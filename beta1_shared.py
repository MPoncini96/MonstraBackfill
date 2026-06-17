from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

DEFAULT_BETA1_STATE_START_DATE = "2025-01-01"
STATE_START_ENV_KEYS = ("BETA1_STATE_START_DATE", "PHANTOM_STATE_START_DATE")


@dataclass
class Beta1Config:
    bot_id: str
    universe: list[str]
    entry_threshold: float
    exit_threshold: float
    max_holdings: int
    fallback_ticker: str


@dataclass
class Beta1ReconstructionResult:
    weekly_returns: pd.Series
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    signal_table: pd.DataFrame
    final_weights: dict[str, float]
    final_asof: str
    apply_date: str


def resolve_beta1_state_start_date() -> str:
    for env_key in STATE_START_ENV_KEYS:
        raw = os.getenv(env_key)
        if not raw:
            continue
        candidate = raw.strip()
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError as exc:
            raise ValueError(
                f"{env_key} must be an ISO date like YYYY-MM-DD; received {candidate!r}"
            ) from exc
    return DEFAULT_BETA1_STATE_START_DATE


def normalize_universe(raw_universe: Any) -> list[str]:
    if raw_universe is None or not isinstance(raw_universe, (list, tuple)):
        return []

    seen: set[str] = set()
    universe: list[str] = []
    for item in raw_universe:
        if item is None:
            continue
        ticker = str(item).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        universe.append(ticker)
    return universe


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def clean_ticker(value: Any, default: str) -> str:
    if value is None:
        return default
    ticker = str(value).strip().upper()
    return ticker or default


def download_weekly_close(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if raw.empty:
        raise ValueError("No data downloaded from yfinance.")

    if not isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns:
            raise ValueError(f"Expected 'Close' column, got: {list(raw.columns)}")
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]
    else:
        close_dict: dict[str, pd.Series] = {}

        for ticker in tickers:
            if (ticker, "Close") in raw.columns:
                close_dict[ticker] = raw[(ticker, "Close")]

        if not close_dict:
            for ticker in tickers:
                if ("Close", ticker) in raw.columns:
                    close_dict[ticker] = raw[("Close", ticker)]

        if not close_dict:
            col_level_0 = list(raw.columns.get_level_values(0))
            col_level_1 = list(raw.columns.get_level_values(1))
            raise ValueError(
                f"Could not locate Close columns. Level 0: {sorted(set(col_level_0))}, "
                f"Level 1: {sorted(set(col_level_1))}"
            )

        close = pd.DataFrame(close_dict)

    close = close.sort_index()
    close = close.dropna(how="all")
    weekly_close = close.resample("W-FRI").last()
    weekly_close = weekly_close.dropna(how="all")
    return weekly_close


def compute_weekly_returns(weekly_close: pd.DataFrame) -> pd.DataFrame:
    return weekly_close.pct_change()


def buy_signal(r2: float, r1: float, threshold: float) -> bool:
    return pd.notna(r2) and pd.notna(r1) and r2 > threshold and r1 > threshold


def sell_signal(r2: float, r1: float, threshold: float) -> bool:
    return pd.notna(r2) and pd.notna(r1) and r2 < threshold and r1 < threshold


def strength_score(r2: float, r1: float) -> float:
    if pd.isna(r2) or pd.isna(r1):
        return 0.0
    return max(r2 + r1, 0.0)


def normalize_weights(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    total = sum(scores.values())
    if total <= 0:
        equal_weight = 1.0 / len(scores)
        return {ticker: equal_weight for ticker in scores}

    return {ticker: score / total for ticker, score in scores.items()}


def _weights_from_row(row: pd.Series) -> dict[str, float]:
    weights: dict[str, float] = {}
    for ticker, weight in row.items():
        val = float(weight)
        if abs(val) > 1e-12:
            weights[str(ticker)] = val
    return weights


def run_beta1_reconstruction(
    universe: list[str],
    start: str,
    end: str,
    entry_threshold: float,
    exit_threshold: float,
    max_holdings: int,
    fallback_ticker: str,
) -> Beta1ReconstructionResult:
    # Beta1 is path-dependent, so preview/live/backfill must replay from the same
    # state start date to avoid drifting into different holdings.
    tickers = sorted(set(universe + [fallback_ticker]))
    weekly_close = download_weekly_close(tickers, start, end)
    weekly_ret = compute_weekly_returns(weekly_close)
    all_dates = weekly_ret.index.tolist()

    if len(all_dates) < 4:
        raise ValueError("Not enough weekly data to run the strategy.")

    held_positions: dict[str, float] = {}
    weights_records: list[dict[str, Any]] = []
    signal_records: list[dict[str, Any]] = []
    portfolio_returns: list[dict[str, Any]] = []

    for i in range(2, len(all_dates) - 1):
        rebalance_date = all_dates[i]
        next_date = all_dates[i + 1]
        next_hold_set: set[str] = set()

        for ticker in list(held_positions.keys()):
            r2 = weekly_ret.loc[all_dates[i - 1], ticker]
            r1 = weekly_ret.loc[all_dates[i], ticker]

            if not sell_signal(r2, r1, exit_threshold):
                next_hold_set.add(ticker)

            signal_records.append(
                {
                    "date": rebalance_date,
                    "ticker": ticker,
                    "status": "held_check",
                    "r2": r2,
                    "r1": r1,
                    "buy_signal": False,
                    "sell_signal": sell_signal(r2, r1, exit_threshold),
                    "score": strength_score(r2, r1),
                }
            )

        candidates: list[tuple[str, float, float, float]] = []
        for ticker in universe:
            if ticker in next_hold_set:
                continue

            r2 = weekly_ret.loc[all_dates[i - 1], ticker]
            r1 = weekly_ret.loc[all_dates[i], ticker]
            is_buy = buy_signal(r2, r1, entry_threshold)
            score = strength_score(r2, r1)

            if is_buy:
                candidates.append((ticker, score, r2, r1))

            signal_records.append(
                {
                    "date": rebalance_date,
                    "ticker": ticker,
                    "status": "candidate_check",
                    "r2": r2,
                    "r1": r1,
                    "buy_signal": is_buy,
                    "sell_signal": False,
                    "score": score,
                }
            )

        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, _, _, _ in candidates:
            if len(next_hold_set) >= max_holdings:
                break
            next_hold_set.add(ticker)

        if len(next_hold_set) == 0:
            new_weights = {fallback_ticker: 1.0}
        else:
            scores: dict[str, float] = {}
            for ticker in next_hold_set:
                r2 = weekly_ret.loc[all_dates[i - 1], ticker]
                r1 = weekly_ret.loc[all_dates[i], ticker]
                scores[ticker] = strength_score(r2, r1)

            new_weights = normalize_weights(scores)

        next_week_ret = 0.0
        for ticker, weight in new_weights.items():
            asset_ret = weekly_ret.loc[next_date, ticker]
            if pd.isna(asset_ret):
                asset_ret = 0.0
            next_week_ret += weight * float(asset_ret)

        portfolio_returns.append(
            {
                "date": next_date,
                "strategy_return": next_week_ret,
            }
        )

        row = {"date": rebalance_date}
        row.update(new_weights)
        weights_records.append(row)
        held_positions = new_weights.copy()

    returns_df = pd.DataFrame(portfolio_returns).set_index("date")
    weights_df = pd.DataFrame(weights_records).set_index("date").fillna(0.0)
    signal_df = pd.DataFrame(signal_records)
    equity_curve = (1.0 + returns_df["strategy_return"]).cumprod()
    final_asof = str(all_dates[-2])
    apply_date = str(all_dates[-1])
    final_weights = _weights_from_row(weights_df.iloc[-1]) if not weights_df.empty else {}

    return Beta1ReconstructionResult(
        weekly_returns=returns_df["strategy_return"],
        equity_curve=equity_curve,
        weights_history=weights_df,
        signal_table=signal_df,
        final_weights=final_weights,
        final_asof=final_asof,
        apply_date=apply_date,
    )
