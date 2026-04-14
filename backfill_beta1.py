#!/usr/bin/env python
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env

load_env()

from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn, get_strategy_origin
from bot_identity import BOT_TYPE_BETA1

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

ENTRY_THRESHOLD = 0.0
EXIT_THRESHOLD = 0.0
MAX_HOLDINGS = 4
FALLBACK_TICKER = "VOO"
UPSERT_BATCH_SIZE = 500

LOG_LEVEL = os.getenv("BETA1_BACKFILL_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("beta1_backfill")


@dataclass
class BacktestResult:
    weekly_returns: pd.Series
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    signal_table: pd.DataFrame


@dataclass
class Beta1BotConfig:
    bot_id: str
    universe: list[str]
    entry_threshold: float
    exit_threshold: float
    max_holdings: int
    fallback_ticker: str


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str
    bot_type: str = BOT_TYPE_BETA1
    origin: str | None = None
    source: str = BOT_EQUITY_SOURCE_BACKFILL


def _normalize_universe(raw_universe: Any) -> list[str]:
    if raw_universe is None:
        return []
    if not isinstance(raw_universe, (list, tuple)):
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


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def _clean_ticker(value: Any, default: str) -> str:
    if value is None:
        return default
    ticker = str(value).strip().upper()
    return ticker or default


def _chunked(values: list[Any], size: int):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def fetch_active_beta1_bots() -> list[Beta1BotConfig]:
    query = """
        SELECT
            bot_id,
            universe,
            COALESCE(entry_threshold, %s) AS entry_threshold,
            COALESCE(exit_threshold, %s) AS exit_threshold,
            COALESCE(max_holdings, %s) AS max_holdings,
            COALESCE(fallback_ticker, %s) AS fallback_ticker
        FROM trading.beta1
        WHERE is_active = TRUE
        ORDER BY bot_id
    """

    configs: list[Beta1BotConfig] = []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (ENTRY_THRESHOLD, EXIT_THRESHOLD, MAX_HOLDINGS, FALLBACK_TICKER),
            )
            rows = cur.fetchall()

    for row in rows:
        bot_id, universe_raw, entry_threshold, exit_threshold, max_holdings, fallback_ticker = row
        configs.append(
            Beta1BotConfig(
                bot_id=str(bot_id).strip().lower(),
                universe=_normalize_universe(universe_raw),
                entry_threshold=_safe_float(entry_threshold, ENTRY_THRESHOLD),
                exit_threshold=_safe_float(exit_threshold, EXIT_THRESHOLD),
                max_holdings=_safe_int(max_holdings, MAX_HOLDINGS),
                fallback_ticker=_clean_ticker(fallback_ticker, FALLBACK_TICKER),
            )
        )

    return configs


def download_weekly_close(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download adjusted close data and convert to weekly close prices.
    Handles both single-ticker and multi-ticker yfinance output shapes.
    """
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

        col_level_0 = list(raw.columns.get_level_values(0))
        col_level_1 = list(raw.columns.get_level_values(1))

        for ticker in tickers:
            if (ticker, "Close") in raw.columns:
                close_dict[ticker] = raw[(ticker, "Close")]

        if not close_dict:
            for ticker in tickers:
                if ("Close", ticker) in raw.columns:
                    close_dict[ticker] = raw[("Close", ticker)]

        if not close_dict:
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


def buy_signal(r2: float, r1: float, threshold: float = ENTRY_THRESHOLD) -> bool:
    return pd.notna(r2) and pd.notna(r1) and r2 > threshold and r1 > threshold


def sell_signal(r2: float, r1: float, threshold: float = EXIT_THRESHOLD) -> bool:
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


def run_backtest(
    universe: list[str],
    start: str = "2020-01-01",
    end: str = "2026-01-01",
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
    max_holdings: int = MAX_HOLDINGS,
    fallback_ticker: str = FALLBACK_TICKER,
) -> BacktestResult:
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

    return BacktestResult(
        weekly_returns=returns_df["strategy_return"],
        equity_curve=equity_curve,
        weights_history=weights_df,
        signal_table=signal_df,
    )


def summarize_performance(weekly_returns: pd.Series) -> pd.Series:
    if weekly_returns.empty:
        raise ValueError("No weekly returns to summarize.")

    n = len(weekly_returns)
    annual_factor = 52

    total_return = (1.0 + weekly_returns).prod() - 1.0
    cagr = (1.0 + total_return) ** (annual_factor / n) - 1.0
    vol = weekly_returns.std(ddof=0) * np.sqrt(annual_factor)

    if weekly_returns.std(ddof=0) > 0:
        sharpe = (weekly_returns.mean() * annual_factor) / vol
    else:
        sharpe = np.nan

    equity = (1.0 + weekly_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    hit_rate = (weekly_returns > 0).mean()

    return pd.Series(
        {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Volatility": vol,
            "Sharpe (rf=0)": sharpe,
            "Max Drawdown": max_drawdown,
            "Weekly Hit Rate": hit_rate,
            "Weeks": n,
        }
    )


def build_benchmark(
    ticker: str = "VOO",
    start: str = "2025-01-01",
    end: str = "2026-01-01",
) -> pd.Series:
    weekly_close = download_weekly_close([ticker], start, end)
    weekly_ret = compute_weekly_returns(weekly_close)[ticker].dropna()
    return weekly_ret


def upsert_backfill_rows(rows: list[BackfillRow]) -> None:
    if not rows:
        return

    sql = """
        INSERT INTO trading.bot_equity (bot_id, bot_type, d, equity, ret, holdings, origin, source)
        VALUES %s
        ON CONFLICT (bot_id, bot_type, d)
        DO UPDATE SET
            equity = EXCLUDED.equity,
            ret = EXCLUDED.ret,
            holdings = EXCLUDED.holdings,
            origin = COALESCE(EXCLUDED.origin, trading.bot_equity.origin),
            source = EXCLUDED.source
    """

    values = [
        (r.bot_id, r.bot_type, r.d, r.equity, r.ret, r.holdings_json, r.origin, r.source)
        for r in rows
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            for batch in _chunked(values, UPSERT_BATCH_SIZE):
                execute_values(cur, sql, batch)
        conn.commit()


def _weights_for_date(weights_history: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, float]:
    if weights_history.empty:
        return {}

    eligible = weights_history.loc[weights_history.index <= target_date]
    if eligible.empty:
        return {}

    row = eligible.iloc[-1]
    weights: dict[str, float] = {}
    for ticker, weight in row.items():
        val = float(weight)
        if abs(val) > 1e-12:
            weights[str(ticker)] = val
    return weights


def backfill_single_bot(bot: Beta1BotConfig) -> dict[str, Any]:
    logger.info(
        "Starting bot_id=%s universe=%d max_holdings=%d entry=%.4f exit=%.4f fallback=%s",
        bot.bot_id,
        len(bot.universe),
        bot.max_holdings,
        bot.entry_threshold,
        bot.exit_threshold,
        bot.fallback_ticker,
    )

    if not bot.universe:
        logger.warning("Skipping bot=%s because universe is empty", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    origin = get_strategy_origin(bot.bot_id, BOT_TYPE_BETA1)

    result = run_backtest(
        universe=bot.universe,
        start=START_DATE,
        end=END_DATE,
        entry_threshold=bot.entry_threshold,
        exit_threshold=bot.exit_threshold,
        max_holdings=bot.max_holdings,
        fallback_ticker=bot.fallback_ticker,
    )

    rows_to_upsert: list[BackfillRow] = []

    for ts, week_ret in result.weekly_returns.items():
        d = ts.date()
        eq = float(result.equity_curve.loc[ts])
        holdings = _weights_for_date(result.weights_history, ts)

        payload = dict(holdings)
        payload["_meta"] = {
            "is_backtest": True,
            "type": "beta1",
            "timing": "rank_on_t_apply_return_on_t_plus_1",
            "entry_threshold": float(bot.entry_threshold),
            "exit_threshold": float(bot.exit_threshold),
            "max_holdings": int(bot.max_holdings),
            "fallback_ticker": bot.fallback_ticker,
            "asof": str(ts),
        }

        rows_to_upsert.append(
            BackfillRow(
                bot_id=bot.bot_id,
                d=d,
                equity=eq,
                ret=float(week_ret),
                holdings_json=json.dumps(payload),
                bot_type=BOT_TYPE_BETA1,
                origin=origin,
            )
        )

    upsert_backfill_rows(rows_to_upsert)

    final_equity = float(result.equity_curve.iloc[-1])
    final_return_pct = (final_equity - 1.0) * 100.0

    logger.info(
        "Completed bot=%s processed_weeks=%d final_equity=%.6f final_return_pct=%.2f",
        bot.bot_id,
        len(result.weekly_returns),
        final_equity,
        final_return_pct,
    )

    return {
        "bot_id": bot.bot_id,
        "processed_days": len(result.weekly_returns),
        "final_equity": final_equity,
        "final_return_pct": final_return_pct,
        "errors": 0,
        "skipped": False,
    }


def backfill_beta1() -> None:
    logger.info("Loading active Beta1 bots from trading.beta1")
    bots = fetch_active_beta1_bots()

    if not bots:
        logger.warning("No active bots found in trading.beta1")
        return

    logger.info(
        "Found %d active bot(s). Backfilling from %s to %s",
        len(bots),
        START_DATE,
        END_DATE,
    )

    results: list[dict[str, Any]] = []

    for bot in bots:
        try:
            results.append(backfill_single_bot(bot))
        except Exception as exc:
            logger.exception("Fatal error in bot=%s: %s", bot.bot_id, exc)
            results.append(
                {
                    "bot_id": bot.bot_id,
                    "processed_days": 0,
                    "final_equity": None,
                    "final_return_pct": None,
                    "errors": 1,
                    "skipped": True,
                }
            )

    logger.info("========== Beta1 Backfill Summary ==========")
    for r in results:
        if r.get("skipped"):
            logger.info(
                "bot=%s status=SKIPPED/FAILED processed_days=%s errors=%s",
                r["bot_id"],
                r.get("processed_days"),
                r.get("errors"),
            )
        else:
            logger.info(
                "bot=%s processed_days=%d final_equity=%.6f final_return_pct=%.2f errors=%d",
                r["bot_id"],
                r["processed_days"],
                r["final_equity"],
                r["final_return_pct"],
                r["errors"],
            )

    logger.info("Beta1 backfill complete.")


if __name__ == "__main__":
    backfill_beta1()
