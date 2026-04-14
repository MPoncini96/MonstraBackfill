from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from bot_identity import BOT_TYPE_BETA1

ENTRY_THRESHOLD = 0.0
EXIT_THRESHOLD = 0.0
MAX_HOLDINGS = 4
FALLBACK_TICKER = "VOO"
DEFAULT_UNIVERSE = [
    "VOO",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "AVGO",
    "AMD",
]
DEFAULT_HISTORY_DAYS = 730


@dataclass
class Beta1Config:
    bot_id: str
    universe: list[str]
    max_holdings: int
    entry_threshold: float
    exit_threshold: float
    fallback_ticker: str


def _normalize_universe(raw_universe: Any) -> list[str]:
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


def _resolve_download_window() -> tuple[str, str]:
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=DEFAULT_HISTORY_DAYS)
    return start.isoformat(), end.isoformat()


def _build_live_config(bot_id: str, use_db_config: bool = True) -> Beta1Config:
    config_data: dict[str, Any] | None = None

    if use_db_config:
        try:
            from db import get_beta1_config

            config_data = get_beta1_config(bot_id)
        except Exception as exc:
            print(f"Warning: Could not load beta1 config for {bot_id} from database, using defaults: {exc}")

    if config_data:
        return Beta1Config(
            bot_id=bot_id,
            universe=_normalize_universe(config_data.get("universe")) or list(DEFAULT_UNIVERSE),
            max_holdings=_safe_int(config_data.get("max_holdings"), MAX_HOLDINGS),
            entry_threshold=_safe_float(config_data.get("entry_threshold"), ENTRY_THRESHOLD),
            exit_threshold=_safe_float(config_data.get("exit_threshold"), EXIT_THRESHOLD),
            fallback_ticker=_clean_ticker(config_data.get("fallback_ticker"), FALLBACK_TICKER),
        )

    return Beta1Config(
        bot_id=bot_id,
        universe=list(DEFAULT_UNIVERSE),
        max_holdings=MAX_HOLDINGS,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        fallback_ticker=FALLBACK_TICKER,
    )


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


def _latest_target_weights(config: Beta1Config, start: str, end: str) -> tuple[dict[str, float], str | None, str]:
    tickers = sorted(set(config.universe + [config.fallback_ticker]))
    weekly_close = download_weekly_close(tickers, start, end)
    weekly_ret = compute_weekly_returns(weekly_close)

    all_dates = weekly_ret.index.tolist()
    if len(all_dates) < 4:
        return {config.fallback_ticker: 1.0}, None, "insufficient_weekly_history"

    held_positions: dict[str, float] = {}

    for i in range(2, len(all_dates) - 1):
        next_hold_set: set[str] = set()

        for ticker in list(held_positions.keys()):
            r2 = weekly_ret.loc[all_dates[i - 1], ticker]
            r1 = weekly_ret.loc[all_dates[i], ticker]
            if not sell_signal(r2, r1, config.exit_threshold):
                next_hold_set.add(ticker)

        candidates: list[tuple[str, float]] = []
        for ticker in config.universe:
            if ticker in next_hold_set:
                continue

            r2 = weekly_ret.loc[all_dates[i - 1], ticker]
            r1 = weekly_ret.loc[all_dates[i], ticker]
            if buy_signal(r2, r1, config.entry_threshold):
                candidates.append((ticker, strength_score(r2, r1)))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, _ in candidates:
            if len(next_hold_set) >= config.max_holdings:
                break
            next_hold_set.add(ticker)

        if not next_hold_set:
            held_positions = {config.fallback_ticker: 1.0}
        else:
            scores: dict[str, float] = {}
            for ticker in next_hold_set:
                r2 = weekly_ret.loc[all_dates[i - 1], ticker]
                r1 = weekly_ret.loc[all_dates[i], ticker]
                scores[ticker] = strength_score(r2, r1)
            held_positions = normalize_weights(scores)

    asof = str(all_dates[-2])
    next_date = str(all_dates[-1])
    return held_positions, asof, next_date


def run_beta1(bot_id: str = "beta1", use_db_config: bool = True) -> dict:
    ts = datetime.now(timezone.utc)
    config = _build_live_config(bot_id, use_db_config=use_db_config)

    try:
        start, end = _resolve_download_window()
        target_weights, asof, applied_to = _latest_target_weights(config, start, end)

        signal = "REBALANCE"
        note = f"As of {asof}: weekly beta1 rebalance"
        if not asof:
            signal = "HOLD"
            note = f"Not enough weekly history; holding {config.fallback_ticker}"

        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_BETA1,
            "ts": ts,
            "signal": signal,
            "note": note,
            "payload": {
                "asof": asof,
                "apply_date": applied_to,
                "interval": "W-FRI",
                "entry_threshold": config.entry_threshold,
                "exit_threshold": config.exit_threshold,
                "max_holdings": config.max_holdings,
                "fallback_ticker": config.fallback_ticker,
                "target_weights": target_weights,
            },
        }

    except Exception as exc:
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_BETA1,
            "ts": ts,
            "signal": "HOLD",
            "note": f"beta1 error: {exc}",
            "payload": {
                "interval": "W-FRI",
                "entry_threshold": config.entry_threshold,
                "exit_threshold": config.exit_threshold,
                "max_holdings": config.max_holdings,
                "fallback_ticker": config.fallback_ticker,
                "target_weights": {config.fallback_ticker: 1.0},
            },
        }
