#!/usr/bin/env python
"""
Production-grade Alpha1 backfill script with:
- ranked 40/30/20/10 allocation
- risk-off / kill-switch support
- cash-equivalent fallback
- transaction costs + slippage
- bulk upserts into trading.bot_equity

Timing convention:
- Holdings for trading date t are selected using data available through t-1
- Return applied is the return realized on trading date t
- Trading costs are charged when moving from prior day's holdings to current day's holdings

Assumptions:
- Universe and parameters are loaded from trading.alpha1
- cash_equivalent is a ticker like SHY, BIL, SGOV, etc.
- Optional benchmark field may exist in trading.alpha1; if not, benchmark filter is skipped
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from psycopg2.extras import execute_values

# Ensure project root import path is available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env

load_env()

from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn, get_strategy_origin
from bots.alpha1 import _normalize_rank_weights_array
from bot_identity import BOT_TYPE_ALPHA1

# ============================================================
# Configuration
# ============================================================

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

LOG_LEVEL = os.getenv("ALPHA1_BACKFILL_LOG_LEVEL", "INFO").upper()

# Default rank weights for Alpha1 (per-bot weights come from trading.alpha1.rank_weights)
RANK_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
DEFAULT_TOP_N_SLOTS = len(RANK_WEIGHTS)

# Eligibility / data hygiene
REQUIRE_FULL_LOOKBACK_WINDOW = True
USE_CASH_EQUIVALENT_FALLBACK = True

# Cost model (expressed in decimal terms)
# Example: 0.0005 = 5 bps = 0.05%
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0

# Bulk DB write size
UPSERT_BATCH_SIZE = 500

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("alpha1_backfill")


# ============================================================
# Data structures
# ============================================================

@dataclass
class Alpha1BotConfig:
    bot_id: str
    universe: list[str]
    top_n: int | None
    lookback_days: int
    cash_equivalent: str | None
    rank_weights: np.ndarray = field(default_factory=lambda: RANK_WEIGHTS.copy())

    # Risk-off / kill-switch controls
    benchmark: str | None = None
    enable_kill_switch: bool = True
    enable_benchmark_filter: bool = False
    kill_switch_mode: str = "top_negative"   # top_negative | avg_negative | off
    benchmark_mode: str = "benchmark_positive"  # benchmark_positive | benchmark_gt_cash | off

    # Optional thresholds
    top_return_threshold: float = 0.0
    benchmark_return_threshold: float = 0.0

    # Cost model
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str
    bot_type: str = BOT_TYPE_ALPHA1
    origin: str | None = None
    source: str = BOT_EQUITY_SOURCE_BACKFILL


# ============================================================
# Utility helpers
# ============================================================

def safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_universe(raw_universe: Any) -> list[str]:
    parsed = safe_json_loads(raw_universe)
    if parsed is None:
        return []

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Universe must be list/tuple, got {type(parsed).__name__}")

    seen: set[str] = set()
    result: list[str] = []
    for item in parsed:
        if item is None:
            continue
        ticker = str(item).strip().upper()
        if not ticker:
            continue
        if ticker not in seen:
            seen.add(ticker)
            result.append(ticker)
    return result


def clean_ticker(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    return s or None


def clamp_top_n(top_n: int | None, rank_weights: np.ndarray) -> int:
    cap = max(1, int(len(rank_weights))) if len(rank_weights) else DEFAULT_TOP_N_SLOTS
    if top_n is None:
        return min(DEFAULT_TOP_N_SLOTS, cap)
    return max(1, min(int(top_n), cap))


def is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def bps_to_decimal(bps: float) -> float:
    return float(bps) / 10_000.0


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# ============================================================
# Database helpers
# ============================================================

def fetch_alpha1_rank_weights_for_active_bots() -> dict[str, Any]:
    """
    Per-bot raw weight payload from trading.alpha1 (rank_weights preferred, else weights).

    Separate small queries survive schemas where the bulk config SELECT omits weights
    (e.g. missing benchmark column) or rank_weights is null but weights is populated.
    """
    rank_rows: list[tuple[Any, ...]] = []
    weight_rows: list[tuple[Any, ...]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT bot_id, rank_weights FROM trading.alpha1 WHERE is_active = true"
                )
                rank_rows = list(cur.fetchall())
            except Exception:
                conn.rollback()
            try:
                cur.execute(
                    "SELECT bot_id, weights FROM trading.alpha1 WHERE is_active = true"
                )
                weight_rows = list(cur.fetchall())
            except Exception:
                conn.rollback()

    by_rank = {str(bid): rw for bid, rw in rank_rows}
    by_w = {str(bid): rw for bid, rw in weight_rows}
    merged: dict[str, Any] = {}
    for bid in set(by_rank) | set(by_w):
        rr = by_rank.get(bid)
        ww = by_w.get(bid)
        raw = rr if rr is not None else ww
        if raw is not None:
            merged[bid] = raw
    return merged


def fetch_active_alpha1_bots() -> list[Alpha1BotConfig]:
    """
    Attempts to load extended Alpha1 config.
    Falls back gracefully if some optional columns do not exist.
    """
    primary_query_with_rank = """
        SELECT
            bot_id,
            universe,
            top_n,
            lookback_days,
            cash_equivalent,
            benchmark,
            {rank_col},
            COALESCE(enable_kill_switch, true) AS enable_kill_switch,
            COALESCE(enable_benchmark_filter, false) AS enable_benchmark_filter,
            COALESCE(kill_switch_mode, 'top_negative') AS kill_switch_mode,
            COALESCE(benchmark_mode, 'benchmark_positive') AS benchmark_mode,
            COALESCE(top_return_threshold, 0.0) AS top_return_threshold,
            COALESCE(benchmark_return_threshold, 0.0) AS benchmark_return_threshold,
            COALESCE(transaction_cost_bps, %s) AS transaction_cost_bps,
            COALESCE(slippage_bps, %s) AS slippage_bps
        FROM trading.alpha1
        WHERE is_active = true
        ORDER BY bot_id
    """

    primary_query = """
        SELECT
            bot_id,
            universe,
            top_n,
            lookback_days,
            cash_equivalent,
            benchmark,
            COALESCE(enable_kill_switch, true) AS enable_kill_switch,
            COALESCE(enable_benchmark_filter, false) AS enable_benchmark_filter,
            COALESCE(kill_switch_mode, 'top_negative') AS kill_switch_mode,
            COALESCE(benchmark_mode, 'benchmark_positive') AS benchmark_mode,
            COALESCE(top_return_threshold, 0.0) AS top_return_threshold,
            COALESCE(benchmark_return_threshold, 0.0) AS benchmark_return_threshold,
            COALESCE(transaction_cost_bps, %s) AS transaction_cost_bps,
            COALESCE(slippage_bps, %s) AS slippage_bps
        FROM trading.alpha1
        WHERE is_active = true
        ORDER BY bot_id
    """

    fallback_query = """
        SELECT
            bot_id,
            universe,
            top_n,
            lookback_days,
            cash_equivalent
        FROM trading.alpha1
        WHERE is_active = true
        ORDER BY bot_id
    """

    weights_from_table = fetch_alpha1_rank_weights_for_active_bots()

    rows: list[tuple[Any, ...]] = []
    extended = True
    extended_with_rank = False

    params = (DEFAULT_TRANSACTION_COST_BPS, DEFAULT_SLIPPAGE_BPS)
    with get_conn() as conn:
        with conn.cursor() as cur:
            for rank_col in ("rank_weights", "weights AS rank_weights"):
                try:
                    cur.execute(primary_query_with_rank.format(rank_col=rank_col), params)
                    rows = cur.fetchall()
                    extended_with_rank = True
                    break
                except Exception:
                    conn.rollback()
                    rows = []
            if not extended_with_rank:
                try:
                    cur.execute(primary_query, params)
                    rows = cur.fetchall()
                except Exception:
                    conn.rollback()
                    extended = False
                    cur.execute(fallback_query)
                    rows = cur.fetchall()

    configs: list[Alpha1BotConfig] = []

    if extended:
        for row in rows:
            if extended_with_rank:
                (
                    bot_id,
                    universe_raw,
                    top_n,
                    lookback_days,
                    cash_equiv,
                    benchmark,
                    rank_raw,
                    enable_kill_switch,
                    enable_benchmark_filter,
                    kill_switch_mode,
                    benchmark_mode,
                    top_return_threshold,
                    benchmark_return_threshold,
                    transaction_cost_bps,
                    slippage_bps,
                ) = row
                raw_rw = rank_raw if rank_raw is not None else weights_from_table.get(str(bot_id))
                rank_w = _normalize_rank_weights_array(raw_rw)
            else:
                (
                    bot_id,
                    universe_raw,
                    top_n,
                    lookback_days,
                    cash_equiv,
                    benchmark,
                    enable_kill_switch,
                    enable_benchmark_filter,
                    kill_switch_mode,
                    benchmark_mode,
                    top_return_threshold,
                    benchmark_return_threshold,
                    transaction_cost_bps,
                    slippage_bps,
                ) = row
                rank_w = _normalize_rank_weights_array(weights_from_table.get(str(bot_id)))

            configs.append(
                Alpha1BotConfig(
                    bot_id=str(bot_id),
                    universe=normalize_universe(universe_raw),
                    top_n=int(top_n) if top_n is not None else None,
                    lookback_days=int(lookback_days),
                    cash_equivalent=clean_ticker(cash_equiv),
                    rank_weights=rank_w,
                    benchmark=clean_ticker(benchmark),
                    enable_kill_switch=bool(enable_kill_switch),
                    enable_benchmark_filter=bool(enable_benchmark_filter),
                    kill_switch_mode=str(kill_switch_mode),
                    benchmark_mode=str(benchmark_mode),
                    top_return_threshold=safe_float(top_return_threshold, 0.0),
                    benchmark_return_threshold=safe_float(benchmark_return_threshold, 0.0),
                    transaction_cost_bps=safe_float(transaction_cost_bps, DEFAULT_TRANSACTION_COST_BPS),
                    slippage_bps=safe_float(slippage_bps, DEFAULT_SLIPPAGE_BPS),
                )
            )
    else:
        for row in rows:
            bot_id, universe_raw, top_n, lookback_days, cash_equiv = row
            bid = str(bot_id)
            configs.append(
                Alpha1BotConfig(
                    bot_id=bid,
                    universe=normalize_universe(universe_raw),
                    top_n=int(top_n) if top_n is not None else None,
                    lookback_days=int(lookback_days),
                    cash_equivalent=clean_ticker(cash_equiv),
                    rank_weights=_normalize_rank_weights_array(weights_from_table.get(bid)),
                )
            )

    return configs


def fetch_previous_equity(
    bot_id: str, start_date: str, bot_type: str = BOT_TYPE_ALPHA1
) -> float:
    query = """
        SELECT equity
        FROM trading.bot_equity
        WHERE bot_id = %s AND bot_type = %s AND d < %s
        ORDER BY d DESC
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (bot_id, bot_type, start_date))
            row = cur.fetchone()

    if row and is_finite_number(row[0]):
        return float(row[0])
    return 1.0


def fetch_previous_holdings(
    bot_id: str, start_date: str, bot_type: str = BOT_TYPE_ALPHA1
) -> dict[str, float]:
    """
    Pull prior holdings if they exist before backfill start.
    Only returns actual ticker weights, excluding _meta.
    """
    query = """
        SELECT holdings
        FROM trading.bot_equity
        WHERE bot_id = %s AND bot_type = %s AND d < %s
        ORDER BY d DESC
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (bot_id, bot_type, start_date))
            row = cur.fetchone()

    if not row or row[0] is None:
        return {}

    raw = row[0]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}

    cleaned: dict[str, float] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        if is_finite_number(v):
            cleaned[str(k).upper()] = float(v)
    return cleaned


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
            for batch in chunked(values, UPSERT_BATCH_SIZE):
                execute_values(cur, sql, batch)
        conn.commit()


# ============================================================
# Market data helpers
# ============================================================

def download_adjusted_close(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data missing 'Close' field.")
        close_df = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data missing 'Close' column.")
        only_symbol = symbols[0]
        close_df = raw[["Close"]].rename(columns={"Close": only_symbol})

    close_df = close_df.sort_index()
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df[~close_df.index.duplicated(keep="last")]
    close_df = close_df.dropna(how="all")
    close_df.columns = [str(c).upper() for c in close_df.columns]
    return close_df


def build_price_frame(bot: Alpha1BotConfig) -> pd.DataFrame:
    symbols = list(bot.universe)

    extra = [bot.cash_equivalent, bot.benchmark]
    for sym in extra:
        if sym and sym not in symbols:
            symbols.append(sym)

    return download_adjusted_close(symbols, START_DATE, END_DATE)


# ============================================================
# Strategy helpers
# ============================================================

def get_trailing_returns(
    prices: pd.DataFrame,
    end_idx_exclusive: int,
    lookback_days: int,
    symbols: set[str] | None = None,
) -> pd.Series:
    start_idx = max(0, end_idx_exclusive - lookback_days)
    lookback = prices.iloc[start_idx:end_idx_exclusive]

    if len(lookback) < 2:
        return pd.Series(dtype=float)

    first_row = lookback.iloc[0]
    last_row = lookback.iloc[-1]

    if REQUIRE_FULL_LOOKBACK_WINDOW:
        valid_mask = lookback.notna().all(axis=0)
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]
    else:
        valid_mask = first_row.notna() & last_row.notna()
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]

    trailing = (last_row / first_row) - 1.0
    trailing = trailing.replace([np.inf, -np.inf], np.nan).dropna()

    if symbols:
        trailing = trailing[trailing.index.isin(symbols)]

    return trailing


def evaluate_kill_switch(
    bot: Alpha1BotConfig,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
) -> tuple[bool, str]:
    """
    Returns:
    - risk_off flag
    - reason
    """
    reasons: list[str] = []

    # Kill switch based on ranked universe
    if bot.enable_kill_switch and bot.kill_switch_mode != "off":
        if ranked_trailing.empty:
            reasons.append("kill_switch:no_ranked_candidates")
        elif bot.kill_switch_mode == "top_negative":
            top_ret = float(ranked_trailing.max())
            if top_ret <= bot.top_return_threshold:
                reasons.append(f"kill_switch:top_ret={top_ret:.6f}<=threshold={bot.top_return_threshold:.6f}")
        elif bot.kill_switch_mode == "avg_negative":
            avg_ret = float(ranked_trailing.mean())
            if avg_ret <= bot.top_return_threshold:
                reasons.append(f"kill_switch:avg_ret={avg_ret:.6f}<=threshold={bot.top_return_threshold:.6f}")

    # Benchmark filter
    if bot.enable_benchmark_filter and bot.benchmark_mode != "off" and bot.benchmark:
        bench_ret = benchmark_trailing.get(bot.benchmark, np.nan)

        if pd.isna(bench_ret):
            reasons.append("benchmark_filter:no_benchmark_data")
        else:
            bench_ret = float(bench_ret)

            if bot.benchmark_mode == "benchmark_positive":
                if bench_ret <= bot.benchmark_return_threshold:
                    reasons.append(
                        f"benchmark_filter:bench_ret={bench_ret:.6f}<=threshold={bot.benchmark_return_threshold:.6f}"
                    )

            elif bot.benchmark_mode == "benchmark_gt_cash":
                cash_ret = np.nan
                if bot.cash_equivalent:
                    cash_ret = cash_trailing.get(bot.cash_equivalent, np.nan)

                if pd.isna(cash_ret):
                    reasons.append("benchmark_filter:no_cash_data")
                else:
                    cash_ret = float(cash_ret)
                    if bench_ret <= cash_ret:
                        reasons.append(
                            f"benchmark_filter:bench_ret={bench_ret:.6f}<=cash_ret={cash_ret:.6f}"
                        )

    if reasons:
        return True, "; ".join(reasons)
    return False, "risk_on"


def choose_risk_on_holdings(
    ranked_trailing: pd.Series, top_n: int, rank_weights: np.ndarray
) -> tuple[list[str], np.ndarray]:
    if ranked_trailing.empty:
        return [], np.array([], dtype=float)

    ranked = ranked_trailing.sort_values(ascending=False)
    selected = ranked.head(top_n).index.tolist()

    rw = np.asarray(rank_weights, dtype=float)[: len(selected)].copy()
    if rw.size == 0:
        return [], np.array([], dtype=float)
    weights = rw / rw.sum()

    return selected, weights


def choose_holdings_for_day(
    bot: Alpha1BotConfig,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
    effective_top_n: int,
) -> tuple[list[str], np.ndarray, bool, str]:
    risk_off, reason = evaluate_kill_switch(
        bot=bot,
        ranked_trailing=ranked_trailing,
        benchmark_trailing=benchmark_trailing,
        cash_trailing=cash_trailing,
    )

    if risk_off:
        if USE_CASH_EQUIVALENT_FALLBACK and bot.cash_equivalent:
            return [bot.cash_equivalent], np.array([1.0], dtype=float), True, reason
        return [], np.array([], dtype=float), True, reason

    selected, weights = choose_risk_on_holdings(ranked_trailing, effective_top_n, bot.rank_weights)

    if not selected and USE_CASH_EQUIVALENT_FALLBACK and bot.cash_equivalent:
        return [bot.cash_equivalent], np.array([1.0], dtype=float), True, "fallback:no_selected_symbols"

    return selected, weights, False, reason


def compute_turnover(prev_holdings: dict[str, float], new_holdings: dict[str, float]) -> float:
    """
    Standard one-way turnover:
    sum(abs(w_new - w_old)) / 2

    Examples:
    - 100% cash -> 100% SPY = 1.0
    - 40/30/20/10 -> same weights next day = 0.0
    """
    all_symbols = set(prev_holdings) | set(new_holdings)
    gross_change = sum(abs(new_holdings.get(sym, 0.0) - prev_holdings.get(sym, 0.0)) for sym in all_symbols)
    return gross_change / 2.0


def compute_cost_drag(
    turnover: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> float:
    """
    Total cost drag in decimal return terms.
    Example:
    turnover=1.0, tc=5bps, slip=5bps => 0.001 = 10bps total drag
    """
    total_bps = float(transaction_cost_bps) + float(slippage_bps)
    return turnover * bps_to_decimal(total_bps)


def compute_day_return_and_holdings(
    day_returns: pd.Series,
    selected_symbols: list[str],
    weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    if not selected_symbols or len(weights) == 0:
        return 0.0, {}

    valid_pairs: list[tuple[str, float]] = []
    for sym, w in zip(selected_symbols, weights):
        if sym in day_returns.index and pd.notna(day_returns[sym]):
            valid_pairs.append((sym, float(w)))

    if not valid_pairs:
        return 0.0, {}

    total_weight = sum(w for _, w in valid_pairs)
    if total_weight <= 0:
        return 0.0, {}

    normalized_pairs = [(sym, w / total_weight) for sym, w in valid_pairs]
    gross_ret = sum(day_returns[sym] * w for sym, w in normalized_pairs)
    holdings = {sym: float(w) for sym, w in normalized_pairs}
    return float(gross_ret), holdings


def build_holdings_payload(
    holdings: dict[str, float],
    bot: Alpha1BotConfig,
    selected_count: int,
    risk_off: bool,
    risk_reason: str,
    gross_ret: float,
    net_ret: float,
    turnover: float,
    cost_drag: float,
) -> str:
    payload = dict(holdings)
    payload["_meta"] = {
        "is_backtest": True,
        "type": "alpha1",
        "data_source": "yfinance",
        "timing": "rank_on_t_minus_1_apply_return_on_t",
        "lookback_days": int(bot.lookback_days),
        "configured_top_n": int(clamp_top_n(bot.top_n, bot.rank_weights)),
        "rank_weights": [float(x) for x in bot.rank_weights.tolist()],
        "selected_holdings": int(selected_count),
        "weight_model": "rank_weights",
        "risk_off": bool(risk_off),
        "risk_reason": str(risk_reason),
        "benchmark": bot.benchmark,
        "cash_equivalent": bot.cash_equivalent,
        "kill_switch_mode": bot.kill_switch_mode,
        "benchmark_mode": bot.benchmark_mode,
        "gross_ret": float(gross_ret),
        "net_ret": float(net_ret),
        "turnover": float(turnover),
        "transaction_cost_bps": float(bot.transaction_cost_bps),
        "slippage_bps": float(bot.slippage_bps),
        "cost_drag": float(cost_drag),
    }
    return json.dumps(payload)


# ============================================================
# Main bot backfill logic
# ============================================================

def backfill_single_bot(bot: Alpha1BotConfig) -> dict[str, Any]:
    logger.info(
        "Starting bot_id=%s universe=%d top_n=%s lookback=%d cash=%s benchmark=%s",
        bot.bot_id,
        len(bot.universe),
        bot.top_n,
        bot.lookback_days,
        bot.cash_equivalent,
        bot.benchmark,
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

    effective_top_n = clamp_top_n(bot.top_n, bot.rank_weights)
    prices = build_price_frame(bot)

    if prices.empty:
        logger.warning("Skipping bot=%s because no price data was downloaded", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    origin = get_strategy_origin(bot.bot_id, BOT_TYPE_ALPHA1)

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    equity = fetch_previous_equity(bot.bot_id, START_DATE)
    prev_holdings = fetch_previous_holdings(bot.bot_id, START_DATE)

    logger.info(
        "bot=%s trading_days=%d starting_equity=%.6f prior_holdings=%s",
        bot.bot_id,
        len(prices),
        equity,
        prev_holdings,
    )

    rows_to_upsert: list[BackfillRow] = []
    errors = 0
    processed_days = 0
    risk_off_days = 0

    ranking_universe = set(bot.universe)
    benchmark_symbols = {bot.benchmark} if bot.benchmark else set()
    cash_symbols = {bot.cash_equivalent} if bot.cash_equivalent else set()

    for i, trading_ts in enumerate(prices.index):
        if i == 0:
            continue

        trading_day = trading_ts.date()

        try:
            ranked_trailing = get_trailing_returns(
                prices=prices,
                end_idx_exclusive=i,
                lookback_days=bot.lookback_days,
                symbols=ranking_universe,
            )

            benchmark_trailing = get_trailing_returns(
                prices=prices,
                end_idx_exclusive=i,
                lookback_days=bot.lookback_days,
                symbols=benchmark_symbols,
            )

            cash_trailing = get_trailing_returns(
                prices=prices,
                end_idx_exclusive=i,
                lookback_days=bot.lookback_days,
                symbols=cash_symbols,
            )

            selected_symbols, weights, risk_off, risk_reason = choose_holdings_for_day(
                bot=bot,
                ranked_trailing=ranked_trailing,
                benchmark_trailing=benchmark_trailing,
                cash_trailing=cash_trailing,
                effective_top_n=effective_top_n,
            )

            if risk_off:
                risk_off_days += 1

            day_ret_vec = returns.loc[trading_ts]
            gross_ret, new_holdings = compute_day_return_and_holdings(
                day_returns=day_ret_vec,
                selected_symbols=selected_symbols,
                weights=weights,
            )

            turnover = compute_turnover(prev_holdings, new_holdings)
            cost_drag = compute_cost_drag(
                turnover=turnover,
                transaction_cost_bps=bot.transaction_cost_bps,
                slippage_bps=bot.slippage_bps,
            )

            net_ret = gross_ret - cost_drag
            equity *= (1.0 + net_ret)

            holdings_json = build_holdings_payload(
                holdings=new_holdings,
                bot=bot,
                selected_count=len(selected_symbols),
                risk_off=risk_off,
                risk_reason=risk_reason,
                gross_ret=gross_ret,
                net_ret=net_ret,
                turnover=turnover,
                cost_drag=cost_drag,
            )

            rows_to_upsert.append(
                BackfillRow(
                    bot_id=bot.bot_id,
                    d=trading_day,
                    equity=float(equity),
                    ret=float(net_ret),
                    holdings_json=holdings_json,
                    bot_type=BOT_TYPE_ALPHA1,
                    origin=origin,
                )
            )

            prev_holdings = new_holdings
            processed_days += 1

        except Exception as exc:
            errors += 1
            logger.exception(
                "Error processing bot=%s date=%s: %s",
                bot.bot_id,
                trading_day,
                exc,
            )

    upsert_backfill_rows(rows_to_upsert)

    final_return_pct = (equity - 1.0) * 100.0
    logger.info(
        "Completed bot=%s processed_days=%d risk_off_days=%d final_equity=%.6f final_return_pct=%.2f errors=%d",
        bot.bot_id,
        processed_days,
        risk_off_days,
        equity,
        final_return_pct,
        errors,
    )

    return {
        "bot_id": bot.bot_id,
        "processed_days": processed_days,
        "risk_off_days": risk_off_days,
        "final_equity": equity,
        "final_return_pct": final_return_pct,
        "errors": errors,
        "skipped": False,
    }


# ============================================================
# Main
# ============================================================

def backfill_alpha1() -> None:
    logger.info("Loading active Alpha1 bots from trading.alpha1")
    bots = fetch_active_alpha1_bots()

    if not bots:
        logger.warning("No active bots found in trading.alpha1")
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
                    "risk_off_days": 0,
                    "final_equity": None,
                    "final_return_pct": None,
                    "errors": 1,
                    "skipped": True,
                }
            )

    logger.info("========== Alpha1 Backfill Summary ==========")
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
                "bot=%s processed_days=%d risk_off_days=%d final_equity=%.6f final_return_pct=%.2f errors=%d",
                r["bot_id"],
                r["processed_days"],
                r["risk_off_days"],
                r["final_equity"],
                r["final_return_pct"],
                r["errors"],
            )

    logger.info("Alpha1 backfill complete.")


if __name__ == "__main__":
    backfill_alpha1()