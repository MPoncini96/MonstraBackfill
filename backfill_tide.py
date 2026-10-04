#!/usr/bin/env python
"""
Tide backfill: long-hold Force variant (from trading.tide).
Writes daily rows to trading.bot_equity with bot_type = tide.

Simulates the live strategy exactly (bots/tide.py run_tide), day by day:
- Composite ranking (50% trailing Sharpe + 30% trailing return + 20% low
  volatility, each z-scored) over the configured universe.
- Membership-stable top-N selection (additive incumbency margin).
- Force's own kill switch (bots/alpha1.py _evaluate_kill_switch), reused
  unmodified via TideConfig.as_alpha1_config().
- Per-position trailing stop-loss + portfolio drawdown circuit breaker
  (tide_math.py apply_guardrails), both re-derived from the equity/price
  curve being built during this same walk-forward pass (no DB round-trips
  mid-backfill, unlike the live runner).

Timing convention (matches other Monstra backfills, e.g. backfill_draco.py):
- Holdings for date t are selected using data available through t-1.
- Return applied is the close-to-close return on date t.

Tide ships hidden/Phase 1 (algorithm_registry.py slug="tide"): this script
is standalone and NOT wired into register_backfill_dispatcher, so it only
ever runs when invoked manually.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
from psycopg2.extras import Json as PgJson, RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env

load_env()

from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn
from bot_identity import BOT_TYPE_TIDE
import tide_math as tm
from bots.alpha1 import DEFAULT_UNIVERSE, _evaluate_kill_switch
from bots.tide import TideConfig, _config_from_db_row

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

# Extra calendar days before START_DATE to fully warm up the configured
# lookback (default 90 trading days) plus a safety margin -- same multiplier
# bots/tide.py's own _resolve_download_window uses for the live runner.
WARMUP_BUFFER_MULTIPLIER = 6
WARMUP_BUFFER_MIN_DAYS = 180
WARMUP_SAFETY_MARGIN_DAYS = 400

UPSERT_BATCH_SIZE = 500

LOG_LEVEL = os.getenv("TIDE_BACKFILL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tide_backfill")


@dataclass
class TideBotRecord:
    bot_id: str
    cfg: TideConfig
    origin: str | None = None


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str
    bot_type: str = BOT_TYPE_TIDE
    origin: str | None = None
    source: str = BOT_EQUITY_SOURCE_BACKFILL


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_active_tide_bots() -> list[TideBotRecord]:
    records: list[TideBotRecord] = []
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM trading.tide WHERE is_active = TRUE ORDER BY bot_id")
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Could not load trading.tide: %s", exc)
        return []

    for row in rows:
        d = dict(row)
        bid = str(d.get("bot_id", "")).strip().lower()
        if not bid:
            continue
        cfg = _config_from_db_row(d)
        records.append(TideBotRecord(bot_id=bid, cfg=cfg, origin=d.get("origin")))
    return records


def upsert_backfill_rows(rows: list[BackfillRow]) -> None:
    if not rows:
        return
    bot_id, bot_type = rows[0].bot_id, rows[0].bot_type
    logger.info("tide backfill upsert: bot_id=%s rows=%d", bot_id, len(rows))
    sql = """
        INSERT INTO trading.bot_equity
            (bot_id, bot_type, d, equity, ret, holdings, origin, source)
        VALUES %s
        ON CONFLICT (bot_id, bot_type, d) DO UPDATE SET
            equity     = EXCLUDED.equity,
            ret        = EXCLUDED.ret,
            holdings   = EXCLUDED.holdings,
            origin     = COALESCE(EXCLUDED.origin, trading.bot_equity.origin),
            updated_at = now(),
            source     = CASE
                WHEN LOWER(TRIM(COALESCE(trading.bot_equity.source, ''))) = 'live trading'
                THEN trading.bot_equity.source
                ELSE EXCLUDED.source
            END
    """
    values = [
        (r.bot_id, r.bot_type, r.d, r.equity, r.ret, PgJson(json.loads(r.holdings_json)), r.origin, r.source)
        for r in rows
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            for batch in chunked(values, UPSERT_BATCH_SIZE):
                execute_values(cur, sql, batch)
        conn.commit()


# ---------------------------------------------------------------------------
# Walk-forward simulation
# ---------------------------------------------------------------------------

def _simulate(cfg: TideConfig, close: pd.DataFrame, raw_returns: pd.DataFrame, start_date: str) -> tuple[list[BackfillRow], int]:
    """Day-by-day walk-forward simulation. Returns (rows, error_count)."""
    fallback = cfg.cash_equivalent
    alpha1_cfg = cfg.as_alpha1_config()
    ranking_universe = set(cfg.universe)
    cash_symbols = {cfg.cash_equivalent} if cfg.cash_equivalent else set()
    benchmark_symbols = {cfg.benchmark} if cfg.benchmark else set()
    empty_series = pd.Series(dtype=float)

    start_idx = int(close.index.searchsorted(pd.Timestamp(start_date)))
    if start_idx < 2:
        start_idx = 2

    current_holdings: list[str] = []
    peak_since_entry: dict[str, float] = {}
    portfolio_peak_equity = 1.0
    circuit_break_cooldown = 0

    equity = 1.0
    rows: list[BackfillRow] = []
    errors = 0

    for i in range(start_idx, len(close)):
        try:
            prev_i = i - 1

            ranked_trailing = tm.trailing_return(close, i, cfg.lookback_days, ranking_universe)
            cash_trailing = tm.trailing_return(close, i, cfg.lookback_days, cash_symbols)
            benchmark_trailing = (
                tm.trailing_return(close, i, cfg.lookback_days, benchmark_symbols) if benchmark_symbols else empty_series
            )

            risk_off, _risk_reason = _evaluate_kill_switch(
                config=alpha1_cfg, ranked_trailing=ranked_trailing,
                benchmark_trailing=benchmark_trailing, cash_trailing=cash_trailing,
            )

            if risk_off and fallback:
                naive_weights = {fallback: 1.0}
            else:
                trailing_vol = tm.trailing_volatility(raw_returns, i, cfg.lookback_days, ranking_universe)
                score = tm.composite_score(ranked_trailing, trailing_vol)
                if score.empty:
                    naive_weights = {fallback: 1.0} if fallback else {}
                else:
                    selected = tm.select_top_n_membership_stable(
                        score.to_dict(), cfg.top_n, current_holdings, cfg.incumbency_margin
                    )
                    naive_weights = {sym: 1.0 / len(selected) for sym in selected} if selected else {}

            guarded_weights, circuit_break_cooldown, portfolio_peak_equity, _stopped_out = tm.apply_guardrails(
                naive_weights, close, i, peak_since_entry, equity, portfolio_peak_equity, fallback,
                circuit_break_cooldown, stop_loss_pct=cfg.stop_loss_pct, circuit_breaker_pct=cfg.circuit_breaker_pct,
                circuit_breaker_cooldown_days=cfg.circuit_breaker_cooldown_days,
            )
            current_holdings = list(guarded_weights.keys())

            # -- Apply today's realized return to yesterday's weights -------
            # float(...) casts guard against numpy.float64 leaking into psycopg2
            # (close[...].iloc[...] returns numpy scalars, not native Python floats).
            day_return = 0.0
            for ticker, weight in guarded_weights.items():
                if ticker not in close.columns:
                    continue
                prev_price = float(close[ticker].iloc[prev_i])
                today_price = float(close[ticker].iloc[i])
                if is_finite(prev_price) and is_finite(today_price) and prev_price > 0:
                    day_return += float(weight) * (today_price / prev_price - 1.0)

            equity = float(equity * (1.0 + day_return))
            rows.append(BackfillRow(
                bot_id="", d=close.index[i].date(), equity=equity, ret=float(day_return),
                holdings_json=json.dumps({k: round(float(v), 6) for k, v in guarded_weights.items()}),
            ))
        except Exception:
            logger.exception("tide backfill: error simulating index=%d", i)
            errors += 1

    return rows, errors


def backfill_single_bot(bot: TideBotRecord) -> dict[str, Any]:
    cfg = bot.cfg
    if not cfg.universe:
        logger.warning("Skipping bot=%s: empty universe", bot.bot_id)
        return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}

    all_tickers = list(dict.fromkeys(cfg.universe + ([cfg.cash_equivalent] if cfg.cash_equivalent else []) + ([cfg.benchmark] if cfg.benchmark else [])))
    warmup_days = max(cfg.lookback_days * WARMUP_BUFFER_MULTIPLIER, WARMUP_BUFFER_MIN_DAYS) + WARMUP_SAFETY_MARGIN_DAYS
    start_d = date.fromisoformat(START_DATE)
    dl_start = (start_d - timedelta(days=warmup_days)).isoformat()

    # bots/tide.py's _download_prices anchors its start date to "today"; the
    # backfill instead needs an explicit warm-up window anchored to
    # START_DATE, so fetch directly rather than reusing that helper.
    from market_data_provider import get_daily_bars
    raw_bars = get_daily_bars(all_tickers, dl_start, END_DATE, adjusted=True)
    if raw_bars is None or raw_bars.empty:
        logger.warning("Skipping bot=%s: no price data", bot.bot_id)
        return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}

    if isinstance(raw_bars.columns, pd.MultiIndex):
        if "Close" not in raw_bars.columns.get_level_values(0):
            return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}
        close = raw_bars["Close"].copy()
    else:
        if "Close" not in raw_bars.columns:
            return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}
        close = raw_bars[["Close"]].rename(columns={"Close": all_tickers[0]})

    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    close = close[~close.index.duplicated(keep="last")]
    close.columns = [str(c).upper() for c in close.columns]
    raw_returns = close.pct_change(fill_method=None)

    rows, errors = _simulate(cfg, close, raw_returns, START_DATE)
    for r in rows:
        r.bot_id = bot.bot_id
        r.origin = bot.origin

    upsert_backfill_rows(rows)
    return {"bot_id": bot.bot_id, "processed_days": len(rows), "errors": errors, "skipped": False}


def main() -> None:
    bots = fetch_active_tide_bots()
    logger.info("tide backfill: %d active bot(s)", len(bots))
    for bot in bots:
        result = backfill_single_bot(bot)
        logger.info("tide backfill result: %s", result)


if __name__ == "__main__":
    main()
