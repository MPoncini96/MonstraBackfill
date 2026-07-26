#!/usr/bin/env python
"""
Draco backfill: multi-timeframe log-linear regression mean-reversion (from trading.draco).
Writes daily rows to trading.bot_equity with bot_type = draco.

Simulates the live strategy exactly (bots/draco.py run_draco), day by day:
- Weekly entry scans (first eligible trading day of each ISO week), daily exit checks.
- Locked regression targets computed once at entry and never recalculated.
- Market regime filter (benchmark vs SMA, N-day confirmation) and portfolio
  circuit breaker (drawdown -> cooldown), both re-derived from the equity/price
  curve being built during this same walk-forward pass (no DB round-trips
  needed mid-backfill, unlike the live runner).

Timing convention (matches other Monstra backfills, e.g. backfill_echo1.py):
- Holdings for date t are selected using data available through t-1.
- Return applied is the close-to-close return on date t.

Warm-up: downloads far enough before START_DATE to cover the longest configured
lookback (default 5Y = 1260 trading days) in full, without truncating it just
because START_DATE is recent - only the *measured* equity curve is anchored to
START_DATE.
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
from bot_identity import BOT_TYPE_DRACO
import draco_math as dm
from draco_config import DracoConfig, draco_config_from_db_dict
from bots.draco import DracoCandidate, evaluate_candidate, evaluate_position_exit, rank_candidates

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

# Extra calendar days before START_DATE to fully warm up the longest lookback
# (default 5Y = 1260 trading days ~= 1825 calendar days) plus a safety margin.
WARMUP_CALENDAR_DAYS = 2200

UPSERT_BATCH_SIZE = 500

LOG_LEVEL = os.getenv("DRACO_BACKFILL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("draco_backfill")


@dataclass
class DracoBotRecord:
    bot_id: str
    cfg: DracoConfig
    origin: str | None = None


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str
    bot_type: str = BOT_TYPE_DRACO
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

def fetch_active_draco_bots() -> list[DracoBotRecord]:
    records: list[DracoBotRecord] = []
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM trading.draco WHERE is_active = TRUE ORDER BY bot_id")
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Could not load trading.draco: %s", exc)
        return []

    for row in rows:
        d = dict(row)
        bid = str(d.get("bot_id", "")).strip().lower()
        if not bid:
            continue
        cfg = draco_config_from_db_dict(d)
        records.append(DracoBotRecord(bot_id=bid, cfg=cfg, origin=d.get("origin")))
    return records


def upsert_backfill_rows(rows: list[BackfillRow]) -> None:
    if not rows:
        return
    bot_id, bot_type = rows[0].bot_id, rows[0].bot_type
    logger.info("draco backfill upsert: bot_id=%s rows=%d", bot_id, len(rows))
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

def _simulate(cfg: DracoConfig, close: pd.DataFrame, start_date: str) -> tuple[list[BackfillRow], int]:
    """Day-by-day walk-forward simulation. Returns (rows, error_count)."""
    fallback = cfg.fallback_ticker
    start_idx = int(close.index.searchsorted(pd.Timestamp(start_date)))
    if start_idx < 2:
        start_idx = 2

    positions: dict[str, dict[str, Any]] = {}
    cooldowns: dict[str, int] = {}  # ticker -> index until which re-entry is blocked (exclusive)
    regime_state = "risk_on"
    regime_pending: str | None = None
    regime_confirm_count = 0
    breaker_active = False
    breaker_cooldown_remaining = 0
    peak_equity = 1.0
    last_entry_scan_iso_week: list[int] | None = None

    equity = 1.0
    rows: list[BackfillRow] = []
    errors = 0

    for i in range(start_idx, len(close)):
        try:
            prev_i = i - 1
            decision_prices = close.iloc[: i]  # known through t-1 (no look-ahead)
            trading_date = close.index[prev_i]
            iso_week = list(trading_date.isocalendar()[:2])
            is_entry_scan_day = cfg.scan_cadence != "weekly" or last_entry_scan_iso_week != iso_week

            # -- Market regime (benchmark vs SMA, N-day confirmation) -------
            regime = "risk_on"
            if cfg.use_market_regime_filter and cfg.market_regime_ticker in decision_prices.columns:
                regime_series = decision_prices[cfg.market_regime_ticker].dropna()
                if len(regime_series) >= cfg.market_sma_period:
                    sma = float(regime_series.tail(cfg.market_sma_period).mean())
                    current = float(regime_series.iloc[-1])
                    raw_signal = "risk_off" if current < sma else "risk_on"
                    if raw_signal == regime_state:
                        regime_pending, regime_confirm_count = None, 0
                    else:
                        if regime_pending == raw_signal:
                            regime_confirm_count += 1
                        else:
                            regime_pending, regime_confirm_count = raw_signal, 1
                        if regime_confirm_count >= cfg.regime_confirmation_days:
                            regime_state, regime_pending, regime_confirm_count = raw_signal, None, 0
            regime = regime_state

            # -- Circuit breaker (drawdown -> cooldown), local equity curve --
            peak_equity = max(peak_equity, equity)
            drawdown = (1 - equity / peak_equity) if peak_equity > 0 else 0.0
            if breaker_active:
                breaker_cooldown_remaining = max(0, breaker_cooldown_remaining - 1)
                if breaker_cooldown_remaining <= 0:
                    breaker_active = False
                    peak_equity = equity
            elif cfg.use_portfolio_circuit_breaker and drawdown >= cfg.maximum_portfolio_drawdown:
                breaker_active = True
                breaker_cooldown_remaining = cfg.circuit_breaker_cooldown_trading_days

            risk_off = regime == "risk_off"
            liquidate = breaker_active or (risk_off and cfg.liquidate_to_fallback_during_risk_off)

            if liquidate:
                positions = {}
                weights = {fallback: cfg.target_gross_exposure} if cfg.hold_fallback_when_empty else {}
            else:
                # -- Exit evaluation (every day) -----------------------------
                for ticker, pos in list(positions.items()):
                    if ticker not in decision_prices.columns:
                        continue
                    series = decision_prices[ticker].dropna()
                    if series.empty:
                        continue
                    current_price = float(series.iloc[-1])
                    entry_price = float(pos["entry_price"])
                    holding_days = prev_i - int(pos["entry_index"])
                    target = dm.LockedTarget(
                        label=pos["target_label"], lookback=int(pos["target_lookback"]),
                        slope=float(pos["target_slope"]), intercept=float(pos["target_intercept"]),
                        endpoint_index=int(pos["target_endpoint_index"]), r_squared=float(pos["target_r_squared"]),
                        entry_price=entry_price, entry_target_price=float(pos["entry_target_price"]),
                        entry_upside=float(pos["entry_upside"]),
                    )
                    exit_reason, _position_return, _projected_price = evaluate_position_exit(
                        current_price, entry_price, holding_days, target, cfg
                    )

                    if exit_reason:
                        del positions[ticker]
                        if cfg.symbol_reentry_cooldown_days > 0:
                            cooldowns[ticker] = prev_i + cfg.symbol_reentry_cooldown_days

                cooldowns = {t: until for t, until in cooldowns.items() if until > prev_i}

                # -- Entry scan (weekly) -------------------------------------
                if is_entry_scan_day and len(positions) < cfg.max_positions:
                    candidates: list[DracoCandidate] = []
                    for ticker in cfg.universe:
                        if ticker in positions or ticker in cooldowns or ticker not in decision_prices.columns:
                            continue
                        series = decision_prices[ticker].dropna()
                        if series.empty:
                            continue
                        candidate = evaluate_candidate(ticker, series, cfg)
                        if candidate is not None:
                            candidates.append(candidate)

                    ranked = rank_candidates(candidates)
                    free_slots = cfg.max_positions - len(positions)
                    take = min(free_slots, cfg.max_new_positions_per_scan, len(ranked))
                    for candidate in ranked[:take]:
                        t = candidate.target
                        positions[candidate.ticker] = {
                            "target_label": t.label, "target_lookback": t.lookback, "target_slope": t.slope,
                            "target_intercept": t.intercept, "target_endpoint_index": t.endpoint_index,
                            "target_r_squared": t.r_squared, "entry_price": t.entry_price,
                            "entry_target_price": t.entry_target_price, "entry_upside": t.entry_upside,
                            "entry_index": prev_i,
                        }
                    last_entry_scan_iso_week = iso_week

                if positions:
                    weight = cfg.target_gross_exposure / len(positions)
                    weights = {ticker: weight for ticker in positions}
                elif cfg.hold_fallback_when_empty:
                    weights = {fallback: cfg.target_gross_exposure}
                else:
                    weights = {}

            # -- Apply today's realized return to yesterday's weights -------
            # float(...) casts guard against numpy.float64 leaking into psycopg2
            # (close[...].iloc[...] returns numpy scalars, not native Python floats).
            day_return = 0.0
            for ticker, weight in weights.items():
                if ticker not in close.columns:
                    continue
                prev_price = float(close[ticker].iloc[prev_i])
                today_price = float(close[ticker].iloc[i])
                if is_finite(prev_price) and is_finite(today_price) and prev_price > 0:
                    day_return += float(weight) * (today_price / prev_price - 1.0)

            equity = float(equity * (1.0 + day_return))
            rows.append(BackfillRow(
                bot_id="", d=close.index[i].date(), equity=equity, ret=float(day_return),
                holdings_json=json.dumps({k: round(float(v), 6) for k, v in weights.items()}),
            ))
        except Exception:
            logger.exception("draco backfill: error simulating index=%d", i)
            errors += 1

    return rows, errors


def backfill_single_bot(bot: DracoBotRecord) -> dict[str, Any]:
    cfg = bot.cfg
    if not cfg.universe:
        logger.warning("Skipping bot=%s: empty universe", bot.bot_id)
        return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}

    all_tickers = list(dict.fromkeys(cfg.universe + [cfg.market_regime_ticker, cfg.fallback_ticker]))
    start_d = date.fromisoformat(START_DATE)
    dl_start = (start_d - timedelta(days=WARMUP_CALENDAR_DAYS)).isoformat()

    # download_draco_prices anchors its start date to "today"; the backfill
    # instead needs an explicit warm-up window anchored to START_DATE, so
    # fetch directly rather than reusing that helper.
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

    rows, errors = _simulate(cfg, close, START_DATE)
    for r in rows:
        r.bot_id = bot.bot_id
        r.origin = bot.origin

    upsert_backfill_rows(rows)
    return {"bot_id": bot.bot_id, "processed_days": len(rows), "errors": errors, "skipped": False}


def main() -> None:
    bots = fetch_active_draco_bots()
    logger.info("draco backfill: %d active bot(s)", len(bots))
    for bot in bots:
        result = backfill_single_bot(bot)
        logger.info("draco backfill result: %s", result)


if __name__ == "__main__":
    main()
