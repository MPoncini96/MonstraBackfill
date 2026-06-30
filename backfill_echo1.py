#!/usr/bin/env python
"""
Echo1 backfill: EchoNetworkPairs lead-lag pair strategy (from trading.echo1).
Writes daily rows to trading.bot_equity with bot_type = echo1.

The backfill simulates the live strategy exactly as the notebook does:
- Monthly pair discovery (no cache — full history available)
- Daily network scoring → top-N score-proportional allocation
- Market filter (VOO SMA-{market_sma_period})
- Drawdown guard (pause at {max_drawdown_pause}, recover when
  drawdown < max_drawdown_pause * drawdown_recovery_pct after cooldown_days)
- Score-proportional weights at target_gross_exposure

Timing convention (matches other Monstra backfills):
- Holdings for date t are selected using data available through t-1.
- Return applied is the close-to-close return on date t.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Sequence

import numpy as np
import pandas as pd
from psycopg2.extras import Json as PgJson, RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env

load_env()

from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn, get_strategy_origin
from bot_identity import BOT_TYPE_ECHO1
from bots.echo1 import (
    Echo1Config,
    discover_pairs,
    build_network_scores,
    echo1_config_from_db_dict,
    download_echo1_prices,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

START_DATE = "2025-01-01"
END_DATE   = date.today().isoformat()

# Extra calendar days before START_DATE to warm up pair discovery lookback.
# discovery_lookback is in trading days (~2 years = 504 td ≈ 730 calendar days).
# Add 100 extra for safety.
WARMUP_CALENDAR_DAYS = 1200

UPSERT_BATCH_SIZE = 500

LOG_LEVEL = os.getenv("ECHO1_BACKFILL_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("echo1_backfill")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Echo1BotRecord:
    bot_id: str
    cfg: Echo1Config


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str
    bot_type: str = BOT_TYPE_ECHO1
    origin: str | None = None
    source: str = BOT_EQUITY_SOURCE_BACKFILL


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def chunked(seq: Sequence[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_active_echo1_bots() -> list[Echo1BotRecord]:
    records: list[Echo1BotRecord] = []
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM trading.echo1 WHERE is_active = TRUE ORDER BY bot_id"
                )
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Could not load trading.echo1: %s", exc)
        return []

    for row in rows:
        d = dict(row)
        bid = str(d.get("bot_id", "")).strip().lower()
        if not bid:
            continue
        cfg = echo1_config_from_db_dict(d)
        records.append(Echo1BotRecord(bot_id=bid, cfg=cfg))

    return records


def fetch_previous_equity(bot_id: str, before_date: str) -> float:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT equity FROM trading.bot_equity
                WHERE bot_id = %s AND bot_type = %s AND d < %s
                ORDER BY d DESC LIMIT 1
                """,
                (bot_id, BOT_TYPE_ECHO1, before_date),
            )
            row = cur.fetchone()
    return float(row[0]) if row and is_finite(row[0]) else 1.0


def upsert_backfill_rows(rows: list[BackfillRow]) -> None:
    if not rows:
        return
    bot_id   = rows[0].bot_id
    bot_type = rows[0].bot_type
    min_d    = min(r.d for r in rows)
    max_d    = max(r.d for r in rows)
    logger.info(
        "echo1 backfill upsert: bot_id=%s bot_type=%s rows=%d min_d=%s max_d=%s",
        bot_id, bot_type, len(rows), min_d, max_d,
    )
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
        (r.bot_id, r.bot_type, r.d, r.equity, r.ret,
         PgJson(json.loads(r.holdings_json)), r.origin, r.source)
        for r in rows
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            for batch in chunked(values, UPSERT_BATCH_SIZE):
                execute_values(cur, sql, batch)
        conn.commit()


# ---------------------------------------------------------------------------
# Pairs-cache helpers  (keyed by (bot_id, month) for historical months)
# ---------------------------------------------------------------------------

_PAIRS_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS trading.echo1_pairs_cache (
    bot_id      TEXT        NOT NULL,
    month       DATE        NOT NULL,
    pairs       JSONB       NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bot_id, month)
)
"""


def ensure_pairs_cache_table() -> None:
    """Create trading.echo1_pairs_cache if it does not already exist."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_PAIRS_CACHE_DDL)
            conn.commit()
    except Exception as exc:
        logger.warning("echo1 backfill: could not ensure pairs cache table: %s", exc)


def load_backfill_pairs_cache(bot_id: str, month: date) -> list[dict] | None:
    """
    Return cached pairs for *month* (first day of that month), or None on miss.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pairs FROM trading.echo1_pairs_cache
                    WHERE bot_id = %s AND month = %s
                    """,
                    (bot_id, month.isoformat()),
                )
                row = cur.fetchone()
        if not row:
            return None
        raw = row[0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        return raw if isinstance(raw, list) else None
    except Exception as exc:
        logger.warning(
            "echo1 backfill: load_pairs_cache failed bot=%s month=%s: %s",
            bot_id, month, exc,
        )
        return None


def save_backfill_pairs_cache(bot_id: str, month: date, pairs: list[dict]) -> None:
    """
    Upsert *pairs* for *month* into trading.echo1_pairs_cache.
    """
    try:
        from psycopg2.extras import Json as PgJson
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trading.echo1_pairs_cache
                        (bot_id, month, pairs, computed_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (bot_id, month) DO UPDATE SET
                        pairs       = EXCLUDED.pairs,
                        computed_at = EXCLUDED.computed_at
                    """,
                    (bot_id, month.isoformat(), PgJson(pairs)),
                )
            conn.commit()
    except Exception as exc:
        logger.warning(
            "echo1 backfill: save_pairs_cache failed bot=%s month=%s: %s",
            bot_id, month, exc,
        )


# ---------------------------------------------------------------------------
# Single-bot backfill
# ---------------------------------------------------------------------------

def backfill_single_bot(bot: Echo1BotRecord) -> dict[str, Any]:
    cfg = bot.cfg
    logger.info(
        "echo1 backfill: bot_id=%s universe=%d fallback=%s portfolio_size=%d lookback=%d",
        bot.bot_id, len(cfg.universe), cfg.fallback_ticker,
        cfg.portfolio_size, cfg.discovery_lookback,
    )

    if not cfg.universe:
        logger.warning("Skipping bot=%s: empty universe", bot.bot_id)
        return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}

    # Ensure the pairs-cache table exists before any cache reads/writes.
    ensure_pairs_cache_table()

    # ── Download full price history ──────────────────────────────────────────
    all_tickers = list(dict.fromkeys(cfg.universe + [cfg.fallback_ticker]))
    # For the backfill we need data from START_DATE - warmup through END_DATE.
    start_d  = date.fromisoformat(START_DATE)
    end_d    = date.fromisoformat(END_DATE)
    dl_start = (start_d - timedelta(days=WARMUP_CALENDAR_DAYS)).isoformat()

    raw = get_conn()  # just a connectivity check; actual fetch below
    raw = None        # reset; use download helper

    from market_data_provider import get_daily_bars
    raw_bars = get_daily_bars(all_tickers, dl_start, END_DATE, adjusted=True)

    if raw_bars is None or raw_bars.empty:
        logger.warning("Skipping bot=%s: no price data", bot.bot_id)
        return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}

    # Extract close prices DataFrame (tickers as columns)
    if isinstance(raw_bars.columns, pd.MultiIndex):
        if "Close" not in raw_bars.columns.get_level_values(0):
            logger.warning("Skipping bot=%s: no Close column", bot.bot_id)
            return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}
        close_all = raw_bars["Close"].copy()
    else:
        if "Close" not in raw_bars.columns:
            logger.warning("Skipping bot=%s: no Close column (single ticker)", bot.bot_id)
            return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}
        close_all = raw_bars[["Close"]].rename(columns={"Close": all_tickers[0]})

    close_all = close_all.dropna(how="all").sort_index()
    close_all.index = pd.to_datetime(close_all.index)
    close_all = close_all[~close_all.index.duplicated(keep="last")]
    close_all.columns = [str(c).upper() for c in close_all.columns]

    # Dates to backfill
    backfill_dates = close_all.loc[START_DATE:END_DATE].index
    if backfill_dates.empty:
        logger.warning("Skipping bot=%s: no trading dates in backfill window", bot.bot_id)
        return {"bot_id": bot.bot_id, "processed_days": 0, "errors": 0, "skipped": True}

    all_dates = close_all.index
    origin    = get_strategy_origin(bot.bot_id, BOT_TYPE_ECHO1)

    # ── Per-date simulation ──────────────────────────────────────────────────
    equity          = fetch_previous_equity(bot.bot_id, START_DATE)
    high_watermark  = equity
    holdings: dict[str, float] = {}          # current target weights
    drawdown_pause_active = False
    drawdown_pause_date: date | None = None
    discovered_pairs: list[dict] = []
    last_pair_refresh_month: int | None = None
    rows_to_upsert: list[BackfillRow] = []
    processed = 0
    errors    = 0

    # Minimum history index needed before we can run discovery
    warmup_end_idx = cfg.discovery_lookback + 50

    for date_ts in backfill_dates:
        date_idx = all_dates.get_loc(date_ts)
        trade_date = date_ts.date()

        if date_idx < warmup_end_idx:
            rows_to_upsert.append(_make_row(
                bot.bot_id, trade_date, equity, 0.0, {}, "warmup", origin,
            ))
            processed += 1
            continue

        # Signals are generated using data through t-1, but portfolio equity for
        # date t realizes the close-to-close return of the holdings carried into t.
        ret = _compute_return(holdings, close_all, date_idx, all_dates)
        equity *= 1.0 + ret
        high_watermark = max(high_watermark, equity)
        drawdown = 1 - equity / high_watermark if high_watermark > 0 else 0.0

        def _fallback_weights(pct: float) -> dict[str, float]:
            return {cfg.fallback_ticker: pct} if cfg.fallback_ticker in close_all.columns else {}

        # ── Drawdown guard ───────────────────────────────────────────────────
        if drawdown >= cfg.max_drawdown_pause:
            if not drawdown_pause_active:
                drawdown_pause_active = True
                drawdown_pause_date   = trade_date
            holdings = _fallback_weights(cfg.target_gross_exposure)
            rows_to_upsert.append(_make_row(
                bot.bot_id, trade_date, equity, ret, holdings,
                f"drawdown_pause({drawdown:.1%})", origin,
            ))
            processed += 1
            continue

        if drawdown_pause_active:
            days_paused = (trade_date - drawdown_pause_date).days if drawdown_pause_date else 0
            recovery_threshold = cfg.max_drawdown_pause * cfg.drawdown_recovery_pct
            if drawdown < recovery_threshold and days_paused >= cfg.drawdown_cooldown_days:
                drawdown_pause_active = False
            else:
                holdings = _fallback_weights(cfg.target_gross_exposure)
                rows_to_upsert.append(_make_row(
                    bot.bot_id, trade_date, equity, ret, holdings,
                    f"drawdown_recovering({drawdown:.1%})", origin,
                ))
                processed += 1
                continue

        # ── Market filter ────────────────────────────────────────────────────
        if cfg.use_market_filter and cfg.fallback_ticker in close_all.columns:
            hist_slice = close_all.iloc[
                max(0, date_idx - cfg.market_sma_period - 5): date_idx + 1
            ]
            fb_px = hist_slice[cfg.fallback_ticker].dropna()
            if len(fb_px) >= cfg.market_sma_period:
                if float(fb_px.iloc[-1]) < float(fb_px.tail(cfg.market_sma_period).mean()):
                    holdings = _fallback_weights(cfg.market_filter_fallback_pct)
                    rows_to_upsert.append(_make_row(
                        bot.bot_id, trade_date, equity, ret, holdings,
                        "market_filter", origin,
                    ))
                    processed += 1
                    continue

        # ── Monthly pair discovery (persistent cache) ────────────────────────
        if last_pair_refresh_month is None or date_ts.month != last_pair_refresh_month:
            month_key = date(date_ts.year, date_ts.month, 1)
            cached = load_backfill_pairs_cache(bot.bot_id, month_key)
            if cached is not None:
                discovered_pairs = cached
                logger.info(
                    "echo1 cache hit  bot=%s month=%s pairs=%d",
                    bot.bot_id, month_key, len(discovered_pairs),
                )
            else:
                cw = close_all.iloc[
                    max(0, date_idx - cfg.discovery_lookback - 30): date_idx + 1
                ]
                discovered_pairs = discover_pairs(cw, cfg)
                if len(discovered_pairs) < 10:
                    discovered_pairs = discover_pairs(cw, cfg, min_corr=cfg.min_avg_abs_corr_loose)
                save_backfill_pairs_cache(bot.bot_id, month_key, discovered_pairs)
                logger.info(
                    "echo1 cache miss bot=%s month=%s pairs=%d → saved",
                    bot.bot_id, month_key, len(discovered_pairs),
                )
            last_pair_refresh_month = date_ts.month

        # ── Score and allocate ───────────────────────────────────────────────
        if not discovered_pairs:
            holdings = _fallback_weights(cfg.target_gross_exposure)
            rows_to_upsert.append(_make_row(
                bot.bot_id, trade_date, equity, ret, holdings, "no_pairs", origin,
            ))
            processed += 1
            continue

        max_lb = max(
            (int(p["lag_days"]) + int(p["return_period"]) + 5 for p in discovered_pairs),
            default=20,
        )
        score_window = close_all.iloc[max(0, date_idx - max_lb): date_idx + 1]
        network_scores = build_network_scores(score_window, discovered_pairs, cfg)

        ranked   = sorted(network_scores.values(), key=lambda x: x["final_score"], reverse=True)
        selected = [x for x in ranked if x["final_score"] > 0][: cfg.portfolio_size]

        if not selected:
            holdings = _fallback_weights(cfg.target_gross_exposure)
            rows_to_upsert.append(_make_row(
                bot.bot_id, trade_date, equity, ret, holdings, "no_signal", origin,
            ))
            processed += 1
            continue

        total_score = sum(x["final_score"] for x in selected)
        new_holdings: dict[str, float] = {
            x["ticker"]: round(
                (x["final_score"] / total_score) * cfg.target_gross_exposure, 6
            )
            for x in selected
        }

        holdings = new_holdings

        payload = dict(new_holdings)
        payload["_meta"] = {
            "is_backtest": True,
            "type": "echo1",
            "regime": "active",
            "num_pairs": len(discovered_pairs),
            "top_pick": selected[0]["ticker"],
            "portfolio_size": cfg.portfolio_size,
            "discovery_lookback": cfg.discovery_lookback,
        }

        rows_to_upsert.append(BackfillRow(
            bot_id=bot.bot_id,
            d=trade_date,
            equity=float(equity),
            ret=float(ret),
            holdings_json=json.dumps(payload),
            bot_type=BOT_TYPE_ECHO1,
            origin=origin,
            source=BOT_EQUITY_SOURCE_BACKFILL,
        ))
        processed += 1

        # Batch flush
        if len(rows_to_upsert) >= UPSERT_BATCH_SIZE:
            upsert_backfill_rows(rows_to_upsert)
            rows_to_upsert = []

    # Final flush
    upsert_backfill_rows(rows_to_upsert)

    final_pct = (equity - 1.0) * 100.0
    logger.info(
        "echo1 backfill complete: bot=%s processed_days=%d final_equity=%.6f "
        "final_return_pct=%.2f errors=%d",
        bot.bot_id, processed, equity, final_pct, errors,
    )
    return {
        "bot_id": bot.bot_id,
        "processed_days": processed,
        "final_equity": equity,
        "final_return_pct": final_pct,
        "errors": errors,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_return(
    holdings: dict[str, float],
    close_all: pd.DataFrame,
    date_idx: int,
    all_dates: pd.DatetimeIndex,
) -> float:
    """
    Compute the weighted return from date_idx-1 to date_idx using *holdings*
    (which are the previous day's target weights).
    """
    if not holdings or date_idx < 1:
        return 0.0
    prev_ts   = all_dates[date_idx - 1]
    curr_ts   = all_dates[date_idx]
    total_ret = 0.0
    for ticker, weight in holdings.items():
        if ticker not in close_all.columns:
            continue
        p_prev = close_all.loc[prev_ts, ticker] if prev_ts in close_all.index else np.nan
        p_curr = close_all.loc[curr_ts, ticker] if curr_ts in close_all.index else np.nan
        if pd.isna(p_prev) or pd.isna(p_curr) or float(p_prev) <= 0:
            continue
        total_ret += float(weight) * (float(p_curr) / float(p_prev) - 1.0)
    return total_ret


def _make_row(
    bot_id: str,
    trade_date: date,
    equity: float,
    ret: float,
    holdings: dict[str, float],
    regime: str,
    origin: str | None,
) -> BackfillRow:
    payload = dict(holdings)
    payload["_meta"] = {"is_backtest": True, "type": "echo1", "regime": regime}
    return BackfillRow(
        bot_id=bot_id,
        d=trade_date,
        equity=float(equity),
        ret=float(ret),
        holdings_json=json.dumps(payload),
        bot_type=BOT_TYPE_ECHO1,
        origin=origin,
        source=BOT_EQUITY_SOURCE_BACKFILL,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def backfill_echo1() -> None:
    logger.info("Loading active Echo1 bots from trading.echo1")
    bots = fetch_active_echo1_bots()

    if not bots:
        logger.warning("No active bots found in trading.echo1")
        return

    logger.info(
        "Found %d active bot(s). Backfilling from %s to %s (warmup from ~%s)",
        len(bots), START_DATE, END_DATE,
        (date.fromisoformat(START_DATE) - timedelta(days=WARMUP_CALENDAR_DAYS)).isoformat(),
    )

    results: list[dict[str, Any]] = []
    for bot in bots:
        try:
            results.append(backfill_single_bot(bot))
        except Exception as exc:
            logger.exception("Fatal error in bot=%s: %s", bot.bot_id, exc)
            results.append({
                "bot_id": bot.bot_id,
                "processed_days": 0,
                "final_equity": None,
                "final_return_pct": None,
                "errors": 1,
                "skipped": True,
            })

    logger.info("========== Echo1 Backfill Summary ==========")
    for r in results:
        if r.get("skipped"):
            logger.info("bot=%s status=SKIPPED processed_days=%s errors=%s",
                        r["bot_id"], r.get("processed_days"), r.get("errors"))
        else:
            logger.info(
                "bot=%s processed_days=%d final_equity=%.6f final_return_pct=%.2f errors=%d",
                r["bot_id"], r["processed_days"], r["final_equity"],
                r["final_return_pct"], r["errors"],
            )
    logger.info("Echo1 backfill complete.")

if __name__ == "__main__":
    backfill_echo1()


