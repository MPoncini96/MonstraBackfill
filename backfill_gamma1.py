#!/usr/bin/env python
"""
Gamma1 backfill: weighted binary-score selector with hysteresis (from trading.gamma1).
Writes daily rows to trading.bot_equity with bot_type = gamma1.

Uses the app trading.gamma1 schema (see DB migration): universe array, rebalance_frequency,
max_positions, entry/exit/hold_bonus, weighting_style, and use_* feature flags.
Benchmark defaults to VOO when not stored. Optional columns (fallback_ticker, history_period,
transaction_cost_bps, benchmark) override defaults when present.

Price history is downloaded with a warm-up window before START_DATE so 252-day
features and EMAs are stable by the backfill start.
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
from psycopg2.extras import RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env

load_env()

from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn, get_strategy_origin
from bot_identity import BOT_TYPE_GAMMA1
from bots.gamma1 import (
    Gamma1Config,
    build_gamma1_score_table,
    download_gamma1_prices,
    gamma1_config_from_db_dict,
    run_weighted_points_backtest,
)

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

WARMUP_CALENDAR_DAYS = 450
UPSERT_BATCH_SIZE = 500

LOG_LEVEL = os.getenv("GAMMA1_BACKFILL_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gamma1_backfill")


@dataclass
class Gamma1BotConfig:
    bot_id: str
    name: str
    cfg: Gamma1Config


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str
    bot_type: str = BOT_TYPE_GAMMA1
    origin: str | None = None
    source: str = BOT_EQUITY_SOURCE_BACKFILL


def is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def chunked(seq: Sequence[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def fetch_active_gamma1_bots() -> list[Gamma1BotConfig]:
    configs: list[Gamma1BotConfig] = []

    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM trading.gamma1
                    WHERE is_active = TRUE
                    ORDER BY bot_id
                    """
                )
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Could not load trading.gamma1: %s", exc)
        return []

    for row in rows:
        d = dict(row)
        bid = str(d.get("bot_id", "")).strip().lower()
        if not bid:
            continue
        name = str(d.get("origin") or d.get("name") or bid)
        cfg = gamma1_config_from_db_dict(d)
        configs.append(Gamma1BotConfig(bot_id=bid, name=name, cfg=cfg))

    return configs


def fetch_previous_equity(bot_id: str, start_date: str) -> float:
    query = """
        SELECT equity
        FROM trading.bot_equity
        WHERE bot_id = %s AND bot_type = %s AND d < %s
        ORDER BY d DESC
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (bot_id, BOT_TYPE_GAMMA1, start_date))
            row = cur.fetchone()

    if row and is_finite_number(row[0]):
        return float(row[0])
    return 1.0


def fetch_previous_holdings(bot_id: str, start_date: str) -> dict[str, float]:
    query = """
        SELECT holdings
        FROM trading.bot_equity
        WHERE bot_id = %s AND bot_type = %s AND d < %s
        ORDER BY d DESC
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (bot_id, BOT_TYPE_GAMMA1, start_date))
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


def price_download_start() -> str:
    start_d = date.fromisoformat(START_DATE)
    return (start_d - timedelta(days=WARMUP_CALENDAR_DAYS)).isoformat()


def backfill_single_bot(bot: Gamma1BotConfig) -> dict[str, Any]:
    c = bot.cfg
    logger.info(
        "Starting bot=%s name=%s universe=%d benchmark=%s rebalance=%s max_pos=%d entry=%.2f exit=%.2f",
        bot.bot_id,
        bot.name,
        len(c.universe),
        c.benchmark,
        c.rebalance,
        c.max_positions,
        c.entry_score,
        c.exit_score,
    )

    if not c.universe:
        logger.warning("Skipping bot=%s because universe is empty", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    tickers = list(dict.fromkeys(c.universe + [c.benchmark]))
    dl_start = price_download_start()

    try:
        data = download_gamma1_prices(tickers, dl_start, END_DATE)
    except Exception as exc:
        logger.exception("Download failed bot=%s: %s", bot.bot_id, exc)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 1,
            "skipped": True,
        }

    if data.empty or c.benchmark not in data.columns:
        logger.warning("Skipping bot=%s: missing price data", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    univ = [t for t in c.universe if t in data.columns]
    if not univ:
        logger.warning("Skipping bot=%s: universe not in download", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    prices = data[univ].copy().dropna(how="all").sort_index()
    bm = data[c.benchmark].reindex(prices.index)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    score_table = build_gamma1_score_table(prices, bm, c.feature_flags)
    raw_scores = score_table["total"]

    prev_hold = fetch_previous_holdings(bot.bot_id, START_DATE)
    init_syms = [s for s in prev_hold if s in prices.columns and prev_hold[s] > 1e-12]
    init_w = pd.Series(0.0, index=prices.columns)
    for s, w in prev_hold.items():
        if s in init_w.index and is_finite_number(w):
            init_w.loc[s] = float(w)

    result = run_weighted_points_backtest(
        prices,
        returns,
        raw_scores,
        c,
        record_weights=True,
        initial_holdings=init_syms if init_syms else None,
        initial_weights=init_w if prev_hold else None,
    )

    weights = result["weights"]
    net_returns = result["net_returns"]

    if weights is None or net_returns.empty:
        logger.warning("Skipping bot=%s: empty backtest output", bot.bot_id)
        return {
            "bot_id": bot.bot_id,
            "processed_days": 0,
            "final_equity": None,
            "final_return_pct": None,
            "errors": 0,
            "skipped": True,
        }

    start_d = date.fromisoformat(START_DATE)
    equity = fetch_previous_equity(bot.bot_id, START_DATE)
    origin = get_strategy_origin(bot.bot_id, BOT_TYPE_GAMMA1)
    rows_to_upsert: list[BackfillRow] = []
    processed = 0

    for ts in net_returns.index:
        if ts.date() < start_d:
            continue

        net_r = float(net_returns.loc[ts])
        equity *= 1.0 + net_r

        wrow = weights.loc[ts]
        holdings = {str(k): float(v) for k, v in wrow.items() if float(v) > 1e-12}

        payload = dict(holdings)
        payload["_meta"] = {
            "is_backtest": True,
            "type": "gamma1",
            "data_source": "yfinance",
            "timing": "rebalance_weights_held_for_daily_returns",
            "benchmark": c.benchmark,
            "rebalance": c.rebalance,
            "weighting_style": c.weighting_style,
            "max_positions": int(c.max_positions),
            "entry_score": float(c.entry_score),
            "exit_score": float(c.exit_score),
            "transaction_cost_bps": float(c.transaction_cost_bps),
            "hold_bonus": float(c.hold_bonus),
        }

        rows_to_upsert.append(
            BackfillRow(
                bot_id=bot.bot_id,
                d=ts.date(),
                equity=float(equity),
                ret=net_r,
                holdings_json=json.dumps(payload),
                bot_type=BOT_TYPE_GAMMA1,
                origin=origin,
            )
        )
        processed += 1

    upsert_backfill_rows(rows_to_upsert)
    final_pct = (equity - 1.0) * 100.0
    logger.info(
        "Completed bot=%s processed_days=%d final_equity=%.6f final_return_pct=%.2f",
        bot.bot_id,
        processed,
        equity,
        final_pct,
    )

    return {
        "bot_id": bot.bot_id,
        "processed_days": processed,
        "final_equity": equity,
        "final_return_pct": final_pct,
        "errors": 0,
        "skipped": False,
    }


def backfill_gamma1() -> None:
    logger.info("Loading active Gamma1 bots from trading.gamma1")
    bots = fetch_active_gamma1_bots()

    if not bots:
        logger.warning("No active bots found in trading.gamma1")
        return

    logger.info(
        "Found %d active bot(s). Backfilling from %s to %s (warmup from ~%s)",
        len(bots),
        START_DATE,
        END_DATE,
        price_download_start(),
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

    logger.info("========== Gamma1 Backfill Summary ==========")
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

    logger.info("Gamma1 backfill complete.")


if __name__ == "__main__":
    backfill_gamma1()
