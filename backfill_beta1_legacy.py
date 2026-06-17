#!/usr/bin/env python
# =============================================================================
# LEGACY SNAPSHOT — backfill_beta1_legacy.py
# =============================================================================
# This file is a preserved copy of backfill_beta1.py as it existed BEFORE the
# composite-scoring redesign of the Beta1 strategy (captured 2026-06-16).
#
# PURPOSE:
#   - Historical reference for the original confirmation-threshold approach.
#   - Comparison baseline against the new composite scoring framework.
#   - Rollback target if the new strategy needs to be reverted.
#
# DO NOT MODIFY THIS FILE.
# Future Beta1 development must target backfill_beta1.py, NOT this file.
# Changes here would corrupt the historical snapshot.
# =============================================================================

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
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beta1_legacy_shared import (
    clean_ticker as _clean_ticker,
    compute_weekly_returns,
    download_weekly_close,
    normalize_universe as _normalize_universe,
    resolve_beta1_state_start_date,
    run_beta1_reconstruction,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from env_loader import load_env

load_env()

from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn, get_strategy_origin
from bot_identity import BOT_TYPE_BETA1

START_DATE = resolve_beta1_state_start_date()
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
logger = logging.getLogger("beta1_legacy_backfill")


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

def run_backtest(
    universe: list[str],
    start: str = "2020-01-01",
    end: str = "2026-01-01",
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
    max_holdings: int = MAX_HOLDINGS,
    fallback_ticker: str = FALLBACK_TICKER,
) -> BacktestResult:
    reconstruction = run_beta1_reconstruction(
        universe=universe,
        start=start,
        end=end,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        max_holdings=max_holdings,
        fallback_ticker=fallback_ticker,
    )
    return BacktestResult(
        weekly_returns=reconstruction.weekly_returns,
        equity_curve=reconstruction.equity_curve,
        weights_history=reconstruction.weights_history,
        signal_table=reconstruction.signal_table,
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
        "Starting beta1_legacy backfill bot_id=%s state_start=%s end=%s universe=%s max_holdings=%d entry=%.4f exit=%.4f fallback=%s",
        bot.bot_id,
        START_DATE,
        END_DATE,
        ",".join(bot.universe),
        bot.max_holdings,
        bot.entry_threshold,
        bot.exit_threshold,
        bot.fallback_ticker,
    )

    if not bot.universe:
        logger.warning(
            "Skipping beta1_legacy backfill bot_id=%s state_start=%s because universe is empty",
            bot.bot_id,
            START_DATE,
        )
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
            "state_start_date": START_DATE,
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
                source=BOT_EQUITY_SOURCE_BACKFILL,
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
    logger.info("Loading active Beta1 bots from trading.beta1 state_start=%s end=%s", START_DATE, END_DATE)
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

    logger.info("========== Beta1 Legacy Backfill Summary ==========")
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

    logger.info("Beta1 legacy backfill complete.")


if __name__ == "__main__":
    backfill_beta1()
