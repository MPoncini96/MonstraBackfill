#!/usr/bin/env python
"""
backfill_beta1_v2.py
====================
Backfill runner for Phantom v2 (composite scoring).

Reads active bots from trading.beta1_v2, runs the composite scoring
reconstruction over the configured date range, and writes daily equity
rows to trading.bot_equity with bot_type = 'beta1_v2'.

ISOLATION GUARANTEE:
  - Reads config from trading.beta1_v2 only (never trading.beta1)
  - Writes bot_type = BOT_TYPE_BETA1_V2 = "beta1_v2"
  - The (bot_id, bot_type, d) composite PK on trading.bot_equity ensures
    beta1 and beta1_v2 rows cannot overwrite each other

Status: backfill_enabled=False in algorithm_registry.py.
Do not run in production until composite scoring is validated.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beta1_v2_shared import (
    BOT_TYPE_BETA1_V2,
    Beta1V2Config,
    Beta1V2ConfigError,
    MAX_HOLDINGS,
    FALLBACK_TICKER,
    MIN_FINAL_SCORE,
    WEIGHT_ROUNDING,
    HISTORY_DAYS,
    MIN_DAILY_BARS,
    MOMENTUM_WEIGHT,
    CONFIRMATION_WEIGHT,
    RISK_WEIGHT,
    USE_MARKET_RISK_GATE,
    MARKET_GATE_OVERRIDE_SCORE,
    normalize_universe,
    safe_float,
    safe_int,
    clean_ticker,
    run_beta1_v2_reconstruction,
    Beta1V2ReconstructionResult,
)
from env_loader import load_env

load_env()

from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn, get_strategy_origin

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START_DATE = os.getenv("BETA1_V2_BACKFILL_START", "2023-01-01")
END_DATE   = date.today().isoformat()
UPSERT_BATCH_SIZE = 500

LOG_LEVEL = os.getenv("BETA1_V2_BACKFILL_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("beta1_v2_backfill")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@dataclass
class BackfillRow:
    bot_id:        str
    d:             date
    equity:        float
    ret:           float
    holdings_json: str
    bot_type:      str = BOT_TYPE_BETA1_V2   # NEVER "beta1"
    origin:        str | None = None
    source:        str = BOT_EQUITY_SOURCE_BACKFILL


def _chunked(values: list[Any], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def fetch_active_beta1_v2_bots() -> list[Beta1V2Config]:
    """Load all is_active=TRUE rows from trading.beta1_v2."""
    query = """
        SELECT
            bot_id,
            universe,
            COALESCE(max_holdings,               %s) AS max_holdings,
            COALESCE(fallback_ticker,             %s) AS fallback_ticker,
            COALESCE(min_final_score,             %s) AS min_final_score,
            COALESCE(weight_rounding,             %s) AS weight_rounding,
            COALESCE(history_days,                %s) AS history_days,
            COALESCE(min_daily_bars,              %s) AS min_daily_bars,
            COALESCE(momentum_weight,             %s) AS momentum_weight,
            COALESCE(confirmation_weight,         %s) AS confirmation_weight,
            COALESCE(risk_weight,                 %s) AS risk_weight,
            COALESCE(use_market_risk_gate,        %s) AS use_market_risk_gate,
            COALESCE(market_gate_override_score,  %s) AS market_gate_override_score
        FROM trading.beta1_v2
        WHERE is_active = TRUE
        ORDER BY bot_id
    """

    params = (
        MAX_HOLDINGS,
        FALLBACK_TICKER,
        MIN_FINAL_SCORE,
        WEIGHT_ROUNDING,
        HISTORY_DAYS,
        MIN_DAILY_BARS,
        MOMENTUM_WEIGHT,
        CONFIRMATION_WEIGHT,
        RISK_WEIGHT,
        USE_MARKET_RISK_GATE,
        MARKET_GATE_OVERRIDE_SCORE,
    )

    configs: list[Beta1V2Config] = []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    for row in rows:
        (
            bot_id, universe_raw, max_holdings, fallback_ticker,
            min_final_score, weight_rounding, history_days, min_daily_bars,
            momentum_weight, confirmation_weight, risk_weight,
            use_market_risk_gate, market_gate_override_score,
        ) = row

        configs.append(
            Beta1V2Config(
                bot_id=str(bot_id).strip().lower(),
                universe=normalize_universe(universe_raw),
                max_holdings=safe_int(max_holdings, MAX_HOLDINGS),
                fallback_ticker=clean_ticker(fallback_ticker, FALLBACK_TICKER),
                min_final_score=safe_float(min_final_score, MIN_FINAL_SCORE),
                weight_rounding=safe_float(weight_rounding, WEIGHT_ROUNDING),
                history_days=safe_int(history_days, HISTORY_DAYS),
                min_daily_bars=safe_int(min_daily_bars, MIN_DAILY_BARS),
                momentum_weight=safe_float(momentum_weight, MOMENTUM_WEIGHT),
                confirmation_weight=safe_float(confirmation_weight, CONFIRMATION_WEIGHT),
                risk_weight=safe_float(risk_weight, RISK_WEIGHT),
                use_market_risk_gate=bool(use_market_risk_gate),
                market_gate_override_score=safe_float(
                    market_gate_override_score, MARKET_GATE_OVERRIDE_SCORE
                ),
            )
        )

    return configs


def upsert_backfill_rows(rows: list[BackfillRow]) -> None:
    if not rows:
        return

    sql = """
        INSERT INTO trading.bot_equity (bot_id, bot_type, d, equity, ret, holdings, origin, source)
        VALUES %s
        ON CONFLICT (bot_id, bot_type, d)
        DO UPDATE SET
            equity   = EXCLUDED.equity,
            ret      = EXCLUDED.ret,
            holdings = EXCLUDED.holdings,
            origin   = COALESCE(EXCLUDED.origin, trading.bot_equity.origin),
            source   = EXCLUDED.source
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


# ---------------------------------------------------------------------------
# Single-bot backfill
# ---------------------------------------------------------------------------

def backfill_single_bot(config: Beta1V2Config) -> dict[str, Any]:
    logger.info(
        "Starting beta1_v2 backfill bot_id=%s start=%s end=%s universe=%s",
        config.bot_id, START_DATE, END_DATE, ",".join(config.universe),
    )

    if not config.universe:
        logger.warning("Skipping beta1_v2 backfill bot_id=%s — empty universe", config.bot_id)
        return {"bot_id": config.bot_id, "processed_days": 0, "errors": 0, "skipped": True}

    origin = get_strategy_origin(config.bot_id, BOT_TYPE_BETA1_V2)

    result: Beta1V2ReconstructionResult = run_beta1_v2_reconstruction(
        config, START_DATE, END_DATE
    )

    rows_to_upsert: list[BackfillRow] = []

    for ts, day_ret in result.daily_returns.items():
        d      = ts.date() if hasattr(ts, "date") else ts
        eq     = float(result.equity_curve.loc[ts])

        # Build holdings from weights_history row closest to this date
        weights_row = result.weights_history.loc[:ts].iloc[-1] if not result.weights_history.empty else pd.Series({})
        holdings: dict[str, float] = {
            str(ticker): float(w)
            for ticker, w in weights_row.items()
            if abs(float(w)) > 1e-12
        }

        payload = dict(holdings)
        payload["_meta"] = {
            "is_backtest":      True,
            "type":             "beta1_v2",
            "bot_type":         BOT_TYPE_BETA1_V2,
            "start_date":       START_DATE,
            "max_holdings":     config.max_holdings,
            "momentum_weight":  config.momentum_weight,
            "confirmation_weight": config.confirmation_weight,
            "risk_weight":      config.risk_weight,
            "asof":             str(ts),
        }

        rows_to_upsert.append(
            BackfillRow(
                bot_id=config.bot_id,
                d=d,
                equity=eq,
                ret=float(day_ret),
                holdings_json=json.dumps(payload),
                bot_type=BOT_TYPE_BETA1_V2,
                origin=origin,
                source=BOT_EQUITY_SOURCE_BACKFILL,
            )
        )

    upsert_backfill_rows(rows_to_upsert)

    final_equity     = float(result.equity_curve.iloc[-1])
    final_return_pct = (final_equity - 1.0) * 100.0

    logger.info(
        "Completed beta1_v2 bot=%s processed_days=%d final_equity=%.6f return_pct=%.2f",
        config.bot_id, len(result.daily_returns), final_equity, final_return_pct,
    )

    return {
        "bot_id":          config.bot_id,
        "processed_days":  len(result.daily_returns),
        "final_equity":    final_equity,
        "final_return_pct": final_return_pct,
        "errors":          0,
        "skipped":         False,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def backfill_beta1_v2() -> None:
    logger.info(
        "Loading active beta1_v2 bots from trading.beta1_v2 start=%s end=%s",
        START_DATE, END_DATE,
    )
    bots = fetch_active_beta1_v2_bots()

    if not bots:
        logger.warning("No active bots found in trading.beta1_v2")
        return

    logger.info("Found %d active bot(s). Starting backfill.", len(bots))

    results: list[dict[str, Any]] = []

    for bot in bots:
        try:
            results.append(backfill_single_bot(bot))
        except Exception as exc:
            logger.exception("Fatal error in beta1_v2 bot=%s: %s", bot.bot_id, exc)
            results.append(
                {"bot_id": bot.bot_id, "processed_days": 0, "errors": 1, "skipped": True}
            )

    logger.info("========== Beta1 v2 Backfill Summary ==========")
    for r in results:
        if r.get("skipped"):
            logger.info("bot=%s status=SKIPPED/FAILED errors=%s", r["bot_id"], r.get("errors"))
        else:
            logger.info(
                "bot=%s processed_days=%d final_equity=%.6f return_pct=%.2f errors=%d",
                r["bot_id"], r["processed_days"], r["final_equity"],
                r["final_return_pct"], r["errors"],
            )

    logger.info("Beta1 v2 backfill complete.")


if __name__ == "__main__":
    backfill_beta1_v2()
