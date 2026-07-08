from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env

load_env()

from bot_identity import BOT_TYPE_APTET
from bots.aptet import AptetConfig, download_aptet_prices, resolve_aptet_decision
from db import BOT_EQUITY_SOURCE_BACKFILL, get_conn, get_strategy_origin

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()
UPSERT_BATCH_SIZE = 500
LOG_LEVEL = os.getenv("APTET_BACKFILL_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("aptet_backfill")


@dataclass
class BackfillRow:
    bot_id: str
    d: date
    equity: float
    ret: float
    holdings_json: str
    bot_type: str = BOT_TYPE_APTET
    origin: str | None = None
    source: str = BOT_EQUITY_SOURCE_BACKFILL


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for index in range(0, len(seq), size):
        yield seq[index:index + size]


def fetch_active_aptet_bots() -> list[AptetConfig]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM trading.aptet WHERE is_active = TRUE ORDER BY bot_id")
            rows = cur.fetchall()
    return [
        AptetConfig(
            universe=list(row.get("universe") or []),
            fallback_ticker=str(row.get("fallback_ticker") or "VOO").strip().upper(),
            benchmark_ticker=str((row.get("benchmark_ticker") or row.get("fallback_ticker") or "VOO")).strip().upper(),
            min_holdings=int(row.get("min_holdings") or 2),
            max_holdings=int(row.get("max_holdings") or 8),
            adaptation_speed=str(row.get("adaptation_speed") or "balanced").strip().lower(),
            risk_off_enabled=bool(True if row.get("risk_off_enabled") is None else row.get("risk_off_enabled")),
            history_period=str(row.get("history_period")).strip() if row.get("history_period") else None,
            interval=str(row.get("interval") or "1d").strip() or "1d",
            bot_id=str(row.get("bot_id") or "").strip(),
        )
        for row in rows
    ]


def fetch_previous_equity(bot_id: str, start_date: str, bot_type: str = BOT_TYPE_APTET) -> float:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT equity FROM trading.bot_equity WHERE bot_id = %s AND bot_type = %s AND d < %s ORDER BY d DESC LIMIT 1", (bot_id, bot_type, start_date))
            row = cur.fetchone()
    return float(row[0]) if row and is_finite_number(row[0]) else 1.0


def fetch_previous_holdings(bot_id: str, start_date: str, bot_type: str = BOT_TYPE_APTET) -> dict[str, float]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT holdings FROM trading.bot_equity WHERE bot_id = %s AND bot_type = %s AND d < %s ORDER BY d DESC LIMIT 1", (bot_id, bot_type, start_date))
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
    return {str(key).upper(): float(value) for key, value in raw.items() if not str(key).startswith("_") and is_finite_number(value)}


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
            source = CASE WHEN LOWER(TRIM(COALESCE(trading.bot_equity.source, ''))) = 'live trading' THEN trading.bot_equity.source ELSE EXCLUDED.source END
    """
    values = [(row.bot_id, row.bot_type, row.d, row.equity, row.ret, row.holdings_json, row.origin, row.source) for row in rows]
    with get_conn() as conn:
        with conn.cursor() as cur:
            for batch in chunked(values, UPSERT_BATCH_SIZE):
                execute_values(cur, sql, batch)
        conn.commit()

def compute_turnover(prev_holdings: dict[str, float], new_holdings: dict[str, float]) -> float:
    all_symbols = set(prev_holdings) | set(new_holdings)
    return sum(abs(new_holdings.get(sym, 0.0) - prev_holdings.get(sym, 0.0)) for sym in all_symbols) / 2.0


def compute_cost_drag(turnover: float, transaction_cost_bps: float = 5.0, slippage_bps: float = 5.0) -> float:
    return turnover * ((float(transaction_cost_bps) + float(slippage_bps)) / 10_000.0)


def compute_day_return_and_holdings(day_returns: pd.Series, selected_symbols: list[str], weights: np.ndarray) -> tuple[float, dict[str, float]]:
    if not selected_symbols or len(weights) == 0:
        return 0.0, {}
    valid_pairs = [(symbol, float(weight)) for symbol, weight in zip(selected_symbols, weights) if symbol in day_returns.index and pd.notna(day_returns[symbol])]
    if not valid_pairs:
        return 0.0, {}
    total_weight = sum(weight for _, weight in valid_pairs)
    if total_weight <= 0:
        return 0.0, {}
    normalized_pairs = [(symbol, weight / total_weight) for symbol, weight in valid_pairs]
    gross_ret = sum(float(day_returns[symbol]) * weight for symbol, weight in normalized_pairs)
    return float(gross_ret), {symbol: float(weight) for symbol, weight in normalized_pairs}


def build_holdings_payload(holdings: dict[str, float], bot: AptetConfig, metadata: dict[str, Any], risk_off: bool, risk_reason: str, gross_ret: float, net_ret: float, turnover: float, cost_drag: float) -> str:
    payload = dict(holdings)
    payload["_meta"] = {
        "is_backtest": True,
        "type": BOT_TYPE_APTET,
        "timing": "rank_on_t_minus_1_apply_return_on_t",
        "risk_off": bool(risk_off),
        "risk_reason": str(risk_reason),
        "fallback_ticker": bot.fallback_ticker,
        "gross_ret": float(gross_ret),
        "net_ret": float(net_ret),
        "turnover": float(turnover),
        "cost_drag": float(cost_drag),
        **metadata,
    }
    return json.dumps(payload)


def backfill_single_bot(bot: AptetConfig) -> dict[str, Any]:
    logger.info("Starting aptet bot_id=%s universe=%d min_holdings=%d max_holdings=%d speed=%s", bot.bot_id, len(bot.universe), bot.min_holdings, bot.max_holdings, bot.adaptation_speed)
    if not bot.universe:
        return {"bot_id": bot.bot_id, "processed_days": 0, "final_equity": None, "final_return_pct": None, "errors": 0, "skipped": True}
    prices = download_aptet_prices(bot, START_DATE, END_DATE)
    if prices.empty:
        return {"bot_id": bot.bot_id, "processed_days": 0, "final_equity": None, "final_return_pct": None, "errors": 0, "skipped": True}
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = fetch_previous_equity(bot.bot_id or "", START_DATE)
    prev_holdings = fetch_previous_holdings(bot.bot_id or "", START_DATE)
    origin = get_strategy_origin(bot.bot_id or "", BOT_TYPE_APTET)
    rows_to_upsert: list[BackfillRow] = []
    errors = 0
    processed_days = 0
    risk_off_days = 0
    for index, trading_ts in enumerate(prices.index):
        if index == 0:
            continue
        trading_day = trading_ts.date()
        try:
            selected_symbols, weights, risk_off, risk_reason, metadata = resolve_aptet_decision(prices, bot, end_idx_exclusive=index)
            if risk_off:
                risk_off_days += 1
            gross_ret, new_holdings = compute_day_return_and_holdings(returns.loc[trading_ts], selected_symbols, weights)
            turnover = compute_turnover(prev_holdings, new_holdings)
            cost_drag = compute_cost_drag(turnover)
            net_ret = gross_ret - cost_drag
            equity *= (1.0 + net_ret)
            rows_to_upsert.append(BackfillRow(bot_id=bot.bot_id or "", d=trading_day, equity=float(equity), ret=float(net_ret), holdings_json=build_holdings_payload(new_holdings, bot, metadata, risk_off, risk_reason, gross_ret, net_ret, turnover, cost_drag), origin=origin))
            prev_holdings = new_holdings
            processed_days += 1
        except Exception as exc:
            errors += 1
            logger.exception("Error processing bot=%s date=%s: %s", bot.bot_id, trading_day, exc)
    upsert_backfill_rows(rows_to_upsert)
    return {
        "bot_id": bot.bot_id,
        "processed_days": processed_days,
        "risk_off_days": risk_off_days,
        "final_equity": equity,
        "final_return_pct": (equity - 1.0) * 100.0,
        "errors": errors,
        "skipped": False,
    }


def backfill_aptet() -> None:
    bots = fetch_active_aptet_bots()
    if not bots:
        logger.warning("No active bots found in trading.aptet")
        return
    for bot in bots:
        backfill_single_bot(bot)


if __name__ == "__main__":
    backfill_aptet()
