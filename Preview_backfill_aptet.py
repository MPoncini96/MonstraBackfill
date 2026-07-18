from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YFINANCE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "monstra_yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
os.environ.setdefault("YFINANCE_TZ_CACHE_LOCATION", YFINANCE_CACHE_DIR)

from env_loader import load_env

load_env()

import yfinance.cache as yf_cache

yf_cache.set_cache_location(YFINANCE_CACHE_DIR)
yf_cache.set_tz_cache_location(YFINANCE_CACHE_DIR)

from backfill_aptet import compute_cost_drag, compute_day_return_and_holdings, compute_turnover
from bots.aptet import AptetConfig, download_aptet_prices, resolve_aptet_decision

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = date.today().isoformat()


@dataclass
class PreviewConfig:
    bot_name: str
    universe: list[str]
    fallback_ticker: str
    min_holdings: int
    max_holdings: int
    adaptation_speed: str
    risk_off_enabled: bool
    start_date: str
    end_date: str


def _clean_ticker(value: Any, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    ticker = str(value).strip().upper()
    return ticker or fallback


def _parse_int(value: Any, fallback: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    parsed = max(minimum, parsed)
    return min(parsed, maximum) if maximum is not None else parsed


def _normalize_universe(raw_universe: Any, fallback_ticker: str) -> list[str]:
    if not isinstance(raw_universe, list):
        return []
    seen: set[str] = set()
    universe: list[str] = []
    for item in raw_universe:
        ticker = _clean_ticker(item)
        if not ticker or ticker == fallback_ticker or ticker in seen:
            continue
        seen.add(ticker)
        universe.append(ticker)
    return universe


def _parse_date_string(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return fallback


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    fallback_ticker = _clean_ticker(payload.get("fallbackTicker"), "VOO") or "VOO"
    universe = _normalize_universe(payload.get("universe"), fallback_ticker)
    max_holdings = _parse_int(payload.get("maxHoldings"), 8, minimum=1, maximum=max(1, len(universe) or 1))
    min_holdings = _parse_int(payload.get("minHoldings"), 2, minimum=1, maximum=max_holdings)
    adaptation_speed = str(payload.get("adaptationSpeed") or "balanced").strip().lower() or "balanced"
    if adaptation_speed not in {"conservative", "balanced", "aggressive"}:
        adaptation_speed = "balanced"
    start_date = _parse_date_string(payload.get("startDate"), DEFAULT_START_DATE)
    end_date = _parse_date_string(payload.get("endDate"), DEFAULT_END_DATE)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return PreviewConfig(
        bot_name=str(payload.get("botName") or "").strip(),
        universe=universe,
        fallback_ticker=fallback_ticker,
        min_holdings=min_holdings,
        max_holdings=max_holdings,
        adaptation_speed=adaptation_speed,
        risk_off_enabled=bool(True if payload.get("riskOffEnabled") is None else payload.get("riskOffEnabled")),
        start_date=start_date,
        end_date=end_date,
    )


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.universe:
        raise ValueError("At least one universe ticker is required.")
    config = AptetConfig(
        universe=preview.universe,
        fallback_ticker=preview.fallback_ticker,
        benchmark_ticker=preview.fallback_ticker,
        min_holdings=preview.min_holdings,
        max_holdings=preview.max_holdings,
        adaptation_speed=preview.adaptation_speed,
        risk_off_enabled=preview.risk_off_enabled,
    )
    prices = download_aptet_prices(config, preview.start_date, preview.end_date)
    if prices.empty or len(prices.index) < 2:
        raise ValueError("Not enough market data was returned for this preview.")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = 1.0
    prev_holdings: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    risk_off_days = 0
    for index, trading_ts in enumerate(prices.index):
        if index == 0:
            continue
        selected_symbols, weights, risk_off, risk_reason, metadata = resolve_aptet_decision(prices, config, end_idx_exclusive=index)
        if risk_off:
            risk_off_days += 1
        gross_ret, current_holdings = compute_day_return_and_holdings(returns.loc[trading_ts], selected_symbols, weights)
        turnover = compute_turnover(prev_holdings, current_holdings)
        cost_drag = compute_cost_drag(turnover)
        net_ret = gross_ret - cost_drag
        equity *= 1.0 + net_ret
        rows.append({
            "d": trading_ts.date().isoformat(),
            "equity": float(equity),
            "ret": float(net_ret),
            "holdings": current_holdings,
            "meta": {"risk_off": bool(risk_off), "risk_reason": risk_reason, "turnover": float(turnover), "cost_drag": float(cost_drag), **metadata},
        })
        prev_holdings = current_holdings.copy()
    final_equity = rows[-1]["equity"] if rows else 1.0
    actual_start_date = rows[0]["d"] if rows else preview.start_date
    actual_end_date = rows[-1]["d"] if rows else preview.end_date
    return {
        "success": True,
        "startDate": actual_start_date,
        "endDate": actual_end_date,
        "initialEquity": 1.0,
        "equity": rows,
        "summary": {
            "botName": preview.bot_name,
            "processedDays": len(rows),
            "riskOffDays": risk_off_days,
            "finalEquity": final_equity,
            "finalReturnPct": (final_equity - 1.0) * 100.0,
        },
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("Preview payload must be a JSON object.")
        preview = build_preview_config(payload)
        result = run_preview(preview)
        json.dump(result, sys.stdout)
        return 0
    except Exception as exc:
        json.dump({"success": False, "error": str(exc)}, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
