#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YFINANCE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "monstra_yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
os.environ.setdefault("YFINANCE_TZ_CACHE_LOCATION", YFINANCE_CACHE_DIR)

from env_loader import load_env

load_env()

import yfinance.cache as yf_cache

yf_cache.set_cache_location(YFINANCE_CACHE_DIR)
yf_cache.set_tz_cache_location(YFINANCE_CACHE_DIR)

from backfill_beta1 import (
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    FALLBACK_TICKER,
    MAX_HOLDINGS,
    START_DATE,
    END_DATE,
    _clean_ticker,
    _normalize_universe,
    _safe_float,
    _safe_int,
    _weights_for_date,
    run_backtest,
)


@dataclass
class PreviewConfig:
    bot_name: str
    universe: list[str]
    entry_threshold: float
    exit_threshold: float
    max_holdings: int
    fallback_ticker: str


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    return PreviewConfig(
        bot_name=str(payload.get("botName", "")).strip(),
        universe=_normalize_universe(payload.get("tickers")),
        entry_threshold=_safe_float(payload.get("entry_threshold"), ENTRY_THRESHOLD),
        exit_threshold=_safe_float(payload.get("exit_threshold"), EXIT_THRESHOLD),
        max_holdings=_safe_int(payload.get("portfolioSize"), MAX_HOLDINGS),
        fallback_ticker=_clean_ticker(payload.get("safetyNetEquity"), FALLBACK_TICKER),
    )


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.universe:
        raise ValueError("At least one universe ticker is required.")

    result = run_backtest(
        universe=preview.universe,
        start=START_DATE,
        end=END_DATE,
        entry_threshold=preview.entry_threshold,
        exit_threshold=preview.exit_threshold,
        max_holdings=preview.max_holdings,
        fallback_ticker=preview.fallback_ticker,
    )

    rows: list[dict[str, Any]] = []
    for ts, week_ret in result.weekly_returns.items():
        holdings = _weights_for_date(result.weights_history, ts)
        rows.append(
            {
                "d": ts.date().isoformat(),
                "equity": float(result.equity_curve.loc[ts]),
                "ret": float(week_ret),
                "holdings": holdings,
                "meta": {
                    "entry_threshold": float(preview.entry_threshold),
                    "exit_threshold": float(preview.exit_threshold),
                    "max_holdings": int(preview.max_holdings),
                    "fallback_ticker": preview.fallback_ticker,
                },
            }
        )

    final_equity = rows[-1]["equity"] if rows else 1.0
    final_return_pct = (final_equity - 1.0) * 100.0

    return {
        "success": True,
        "startDate": START_DATE,
        "endDate": END_DATE,
        "initialEquity": 1.0,
        "equity": rows,
        "summary": {
            "botName": preview.bot_name,
            "processedDays": len(rows),
            "finalEquity": final_equity,
            "finalReturnPct": final_return_pct,
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
