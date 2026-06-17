#!/usr/bin/env python
"""
Preview_backfill_beta1_v2.py
============================
Preview runner for Phantom v2 (composite scoring).

Reads bot config from JSON on stdin, runs the composite scoring reconstruction,
and writes the equity curve to stdout as JSON.  No database writes.

Uses the same run_beta1_v2_reconstruction() as backfill_beta1_v2.py to
guarantee that preview and backfill weights are identical for the same inputs.

Input (stdin): JSON object with keys matching normalizeBeta1V2Config output:
  botName, tickers, safetyNetEquity, portfolioSize,
  momentumWeight, confirmationWeight, riskWeight,
  minFinalScore, weightRounding, historyDays, minDailyBars,
  useMarketRiskGate, marketGateOverrideScore

Output (stdout): JSON object:
  { success, startDate, endDate, initialEquity, equity: [...], summary: {...} }
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
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

from beta1_v2_shared import (
    Beta1V2Config,
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
)

from datetime import date

START_DATE = os.getenv("BETA1_V2_PREVIEW_START", "2023-01-01")
END_DATE   = date.today().isoformat()


# ---------------------------------------------------------------------------
# Config builder from preview payload
# ---------------------------------------------------------------------------

@dataclass
class PreviewConfig:
    bot_name:               str
    universe:               list[str]
    max_holdings:           int
    fallback_ticker:        str
    min_final_score:        float
    weight_rounding:        float
    history_days:           int
    min_daily_bars:         int
    momentum_weight:        float
    confirmation_weight:    float
    risk_weight:            float
    use_market_risk_gate:   bool
    market_gate_override_score: float


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    return PreviewConfig(
        bot_name=str(payload.get("botName", "")).strip(),
        universe=normalize_universe(payload.get("tickers")),
        max_holdings=safe_int(payload.get("portfolioSize"), MAX_HOLDINGS),
        fallback_ticker=clean_ticker(payload.get("safetyNetEquity"), FALLBACK_TICKER),
        min_final_score=safe_float(payload.get("minFinalScore"), MIN_FINAL_SCORE),
        weight_rounding=safe_float(payload.get("weightRounding"), WEIGHT_ROUNDING),
        history_days=safe_int(payload.get("historyDays"), HISTORY_DAYS),
        min_daily_bars=safe_int(payload.get("minDailyBars"), MIN_DAILY_BARS),
        momentum_weight=safe_float(payload.get("momentumWeight"), MOMENTUM_WEIGHT),
        confirmation_weight=safe_float(payload.get("confirmationWeight"), CONFIRMATION_WEIGHT),
        risk_weight=safe_float(payload.get("riskWeight"), RISK_WEIGHT),
        use_market_risk_gate=bool(payload.get("useMarketRiskGate", USE_MARKET_RISK_GATE)),
        market_gate_override_score=safe_float(
            payload.get("marketGateOverrideScore"), MARKET_GATE_OVERRIDE_SCORE
        ),
    )


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.universe:
        raise ValueError("At least one universe ticker is required.")

    config = Beta1V2Config(
        bot_id="preview",
        universe=preview.universe,
        max_holdings=preview.max_holdings,
        fallback_ticker=preview.fallback_ticker,
        min_final_score=preview.min_final_score,
        weight_rounding=preview.weight_rounding,
        history_days=preview.history_days,
        min_daily_bars=preview.min_daily_bars,
        momentum_weight=preview.momentum_weight,
        confirmation_weight=preview.confirmation_weight,
        risk_weight=preview.risk_weight,
        use_market_risk_gate=preview.use_market_risk_gate,
        market_gate_override_score=preview.market_gate_override_score,
    )

    # run_beta1_v2_reconstruction is the same function used by backfill,
    # guaranteeing preview and backfill weights are identical for identical inputs.
    result = run_beta1_v2_reconstruction(config, START_DATE, END_DATE)

    rows: list[dict[str, Any]] = []
    for ts, day_ret in result.daily_returns.items():
        weights_row = result.weights_history.loc[:ts].iloc[-1] if not result.weights_history.empty else {}
        holdings = {
            str(ticker): float(w)
            for ticker, w in (weights_row.items() if hasattr(weights_row, "items") else {}.items())
            if abs(float(w)) > 1e-12
        }
        rows.append(
            {
                "d":       (ts.date() if hasattr(ts, "date") else ts).isoformat(),
                "equity":  float(result.equity_curve.loc[ts]),
                "ret":     float(day_ret),
                "holdings": holdings,
                "meta": {
                    "max_holdings":    config.max_holdings,
                    "fallback_ticker": config.fallback_ticker,
                    "momentum_weight": config.momentum_weight,
                    "confirmation_weight": config.confirmation_weight,
                    "risk_weight":     config.risk_weight,
                },
            }
        )

    final_equity     = rows[-1]["equity"] if rows else 1.0
    final_return_pct = (final_equity - 1.0) * 100.0

    return {
        "success":        True,
        "startDate":      START_DATE,
        "endDate":        END_DATE,
        "initialEquity":  1.0,
        "equity":         rows,
        "summary": {
            "botName":        preview.bot_name,
            "processedDays":  len(rows),
            "finalEquity":    final_equity,
            "finalReturnPct": final_return_pct,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("Preview payload must be a JSON object.")
        preview = build_preview_config(payload)
        result  = run_preview(preview)
        json.dump(result, sys.stdout)
        return 0
    except Exception as exc:
        json.dump({"success": False, "error": str(exc)}, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
