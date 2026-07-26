#!/usr/bin/env python
"""Draco preview/backtest: reuses the exact walk-forward simulation from
backfill_draco.py (_simulate) so preview results and production backfills can
never drift apart - there is only one implementation of the day-by-day logic.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from market_data_provider import get_daily_bars
from draco_config import DracoConfig, validate_draco_config
from backfill_draco import END_DATE, START_DATE, WARMUP_CALENDAR_DAYS, _simulate


@dataclass
class PreviewConfig:
    bot_name: str
    cfg: DracoConfig


def _clean_ticker(value: Any, fallback: str) -> str:
    text = str(value or "").strip().upper()
    return text or fallback


def _safe_int(value: Any, fallback: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, parsed)


def _safe_fraction(value: Any, fallback: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    if parsed > 1:
        parsed = parsed / 100.0
    return max(minimum, min(maximum, parsed))


def _normalize_universe(raw_tickers: Any, exclude: str | None = None) -> list[str]:
    if not isinstance(raw_tickers, list):
        return []
    exclude_norm = (exclude or "").strip().upper()
    seen: set[str] = set()
    out: list[str] = []
    for raw_ticker in raw_tickers:
        ticker = _clean_ticker(raw_ticker, "")
        if ticker and ticker != exclude_norm and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    fallback_ticker = _clean_ticker(payload.get("fallbackTicker"), "QQQ")
    universe = _normalize_universe(payload.get("tickers") or payload.get("universe"), exclude=fallback_ticker)
    defaults = DracoConfig(universe=list(universe))

    cfg = DracoConfig(
        universe=universe or defaults.universe,
        benchmark_ticker=_clean_ticker(payload.get("benchmarkTicker"), defaults.benchmark_ticker),
        market_regime_ticker=_clean_ticker(payload.get("marketRegimeTicker"), defaults.market_regime_ticker),
        fallback_ticker=fallback_ticker,
        max_positions=_safe_int(payload.get("maxPositions"), defaults.max_positions),
        max_new_positions_per_scan=_safe_int(payload.get("maxNewPositionsPerScan"), defaults.max_new_positions_per_scan),
        target_gross_exposure=_safe_fraction(payload.get("targetGrossExposure"), defaults.target_gross_exposure, 0.1, 1.0),
        minimum_entry_score=float(payload.get("minimumEntryScore", defaults.minimum_entry_score)),
        minimum_valid_timeframes=_safe_int(payload.get("minimumValidTimeframes"), defaults.minimum_valid_timeframes, minimum=0),
        minimum_below_regression_timeframes=_safe_int(payload.get("minimumBelowRegressionTimeframes"), defaults.minimum_below_regression_timeframes, minimum=0),
        minimum_positive_slope_timeframes=_safe_int(payload.get("minimumPositiveSlopeTimeframes"), defaults.minimum_positive_slope_timeframes, minimum=0),
        require_positive_long_term_slope=bool(payload.get("requirePositiveLongTermSlope", defaults.require_positive_long_term_slope)),
        minimum_target_r_squared=_safe_fraction(payload.get("minimumTargetRSquared"), defaults.minimum_target_r_squared, 0.0, 1.0),
        minimum_entry_target_upside=_safe_fraction(payload.get("minimumEntryTargetUpside"), defaults.minimum_entry_target_upside, 0.0, 5.0),
        maximum_entry_target_upside=_safe_fraction(payload.get("maximumEntryTargetUpside"), defaults.maximum_entry_target_upside, 0.0, 5.0),
        stop_loss_percent=_safe_fraction(payload.get("stopLossPercent"), defaults.stop_loss_percent, 0.01, 0.95),
        maximum_holding_trading_days=_safe_int(payload.get("maximumHoldingTradingDays"), defaults.maximum_holding_trading_days),
        symbol_reentry_cooldown_days=_safe_int(payload.get("symbolReentryCooldownDays"), defaults.symbol_reentry_cooldown_days, minimum=0),
        use_market_regime_filter=bool(payload.get("useMarketRegimeFilter", defaults.use_market_regime_filter)),
        market_sma_period=_safe_int(payload.get("marketSmaPeriod"), defaults.market_sma_period),
        regime_confirmation_days=_safe_int(payload.get("regimeConfirmationDays"), defaults.regime_confirmation_days),
        use_portfolio_circuit_breaker=bool(payload.get("usePortfolioCircuitBreaker", defaults.use_portfolio_circuit_breaker)),
        maximum_portfolio_drawdown=_safe_fraction(payload.get("maximumPortfolioDrawdown"), defaults.maximum_portfolio_drawdown, 0.01, 0.95),
        circuit_breaker_cooldown_trading_days=_safe_int(payload.get("circuitBreakerCooldownTradingDays"), defaults.circuit_breaker_cooldown_trading_days),
    )
    return PreviewConfig(bot_name=str(payload.get("botName", "")).strip(), cfg=cfg)


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    cfg = preview.cfg
    errors = validate_draco_config(cfg)
    if errors:
        raise ValueError("; ".join(errors))

    all_tickers = list(dict.fromkeys(cfg.universe + [cfg.market_regime_ticker, cfg.fallback_ticker]))
    start_d = date.fromisoformat(START_DATE)
    dl_start = (start_d - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)).isoformat()

    raw_bars = get_daily_bars(all_tickers, dl_start, END_DATE, adjusted=True)
    if raw_bars is None or raw_bars.empty:
        raise ValueError("Not enough market data was returned for this preview.")

    if isinstance(raw_bars.columns, pd.MultiIndex):
        if "Close" not in raw_bars.columns.get_level_values(0):
            raise ValueError("Not enough market data was returned for this preview.")
        close = raw_bars["Close"].copy()
    else:
        if "Close" not in raw_bars.columns:
            raise ValueError("Not enough market data was returned for this preview.")
        close = raw_bars[["Close"]].rename(columns={"Close": all_tickers[0]})

    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    close = close[~close.index.duplicated(keep="last")]
    close.columns = [str(c).upper() for c in close.columns]

    backfill_rows, sim_errors = _simulate(cfg, close, START_DATE)
    if not backfill_rows:
        raise ValueError("No trading dates were available in the preview window.")

    rows = [
        {
            "d": r.d.isoformat(),
            "equity": float(r.equity),
            "ret": float(r.ret),
            "holdings": {**json.loads(r.holdings_json), "_meta": {"is_backtest": True, "type": "draco"}},
        }
        for r in backfill_rows
    ]
    final_equity = rows[-1]["equity"]

    return {
        "success": True,
        "startDate": START_DATE,
        "endDate": END_DATE,
        "initialEquity": 1.0,
        "equity": rows,
        "summary": {
            "botName": preview.bot_name,
            "processedDays": len(rows),
            "simulationErrors": sim_errors,
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
    sys.exit(main())
