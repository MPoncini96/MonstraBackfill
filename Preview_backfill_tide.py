#!/usr/bin/env python
"""Tide preview/backtest: reuses the exact walk-forward simulation from
backfill_tide.py (_simulate) so preview results and production backfills can
never drift apart - there is only one implementation of the day-by-day logic.

Tide ships hidden/Phase 1 (user_creatable=False, see algorithm_registry.py's
"tide" entry) so this script has no live caller yet - it exists for parity
with the other algorithm families' Preview_backfill_*.py scripts and so a
preview path is already in place if/when Tide is promoted.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from market_data_provider import get_daily_bars
from bots.tide import TideConfig
from backfill_tide import END_DATE, START_DATE, WARMUP_BUFFER_MIN_DAYS, WARMUP_BUFFER_MULTIPLIER, WARMUP_SAFETY_MARGIN_DAYS, _simulate


@dataclass
class PreviewConfig:
    bot_name: str
    cfg: TideConfig


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
    cash_equivalent = _clean_ticker(payload.get("cashEquivalent"), "VOO")
    universe = _normalize_universe(payload.get("tickers") or payload.get("universe"), exclude=cash_equivalent)
    defaults = TideConfig(universe=list(universe))

    benchmark_raw = payload.get("benchmark")
    benchmark = _clean_ticker(benchmark_raw, "") or None if benchmark_raw else None

    cfg = TideConfig(
        universe=universe or defaults.universe,
        cash_equivalent=cash_equivalent,
        top_n=_safe_int(payload.get("topN"), defaults.top_n),
        lookback_days=_safe_int(payload.get("lookbackDays"), defaults.lookback_days),
        incumbency_margin=float(payload.get("incumbencyMargin", defaults.incumbency_margin)),
        stop_loss_pct=_safe_fraction(payload.get("stopLossPct"), defaults.stop_loss_pct, 0.01, 0.95),
        circuit_breaker_pct=_safe_fraction(payload.get("circuitBreakerPct"), defaults.circuit_breaker_pct, 0.01, 0.95),
        circuit_breaker_cooldown_days=_safe_int(payload.get("circuitBreakerCooldownDays"), defaults.circuit_breaker_cooldown_days, minimum=0),
        benchmark=benchmark,
        enable_kill_switch=bool(payload.get("enableKillSwitch", defaults.enable_kill_switch)),
        enable_benchmark_filter=bool(payload.get("enableBenchmarkFilter", defaults.enable_benchmark_filter)),
        kill_switch_mode=str(payload.get("killSwitchMode", defaults.kill_switch_mode)),
        benchmark_mode=str(payload.get("benchmarkMode", defaults.benchmark_mode)),
        top_return_threshold=float(payload.get("topReturnThreshold", defaults.top_return_threshold)),
        benchmark_return_threshold=float(payload.get("benchmarkReturnThreshold", defaults.benchmark_return_threshold)),
        transaction_cost_bps=float(payload.get("transactionCostBps", defaults.transaction_cost_bps)),
        slippage_bps=float(payload.get("slippageBps", defaults.slippage_bps)),
    )
    return PreviewConfig(bot_name=str(payload.get("botName", "")).strip(), cfg=cfg)


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    cfg = preview.cfg
    if not cfg.universe:
        raise ValueError("At least one ticker is required.")

    all_tickers = list(dict.fromkeys(cfg.universe + ([cfg.cash_equivalent] if cfg.cash_equivalent else []) + ([cfg.benchmark] if cfg.benchmark else [])))
    warmup_days = max(cfg.lookback_days * WARMUP_BUFFER_MULTIPLIER, WARMUP_BUFFER_MIN_DAYS) + WARMUP_SAFETY_MARGIN_DAYS
    start_d = date.fromisoformat(START_DATE)
    dl_start = (start_d - pd.Timedelta(days=warmup_days)).isoformat()

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
    raw_returns = close.pct_change(fill_method=None)

    backfill_rows, sim_errors = _simulate(cfg, close, raw_returns, START_DATE)
    if not backfill_rows:
        raise ValueError("No trading dates were available in the preview window.")

    rows = [
        {
            "d": r.d.isoformat(),
            "equity": float(r.equity),
            "ret": float(r.ret),
            "holdings": {**json.loads(r.holdings_json), "_meta": {"is_backtest": True, "type": "tide"}},
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
