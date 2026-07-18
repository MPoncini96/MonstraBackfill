#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YFINANCE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "monstra_yfinance_cache")
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
os.environ.setdefault("YFINANCE_TZ_CACHE_LOCATION", YFINANCE_CACHE_DIR)

from env_loader import load_env

load_env()

import yfinance.cache as yf_cache

yf_cache.set_cache_location(YFINANCE_CACHE_DIR)
yf_cache.set_tz_cache_location(YFINANCE_CACHE_DIR)

from bots.gamma1 import (
    DEFAULT_BENCHMARK,
    DEFAULT_ENTRY_SCORE,
    DEFAULT_EXIT_SCORE,
    DEFAULT_MAX_POSITIONS,
    Gamma1Config,
    Gamma1FeatureFlags,
    _clean_ticker,
    _normalize_universe,
    _safe_bool,
    _safe_float,
    _safe_int,
    build_gamma1_score_table,
    download_gamma1_prices,
    gamma1_feature_flags_from_row,
    normalize_rebalance_rule,
    run_weighted_points_backtest,
)

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = date.today().isoformat()
WARMUP_CALENDAR_DAYS = 450

_FEATURE_FLAG_FIELDS = [
    "use_ret10_positive",
    "use_ret20_positive",
    "use_ret20_vs_benchmark",
    "use_price_above_ema20",
    "use_ema10_above_ema20",
    "use_ema20_above_ema50",
    "use_within_10pct_high",
    "use_rsi_55_70",
    "use_penalty_ret10_negative",
    "use_penalty_price_below_ema50",
    "use_penalty_rsi_above_80",
    "use_penalty_far_from_high",
    "use_penalty_volatility_spike",
    "use_penalty_fading_activity",
    "use_market_regime_penalty",
]
_LOCKED_FLAGS = {"use_market_regime_penalty"}
_KNOWN_FLAGS = set(_FEATURE_FLAG_FIELDS)


def _parse_vex_rule_config(raw: Any) -> dict[str, bool]:
    if raw is None:
        return {f: True for f in _FEATURE_FLAG_FIELDS}
    if not isinstance(raw, dict):
        raise ValueError("vexRuleConfig must be a JSON object.")
    for key in raw:
        if key not in _KNOWN_FLAGS:
            raise ValueError(f"Unknown Vex rule ID: '{key}'. Rejected to prevent misconfiguration.")

    result: dict[str, bool] = {}
    for flag in _FEATURE_FLAG_FIELDS:
        if flag in _LOCKED_FLAGS:
            result[flag] = True
        else:
            raw_val = raw.get(flag)
            result[flag] = _safe_bool(raw_val, True) if raw_val is not None else True
    return result


@dataclass
class PreviewConfig:
    bot_name: str
    universe: list[str]
    benchmark: str
    rebalance: str
    max_positions: int
    entry_score: float
    exit_score: float
    fallback_ticker: str | None
    start_date: str
    end_date: str


def parse_date_string(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return fallback


def price_download_start_for(start_date: str) -> str:
    start_d = date.fromisoformat(start_date)
    return (start_d - timedelta(days=WARMUP_CALENDAR_DAYS)).isoformat()


def build_preview_config(payload: dict[str, Any]) -> tuple[PreviewConfig, Gamma1FeatureFlags]:
    fallback_ticker = str(payload.get("safetyNetEquity") or "").strip().upper()
    custom_safety = str(payload.get("customSafetyNetEquity") or "").strip().upper()
    if fallback_ticker == "CUSTOM":
        fallback_ticker = custom_safety

    start_date = parse_date_string(payload.get("startDate"), DEFAULT_START_DATE)
    end_date = parse_date_string(payload.get("endDate"), DEFAULT_END_DATE)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    rule_config = _parse_vex_rule_config(payload.get("vexRuleConfig"))
    feature_flags = gamma1_feature_flags_from_row(rule_config)

    return PreviewConfig(
        bot_name=str(payload.get("botName", "")).strip(),
        universe=_normalize_universe(payload.get("tickers")),
        benchmark=_clean_ticker(payload.get("benchmark"), DEFAULT_BENCHMARK),
        rebalance=normalize_rebalance_rule(payload.get("rebalance_frequency") or payload.get("rebalance")),
        max_positions=_safe_int(payload.get("portfolioSize"), DEFAULT_MAX_POSITIONS),
        entry_score=_safe_float(payload.get("entry_score"), DEFAULT_ENTRY_SCORE),
        exit_score=_safe_float(payload.get("exit_score"), DEFAULT_EXIT_SCORE),
        fallback_ticker=fallback_ticker or None,
        start_date=start_date,
        end_date=end_date,
    ), feature_flags


def build_gamma1_config(preview: PreviewConfig, feature_flags: Gamma1FeatureFlags | None = None) -> Gamma1Config:
    return Gamma1Config(
        universe=preview.universe,
        benchmark=preview.benchmark,
        rebalance=preview.rebalance,
        max_positions=preview.max_positions,
        entry_score=preview.entry_score,
        exit_score=preview.exit_score,
        fallback_ticker=preview.fallback_ticker,
        weighting_style="confidence_shield",
        feature_flags=feature_flags if feature_flags is not None else Gamma1FeatureFlags(),
        use_sector_cap=False,
        use_volatility_filter=False,
    )


def run_preview(preview: PreviewConfig, feature_flags: Gamma1FeatureFlags | None = None) -> dict[str, Any]:
    if not preview.universe:
        raise ValueError("At least one universe ticker is required.")

    config = build_gamma1_config(preview, feature_flags)
    tickers = list(dict.fromkeys(list(config.universe) + [config.benchmark] + ([config.fallback_ticker] if config.fallback_ticker else [])))
    data = download_gamma1_prices(tickers, price_download_start_for(preview.start_date), preview.end_date)

    if data.empty or config.benchmark not in data.columns:
        raise ValueError("Not enough market data was returned for this preview.")

    universe = [ticker for ticker in config.universe if ticker in data.columns]
    if not universe:
        raise ValueError("No preview universe symbols were found in the downloaded market data.")

    prices = data[universe].copy().dropna(how="all").sort_index()
    benchmark_prices = data[config.benchmark].reindex(prices.index)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score_table = build_gamma1_score_table(prices, benchmark_prices, config.feature_flags)
    raw_scores = score_table["total"].copy()
    raw_scores["_benchmark"] = benchmark_prices

    result = run_weighted_points_backtest(
        prices,
        returns,
        raw_scores,
        config,
        record_weights=True,
    )

    weights = result["weights"]
    net_returns = result["net_returns"]
    equity_curve = result["equity_curve"]

    if weights is None or net_returns.empty or equity_curve.empty:
        raise ValueError("Preview backtest did not return any equity data.")

    rows: list[dict[str, Any]] = []
    for ts in net_returns.index:
        iso_date = ts.date().isoformat()
        if iso_date < preview.start_date or iso_date > preview.end_date:
            continue

        weight_row = weights.loc[ts]
        holdings = {str(k): float(v) for k, v in weight_row.items() if float(v) > 1e-12}
        rows.append(
            {
                "d": iso_date,
                "equity": float(equity_curve.loc[ts, "equity"]),
                "ret": float(net_returns.loc[ts]),
                "holdings": holdings,
                "meta": {
                    "benchmark": config.benchmark,
                    "rebalance": config.rebalance,
                    "max_positions": int(config.max_positions),
                    "entry_score": float(config.entry_score),
                    "exit_score": float(config.exit_score),
                },
            }
        )

    final_equity = rows[-1]["equity"] if rows else 1.0
    final_return_pct = (final_equity - 1.0) * 100.0
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
            "finalEquity": final_equity,
            "finalReturnPct": final_return_pct,
        },
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("Preview payload must be a JSON object.")
        preview, feature_flags = build_preview_config(payload)
        result = run_preview(preview, feature_flags)
        json.dump(result, sys.stdout)
        return 0
    except Exception as exc:
        json.dump({"success": False, "error": str(exc)}, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
