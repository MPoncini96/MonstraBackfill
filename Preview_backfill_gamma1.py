#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
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

from backfill_gamma1 import END_DATE, START_DATE, price_download_start
from bots.gamma1 import (
    DEFAULT_BENCHMARK,
    DEFAULT_ENTRY_SCORE,
    DEFAULT_EXIT_SCORE,
    DEFAULT_MAX_POSITIONS,
    Gamma1Config,
    Gamma1FeatureFlags,
    _clean_ticker,
    _normalize_universe,
    _safe_float,
    _safe_int,
    build_gamma1_score_table,
    download_gamma1_prices,
    normalize_rebalance_rule,
    run_weighted_points_backtest,
)


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


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    fallback_ticker = str(payload.get("safetyNetEquity") or "").strip().upper()
    custom_safety = str(payload.get("customSafetyNetEquity") or "").strip().upper()
    if fallback_ticker == "CUSTOM":
        fallback_ticker = custom_safety

    return PreviewConfig(
        bot_name=str(payload.get("botName", "")).strip(),
        universe=_normalize_universe(payload.get("tickers")),
        benchmark=_clean_ticker(payload.get("benchmark"), DEFAULT_BENCHMARK),
        rebalance=normalize_rebalance_rule(payload.get("rebalance_frequency") or payload.get("rebalance")),
        max_positions=_safe_int(payload.get("portfolioSize"), DEFAULT_MAX_POSITIONS),
        entry_score=_safe_float(payload.get("entry_score"), DEFAULT_ENTRY_SCORE),
        exit_score=_safe_float(payload.get("exit_score"), DEFAULT_EXIT_SCORE),
        fallback_ticker=fallback_ticker or None,
    )


def build_gamma1_config(preview: PreviewConfig) -> Gamma1Config:
    return Gamma1Config(
        universe=preview.universe,
        benchmark=preview.benchmark,
        rebalance=preview.rebalance,
        max_positions=preview.max_positions,
        entry_score=preview.entry_score,
        exit_score=preview.exit_score,
        fallback_ticker=preview.fallback_ticker,
        weighting_style="equal",
        feature_flags=Gamma1FeatureFlags(),
        use_sector_cap=False,
        use_volatility_filter=False,
    )


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.universe:
        raise ValueError("At least one universe ticker is required.")

    config = build_gamma1_config(preview)
    tickers = list(dict.fromkeys(list(config.universe) + [config.benchmark]))
    data = download_gamma1_prices(tickers, price_download_start(), END_DATE)

    if data.empty or config.benchmark not in data.columns:
        raise ValueError("Not enough market data was returned for this preview.")

    universe = [ticker for ticker in config.universe if ticker in data.columns]
    if not universe:
        raise ValueError("No preview universe symbols were found in the downloaded market data.")

    prices = data[universe].copy().dropna(how="all").sort_index()
    benchmark_prices = data[config.benchmark].reindex(prices.index)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score_table = build_gamma1_score_table(prices, benchmark_prices, config.feature_flags)
    raw_scores = score_table["total"]

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
        if ts.date().isoformat() < START_DATE:
            continue

        weight_row = weights.loc[ts]
        holdings = {str(k): float(v) for k, v in weight_row.items() if float(v) > 1e-12}
        rows.append(
            {
                "d": ts.date().isoformat(),
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
