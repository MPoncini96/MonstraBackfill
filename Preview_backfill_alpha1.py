#!/usr/bin/env python
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

from bots.alpha1 import (
    Alpha1Config,
    _build_price_frame,
    _choose_holdings_for_day,
    _clamp_top_n,
    _get_trailing_returns,
    _normalize_rank_weights_array,
)
from backfill_alpha1 import compute_cost_drag, compute_day_return_and_holdings, compute_turnover

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass
class PreviewConfig:
    bot_name: str
    tickers: list[str]
    cash_equivalent: str
    lookback: int
    portfolio_size: int
    weights: list[float]


def clean_ticker(value: Any, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    text = str(value).strip().upper()
    return text or fallback


def parse_int(value: Any, fallback: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def normalize_tickers(raw_tickers: Any) -> list[str]:
    if not isinstance(raw_tickers, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_ticker in raw_tickers:
        ticker = clean_ticker(raw_ticker)
        if ticker and ticker not in seen:
            seen.add(ticker)
            normalized.append(ticker)
    return normalized


def normalize_weights(raw_weights: Any, portfolio_size: int) -> list[float]:
    if not isinstance(raw_weights, list):
        raw_weights = []

    numeric: list[float] = []
    for raw_weight in raw_weights:
        try:
            numeric.append(float(raw_weight))
        except (TypeError, ValueError):
            continue

    if not numeric:
        fallback = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
        return fallback[:portfolio_size].tolist()

    normalized = _normalize_rank_weights_array(numeric)
    return normalized[:portfolio_size].tolist()


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    portfolio_size = parse_int(payload.get("portfolioSize"), 4, minimum=1, maximum=20)
    return PreviewConfig(
        bot_name=str(payload.get("botName", "")).strip(),
        tickers=normalize_tickers(payload.get("tickers")),
        cash_equivalent=clean_ticker(payload.get("safetyNetEquity"), "VOO") or "VOO",
        lookback=parse_int(payload.get("lookback"), 60, minimum=1, maximum=365),
        portfolio_size=portfolio_size,
        weights=normalize_weights(payload.get("weights"), portfolio_size),
    )


def build_alpha1_config(preview: PreviewConfig) -> Alpha1Config:
    rank_weights = _normalize_rank_weights_array(preview.weights)
    top_n = min(preview.portfolio_size, len(rank_weights)) if len(rank_weights) else preview.portfolio_size
    top_n = max(1, top_n)

    return Alpha1Config(
        universe=preview.tickers,
        cash_equivalent=preview.cash_equivalent,
        top_n=top_n,
        lookback_days=preview.lookback,
        benchmark=None,
        enable_kill_switch=True,
        enable_benchmark_filter=False,
        kill_switch_mode="top_negative",
        benchmark_mode="benchmark_positive",
        top_return_threshold=0.0,
        benchmark_return_threshold=0.0,
        transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        rank_weights=rank_weights,
    )


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.tickers:
        raise ValueError("At least one universe ticker is required.")

    config = build_alpha1_config(preview)
    effective_top_n = _clamp_top_n(config.top_n, config.rank_weights)
    prices = _build_price_frame(config, START_DATE, END_DATE)
    if prices.empty or len(prices.index) < 2:
        raise ValueError("Not enough market data was returned for this preview.")

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    equity = 1.0
    prev_holdings: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    risk_off_days = 0

    ranking_universe = set(config.universe)
    benchmark_symbols = {config.benchmark} if config.benchmark else set()
    cash_symbols = {config.cash_equivalent} if config.cash_equivalent else set()

    for index, trading_ts in enumerate(prices.index):
        if index == 0:
            continue

        ranked_trailing = _get_trailing_returns(
            prices=prices,
            end_idx_exclusive=index,
            lookback_days=config.lookback_days,
            symbols=ranking_universe,
        )
        benchmark_trailing = _get_trailing_returns(
            prices=prices,
            end_idx_exclusive=index,
            lookback_days=config.lookback_days,
            symbols=benchmark_symbols,
        )
        cash_trailing = _get_trailing_returns(
            prices=prices,
            end_idx_exclusive=index,
            lookback_days=config.lookback_days,
            symbols=cash_symbols,
        )

        selected_symbols, weights, risk_off, risk_reason = _choose_holdings_for_day(
            config=config,
            ranked_trailing=ranked_trailing,
            benchmark_trailing=benchmark_trailing,
            cash_trailing=cash_trailing,
        )

        if risk_off:
            risk_off_days += 1

        gross_ret, current_holdings = compute_day_return_and_holdings(
            day_returns=returns.loc[trading_ts],
            selected_symbols=selected_symbols,
            weights=weights,
        )

        turnover = compute_turnover(prev_holdings, current_holdings)
        cost_drag = compute_cost_drag(
            turnover=turnover,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
        )
        net_ret = gross_ret - cost_drag
        equity *= 1.0 + net_ret

        rows.append(
            {
                "d": trading_ts.date().isoformat(),
                "equity": float(equity),
                "ret": float(net_ret),
                "holdings": current_holdings,
                "meta": {
                    "risk_off": bool(risk_off),
                    "risk_reason": risk_reason,
                    "selected_stocks": selected_symbols,
                    "turnover": float(turnover),
                    "cost_drag": float(cost_drag),
                    "top_n": effective_top_n,
                },
            }
        )
        prev_holdings = current_holdings.copy()

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
            "riskOffDays": risk_off_days,
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
