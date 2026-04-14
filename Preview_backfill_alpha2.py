#!/usr/bin/env python
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env

load_env()

from bots.alpha2 import (
    Alpha2Config,
    build_universe_price_frame,
    choose_universe_triggered_holdings,
    compute_day_return_and_holdings,
    is_rebalance_day,
    parse_rebalance_freq,
)

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass
class PreviewConfig:
    bot_name: str
    groups: list[dict[str, Any]]
    benchmark: str
    cash_equivalent: str
    lookback: int
    rebalance_freq: int
    slot_count: int
    weights: list[float]
    enable_kill_switch: bool
    enable_benchmark_filter: bool
    kill_switch_mode: str
    benchmark_mode: str
    top_return_threshold: float
    benchmark_return_threshold: float
    transaction_cost_bps: float
    slippage_bps: float


def clean_ticker(value: Any, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    text = str(value).strip().upper()
    return text or fallback


def parse_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return fallback


def parse_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else fallback
    except (TypeError, ValueError):
        return fallback


def parse_int(value: Any, fallback: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def normalize_groups(raw_groups: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_groups, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue

        name = str(raw_group.get("name", "")).strip().lower()
        proxy = clean_ticker(raw_group.get("proxy"))
        raw_stocks = raw_group.get("stocks", [])
        stocks: list[str] = []
        seen: set[str] = set()

        if isinstance(raw_stocks, list):
            for raw_stock in raw_stocks:
                ticker = clean_ticker(raw_stock)
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    stocks.append(ticker)

        if name and proxy and stocks:
            normalized.append({"name": name, "proxy": proxy, "stocks": stocks})

    return normalized


def normalize_weights(raw_weights: Any) -> list[float]:
    if not isinstance(raw_weights, list):
        return [0.4, 0.3, 0.2, 0.1]

    values = [parse_float(weight, 0.0) for weight in raw_weights]
    values = [value for value in values if value >= 0]
    total = sum(values)
    if total <= 0:
        return [0.4, 0.3, 0.2, 0.1]
    return [value / total for value in values]


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    return PreviewConfig(
        bot_name=str(payload.get("botName", "")).strip(),
        groups=normalize_groups(payload.get("groups")),
        benchmark=clean_ticker(payload.get("benchmark"), "VOO") or "VOO",
        cash_equivalent=clean_ticker(payload.get("cashEquivalent"), "VOO") or "VOO",
        lookback=parse_int(payload.get("lookback"), 20, minimum=1, maximum=120),
        rebalance_freq=parse_int(payload.get("rebalanceFreq"), 5, minimum=1, maximum=252),
        slot_count=parse_int(payload.get("slotCount"), 4, minimum=1, maximum=20),
        weights=normalize_weights(payload.get("weights")),
        enable_kill_switch=parse_bool(payload.get("enableKillSwitch"), True),
        enable_benchmark_filter=parse_bool(payload.get("enableBenchmarkFilter"), False),
        kill_switch_mode=str(payload.get("killSwitchMode", "top_negative")).strip() or "top_negative",
        benchmark_mode=str(payload.get("benchmarkMode", "benchmark_positive")).strip() or "benchmark_positive",
        top_return_threshold=parse_float(payload.get("topReturnThreshold"), 0.0),
        benchmark_return_threshold=parse_float(payload.get("benchmarkReturnThreshold"), 0.0),
        transaction_cost_bps=parse_float(payload.get("transactionCostBps"), DEFAULT_TRANSACTION_COST_BPS),
        slippage_bps=parse_float(payload.get("slippageBps"), DEFAULT_SLIPPAGE_BPS),
    )


def build_alpha2_config(preview: PreviewConfig) -> Alpha2Config:
    proxies = {group["name"]: group["proxy"] for group in preview.groups}
    stocks = {group["name"]: group["stocks"] for group in preview.groups}

    return Alpha2Config(
        bot_id=(preview.bot_name or "preview_alpha2").lower().replace(" ", "_"),
        proxies=proxies,
        stocks=stocks,
        lookback_days=preview.lookback,
        rebalance_freq=preview.rebalance_freq,
        stock_bench_mode="sector",
        rank_weights=np.array(preview.weights, dtype=float),
        cash_equivalent=preview.cash_equivalent,
        benchmark=preview.benchmark,
        enable_kill_switch=preview.enable_kill_switch,
        enable_benchmark_filter=preview.enable_benchmark_filter,
        kill_switch_mode=preview.kill_switch_mode,
        benchmark_mode=preview.benchmark_mode,
        top_return_threshold=preview.top_return_threshold,
        benchmark_return_threshold=preview.benchmark_return_threshold,
        transaction_cost_bps=preview.transaction_cost_bps,
        slippage_bps=preview.slippage_bps,
        num_stock_picks=preview.slot_count,
    )


def compute_cost_drag(turnover: float, transaction_cost_bps: float, slippage_bps: float) -> float:
    total_bps = float(transaction_cost_bps) + float(slippage_bps)
    return float(turnover) * (total_bps / 10_000.0)


def compute_turnover(previous_holdings: dict[str, float], current_holdings: dict[str, float]) -> float:
    symbols = set(previous_holdings.keys()) | set(current_holdings.keys())
    return sum(abs(float(current_holdings.get(symbol, 0.0)) - float(previous_holdings.get(symbol, 0.0))) for symbol in symbols)


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.groups:
        raise ValueError("At least one group with a proxy and stock list is required.")

    config = build_alpha2_config(preview)
    fallback_symbol = config.cash_equivalent or config.benchmark or "VOO"
    prices = build_universe_price_frame(config, START_DATE, END_DATE)
    if prices.empty or len(prices.index) < 2:
        raise ValueError("Not enough market data was returned for this preview.")

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rebalance_every_n_days = parse_rebalance_freq(config.rebalance_freq)

    equity = 1.0
    strategy_day_counter = 0
    prev_holdings: dict[str, float] = {}
    current_holdings: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    risk_off_days = 0

    for index, trading_ts in enumerate(prices.index):
        if index == 0:
            continue

        strategy_day_counter += 1
        rebalanced_today = False
        risk_off = False
        risk_reason = "carry_forward"
        top_sectors: list[str] = []
        selected_stocks: list[str] = []

        if is_rebalance_day(strategy_day_counter - 1, rebalance_every_n_days):
            pick = choose_universe_triggered_holdings(config, prices, index)
            risk_off = bool(pick.risk_off)
            risk_reason = pick.risk_reason
            top_sectors = pick.top_sectors
            selected_stocks = pick.selected_stocks
            rebalanced_today = True

            if risk_off:
                risk_off_days += 1

            _, next_holdings = compute_day_return_and_holdings(
                day_returns=returns.loc[trading_ts],
                selected_symbols=pick.symbols,
                weights=pick.weights,
            )
            current_holdings = next_holdings or prev_holdings.copy() or {fallback_symbol: 1.0}
        else:
            current_holdings = prev_holdings.copy() or {fallback_symbol: 1.0}

        gross_ret = 0.0
        day_returns = returns.loc[trading_ts]
        for symbol, weight in current_holdings.items():
            if symbol in day_returns.index and pd.notna(day_returns[symbol]):
                gross_ret += float(weight) * float(day_returns[symbol])

        turnover = compute_turnover(prev_holdings, current_holdings) if rebalanced_today else 0.0
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
                    "rebalanced_today": rebalanced_today,
                    "risk_off": risk_off,
                    "risk_reason": risk_reason,
                    "top_sectors": top_sectors,
                    "selected_stocks": selected_stocks,
                    "turnover": float(turnover),
                    "cost_drag": float(cost_drag),
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
