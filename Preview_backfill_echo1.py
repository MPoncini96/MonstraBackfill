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

from bots.echo1 import (
    DEFAULT_LAG_RANGE,
    DEFAULT_RETURN_PERIOD_RANGE,
    Echo1Config,
    build_network_scores,
    discover_pairs,
    download_echo1_prices,
)

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()


@dataclass
class PreviewConfig:
    bot_name: str
    universe: list[str]
    fallback_ticker: str
    portfolio_size: int
    target_gross_exposure: float
    discovery_lookback: int
    max_pairs: int
    lag_range: list[int]
    return_period_range: list[int]
    use_market_filter: bool
    market_filter_fallback_pct: float
    max_drawdown_pause: float


def _clean_ticker(value: Any, fallback: str = "VOO") -> str:
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


def _normalize_universe(raw_tickers: Any) -> list[str]:
    if not isinstance(raw_tickers, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_ticker in raw_tickers:
        ticker = _clean_ticker(raw_ticker, "")
        if ticker and ticker not in seen:
            seen.add(ticker)
            normalized.append(ticker)
    return normalized


def _parse_int_list(raw_values: Any, fallback: list[int]) -> list[int]:
    if not isinstance(raw_values, list):
        return list(fallback)
    values: list[int] = []
    for raw_value in raw_values:
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            values.append(parsed)
    unique = sorted(set(values))
    return unique or list(fallback)


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    return PreviewConfig(
        bot_name=str(payload.get("botName", "")).strip(),
        universe=_normalize_universe(payload.get("tickers")),
        fallback_ticker=_clean_ticker(payload.get("fallbackTicker"), "VOO"),
        portfolio_size=_safe_int(payload.get("portfolioSize"), 4),
        target_gross_exposure=_safe_fraction(payload.get("targetGrossExposure"), 0.95, 0.1, 1.0),
        discovery_lookback=_safe_int(payload.get("discoveryLookback"), 504, 60),
        max_pairs=_safe_int(payload.get("maxPairs"), 75),
        lag_range=_parse_int_list(payload.get("lagRange"), list(DEFAULT_LAG_RANGE)),
        return_period_range=_parse_int_list(payload.get("returnPeriodRange"), list(DEFAULT_RETURN_PERIOD_RANGE)),
        use_market_filter=bool(payload.get("useMarketFilter", True)),
        market_filter_fallback_pct=_safe_fraction(payload.get("marketFilterFallbackPct"), 0.5, 0.0, 1.0),
        max_drawdown_pause=_safe_fraction(payload.get("maxDrawdownPause"), 0.2, 0.01, 0.95),
    )


def build_echo1_config(preview: PreviewConfig) -> Echo1Config:
    return Echo1Config(
        universe=preview.universe,
        fallback_ticker=preview.fallback_ticker,
        portfolio_size=preview.portfolio_size,
        target_gross_exposure=preview.target_gross_exposure,
        discovery_lookback=preview.discovery_lookback,
        max_pairs=preview.max_pairs,
        lag_range=preview.lag_range,
        return_period_range=preview.return_period_range,
        use_market_filter=preview.use_market_filter,
        market_filter_fallback_pct=preview.market_filter_fallback_pct,
        max_drawdown_pause=preview.max_drawdown_pause,
    )


def _compute_return(holdings: dict[str, float], close_all, date_idx: int, all_dates) -> float:
    if not holdings or date_idx < 1:
        return 0.0

    prev_ts = all_dates[date_idx - 1]
    curr_ts = all_dates[date_idx]
    total_ret = 0.0
    for ticker, weight in holdings.items():
        if ticker not in close_all.columns:
            continue
        p_prev = close_all.loc[prev_ts, ticker]
        p_curr = close_all.loc[curr_ts, ticker]
        if p_prev is None or p_curr is None or p_prev <= 0:
            continue
        total_ret += float(weight) * (float(p_curr) / float(p_prev) - 1.0)
    return total_ret


def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.universe:
        raise ValueError("At least one universe ticker is required.")

    cfg = build_echo1_config(preview)
    all_tickers = list(dict.fromkeys(cfg.universe + [cfg.fallback_ticker]))
    close_all = download_echo1_prices(all_tickers, cfg.discovery_lookback + 420)
    if close_all.empty:
        raise ValueError("Not enough market data was returned for this preview.")

    backfill_dates = close_all.loc[START_DATE:END_DATE].index
    if backfill_dates.empty:
        raise ValueError("No trading dates were available in the preview window.")

    all_dates = close_all.index
    warmup_end_idx = cfg.discovery_lookback + 50

    equity = 1.0
    high_watermark = equity
    holdings: dict[str, float] = {}
    drawdown_pause_active = False
    drawdown_pause_idx: int | None = None
    discovered_pairs: list[dict[str, Any]] = []
    last_pair_refresh_month: int | None = None
    rows: list[dict[str, Any]] = []
    fallback_days = 0
    pair_refreshes = 0

    def _fallback_weights(pct: float) -> dict[str, float]:
        return {cfg.fallback_ticker: pct} if cfg.fallback_ticker in close_all.columns and pct > 0 else {}

    for date_ts in backfill_dates:
        date_idx = all_dates.get_loc(date_ts)
        trade_date = date_ts.date().isoformat()

        if date_idx < warmup_end_idx:
            rows.append({
                "d": trade_date,
                "equity": float(equity),
                "ret": 0.0,
                "holdings": {},
                "meta": {"regime": "warmup"},
            })
            continue

        ret = _compute_return(holdings, close_all, date_idx, all_dates)
        equity *= 1.0 + ret
        high_watermark = max(high_watermark, equity)
        drawdown = 1 - equity / high_watermark if high_watermark > 0 else 0.0

        regime = "active"

        if drawdown >= cfg.max_drawdown_pause:
            if not drawdown_pause_active:
                drawdown_pause_active = True
                drawdown_pause_idx = date_idx
            holdings = _fallback_weights(cfg.target_gross_exposure)
            fallback_days += 1
            regime = "drawdown_pause"
        else:
            if drawdown_pause_active:
                recovery_threshold = cfg.max_drawdown_pause * cfg.drawdown_recovery_pct
                days_paused = 0 if drawdown_pause_idx is None else max(0, date_idx - drawdown_pause_idx)
                if drawdown < recovery_threshold and days_paused >= cfg.drawdown_cooldown_days:
                    drawdown_pause_active = False
                    drawdown_pause_idx = None
                else:
                    holdings = _fallback_weights(cfg.target_gross_exposure)
                    fallback_days += 1
                    regime = "drawdown_recovering"

            if regime == "active" and cfg.use_market_filter and cfg.fallback_ticker in close_all.columns:
                hist_slice = close_all.iloc[max(0, date_idx - cfg.market_sma_period - 5): date_idx + 1]
                fb_px = hist_slice[cfg.fallback_ticker].dropna()
                if len(fb_px) >= cfg.market_sma_period:
                    if float(fb_px.iloc[-1]) < float(fb_px.tail(cfg.market_sma_period).mean()):
                        holdings = _fallback_weights(cfg.market_filter_fallback_pct)
                        fallback_days += 1
                        regime = "market_filter"

        if regime == "active":
            if last_pair_refresh_month is None or date_ts.month != last_pair_refresh_month:
                close_window = close_all.iloc[max(0, date_idx - cfg.discovery_lookback - 30): date_idx + 1]
                discovered_pairs = discover_pairs(close_window, cfg)
                if len(discovered_pairs) < 10:
                    discovered_pairs = discover_pairs(close_window, cfg, min_corr=cfg.min_avg_abs_corr_loose)
                last_pair_refresh_month = date_ts.month
                pair_refreshes += 1

            if not discovered_pairs:
                holdings = _fallback_weights(cfg.target_gross_exposure)
                fallback_days += 1
                regime = "no_pairs"
            else:
                max_lb = max(
                    (int(pair["lag_days"]) + int(pair["return_period"]) + 5 for pair in discovered_pairs),
                    default=20,
                )
                score_window = close_all.iloc[max(0, date_idx - max_lb): date_idx + 1]
                network_scores = build_network_scores(score_window, discovered_pairs, cfg)
                ranked = sorted(network_scores.values(), key=lambda item: item["final_score"], reverse=True)
                selected = [item for item in ranked if item["final_score"] > 0][: cfg.portfolio_size]

                if not selected:
                    holdings = _fallback_weights(cfg.target_gross_exposure)
                    fallback_days += 1
                    regime = "no_signal"
                else:
                    total_score = sum(item["final_score"] for item in selected)
                    holdings = {
                        item["ticker"]: round(
                            (item["final_score"] / total_score) * cfg.target_gross_exposure,
                            6,
                        )
                        for item in selected
                    }

        rows.append({
            "d": trade_date,
            "equity": float(equity),
            "ret": float(ret),
            "holdings": holdings,
            "meta": {
                "regime": regime,
                "num_pairs": len(discovered_pairs),
                "portfolio_size": cfg.portfolio_size,
            },
        })

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
            "fallbackDays": fallback_days,
            "pairRefreshes": pair_refreshes,
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
