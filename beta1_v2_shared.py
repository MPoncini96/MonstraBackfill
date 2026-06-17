"""
beta1_v2_shared.py
==================
Phantom v2 composite scoring — shared core for live runner, backfill, and preview.

Source of truth: bots/beta1V2.py (uploaded 2026-06-17).
Do NOT import from beta1_shared or beta1_legacy_shared — this module is
intentionally isolated to prevent contamination between algorithm families.

Exports used by each consumer:
  bots/beta1_v2.py          : Beta1V2Config, Beta1V2ConfigError, build_live_config,
                               latest_target_weights
  backfill_beta1_v2.py      : Beta1V2Config, Beta1V2ConfigError,
                               run_beta1_v2_reconstruction, Beta1V2ReconstructionResult
  Preview_backfill_beta1_v2 : Beta1V2Config, run_beta1_v2_reconstruction,
                               Beta1V2ReconstructionResult
  tests/                    : compute_target_weights (for parity assertions)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from bot_identity import BOT_TYPE_BETA1_V2  # noqa: F401 — re-exported for consumers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — defaults used when the DB row is NULL or absent
# ---------------------------------------------------------------------------

MAX_HOLDINGS            = 4
FALLBACK_TICKER         = "VOO"
MIN_FINAL_SCORE         = 0.0
WEIGHT_ROUNDING         = 0.05
HISTORY_DAYS            = 220
MIN_DAILY_BARS          = 80

MOMENTUM_WEIGHT         = 1.00
CONFIRMATION_WEIGHT     = 0.10
RISK_WEIGHT             = 0.20

USE_MARKET_RISK_GATE    = True
MARKET_GATE_OVERRIDE_SCORE = 0.20


# ---------------------------------------------------------------------------
# Isolation utilities
# Do NOT import normalize_universe / safe_float / safe_int / clean_ticker from
# beta1_shared.  Duplicating these four trivial helpers here keeps beta1_v2
# fully isolated from the legacy algorithm family.
# ---------------------------------------------------------------------------

def normalize_universe(raw_universe: Any) -> list[str]:
    if raw_universe is None or not isinstance(raw_universe, (list, tuple)):
        return []
    seen: set[str] = set()
    universe: list[str] = []
    for item in raw_universe:
        if item is None:
            continue
        ticker = str(item).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        universe.append(ticker)
    return universe


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def clean_ticker(value: Any, default: str) -> str:
    if value is None:
        return default
    ticker = str(value).strip().upper()
    return ticker or default


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

class Beta1V2ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Beta1V2Config:
    """All parameters required to run the Phantom v2 composite scoring strategy."""

    bot_id:                     str
    universe:                   list[str]
    max_holdings:               int   = MAX_HOLDINGS
    fallback_ticker:            str   = FALLBACK_TICKER
    min_final_score:            float = MIN_FINAL_SCORE
    weight_rounding:            float = WEIGHT_ROUNDING
    history_days:               int   = HISTORY_DAYS
    min_daily_bars:             int   = MIN_DAILY_BARS
    momentum_weight:            float = MOMENTUM_WEIGHT
    confirmation_weight:        float = CONFIRMATION_WEIGHT
    risk_weight:                float = RISK_WEIGHT
    use_market_risk_gate:       bool  = USE_MARKET_RISK_GATE
    market_gate_override_score: float = MARKET_GATE_OVERRIDE_SCORE


# ---------------------------------------------------------------------------
# Config loader (DB-backed, for live runner)
# ---------------------------------------------------------------------------

def build_live_config(bot_id: str, use_db_config: bool = True) -> Beta1V2Config:
    """Load Beta1V2Config from trading.beta1_v2 via db.get_beta1_v2_config().

    Raises Beta1V2ConfigError on any failure so the live runner can return
    a HOLD signal rather than crashing.
    """
    if not use_db_config:
        raise Beta1V2ConfigError(
            "Live beta1_v2 requires DB-backed config. "
            "Pass an explicit Beta1V2Config to helper utilities instead."
        )

    try:
        from db import get_beta1_v2_config
    except Exception as exc:
        raise Beta1V2ConfigError(
            f"Could not import beta1_v2 config loader for {bot_id}: {exc}"
        ) from exc

    try:
        config_data: dict[str, Any] | None = get_beta1_v2_config(bot_id)
    except Exception as exc:
        raise Beta1V2ConfigError(
            f"Could not load beta1_v2 config for {bot_id}: {exc}"
        ) from exc

    if not config_data:
        raise Beta1V2ConfigError(
            f"No active beta1_v2 config found for {bot_id}."
        )

    universe = normalize_universe(config_data.get("universe"))
    if not universe:
        raise Beta1V2ConfigError(
            f"Active beta1_v2 config for {bot_id} has an empty or invalid universe."
        )

    return Beta1V2Config(
        bot_id=bot_id,
        universe=universe,
        max_holdings=safe_int(config_data.get("max_holdings"), MAX_HOLDINGS),
        fallback_ticker=clean_ticker(config_data.get("fallback_ticker"), FALLBACK_TICKER),
        min_final_score=safe_float(config_data.get("min_final_score"), MIN_FINAL_SCORE),
        weight_rounding=safe_float(config_data.get("weight_rounding"), WEIGHT_ROUNDING),
        history_days=safe_int(config_data.get("history_days"), HISTORY_DAYS),
        min_daily_bars=safe_int(config_data.get("min_daily_bars"), MIN_DAILY_BARS),
        momentum_weight=safe_float(config_data.get("momentum_weight"), MOMENTUM_WEIGHT),
        confirmation_weight=safe_float(config_data.get("confirmation_weight"), CONFIRMATION_WEIGHT),
        risk_weight=safe_float(config_data.get("risk_weight"), RISK_WEIGHT),
        use_market_risk_gate=bool(config_data.get("use_market_risk_gate", USE_MARKET_RISK_GATE)),
        market_gate_override_score=safe_float(
            config_data.get("market_gate_override_score"), MARKET_GATE_OVERRIDE_SCORE
        ),
    )


# ---------------------------------------------------------------------------
# Core scoring — do NOT alter these functions.
# Source of truth: bots/beta1V2.py uploaded 2026-06-17.
# ---------------------------------------------------------------------------

def _score_asset(series: pd.Series, config: Beta1V2Config) -> dict[str, float] | None:
    try:
        series = series.dropna()

        if len(series) < config.min_daily_bars:
            return None

        price       = float(series.iloc[-1])
        ret_2w      = float(series.iloc[-1] / series.iloc[-10] - 1)

        sma_20      = float(series.iloc[-20:].mean())
        sma_50      = float(series.iloc[-50:].mean())

        daily_returns   = series.pct_change().dropna()
        vol_20          = float(daily_returns.iloc[-20:].std() * math.sqrt(252))

        rolling_high_20 = float(series.iloc[-20:].max())
        drawdown_20     = float(price / rolling_high_20 - 1)
        drawdown_penalty = abs(min(drawdown_20, 0.0))

        downside_week    = min(ret_2w, 0.0)
        downside_penalty = abs(downside_week)

        momentum_score = ret_2w

        confirmation_score = (
            0.40 * np.tanh((price / sma_20 - 1) / 0.03)
            + 0.35 * np.tanh((price / sma_50 - 1) / 0.06)
            + 0.25 * np.tanh((sma_20 / sma_50 - 1) / 0.04)
        )

        risk_score = (
            0.55 * vol_20
            + 0.30 * drawdown_penalty
            + 0.15 * downside_penalty
        )

        final_score = (
            config.momentum_weight      * momentum_score
            + config.confirmation_weight * confirmation_score
            - config.risk_weight         * risk_score
        )

        return {
            "final_score":        float(final_score),
            "momentum_score":     float(momentum_score),
            "confirmation_score": float(confirmation_score),
            "risk_score":         float(risk_score),
        }

    except Exception:
        return None


def _market_is_healthy(close: pd.DataFrame, config: Beta1V2Config) -> bool:
    try:
        if config.fallback_ticker not in close.columns:
            return True
        series = close[config.fallback_ticker].dropna()
        if len(series) < 50:
            return True
        return float(series.iloc[-1]) > float(series.iloc[-50:].mean())
    except Exception:
        return True


def _normalize_weights(scores: dict[str, float]) -> dict[str, float]:
    positive_scores = {ticker: max(float(score), 0.0) for ticker, score in scores.items()}
    total = sum(positive_scores.values())
    if total <= 0:
        equal_weight = 1.0 / len(positive_scores)
        return {ticker: equal_weight for ticker in positive_scores}
    return {ticker: score / total for ticker, score in positive_scores.items()}


def _round_weights_to_sum(weights: dict[str, float], step: float) -> dict[str, float]:
    if not weights:
        return {}

    rounded = {
        ticker: round(weight / step) * step
        for ticker, weight in weights.items()
    }
    rounded = {ticker: weight for ticker, weight in rounded.items() if weight > 0}

    if not rounded:
        top = max(weights.items(), key=lambda x: x[1])[0]
        return {top: 1.0}

    diff         = round(1.0 - sum(rounded.values()), 10)
    steps_needed = int(round(diff / step))

    if steps_needed > 0:
        order = sorted(weights.keys(), key=lambda t: weights[t], reverse=True)
        for _ in range(steps_needed):
            for ticker in order:
                if ticker in rounded:
                    rounded[ticker] = round(rounded[ticker] + step, 10)
                    break

    elif steps_needed < 0:
        order = sorted(rounded.keys(), key=lambda t: weights.get(t, 0.0))
        for _ in range(abs(steps_needed)):
            for ticker in order:
                if rounded.get(ticker, 0.0) > step:
                    rounded[ticker] = round(rounded[ticker] - step, 10)
                    break

    rounded = {ticker: round(weight, 2) for ticker, weight in rounded.items() if weight > 0}

    total = round(sum(rounded.values()), 10)
    if abs(total - 1.0) > 0.0001 and rounded:
        top = max(rounded.keys(), key=lambda t: rounded[t])
        rounded[top] = round(rounded[top] + (1.0 - total), 2)

    return rounded


# ---------------------------------------------------------------------------
# Weight computation — the shared kernel used by all three paths
# ---------------------------------------------------------------------------

def compute_target_weights(
    config: Beta1V2Config,
    close: pd.DataFrame,
) -> tuple[dict[str, float], str | None, dict[str, Any]]:
    """Compute target weights from a pre-downloaded (or pre-sliced) price DataFrame.

    This is the single authoritative computation shared by the live runner,
    backfill, and preview.  The caller is responsible for providing a correctly
    sized close DataFrame (at least config.min_daily_bars rows).

    Returns:
        (target_weights, asof_date_str, diagnostics)
    """
    if close.empty or len(close) < config.min_daily_bars:
        return {config.fallback_ticker: 1.0}, None, {"reason": "insufficient_history"}

    market_is_healthy = _market_is_healthy(close, config)
    scores: dict[str, float] = {}
    diagnostics: dict[str, Any] = {
        "market_is_healthy": market_is_healthy,
        "scores": {},
    }

    for ticker in config.universe:
        if ticker not in close.columns:
            continue

        score_data = _score_asset(close[ticker], config)
        if not score_data:
            continue

        final_score = score_data["final_score"]
        diagnostics["scores"][ticker] = score_data

        if final_score <= config.min_final_score:
            continue

        if config.use_market_risk_gate:
            if not market_is_healthy and final_score < config.market_gate_override_score:
                continue

        scores[ticker] = final_score

    asof = str(close.index[-1].date())

    if not scores:
        return {config.fallback_ticker: 1.0}, asof, diagnostics

    ranked       = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected     = dict(ranked[: config.max_holdings])
    raw_weights  = _normalize_weights(selected)
    final_weights = _round_weights_to_sum(raw_weights, config.weight_rounding)

    return final_weights, asof, diagnostics


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_close(config: Beta1V2Config) -> pd.DataFrame:
    """Download recent closing prices for the live runner (uses period= parameter)."""
    tickers = sorted(set(config.universe + [config.fallback_ticker]))

    prices = yf.download(
        tickers,
        period=f"{config.history_days}d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.dropna(how="all")
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices


def download_close_range(config: Beta1V2Config, start: str, end: str) -> pd.DataFrame:
    """Download closing prices over an explicit date range for backfill / preview."""
    tickers = sorted(set(config.universe + [config.fallback_ticker]))

    prices = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.dropna(how="all")
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices


def latest_target_weights(
    config: Beta1V2Config,
) -> tuple[dict[str, float], str | None, dict[str, Any]]:
    """Download prices then compute weights.  Used by the live runner only."""
    close = download_close(config)
    return compute_target_weights(config, close)


# ---------------------------------------------------------------------------
# Reconstruction — used by backfill and preview
# ---------------------------------------------------------------------------

@dataclass
class Beta1V2ReconstructionResult:
    """Output of run_beta1_v2_reconstruction()."""

    daily_returns:  pd.Series           # index = date, values = daily portfolio return
    equity_curve:   pd.Series           # index = date, values = cumulative equity (starts at 1.0)
    weights_history: pd.DataFrame       # index = date, columns = tickers + fallback
    final_weights:  dict[str, float]    # target weights as of the last date
    final_asof:     str                 # ISO date string of final weights


def run_beta1_v2_reconstruction(
    config: Beta1V2Config,
    start: str,
    end: str,
) -> Beta1V2ReconstructionResult:
    """
    Walk through price history day by day, computing daily target weights and
    portfolio returns.  Used by backfill and preview.

    The algorithm is NOT path-dependent: target weights at date T depend only
    on prices up to T.  Each day the full scoring logic is re-evaluated on the
    expanding window close[:T].

    Args:
        config: Beta1V2Config (universe, scoring weights, risk gate params, etc.)
        start:  ISO date string — beginning of price download
        end:    ISO date string — end of price download (exclusive, yfinance convention)

    Returns:
        Beta1V2ReconstructionResult with daily_returns, equity_curve, and weights_history.

    Raises:
        ValueError if insufficient price history is available.
    """
    close = download_close_range(config, start, end)

    if close.empty or len(close) < config.min_daily_bars:
        raise ValueError(
            f"Insufficient price history for beta1_v2 reconstruction "
            f"(got {len(close)} bars, need {config.min_daily_bars})."
        )

    dates = close.index.tolist()

    # Start fully invested in the fallback ticker
    current_weights: dict[str, float] = {config.fallback_ticker: 1.0}

    daily_ret_records:   list[dict[str, Any]] = []
    weights_records:     list[dict[str, Any]] = []
    equity              = 1.0

    for i, d in enumerate(dates):
        # ---- 1. Compute the portfolio return for day i ----------------------
        # On day 0 there is no previous day, so no return to record.
        if i > 0:
            day_ret = 0.0
            for ticker, w in current_weights.items():
                if ticker not in close.columns:
                    continue
                p_prev = close[ticker].iloc[i - 1]
                p_curr = close[ticker].iloc[i]
                if (
                    pd.notna(p_prev) and pd.notna(p_curr)
                    and float(p_prev) > 0
                ):
                    day_ret += w * (float(p_curr) / float(p_prev) - 1.0)
            equity *= (1.0 + day_ret)
            daily_ret_records.append({"date": d, "ret": day_ret, "equity": equity})

        # ---- 2. Recompute target weights using prices up to and including day i
        window = close.iloc[: i + 1]
        new_weights, _, _ = compute_target_weights(config, window)
        current_weights = new_weights

        row: dict[str, Any] = {"date": d}
        row.update(current_weights)
        weights_records.append(row)

    if not daily_ret_records:
        raise ValueError("No daily return records produced — price history too short.")

    returns_df = (
        pd.DataFrame(daily_ret_records)
        .set_index("date")[["ret"]]
        .rename(columns={"ret": "daily_return"})
    )
    equity_df = (
        pd.DataFrame(daily_ret_records)
        .set_index("date")[["equity"]]
        .rename(columns={"equity": "equity_value"})
    )
    weights_df = (
        pd.DataFrame(weights_records)
        .set_index("date")
        .fillna(0.0)
    )

    return Beta1V2ReconstructionResult(
        daily_returns=returns_df["daily_return"],
        equity_curve=equity_df["equity_value"],
        weights_history=weights_df,
        final_weights=current_weights,
        final_asof=str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1]),
    )
