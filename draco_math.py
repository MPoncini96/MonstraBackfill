"""draco_math.py - Pure regression/scoring/target functions for the Draco algorithm family.

No I/O; every function here is deterministic given plain numeric inputs, so it is
directly unit-testable without mocking price downloads or the database. Live
signal generation, backfill, and preview/backtest all call into this module so
the mathematical behavior is identical across every execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants (Draco strategy defaults - see ADDING_NEW_ALGORITHM.md and the
# Draco design spec in Monstra-Worker/FromQC/DracoQC.py)
# ---------------------------------------------------------------------------

LOOKBACKS: dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "4Y": 1008,
    "5Y": 1260,
}

TIMEFRAME_WEIGHTS: dict[str, float] = {
    "1M": 0.75,
    "3M": 1.00,
    "6M": 1.15,
    "1Y": 1.30,
    "2Y": 1.50,
    "3Y": 1.70,
    "4Y": 1.85,
    "5Y": 2.00,
}

TARGET_LOOKBACKS: tuple[str, ...] = ("3M", "6M", "1Y", "2Y")
LONG_TERM_LOOKBACKS: tuple[str, ...] = ("2Y", "3Y", "4Y", "5Y")

Z_SCORE_CLAMP = 4.0
ANNUALIZED_SLOPE_EXPONENT_CLAMP = 20.0
EXP_OVERFLOW_GUARD = 700.0  # math.exp overflows ~709.78 for float64
TRADING_DAYS_PER_YEAR = 252

MINIMUM_WINDOW_LENGTH = 3


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return hi if value > 0 else lo
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Regression fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeframeMetrics:
    label: str
    lookback: int
    slope: float
    intercept: float
    endpoint_index: int
    current_price: float
    fitted_price: float
    residual_std: float
    z_score: float
    annualized_slope: float
    r_squared: float


def fit_log_linear_regression(
    prices: Sequence[float],
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """OLS fit of ln(price) = intercept + slope * index over `prices`.

    Returns (slope, intercept, index_array, log_prices), or None if the series
    is too short or contains non-finite / non-positive values.
    """
    arr = np.asarray(prices, dtype=float)
    if arr.size < MINIMUM_WINDOW_LENGTH:
        return None
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        return None

    index = np.arange(arr.size, dtype=float)
    log_prices = np.log(arr)
    slope, intercept = np.polyfit(index, log_prices, 1)
    return float(slope), float(intercept), index, log_prices


def compute_timeframe_metrics(
    label: str,
    lookback: int,
    prices: Sequence[float],
) -> TimeframeMetrics | None:
    """Compute regression metrics for one timeframe using the trailing `lookback` prices.

    `prices` should be the full adjusted-close history ending at "today"; only
    the trailing `lookback` entries are used. Returns None when there isn't
    enough valid history for this timeframe.
    """
    if len(prices) < lookback:
        return None
    window = prices[-lookback:]
    fit = fit_log_linear_regression(window)
    if fit is None:
        return None
    slope, intercept, index, log_prices = fit

    endpoint_index = len(window) - 1
    fitted_log = intercept + slope * index
    residuals = log_prices - fitted_log
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else 0.0

    current_price = float(window[-1])
    fitted_current_log = intercept + slope * endpoint_index
    fitted_price = math.exp(_clamp(fitted_current_log, -EXP_OVERFLOW_GUARD, EXP_OVERFLOW_GUARD))

    if residual_std > 1e-12:
        raw_z = (math.log(current_price) - fitted_current_log) / residual_std
    else:
        raw_z = 0.0
    z_score = _clamp(raw_z, -Z_SCORE_CLAMP, Z_SCORE_CLAMP)

    annualized_exponent = _clamp(
        slope * TRADING_DAYS_PER_YEAR,
        -ANNUALIZED_SLOPE_EXPONENT_CLAMP,
        ANNUALIZED_SLOPE_EXPONENT_CLAMP,
    )
    annualized_slope = math.exp(annualized_exponent) - 1.0

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_prices - np.mean(log_prices)) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    return TimeframeMetrics(
        label=label,
        lookback=lookback,
        slope=slope,
        intercept=intercept,
        endpoint_index=endpoint_index,
        current_price=current_price,
        fitted_price=fitted_price,
        residual_std=residual_std,
        z_score=z_score,
        annualized_slope=annualized_slope,
        r_squared=r_squared,
    )


def compute_all_timeframe_metrics(
    prices: Sequence[float],
    *,
    lookbacks: dict[str, int] = LOOKBACKS,
) -> dict[str, TimeframeMetrics]:
    """Compute metrics for every configured timeframe with sufficient history."""
    metrics: dict[str, TimeframeMetrics] = {}
    for label, lookback in lookbacks.items():
        m = compute_timeframe_metrics(label, lookback, prices)
        if m is not None:
            metrics[label] = m
    return metrics


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def deviation_points(z_score: float) -> int:
    if z_score <= -2.50:
        return 6
    if z_score <= -2.00:
        return 5
    if z_score <= -1.50:
        return 4
    if z_score <= -1.00:
        return 3
    if z_score <= -0.50:
        return 2
    if z_score <= 0.00:
        return 1
    return 0


def annualized_slope_points(annualized_slope: float) -> int:
    if annualized_slope >= 0.40:
        return 6
    if annualized_slope >= 0.25:
        return 5
    if annualized_slope >= 0.15:
        return 4
    if annualized_slope >= 0.08:
        return 3
    if annualized_slope >= 0.03:
        return 2
    if annualized_slope >= 0.00:
        return 1
    return 0


def r_squared_points(r_squared: float) -> int:
    if r_squared >= 0.80:
        return 3
    if r_squared >= 0.60:
        return 2
    if r_squared >= 0.35:
        return 1
    return 0


def score_timeframe(metrics: TimeframeMetrics, weight: float) -> float:
    points = (
        deviation_points(metrics.z_score)
        + annualized_slope_points(metrics.annualized_slope)
        + r_squared_points(metrics.r_squared)
    )
    return points * weight


def compute_total_score(
    metrics: dict[str, TimeframeMetrics],
    *,
    weights: dict[str, float] = TIMEFRAME_WEIGHTS,
) -> float:
    return sum(score_timeframe(m, weights.get(label, 1.0)) for label, m in metrics.items())


# ---------------------------------------------------------------------------
# Entry requirements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntryRequirements:
    minimum_score: float = 30.0
    minimum_valid_timeframes: int = 6
    minimum_below_line_timeframes: int = 2
    minimum_positive_slope_timeframes: int = 4
    require_positive_long_term_slope: bool = True
    long_term_lookbacks: tuple[str, ...] = LONG_TERM_LOOKBACKS


DEFAULT_ENTRY_REQUIREMENTS = EntryRequirements()


def evaluate_entry_requirements(
    metrics: dict[str, TimeframeMetrics],
    total_score: float,
    current_price: float,
    requirements: EntryRequirements = DEFAULT_ENTRY_REQUIREMENTS,
) -> bool:
    if len(metrics) < requirements.minimum_valid_timeframes:
        return False

    below_line = sum(1 for m in metrics.values() if current_price < m.fitted_price)
    if below_line < requirements.minimum_below_line_timeframes:
        return False

    positive_slope = sum(1 for m in metrics.values() if m.slope > 0)
    if positive_slope < requirements.minimum_positive_slope_timeframes:
        return False

    if requirements.require_positive_long_term_slope:
        has_positive_long_term = any(
            metrics[label].slope > 0
            for label in requirements.long_term_lookbacks
            if label in metrics
        )
        if not has_positive_long_term:
            return False

    if total_score < requirements.minimum_score:
        return False

    return True


# ---------------------------------------------------------------------------
# Locked target selection + projection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LockedTarget:
    label: str
    lookback: int
    slope: float
    intercept: float
    endpoint_index: int
    r_squared: float
    entry_price: float
    entry_target_price: float
    entry_upside: float


@dataclass(frozen=True)
class TargetRequirements:
    minimum_r_squared: float = 0.30
    require_positive_slope: bool = True
    require_price_below_target_line: bool = True
    minimum_entry_upside: float = 0.01
    maximum_entry_upside: float = 0.50
    candidate_labels: tuple[str, ...] = TARGET_LOOKBACKS


DEFAULT_TARGET_REQUIREMENTS = TargetRequirements()


def select_locked_target(
    metrics: dict[str, TimeframeMetrics],
    current_price: float,
    requirements: TargetRequirements = DEFAULT_TARGET_REQUIREMENTS,
) -> LockedTarget | None:
    """Choose the locked exit target per Draco's eligibility + tie-break rules.

    Eligibility: R² >= minimum, positive slope, price below the target line,
    upside within [minimum, maximum]. Tie-break: highest R² -> highest upside
    -> longest lookback.
    """
    candidates: list[tuple[TimeframeMetrics, float]] = []
    for label in requirements.candidate_labels:
        m = metrics.get(label)
        if m is None:
            continue
        if m.r_squared < requirements.minimum_r_squared:
            continue
        if requirements.require_positive_slope and m.slope <= 0:
            continue
        if requirements.require_price_below_target_line and current_price >= m.fitted_price:
            continue
        if current_price <= 0:
            continue
        upside = m.fitted_price / current_price - 1.0
        if upside < requirements.minimum_entry_upside or upside > requirements.maximum_entry_upside:
            continue
        candidates.append((m, upside))

    if not candidates:
        return None

    best_m, best_upside = max(candidates, key=lambda item: (item[0].r_squared, item[1], item[0].lookback))

    return LockedTarget(
        label=best_m.label,
        lookback=best_m.lookback,
        slope=best_m.slope,
        intercept=best_m.intercept,
        endpoint_index=best_m.endpoint_index,
        r_squared=best_m.r_squared,
        entry_price=current_price,
        entry_target_price=best_m.fitted_price,
        entry_upside=best_upside,
    )


def project_locked_target(target: LockedTarget, trading_days_held: int) -> float:
    """Project the locked regression line forward `trading_days_held` sessions.

    The target's slope/intercept/endpoint_index are frozen at entry and never
    recalculated; only the projection index advances.
    """
    projected_index = target.endpoint_index + trading_days_held
    exponent = _clamp(
        target.intercept + target.slope * projected_index,
        -EXP_OVERFLOW_GUARD,
        EXP_OVERFLOW_GUARD,
    )
    return math.exp(exponent)
