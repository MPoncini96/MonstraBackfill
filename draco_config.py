"""draco_config.py - DracoConfig schema, validation, and DB config loading.

Split out of bots/draco.py to keep that module focused on live-run
orchestration (CLAUDE.md: keep files under ~500 lines).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import draco_math as dm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (Draco strategy defaults)
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_MARKET_REGIME_TICKER = "SPY"
DEFAULT_FALLBACK_TICKER = "QQQ"

DEFAULT_MAX_POSITIONS = 8
DEFAULT_MAX_NEW_POSITIONS_PER_SCAN = 8
DEFAULT_TARGET_GROSS_EXPOSURE = 0.96
DEFAULT_SCAN_CADENCE = "weekly"

DEFAULT_STOP_LOSS_PERCENT = 0.10
DEFAULT_MAXIMUM_HOLDING_TRADING_DAYS = 126
DEFAULT_SYMBOL_REENTRY_COOLDOWN_DAYS = 5

DEFAULT_USE_MARKET_REGIME_FILTER = True
DEFAULT_MARKET_SMA_PERIOD = 200
DEFAULT_REGIME_CONFIRMATION_DAYS = 5
DEFAULT_LIQUIDATE_TO_FALLBACK_DURING_RISK_OFF = True
DEFAULT_HOLD_FALLBACK_WHEN_EMPTY = True

DEFAULT_USE_PORTFOLIO_CIRCUIT_BREAKER = True
DEFAULT_MAXIMUM_PORTFOLIO_DRAWDOWN = 0.15
DEFAULT_CIRCUIT_BREAKER_COOLDOWN_TRADING_DAYS = 20


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DracoConfig:
    universe: list[str]
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    market_regime_ticker: str = DEFAULT_MARKET_REGIME_TICKER
    fallback_ticker: str = DEFAULT_FALLBACK_TICKER

    max_positions: int = DEFAULT_MAX_POSITIONS
    max_new_positions_per_scan: int = DEFAULT_MAX_NEW_POSITIONS_PER_SCAN
    target_gross_exposure: float = DEFAULT_TARGET_GROSS_EXPOSURE
    scan_cadence: str = DEFAULT_SCAN_CADENCE

    entry_lookbacks: dict[str, int] = field(default_factory=lambda: dict(dm.LOOKBACKS))
    target_lookbacks: tuple[str, ...] = dm.TARGET_LOOKBACKS
    timeframe_weights: dict[str, float] = field(default_factory=lambda: dict(dm.TIMEFRAME_WEIGHTS))
    long_term_lookbacks: tuple[str, ...] = dm.LONG_TERM_LOOKBACKS

    minimum_entry_score: float = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_score
    minimum_valid_timeframes: int = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_valid_timeframes
    minimum_below_regression_timeframes: int = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_below_line_timeframes
    minimum_positive_slope_timeframes: int = dm.DEFAULT_ENTRY_REQUIREMENTS.minimum_positive_slope_timeframes
    require_positive_long_term_slope: bool = dm.DEFAULT_ENTRY_REQUIREMENTS.require_positive_long_term_slope

    minimum_target_r_squared: float = dm.DEFAULT_TARGET_REQUIREMENTS.minimum_r_squared
    require_positive_target_slope: bool = dm.DEFAULT_TARGET_REQUIREMENTS.require_positive_slope
    require_price_below_target_line: bool = dm.DEFAULT_TARGET_REQUIREMENTS.require_price_below_target_line
    minimum_entry_target_upside: float = dm.DEFAULT_TARGET_REQUIREMENTS.minimum_entry_upside
    maximum_entry_target_upside: float = dm.DEFAULT_TARGET_REQUIREMENTS.maximum_entry_upside

    stop_loss_percent: float = DEFAULT_STOP_LOSS_PERCENT
    maximum_holding_trading_days: int = DEFAULT_MAXIMUM_HOLDING_TRADING_DAYS
    symbol_reentry_cooldown_days: int = DEFAULT_SYMBOL_REENTRY_COOLDOWN_DAYS

    use_market_regime_filter: bool = DEFAULT_USE_MARKET_REGIME_FILTER
    market_sma_period: int = DEFAULT_MARKET_SMA_PERIOD
    regime_confirmation_days: int = DEFAULT_REGIME_CONFIRMATION_DAYS
    liquidate_to_fallback_during_risk_off: bool = DEFAULT_LIQUIDATE_TO_FALLBACK_DURING_RISK_OFF
    hold_fallback_when_empty: bool = DEFAULT_HOLD_FALLBACK_WHEN_EMPTY

    use_portfolio_circuit_breaker: bool = DEFAULT_USE_PORTFOLIO_CIRCUIT_BREAKER
    maximum_portfolio_drawdown: float = DEFAULT_MAXIMUM_PORTFOLIO_DRAWDOWN
    circuit_breaker_cooldown_trading_days: int = DEFAULT_CIRCUIT_BREAKER_COOLDOWN_TRADING_DAYS

    def entry_requirements(self) -> dm.EntryRequirements:
        return dm.EntryRequirements(
            minimum_score=self.minimum_entry_score,
            minimum_valid_timeframes=self.minimum_valid_timeframes,
            minimum_below_line_timeframes=self.minimum_below_regression_timeframes,
            minimum_positive_slope_timeframes=self.minimum_positive_slope_timeframes,
            require_positive_long_term_slope=self.require_positive_long_term_slope,
            long_term_lookbacks=tuple(self.long_term_lookbacks),
        )

    def target_requirements(self) -> dm.TargetRequirements:
        return dm.TargetRequirements(
            minimum_r_squared=self.minimum_target_r_squared,
            require_positive_slope=self.require_positive_target_slope,
            require_price_below_target_line=self.require_price_below_target_line,
            minimum_entry_upside=self.minimum_entry_target_upside,
            maximum_entry_upside=self.maximum_entry_target_upside,
            candidate_labels=tuple(self.target_lookbacks),
        )


def validate_draco_config(config: DracoConfig) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: list[str] = []
    universe = [t for t in config.universe if t]
    if len(set(universe)) != len(universe):
        errors.append("universe contains duplicate tickers")
    if not universe:
        errors.append("universe must not be empty")
    if not config.fallback_ticker:
        errors.append("fallbackTicker must be set")
    if config.fallback_ticker in universe:
        errors.append("fallbackTicker must not also appear in universe")
    if not config.benchmark_ticker:
        errors.append("benchmarkTicker must be set")
    if config.max_positions < 1:
        errors.append("maxPositions must be >= 1")
    if not (0 < config.target_gross_exposure <= 1):
        errors.append("targetGrossExposure must be in (0, 1]")
    if not (0 < config.stop_loss_percent < 1):
        errors.append("stopLossPercent must be in (0, 1)")
    if not (0 < config.maximum_portfolio_drawdown < 1):
        errors.append("maximumPortfolioDrawdown must be in (0, 1)")
    if config.minimum_entry_target_upside >= config.maximum_entry_target_upside:
        errors.append("minimumEntryTargetUpside must be less than maximumEntryTargetUpside")
    if config.minimum_valid_timeframes > len(config.entry_lookbacks):
        errors.append("minimumValidTimeframes cannot exceed the number of entryLookbacks")
    if config.minimum_positive_slope_timeframes > len(config.entry_lookbacks):
        errors.append("minimumPositiveSlopeTimeframes cannot exceed the number of entryLookbacks")
    missing_target_lookbacks = [lb for lb in config.target_lookbacks if lb not in config.entry_lookbacks]
    if missing_target_lookbacks:
        errors.append(f"targetLookbacks not present in entryLookbacks: {missing_target_lookbacks}")
    if any(w < 0 for w in config.timeframe_weights.values()):
        errors.append("timeframeWeights must not be negative")
    return errors


def _safe_float(v: Any, default: float) -> float:
    try:
        parsed = float(v)
        return parsed if np.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(v))
    except (TypeError, ValueError):
        return default


def _safe_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(v)


def _normalize_universe(v: Any, exclude: str | None = None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    if not isinstance(v, (list, tuple)):
        return []
    exclude_norm = (exclude or "").strip().upper()
    seen: set[str] = set()
    out: list[str] = []
    for item in v:
        t = str(item).strip().upper()
        if t and t != exclude_norm and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def draco_config_from_db_dict(d: dict[str, Any]) -> DracoConfig:
    """Build a runtime DracoConfig from a trading.draco row dict (params JSONB + top-level cols)."""
    params_raw = d.get("params")
    if isinstance(params_raw, str):
        try:
            params_raw = json.loads(params_raw)
        except Exception:
            params_raw = {}
    params: dict[str, Any] = params_raw if isinstance(params_raw, dict) else {}

    fallback_ticker = str(d.get("fallback_ticker") or params.get("fallbackTicker") or DEFAULT_FALLBACK_TICKER).strip().upper()
    universe = _normalize_universe(d.get("universe") or params.get("universe"), exclude=fallback_ticker)

    defaults = DracoConfig(universe=list(universe))
    return DracoConfig(
        universe=universe or defaults.universe,
        benchmark_ticker=str(d.get("benchmark_ticker") or params.get("benchmarkTicker") or DEFAULT_BENCHMARK_TICKER).strip().upper(),
        market_regime_ticker=str(d.get("market_regime_ticker") or params.get("marketRegimeTicker") or DEFAULT_MARKET_REGIME_TICKER).strip().upper(),
        fallback_ticker=fallback_ticker,
        max_positions=_safe_int(params.get("maxPositions"), DEFAULT_MAX_POSITIONS),
        max_new_positions_per_scan=_safe_int(params.get("maxNewPositionsPerScan"), DEFAULT_MAX_NEW_POSITIONS_PER_SCAN),
        target_gross_exposure=_safe_float(params.get("targetGrossExposure"), DEFAULT_TARGET_GROSS_EXPOSURE),
        scan_cadence=str(params.get("scanCadence") or DEFAULT_SCAN_CADENCE),
        minimum_entry_score=_safe_float(params.get("minimumEntryScore"), defaults.minimum_entry_score),
        minimum_valid_timeframes=_safe_int(params.get("minimumValidTimeframes"), defaults.minimum_valid_timeframes),
        minimum_below_regression_timeframes=_safe_int(params.get("minimumBelowRegressionTimeframes"), defaults.minimum_below_regression_timeframes, minimum=0),
        minimum_positive_slope_timeframes=_safe_int(params.get("minimumPositiveSlopeTimeframes"), defaults.minimum_positive_slope_timeframes, minimum=0),
        require_positive_long_term_slope=_safe_bool(params.get("requirePositiveLongTermSlope"), defaults.require_positive_long_term_slope),
        minimum_target_r_squared=_safe_float(params.get("minimumTargetRSquared"), defaults.minimum_target_r_squared),
        require_positive_target_slope=_safe_bool(params.get("requirePositiveTargetSlope"), defaults.require_positive_target_slope),
        require_price_below_target_line=_safe_bool(params.get("requirePriceBelowTargetLine"), defaults.require_price_below_target_line),
        minimum_entry_target_upside=_safe_float(params.get("minimumEntryTargetUpside"), defaults.minimum_entry_target_upside),
        maximum_entry_target_upside=_safe_float(params.get("maximumEntryTargetUpside"), defaults.maximum_entry_target_upside),
        stop_loss_percent=_safe_float(params.get("stopLossPercent"), DEFAULT_STOP_LOSS_PERCENT),
        maximum_holding_trading_days=_safe_int(params.get("maximumHoldingTradingDays"), DEFAULT_MAXIMUM_HOLDING_TRADING_DAYS),
        symbol_reentry_cooldown_days=_safe_int(params.get("symbolReentryCooldownDays"), DEFAULT_SYMBOL_REENTRY_COOLDOWN_DAYS, minimum=0),
        use_market_regime_filter=_safe_bool(params.get("useMarketRegimeFilter"), DEFAULT_USE_MARKET_REGIME_FILTER),
        market_sma_period=_safe_int(params.get("marketSmaPeriod"), DEFAULT_MARKET_SMA_PERIOD),
        regime_confirmation_days=_safe_int(params.get("regimeConfirmationDays"), DEFAULT_REGIME_CONFIRMATION_DAYS, minimum=1),
        liquidate_to_fallback_during_risk_off=_safe_bool(params.get("liquidateToFallbackDuringRiskOff"), DEFAULT_LIQUIDATE_TO_FALLBACK_DURING_RISK_OFF),
        hold_fallback_when_empty=_safe_bool(params.get("holdFallbackWhenEmpty"), DEFAULT_HOLD_FALLBACK_WHEN_EMPTY),
        use_portfolio_circuit_breaker=_safe_bool(params.get("usePortfolioCircuitBreaker"), DEFAULT_USE_PORTFOLIO_CIRCUIT_BREAKER),
        maximum_portfolio_drawdown=_safe_float(params.get("maximumPortfolioDrawdown"), DEFAULT_MAXIMUM_PORTFOLIO_DRAWDOWN),
        circuit_breaker_cooldown_trading_days=_safe_int(params.get("circuitBreakerCooldownTradingDays"), DEFAULT_CIRCUIT_BREAKER_COOLDOWN_TRADING_DAYS, minimum=1),
    )


class DracoConfigError(RuntimeError):
    """Raised when a live Draco run cannot obtain a valid trading.draco config.

    This is intentionally loud: a bot_id the worker selected from
    `trading.draco WHERE is_active = true` must have a corresponding config
    row. If it doesn't, the live runner should fail the job run with a clear
    reason instead of silently falling back to an empty universe.
    """


def load_db_config(bot_id: str, *, strict: bool = False) -> DracoConfig:
    try:
        from db import get_conn
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM trading.draco WHERE bot_id = %s AND is_active = TRUE LIMIT 1",
                    (bot_id,),
                )
                row = cur.fetchone()
    except Exception as exc:
        if strict:
            raise DracoConfigError(f"draco: DB config lookup failed for bot_id={bot_id!r}: {exc}") from exc
        logger.warning("draco: could not load config for bot_id=%s, using defaults: %s", bot_id, exc)
        return DracoConfig(universe=[])

    if row:
        return draco_config_from_db_dict(dict(row))

    if strict:
        raise DracoConfigError(
            f"draco: no active trading.draco config row for bot_id={bot_id!r}; "
            "cannot run live signal without config."
        )
    logger.warning("draco: no active config row for bot_id=%s, using empty universe", bot_id)
    return DracoConfig(universe=[])
