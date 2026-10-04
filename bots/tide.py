"""bots/tide.py (MonstraBackfill)

Tide config: a long-hold Force variant. Ranks candidates on a composite
Sharpe/return/low-volatility score (not Force's raw trailing return),
holds an equal-weighted top-N (not Force's rank-weighted tiers), and
trades only when the resulting holdings set actually changes. Adds a
per-position trailing stop-loss and a portfolio drawdown circuit breaker
on top of Force's own kill switch.

This is the backfill-side mirror of Monstra-Worker/bots/tide.py, trimmed to
just the DB-free pieces backfill_tide.py needs (TideConfig + DB-row
parsing) -- no AlgorithmContext dependency, no live run_tide, matching how
this repo already keeps bots/draco.py's pure candidate logic separate from
its own live-runner DB plumbing.

All scoring/selection/guardrail math lives in tide_math.py (pure functions,
shared with backfill_tide.py so behavior is identical everywhere) -- ported
directly from the already-validated Labs backtests. See /labs/tide for the
full methodology write-up and backtest evidence.

Hidden/Phase-1 (see algorithm_registry.py's "tide" entry): standalone,
not wired into any dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bots.alpha1 import (
    DEFAULT_CASH_EQUIVALENT,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_UNIVERSE,
    Alpha1Config,
)
import tide_math as tm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TideConfig:
    universe: list[str]
    cash_equivalent: str | None = DEFAULT_CASH_EQUIVALENT
    top_n: int = tm.DEFAULT_TOP_N
    lookback_days: int = tm.DEFAULT_LOOKBACK_DAYS
    incumbency_margin: float = tm.DEFAULT_INCUMBENCY_MARGIN
    stop_loss_pct: float = tm.DEFAULT_STOP_LOSS_PCT
    circuit_breaker_pct: float = tm.DEFAULT_CIRCUIT_BREAKER_PCT
    circuit_breaker_cooldown_days: int = tm.DEFAULT_CIRCUIT_BREAKER_COOLDOWN_DAYS
    benchmark: str | None = None
    enable_kill_switch: bool = True
    enable_benchmark_filter: bool = False
    kill_switch_mode: str = "top_negative"
    benchmark_mode: str = "benchmark_positive"
    top_return_threshold: float = 0.0
    benchmark_return_threshold: float = 0.0
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS

    def as_alpha1_config(self) -> Alpha1Config:
        """An Alpha1Config view for reusing bots/alpha1.py's kill switch
        unmodified -- Tide's kill switch is Force's own, evaluated on raw
        trailing return, not the composite score (see tide_math.py's module
        docstring for why a risk-adjusted score doesn't answer "is anything
        working at all")."""
        top_n = max(1, int(self.top_n))
        return Alpha1Config(
            universe=list(self.universe),
            cash_equivalent=self.cash_equivalent,
            top_n=top_n,
            lookback_days=self.lookback_days,
            benchmark=self.benchmark,
            enable_kill_switch=self.enable_kill_switch,
            enable_benchmark_filter=self.enable_benchmark_filter,
            kill_switch_mode=self.kill_switch_mode,
            benchmark_mode=self.benchmark_mode,
            top_return_threshold=self.top_return_threshold,
            benchmark_return_threshold=self.benchmark_return_threshold,
            transaction_cost_bps=self.transaction_cost_bps,
            slippage_bps=self.slippage_bps,
            incumbency_margin=self.incumbency_margin,  # unused by the kill switch itself
            rank_weights=np.full(top_n, 1.0 / top_n, dtype=float),
        )


class TideConfigError(RuntimeError):
    """Raised when a Tide run cannot obtain a valid trading.tide config.

    Loud on purpose: a bot_id selected from `trading.tide WHERE is_active =
    true` must have a corresponding config row. If it doesn't, the caller
    should fail with a clear reason instead of silently falling back to an
    empty universe."""


def _normalize_universe(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    seen: set[str] = set()
    universe: list[str] = []
    for item in raw:
        if item is None:
            continue
        ticker = str(item).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        universe.append(ticker)
    return universe


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _config_from_db_row(row: dict[str, Any]) -> TideConfig:
    return TideConfig(
        universe=_normalize_universe(row.get("universe")) or list(DEFAULT_UNIVERSE),
        cash_equivalent=(str(row["cash_equivalent"]).strip().upper() if row.get("cash_equivalent") else DEFAULT_CASH_EQUIVALENT),
        top_n=_safe_int(row.get("top_n"), tm.DEFAULT_TOP_N),
        lookback_days=_safe_int(row.get("lookback_days"), tm.DEFAULT_LOOKBACK_DAYS),
        incumbency_margin=_safe_float(row.get("incumbency_margin"), tm.DEFAULT_INCUMBENCY_MARGIN),
        stop_loss_pct=_safe_float(row.get("stop_loss_pct"), tm.DEFAULT_STOP_LOSS_PCT),
        circuit_breaker_pct=_safe_float(row.get("circuit_breaker_pct"), tm.DEFAULT_CIRCUIT_BREAKER_PCT),
        benchmark=(str(row["benchmark"]).strip().upper() if row.get("benchmark") else None),
        transaction_cost_bps=_safe_float(row.get("transaction_cost_bps"), DEFAULT_TRANSACTION_COST_BPS),
        slippage_bps=_safe_float(row.get("slippage_bps"), DEFAULT_SLIPPAGE_BPS),
    )
