"""tide_math.py - Pure scoring/selection/guardrail functions for the Tide
algorithm family.

No I/O; every function here is deterministic given plain price-history
inputs, so it is directly unit-testable without mocking price downloads or
the database. Live signal generation (bots/tide.py) and backfill
(backfill_tide.py) both call into this module so behavior is identical
across every execution path -- same separation-of-concerns pattern as
draco_math.py for Draco.

Ported from the already-validated Labs backtests, not reinvented:
  - composite_score / trailing_volatility / select_top_n_membership_stable:
    originally Monstra-Worker/tide_backtest.py
  - apply_guardrails (stop-loss + circuit breaker): originally
    Monstra-Worker/tide_adaptive_backtest.py
See /labs/tide for the full methodology write-up and backtest evidence
behind these specific formulas and defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (see /labs/tide for why these specific values)
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_TOP_N = 4
# Additive, in composite z-score units -- NOT Force's multiplicative margin
# (a fraction of raw return). A z-scored metric is centered at zero by
# construction, so a multiplicative margin would collapse toward no cushion
# for any middling-but-fine incumbent. See tide_backtest.py's module
# docstring for the full reasoning.
DEFAULT_INCUMBENCY_MARGIN = 0.30

# Composite ranking score: 50% trailing Sharpe + 30% trailing return + 20%
# low volatility, each z-scored across the eligible set before blending.
SHARPE_SCORE_WEIGHT = 0.5
RETURN_SCORE_WEIGHT = 0.3
LOW_VOL_SCORE_WEIGHT = 0.2
# Daily-return stdev floor for Sharpe's denominator -- a numerical guard
# against near-zero/degenerate volatility, not a realistic vol level.
VOL_FLOOR = 0.002

# Risk controls (see tide_bots_backtest.py for why these specific values,
# and apply_guardrails below for the exact mechanics).
DEFAULT_STOP_LOSS_PCT = 0.20
DEFAULT_CIRCUIT_BREAKER_PCT = 0.25
DEFAULT_CIRCUIT_BREAKER_COOLDOWN_DAYS = 21


def trailing_return(
    prices: pd.DataFrame,
    end_idx_exclusive: int,
    lookback_days: int,
    symbols: set[str] | None = None,
    *,
    require_full_window: bool = True,
) -> pd.Series:
    """Trailing (last/first - 1) return over the lookback window ending just
    before `end_idx_exclusive`, requiring every included symbol to have a
    non-null price on both the first and last day of the window (so a
    symbol with gappy history doesn't silently drop to a shorter window)."""
    start_idx = max(0, end_idx_exclusive - lookback_days)
    window = prices.iloc[start_idx:end_idx_exclusive]
    if len(window) < 2:
        return pd.Series(dtype=float)

    first_row = window.iloc[0]
    last_row = window.iloc[-1]
    if require_full_window:
        valid_mask = window.notna().all(axis=0)
    else:
        valid_mask = first_row.notna() & last_row.notna()
    first_row = first_row[valid_mask]
    last_row = last_row[valid_mask]

    trailing = (last_row / first_row) - 1.0
    trailing = trailing.replace([np.inf, -np.inf], np.nan).dropna()
    if symbols is not None:
        trailing = trailing[trailing.index.isin(symbols)]
    return trailing


def trailing_volatility(
    raw_returns: pd.DataFrame,
    end_idx_exclusive: int,
    lookback_days: int,
    symbols: set[str] | None = None,
) -> pd.Series:
    """Stdev of daily returns over the same trailing window `trailing_return`
    uses, requiring the full window so a symbol with gappy history doesn't
    get a fabricated low volatility from zero-filled gap days. `raw_returns`
    must have true NaNs for missing days (not filled with 0), unlike the
    return series used for applying realized P&L."""
    start_idx = max(0, end_idx_exclusive - lookback_days)
    window = raw_returns.iloc[start_idx:end_idx_exclusive]
    if len(window) < 2:
        return pd.Series(dtype=float)
    valid_mask = window.notna().all(axis=0)
    vol = window.loc[:, valid_mask].std(axis=0, ddof=1)
    if symbols:
        vol = vol[vol.index.isin(symbols)]
    return vol.dropna()


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def composite_score(
    trailing_ret: pd.Series,
    trailing_vol: pd.Series,
    *,
    sharpe_weight: float = SHARPE_SCORE_WEIGHT,
    return_weight: float = RETURN_SCORE_WEIGHT,
    low_vol_weight: float = LOW_VOL_SCORE_WEIGHT,
    vol_floor: float = VOL_FLOOR,
) -> pd.Series:
    """50% trailing Sharpe + 30% trailing return + 20% low volatility (by
    default), each z-scored across the shared symbols before blending so
    the three -- a percent return, a ratio, and a percent vol -- sit on a
    comparable scale. Sharpe's denominator is floored against near-zero
    volatility."""
    common = trailing_ret.index.intersection(trailing_vol.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    ret = trailing_ret[common]
    vol = trailing_vol[common].clip(lower=vol_floor)
    sharpe = ret / vol
    score = sharpe_weight * _zscore(sharpe) + return_weight * _zscore(ret) + low_vol_weight * _zscore(-vol)
    return score.dropna()


def select_top_n_membership_stable(
    scores: dict[str, float],
    top_n: int,
    current_holdings: list[str],
    margin: float,
) -> list[str]:
    """Membership-only incumbency defense: since Tide's top-n is equally
    weighted, position/order carries no weight, only whether a name is in
    the set at all. An incumbent's score gets an additive +margin cushion
    (z-score units); a challenger needs to clear an incumbent's cushioned
    score to take its seat. A single sort by one comparison value (not a
    chained pairwise rule) can't cycle the way an order-preserving
    comparator could with 3+ incumbents."""
    if not scores:
        return []
    current_set = set(current_holdings or [])

    def comparison_value(symbol: str) -> float:
        s = scores[symbol]
        return s + margin if symbol in current_set else s

    ordered = sorted(scores.keys(), key=lambda s: (-comparison_value(s), s))
    return ordered[:top_n]


def apply_guardrails(
    naive_weights: dict[str, float],
    prices: pd.DataFrame,
    i: int,
    peak_since_entry: dict[str, float],
    equity: float,
    portfolio_peak_equity: float,
    cash_equivalent: str | None,
    circuit_break_cooldown: int,
    *,
    stop_loss_pct: float | None,
    circuit_breaker_pct: float | None,
    circuit_breaker_cooldown_days: int = DEFAULT_CIRCUIT_BREAKER_COOLDOWN_DAYS,
) -> tuple[dict[str, float], int, float, list[str]]:
    """Per-position trailing stop-loss + portfolio drawdown circuit
    breaker, applied to a day's naive (ranking-only) selection. Both use
    only data known at the START of day i -- prices.iloc[i-1] (the most
    recent known close) and the equity level entering day i -- no
    look-ahead.

    Circuit breaker: if equity has drawn down more than `circuit_breaker_pct`
    from its running peak, force the whole portfolio to cash_equivalent for
    `circuit_breaker_cooldown_days` trading days, overriding the ranking
    entirely. This is a FIXED time-based cooldown, not "wait until equity
    recovers": an equity-recovery condition is self-defeating here -- once
    in cash, equity barely moves, so it can almost never climb back to
    within half the trigger threshold of the OLD (pre-crash) peak through
    cash drift alone (confirmed directly in an earlier backtest: 300+ of
    ~630 trading days stuck in the breaker for some bots). The peak resets
    to the equity level at the moment the cooldown ends, so re-entry starts
    measuring drawdown fresh rather than against a now-unreachable old high.

    Stop-loss: a held name is exited to cash if its own last known price is
    `stop_loss_pct` or more below its peak price since it was added to the
    portfolio -- independent of what the ranking still wants to hold. A
    stopped-out name's weight goes to cash, not to the next-ranked
    challenger (de-risks rather than immediately rotating), and its peak
    tracking resets, so re-entry later starts a fresh peak.

    Returns (adjusted weights, remaining cooldown days [0 = not tripped],
    new peak portfolio equity, symbols stopped out today)."""
    stopped_out: list[str] = []

    if circuit_breaker_pct is not None:
        if circuit_break_cooldown > 0:
            cash_weights = {cash_equivalent: 1.0} if cash_equivalent else {}
            return cash_weights, circuit_break_cooldown - 1, portfolio_peak_equity, stopped_out

        portfolio_peak_equity = max(portfolio_peak_equity, equity)
        drawdown = equity / portfolio_peak_equity - 1.0
        if drawdown <= -circuit_breaker_pct:
            cash_weights = {cash_equivalent: 1.0} if cash_equivalent else {}
            return cash_weights, circuit_breaker_cooldown_days, equity, stopped_out
    else:
        portfolio_peak_equity = max(portfolio_peak_equity, equity)

    if stop_loss_pct is None or i == 0:
        return naive_weights, 0, portfolio_peak_equity, stopped_out

    last_prices = prices.iloc[i - 1]
    surviving: dict[str, float] = {}
    removed_weight = 0.0
    for symbol, weight in naive_weights.items():
        if symbol == cash_equivalent:
            surviving[symbol] = surviving.get(symbol, 0.0) + weight
            continue
        price = last_prices.get(symbol)
        if price is None or pd.isna(price):
            surviving[symbol] = surviving.get(symbol, 0.0) + weight
            continue
        peak = max(peak_since_entry.get(symbol, float(price)), float(price))
        is_new_entry = symbol not in peak_since_entry
        peak_since_entry[symbol] = peak
        if not is_new_entry and price <= peak * (1.0 - stop_loss_pct):
            stopped_out.append(symbol)
            removed_weight += weight
            del peak_since_entry[symbol]
        else:
            surviving[symbol] = surviving.get(symbol, 0.0) + weight

    if removed_weight > 0 and cash_equivalent:
        surviving[cash_equivalent] = surviving.get(cash_equivalent, 0.0) + removed_weight

    for symbol in list(peak_since_entry.keys()):
        if symbol not in naive_weights:
            del peak_since_entry[symbol]

    return surviving, 0, portfolio_peak_equity, stopped_out
