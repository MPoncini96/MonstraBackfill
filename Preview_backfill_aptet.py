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

from bots.aptet import (
    MIN_OPTIMIZATION_SAMPLES,
    PARAMETER_REVIEW_DAYS,
    TOP_RETURN_THRESHOLD,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_HOLDINGS,
    AptetConfig,
    _adaptation_profile,
    _bounded_candidate_top_ns,
    _rolling_compounded_return,
    download_aptet_prices,
)
from backfill_aptet import compute_cost_drag, compute_turnover

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = date.today().isoformat()


@dataclass
class PreviewConfig:
    bot_name: str
    universe: list[str]
    fallback_ticker: str
    min_holdings: int
    max_holdings: int
    adaptation_speed: str
    risk_off_enabled: bool
    start_date: str
    end_date: str


def _clean_ticker(value: Any, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    ticker = str(value).strip().upper()
    return ticker or fallback


def _parse_int(value: Any, fallback: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    parsed = max(minimum, parsed)
    return min(parsed, maximum) if maximum is not None else parsed


def _normalize_universe(raw_universe: Any, fallback_ticker: str) -> list[str]:
    if not isinstance(raw_universe, list):
        return []
    seen: set[str] = set()
    universe: list[str] = []
    for item in raw_universe:
        ticker = _clean_ticker(item)
        if not ticker or ticker == fallback_ticker or ticker in seen:
            continue
        seen.add(ticker)
        universe.append(ticker)
    return universe


def _parse_date_string(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return fallback


def build_preview_config(payload: dict[str, Any]) -> PreviewConfig:
    fallback_ticker = _clean_ticker(payload.get("fallbackTicker"), "VOO") or "VOO"
    universe = _normalize_universe(payload.get("universe"), fallback_ticker)
    max_holdings = _parse_int(payload.get("maxHoldings"), 8, minimum=1, maximum=max(1, len(universe) or 1))
    min_holdings = _parse_int(payload.get("minHoldings"), 2, minimum=1, maximum=max_holdings)
    adaptation_speed = str(payload.get("adaptationSpeed") or "balanced").strip().lower() or "balanced"
    if adaptation_speed not in {"conservative", "balanced", "aggressive"}:
        adaptation_speed = "balanced"
    start_date = _parse_date_string(payload.get("startDate"), DEFAULT_START_DATE)
    end_date = _parse_date_string(payload.get("endDate"), DEFAULT_END_DATE)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return PreviewConfig(
        bot_name=str(payload.get("botName") or "").strip(),
        universe=universe,
        fallback_ticker=fallback_ticker,
        min_holdings=min_holdings,
        max_holdings=max_holdings,
        adaptation_speed=adaptation_speed,
        risk_off_enabled=bool(True if payload.get("riskOffEnabled") is None else payload.get("riskOffEnabled")),
        start_date=start_date,
        end_date=end_date,
    )


# ─── Vectorized pre-computation ───────────────────────────────────────────────

def _build_trailing_matrices(prices_np: np.ndarray, lookbacks: list[int]) -> dict[int, np.ndarray]:
    """
    Pre-compute trailing return matrices for all lookbacks.

    trailing_matrices[L][E, j] matches _get_trailing_returns(prices, E, L)[col_j]:
      = prices_np[E-1, j] / prices_np[E-L, j] - 1.0
        when all prices_np[E-L : E, j] are finite AND prices_np[E-L, j] > 0,
        otherwise NaN.

    Shape of each matrix: (n+1, ncols).  Row index = end_idx_exclusive (E).
    Rows 0..L-1 are always NaN (insufficient history).
    """
    n, ncols = prices_np.shape

    # Prefix-sum of finite flags for O(1) window-validity checks.
    # cum_finite[t] = number of finite values in prices_np[0:t, :] per column.
    finite_mask = np.isfinite(prices_np).astype(np.int32)  # (n, ncols)
    cum_finite = np.zeros((n + 1, ncols), dtype=np.int32)
    cum_finite[1:] = np.cumsum(finite_mask, axis=0)

    matrices: dict[int, np.ndarray] = {}
    for L in lookbacks:
        mat = np.full((n + 1, ncols), np.nan, dtype=np.float64)

        # E ranges from L to n (so the window prices_np[E-L : E] has exactly L rows)
        E_arr = np.arange(L, n + 1)        # shape (m,), m = n - L + 1
        first_row = E_arr - L              # index of window-start row in prices_np
        last_row = E_arr - 1              # index of window-end   row in prices_np

        # Count finite values in each window using prefix sums.
        window_finite = cum_finite[E_arr] - cum_finite[first_row]  # (m, ncols)
        full_window_ok = window_finite == L                        # all L rows finite

        first_prices = prices_np[first_row]   # (m, ncols)
        last_prices  = prices_np[last_row]    # (m, ncols)

        valid = full_window_ok & (first_prices > 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            trailing = np.where(valid, last_prices / first_prices - 1.0, np.nan)

        mat[E_arr] = trailing
        matrices[L] = mat

    return matrices


# ─── Fast optimization inner loop ────────────────────────────────────────────

def _fast_simulate_combo(
    tr_mat: np.ndarray,
    ret_sim: np.ndarray,
    universe_arr: np.ndarray,
    fallback_col: int,
    risk_off_enabled: bool,
    top_n: int,
    loop_start: int,
    loop_end: int,
) -> list[float]:
    """
    Vectorized replacement for the inner loop of _simulate_param_combo.

    tr_mat    : trailing-return matrix for this lookback, shape (n+1, ncols).
    ret_sim   : day-over-day price returns, shape (n-1, ncols).
                ret_sim[t, j] = price[t+1,j]/price[t,j] - 1  (NaN where invalid).
    loop_start, loop_end : half-open range matching original range(start_idx, end_idx).

    Return value semantics match the original:
      - risk-on day with all valid prices  → append equal-weight return
      - risk-off day (fallback)            → append fallback return if valid, else skip
      - any invalid price in selected      → skip day (matches `valid = False` path)
    """
    daily_returns: list[float] = []
    n_universe = len(universe_arr)
    effective_top_n = min(top_n, n_universe)

    for idx in range(loop_start, loop_end):
        uni_tr = tr_mat[idx, universe_arr]          # (n_universe,)
        valid_mask = np.isfinite(uni_tr)

        if risk_off_enabled:
            n_valid = int(valid_mask.sum())
            if n_valid == 0 or float(uni_tr[valid_mask].max()) <= TOP_RETURN_THRESHOLD:
                # Risk-off: use fallback ticker
                fb_ret = ret_sim[idx, fallback_col]
                if np.isfinite(fb_ret):
                    daily_returns.append(float(fb_ret))
                # else: skip (invalid fallback price — matches original `valid = False`)
                continue

        # Risk-on: select top_n from valid universe tickers
        if not valid_mask.any():
            # USE_FALLBACK_TICKER = True
            fb_ret = ret_sim[idx, fallback_col]
            if np.isfinite(fb_ret):
                daily_returns.append(float(fb_ret))
            continue

        valid_cols = universe_arr[valid_mask]
        valid_tr   = uni_tr[valid_mask]
        n_take = min(effective_top_n, len(valid_cols))

        # argpartition is O(n) vs argsort O(n log n); for small universes argsort is fine
        sort_order = np.argsort(valid_tr)[::-1][:n_take]
        selected_cols = valid_cols[sort_order]

        # Equal-weight period return (day idx → idx+1)
        period_rets = ret_sim[idx, selected_cols]
        if not np.all(np.isfinite(period_rets)):
            # Any invalid price → skip day (original: `valid = False; break`)
            continue

        daily_returns.append(float(np.mean(period_rets)))

    return daily_returns


def _fast_optimize_params(
    trailing_matrices: dict[int, np.ndarray],
    ret_sim: np.ndarray,
    universe_arr: np.ndarray,
    fallback_col: int,
    config: AptetConfig,
    end_idx_exclusive: int,
    previous_selected_top_n: int | None,
) -> dict[str, Any] | None:
    """
    Vectorized replacement for optimize_aptet_params.
    Uses pre-computed trailing matrices instead of calling _get_trailing_returns.
    """
    profile = _adaptation_profile(config)
    top_ns = _bounded_candidate_top_ns(config, previous_selected_top_n)
    train_days = profile.optimization_train_days
    n_view = end_idx_exclusive   # view = prices.iloc[:end_idx_exclusive]

    best: dict[str, Any] | None = None
    best_score = -np.inf

    for L in profile.candidate_lookbacks:
        if L not in trailing_matrices:
            continue
        tr_mat = trailing_matrices[L]
        # Matches original: start_idx = max(lookback+1, len(view)-train_days), end_idx = len(view)-1
        loop_start = max(L + 1, n_view - train_days)
        loop_end   = n_view - 1    # range(loop_start, loop_end) == original range(start_idx, end_idx)

        if loop_start >= loop_end:
            continue

        for top_n in top_ns:
            daily_rets = _fast_simulate_combo(
                tr_mat, ret_sim, universe_arr, fallback_col,
                config.risk_off_enabled, top_n, loop_start, loop_end,
            )

            if len(daily_rets) < MIN_OPTIMIZATION_SAMPLES:
                continue

            rets_arr    = np.asarray(daily_rets, dtype=np.float64)
            equity      = np.cumprod(1.0 + rets_arr)
            total_ret   = float(equity[-1] - 1.0)
            running_max = np.maximum.accumulate(equity)
            max_dd      = float((equity / running_max - 1.0).min())
            vol         = float(rets_arr.std())
            sharpe_like = float((rets_arr.mean() / vol) * np.sqrt(252)) if vol > 0 else 0.0
            score       = total_ret + sharpe_like * 0.05 + max_dd * 0.50

            if score > best_score:
                best_score = score
                best = {
                    "selectedLookbackDays": int(L),
                    "selectedTopN":         int(top_n),
                    "score":                score,
                    "totalReturn":          total_ret,
                    "maxDrawdown":          max_dd,
                    "sharpeLike":           sharpe_like,
                }

    return best


# ─── Optimized run_preview ────────────────────────────────────────────────────

def run_preview(preview: PreviewConfig) -> dict[str, Any]:
    if not preview.universe:
        raise ValueError("At least one universe ticker is required.")

    config = AptetConfig(
        universe=preview.universe,
        fallback_ticker=preview.fallback_ticker,
        benchmark_ticker=preview.fallback_ticker,
        min_holdings=preview.min_holdings,
        max_holdings=preview.max_holdings,
        adaptation_speed=preview.adaptation_speed,
        risk_off_enabled=preview.risk_off_enabled,
    )

    prices = download_aptet_prices(config, preview.start_date, preview.end_date)
    if prices.empty or len(prices.index) < 2:
        raise ValueError("Not enough market data was returned for this preview.")

    n          = len(prices.index)
    col_names  = list(prices.columns)
    col_index  = {col: i for i, col in enumerate(col_names)}

    # Column index arrays for universe and fallback ticker
    universe_arr = np.array(
        [col_index[s] for s in config.universe if s in col_index], dtype=np.int32
    )
    fallback_col = col_index.get(config.fallback_ticker, 0)

    prices_np = prices.to_numpy(dtype=np.float64, copy=False)   # (n, ncols)

    # ret_sim[t, j] = prices_np[t+1,j] / prices_np[t,j] - 1  (NaN where invalid)
    # Used by _fast_simulate_combo (skips invalid days, matching original behaviour).
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_sim = np.where(
            prices_np[:-1] > 0,
            prices_np[1:] / prices_np[:-1] - 1.0,
            np.nan,
        )  # shape (n-1, ncols)

    # ret_outer[t, j] = same but NaN → 0.0, matching pct_change().fillna(0.0)
    # Used in the outer equity loop (matches compute_day_return_and_holdings).
    ret_outer = np.where(np.isfinite(ret_sim), ret_sim, 0.0)

    # Pre-compute trailing-return matrices for all candidate lookbacks.
    # Also include DEFAULT_LOOKBACK_DAYS in case prior_lookback is None on first day.
    profile   = _adaptation_profile(config)
    all_lookbacks = sorted(set(profile.candidate_lookbacks) | {DEFAULT_LOOKBACK_DAYS})
    trailing_matrices = _build_trailing_matrices(prices_np, all_lookbacks)

    # ── Adaptation state (inlined for speed) ──────────────────────────────────
    prior_lookback: int | None = None
    prior_top_n:    int | None = None
    realized_since_change: list[float] = []   # last ≤ PARAMETER_REVIEW_DAYS daily returns

    # ── Main loop ─────────────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    risk_off_days = 0
    equity        = 1.0
    prev_holdings: dict[str, float] = {}

    for index in range(1, n):
        trading_day = prices.index[index].date()

        # -- Decide whether to re-optimize -----------------------------------
        n_recent = len(realized_since_change)
        trailing_review = (
            _rolling_compounded_return(realized_since_change)
            if n_recent >= PARAMETER_REVIEW_DAYS else None
        )
        should_search = prior_top_n is None or prior_lookback is None
        search_reason = "initial_selection" if should_search else "hold_current_parameters"
        if not should_search and trailing_review is not None and trailing_review < 0.0:
            should_search = True
            search_reason = "negative_last_5d_since_change"

        # -- Run fast optimization if needed ---------------------------------
        best: dict[str, Any] | None = None
        opt_date_str: str | None = None
        if should_search:
            best = _fast_optimize_params(
                trailing_matrices, ret_sim, universe_arr, fallback_col,
                config, index, prior_top_n,
            )
            if best is not None:
                # lastOptimizationDate = last row of the view (= prices.index[index-1])
                opt_date_str = str(prices.index[index - 1].date())

        # -- Select parameters -----------------------------------------------
        if best is not None:
            selected_lookback = int(best["selectedLookbackDays"])
            selected_top_n    = int(best["selectedTopN"])
        else:
            selected_lookback = prior_lookback if prior_lookback is not None else DEFAULT_LOOKBACK_DAYS
            selected_top_n    = (
                prior_top_n if prior_top_n is not None
                else min(DEFAULT_MIN_HOLDINGS, max(1, len(config.universe)))
            )
        parameter_changed = prior_lookback != selected_lookback or prior_top_n != selected_top_n

        # -- Daily holdings decision using pre-computed trailing matrix ------
        ranked_trailing_returns: dict[str, float] = {}

        if index < selected_lookback + 1 or selected_lookback not in trailing_matrices:
            # Insufficient history
            selected_symbols = [config.fallback_ticker]
            weights          = np.array([1.0])
            risk_off         = True
            risk_reason      = "no_history"
        else:
            tr_row   = trailing_matrices[selected_lookback][index, :]  # (ncols,)
            uni_tr   = tr_row[universe_arr]                             # (n_universe,)
            valid_mask = np.isfinite(uni_tr)

            # Evaluate risk-off (mirrors _evaluate_risk_off; BENCHMARK_FILTER_ENABLED=False)
            risk_off   = False
            risk_reason = "risk_on"

            if config.risk_off_enabled:
                if not valid_mask.any():
                    risk_off    = True
                    risk_reason = "no_ranked_candidates"
                else:
                    top_ret = float(uni_tr[valid_mask].max())
                    if top_ret <= TOP_RETURN_THRESHOLD:
                        risk_off    = True
                        risk_reason = f"top_non_positive:{top_ret:.6f}"

            if risk_off:
                selected_symbols = [config.fallback_ticker]
                weights          = np.array([1.0])
            else:
                if not valid_mask.any():
                    # USE_FALLBACK_TICKER = True
                    selected_symbols = [config.fallback_ticker]
                    weights          = np.array([1.0])
                    risk_off         = True
                    risk_reason      = "no_selected_symbols"
                else:
                    valid_cols = universe_arr[valid_mask]
                    valid_tr   = uni_tr[valid_mask]
                    n_take     = min(selected_top_n, len(valid_cols))
                    sort_order = np.argsort(valid_tr)[::-1][:n_take]
                    sel_cols   = valid_cols[sort_order]

                    selected_symbols = [col_names[c] for c in sel_cols]
                    n_sel            = len(selected_symbols)
                    weights          = np.ones(n_sel) / n_sel
                    ranked_trailing_returns = {
                        col_names[sel_cols[i]]: float(valid_tr[sort_order[i]])
                        for i in range(n_sel)
                    }

        if risk_off:
            risk_off_days += 1

        # -- Compute equity return for this day ------------------------------
        # Matches compute_day_return_and_holdings(returns.loc[trading_ts], ...)
        # where returns = prices.pct_change().fillna(0.0)
        day_ret_row = ret_outer[index - 1]    # (ncols,) returns from day index-1 → index

        valid_pairs: list[tuple[str, float]] = [
            (sym, float(w))
            for sym, w in zip(selected_symbols, weights)
            if sym in col_index
        ]

        gross_ret = 0.0
        current_holdings: dict[str, float] = {}
        if valid_pairs:
            total_w = sum(w for _, w in valid_pairs)
            if total_w > 0:
                norm = [(sym, w / total_w) for sym, w in valid_pairs]
                gross_ret        = sum(day_ret_row[col_index[sym]] * w for sym, w in norm)
                current_holdings = {sym: w for sym, w in norm}

        turnover  = compute_turnover(prev_holdings, current_holdings)
        cost_drag = compute_cost_drag(turnover)
        net_ret   = gross_ret - cost_drag
        equity   *= 1.0 + net_ret

        # -- Build metadata --------------------------------------------------
        top_ns_list = _bounded_candidate_top_ns(config, prior_top_n)
        metadata: dict[str, Any] = {
            "selectedLookbackDays":           selected_lookback,
            "selectedTopN":                   selected_top_n,
            "candidateLookbacks":             list(profile.candidate_lookbacks),
            "candidateTopNs":                 list(top_ns_list),
            "adaptationSpeed":                config.adaptation_speed,
            "optimizationTrainDays":          int(profile.optimization_train_days),
            "optimizationCheckDays":          int(profile.optimization_check_days),
            "lastOptimizationDate":           opt_date_str,
            "riskOffReason":                  risk_reason if risk_off else None,
            "fallbackTicker":                 config.fallback_ticker,
            "rankedTrailingReturns":          ranked_trailing_returns,
            "optimizationScore":              best.get("score") if best else None,
            "usedDefaultParameters":          best is None and prior_top_n is None,
            "previousSelectedTopN":           prior_top_n,
            "previousSelectedLookbackDays":   prior_lookback,
            "parameterSearchTriggered":       bool(should_search),
            "parameterSearchReason":          search_reason,
            "parameterChanged":               bool(parameter_changed),
            "parameterReviewDays":            PARAMETER_REVIEW_DAYS,
            "daysSinceParameterChange":       n_recent,
            "trailingReturnSinceLastChange5D": trailing_review,
        }

        rows.append({
            "d":        trading_day.isoformat(),
            "equity":   float(equity),
            "ret":      float(net_ret),
            "holdings": current_holdings,
            "meta": {
                "risk_off":  bool(risk_off),
                "risk_reason": risk_reason,
                "turnover":  float(turnover),
                "cost_drag": float(cost_drag),
                **metadata,
            },
        })

        prev_holdings = current_holdings.copy()

        # -- Advance adaptation state ----------------------------------------
        param_changed_state = prior_lookback != selected_lookback or prior_top_n != selected_top_n
        if param_changed_state:
            realized_since_change = []
        realized_since_change.append(float(net_ret))
        realized_since_change = realized_since_change[-PARAMETER_REVIEW_DAYS:]

        prior_lookback = selected_lookback
        prior_top_n    = selected_top_n

    # ── Assemble result ───────────────────────────────────────────────────────
    final_equity      = rows[-1]["equity"] if rows else 1.0
    actual_start_date = rows[0]["d"] if rows else preview.start_date
    actual_end_date   = rows[-1]["d"] if rows else preview.end_date

    return {
        "success":      True,
        "startDate":    actual_start_date,
        "endDate":      actual_end_date,
        "initialEquity": 1.0,
        "equity":       rows,
        "summary": {
            "botName":        preview.bot_name,
            "processedDays":  len(rows),
            "riskOffDays":    risk_off_days,
            "finalEquity":    final_equity,
            "finalReturnPct": (final_equity - 1.0) * 100.0,
        },
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("Preview payload must be a JSON object.")
        preview = build_preview_config(payload)
        result  = run_preview(preview)
        json.dump(result, sys.stdout)
        return 0
    except Exception as exc:
        json.dump({"success": False, "error": str(exc)}, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
