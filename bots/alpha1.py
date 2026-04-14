# bots/alpha1.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from bot_identity import BOT_TYPE_ALPHA1

# Default fallback configuration (used if database is unavailable)
DEFAULT_UNIVERSE = [
    "VOO", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "COST", "JPM", "UNH", "HD",
    "AMD", "AVGO",
]

DEFAULT_CASH_EQUIVALENT = "VOO"
DEFAULT_TOP_N = 4
DEFAULT_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)

REQUIRE_FULL_LOOKBACK_WINDOW = True
USE_CASH_EQUIVALENT_FALLBACK = True

DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass
class Alpha1Config:
    universe: list[str]
    cash_equivalent: str | None
    top_n: int | None
    lookback_days: int
    benchmark: str | None = None
    enable_kill_switch: bool = True
    enable_benchmark_filter: bool = False
    kill_switch_mode: str = "top_negative"
    benchmark_mode: str = "benchmark_positive"
    top_return_threshold: float = 0.0
    benchmark_return_threshold: float = 0.0
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    rank_weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())


def _default_top_weights(n: int = 4, base=(0.40, 0.30, 0.20, 0.10)) -> np.ndarray:
    if n == 4:
        return np.array(base, dtype=float)
    w = np.linspace(n, 1, n, dtype=float)
    w /= w.sum()
    return w


def _compute_growth(prices: pd.DataFrame, lookback_days: int = 14, bars_per_day: int = 24) -> pd.DataFrame:
    lb = int(lookback_days * bars_per_day)
    return prices / prices.shift(lb) - 1.0


def _top_n_by_growth(growth_row: pd.Series, n: int = 4) -> list[str]:
    g = growth_row.dropna().sort_values(ascending=False)
    return list(g.index[:n])


def _normalize_universe(raw_universe: Any) -> list[str]:
    if raw_universe is None:
        return []
    if not isinstance(raw_universe, (list, tuple)):
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


def _clean_ticker(value: Any) -> str | None:
    if value is None:
        return None
    ticker = str(value).strip().upper()
    return ticker or None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _normalize_rank_weights_array(raw: Any) -> np.ndarray:
    """Coerce DB/user rank weights to a positive simplex; fallback to DEFAULT_WEIGHTS."""
    if raw is None:
        return DEFAULT_WEIGHTS.copy()
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw)
        except Exception:
            return DEFAULT_WEIGHTS.copy()
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        return DEFAULT_WEIGHTS.copy()
    try:
        arr = np.array([float(x) for x in raw], dtype=float)
    except (TypeError, ValueError):
        return DEFAULT_WEIGHTS.copy()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return DEFAULT_WEIGHTS.copy()
    s = float(arr.sum())
    if s > 1.5:
        arr = arr / 100.0
        s = float(arr.sum())
    if s <= 0:
        return DEFAULT_WEIGHTS.copy()
    return arr / s


def _clamp_top_n(top_n: int | None, rank_weights: np.ndarray) -> int:
    cap = int(len(rank_weights)) if len(rank_weights) > 0 else DEFAULT_TOP_N
    cap = max(1, cap)
    if top_n is None:
        return min(DEFAULT_TOP_N, cap)
    return max(1, min(int(top_n), cap))


def _parse_period_days(history_period: str | None) -> int:
    if not history_period:
        return 0

    token = history_period.strip().lower()
    if len(token) < 2:
        return 0

    try:
        qty = int(token[:-1])
    except ValueError:
        return 0

    unit = token[-1]
    multipliers = {
        "d": 1,
        "w": 7,
        "m": 30,
        "y": 365,
    }
    return qty * multipliers.get(unit, 0)


def _build_live_config(bot_id: str, use_db_config: bool = True) -> tuple[Alpha1Config, str | None]:
    config_data: dict[str, Any] | None = None
    history_period: str | None = None

    if use_db_config:
        try:
            from db import get_bot_config

            config_data = get_bot_config(bot_id)
        except Exception as exc:
            print(f"Warning: Could not load config for {bot_id} from database, using defaults: {exc}")

    if config_data:
        history_period = config_data.get("history_period")
        rw_raw = config_data.get("rank_weights")
        if rw_raw is None:
            rw_raw = config_data.get("weights")
        rank_w = _normalize_rank_weights_array(rw_raw)
        config = Alpha1Config(
            universe=_normalize_universe(config_data.get("universe")) or list(DEFAULT_UNIVERSE),
            cash_equivalent=_clean_ticker(config_data.get("cash_equivalent", DEFAULT_CASH_EQUIVALENT)),
            top_n=config_data.get("top_n", DEFAULT_TOP_N),
            lookback_days=int(config_data.get("lookback_days", 14)),
            benchmark=_clean_ticker(config_data.get("benchmark")),
            enable_kill_switch=_safe_bool(config_data.get("enable_kill_switch"), True),
            enable_benchmark_filter=_safe_bool(config_data.get("enable_benchmark_filter"), False),
            kill_switch_mode=str(config_data.get("kill_switch_mode", "top_negative")),
            benchmark_mode=str(config_data.get("benchmark_mode", "benchmark_positive")),
            top_return_threshold=_safe_float(config_data.get("top_return_threshold"), 0.0),
            benchmark_return_threshold=_safe_float(config_data.get("benchmark_return_threshold"), 0.0),
            transaction_cost_bps=_safe_float(config_data.get("transaction_cost_bps"), DEFAULT_TRANSACTION_COST_BPS),
            slippage_bps=_safe_float(config_data.get("slippage_bps"), DEFAULT_SLIPPAGE_BPS),
            rank_weights=rank_w,
        )
    else:
        config = Alpha1Config(
            universe=list(DEFAULT_UNIVERSE),
            cash_equivalent=DEFAULT_CASH_EQUIVALENT,
            top_n=DEFAULT_TOP_N,
            lookback_days=14,
            rank_weights=DEFAULT_WEIGHTS.copy(),
        )

    config.top_n = _clamp_top_n(config.top_n, config.rank_weights)
    if config.lookback_days < 1:
        config.lookback_days = 14

    return config, history_period


def _resolve_download_window(lookback_days: int, history_period: str | None) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    configured_days = _parse_period_days(history_period)
    buffer_days = max(lookback_days * 6, 180, configured_days)
    start_date = today - timedelta(days=buffer_days)
    end_date = today + timedelta(days=1)
    return start_date.isoformat(), end_date.isoformat()


def _download_adjusted_close(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data missing 'Close'.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data missing 'Close'.")
        prices = raw[["Close"]].rename(columns={"Close": symbols[0]})

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")
    prices.columns = [str(col).upper() for col in prices.columns]
    return prices


def _build_price_frame(config: Alpha1Config, start_date: str, end_date: str) -> pd.DataFrame:
    symbols = list(config.universe)
    for sym in (config.cash_equivalent, config.benchmark):
        if sym and sym not in symbols:
            symbols.append(sym)
    return _download_adjusted_close(symbols, start_date, end_date)


def _get_trailing_returns(
    prices: pd.DataFrame,
    end_idx_exclusive: int,
    lookback_days: int,
    symbols: set[str] | None = None,
) -> pd.Series:
    start_idx = max(0, end_idx_exclusive - lookback_days)
    lookback = prices.iloc[start_idx:end_idx_exclusive]

    if len(lookback) < 2:
        return pd.Series(dtype=float)

    first_row = lookback.iloc[0]
    last_row = lookback.iloc[-1]

    if REQUIRE_FULL_LOOKBACK_WINDOW:
        valid_mask = lookback.notna().all(axis=0)
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]
    else:
        valid_mask = first_row.notna() & last_row.notna()
        first_row = first_row[valid_mask]
        last_row = last_row[valid_mask]

    trailing = (last_row / first_row) - 1.0
    trailing = trailing.replace([np.inf, -np.inf], np.nan).dropna()

    if symbols is not None:
        trailing = trailing[trailing.index.isin(symbols)]

    return trailing


def _evaluate_kill_switch(
    config: Alpha1Config,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if config.enable_kill_switch and config.kill_switch_mode != "off":
        if ranked_trailing.empty:
            reasons.append("kill_switch:no_ranked_candidates")
        elif config.kill_switch_mode == "top_negative":
            top_ret = float(ranked_trailing.max())
            if top_ret <= config.top_return_threshold:
                reasons.append(
                    f"kill_switch:top_ret={top_ret:.6f}<=threshold={config.top_return_threshold:.6f}"
                )
        elif config.kill_switch_mode == "avg_negative":
            avg_ret = float(ranked_trailing.mean())
            if avg_ret <= config.top_return_threshold:
                reasons.append(
                    f"kill_switch:avg_ret={avg_ret:.6f}<=threshold={config.top_return_threshold:.6f}"
                )

    if config.enable_benchmark_filter and config.benchmark_mode != "off" and config.benchmark:
        bench_ret = benchmark_trailing.get(config.benchmark, np.nan)

        if pd.isna(bench_ret):
            reasons.append("benchmark_filter:no_benchmark_data")
        else:
            bench_ret = float(bench_ret)

            if config.benchmark_mode == "benchmark_positive":
                if bench_ret <= config.benchmark_return_threshold:
                    reasons.append(
                        f"benchmark_filter:bench_ret={bench_ret:.6f}<=threshold={config.benchmark_return_threshold:.6f}"
                    )
            elif config.benchmark_mode == "benchmark_gt_cash":
                cash_ret = np.nan
                if config.cash_equivalent:
                    cash_ret = cash_trailing.get(config.cash_equivalent, np.nan)

                if pd.isna(cash_ret):
                    reasons.append("benchmark_filter:no_cash_data")
                else:
                    cash_ret = float(cash_ret)
                    if bench_ret <= cash_ret:
                        reasons.append(
                            f"benchmark_filter:bench_ret={bench_ret:.6f}<=cash_ret={cash_ret:.6f}"
                        )

    if reasons:
        return True, "; ".join(reasons)

    return False, "risk_on"


def _choose_risk_on_holdings(
    ranked_trailing: pd.Series, top_n: int, rank_weights: np.ndarray
) -> tuple[list[str], np.ndarray]:
    if ranked_trailing.empty:
        return [], np.array([], dtype=float)

    ranked = ranked_trailing.sort_values(ascending=False)
    selected = ranked.head(top_n).index.tolist()
    rw = np.asarray(rank_weights, dtype=float)
    weights = rw[: len(selected)].copy()

    if len(weights) == 0:
        return [], np.array([], dtype=float)

    weights = weights / weights.sum()
    return selected, weights


def _choose_holdings_for_day(
    config: Alpha1Config,
    ranked_trailing: pd.Series,
    benchmark_trailing: pd.Series,
    cash_trailing: pd.Series,
) -> tuple[list[str], np.ndarray, bool, str]:
    risk_off, reason = _evaluate_kill_switch(
        config=config,
        ranked_trailing=ranked_trailing,
        benchmark_trailing=benchmark_trailing,
        cash_trailing=cash_trailing,
    )

    if risk_off:
        if USE_CASH_EQUIVALENT_FALLBACK and config.cash_equivalent:
            return [config.cash_equivalent], np.array([1.0], dtype=float), True, reason
        return [], np.array([], dtype=float), True, reason

    selected, weights = _choose_risk_on_holdings(
        ranked_trailing, int(config.top_n or DEFAULT_TOP_N), config.rank_weights
    )

    if not selected and USE_CASH_EQUIVALENT_FALLBACK and config.cash_equivalent:
        return [config.cash_equivalent], np.array([1.0], dtype=float), True, "fallback:no_selected_symbols"

    return selected, weights, False, reason


def run_alpha1(
    bot_id: str,
    lookback_days: int | None = None,
    top_n: int | None = None,
    history_period: str | None = None,
    interval: str | None = None,
    use_db_config: bool = True,
) -> dict:
    """
    Live Alpha1 signal generator using the same daily strategy rules as backfill_alpha1.

    Args:
        bot_id: one of {bellator, imperium, medicus, vectura, vis}
    """
    ts = datetime.now(timezone.utc)
    config, history_period_config = _build_live_config(bot_id, use_db_config=use_db_config)

    if lookback_days is not None:
        config.lookback_days = max(1, int(lookback_days))
    if top_n is not None:
        config.top_n = _clamp_top_n(top_n, config.rank_weights)

    history_period = history_period if history_period is not None else history_period_config
    start_date, end_date = _resolve_download_window(config.lookback_days, history_period)
    prices = _build_price_frame(config, start_date, end_date)
    interval = "1d"

    if prices.empty or len(prices.index) < 2:
        fallback = config.cash_equivalent or DEFAULT_CASH_EQUIVALENT
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_ALPHA1,
            "ts": ts,
            "signal": "HOLD",
            "note": f"Not enough {interval} history for lookback={config.lookback_days}d; holding {fallback}",
            "payload": {
                "asof": str(prices.index[-1]) if not prices.empty else None,
                "interval": interval,
                "lookback_days": config.lookback_days,
                "top_n": config.top_n,
                "target_weights": {fallback: 1.0},
            },
        }

    ranking_universe = set(config.universe)
    benchmark_symbols = {config.benchmark} if config.benchmark else set()
    cash_symbols = {config.cash_equivalent} if config.cash_equivalent else set()

    ranked_trailing = _get_trailing_returns(
        prices=prices,
        end_idx_exclusive=len(prices.index),
        lookback_days=config.lookback_days,
        symbols=ranking_universe,
    )
    benchmark_trailing = _get_trailing_returns(
        prices=prices,
        end_idx_exclusive=len(prices.index),
        lookback_days=config.lookback_days,
        symbols=benchmark_symbols,
    )
    cash_trailing = _get_trailing_returns(
        prices=prices,
        end_idx_exclusive=len(prices.index),
        lookback_days=config.lookback_days,
        symbols=cash_symbols,
    )

    selected_symbols, weights, risk_off, risk_reason = _choose_holdings_for_day(
        config=config,
        ranked_trailing=ranked_trailing,
        benchmark_trailing=benchmark_trailing,
        cash_trailing=cash_trailing,
    )

    asof = prices.index[-1]
    target_w = {sym: float(w) for sym, w in zip(selected_symbols, weights)}

    if not target_w:
        fallback = config.cash_equivalent or DEFAULT_CASH_EQUIVALENT
        target_w = {fallback: 1.0}
        signal = "HOLD"
        note = f"As of {asof}: no valid growth ranks; holding {fallback}"
    elif risk_off:
        signal = "REBALANCE"
        note = f"As of {asof}: risk_off ({risk_reason}); holding {', '.join(target_w.keys())}"
    else:
        signal = "REBALANCE"
        note = f"As of {asof}: top={', '.join(selected_symbols)}"

    return {
        "bot_id": bot_id,
        "bot_type": BOT_TYPE_ALPHA1,
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": {
            "asof": str(asof),
            "interval": interval,
            "lookback_days": config.lookback_days,
            "top_n": config.top_n,
            "universe_size": len(config.universe),
            "benchmark": config.benchmark,
            "cash_equivalent": config.cash_equivalent,
            "risk_off": risk_off,
            "risk_reason": risk_reason,
            "kill_switch_mode": config.kill_switch_mode,
            "benchmark_mode": config.benchmark_mode,
            "top_return_threshold": config.top_return_threshold,
            "benchmark_return_threshold": config.benchmark_return_threshold,
            "transaction_cost_bps": config.transaction_cost_bps,
            "slippage_bps": config.slippage_bps,
            "rank_weights": [float(x) for x in config.rank_weights.tolist()],
            "weight_model": "rank_weights",
            "timing": "rank_on_t_minus_1_apply_return_on_t",
            "target_weights": target_w,
        },
    }
