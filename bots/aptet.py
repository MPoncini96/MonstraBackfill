from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from bot_identity import BOT_TYPE_APTET

DEFAULT_UNIVERSE = [
    "LLY", "NVO", "VRTX", "REGN", "ALNY", "CRSP", "ISRG", "SYK", "BSX",
    "ABT", "MDT", "TMO", "DHR", "ILMN", "UNH", "HCA", "GILD", "MRNA",
    "DXCM", "GEHC", "PODD", "CVS", "BDX", "AZN", "ABBV",
]
DEFAULT_FALLBACK_TICKER = "VOO"
DEFAULT_BENCHMARK_TICKER = "VOO"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MIN_HOLDINGS = 2
DEFAULT_MAX_HOLDINGS = 8
DEFAULT_ADAPTATION_SPEED = "balanced"
PARAMETER_REVIEW_DAYS = 5
MIN_OPTIMIZATION_SAMPLES = 20
TOP_RETURN_THRESHOLD = 0.0
BENCHMARK_RETURN_THRESHOLD = 0.0
REQUIRE_FULL_LOOKBACK_WINDOW = True
USE_FALLBACK_TICKER = True
BENCHMARK_FILTER_ENABLED = False
APTET_LIVE_EQUITY_SOURCE = "live trading"

logger = logging.getLogger(__name__)
DOWNLOAD_BATCH_SIZE = 5


@dataclass(frozen=True)
class AdaptationProfile:
    candidate_lookbacks: list[int]
    optimization_train_days: int
    optimization_check_days: int


ADAPTATION_PROFILES: dict[str, AdaptationProfile] = {
    "conservative": AdaptationProfile([20, 25, 30], 63, 15),
    "balanced": AdaptationProfile([10, 15, 20, 25, 30], 42, 10),
    "aggressive": AdaptationProfile([5, 10, 15, 20], 30, 5),
}


@dataclass
class AptetConfig:
    universe: list[str]
    fallback_ticker: str
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    min_holdings: int = DEFAULT_MIN_HOLDINGS
    max_holdings: int = DEFAULT_MAX_HOLDINGS
    adaptation_speed: str = DEFAULT_ADAPTATION_SPEED
    risk_off_enabled: bool = True
    history_period: str | None = None
    interval: str = "1d"
    bot_id: str | None = None


@dataclass
class AptetAdaptationState:
    selected_lookback_days: int | None = None
    selected_top_n: int | None = None
    realized_returns_since_change: list[float] = field(default_factory=list)


def _clean_ticker(value: Any, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    ticker = str(value).strip().upper()
    return ticker or fallback


def _safe_int(value: Any, fallback: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, parsed)


def _safe_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _normalize_universe(raw_universe: Any, fallback_ticker: str | None = None) -> list[str]:
    if isinstance(raw_universe, str):
        try:
            import json
            raw_universe = json.loads(raw_universe)
        except Exception:
            return []
    if not isinstance(raw_universe, (list, tuple)):
        return []
    fallback = _clean_ticker(fallback_ticker)
    seen: set[str] = set()
    normalized: list[str] = []
    for item in raw_universe:
        ticker = _clean_ticker(item)
        if not ticker or ticker == fallback or ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return normalized


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
    return qty * {"d": 1, "w": 7, "m": 30, "y": 365}.get(token[-1], 0)


def _adaptation_profile(config: AptetConfig) -> AdaptationProfile:
    return ADAPTATION_PROFILES.get(config.adaptation_speed, ADAPTATION_PROFILES[DEFAULT_ADAPTATION_SPEED])


def candidate_top_ns(config: AptetConfig) -> list[int]:
    max_holdings = min(len(config.universe), max(1, int(config.max_holdings)))
    min_holdings = min(max_holdings, max(1, int(config.min_holdings)))
    return list(range(min_holdings, max_holdings + 1)) if max_holdings >= 1 else [DEFAULT_MIN_HOLDINGS]


def _bounded_candidate_top_ns(config: AptetConfig, previous_selected_top_n: int | None = None) -> list[int]:
    top_ns = candidate_top_ns(config)
    if previous_selected_top_n is None:
        return top_ns
    bounded = [top_n for top_n in top_ns if abs(int(top_n) - int(previous_selected_top_n)) <= 1]
    return bounded or top_ns


def _rolling_compounded_return(daily_returns: list[float]) -> float | None:
    if not daily_returns:
        return None
    compounded = 1.0
    for daily_return in daily_returns:
        compounded *= 1.0 + float(daily_return)
    return compounded - 1.0


def _advance_adaptation_state(
    state: AptetAdaptationState | None,
    metadata: dict[str, Any],
    realized_return: float,
) -> AptetAdaptationState:
    selected_lookback = metadata.get("selectedLookbackDays")
    selected_top_n = metadata.get("selectedTopN")
    lookback_value = int(selected_lookback) if selected_lookback is not None else None
    top_n_value = int(selected_top_n) if selected_top_n is not None else None
    current = state or AptetAdaptationState()
    parameter_changed = current.selected_lookback_days != lookback_value or current.selected_top_n != top_n_value
    realized_returns = [] if parameter_changed else list(current.realized_returns_since_change)
    realized_returns.append(float(realized_return))
    return AptetAdaptationState(
        selected_lookback_days=lookback_value,
        selected_top_n=top_n_value,
        realized_returns_since_change=realized_returns[-PARAMETER_REVIEW_DAYS:],
    )


def aptet_config_from_db_dict(data: dict[str, Any] | None, *, bot_id: str | None = None) -> AptetConfig:
    raw = data or {}
    fallback_ticker = _clean_ticker(raw.get("fallback_ticker"), DEFAULT_FALLBACK_TICKER) or DEFAULT_FALLBACK_TICKER
    universe = _normalize_universe(raw.get("universe"), fallback_ticker) or list(DEFAULT_UNIVERSE)
    max_holdings = min(len(universe), _safe_int(raw.get("max_holdings"), DEFAULT_MAX_HOLDINGS))
    min_holdings = min(max_holdings, _safe_int(raw.get("min_holdings"), DEFAULT_MIN_HOLDINGS))
    adaptation_speed = str(raw.get("adaptation_speed") or DEFAULT_ADAPTATION_SPEED).strip().lower() or DEFAULT_ADAPTATION_SPEED
    if adaptation_speed not in ADAPTATION_PROFILES:
        adaptation_speed = DEFAULT_ADAPTATION_SPEED
    return AptetConfig(
        universe=universe,
        fallback_ticker=fallback_ticker,
        benchmark_ticker=_clean_ticker(raw.get("benchmark_ticker"), DEFAULT_BENCHMARK_TICKER) or DEFAULT_BENCHMARK_TICKER,
        min_holdings=min_holdings,
        max_holdings=max_holdings,
        adaptation_speed=adaptation_speed,
        risk_off_enabled=_safe_bool(raw.get("risk_off_enabled"), True),
        history_period=(str(raw.get("history_period") or "").strip() or None),
        interval=str(raw.get("interval") or "1d").strip() or "1d",
        bot_id=bot_id or (str(raw.get("bot_id")).strip() if raw.get("bot_id") is not None else None),
    )


def _load_db_config(bot_id: str) -> AptetConfig:
    try:
        from db import get_conn
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM trading.aptet WHERE bot_id = %s AND is_active = TRUE LIMIT 1", (bot_id,))
                row = cur.fetchone()
    except Exception:
        row = None
    return aptet_config_from_db_dict(dict(row) if row else None, bot_id=bot_id)


def _resolve_download_window(config: AptetConfig) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    profile = _adaptation_profile(config)
    buffer_days = max(_parse_period_days(config.history_period), profile.optimization_train_days + max(profile.candidate_lookbacks) + 45, 180)
    return (today - timedelta(days=buffer_days)).isoformat(), (today + timedelta(days=1)).isoformat()


def _requested_symbols(config: AptetConfig) -> list[str]:
    symbols = list(config.universe)
    for symbol in (config.fallback_ticker, config.benchmark_ticker):
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _chunk_symbols(symbols: list[str], size: int) -> list[list[str]]:
    return [symbols[index:index + size] for index in range(0, len(symbols), size)]


def _normalize_downloaded_prices(prices: pd.DataFrame, requested_symbols: list[str]) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    normalized = prices.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized.columns = [str(col).upper() for col in normalized.columns]
    normalized = normalized.loc[:, normalized.columns.isin([symbol.upper() for symbol in requested_symbols])]
    normalized = normalized.apply(pd.to_numeric, errors="coerce").astype("float64", copy=False)
    normalized = normalized.dropna(axis=1, how="all")
    normalized = normalized.dropna(how="all")
    return normalized


def _extract_price_frame_from_download(raw: Any, requested_symbols: list[str]) -> pd.DataFrame:
    if raw is None or not isinstance(raw, (pd.DataFrame, pd.Series)):
        return pd.DataFrame()
    if isinstance(raw, pd.Series):
        if len(requested_symbols) != 1:
            return pd.DataFrame()
        frame = raw.to_frame(name=requested_symbols[0])
        return _normalize_downloaded_prices(frame, requested_symbols)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        extracted: pd.DataFrame | None = None
        for field in ("Close", "Adj Close"):
            for level in range(raw.columns.nlevels):
                try:
                    if field in raw.columns.get_level_values(level):
                        candidate = raw.xs(field, axis=1, level=level, drop_level=True)
                        extracted = candidate.to_frame() if isinstance(candidate, pd.Series) else candidate
                        break
                except Exception:
                    continue
            if extracted is not None:
                break
        if extracted is None:
            return pd.DataFrame()
        if isinstance(extracted.columns, pd.MultiIndex):
            extracted.columns = [
                str(next((part for part in reversed(col) if str(part).upper() in requested_symbols), col[-1])).upper()
                for col in extracted.columns
            ]
        return _normalize_downloaded_prices(extracted, requested_symbols)
    direct_symbol_columns = [column for column in raw.columns if str(column).upper() in requested_symbols]
    if direct_symbol_columns:
        return _normalize_downloaded_prices(raw[direct_symbol_columns], requested_symbols)
    if len(requested_symbols) == 1:
        for field in ("Adj Close", "Close"):
            if field in raw.columns:
                return _normalize_downloaded_prices(raw[[field]].rename(columns={field: requested_symbols[0]}), requested_symbols)
    return pd.DataFrame()


def _download_price_frame(symbols: list[str], start_date: str, end_date: str, *, threads: bool) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=threads,
            timeout=20,
        )
    except Exception as exc:
        logger.warning("Aptet price download failed symbols=%s threads=%s error=%s", symbols, threads, exc)
        return pd.DataFrame()
    prices = _extract_price_frame_from_download(raw, [symbol.upper() for symbol in symbols])
    if prices.empty:
        logger.warning("Aptet price download returned unusable data symbols=%s threads=%s", symbols, threads)
    return prices


def _merge_price_frames(frames: list[pd.DataFrame], requested_symbols: list[str]) -> pd.DataFrame:
    combined = pd.DataFrame()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if combined.empty:
            combined = frame.copy()
        else:
            combined = combined.combine_first(frame)
    if combined.empty:
        return combined
    combined = _normalize_downloaded_prices(combined, requested_symbols)
    ordered_columns = [symbol for symbol in requested_symbols if symbol in combined.columns]
    return combined.reindex(columns=ordered_columns)


def _has_minimum_usable_symbols(prices: pd.DataFrame, config: AptetConfig) -> bool:
    if prices.empty:
        logger.error("Aptet price load produced no usable prices bot_id=%s", config.bot_id)
        return False
    available_symbols = {str(column).upper() for column in prices.columns if prices[column].notna().any()}
    if config.fallback_ticker.upper() not in available_symbols:
        logger.error("Aptet price load missing fallback ticker bot_id=%s fallback=%s", config.bot_id, config.fallback_ticker)
        return False
    ranked_symbols = [symbol for symbol in config.universe if symbol.upper() in available_symbols]
    if len(ranked_symbols) < int(config.min_holdings):
        logger.error(
            "Aptet price load insufficient ranked symbols bot_id=%s available=%d required=%d",
            config.bot_id,
            len(ranked_symbols),
            int(config.min_holdings),
        )
        return False
    return True


def _load_prices_from_market_data(config: AptetConfig, start_date: str, end_date: str) -> pd.DataFrame:
    requested_symbols = [symbol.upper() for symbol in _requested_symbols(config)]
    if not requested_symbols:
        return pd.DataFrame()
    try:
        from db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ticker, price_date, closing_price
                    FROM trading."Market_Data"
                    WHERE ticker = ANY(%s)
                      AND price_date >= %s
                      AND price_date <= %s
                    ORDER BY price_date ASC, ticker ASC
                    """,
                    (requested_symbols, start_date, end_date),
                )
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Aptet Market_Data load failed bot_id=%s error=%s", config.bot_id, exc)
        return pd.DataFrame()
    if not rows:
        logger.warning(
            "Aptet Market_Data returned no rows bot_id=%s start=%s end=%s symbols=%s",
            config.bot_id,
            start_date,
            end_date,
            requested_symbols,
        )
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["ticker", "price_date", "closing_price"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["price_date"] = pd.to_datetime(frame["price_date"])
    frame["closing_price"] = pd.to_numeric(frame["closing_price"], errors="coerce")
    prices = frame.pivot(index="price_date", columns="ticker", values="closing_price")
    prices = _normalize_downloaded_prices(prices, requested_symbols)
    ordered_columns = [symbol for symbol in requested_symbols if symbol in prices.columns]
    prices = prices.reindex(columns=ordered_columns)
    if prices.empty:
        logger.warning(
            "Aptet Market_Data pivot produced no usable prices bot_id=%s start=%s end=%s",
            config.bot_id,
            start_date,
            end_date,
        )
    return prices


def _build_price_frame(config: AptetConfig, start_date: str, end_date: str) -> pd.DataFrame:
    prices = _load_prices_from_market_data(config, start_date, end_date)
    if prices.empty:
        symbols = [symbol.upper() for symbol in _requested_symbols(config)]
        downloaded_frames: list[pd.DataFrame] = []
        remaining = list(symbols)

        initial = _download_price_frame(remaining, start_date, end_date, threads=True)
        if not initial.empty:
            downloaded_frames.append(initial)
            remaining = [symbol for symbol in remaining if symbol not in initial.columns]

        if remaining:
            for chunk in _chunk_symbols(remaining, DOWNLOAD_BATCH_SIZE):
                chunk_frame = _download_price_frame(chunk, start_date, end_date, threads=False)
                if not chunk_frame.empty:
                    downloaded_frames.append(chunk_frame)
            recovered_symbols = {column for frame in downloaded_frames for column in frame.columns}
            remaining = [symbol for symbol in symbols if symbol not in recovered_symbols]

        if remaining:
            for symbol in remaining:
                single_frame = _download_price_frame([symbol], start_date, end_date, threads=False)
                if not single_frame.empty:
                    downloaded_frames.append(single_frame)

        prices = _merge_price_frames(downloaded_frames, symbols)
    missing_symbols = [symbol for symbol in _requested_symbols(config) if symbol.upper() not in prices.columns]
    if missing_symbols:
        logger.warning("Aptet missing price history bot_id=%s symbols=%s", config.bot_id, missing_symbols)
    if not _has_minimum_usable_symbols(prices, config):
        return pd.DataFrame()
    return prices

def download_aptet_prices(config: AptetConfig, start_date: str, end_date: str) -> pd.DataFrame:
    return _build_price_frame(config, start_date, end_date)


def _get_trailing_returns(prices: pd.DataFrame, end_idx_exclusive: int, lookback_days: int, symbols: set[str] | None = None) -> pd.Series:
    start_idx = max(0, end_idx_exclusive - lookback_days)
    window = prices.iloc[start_idx:end_idx_exclusive]
    if len(window) < 2:
        return pd.Series(dtype=float)

    window_values = window.to_numpy(dtype="float64", copy=False)
    if window_values.ndim != 2 or window_values.shape[0] < 2:
        return pd.Series(dtype=float)

    if REQUIRE_FULL_LOOKBACK_WINDOW:
        valid_mask = np.isfinite(window_values).all(axis=0)
    else:
        valid_mask = np.isfinite(window_values[0]) & np.isfinite(window_values[-1])

    if symbols is not None:
        symbol_mask = np.fromiter((column in symbols for column in window.columns), dtype=bool, count=len(window.columns))
        valid_mask = valid_mask & symbol_mask

    if not bool(valid_mask.any()):
        return pd.Series(dtype=float)

    first = window_values[0, valid_mask]
    last = window_values[-1, valid_mask]
    columns = window.columns.to_numpy(copy=False)[valid_mask]

    positive_first = np.isfinite(first) & np.isfinite(last) & (first > 0.0)
    if not bool(positive_first.any()):
        return pd.Series(dtype=float)

    first = first[positive_first]
    last = last[positive_first]
    columns = columns[positive_first]

    trailing_values = (last / first) - 1.0
    finite_mask = np.isfinite(trailing_values)
    if not bool(finite_mask.any()):
        return pd.Series(dtype=float)

    return pd.Series(trailing_values[finite_mask], index=columns[finite_mask], dtype=float)

def _evaluate_risk_off(config: AptetConfig, ranked_trailing: pd.Series, benchmark_trailing: pd.Series) -> tuple[bool, str]:
    reasons: list[str] = []
    if config.risk_off_enabled:
        if ranked_trailing.empty:
            reasons.append("no_ranked_candidates")
        else:
            top_ret = float(ranked_trailing.max())
            if top_ret <= TOP_RETURN_THRESHOLD:
                reasons.append(f"top_non_positive:{top_ret:.6f}")
    if BENCHMARK_FILTER_ENABLED:
        benchmark_ret = benchmark_trailing.get(config.benchmark_ticker, np.nan)
        if pd.isna(benchmark_ret) or float(benchmark_ret) <= BENCHMARK_RETURN_THRESHOLD:
            reasons.append("benchmark_filter_failed")
    return (True, "; ".join(reasons)) if reasons else (False, "risk_on")


def _equal_weight_holdings(ranked_trailing: pd.Series, top_n: int) -> tuple[list[str], np.ndarray]:
    if ranked_trailing.empty:
        return [], np.array([], dtype=float)
    selected = ranked_trailing.sort_values(ascending=False).head(top_n).index.tolist()
    if not selected:
        return [], np.array([], dtype=float)
    return selected, np.ones(len(selected), dtype=float) / float(len(selected))


def _choose_holdings_for_day(config: AptetConfig, ranked_trailing: pd.Series, benchmark_trailing: pd.Series, top_n: int) -> tuple[list[str], np.ndarray, bool, str]:
    risk_off, reason = _evaluate_risk_off(config, ranked_trailing, benchmark_trailing)
    if risk_off:
        return [config.fallback_ticker], np.array([1.0], dtype=float), True, reason
    selected, weights = _equal_weight_holdings(ranked_trailing, top_n)
    if not selected and USE_FALLBACK_TICKER:
        return [config.fallback_ticker], np.array([1.0], dtype=float), True, "no_selected_symbols"
    return selected, weights, False, reason


def _compute_weighted_period_return(today: pd.Series, tomorrow: pd.Series, selected_symbols: list[str], weights: np.ndarray) -> float:
    period_return = 0.0
    for ticker, weight in zip(selected_symbols, weights):
        p0 = today.get(ticker, np.nan)
        p1 = tomorrow.get(ticker, np.nan)
        if pd.isna(p0) or pd.isna(p1) or float(p0) <= 0:
            return 0.0
        period_return += float(weight) * ((float(p1) / float(p0)) - 1.0)
    return float(period_return)


def _simulate_param_combo(prices: pd.DataFrame, config: AptetConfig, lookback: int, top_n: int) -> dict[str, Any] | None:
    profile = _adaptation_profile(config)
    start_idx = max(lookback + 1, len(prices.index) - profile.optimization_train_days)
    end_idx = len(prices.index) - 1
    ranking_universe = set(config.universe)
    benchmark_symbols = {config.benchmark_ticker} if config.benchmark_ticker else set()
    daily_returns: list[float] = []
    for idx in range(start_idx, end_idx):
        ranked_trailing = _get_trailing_returns(prices, idx, lookback, ranking_universe)
        benchmark_trailing = _get_trailing_returns(prices, idx, lookback, benchmark_symbols)
        selected, weights, _risk_off, _reason = _choose_holdings_for_day(config, ranked_trailing, benchmark_trailing, top_n)
        today = prices.iloc[idx]
        tomorrow = prices.iloc[idx + 1]
        period_return = 0.0
        valid = True
        for ticker, weight in zip(selected, weights):
            p0 = today.get(ticker, np.nan)
            p1 = tomorrow.get(ticker, np.nan)
            if pd.isna(p0) or pd.isna(p1) or float(p0) <= 0:
                valid = False
                break
            period_return += float(weight) * ((float(p1) / float(p0)) - 1.0)
        if valid:
            daily_returns.append(period_return)
    if len(daily_returns) < MIN_OPTIMIZATION_SAMPLES:
        return None
    returns = pd.Series(daily_returns, dtype=float)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_drawdown = float(drawdown.min())
    volatility = float(returns.std())
    sharpe_like = float((returns.mean() / volatility) * np.sqrt(252)) if volatility > 0 else 0.0
    return {
        "selectedLookbackDays": int(lookback),
        "selectedTopN": int(top_n),
        "score": total_return + sharpe_like * 0.05 + max_drawdown * 0.50,
        "totalReturn": total_return,
        "maxDrawdown": max_drawdown,
        "sharpeLike": sharpe_like,
    }


def optimize_aptet_params(
    prices: pd.DataFrame,
    config: AptetConfig,
    *,
    end_idx_exclusive: int | None = None,
    previous_selected_top_n: int | None = None,
) -> dict[str, Any] | None:
    view = prices if end_idx_exclusive is None else prices.iloc[:end_idx_exclusive]
    if view.empty:
        return None
    profile = _adaptation_profile(config)
    top_ns = _bounded_candidate_top_ns(config, previous_selected_top_n)
    best: dict[str, Any] | None = None
    for lookback in profile.candidate_lookbacks:
        for top_n in top_ns:
            result = _simulate_param_combo(view, config, lookback, top_n)
            if result is None:
                continue
            if best is None or float(result["score"]) > float(best["score"]):
                best = result
    if best is None:
        return None
    return {
        **best,
        "candidateLookbacks": list(profile.candidate_lookbacks),
        "candidateTopNs": list(top_ns),
        "adaptationSpeed": config.adaptation_speed,
        "optimizationTrainDays": int(profile.optimization_train_days),
        "optimizationCheckDays": int(profile.optimization_check_days),
        "lastOptimizationDate": str(view.index[-1].date()),
    }

def resolve_aptet_decision(
    prices: pd.DataFrame,
    config: AptetConfig,
    *,
    end_idx_exclusive: int | None = None,
    previous_selected_top_n: int | None = None,
    adaptation_state: AptetAdaptationState | None = None,
) -> tuple[list[str], np.ndarray, bool, str, dict[str, Any]]:
    if not all(pd.api.types.is_float_dtype(dtype) for dtype in prices.dtypes):
        prices = prices.astype("float64", copy=False)
    view_end = int(end_idx_exclusive if end_idx_exclusive is not None else len(prices.index))
    profile = _adaptation_profile(config)
    prior_top_n = int(adaptation_state.selected_top_n) if adaptation_state and adaptation_state.selected_top_n is not None else (int(previous_selected_top_n) if previous_selected_top_n is not None else None)
    prior_lookback = int(adaptation_state.selected_lookback_days) if adaptation_state and adaptation_state.selected_lookback_days is not None else None
    recent_returns = list(adaptation_state.realized_returns_since_change) if adaptation_state else []
    trailing_review_return = _rolling_compounded_return(recent_returns[-PARAMETER_REVIEW_DAYS:]) if len(recent_returns) >= PARAMETER_REVIEW_DAYS else None
    should_search = prior_top_n is None or prior_lookback is None
    search_reason = "initial_selection" if should_search else "hold_current_parameters"
    if not should_search and trailing_review_return is not None and trailing_review_return < 0.0:
        should_search = True
        search_reason = "negative_last_5d_since_change"
    top_ns = _bounded_candidate_top_ns(config, prior_top_n)
    best: dict[str, Any] | None = None
    if should_search:
        best = optimize_aptet_params(
            prices,
            config,
            end_idx_exclusive=view_end,
            previous_selected_top_n=prior_top_n,
        )
    selected_lookback = int(best["selectedLookbackDays"]) if best else (prior_lookback if prior_lookback is not None else DEFAULT_LOOKBACK_DAYS)
    selected_top_n = int(best["selectedTopN"]) if best else (prior_top_n if prior_top_n is not None else min(DEFAULT_MIN_HOLDINGS, max(1, len(config.universe))))
    parameter_changed = prior_lookback != selected_lookback or prior_top_n != selected_top_n
    metadata: dict[str, Any] = {
        "selectedLookbackDays": selected_lookback,
        "selectedTopN": selected_top_n,
        "candidateLookbacks": list(profile.candidate_lookbacks),
        "candidateTopNs": list(top_ns),
        "adaptationSpeed": config.adaptation_speed,
        "optimizationTrainDays": int(profile.optimization_train_days),
        "optimizationCheckDays": int(profile.optimization_check_days),
        "lastOptimizationDate": best.get("lastOptimizationDate") if best else None,
        "riskOffReason": None,
        "fallbackTicker": config.fallback_ticker,
        "rankedTrailingReturns": {},
        "optimizationScore": best.get("score") if best else None,
        "usedDefaultParameters": best is None and prior_top_n is None,
        "previousSelectedTopN": prior_top_n,
        "previousSelectedLookbackDays": prior_lookback,
        "parameterSearchTriggered": bool(should_search),
        "parameterSearchReason": search_reason,
        "parameterChanged": bool(parameter_changed),
        "parameterReviewDays": PARAMETER_REVIEW_DAYS,
        "daysSinceParameterChange": len(recent_returns),
        "trailingReturnSinceLastChange5D": trailing_review_return,
    }
    if view_end < selected_lookback + 1:
        metadata["riskOffReason"] = "no_history"
        return [config.fallback_ticker], np.array([1.0], dtype=float), True, "no_history", metadata
    ranking_universe = set(config.universe)
    benchmark_symbols = {config.benchmark_ticker} if config.benchmark_ticker else set()
    ranked_trailing = _get_trailing_returns(prices, view_end, selected_lookback, ranking_universe)
    benchmark_trailing = _get_trailing_returns(prices, view_end, selected_lookback, benchmark_symbols)
    selected_symbols, weights, risk_off, risk_reason = _choose_holdings_for_day(config, ranked_trailing, benchmark_trailing, selected_top_n)
    if selected_symbols:
        metadata["rankedTrailingReturns"] = {symbol: float(ranked_trailing.get(symbol)) for symbol in selected_symbols if symbol in ranked_trailing.index}
    metadata["riskOffReason"] = risk_reason if risk_off else None
    return selected_symbols, weights, risk_off, risk_reason, metadata


def run_aptet(bot_id: str, use_db_config: bool = True) -> dict[str, Any]:
    ts = datetime.now(timezone.utc)
    config = _load_db_config(bot_id) if use_db_config else aptet_config_from_db_dict({"bot_id": bot_id}, bot_id=bot_id)
    start_date, end_date = _resolve_download_window(config)
    prices = _build_price_frame(config, start_date, end_date)
    if prices.empty or len(prices.index) < 2:
        fallback = config.fallback_ticker or DEFAULT_FALLBACK_TICKER
        profile = _adaptation_profile(config)
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_APTET,
            "ts": ts,
            "signal": "HOLD",
            "note": f"Not enough {config.interval} history; holding {fallback}",
            "payload": {
                "asof": None,
                "interval": config.interval,
                "target_weights": {fallback: 1.0},
                "selectedLookbackDays": DEFAULT_LOOKBACK_DAYS,
                "selectedTopN": min(DEFAULT_MIN_HOLDINGS, max(1, len(config.universe))),
                "candidateLookbacks": list(profile.candidate_lookbacks),
                "candidateTopNs": list(candidate_top_ns(config)),
                "adaptationSpeed": config.adaptation_speed,
                "optimizationTrainDays": profile.optimization_train_days,
                "optimizationCheckDays": profile.optimization_check_days,
                "lastOptimizationDate": None,
                "riskOffReason": "no_history",
                "fallbackTicker": fallback,
                "rankedTrailingReturns": {},
            },
        }
    adaptation_state: AptetAdaptationState | None = None
    if len(prices.index) > 2:
        for index in range(1, len(prices.index) - 1):
            selected_symbols, weights, _risk_off, _risk_reason, replay_metadata = resolve_aptet_decision(
                prices,
                config,
                end_idx_exclusive=index,
                adaptation_state=adaptation_state,
            )
            realized_return = _compute_weighted_period_return(prices.iloc[index], prices.iloc[index + 1], selected_symbols, weights)
            adaptation_state = _advance_adaptation_state(adaptation_state, replay_metadata, realized_return)
    selected_symbols, weights, risk_off, risk_reason, metadata = resolve_aptet_decision(
        prices,
        config,
        adaptation_state=adaptation_state,
    )
    asof = prices.index[-1]
    target_weights = {symbol: float(weight) for symbol, weight in zip(selected_symbols, weights)}
    if not target_weights:
        fallback = config.fallback_ticker or DEFAULT_FALLBACK_TICKER
        target_weights = {fallback: 1.0}
        risk_off = True
        risk_reason = risk_reason or "no_selected_symbols"
        metadata["riskOffReason"] = risk_reason
    note = f"As of {asof.date()}: fallback to {', '.join(target_weights.keys())} ({risk_reason})" if risk_off else f"As of {asof.date()}: selected {', '.join(selected_symbols)}"
    return {
        "bot_id": bot_id,
        "bot_type": BOT_TYPE_APTET,
        "ts": ts,
        "signal": "REBALANCE" if target_weights else "HOLD",
        "note": note,
        "payload": {
            "asof": str(asof),
            "interval": config.interval,
            "universeSize": len(config.universe),
            "benchmarkTicker": config.benchmark_ticker,
            "riskOff": bool(risk_off),
            "target_weights": target_weights,
            **metadata,
        },
        "equity_source": APTET_LIVE_EQUITY_SOURCE,
    }
