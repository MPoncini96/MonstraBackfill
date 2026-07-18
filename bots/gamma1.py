# bots/gamma1.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from bot_identity import BOT_TYPE_GAMMA1
from market_data_provider import get_daily_bars

DEFAULT_UNIVERSE = [
    "VOO",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "COST",
    "JPM",
    "UNH",
    "HD",
    "AMD",
    "AVGO",
]
DEFAULT_BENCHMARK = "VOO"
DEFAULT_REBALANCE = "W-FRI"
DEFAULT_MAX_POSITIONS = 5
DEFAULT_ENTRY_SCORE = 5.0
DEFAULT_EXIT_SCORE = 4.0
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_HOLD_BONUS = 0.25
DEFAULT_HISTORY_DAYS = 900
DEFAULT_GROUP = "default"
INVERSE_GROUPS = {"inverse"}
TICKER_GROUPS = {
    "NVDA": "mega_cap",
    "AVGO": "mega_cap",
    "TSM": "mega_cap",
    "META": "mega_cap",
    "AMZN": "mega_cap",
    "MSFT": "mega_cap",
    "AAPL": "mega_cap",
    "GOOGL": "mega_cap",
    "QQQ": "benchmark_core",
    "VOO": "benchmark_core",
    "SPY": "benchmark_core",
    "IVV": "benchmark_core",
    "ASML": "upper_large_cap",
    "LLY": "upper_large_cap",
    "JPM": "upper_large_cap",
    "WMT": "upper_large_cap",
    "NFLX": "upper_large_cap",
    "XOM": "upper_large_cap",
    "COST": "upper_large_cap",
    "UNH": "upper_large_cap",
    "PLTR": "large_cap_growth",
    "ANET": "large_cap_growth",
    "BKNG": "large_cap_growth",
    "ISRG": "large_cap_growth",
    "IBKR": "large_cap_growth",
    "AMD": "large_cap_growth",
    "GEV": "industrial_large_cap",
    "ETN": "industrial_large_cap",
    "CAT": "industrial_large_cap",
    "PH": "industrial_large_cap",
    "TT": "industrial_large_cap",
    "URI": "industrial_large_cap",
    "HD": "industrial_large_cap",
    "VST": "energy_power_cyclical",
    "CEG": "energy_power_cyclical",
    "FCX": "energy_power_cyclical",
}
REWARD_TEMPLATE: dict[str, float] = {
    "ret10_positive": 1.0,
    "ret20_positive": 2.0,
    "ret20_vs_benchmark": 2.0,
    "price_above_ema20": 1.0,
    "ema10_above_ema20": 1.0,
    "ema20_above_ema50": 1.0,
    "within_10pct_high": 2.0,
    "rsi_55_70": 1.0,
}
PENALTY_TEMPLATE: dict[str, float] = {
    "ret10_negative": 2.0,
    "price_below_ema50": 2.0,
    "rsi_above_80": 1.0,
    "far_from_high": 1.0,
    "volatility_spike": 1.0,
    "fading_activity": 1.0,
    "market_regime_penalty": 2.0,
}
AVERAGE_SCORE_ALLOCATION_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (9.0, 1.0),
    (7.0, 0.75),
    (5.0, 0.50),
    (3.0, 0.25),
    (0.0, 0.0),
)


@dataclass
class Gamma1FeatureFlags:
    """Feature toggles matching trading.gamma1 columns."""

    use_ret10_positive: bool = True
    use_ret20_positive: bool = True
    use_ret20_vs_benchmark: bool = True
    use_price_above_ema20: bool = True
    use_ema10_above_ema20: bool = True
    use_ema20_above_ema50: bool = True
    use_within_10pct_high: bool = True
    use_rsi_55_70: bool = True
    use_penalty_ret10_negative: bool = True
    use_penalty_price_below_ema50: bool = True
    use_penalty_rsi_above_80: bool = True
    use_penalty_far_from_high: bool = True
    use_penalty_volatility_spike: bool = True
    use_penalty_fading_activity: bool = True
    use_market_regime_penalty: bool = True


@dataclass
class Gamma1Config:
    universe: list[str]
    benchmark: str = DEFAULT_BENCHMARK
    rebalance: str = DEFAULT_REBALANCE
    max_positions: int = DEFAULT_MAX_POSITIONS
    entry_score: float = DEFAULT_ENTRY_SCORE
    exit_score: float = DEFAULT_EXIT_SCORE
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    hold_bonus: float = DEFAULT_HOLD_BONUS
    fallback_ticker: str | None = None
    history_period: str | None = None
    weighting_style: str = "confidence_shield"
    feature_flags: Gamma1FeatureFlags = field(default_factory=Gamma1FeatureFlags)
    use_sector_cap: bool = False
    sector_cap_limit: int | None = None
    use_volatility_filter: bool = False
    adaptive_enabled: bool = True
    learn_from_holdings_only: bool = False
    learning_deadzone: float = 0.01
    return_clip: float = 0.10
    max_positive_reward_boost: float = 0.045
    max_negative_reward_cut: float = 0.075
    max_positive_penalty_cut: float = 0.060
    max_negative_penalty_boost: float = 0.085
    reward_mistake_cost: float = 1.25
    penalty_mistake_cost: float = 1.10
    baseline_decay: float = 0.04
    min_rule_weight: float = 0.50
    max_rule_weight: float = 3.25


def _normalize_universe(raw_universe: Any) -> list[str]:
    if raw_universe is None:
        return []
    if isinstance(raw_universe, str):
        try:
            raw_universe = json.loads(raw_universe)
        except Exception:
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


def _safe_bool(value: Any, default: bool = True) -> bool:
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


def normalize_rebalance_rule(raw: Any) -> str:
    if raw is None:
        return DEFAULT_REBALANCE
    s = str(raw).strip().lower()
    if s in ("weekly", "week", "w-fri", "w_fri", "fri"):
        return "W-FRI"
    if s in ("w-mon", "w_mon", "mon"):
        return "W-MON"
    if s in ("daily", "d"):
        return "B"
    u = str(raw).strip().upper().replace("_", "-")
    return u if u else DEFAULT_REBALANCE


def _clean_ticker(value: Any, default: str) -> str:
    if value is None:
        return default
    ticker = str(value).strip().upper()
    return ticker or default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def gamma1_feature_flags_from_row(d: dict[str, Any]) -> Gamma1FeatureFlags:
    return Gamma1FeatureFlags(
        use_ret10_positive=_safe_bool(d.get("use_ret10_positive"), True),
        use_ret20_positive=_safe_bool(d.get("use_ret20_positive"), True),
        use_ret20_vs_benchmark=_safe_bool(d.get("use_ret20_vs_benchmark"), True),
        use_price_above_ema20=_safe_bool(d.get("use_price_above_ema20"), True),
        use_ema10_above_ema20=_safe_bool(d.get("use_ema10_above_ema20"), True),
        use_ema20_above_ema50=_safe_bool(d.get("use_ema20_above_ema50"), True),
        use_within_10pct_high=_safe_bool(d.get("use_within_10pct_high"), True),
        use_rsi_55_70=_safe_bool(d.get("use_rsi_55_70"), True),
        use_penalty_ret10_negative=_safe_bool(d.get("use_penalty_ret10_negative"), True),
        use_penalty_price_below_ema50=_safe_bool(d.get("use_penalty_price_below_ema50"), True),
        use_penalty_rsi_above_80=_safe_bool(d.get("use_penalty_rsi_above_80"), True),
        use_penalty_far_from_high=_safe_bool(d.get("use_penalty_far_from_high"), True),
        use_penalty_volatility_spike=_safe_bool(d.get("use_penalty_volatility_spike"), True),
        use_penalty_fading_activity=_safe_bool(d.get("use_penalty_fading_activity"), True),
        use_market_regime_penalty=_safe_bool(d.get("use_market_regime_penalty"), True),
    )


def gamma1_config_from_db_dict(d: dict[str, Any]) -> Gamma1Config:
    cap = d.get("sector_cap_limit")
    try:
        cap_i = int(cap) if cap is not None else None
    except (TypeError, ValueError):
        cap_i = None

    hp = d.get("history_period")
    history_period = str(hp) if hp is not None and str(hp).strip() else None
    fb = str(d.get("fallback_ticker") or "").strip().upper() or None

    return Gamma1Config(
        universe=_normalize_universe(d.get("universe")) or list(DEFAULT_UNIVERSE),
        benchmark=_clean_ticker(d.get("benchmark"), DEFAULT_BENCHMARK),
        rebalance=normalize_rebalance_rule(d.get("rebalance_frequency") or d.get("rebalance")),
        max_positions=_safe_int(d.get("max_positions"), DEFAULT_MAX_POSITIONS),
        entry_score=_safe_float(d.get("entry_score"), DEFAULT_ENTRY_SCORE),
        exit_score=_safe_float(d.get("exit_score"), DEFAULT_EXIT_SCORE),
        transaction_cost_bps=_safe_float(d.get("transaction_cost_bps"), DEFAULT_TRANSACTION_COST_BPS),
        hold_bonus=_safe_float(d.get("hold_bonus"), DEFAULT_HOLD_BONUS),
        fallback_ticker=fb,
        history_period=history_period,
        weighting_style=str(d.get("weighting_style") or "confidence_shield").strip().lower() or "confidence_shield",
        feature_flags=gamma1_feature_flags_from_row(d),
        use_sector_cap=_safe_bool(d.get("use_sector_cap"), False),
        sector_cap_limit=cap_i,
        use_volatility_filter=_safe_bool(d.get("use_volatility_filter"), False),
        adaptive_enabled=_safe_bool(d.get("adaptive_enabled"), True),
        learn_from_holdings_only=_safe_bool(d.get("learn_from_holdings_only"), False),
        learning_deadzone=_safe_float(d.get("learning_deadzone"), 0.01),
        return_clip=_safe_float(d.get("return_clip"), 0.10),
        max_positive_reward_boost=_safe_float(d.get("max_positive_reward_boost"), 0.045),
        max_negative_reward_cut=_safe_float(d.get("max_negative_reward_cut"), 0.075),
        max_positive_penalty_cut=_safe_float(d.get("max_positive_penalty_cut"), 0.060),
        max_negative_penalty_boost=_safe_float(d.get("max_negative_penalty_boost"), 0.085),
        reward_mistake_cost=_safe_float(d.get("reward_mistake_cost"), 1.25),
        penalty_mistake_cost=_safe_float(d.get("penalty_mistake_cost"), 1.10),
        baseline_decay=_safe_float(d.get("baseline_decay"), 0.04),
        min_rule_weight=_safe_float(d.get("min_rule_weight"), 0.50),
        max_rule_weight=_safe_float(d.get("max_rule_weight"), 3.25),
    )


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


def _build_live_config(bot_id: str, use_db_config: bool = True) -> Gamma1Config:
    config_data: dict[str, Any] | None = None
    if use_db_config:
        try:
            from db import get_gamma1_config

            config_data = get_gamma1_config(bot_id)
        except Exception as exc:
            print(f"Warning: Could not load gamma1 config for {bot_id} from database, using defaults: {exc}")
    if config_data:
        return gamma1_config_from_db_dict(config_data)
    return Gamma1Config(universe=list(DEFAULT_UNIVERSE))


def _resolve_download_window(config: Gamma1Config) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    end_date = today + timedelta(days=1)
    configured_days = _parse_period_days(config.history_period)
    buffer_days = max(DEFAULT_HISTORY_DAYS, configured_days, 400)
    start_date = today - timedelta(days=buffer_days)
    return start_date.isoformat(), end_date.isoformat()

def _group_for_ticker(ticker: str) -> str:
    return TICKER_GROUPS.get(str(ticker).upper(), DEFAULT_GROUP)


def _group_names(tickers: list[str]) -> list[str]:
    return sorted({_group_for_ticker(ticker) for ticker in tickers} | {DEFAULT_GROUP})


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _build_factor_tables(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    feature_flags: Gamma1FeatureFlags | None = None,
) -> dict[str, Any]:
    px = prices.copy()
    bm = benchmark_prices.copy().reindex(px.index)
    ff = feature_flags or Gamma1FeatureFlags()

    ema10 = px.ewm(span=10, adjust=False, min_periods=10).mean()
    ema20 = px.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = px.ewm(span=50, adjust=False, min_periods=50).mean()
    ret10 = px / px.shift(10) - 1.0
    ret20 = px / px.shift(20) - 1.0
    bm_ret20 = bm / bm.shift(20) - 1.0
    bm_ema50 = bm.ewm(span=50, adjust=False, min_periods=50).mean()
    bm_above_ema50 = bm > bm_ema50
    rsi14 = px.apply(rsi, axis=0)
    high_252 = px.rolling(252, min_periods=50).max()
    within_10pct_high = px >= 0.90 * high_252
    rets = px.pct_change().fillna(0.0)
    vol5 = rets.rolling(5).std()
    vol20 = rets.rolling(20).std()
    avgvol5 = px.pct_change().abs().rolling(5).mean()
    avgvol20 = px.pct_change().abs().rolling(20).mean()

    reward_factors = {
        "ret10_positive": (ret10 > 0).astype(int) * int(bool(ff.use_ret10_positive)),
        "ret20_positive": (ret20 > 0).astype(int) * int(bool(ff.use_ret20_positive)),
        "ret20_vs_benchmark": ret20.gt(bm_ret20, axis=0).astype(int) * int(bool(ff.use_ret20_vs_benchmark)),
        "price_above_ema20": (px > ema20).astype(int) * int(bool(ff.use_price_above_ema20)),
        "ema10_above_ema20": (ema10 > ema20).astype(int) * int(bool(ff.use_ema10_above_ema20)),
        "ema20_above_ema50": (ema20 > ema50).astype(int) * int(bool(ff.use_ema20_above_ema50)),
        "within_10pct_high": within_10pct_high.astype(int) * int(bool(ff.use_within_10pct_high)),
        "rsi_55_70": ((rsi14 >= 55) & (rsi14 <= 70)).astype(int) * int(bool(ff.use_rsi_55_70)),
    }
    penalty_factors = {
        "ret10_negative": (ret10 < 0).astype(int) * int(bool(ff.use_penalty_ret10_negative)),
        "price_below_ema50": (px < ema50).astype(int) * int(bool(ff.use_penalty_price_below_ema50)),
        "rsi_above_80": (rsi14 > 80).astype(int) * int(bool(ff.use_penalty_rsi_above_80)),
        "far_from_high": (px < 0.85 * high_252).astype(int) * int(bool(ff.use_penalty_far_from_high)),
        "volatility_spike": (vol5 > vol20 * 1.5).astype(int) * int(bool(ff.use_penalty_volatility_spike)),
        "fading_activity": (avgvol5 < avgvol20).astype(int) * int(bool(ff.use_penalty_fading_activity)),
    }
    market_signal = (~bm_above_ema50).astype(int) * int(bool(ff.use_market_regime_penalty))
    market_df = pd.DataFrame(
        np.repeat(market_signal.values.reshape(-1, 1), len(px.columns), axis=1),
        index=px.index,
        columns=px.columns,
    )
    penalty_factors["market_regime_penalty"] = market_df

    positive = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    penalty = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    total = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    for name, factor in reward_factors.items():
        weighted = factor.astype(float) * REWARD_TEMPLATE[name]
        positive = positive.add(weighted, fill_value=0.0)
        total = total.add(weighted, fill_value=0.0)

    for name, factor in penalty_factors.items():
        weighted = factor.astype(float) * PENALTY_TEMPLATE[name]
        penalty = penalty.add(weighted, fill_value=0.0)
        total = total.sub(weighted, fill_value=0.0)

    return {
        "total": total,
        "positive": positive,
        "penalty": penalty,
        "reward_factors": reward_factors,
        "penalty_factors": penalty_factors,
        "ret10": ret10,
        "ret20": ret20,
        "ema10": ema10,
        "ema20": ema20,
        "ema50": ema50,
        "rsi14": rsi14,
    }


def build_gamma1_score_table(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    feature_flags: Gamma1FeatureFlags | None = None,
) -> dict[str, pd.DataFrame]:
    return _build_factor_tables(prices, benchmark_prices, feature_flags)


def get_gamma1_rebalance_dates(prices_index: pd.DatetimeIndex, rebalance_rule: str) -> list[pd.Timestamp]:
    if prices_index.empty:
        return []
    normalized_index = pd.DatetimeIndex(prices_index).normalize()
    stub = pd.DataFrame(index=normalized_index, data={"_": 0.0})
    rebal = stub.resample(rebalance_rule).last().dropna().index
    actual_by_day: dict[pd.Timestamp, pd.Timestamp] = {}
    for ts in pd.DatetimeIndex(prices_index):
        actual_by_day[ts.normalize()] = ts
    return [actual_by_day[d.normalize()] for d in rebal if d.normalize() in actual_by_day]


def download_gamma1_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = get_daily_bars(tickers, start_date, end_date, adjusted=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data missing 'Close'.")
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all").sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.columns = [str(col).upper() for col in prices.columns]
    return prices


def _initial_weight_maps(tickers: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    groups = _group_names(tickers)
    return (
        {group: REWARD_TEMPLATE.copy() for group in groups},
        {group: PENALTY_TEMPLATE.copy() for group in groups},
    )


def _return_impact(config: Gamma1Config, value: float) -> float:
    abs_return = abs(float(value))
    if abs_return <= config.learning_deadzone:
        return 0.0
    clipped = min(abs_return, config.return_clip)
    impact = (clipped - config.learning_deadzone) / (config.return_clip - config.learning_deadzone)
    return float(np.clip(impact, 0.0, 1.0))


def _apply_baseline_gravity(config: Gamma1Config, weight: float, base_weight: float) -> float:
    return weight * (1.0 - config.baseline_decay) + base_weight * config.baseline_decay


def _update_reward_weight(config: Gamma1Config, current_weight: float, base_weight: float, avg_return: float) -> float:
    impact = _return_impact(config, avg_return)
    if impact == 0.0:
        next_weight = current_weight
    elif avg_return > config.learning_deadzone:
        next_weight = current_weight * (1.0 + config.max_positive_reward_boost * impact)
    elif avg_return < -config.learning_deadzone:
        cut = config.max_negative_reward_cut * impact * config.reward_mistake_cost
        next_weight = current_weight * max(0.70, 1.0 - cut)
    else:
        next_weight = current_weight
    next_weight = _apply_baseline_gravity(config, next_weight, base_weight)
    return float(np.clip(next_weight, config.min_rule_weight, config.max_rule_weight))


def _update_penalty_weight(config: Gamma1Config, current_weight: float, base_weight: float, avg_return: float) -> float:
    impact = _return_impact(config, avg_return)
    if impact == 0.0:
        next_weight = current_weight
    elif avg_return < -config.learning_deadzone:
        next_weight = current_weight * (1.0 + config.max_negative_penalty_boost * impact)
    elif avg_return > config.learning_deadzone:
        cut = config.max_positive_penalty_cut * impact * config.penalty_mistake_cost
        next_weight = current_weight * max(0.70, 1.0 - cut)
    else:
        next_weight = current_weight
    next_weight = _apply_baseline_gravity(config, next_weight, base_weight)
    return float(np.clip(next_weight, config.min_rule_weight, config.max_rule_weight))

def _score_for_date(
    dt: pd.Timestamp,
    tickers: list[str],
    reward_factors: dict[str, pd.DataFrame],
    penalty_factors: dict[str, pd.DataFrame],
    reward_weights: dict[str, dict[str, float]],
    penalty_weights: dict[str, dict[str, float]],
) -> pd.Series:
    scores = pd.Series(0.0, index=tickers, dtype=float)
    for ticker in tickers:
        group = _group_for_ticker(ticker)
        for name, factor in reward_factors.items():
            scores.loc[ticker] += float(factor.loc[dt, ticker]) * reward_weights[group][name]
        for name, factor in penalty_factors.items():
            scores.loc[ticker] -= float(factor.loc[dt, ticker]) * penalty_weights[group][name]
    return scores


def _update_adaptive_weights(
    config: Gamma1Config,
    current_prices: pd.Series,
    previous_price_snapshot: pd.Series | None,
    previous_reward_factors: dict[str, pd.Series] | None,
    previous_penalty_factors: dict[str, pd.Series] | None,
    previous_holdings: list[str],
    reward_weights: dict[str, dict[str, float]],
    penalty_weights: dict[str, dict[str, float]],
) -> None:
    if not config.adaptive_enabled:
        return
    if previous_price_snapshot is None or previous_reward_factors is None or previous_penalty_factors is None:
        return

    common = [ticker for ticker in previous_price_snapshot.index if ticker in current_prices.index]
    if not common:
        return

    previous = previous_price_snapshot[common]
    current = current_prices[common]
    realized = (current / previous - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    if realized.empty:
        return
    if config.learn_from_holdings_only:
        held = [ticker for ticker in previous_holdings if ticker in realized.index]
        if not held:
            return
        realized = realized[held]
        if realized.empty:
            return

    for group in list(reward_weights.keys()):
        group_tickers = [ticker for ticker in realized.index if _group_for_ticker(ticker) == group]
        if not group_tickers:
            continue
        group_returns = realized[group_tickers]

        for name, factor_values in previous_reward_factors.items():
            active = factor_values.reindex(group_returns.index).fillna(0)
            active_returns = group_returns[active > 0]
            if active_returns.empty:
                continue
            reward_weights[group][name] = _update_reward_weight(
                config,
                reward_weights[group][name],
                REWARD_TEMPLATE[name],
                float(active_returns.mean()),
            )

        for name, factor_values in previous_penalty_factors.items():
            active = factor_values.reindex(group_returns.index).fillna(0)
            active_returns = group_returns[active > 0]
            if active_returns.empty:
                continue
            penalty_weights[group][name] = _update_penalty_weight(
                config,
                penalty_weights[group][name],
                PENALTY_TEMPLATE[name],
                float(active_returns.mean()),
            )


def _ensure_fallback_prices(
    universe_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    config: Gamma1Config,
) -> pd.DataFrame:
    tradable = universe_prices.copy()
    fallback = config.fallback_ticker
    if not fallback or fallback in tradable.columns:
        return tradable
    if fallback == config.benchmark:
        tradable[fallback] = benchmark_prices.reindex(tradable.index)
        return tradable.ffill().bfill()

    start_date = tradable.index[0].date().isoformat()
    end_date = (tradable.index[-1].date() + timedelta(days=1)).isoformat()
    try:
        fallback_prices = download_gamma1_prices([fallback], start_date, end_date)
        if fallback in fallback_prices.columns:
            tradable[fallback] = fallback_prices[fallback].reindex(tradable.index)
            return tradable.ffill().bfill()
    except Exception:
        pass

    tradable[fallback] = 1.0
    return tradable


def _derive_initial_holdings(
    tradable_columns: list[str],
    fallback_ticker: str | None,
    initial_holdings: list[str] | None,
    initial_weights: pd.Series | None,
) -> list[str]:
    if initial_holdings:
        return [ticker for ticker in initial_holdings if ticker in tradable_columns]
    if initial_weights is None:
        return []
    fallback = fallback_ticker.upper() if fallback_ticker else None
    holdings: list[str] = []
    for ticker in tradable_columns:
        if ticker == fallback:
            continue
        if float(initial_weights.get(ticker, 0.0)) > 1e-12:
            holdings.append(str(ticker))
    return holdings


def _stock_allocation(score_sum: float, selected_count: int, fallback_ticker: str | None) -> float:
    if selected_count <= 0:
        return 0.0
    if not fallback_ticker:
        return 1.0
    average_score = float(score_sum) / float(selected_count)
    for threshold, allocation in AVERAGE_SCORE_ALLOCATION_THRESHOLDS:
        if average_score >= threshold:
            return allocation
    return 0.0


def _run_adaptive_backtest(
    universe_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    config: Gamma1Config,
    *,
    record_weights: bool,
    initial_holdings: list[str] | None,
    initial_weights: pd.Series | None,
) -> dict[str, Any]:
    dates = universe_prices.index
    rebalance_dates = get_gamma1_rebalance_dates(dates, config.rebalance)
    tradable_prices = _ensure_fallback_prices(universe_prices, benchmark_prices, config)
    tradable_returns = tradable_prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tradable_columns = [str(col) for col in tradable_prices.columns]
    universe_columns = [str(col) for col in universe_prices.columns]
    weights = pd.DataFrame(0.0, index=dates, columns=tradable_columns) if record_weights else None
    current_weights = initial_weights.reindex(tradable_columns).fillna(0.0) if initial_weights is not None else pd.Series(0.0, index=tradable_columns)
    current_holdings = _derive_initial_holdings(universe_columns, config.fallback_ticker, initial_holdings, initial_weights)
    trade_rows: list[dict[str, Any]] = []

    factor_bundle = _build_factor_tables(universe_prices, benchmark_prices, config.feature_flags)
    reward_factors = factor_bundle["reward_factors"]
    penalty_factors = factor_bundle["penalty_factors"]
    reward_weights, penalty_weights = _initial_weight_maps(universe_columns)
    previous_reward_factors: dict[str, pd.Series] | None = None
    previous_penalty_factors: dict[str, pd.Series] | None = None
    previous_prices: pd.Series | None = None
    previous_holdings: list[str] = []

    for index, dt in enumerate(rebalance_dates):
        current_prices = universe_prices.loc[dt]
        _update_adaptive_weights(
            config,
            current_prices,
            previous_prices,
            previous_reward_factors,
            previous_penalty_factors,
            previous_holdings,
            reward_weights,
            penalty_weights,
        )

        today_scores = _score_for_date(
            dt,
            universe_columns,
            reward_factors,
            penalty_factors,
            reward_weights,
            penalty_weights,
        ).dropna()
        adjusted_scores = today_scores.copy()
        for ticker in current_holdings:
            if ticker in adjusted_scores.index:
                adjusted_scores.loc[ticker] += config.hold_bonus
        ranked = adjusted_scores.sort_values(ascending=False)

        selected: list[str] = []
        entered: list[str] = []
        for ticker in ranked.index:
            if float(today_scores.loc[ticker]) < config.entry_score:
                continue
            selected.append(str(ticker))
            if ticker not in current_holdings:
                entered.append(str(ticker))
            if len(selected) >= int(config.max_positions):
                break
        exited = [ticker for ticker in current_holdings if ticker not in selected]
        current_holdings = selected

        selected_score_sum = float(sum(float(today_scores.loc[ticker]) for ticker in selected))
        stock_allocation = _stock_allocation(selected_score_sum, len(selected), config.fallback_ticker)
        fallback_allocation = 1.0 - stock_allocation if config.fallback_ticker else 0.0

        target_weights = pd.Series(0.0, index=tradable_columns)
        if selected and stock_allocation > 0:
            per_stock_weight = stock_allocation / len(selected)
            for ticker in selected:
                target_weights.loc[ticker] = per_stock_weight
        if config.fallback_ticker and fallback_allocation > 0 and config.fallback_ticker in target_weights.index:
            target_weights.loc[config.fallback_ticker] = fallback_allocation

        turnover = float(np.abs(target_weights - current_weights).sum())
        next_dt = rebalance_dates[index + 1] if index + 1 < len(rebalance_dates) else dates[-1]
        mask = (dates >= dt) & (dates < next_dt) if next_dt != dates[-1] else (dates >= dt) & (dates <= next_dt)
        if weights is not None:
            weights.loc[mask] = target_weights.values

        trade_rows.append({
            "date": dt,
            "selected": selected.copy(),
            "raw_scores": {ticker: float(today_scores.loc[ticker]) for ticker in selected},
            "adjusted_scores": {ticker: float(adjusted_scores.loc[ticker]) for ticker in selected},
            "turnover": turnover,
            "entered": entered,
            "exited": exited,
            "selected_score_sum": selected_score_sum,
            "stock_allocation": stock_allocation,
            "fallback_allocation": fallback_allocation,
        })

        current_weights = target_weights.copy()
        previous_reward_factors = {name: factor.loc[dt].copy() for name, factor in reward_factors.items()}
        previous_penalty_factors = {name: factor.loc[dt].copy() for name, factor in penalty_factors.items()}
        previous_prices = current_prices.copy()
        previous_holdings = selected.copy()

    if weights is None:
        weights = pd.DataFrame(0.0, index=dates, columns=tradable_columns)

    shifted_weights = weights.shift(1).fillna(0.0)
    gross_returns = (shifted_weights * tradable_returns).sum(axis=1)
    cost_series = pd.Series(0.0, index=dates)
    trade_cost = float(config.transaction_cost_bps) / 10000.0
    for row in trade_rows:
        cost_series.loc[row["date"]] = float(row["turnover"]) * trade_cost
    net_returns = gross_returns - cost_series
    equity = (1 + net_returns).cumprod()

    return {
        "weights": weights,
        "trade_log": pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(),
        "gross_returns": gross_returns,
        "cost_series": cost_series,
        "net_returns": net_returns,
        "equity_curve": pd.DataFrame({
            "gross_return": gross_returns,
            "cost": cost_series,
            "net_return": net_returns,
            "equity": equity,
        }),
    }

def run_weighted_points_backtest(
    universe_prices: pd.DataFrame,
    returns: pd.DataFrame,
    raw_scores: pd.DataFrame,
    config: Gamma1Config,
    *,
    record_weights: bool = True,
    initial_holdings: list[str] | None = None,
    initial_weights: pd.Series | None = None,
) -> dict[str, Any]:
    benchmark_prices = returns.mean(axis=1)
    if isinstance(raw_scores, pd.DataFrame) and "_benchmark" in raw_scores.columns:
        benchmark_prices = raw_scores["_benchmark"]
    return _run_adaptive_backtest(
        universe_prices,
        benchmark_prices,
        config,
        record_weights=record_weights,
        initial_holdings=initial_holdings,
        initial_weights=initial_weights,
    )


def _final_target_weights_from_state(
    universe_prices: pd.DataFrame,
    raw_scores: pd.DataFrame,
    config: Gamma1Config,
    *,
    initial_holdings: list[str] | None = None,
    initial_weights: pd.Series | None = None,
) -> tuple[dict[str, float], pd.Timestamp | None, str]:
    benchmark_prices = raw_scores["_benchmark"] if "_benchmark" in raw_scores.columns else universe_prices.mean(axis=1)
    result = _run_adaptive_backtest(
        universe_prices,
        benchmark_prices,
        config,
        record_weights=True,
        initial_holdings=initial_holdings,
        initial_weights=initial_weights,
    )
    trade_log = result["trade_log"]
    weights = result["weights"]
    if trade_log.empty or weights is None or weights.empty:
        return {}, None, "no_rebalance_dates"
    asof = pd.Timestamp(trade_log.iloc[-1]["date"])
    final_weights = weights.loc[asof]
    target_weights = {str(symbol): float(weight) for symbol, weight in final_weights.items() if float(weight) > 1e-12}
    return target_weights, asof, "ok"


def run_gamma1(bot_id: str, use_db_config: bool = True) -> dict:
    ts = datetime.now(timezone.utc)
    config = _build_live_config(bot_id, use_db_config=use_db_config)

    try:
        tickers = list(dict.fromkeys(list(config.universe) + [config.benchmark] + ([config.fallback_ticker] if config.fallback_ticker else [])))
        start_date, end_date = _resolve_download_window(config)
        data = download_gamma1_prices(tickers, start_date, end_date)

        if data.empty or config.benchmark not in data.columns:
            fb = config.fallback_ticker or DEFAULT_BENCHMARK
            return {
                "bot_id": bot_id,
                "bot_type": BOT_TYPE_GAMMA1,
                "ts": ts,
                "signal": "HOLD",
                "note": f"Insufficient price data; holding {fb}",
                "payload": {"asof": None, "benchmark": config.benchmark, "target_weights": {fb: 1.0}},
            }

        universe = [ticker for ticker in config.universe if ticker in data.columns]
        if not universe:
            fb = config.fallback_ticker or DEFAULT_BENCHMARK
            return {
                "bot_id": bot_id,
                "bot_type": BOT_TYPE_GAMMA1,
                "ts": ts,
                "signal": "HOLD",
                "note": f"No universe symbols in download; holding {fb}",
                "payload": {"asof": str(data.index[-1]), "benchmark": config.benchmark, "target_weights": {fb: 1.0}},
            }

        prices = data[universe].copy().dropna(how="all").sort_index()
        benchmark_prices = data[config.benchmark].reindex(prices.index)
        score_table = build_gamma1_score_table(prices, benchmark_prices, config.feature_flags)
        raw_scores = score_table["total"].copy()
        raw_scores["_benchmark"] = benchmark_prices

        target_weights, asof, status = _final_target_weights_from_state(
            prices,
            raw_scores,
            config,
        )
        if not target_weights and config.fallback_ticker:
            target_weights = {config.fallback_ticker: 1.0}

        note = f"As of {asof}: gamma1 adaptive rebalance ({status})" if asof else f"gamma1 adaptive: {status}"
        signal = "REBALANCE" if target_weights else "HOLD"
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_GAMMA1,
            "ts": ts,
            "signal": signal,
            "note": note,
            "payload": {
                "asof": str(asof) if asof else str(prices.index[-1]),
                "interval": config.rebalance,
                "benchmark": config.benchmark,
                "max_positions": config.max_positions,
                "entry_score": config.entry_score,
                "exit_score": config.exit_score,
                "transaction_cost_bps": config.transaction_cost_bps,
                "hold_bonus": config.hold_bonus,
                "target_weights": target_weights,
            },
        }
    except Exception as exc:
        fb = config.fallback_ticker or DEFAULT_BENCHMARK
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_GAMMA1,
            "ts": ts,
            "signal": "HOLD",
            "note": f"gamma1 error: {exc}",
            "payload": {"benchmark": config.benchmark, "target_weights": {fb: 1.0}},
        }
