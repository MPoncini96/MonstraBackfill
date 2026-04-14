# bots/gamma1.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from bot_identity import BOT_TYPE_GAMMA1

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
DEFAULT_ENTRY_SCORE = 7.0
DEFAULT_EXIT_SCORE = 4.0
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_HOLD_BONUS = 1.0
DEFAULT_HISTORY_DAYS = 900


@dataclass
class Gamma1FeatureFlags:
    """Feature toggles matching trading.gamma1 columns (scoring weights are fixed by design)."""

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
    weighting_style: str = "equal"
    feature_flags: Gamma1FeatureFlags = field(default_factory=Gamma1FeatureFlags)
    use_sector_cap: bool = False
    sector_cap_limit: int | None = None
    use_volatility_filter: bool = False


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
    """Map DB values (e.g. weekly) to pandas resample offsets."""
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
        v = int(value)
        return max(1, v)
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
    """Build runtime config from a trading.gamma1 row (dict)."""
    uni = _normalize_universe(d.get("universe")) or list(DEFAULT_UNIVERSE)
    fb = d.get("fallback_ticker")
    fb = str(fb).strip().upper() if fb else None
    cap = d.get("sector_cap_limit")
    cap_i: int | None = None
    if cap is not None:
        try:
            cap_i = int(cap)
        except (TypeError, ValueError):
            cap_i = None

    hp = d.get("history_period")
    history_period = str(hp) if hp is not None and str(hp).strip() else None

    return Gamma1Config(
        universe=uni,
        benchmark=_clean_ticker(d.get("benchmark"), DEFAULT_BENCHMARK),
        rebalance=normalize_rebalance_rule(d.get("rebalance_frequency") or d.get("rebalance")),
        max_positions=_safe_int(d.get("max_positions"), DEFAULT_MAX_POSITIONS),
        entry_score=_safe_float(d.get("entry_score"), DEFAULT_ENTRY_SCORE),
        exit_score=_safe_float(d.get("exit_score"), DEFAULT_EXIT_SCORE),
        transaction_cost_bps=_safe_float(d.get("transaction_cost_bps"), DEFAULT_TRANSACTION_COST_BPS),
        hold_bonus=_safe_float(d.get("hold_bonus"), DEFAULT_HOLD_BONUS),
        fallback_ticker=fb,
        history_period=history_period,
        weighting_style=str(d.get("weighting_style") or "equal").strip().lower() or "equal",
        feature_flags=gamma1_feature_flags_from_row(d),
        use_sector_cap=_safe_bool(d.get("use_sector_cap"), False),
        sector_cap_limit=cap_i,
        use_volatility_filter=_safe_bool(d.get("use_volatility_filter"), False),
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

    unit = token[-1]
    multipliers = {"d": 1, "w": 7, "m": 30, "y": 365}
    return qty * multipliers.get(unit, 0)


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


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_out = 100 - (100 / (1 + rs))
    return rsi_out.fillna(50)


def build_gamma1_score_table(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    feature_flags: Gamma1FeatureFlags | None = None,
) -> dict[str, pd.DataFrame]:
    px = prices.copy()
    bm = benchmark_prices.copy()
    ff = feature_flags or Gamma1FeatureFlags()

    ema10 = px.ewm(span=10, adjust=False, min_periods=10).mean()
    ema20 = px.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = px.ewm(span=50, adjust=False, min_periods=50).mean()

    ret10 = px / px.shift(10) - 1.0
    ret20 = px / px.shift(20) - 1.0

    bm_ret20 = bm / bm.shift(20) - 1.0
    bm_above_ema50 = bm > bm.ewm(span=50, adjust=False, min_periods=50).mean()

    rsi14 = px.apply(rsi, axis=0)

    high_252 = px.rolling(252, min_periods=50).max()
    within_10_of_high = px >= 0.90 * high_252

    rets = px.pct_change().fillna(0.0)
    vol5 = rets.rolling(5).std()
    vol20 = rets.rolling(20).std()
    avgvol5 = px.pct_change().abs().rolling(5).mean()
    avgvol20 = px.pct_change().abs().rolling(20).mean()

    s1 = (ret10 > 0).astype(int) * 1 * int(bool(ff.use_ret10_positive))
    s2 = (ret20 > 0).astype(int) * 2 * int(bool(ff.use_ret20_positive))
    s3 = ret20.gt(bm_ret20, axis=0).astype(int) * 2 * int(bool(ff.use_ret20_vs_benchmark))
    s4 = (px > ema20).astype(int) * 1 * int(bool(ff.use_price_above_ema20))
    s5 = (ema10 > ema20).astype(int) * 1 * int(bool(ff.use_ema10_above_ema20))
    s6 = (ema20 > ema50).astype(int) * 1 * int(bool(ff.use_ema20_above_ema50))
    s7 = within_10_of_high.astype(int) * 2 * int(bool(ff.use_within_10pct_high))
    s8 = ((rsi14 >= 55) & (rsi14 <= 70)).astype(int) * 1 * int(bool(ff.use_rsi_55_70))

    positive = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8

    p1 = (ret10 < 0).astype(int) * 2 * int(bool(ff.use_penalty_ret10_negative))
    p2 = (px < ema50).astype(int) * 2 * int(bool(ff.use_penalty_price_below_ema50))
    p3 = (rsi14 > 80).astype(int) * 1 * int(bool(ff.use_penalty_rsi_above_80))
    p4 = (px < 0.85 * high_252).astype(int) * 1 * int(bool(ff.use_penalty_far_from_high))
    p5 = (vol5 > vol20 * 1.5).astype(int) * 1 * int(bool(ff.use_penalty_volatility_spike))
    p6 = (avgvol5 < avgvol20).astype(int) * 1 * int(bool(ff.use_penalty_fading_activity))

    penalty = p1 + p2 + p3 + p4 + p5 + p6

    total_score = positive - penalty
    market_penalty = (~bm_above_ema50).astype(int) * 2 * int(bool(ff.use_market_regime_penalty))
    total_score = total_score.sub(market_penalty, axis=0)

    return {
        "total": total_score,
        "positive": positive,
        "penalty": penalty,
        "ret10": ret10,
        "ret20": ret20,
        "ema10": ema10,
        "ema20": ema20,
        "ema50": ema50,
        "rsi14": rsi14,
    }


def get_gamma1_rebalance_dates(prices_index: pd.DatetimeIndex, rebalance_rule: str) -> list[pd.Timestamp]:
    stub = pd.DataFrame(index=prices_index, data={"_": 0.0})
    rebal = stub.resample(rebalance_rule).last().index
    return [d for d in rebal if d in prices_index]


def download_gamma1_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers=tickers,
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
        first = tickers[0]
        prices = raw[["Close"]].rename(columns={"Close": first})

    prices = prices.dropna(how="all").sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices


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
    """
    Weighted binary-score backtest: weekly (or other) rebalance with hysteresis.
    """
    dates = universe_prices.index
    rebal_dates = get_gamma1_rebalance_dates(dates, config.rebalance)

    cols = universe_prices.columns
    weights = (
        pd.DataFrame(0.0, index=dates, columns=cols)
        if record_weights
        else None
    )
    trade_rows: list[dict[str, Any]] = []

    current_holdings = [t for t in (initial_holdings or []) if t in cols]
    if initial_weights is not None:
        current_weights = initial_weights.reindex(cols).fillna(0.0)
    else:
        current_weights = pd.Series(0.0, index=cols)

    if not current_holdings and initial_weights is not None:
        current_holdings = [
            str(t) for t in cols if float(current_weights.get(t, 0.0)) > 1e-12
        ]

    max_pos = int(config.max_positions)

    for i, dt in enumerate(rebal_dates):
        today_raw = raw_scores.loc[dt].dropna().copy()

        kept_holdings: list[str] = []
        exited: list[str] = []

        for ticker in current_holdings:
            score = today_raw.get(ticker, np.nan)
            if pd.notna(score) and score >= config.exit_score:
                kept_holdings.append(ticker)
            else:
                exited.append(ticker)

        current_holdings = kept_holdings

        adjusted_scores = today_raw.copy()
        for ticker in current_holdings:
            if ticker in adjusted_scores.index:
                adjusted_scores.loc[ticker] += config.hold_bonus

        candidates = adjusted_scores.sort_values(ascending=False)

        entered: list[str] = []
        for ticker in candidates.index:
            if ticker in current_holdings:
                continue
            if adjusted_scores.loc[ticker] < config.entry_score:
                continue
            if len(current_holdings) >= max_pos:
                break
            current_holdings.append(ticker)
            entered.append(ticker)

        target_weights = pd.Series(0.0, index=cols)
        if current_holdings:
            target_weights.loc[current_holdings] = 1.0 / len(current_holdings)

        turnover = float(np.abs(target_weights - current_weights).sum())

        next_dt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        if next_dt != dates[-1]:
            mask = (dates >= dt) & (dates < next_dt)
        else:
            mask = (dates >= dt) & (dates <= next_dt)

        if weights is not None:
            weights.loc[mask] = target_weights.values

        trade_rows.append(
            {
                "date": dt,
                "selected": current_holdings.copy(),
                "raw_scores": {
                    t: float(today_raw[t])
                    for t in current_holdings
                    if t in today_raw.index
                },
                "adjusted_scores": {
                    t: float(adjusted_scores[t])
                    for t in current_holdings
                    if t in adjusted_scores.index
                },
                "turnover": turnover,
                "entered": entered,
                "exited": exited,
            }
        )

        current_weights = target_weights.copy()

    shifted_w = (
        weights.shift(1).fillna(0.0) if weights is not None else pd.DataFrame(0.0, index=dates, columns=cols)
    )
    gross_returns = (shifted_w * returns).sum(axis=1)

    cost_series = pd.Series(0.0, index=dates)
    tc = float(config.transaction_cost_bps) / 10000.0
    for row in trade_rows:
        rdt = row["date"]
        cost_series.loc[rdt] = float(row["turnover"]) * tc

    net_returns = gross_returns - cost_series
    equity = (1 + net_returns).cumprod()

    return {
        "weights": weights,
        "trade_log": pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(),
        "gross_returns": gross_returns,
        "cost_series": cost_series,
        "net_returns": net_returns,
        "equity_curve": pd.DataFrame(
            {
                "gross_return": gross_returns,
                "cost": cost_series,
                "net_return": net_returns,
                "equity": equity,
            }
        ),
    }


def _final_target_weights_from_state(
    universe_prices: pd.DataFrame,
    raw_scores: pd.DataFrame,
    config: Gamma1Config,
    *,
    initial_holdings: list[str] | None = None,
    initial_weights: pd.Series | None = None,
) -> tuple[dict[str, float], pd.Timestamp | None, str]:
    dates = universe_prices.index
    cols = universe_prices.columns
    rebal_dates = get_gamma1_rebalance_dates(dates, config.rebalance)
    if not rebal_dates:
        return {}, None, "no_rebalance_dates"

    current_holdings = [t for t in (initial_holdings or []) if t in cols]
    if initial_weights is not None:
        current_weights = initial_weights.reindex(cols).fillna(0.0)
    else:
        current_weights = pd.Series(0.0, index=cols)

    if not current_holdings and initial_weights is not None:
        current_holdings = [
            str(t) for t in cols if float(current_weights.get(t, 0.0)) > 1e-12
        ]

    max_pos = int(config.max_positions)

    for i, dt in enumerate(rebal_dates):
        today_raw = raw_scores.loc[dt].dropna().copy()

        kept_holdings: list[str] = []
        for ticker in current_holdings:
            score = today_raw.get(ticker, np.nan)
            if pd.notna(score) and score >= config.exit_score:
                kept_holdings.append(ticker)

        current_holdings = kept_holdings

        adjusted_scores = today_raw.copy()
        for ticker in current_holdings:
            if ticker in adjusted_scores.index:
                adjusted_scores.loc[ticker] += config.hold_bonus

        candidates = adjusted_scores.sort_values(ascending=False)

        for ticker in candidates.index:
            if ticker in current_holdings:
                continue
            if adjusted_scores.loc[ticker] < config.entry_score:
                continue
            if len(current_holdings) >= max_pos:
                break
            current_holdings.append(ticker)

        target_weights = pd.Series(0.0, index=cols)
        if current_holdings:
            target_weights.loc[current_holdings] = 1.0 / len(current_holdings)

        current_weights = target_weights.copy()

        if i == len(rebal_dates) - 1:
            asof = dt
            tw = {s: float(w) for s, w in target_weights.items() if w > 1e-12}
            return tw, asof, "ok"

    return {}, None, "no_iterations"


def run_gamma1(bot_id: str, use_db_config: bool = True) -> dict:
    """Live gamma1: weighted points + hysteresis, parameters from trading.gamma1."""
    ts = datetime.now(timezone.utc)
    config = _build_live_config(bot_id, use_db_config=use_db_config)

    try:
        tickers = list(dict.fromkeys(list(config.universe) + [config.benchmark]))
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
                "payload": {
                    "asof": None,
                    "benchmark": config.benchmark,
                    "target_weights": {fb: 1.0},
                },
            }

        univ = [t for t in config.universe if t in data.columns]
        if not univ:
            fb = config.fallback_ticker or DEFAULT_BENCHMARK
            return {
                "bot_id": bot_id,
                "bot_type": BOT_TYPE_GAMMA1,
                "ts": ts,
                "signal": "HOLD",
                "note": f"No universe symbols in download; holding {fb}",
                "payload": {
                    "asof": str(data.index[-1]),
                    "benchmark": config.benchmark,
                    "target_weights": {fb: 1.0},
                },
            }

        prices = data[univ].copy().dropna(how="all")
        bm = data[config.benchmark].copy()
        prices = prices.dropna(how="all").sort_index()
        bm = bm.reindex(prices.index)

        if len(prices.index) < 5:
            fb = config.fallback_ticker or DEFAULT_BENCHMARK
            return {
                "bot_id": bot_id,
                "bot_type": BOT_TYPE_GAMMA1,
                "ts": ts,
                "signal": "HOLD",
                "note": "Not enough history",
                "payload": {
                    "asof": str(prices.index[-1]) if len(prices.index) else None,
                    "benchmark": config.benchmark,
                    "target_weights": {fb: 1.0},
                },
            }

        score_table = build_gamma1_score_table(prices, bm, config.feature_flags)
        raw_scores = score_table["total"]

        target_w, asof, status = _final_target_weights_from_state(
            prices,
            raw_scores,
            config,
        )

        if not target_w and config.fallback_ticker:
            target_w = {config.fallback_ticker: 1.0}

        note = f"As of {asof}: gamma1 rebalance ({status})" if asof else f"gamma1: {status}"
        signal = "REBALANCE" if target_w else "HOLD"

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
                "target_weights": target_w,
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
            "payload": {
                "benchmark": config.benchmark,
                "target_weights": {fb: 1.0},
            },
        }
