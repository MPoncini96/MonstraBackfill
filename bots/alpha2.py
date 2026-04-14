# bots/alpha2.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from bot_identity import BOT_TYPE_ALPHA2

DEFAULT_BENCH = "VOO"

DEFAULT_PROXIES = {
    "ai": "SMH",
    "tech": "XLK",
    "finance": "XLF",
    "energy": "XLE",
    "logistics": "IYT",
    "defense": "ITA",
}

DEFAULT_STOCKS = {
    "ai": ["NVDA", "AMD", "AVGO", "ASML", "TSM", "AMAT", "LRCX", "KLAC", "MU", "QCOM", "ARM", "MRVL"],
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRM", "ADBE", "INTU", "NOW", "CSCO", "IBM"],
    "finance": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB", "PGR", "AIG"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "KMI", "WMB", "OKE", "TRGP"],
    "logistics": ["UPS", "FDX", "UNP", "CSX", "NSC", "CP", "CNI", "JBHT", "ODFL", "XPO", "SAIA", "KNX"],
    "defense": ["LMT", "NOC", "RTX", "GD", "BA", "LHX", "HII", "TDG", "HEI", "TXT", "BWXT", "KTOS"],
}

DEFAULT_RANK_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_HISTORY_PERIOD = "12mo"
DEFAULT_REBALANCE_FREQ = 5
DEFAULT_STOCK_BENCH_MODE = "sector"
DEFAULT_CASH_EQUIVALENT = DEFAULT_BENCH
DEFAULT_NUM_TRIGGERED_SECTORS = 1
DEFAULT_NUM_STOCK_PICKS = 4

REQUIRE_FULL_LOOKBACK_WINDOW = True
USE_CASH_EQUIVALENT_FALLBACK = True

DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass
class Alpha2Config:
    bot_id: str
    proxies: dict[str, str]
    stocks: dict[str, list[str]]
    lookback_days: int
    rebalance_freq: str | int | None
    stock_bench_mode: str | None
    rank_weights: np.ndarray
    cash_equivalent: str | None = None
    benchmark: str | None = None
    enable_kill_switch: bool = True
    enable_benchmark_filter: bool = False
    kill_switch_mode: str = "top_negative"
    benchmark_mode: str = "benchmark_positive"
    top_return_threshold: float = 0.0
    benchmark_return_threshold: float = 0.0
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    num_triggered_sectors: int = DEFAULT_NUM_TRIGGERED_SECTORS
    num_stock_picks: int = DEFAULT_NUM_STOCK_PICKS


@dataclass
class Alpha2UniversePick:
    symbols: list[str]
    weights: np.ndarray
    risk_off: bool
    risk_reason: str
    top_sectors: list[str]
    selected_stocks: list[str]


def normalize_proxy_map(raw: Any) -> dict[str, str]:
    if raw is None or not isinstance(raw, dict):
        return {}

    result: dict[str, str] = {}
    for key, value in raw.items():
        logical = str(key).strip()
        ticker = clean_ticker(value)
        if logical and ticker:
            result[logical] = ticker
    return result


def normalize_stocks_map(raw: Any) -> dict[str, list[str]]:
    if raw is None or not isinstance(raw, dict):
        return {}

    result: dict[str, list[str]] = {}
    for key, value in raw.items():
        logical = str(key).strip()
        if not logical:
            continue

        if isinstance(value, list):
            seen: set[str] = set()
            tickers: list[str] = []
            for item in value:
                ticker = clean_ticker(item)
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)
            result[logical] = tickers
        else:
            ticker = clean_ticker(value)
            result[logical] = [ticker] if ticker else []
    return result


def normalize_rank_weights(raw: Any) -> np.ndarray:
    if raw is None or not isinstance(raw, (list, tuple)):
        return DEFAULT_RANK_WEIGHTS.copy()

    arr = np.array(raw, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]

    if arr.size == 0 or arr.sum() <= 0:
        return DEFAULT_RANK_WEIGHTS.copy()

    return arr / arr.sum()


def clean_ticker(value: Any) -> str | None:
    if value is None:
        return None
    ticker = str(value).strip().upper()
    return ticker or None


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_bool(value: Any, default: bool) -> bool:
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


def parse_rebalance_freq(value: Any) -> int:
    if value is None:
        return 1

    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)

    token = str(value).strip().lower()
    mapping = {
        "daily": 1,
        "every_day": 1,
        "weekly": 5,
        "every_week": 5,
        "monthly": 21,
        "every_month": 21,
        "w-fri": 5,
        "d": 1,
    }
    if token in mapping:
        return mapping[token]

    try:
        return max(1, int(token))
    except ValueError:
        return 1


def parse_period_days(history_period: str | None) -> int:
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


def build_live_config(bot_id: str, use_db_config: bool = True) -> tuple[Alpha2Config, str | None]:
    config_data: dict[str, Any] | None = None
    history_period: str | None = None

    if use_db_config:
        try:
            from db import get_alpha2_config

            config_data = get_alpha2_config(bot_id)
        except Exception as exc:
            print(f"Warning: Could not load {bot_id} config from database, using defaults: {exc}")

    if config_data:
        history_period = config_data.get("history_period")
        config = Alpha2Config(
            bot_id=bot_id,
            proxies=normalize_proxy_map(config_data.get("proxies")) or dict(DEFAULT_PROXIES),
            stocks=normalize_stocks_map(config_data.get("stocks")) or dict(DEFAULT_STOCKS),
            lookback_days=int(config_data.get("lookback_days", DEFAULT_LOOKBACK_DAYS)),
            rebalance_freq=config_data.get("rebalance_freq", DEFAULT_REBALANCE_FREQ),
            stock_bench_mode=str(config_data.get("stock_bench_mode", DEFAULT_STOCK_BENCH_MODE)),
            rank_weights=normalize_rank_weights(config_data.get("rank_weights")),
            cash_equivalent=clean_ticker(config_data.get("cash_equivalent", DEFAULT_CASH_EQUIVALENT)),
            benchmark=clean_ticker(config_data.get("benchmark", DEFAULT_BENCH)),
            enable_kill_switch=safe_bool(config_data.get("enable_kill_switch"), True),
            enable_benchmark_filter=safe_bool(config_data.get("enable_benchmark_filter"), False),
            kill_switch_mode=str(config_data.get("kill_switch_mode", "top_negative")),
            benchmark_mode=str(config_data.get("benchmark_mode", "benchmark_positive")),
            top_return_threshold=safe_float(config_data.get("top_return_threshold"), 0.0),
            benchmark_return_threshold=safe_float(config_data.get("benchmark_return_threshold"), 0.0),
            transaction_cost_bps=safe_float(config_data.get("transaction_cost_bps"), DEFAULT_TRANSACTION_COST_BPS),
            slippage_bps=safe_float(config_data.get("slippage_bps"), DEFAULT_SLIPPAGE_BPS),
            num_triggered_sectors=max(1, safe_int(config_data.get("num_triggered_sectors"), DEFAULT_NUM_TRIGGERED_SECTORS)),
            num_stock_picks=max(1, safe_int(config_data.get("num_stock_picks"), DEFAULT_NUM_STOCK_PICKS)),
        )
    else:
        config = Alpha2Config(
            bot_id=bot_id,
            proxies=dict(DEFAULT_PROXIES),
            stocks=dict(DEFAULT_STOCKS),
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            rebalance_freq=DEFAULT_REBALANCE_FREQ,
            stock_bench_mode=DEFAULT_STOCK_BENCH_MODE,
            rank_weights=DEFAULT_RANK_WEIGHTS.copy(),
            cash_equivalent=DEFAULT_CASH_EQUIVALENT,
            benchmark=DEFAULT_BENCH,
            num_triggered_sectors=DEFAULT_NUM_TRIGGERED_SECTORS,
            num_stock_picks=DEFAULT_NUM_STOCK_PICKS,
        )

    if config.lookback_days < 1:
        config.lookback_days = DEFAULT_LOOKBACK_DAYS

    return config, history_period


def resolve_download_window(lookback_days: int, history_period: str | None) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    configured_days = parse_period_days(history_period)
    buffer_days = max(lookback_days * 8, configured_days, 252)
    start_date = today - timedelta(days=buffer_days)
    end_date = today + timedelta(days=1)
    return start_date.isoformat(), end_date.isoformat()


def download_adjusted_close(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
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
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices


def collect_universe_symbols(config: Alpha2Config) -> list[str]:
    """Proxies, all configured stocks, benchmark, and cash equivalent (deduped, stable order)."""
    symbols: list[str] = []
    seen: set[str] = set()
    for t in config.proxies.values():
        ct = clean_ticker(t)
        if ct and ct not in seen:
            seen.add(ct)
            symbols.append(ct)
    for tick_list in config.stocks.values():
        for raw in tick_list:
            ct = clean_ticker(raw)
            if ct and ct not in seen:
                seen.add(ct)
                symbols.append(ct)
    for sym in (config.cash_equivalent, config.benchmark):
        ct = clean_ticker(sym)
        if ct and ct not in seen:
            seen.add(ct)
            symbols.append(ct)
    return symbols


def build_universe_price_frame(config: Alpha2Config, start_date: str, end_date: str) -> pd.DataFrame:
    symbols = collect_universe_symbols(config)
    return download_adjusted_close(symbols, start_date, end_date)


def get_trailing_returns(
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

    if symbols:
        trailing = trailing[trailing.index.isin(symbols)]

    return trailing


def evaluate_risk_off(
    config: Alpha2Config,
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


def choose_universe_triggered_holdings(
    config: Alpha2Config,
    prices: pd.DataFrame,
    end_idx_exclusive: int,
) -> Alpha2UniversePick:
    """
    Rank sectors by proxy ETF trailing return; take top ``num_triggered_sectors``;
    rank stocks in those sectors by trailing return; equal-weight top ``num_stock_picks``.
    Mirrors QuantConnect Alpha2UniverseTriggeredQC rebalance logic.
    """
    proxy_symbols = set(config.proxies.values())
    benchmark_symbols = {config.benchmark} if config.benchmark else set()
    cash_symbols = {config.cash_equivalent} if config.cash_equivalent else set()

    proxy_trailing = get_trailing_returns(
        prices=prices,
        end_idx_exclusive=end_idx_exclusive,
        lookback_days=config.lookback_days,
        symbols=proxy_symbols,
    )
    benchmark_trailing = get_trailing_returns(
        prices=prices,
        end_idx_exclusive=end_idx_exclusive,
        lookback_days=config.lookback_days,
        symbols=benchmark_symbols,
    )
    cash_trailing = get_trailing_returns(
        prices=prices,
        end_idx_exclusive=end_idx_exclusive,
        lookback_days=config.lookback_days,
        symbols=cash_symbols,
    )

    risk_off, risk_reason = evaluate_risk_off(
        config=config,
        ranked_trailing=proxy_trailing,
        benchmark_trailing=benchmark_trailing,
        cash_trailing=cash_trailing,
    )

    fallback_ticker = config.cash_equivalent or config.benchmark or DEFAULT_BENCH

    if risk_off:
        if USE_CASH_EQUIVALENT_FALLBACK and fallback_ticker:
            w = np.array([1.0], dtype=float)
            return Alpha2UniversePick(
                symbols=[fallback_ticker],
                weights=w,
                risk_off=True,
                risk_reason=risk_reason,
                top_sectors=[],
                selected_stocks=[],
            )
        return Alpha2UniversePick(
            symbols=[],
            weights=np.array([], dtype=float),
            risk_off=True,
            risk_reason=risk_reason,
            top_sectors=[],
            selected_stocks=[],
        )

    ranked_sectors: list[tuple[str, float]] = []
    for logical_sector, proxy_ticker in config.proxies.items():
        pt = clean_ticker(proxy_ticker)
        if pt and pt in proxy_trailing.index:
            ranked_sectors.append((logical_sector, float(proxy_trailing[pt])))

    ranked_sectors.sort(key=lambda x: x[1], reverse=True)
    n_sec = max(1, int(config.num_triggered_sectors))
    top_sectors = [s for s, _ in ranked_sectors[:n_sec]]

    if not top_sectors:
        reason = "fallback:no_selected_sectors"
        w = np.array([1.0], dtype=float)
        return Alpha2UniversePick(
            symbols=[fallback_ticker],
            weights=w,
            risk_off=True,
            risk_reason=reason,
            top_sectors=[],
            selected_stocks=[],
        )

    triggered: list[str] = []
    for sector in top_sectors:
        triggered.extend(config.stocks.get(sector, []))
    triggered = list(dict.fromkeys(triggered))

    stock_trailing = get_trailing_returns(
        prices=prices,
        end_idx_exclusive=end_idx_exclusive,
        lookback_days=config.lookback_days,
        symbols=set(triggered),
    )
    stock_trailing = stock_trailing.sort_values(ascending=False)

    n_pick = max(1, int(config.num_stock_picks))
    selected_stocks = stock_trailing.head(n_pick).index.tolist()

    if not selected_stocks:
        reason = "fallback:no_selected_stocks"
        w = np.array([1.0], dtype=float)
        return Alpha2UniversePick(
            symbols=[fallback_ticker],
            weights=w,
            risk_off=True,
            risk_reason=reason,
            top_sectors=top_sectors,
            selected_stocks=[],
        )

    w = np.full(len(selected_stocks), 1.0 / len(selected_stocks), dtype=float)
    return Alpha2UniversePick(
        symbols=selected_stocks,
        weights=w,
        risk_off=False,
        risk_reason=risk_reason,
        top_sectors=top_sectors,
        selected_stocks=selected_stocks,
    )


def compute_day_return_and_holdings(
    day_returns: pd.Series,
    selected_symbols: list[str],
    weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    if not selected_symbols or len(weights) == 0:
        return 0.0, {}

    valid_pairs: list[tuple[str, float]] = []
    for sym, weight in zip(selected_symbols, weights):
        if sym in day_returns.index and pd.notna(day_returns[sym]):
            valid_pairs.append((sym, float(weight)))

    if not valid_pairs:
        return 0.0, {}

    total_weight = sum(weight for _, weight in valid_pairs)
    if total_weight <= 0:
        return 0.0, {}

    normalized_pairs = [(sym, weight / total_weight) for sym, weight in valid_pairs]
    holdings = {sym: float(weight) for sym, weight in normalized_pairs}
    gross_ret = sum(day_returns[sym] * weight for sym, weight in normalized_pairs)
    return float(gross_ret), holdings


def is_rebalance_day(day_counter: int, rebalance_every_n_days: int) -> bool:
    if rebalance_every_n_days <= 1:
        return True
    return (day_counter % rebalance_every_n_days) == 0


def run_alpha2(
    bot_id: str,
    lookback_days: int | None = None,
    history_period: str | None = None,
    rebalance_freq: str | None = None,
    stock_bench_mode: str | None = None,
    use_db_config: bool = True,
) -> dict:
    """
    Live Alpha2 signal generator using the same daily proxy-ranking rules as backfill_alpha2.
    """
    ts = datetime.now(timezone.utc)
    config, history_period_cfg = build_live_config(bot_id, use_db_config=use_db_config)

    if lookback_days is not None:
        config.lookback_days = max(1, int(lookback_days))
    if rebalance_freq is not None:
        config.rebalance_freq = rebalance_freq
    if stock_bench_mode is not None:
        config.stock_bench_mode = stock_bench_mode

    history_period = history_period if history_period is not None else history_period_cfg
    start_date, end_date = resolve_download_window(config.lookback_days, history_period)
    prices = build_universe_price_frame(config, start_date, end_date)

    if prices.empty or len(prices.index) < 2:
        fallback = config.cash_equivalent or config.benchmark or DEFAULT_BENCH
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_ALPHA2,
            "ts": ts,
            "signal": "HOLD",
            "note": f"Not enough 1d history; holding {fallback}",
            "payload": {
                "asof": str(prices.index[-1]) if not prices.empty else None,
                "interval": "1d",
                "lookback_days": config.lookback_days,
                "rebalance_freq": config.rebalance_freq,
                "target_weights": {fallback: 1.0},
            },
        }

    if len(prices.index) < config.lookback_days + 2:
        fallback = config.cash_equivalent or config.benchmark or DEFAULT_BENCH
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_ALPHA2,
            "ts": ts,
            "signal": "HOLD",
            "note": f"Not enough lookback window (need {config.lookback_days + 2} days); holding {fallback}",
            "payload": {
                "asof": str(prices.index[-1]),
                "interval": "1d",
                "lookback_days": config.lookback_days,
                "rebalance_freq": config.rebalance_freq,
                "target_weights": {fallback: 1.0},
            },
        }

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rebalance_every_n_days = parse_rebalance_freq(config.rebalance_freq)
    strategy_day_counter = len(prices.index) - 1
    asof = prices.index[-1]
    prev_date = prices.index[-2] if len(prices.index) >= 2 else prices.index[-1]

    rebalanced_today = is_rebalance_day(strategy_day_counter - 1, rebalance_every_n_days)
    pick: Alpha2UniversePick | None = None

    if rebalanced_today:
        pick = choose_universe_triggered_holdings(
            config=config,
            prices=prices,
            end_idx_exclusive=len(prices.index) - 1,
        )

        risk_off = pick.risk_off
        risk_reason = pick.risk_reason

        _, target_holdings = compute_day_return_and_holdings(
            day_returns=returns.loc[asof],
            selected_symbols=pick.symbols,
            weights=pick.weights,
        )

        if not target_holdings:
            fallback = config.cash_equivalent or config.benchmark or DEFAULT_BENCH
            target_holdings = {fallback: 1.0}

        signal = "REBALANCE"
        note = (
            f"As of {asof.date()}: rebalance sectors={pick.top_sectors} "
            f"stocks={pick.selected_stocks} risk_reason={risk_reason} "
            f"holdings={', '.join(target_holdings.keys())}"
        )
    else:
        risk_off = False
        risk_reason = "carry_forward"
        target_holdings = {}
        signal = "HOLD"
        note = f"As of {asof.date()}: carry forward prior holdings (next rebalance in {rebalance_every_n_days}-day cycle)"

    payload = {
        "asof": str(asof),
        "prev_date": str(prev_date),
        "interval": "1d",
        "lookback_days": config.lookback_days,
        "rebalance_freq": config.rebalance_freq,
        "rebalanced_today": rebalanced_today,
        "rank_weights": [float(x) for x in config.rank_weights.tolist()],
        "num_triggered_sectors": config.num_triggered_sectors,
        "num_stock_picks": config.num_stock_picks,
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
        "mode": config.stock_bench_mode,
        "top_sectors": pick.top_sectors if pick else [],
        "selected_stocks": pick.selected_stocks if pick else [],
        "target_weights": target_holdings,
    }

    return {
        "bot_id": bot_id,
        "bot_type": BOT_TYPE_ALPHA2,
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": payload,
    }
