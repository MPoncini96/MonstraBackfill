# bots/echo1.py
"""
EchoNetworkPairs live signal runner.

Strategy: discovers lead-lag correlations across a configurable universe,
scores each follower by aggregating signals from all active leaders, then
allocates proportionally to the top-N picks.

Key design notes vs. the notebook:
- Price data comes from market_data_provider (Alpaca or yfinance), not raw yfinance.
- Monthly pair discovery is cached in trading.echo1_pairs_cache so the live
  runner doesn't re-run 15s discovery on every hourly tick.
- Market filter (VOO SMA-100) and drawdown guard (20% pause) are preserved.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from bot_identity import BOT_TYPE_ECHO1
from market_data_provider import get_daily_bars

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (mirror notebook constants)
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE = [
    "NVDA", "PLTR", "ANET", "AVGO", "TSM", "ASML",
    "META", "NFLX", "AMZN", "BKNG",
    "LLY", "ISRG",
    "GEV", "ETN", "CAT", "PH", "TT", "URI",
    "VST", "CEG",
    "XOM", "FCX",
    "IBKR", "JPM", "WMT",
]
DEFAULT_FALLBACK_TICKER      = "VOO"
DEFAULT_PORTFOLIO_SIZE       = 4
DEFAULT_GROSS_EXPOSURE       = 0.95
DEFAULT_DISCOVERY_LOOKBACK   = 504   # trading days (~2 years)
DEFAULT_MAX_PAIRS            = 75
DEFAULT_LAG_RANGE            = [3, 5, 7, 9]
DEFAULT_RETURN_PERIOD_RANGE  = [3, 5, 7, 9]
DEFAULT_MIN_AVG_ABS_CORR     = 0.20
DEFAULT_MIN_AVG_ABS_CORR_LOOSE = 0.12
DEFAULT_MIN_CONSISTENCY_SCORE = 0.60
DEFAULT_MIN_SAMPLES          = 40
DEFAULT_MIN_LEADER_MOVE      = 0.005
DEFAULT_SIGNAL_MULTIPLIER    = 2.0
DEFAULT_MAX_SINGLE_SIGNAL    = 0.50
DEFAULT_MAX_TOTAL_SCORE      = 2.00
DEFAULT_USE_MARKET_FILTER    = True
DEFAULT_MARKET_SMA_PERIOD    = 100
DEFAULT_MARKET_FILTER_FALLBACK_PCT = 0.50
DEFAULT_MAX_DRAWDOWN_PAUSE   = 0.20
DEFAULT_DRAWDOWN_RECOVERY_PCT = 0.60
DEFAULT_DRAWDOWN_COOLDOWN_DAYS = 10

# Extra calendar-day buffer added on top of DISCOVERY_LOOKBACK when fetching
# price history (accounts for weekends/holidays in trading-day counts).
_LOOKBACK_CALENDAR_BUFFER = 100


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Echo1Config:
    universe: list[str]
    fallback_ticker: str                        = DEFAULT_FALLBACK_TICKER
    portfolio_size: int                         = DEFAULT_PORTFOLIO_SIZE
    target_gross_exposure: float                = DEFAULT_GROSS_EXPOSURE
    discovery_lookback: int                     = DEFAULT_DISCOVERY_LOOKBACK
    max_pairs: int                              = DEFAULT_MAX_PAIRS
    lag_range: list[int]                        = field(default_factory=lambda: list(DEFAULT_LAG_RANGE))
    return_period_range: list[int]              = field(default_factory=lambda: list(DEFAULT_RETURN_PERIOD_RANGE))
    min_avg_abs_corr: float                     = DEFAULT_MIN_AVG_ABS_CORR
    min_avg_abs_corr_loose: float               = DEFAULT_MIN_AVG_ABS_CORR_LOOSE
    min_consistency_score: float                = DEFAULT_MIN_CONSISTENCY_SCORE
    min_samples: int                            = DEFAULT_MIN_SAMPLES
    min_leader_move: float                      = DEFAULT_MIN_LEADER_MOVE
    signal_multiplier: float                    = DEFAULT_SIGNAL_MULTIPLIER
    max_single_signal: float                    = DEFAULT_MAX_SINGLE_SIGNAL
    max_total_score: float                      = DEFAULT_MAX_TOTAL_SCORE
    use_market_filter: bool                     = DEFAULT_USE_MARKET_FILTER
    market_sma_period: int                      = DEFAULT_MARKET_SMA_PERIOD
    market_filter_fallback_pct: float           = DEFAULT_MARKET_FILTER_FALLBACK_PCT
    max_drawdown_pause: float                   = DEFAULT_MAX_DRAWDOWN_PAUSE
    drawdown_recovery_pct: float                = DEFAULT_DRAWDOWN_RECOVERY_PCT
    drawdown_cooldown_days: int                 = DEFAULT_DRAWDOWN_COOLDOWN_DAYS


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int) -> int:
    try:
        return max(1, int(v))
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


def _parse_int_list(v: Any, default: list[int]) -> list[int]:
    """Coerce a DB value (JSONB list, JSON string, or Python list) to list[int]."""
    if v is None:
        return list(default)
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return list(default)
    if isinstance(v, (list, tuple)):
        try:
            return [int(x) for x in v]
        except (TypeError, ValueError):
            pass
    return list(default)


def _normalize_universe(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    if not isinstance(v, (list, tuple)):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in v:
        t = str(item).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def echo1_config_from_db_dict(d: dict[str, Any]) -> Echo1Config:
    """Build a runtime Echo1Config from a trading.echo1 row dict."""
    uni = _normalize_universe(d.get("universe")) or list(DEFAULT_UNIVERSE)
    return Echo1Config(
        universe=uni,
        fallback_ticker=str(d.get("fallback_ticker") or DEFAULT_FALLBACK_TICKER).strip().upper(),
        portfolio_size=_safe_int(d.get("portfolio_size"), DEFAULT_PORTFOLIO_SIZE),
        target_gross_exposure=_safe_float(d.get("target_gross_exposure"), DEFAULT_GROSS_EXPOSURE),
        discovery_lookback=_safe_int(d.get("discovery_lookback"), DEFAULT_DISCOVERY_LOOKBACK),
        max_pairs=_safe_int(d.get("max_pairs"), DEFAULT_MAX_PAIRS),
        lag_range=_parse_int_list(d.get("lag_range"), DEFAULT_LAG_RANGE),
        return_period_range=_parse_int_list(d.get("return_period_range"), DEFAULT_RETURN_PERIOD_RANGE),
        min_avg_abs_corr=_safe_float(d.get("min_avg_abs_corr"), DEFAULT_MIN_AVG_ABS_CORR),
        min_avg_abs_corr_loose=_safe_float(d.get("min_avg_abs_corr_loose"), DEFAULT_MIN_AVG_ABS_CORR_LOOSE),
        min_consistency_score=_safe_float(d.get("min_consistency_score"), DEFAULT_MIN_CONSISTENCY_SCORE),
        min_samples=_safe_int(d.get("min_samples"), DEFAULT_MIN_SAMPLES),
        min_leader_move=_safe_float(d.get("min_leader_move"), DEFAULT_MIN_LEADER_MOVE),
        signal_multiplier=_safe_float(d.get("signal_multiplier"), DEFAULT_SIGNAL_MULTIPLIER),
        max_single_signal=_safe_float(d.get("max_single_signal"), DEFAULT_MAX_SINGLE_SIGNAL),
        max_total_score=_safe_float(d.get("max_total_score"), DEFAULT_MAX_TOTAL_SCORE),
        use_market_filter=_safe_bool(d.get("use_market_filter"), DEFAULT_USE_MARKET_FILTER),
        market_sma_period=_safe_int(d.get("market_sma_period"), DEFAULT_MARKET_SMA_PERIOD),
        market_filter_fallback_pct=_safe_float(d.get("market_filter_fallback_pct"), DEFAULT_MARKET_FILTER_FALLBACK_PCT),
        max_drawdown_pause=_safe_float(d.get("max_drawdown_pause"), DEFAULT_MAX_DRAWDOWN_PAUSE),
        drawdown_recovery_pct=_safe_float(d.get("drawdown_recovery_pct"), DEFAULT_DRAWDOWN_RECOVERY_PCT),
        drawdown_cooldown_days=_safe_int(d.get("drawdown_cooldown_days"), DEFAULT_DRAWDOWN_COOLDOWN_DAYS),
    )


# ---------------------------------------------------------------------------
# Price download
# ---------------------------------------------------------------------------

def download_echo1_prices(
    tickers: list[str],
    lookback_trading_days: int,
) -> pd.DataFrame:
    """
    Fetch adjusted daily close prices for *tickers* going back far enough to
    cover *lookback_trading_days* trading days plus a calendar buffer.

    Returns a DataFrame with tickers as columns, DatetimeIndex as rows.
    Missing tickers are silently dropped (logged at WARNING).
    """
    today = datetime.now(timezone.utc).date()
    end_date = today + timedelta(days=1)
    calendar_days = lookback_trading_days * 2 + _LOOKBACK_CALENDAR_BUFFER
    start_date = today - timedelta(days=calendar_days)

    raw = get_daily_bars(tickers, start_date.isoformat(), end_date.isoformat(), adjusted=True)

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"].copy()
    else:
        # Single ticker
        if "Close" not in raw.columns:
            return pd.DataFrame()
        first = tickers[0]
        close = raw[["Close"]].rename(columns={"Close": first})

    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    close = close[~close.index.duplicated(keep="last")]
    close.columns = [str(c).upper() for c in close.columns]

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        logger.warning("echo1: tickers missing from price data: %s", missing)

    return close


# ---------------------------------------------------------------------------
# Pair discovery (ported from notebook, yfinance-free)
# ---------------------------------------------------------------------------

def _dedupe_nearby_parameter_clusters(
    df: pd.DataFrame,
    lag_tolerance: int = 2,
    return_period_tolerance: int = 2,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    selected: list[dict] = []
    for _, row in df.iterrows():
        dup = any(
            row["leader"] == s["leader"]
            and row["follower"] == s["follower"]
            and abs(row["lag_days"] - s["lag_days"]) <= lag_tolerance
            and abs(row["return_period"] - s["return_period"]) <= return_period_tolerance
            for s in selected
        )
        if not dup:
            selected.append(row.to_dict())
    return pd.DataFrame(selected).reset_index(drop=True)


def discover_pairs(
    close: pd.DataFrame,
    cfg: Echo1Config,
    *,
    min_corr: float | None = None,
) -> list[dict]:
    """
    Discover lead-lag pairs from *close* (columns = tickers, index = dates).

    Returns a list of pair dicts suitable for build_network_scores().
    """
    if min_corr is None:
        min_corr = cfg.min_avg_abs_corr

    tickers = [t for t in cfg.universe if t in close.columns]
    if len(tickers) < 2:
        return []

    windows = {"corr_3m": 63, "corr_6m": 126, "corr_1y": 252, "corr_2y": 504}
    corr_cols = list(windows.keys())

    # Pre-compute return cache: {period: {ticker: Series}}
    returns_cache: dict[int, dict[str, pd.Series]] = {}
    for period in set(cfg.return_period_range):
        returns_cache[period] = {
            t: close[t].pct_change(period, fill_method=None)
            for t in tickers
        }

    rows: list[dict] = []
    for leader in tickers:
        for follower in tickers:
            for lag_days in cfg.lag_range:
                for return_period in cfg.return_period_range:
                    if lag_days < return_period:
                        continue

                    row: dict[str, Any] = {
                        "leader": leader,
                        "follower": follower,
                        "lag_days": lag_days,
                        "return_period": return_period,
                        **{k: np.nan for k in corr_cols},
                    }
                    skip = False

                    for label, length in windows.items():
                        ls = returns_cache[return_period][leader].tail(length)
                        ff = returns_cache[return_period][follower].shift(-lag_days).tail(length)
                        mask = ls.notna() & ff.notna()
                        ls_v = ls[mask].values
                        ff_v = ff[mask].values

                        if len(ls_v) < cfg.min_samples:
                            row[label] = np.nan
                        else:
                            row[label] = float(np.corrcoef(ls_v, ff_v)[0, 1])

                        # Early exit on weak corr_3m
                        if label == "corr_3m":
                            if pd.isna(row[label]) or abs(row[label]) < 0.10:
                                skip = True
                                break

                    if not skip:
                        rows.append(row)

    if not rows:
        return []

    df = pd.DataFrame(rows)
    abs_corrs = df[corr_cols].abs()

    df["avg_abs_corr"]  = abs_corrs.mean(axis=1, skipna=True)
    df["corr_std"]      = abs_corrs.std(axis=1, skipna=True).fillna(1)
    df["valid_windows"] = df[corr_cols].notna().sum(axis=1)
    df["same_sign_windows"] = df[corr_cols].apply(
        lambda r: max((r.dropna() > 0).sum(), (r.dropna() < 0).sum())
        if r.dropna().size > 0 else 0,
        axis=1,
    )
    df["sign_consistency"] = df["same_sign_windows"] / df["valid_windows"].replace(0, np.nan)
    df["stability_score"]  = 1 / (1 + df["corr_std"])
    df["consistency_score"] = (
        df["sign_consistency"].fillna(0)
        * df["stability_score"]
        * (df["valid_windows"] / len(corr_cols))
    )
    df["score"] = df["avg_abs_corr"].fillna(0) * df["consistency_score"].fillna(0)

    df = df[
        (df["avg_abs_corr"] >= min_corr)
        & (df["consistency_score"] >= cfg.min_consistency_score)
        & (df["valid_windows"] == 4)
    ]

    if df.empty:
        return []

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df = _dedupe_nearby_parameter_clusters(df)
    df = df.head(cfg.max_pairs)
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Pairs cache (trading.echo1_pairs_cache)
# ---------------------------------------------------------------------------

def _month_key(dt: datetime | None = None) -> str:
    """Return 'YYYY-MM-01' for the first day of the given month (UTC today if None)."""
    d = (dt or datetime.now(timezone.utc)).date().replace(day=1)
    return d.isoformat()


def load_cached_pairs(bot_id: str) -> list[dict] | None:
    """
    Load this month's discovered pairs from trading.echo1_pairs_cache.
    Returns None if no row exists for this month.
    """
    month = _month_key()
    try:
        from db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pairs FROM trading.echo1_pairs_cache
                    WHERE bot_id = %s AND month = %s
                    """,
                    (bot_id, month),
                )
                row = cur.fetchone()
        if not row:
            return None
        raw = row[0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        return raw if isinstance(raw, list) else None
    except Exception as exc:
        logger.warning("echo1: failed to load pairs cache bot_id=%s: %s", bot_id, exc)
        return None


def save_cached_pairs(bot_id: str, pairs: list[dict]) -> None:
    """
    Upsert this month's discovered pairs into trading.echo1_pairs_cache.
    """
    month = _month_key()
    try:
        from db import get_conn
        from psycopg2.extras import Json
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trading.echo1_pairs_cache (bot_id, month, pairs, computed_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (bot_id, month) DO UPDATE SET
                        pairs       = EXCLUDED.pairs,
                        computed_at = EXCLUDED.computed_at
                    """,
                    (bot_id, month, Json(pairs)),
                )
            conn.commit()
    except Exception as exc:
        logger.warning("echo1: failed to save pairs cache bot_id=%s: %s", bot_id, exc)


# ---------------------------------------------------------------------------
# Network scoring
# ---------------------------------------------------------------------------

def _average_signed_correlation(pair: dict) -> float:
    vals = [pair.get(k) for k in ("corr_3m", "corr_6m", "corr_1y", "corr_2y")]
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else 0.0


def build_network_scores(
    close_window: pd.DataFrame,
    discovered_pairs: list[dict],
    cfg: Echo1Config,
) -> dict[str, dict]:
    """
    Score each follower ticker by aggregating lead signals from all pairs.

    Returns {ticker: {"ticker": str, "network_score": float, "final_score": float, "signals": list}}
    """
    tickers_in_window = set(close_window.columns)
    scores: dict[str, dict] = {
        t: {"ticker": t, "network_score": 0.0, "final_score": 0.0, "signals": []}
        for t in cfg.universe
        if t in tickers_in_window
    }

    for pair in discovered_pairs:
        leader   = pair["leader"]
        follower = pair["follower"]
        if follower not in scores or leader not in tickers_in_window:
            continue

        lag_days      = int(pair["lag_days"])
        return_period = int(pair["return_period"])
        prices        = close_window[leader].dropna()

        if len(prices) < lag_days + return_period + 1:
            continue

        past_end   = prices.iloc[-lag_days]
        past_start = prices.iloc[-lag_days - return_period]
        if past_start <= 0:
            continue

        leader_return = past_end / past_start - 1
        if abs(leader_return) < cfg.min_leader_move:
            continue

        avg_corr = _average_signed_correlation(pair)
        if avg_corr * leader_return <= 0:
            continue

        signal_score = min(
            abs(avg_corr) * abs(leader_return) * cfg.signal_multiplier,
            cfg.max_single_signal,
        )
        scores[follower]["network_score"] += signal_score
        scores[follower]["signals"].append({
            "leader": leader,
            "avg_corr": avg_corr,
            "leader_return": leader_return,
            "signal_score": signal_score,
        })

    for item in scores.values():
        item["final_score"] = min(item["network_score"], cfg.max_total_score)

    return scores


# ---------------------------------------------------------------------------
# Market filter
# ---------------------------------------------------------------------------

def _market_filter_active(close: pd.DataFrame, cfg: Echo1Config) -> bool:
    """
    Return True if the market filter is triggered (bearish regime).
    Triggered when VOO is below its SMA-{market_sma_period}.
    """
    if not cfg.use_market_filter:
        return False
    fb = cfg.fallback_ticker
    if fb not in close.columns:
        return False
    voo = close[fb].dropna()
    if len(voo) < cfg.market_sma_period:
        return False
    return float(voo.iloc[-1]) < float(voo.tail(cfg.market_sma_period).mean())


# ---------------------------------------------------------------------------
# DB config loader
# ---------------------------------------------------------------------------

def _load_db_config(bot_id: str) -> Echo1Config:
    try:
        from db import get_conn
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM trading.echo1 WHERE bot_id = %s AND is_active = TRUE LIMIT 1",
                    (bot_id,),
                )
                row = cur.fetchone()
        if row:
            return echo1_config_from_db_dict(dict(row))
    except Exception as exc:
        logger.warning(
            "echo1: could not load config for bot_id=%s from DB, using defaults: %s",
            bot_id, exc,
        )
    return Echo1Config(universe=list(DEFAULT_UNIVERSE))


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------

def run_echo1(bot_id: str, use_db_config: bool = True) -> dict:
    """
    Live Echo1 signal runner.

    Returns a signal dict compatible with the existing worker write_signal /
    update_bot_equity pipeline:
        {bot_id, bot_type, ts, signal, note, payload:{target_weights, ...}}
    """
    ts = datetime.now(timezone.utc)
    cfg = _load_db_config(bot_id) if use_db_config else Echo1Config(universe=list(DEFAULT_UNIVERSE))
    fallback = cfg.fallback_ticker

    def _fallback_signal(note: str, pct: float = 1.0) -> dict:
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_ECHO1,
            "ts": ts,
            "signal": "HOLD",
            "note": note,
            "payload": {
                "target_weights": {fallback: pct} if pct > 0 else {},
                "regime": "fallback",
            },
        }

    # ── Download price history ───────────────────────────────────────────────
    all_tickers = list(dict.fromkeys(cfg.universe + [fallback]))
    try:
        close = download_echo1_prices(all_tickers, cfg.discovery_lookback)
    except Exception as exc:
        logger.exception("echo1 bot_id=%s price download failed: %s", bot_id, exc)
        return _fallback_signal(f"price download error: {exc}")

    if close.empty or len(close) < cfg.market_sma_period + 10:
        return _fallback_signal("insufficient price history")

    # ── Market filter ────────────────────────────────────────────────────────
    if _market_filter_active(close, cfg):
        pct = cfg.market_filter_fallback_pct
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_ECHO1,
            "ts": ts,
            "signal": "REBALANCE",
            "note": f"market filter active: {pct:.0%} {fallback}",
            "payload": {
                "target_weights": {fallback: pct},
                "regime": "market_filter",
            },
        }

    # ── Drawdown guard ───────────────────────────────────────────────────────
    # Load equity history from bot_equity to check current drawdown.
    try:
        from db import get_conn
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT d, equity
                    FROM trading.bot_equity
                    WHERE bot_id = %s AND bot_type = %s
                    ORDER BY d DESC
                    LIMIT 60
                    """,
                    (bot_id, BOT_TYPE_ECHO1),
                )
                eq_rows = cur.fetchall()
    except Exception as exc:
        logger.warning("echo1 bot_id=%s equity load failed: %s", bot_id, exc)
        eq_rows = []

    if eq_rows:
        equities = [float(r["equity"]) for r in eq_rows]
        peak     = max(equities)
        current  = equities[0]
        drawdown = 1 - current / peak if peak > 0 else 0.0

        if drawdown >= cfg.max_drawdown_pause:
            pct = cfg.target_gross_exposure
            return {
                "bot_id": bot_id,
                "bot_type": BOT_TYPE_ECHO1,
                "ts": ts,
                "signal": "REBALANCE",
                "note": f"drawdown guard active: {drawdown:.1%} drawdown → {pct:.0%} {fallback}",
                "payload": {
                    "target_weights": {fallback: pct},
                    "regime": f"drawdown_pause({drawdown:.1%})",
                    "drawdown": drawdown,
                },
            }

    # ── Monthly pair discovery (with cache) ──────────────────────────────────
    pairs = load_cached_pairs(bot_id)

    if pairs is None:
        logger.info("echo1 bot_id=%s: running pair discovery (no cache for this month)", bot_id)
        t0 = time.perf_counter()

        # Use the last discovery_lookback trading days of close data.
        close_window = close.tail(cfg.discovery_lookback + 30)
        pairs = discover_pairs(close_window, cfg)

        if len(pairs) < 10:
            logger.info(
                "echo1 bot_id=%s: only %d pairs with strict corr; retrying loose",
                bot_id, len(pairs),
            )
            pairs = discover_pairs(close_window, cfg, min_corr=cfg.min_avg_abs_corr_loose)

        logger.info(
            "echo1 bot_id=%s: discovery found %d pairs in %.1fs",
            bot_id, len(pairs), time.perf_counter() - t0,
        )
        save_cached_pairs(bot_id, pairs)

    if not pairs:
        return _fallback_signal("no pairs discovered")

    # ── Score and select ─────────────────────────────────────────────────────
    max_lb = max(
        (int(p["lag_days"]) + int(p["return_period"]) + 5 for p in pairs),
        default=20,
    )
    score_window = close.tail(max_lb + 5)
    network_scores = build_network_scores(score_window, pairs, cfg)

    ranked   = sorted(network_scores.values(), key=lambda x: x["final_score"], reverse=True)
    selected = [x for x in ranked if x["final_score"] > 0][: cfg.portfolio_size]

    if not selected:
        return _fallback_signal(
            f"no positive signals; holding {fallback}",
            pct=cfg.target_gross_exposure,
        )

    # ── Score-proportional target weights ────────────────────────────────────
    total_score = sum(x["final_score"] for x in selected)
    target_weights: dict[str, float] = {}
    for item in selected:
        weight = (item["final_score"] / total_score) * cfg.target_gross_exposure
        target_weights[item["ticker"]] = round(weight, 6)

    top_pick = selected[0]["ticker"]
    note = (
        f"echo1 rebalance: {len(selected)} picks "
        f"(top={top_pick} score={selected[0]['final_score']:.3f})"
    )

    return {
        "bot_id": bot_id,
        "bot_type": BOT_TYPE_ECHO1,
        "ts": ts,
        "signal": "REBALANCE",
        "note": note,
        "payload": {
            "target_weights": target_weights,
            "regime": "active",
            "top_pick": top_pick,
            "num_pairs": len(pairs),
            "selected": [
                {
                    "ticker": x["ticker"],
                    "score": round(x["final_score"], 4),
                    "weight": round(target_weights.get(x["ticker"], 0.0), 4),
                }
                for x in selected
            ],
        },
    }
