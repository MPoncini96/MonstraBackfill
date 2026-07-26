# bots/draco.py
"""
Draco live signal runner.

Strategy: multi-timeframe log-linear regression mean-reversion / trend-quality.
For each configured lookback, fits ln(price) = intercept + slope*index, scores
candidates on how far below their regression line they trade, how strong and
consistent their trend is, and how well the line fits. Eligible candidates
lock a regression-projected exit target at entry that is never recalculated.

Key design notes:
- All regression/scoring/target math lives in draco_math.py (pure functions,
  shared with backfill and preview/backtest so behavior is identical everywhere).
- Config schema/validation/DB loading lives in draco_config.py.
- Locked targets, open positions, cooldowns, market-regime confirmation state,
  and circuit-breaker cooldown are persisted in trading.draco_state (JSONB per
  bot_id) so they survive worker restarts/retries - never held only in memory.
- Counters/cooldowns advance at most once per completed trading date, keyed off
  the last bar of the downloaded price series (not wall-clock time), so a
  retried run for the same trading date is a no-op state-wise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from bot_identity import BOT_TYPE_DRACO
from market_data_provider import get_daily_bars
import draco_math as dm
from draco_config import DracoConfig, load_db_config

DRACO_LIVE_EQUITY_SOURCE = "live trading"

logger = logging.getLogger(__name__)

# Extra calendar-day buffer on top of the longest lookback when fetching history.
_LOOKBACK_CALENDAR_BUFFER = 220


# ---------------------------------------------------------------------------
# State persistence (trading.draco_state) - survives worker restarts/retries
# ---------------------------------------------------------------------------

def _empty_state() -> dict[str, Any]:
    return {
        "positions": {},
        "cooldowns": {},
        "market_regime": {"state": "risk_on", "confirmation_count": 0, "pending_state": None},
        "circuit_breaker": {"active": False, "cooldown_remaining_days": 0},
        "last_entry_scan_iso_week": None,
        "last_processed_trading_date": None,
    }


def load_draco_state(bot_id: str) -> dict[str, Any]:
    try:
        from db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT state FROM trading.draco_state WHERE bot_id = %s", (bot_id,))
                row = cur.fetchone()
        if not row:
            return _empty_state()
        raw = row[0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return _empty_state()
        merged = _empty_state()
        merged.update(raw)
        return merged
    except Exception as exc:
        logger.warning("draco: failed to load state for bot_id=%s, using empty state: %s", bot_id, exc)
        return _empty_state()


def save_draco_state(bot_id: str, state: dict[str, Any]) -> None:
    try:
        from db import get_conn
        from psycopg2.extras import Json
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trading.draco_state (bot_id, state, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (bot_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (bot_id, Json(state)),
                )
            conn.commit()
    except Exception as exc:
        logger.warning("draco: failed to save state for bot_id=%s: %s", bot_id, exc)


# ---------------------------------------------------------------------------
# Price download
# ---------------------------------------------------------------------------

def download_draco_prices(tickers: list[str], lookback_trading_days: int) -> pd.DataFrame:
    """Fetch adjusted daily closes far enough back to cover `lookback_trading_days`."""
    today = datetime.now(timezone.utc).date()
    end_date = today.isoformat()
    calendar_days = lookback_trading_days * 2 + _LOOKBACK_CALENDAR_BUFFER
    start_date = (today - pd.Timedelta(days=calendar_days)).isoformat()

    raw = get_daily_bars(tickers, start_date, end_date, adjusted=True)
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    close = close[~close.index.duplicated(keep="last")]
    close.columns = [str(c).upper() for c in close.columns]
    return close


def _trading_days_between(index: pd.DatetimeIndex, start_date: str, end_date: str) -> int:
    """Count trading-day steps between two dates using the actual observed index."""
    try:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        start_pos = index.get_indexer([start_ts], method="nearest")[0]
        end_pos = index.get_indexer([end_ts], method="nearest")[0]
        return max(0, int(end_pos) - int(start_pos))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Market regime filter (SPY vs 200-day SMA, N-day confirmation)
# ---------------------------------------------------------------------------

def _update_market_regime(
    state: dict[str, Any],
    regime_series: pd.Series,
    cfg: DracoConfig,
    *,
    is_new_trading_date: bool,
) -> str:
    """Update (in place) and return the confirmed regime: 'risk_on' or 'risk_off'."""
    regime = state["market_regime"]
    if not cfg.use_market_regime_filter or len(regime_series) < cfg.market_sma_period:
        regime["state"] = "risk_on"
        return "risk_on"

    sma = float(regime_series.tail(cfg.market_sma_period).mean())
    current = float(regime_series.iloc[-1])
    raw_signal = "risk_off" if current < sma else "risk_on"

    if not is_new_trading_date:
        return regime.get("state", "risk_on")

    if raw_signal == regime.get("state", "risk_on"):
        regime["pending_state"] = None
        regime["confirmation_count"] = 0
    else:
        if regime.get("pending_state") == raw_signal:
            regime["confirmation_count"] = int(regime.get("confirmation_count", 0)) + 1
        else:
            regime["pending_state"] = raw_signal
            regime["confirmation_count"] = 1

        if regime["confirmation_count"] >= cfg.regime_confirmation_days:
            regime["state"] = raw_signal
            regime["pending_state"] = None
            regime["confirmation_count"] = 0

    return regime.get("state", "risk_on")


# ---------------------------------------------------------------------------
# Circuit breaker (portfolio drawdown -> multi-day cooldown)
# ---------------------------------------------------------------------------

def _update_circuit_breaker(
    state: dict[str, Any],
    bot_id: str,
    cfg: DracoConfig,
    *,
    is_new_trading_date: bool,
) -> bool:
    """Update (in place) and return whether the circuit breaker is active."""
    breaker = state["circuit_breaker"]
    if not cfg.use_portfolio_circuit_breaker:
        breaker["active"] = False
        breaker["cooldown_remaining_days"] = 0
        return False

    if is_new_trading_date and breaker.get("active"):
        remaining = max(0, int(breaker.get("cooldown_remaining_days", 0)) - 1)
        breaker["cooldown_remaining_days"] = remaining
        if remaining <= 0:
            breaker["active"] = False
        return breaker["active"]

    if breaker.get("active"):
        return True

    try:
        from db import get_conn
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT equity FROM trading.bot_equity
                    WHERE bot_id = %s AND bot_type = %s
                    ORDER BY d DESC LIMIT 400
                    """,
                    (bot_id, BOT_TYPE_DRACO),
                )
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("draco: circuit breaker equity load failed bot_id=%s: %s", bot_id, exc)
        rows = []

    if not rows:
        return False

    equities = [float(r["equity"]) for r in rows]
    peak = max(equities)
    current = equities[0]
    drawdown = (1 - current / peak) if peak > 0 else 0.0

    if drawdown >= cfg.maximum_portfolio_drawdown and is_new_trading_date:
        breaker["active"] = True
        breaker["cooldown_remaining_days"] = cfg.circuit_breaker_cooldown_trading_days
        return True

    return False


# ---------------------------------------------------------------------------
# Candidate evaluation / portfolio construction
# ---------------------------------------------------------------------------

@dataclass
class DracoCandidate:
    ticker: str
    total_score: float
    metrics: dict[str, dm.TimeframeMetrics]
    target: dm.LockedTarget


def evaluate_candidate(ticker: str, prices: pd.Series, cfg: DracoConfig) -> DracoCandidate | None:
    metrics = dm.compute_all_timeframe_metrics(prices.values.tolist(), lookbacks=cfg.entry_lookbacks)
    if not metrics:
        return None
    current_price = float(prices.iloc[-1])
    total_score = dm.compute_total_score(metrics, weights=cfg.timeframe_weights)

    if not dm.evaluate_entry_requirements(metrics, total_score, current_price, cfg.entry_requirements()):
        return None

    target = dm.select_locked_target(metrics, current_price, cfg.target_requirements())
    if target is None:
        return None

    return DracoCandidate(ticker=ticker, total_score=total_score, metrics=metrics, target=target)


def rank_candidates(candidates: list[DracoCandidate]) -> list[DracoCandidate]:
    return sorted(
        candidates,
        key=lambda c: (c.total_score, c.target.r_squared, c.target.entry_upside),
        reverse=True,
    )


def is_new_trading_date(state: dict[str, Any], trading_date: str) -> bool:
    return state.get("last_processed_trading_date") != trading_date


def is_entry_scan_eligible(
    cfg: DracoConfig,
    state: dict[str, Any],
    *,
    new_trading_date: bool,
    current_iso_week: list[int],
) -> bool:
    """Weekly scans only evaluate new entries on the first eligible completed
    session of each ISO week; other cadences scan every eligible run. Exits
    are always evaluated separately regardless of this gate."""
    if not new_trading_date:
        return False
    if cfg.scan_cadence != "weekly":
        return True
    return state.get("last_entry_scan_iso_week") != current_iso_week


def evaluate_position_exit(
    current_price: float,
    entry_price: float,
    holding_days: int,
    target: dm.LockedTarget,
    cfg: DracoConfig,
) -> tuple[str | None, float, float]:
    """Return (exit_reason, position_return, projected_target_price).

    exit_reason is None while the position should remain open. Priority order
    when multiple conditions are met simultaneously: stop loss, then locked
    target reached, then maximum holding period (matches Draco's documented
    exit precedence; risk-off and circuit-breaker exits are handled one level
    up since they liquidate the whole portfolio rather than one position).
    """
    projected_price = dm.project_locked_target(target, holding_days)
    position_return = (current_price / entry_price) - 1.0 if entry_price > 0 else 0.0

    exit_reason: str | None = None
    if position_return <= -cfg.stop_loss_percent:
        exit_reason = "stop_loss"
    elif current_price >= projected_price:
        exit_reason = "target_reached"
    elif holding_days >= cfg.maximum_holding_trading_days:
        exit_reason = "max_hold"

    return exit_reason, position_return, projected_price


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------

def _live_signal_dict(bot_id: str, ts: datetime, *, signal: str, note: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "bot_id": bot_id,
        "bot_type": BOT_TYPE_DRACO,
        "ts": ts,
        "equity_source": DRACO_LIVE_EQUITY_SOURCE,
        "signal": signal,
        "note": note,
        "payload": payload,
    }


def run_draco(bot_id: str, use_db_config: bool = True) -> dict[str, Any]:
    """Live Draco signal runner. Returns a worker-compatible signal dict."""
    ts = datetime.now(timezone.utc)
    cfg = load_db_config(bot_id, strict=True) if use_db_config else DracoConfig(universe=[])
    fallback = cfg.fallback_ticker
    state = load_draco_state(bot_id)

    def _fallback_signal(note: str, pct: float = 1.0, extra: dict | None = None) -> dict:
        payload = {"target_weights": {fallback: pct} if pct > 0 else {}, "regime": "fallback"}
        if extra:
            payload.update(extra)
        return _live_signal_dict(bot_id, ts, signal="HOLD" if pct <= 0 else "REBALANCE", note=note, payload=payload)

    longest_lookback = max(cfg.entry_lookbacks.values()) if cfg.entry_lookbacks else max(dm.LOOKBACKS.values())
    all_tickers = list(dict.fromkeys(cfg.universe + [cfg.market_regime_ticker, fallback]))
    try:
        close = download_draco_prices(all_tickers, longest_lookback)
    except Exception as exc:
        logger.exception("draco bot_id=%s price download failed: %s", bot_id, exc)
        return _fallback_signal(f"price download error: {exc}")

    if close.empty or cfg.market_regime_ticker not in close.columns:
        return _fallback_signal("insufficient price history")

    trading_date = close.index[-1].date().isoformat()
    new_trading_date = is_new_trading_date(state, trading_date)

    current_iso_week = list(close.index[-1].isocalendar()[:2])
    is_entry_scan_day = is_entry_scan_eligible(
        cfg, state, new_trading_date=new_trading_date, current_iso_week=current_iso_week
    )

    regime = _update_market_regime(state, close[cfg.market_regime_ticker], cfg, is_new_trading_date=new_trading_date)
    circuit_breaker_active = _update_circuit_breaker(state, bot_id, cfg, is_new_trading_date=new_trading_date)
    risk_off = regime == "risk_off"

    explain: dict[str, Any] = {
        "tradingDate": trading_date,
        "session": "regular",
        "runId": f"{trading_date}-draco",
        "marketRegime": regime,
        "circuitBreakerActive": circuit_breaker_active,
    }

    # ── Risk-off / circuit breaker: liquidate to fallback, block new entries ──
    if circuit_breaker_active or (risk_off and cfg.liquidate_to_fallback_during_risk_off):
        state["positions"] = {}
        if new_trading_date:
            state["last_processed_trading_date"] = trading_date
        save_draco_state(bot_id, state)
        reason = "circuit_breaker" if circuit_breaker_active else "risk_off"
        return _fallback_signal(
            f"draco liquidated to {fallback}: {reason}",
            pct=cfg.target_gross_exposure,
            extra={**explain, "exitReason": reason},
        )

    # ── Exit evaluation (every eligible completed run) ─────────────────────
    positions: dict[str, Any] = dict(state.get("positions", {}))
    cooldowns: dict[str, str] = dict(state.get("cooldowns", {}))
    exited: list[dict[str, Any]] = []

    for ticker, pos in list(positions.items()):
        if ticker not in close.columns:
            continue
        series = close[ticker].dropna()
        if series.empty:
            continue
        current_price = float(series.iloc[-1])
        entry_price = float(pos["entry_price"])
        holding_days = _trading_days_between(close.index, pos["entry_date"], trading_date)

        target = dm.LockedTarget(
            label=pos["target_label"],
            lookback=int(pos["target_lookback"]),
            slope=float(pos["target_slope"]),
            intercept=float(pos["target_intercept"]),
            endpoint_index=int(pos["target_endpoint_index"]),
            r_squared=float(pos["target_r_squared"]),
            entry_price=entry_price,
            entry_target_price=float(pos["entry_target_price"]),
            entry_upside=float(pos["entry_upside"]),
        )
        exit_reason, position_return, projected_price = evaluate_position_exit(
            current_price, entry_price, holding_days, target, cfg
        )

        if exit_reason:
            del positions[ticker]
            if new_trading_date and cfg.symbol_reentry_cooldown_days > 0:
                cooldown_until = close.index[-1] + pd.Timedelta(days=cfg.symbol_reentry_cooldown_days)
                cooldowns[ticker] = cooldown_until.date().isoformat()
            exited.append({
                "ticker": ticker, "exitReason": exit_reason, "positionReturn": round(position_return, 4),
                "tradingDaysHeld": holding_days,
            })
        else:
            positions[ticker]["holding_days"] = holding_days
            positions[ticker]["position_return"] = round(position_return, 4)
            positions[ticker]["projected_target_price"] = projected_price

    # Drop expired cooldowns.
    if new_trading_date:
        cooldowns = {t: d for t, d in cooldowns.items() if d > trading_date}

    # ── Entry scan (weekly, only when slots are free) ───────────────────────
    new_entries: list[dict[str, Any]] = []
    if is_entry_scan_day and len(positions) < cfg.max_positions:
        candidates: list[DracoCandidate] = []
        for ticker in cfg.universe:
            if ticker in positions or ticker in cooldowns or ticker not in close.columns:
                continue
            series = close[ticker].dropna()
            if series.empty:
                continue
            candidate = evaluate_candidate(ticker, series, cfg)
            if candidate is not None:
                candidates.append(candidate)

        ranked = rank_candidates(candidates)
        free_slots = cfg.max_positions - len(positions)
        take = min(free_slots, cfg.max_new_positions_per_scan, len(ranked))
        for candidate in ranked[:take]:
            t = candidate.target
            positions[candidate.ticker] = {
                "target_label": t.label,
                "target_lookback": t.lookback,
                "target_slope": t.slope,
                "target_intercept": t.intercept,
                "target_endpoint_index": t.endpoint_index,
                "target_r_squared": t.r_squared,
                "entry_price": t.entry_price,
                "entry_target_price": t.entry_target_price,
                "entry_upside": t.entry_upside,
                "entry_date": trading_date,
                "score_at_entry": candidate.total_score,
                "holding_days": 0,
                "position_return": 0.0,
                "projected_target_price": t.entry_target_price,
            }
            new_entries.append({"ticker": candidate.ticker, "score": round(candidate.total_score, 2), "targetUpside": round(t.entry_upside, 4)})

    # ── Portfolio construction (equal weight, or fallback if empty) ─────────
    if positions:
        weight = cfg.target_gross_exposure / len(positions)
        target_weights = {ticker: round(weight, 6) for ticker in positions}
        signal = "REBALANCE"
        note = f"draco: {len(positions)} position(s), {len(new_entries)} new, {len(exited)} exited"
    elif cfg.hold_fallback_when_empty:
        target_weights = {fallback: cfg.target_gross_exposure}
        signal = "REBALANCE"
        note = f"draco: no positions held; {cfg.target_gross_exposure:.0%} {fallback}"
    else:
        target_weights = {}
        signal = "HOLD"
        note = "draco: no positions held; no fallback allocation configured"

    state["positions"] = positions
    state["cooldowns"] = cooldowns
    if new_trading_date:
        state["last_processed_trading_date"] = trading_date
        if is_entry_scan_day:
            state["last_entry_scan_iso_week"] = current_iso_week
    save_draco_state(bot_id, state)

    return _live_signal_dict(
        bot_id,
        ts,
        signal=signal,
        note=note,
        payload={
            "target_weights": target_weights,
            **explain,
            "isEntryScanDay": is_entry_scan_day,
            "positions": [
                {"ticker": t, **{k: v for k, v in p.items() if k not in {"target_intercept", "target_slope"}}}
                for t, p in positions.items()
            ],
            "newEntries": new_entries,
            "exited": exited,
        },
    )
