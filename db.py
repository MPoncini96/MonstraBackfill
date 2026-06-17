"""
Database module for MonstraBackfill.

Operates in two modes:
- Preview-only (no DATABASE_URL): in-memory previews only; get_conn() raises a clear
  error if a code path accidentally calls it.
- Backfill mode (DATABASE_URL set): real psycopg2 connection for reading bot configs
  and writing bot_equity rows.  Required by the /backfill/* endpoints.
"""

from __future__ import annotations

import os
from typing import Any

# trading.bot_equity.source — must match add_bot_equity_source.sql labels
BOT_EQUITY_SOURCE_BACKFILL = "backfill"
BOT_EQUITY_SOURCE_LIVE = "live trading"

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

_STRATEGY_TABLE_BY_TYPE = {
    "alpha1":    "trading.alpha1",
    "alpha2":    "trading.alpha2",
    "beta1":     "trading.beta1",
    "beta1_v2":  "trading.beta1_v2",  # Phantom v2 — composite scoring
    "gamma1":    "trading.gamma1",
}


def get_conn() -> Any:
    """
    Return a psycopg2 connection when DATABASE_URL is configured.

    Raises RuntimeError (not ImportError) when DATABASE_URL is absent so that
    the /backfill/* endpoints surface a clear 500 with an actionable message
    instead of a silent or cryptic failure.
    """
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set on this MonstraBackfill instance. "
            "The /backfill/* endpoints require a live database connection. "
            "Set DATABASE_URL in the Render service environment variables."
        )
    import psycopg2  # noqa: PLC0415 — imported lazily; not needed for preview-only mode

    return psycopg2.connect(DATABASE_URL)


def get_strategy_origin(bot_id: str, bot_type: str) -> str | None:
    """Return trading.<strategy>.origin for this bot_id + bot_type, or None on failure."""
    if not DATABASE_URL:
        return None
    base = (bot_id or "").strip()
    if not base:
        return None
    tbl = _STRATEGY_TABLE_BY_TYPE.get((bot_type or "").strip().lower())
    if not tbl:
        return None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT origin FROM {tbl} WHERE bot_id = %s LIMIT 1",
                    (base,),
                )
                row = cur.fetchone()
    except Exception:
        return None
    if not row:
        return None
    val = row[0]
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def get_beta1_config(bot_id: str | None = None) -> dict[str, Any] | None:
    """
    Returns None when DATABASE_URL is absent (preview-only mode).
    Beta1 live-style helpers handle None by failing closed.
    """
    return None


def get_beta1_v2_config(bot_id: str | None = None) -> dict[str, Any] | None:
    """
    Fetch bot configuration from trading.beta1_v2 table.

    Returns all Beta1V2Config fields as a plain dict, or None when
    DATABASE_URL is absent (preview-only mode) or no active row is found.

    ISOLATION: reads ONLY from trading.beta1_v2.  Never touches trading.beta1.
    """
    if not DATABASE_URL:
        return None

    import psycopg2
    from psycopg2.extras import RealDictCursor

    if bot_id is None:
        bot_id_clean = "beta1_v2"
    else:
        # Strip the _beta1_v2 suffix if present (bot_id stored without suffix)
        raw = bot_id.strip()
        import re as _re
        bot_id_clean = _re.sub(r"_beta1_v2$", "", raw, flags=_re.IGNORECASE) or raw

    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        bot_id,
                        universe,
                        COALESCE(max_holdings,               4)       AS max_holdings,
                        COALESCE(fallback_ticker,            'VOO')   AS fallback_ticker,
                        COALESCE(min_final_score,            0.0)     AS min_final_score,
                        COALESCE(weight_rounding,            0.05)    AS weight_rounding,
                        COALESCE(history_days,               220)     AS history_days,
                        COALESCE(min_daily_bars,             80)      AS min_daily_bars,
                        COALESCE(momentum_weight,            1.00)    AS momentum_weight,
                        COALESCE(confirmation_weight,        0.10)    AS confirmation_weight,
                        COALESCE(risk_weight,                0.20)    AS risk_weight,
                        COALESCE(use_market_risk_gate,       TRUE)    AS use_market_risk_gate,
                        COALESCE(market_gate_override_score, 0.20)    AS market_gate_override_score,
                        is_active
                    FROM trading.beta1_v2
                    WHERE bot_id = %s AND is_active = TRUE
                    LIMIT 1
                    """,
                    (bot_id_clean,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                config = dict(row)
                config["universe"] = list(config.get("universe") or [])
                return config
    except Exception as e:
        print(f"Warning: Could not fetch beta1_v2 config for {bot_id}: {e}")
        return None
