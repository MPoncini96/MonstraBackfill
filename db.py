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
