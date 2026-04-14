"""
Minimal database shim for the preview-only HTTP service.

Production Monstra-Worker uses `DATABASE_URL` and full PostgreSQL access. This service
only runs in-memory previews; the backfill modules still import `db` at load time, so
we provide the same symbols without requiring a live database connection.
"""

from __future__ import annotations

import os
from typing import Any

# trading.bot_equity.source — must match add_bot_equity_source.sql labels
BOT_EQUITY_SOURCE_BACKFILL = "backfill"
BOT_EQUITY_SOURCE_LIVE = "live trading"

# Allow optional DATABASE_URL for future use; previews never call get_conn().
DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_conn() -> Any:
    raise RuntimeError(
        "Database access is not available in MonstraBackfill preview service. "
        "If you see this, a code path is calling the DB stub."
    )


def get_strategy_origin(bot_id: str, bot_type: str) -> str | None:
    return None
