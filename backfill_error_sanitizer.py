"""
backfill_error_sanitizer.py

Shared helper that converts raw backfill/worker exceptions into safe,
user-facing messages before they are returned to API callers.

NEVER expose to clients:
  - DATABASE_URL or any connection string
  - Environment variable names
  - Render service names or internal endpoint paths
  - Stack traces or Python exception details
  - psycopg2 / database driver internals

The real error must always be logged server-side before calling this function.
"""

from __future__ import annotations

# Substrings (lowercase) that flag internal infrastructure problems.
_INFRA_PATTERNS: tuple[str, ...] = (
    "database_url",
    "postgresql://",
    "postgres://",
    "render service",
    "traceback",
    "psycopg",
    "/backfill/",
    "monstrabackfill",
    "set database_url",
    "environment variables",
)


def sanitize_backfill_error(exc: BaseException) -> str:
    """Return a safe, user-facing error message for a backfill exception.

    Infrastructure details (DATABASE_URL, connection strings, stack traces,
    Render service internals) are replaced with a generic message so they
    never reach the client.  The original exception is always logged before
    this function is called so operators retain full visibility.
    """
    raw = str(exc).lower()
    if any(pattern in raw for pattern in _INFRA_PATTERNS):
        return "Backfill service is temporarily unavailable. Please try again in a few minutes."
    if "timeout" in raw or "timed out" in raw:
        return "Backfill took too long to complete. Please try again."
    return "An unexpected error occurred while generating the backfill."
