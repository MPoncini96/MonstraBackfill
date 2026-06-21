"""
Regression tests for sanitize_backfill_error() in app.py.

These tests verify that:
1. DATABASE_URL errors are replaced with a safe message.
2. Stack traces / Python internals are never returned verbatim.
3. Infrastructure details (Render, psycopg2, connection strings) are hidden.
4. Timeout errors produce a distinct user-facing message.
5. Generic errors produce the generic fallback message.

Run with:  python -m pytest test_sanitize_backfill_error.py -v
       or:  python -m unittest test_sanitize_backfill_error -v
"""

from __future__ import annotations

import os
import unittest

from backfill_error_sanitizer import sanitize_backfill_error

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INFRA_LEAK_STRINGS = [
    "DATABASE_URL",
    "postgresql://",
    "postgres://",
    "Render service",
    "psycopg2",
    "Traceback",
    "/backfill/",
    "MonstraBackfill",
    "environment variables",
    "Set DATABASE_URL",
]


def assert_no_infra_leak(test: unittest.TestCase, message: str) -> None:
    """Assert that none of the forbidden infrastructure strings appear in message."""
    lower = message.lower()
    for leak in INFRA_LEAK_STRINGS:
        test.assertNotIn(
            leak.lower(),
            lower,
            f"Infrastructure detail leaked in client message: {leak!r} found in {message!r}",
        )


# ---------------------------------------------------------------------------
# DATABASE_URL / database connection errors
# ---------------------------------------------------------------------------

class TestDatabaseUrlErrors(unittest.TestCase):
    def test_missing_database_url_error_is_sanitized(self):
        exc = RuntimeError(
            "DATABASE_URL is not set on this MonstraBackfill instance. "
            "The /backfill/* endpoints require a live database connection. "
            "Set DATABASE_URL in the Render service environment variables."
        )
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertIn("temporarily unavailable", result.lower())

    def test_postgres_connection_string_is_sanitized(self):
        exc = RuntimeError("could not connect to postgresql://user:pass@host/db")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertIn("temporarily unavailable", result.lower())

    def test_psycopg2_error_is_sanitized(self):
        exc = Exception("psycopg2.OperationalError: SSL connection required")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertIn("temporarily unavailable", result.lower())

    def test_render_service_reference_is_sanitized(self):
        exc = RuntimeError("Render service MonstraBackfill is unreachable")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertIn("temporarily unavailable", result.lower())

    def test_environment_variables_reference_is_sanitized(self):
        exc = RuntimeError("Set DATABASE_URL in the Render service environment variables")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)

    def test_backfill_endpoint_path_is_sanitized(self):
        exc = RuntimeError("The /backfill/* endpoints require a live database connection.")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)

    def test_postgres_url_scheme_is_sanitized(self):
        exc = RuntimeError("postgres://readonly@db.render.com/monstra refused connection")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)


# ---------------------------------------------------------------------------
# Timeout errors
# ---------------------------------------------------------------------------

class TestTimeoutErrors(unittest.TestCase):
    def test_timeout_produces_try_again_message(self):
        exc = Exception("Request timed out after 30s")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertTrue(
            "too long" in result.lower() or "try again" in result.lower(),
            f"Expected try-again language in: {result!r}",
        )

    def test_timed_out_variant(self):
        exc = TimeoutError("operation timed out")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertTrue(
            "too long" in result.lower() or "try again" in result.lower(),
        )


# ---------------------------------------------------------------------------
# Stack traces must never reach the client
# ---------------------------------------------------------------------------

class TestStackTraceNotLeaked(unittest.TestCase):
    def test_traceback_keyword_is_sanitized(self):
        exc = Exception(
            'Traceback (most recent call last):\n  File "backfill_alpha1.py", line 42\nRuntimeError: ...'
        )
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)

    def test_python_file_path_with_database_url_is_sanitized(self):
        exc = RuntimeError('/app/MonstraBackfill/backfill_alpha1.py line 100: DATABASE_URL missing')
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertNotIn("line 100", result)
        self.assertNotIn("backfill_alpha1.py", result)


# ---------------------------------------------------------------------------
# Generic / unexpected errors — raw detail must never appear in output
# ---------------------------------------------------------------------------

class TestGenericErrors(unittest.TestCase):
    def test_unknown_exception_returns_generic_message(self):
        exc = Exception("some internal detail XYZZY that should not leak")
        result = sanitize_backfill_error(exc)
        assert_no_infra_leak(self, result)
        self.assertNotIn("XYZZY", result)
        self.assertNotIn("some internal detail", result)

    def test_return_value_is_always_a_non_empty_string(self):
        cases = [
            RuntimeError("anything"),
            ValueError("val error"),
            Exception(""),
            MemoryError("oom"),
        ]
        for exc in cases:
            result = sanitize_backfill_error(exc)
            self.assertIsInstance(result, str, f"Expected str, got {type(result)} for {exc!r}")
            self.assertGreater(len(result), 0, f"Empty result for {exc!r}")

    def test_empty_exception_message_returns_generic_fallback(self):
        exc = Exception("")
        result = sanitize_backfill_error(exc)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
