from __future__ import annotations

import unittest

import app as monstra_app


class TestAptetRouteRegistration(unittest.TestCase):
    def test_health_snapshot_lists_aptet_preview_and_backfill_routes(self):
        snapshot = monstra_app._readiness_snapshot()

        self.assertIn("aptet", snapshot["requiredModules"]["preview"])
        self.assertIn("aptet", snapshot["requiredModules"]["backfill"])
        self.assertIn("/preview/aptet", snapshot["registeredRoutes"]["preview"])
        self.assertIn("/backfill/aptet", snapshot["registeredRoutes"]["backfill"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
