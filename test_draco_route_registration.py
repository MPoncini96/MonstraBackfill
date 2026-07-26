from __future__ import annotations

import unittest

import app as monstra_app


class TestDracoRouteRegistration(unittest.TestCase):
    def test_health_snapshot_lists_draco_preview_and_backfill_routes(self):
        snapshot = monstra_app._readiness_snapshot()

        self.assertIn("draco", snapshot["requiredModules"]["preview"])
        self.assertIn("draco", snapshot["requiredModules"]["backfill"])
        self.assertIn("/preview/draco", snapshot["registeredRoutes"]["preview"])
        self.assertIn("/backfill/draco", snapshot["registeredRoutes"]["backfill"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
