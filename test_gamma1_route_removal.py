from __future__ import annotations

import unittest

import app as monstra_app


class TestGamma1RouteRemoval(unittest.TestCase):
    def test_health_snapshot_does_not_advertise_gamma1(self):
        snapshot = monstra_app._readiness_snapshot()

        self.assertNotIn("gamma1", snapshot["requiredModules"]["preview"])
        self.assertNotIn("gamma1", snapshot["requiredModules"]["backfill"])
        self.assertNotIn("/preview/gamma1", snapshot["registeredRoutes"]["preview"])
        self.assertNotIn("/backfill/gamma1", snapshot["registeredRoutes"]["backfill"])

    def test_internal_dispatch_rejects_gamma1(self):
        with self.assertRaises(monstra_app.HTTPException) as preview_exc:
            monstra_app._run("gamma1", {})
        self.assertEqual(preview_exc.exception.status_code, 404)

        with self.assertRaises(monstra_app.HTTPException) as backfill_exc:
            monstra_app._run_backfill("gamma1", {"botId": "deprecated-vex"})
        self.assertEqual(backfill_exc.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)