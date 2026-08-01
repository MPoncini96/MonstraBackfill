from __future__ import annotations

import unittest

import app as monstra_app
from algorithm_registry import ALGORITHM_REGISTRY, get_algorithm


class TestGamma1Removal(unittest.TestCase):
    def test_gamma1_is_not_in_algorithm_registry(self):
        self.assertIsNone(get_algorithm("gamma1"))
        self.assertNotIn("gamma1", [entry.slug for entry in ALGORITHM_REGISTRY])

    def test_health_snapshot_does_not_advertise_gamma1(self):
        snapshot = monstra_app._readiness_snapshot()

        self.assertNotIn("gamma1", snapshot["requiredModules"]["preview"])
        self.assertNotIn("gamma1", snapshot["requiredModules"]["backfill"])
        self.assertNotIn("/preview/gamma1", snapshot["registeredRoutes"]["preview"])
        self.assertNotIn("/backfill/gamma1", snapshot["registeredRoutes"]["backfill"])

    def test_internal_dispatch_rejects_gamma1(self):
        with self.assertRaises(monstra_app.HTTPException) as preview_exc:
            monstra_app._run("gamma1", {})
        self.assertEqual(preview_exc.exception.status_code, 400)

        with self.assertRaises(monstra_app.HTTPException) as backfill_exc:
            monstra_app._run_backfill("gamma1", {"botId": "deprecated-vex"})
        self.assertEqual(backfill_exc.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
