from __future__ import annotations

import asyncio
import json
import unittest

from scripts.dashboard.app import mesh_repair_confirm, session


class MeshRepairConfirmApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "running": session.running,
            "mesh_repair_ready": session.mesh_repair_ready,
            "mesh_repair_candidates": list(session.mesh_repair_candidates),
            "mesh_repair_selected_loop_ids": list(session.mesh_repair_selected_loop_ids),
            "mesh_repair_confirm_event": session.mesh_repair_confirm_event,
        }
        session.running = True
        session.mesh_repair_ready = True
        session.mesh_repair_candidates = [
            {"loop_id": 1, "points": [[0.0, 0.0, 0.0]]},
            {"loop_id": 2, "points": [[0.1, 0.0, 0.0]]},
        ]
        session.mesh_repair_selected_loop_ids = []
        session.mesh_repair_confirm_event = asyncio.Event()

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.mesh_repair_ready = self._snapshot["mesh_repair_ready"]
        session.mesh_repair_candidates = self._snapshot["mesh_repair_candidates"]
        session.mesh_repair_selected_loop_ids = self._snapshot["mesh_repair_selected_loop_ids"]
        session.mesh_repair_confirm_event = self._snapshot["mesh_repair_confirm_event"]

    @staticmethod
    def _json_payload(response) -> dict:
        return json.loads(response.body.decode("utf-8"))

    async def test_confirm_allows_empty_selection_for_skip(self) -> None:
        response = await mesh_repair_confirm({"selected_loop_ids": []})

        self.assertEqual(response.status_code, 200)
        payload = self._json_payload(response)
        self.assertEqual(payload["status"], "confirmed")
        self.assertEqual(payload["mode"], "skip")
        self.assertEqual(payload["selected_count"], 0)
        self.assertEqual(payload["selected_loop_ids"], [])
        self.assertEqual(session.mesh_repair_selected_loop_ids, [])
        self.assertTrue(session.mesh_repair_confirm_event.is_set())

    async def test_confirm_accepts_valid_loop_ids(self) -> None:
        response = await mesh_repair_confirm({"selected_loop_ids": [2, 2, 1]})

        self.assertEqual(response.status_code, 200)
        payload = self._json_payload(response)
        self.assertEqual(payload["status"], "confirmed")
        self.assertEqual(payload["mode"], "repair")
        self.assertEqual(payload["selected_count"], 2)
        self.assertEqual(set(payload["selected_loop_ids"]), {1, 2})
        self.assertEqual(set(session.mesh_repair_selected_loop_ids), {1, 2})
        self.assertTrue(session.mesh_repair_confirm_event.is_set())

    async def test_confirm_rejects_unknown_loop_ids(self) -> None:
        response = await mesh_repair_confirm({"selected_loop_ids": [9]})

        self.assertEqual(response.status_code, 400)
        payload = self._json_payload(response)
        self.assertIn("Unknown loop ids", payload["error"])
        self.assertEqual(session.mesh_repair_selected_loop_ids, [])
        self.assertFalse(session.mesh_repair_confirm_event.is_set())


if __name__ == "__main__":
    unittest.main()
