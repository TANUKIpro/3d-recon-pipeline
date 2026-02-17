from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.dashboard.state import PipelineSession


class PipelineSessionMeshRepairTests(unittest.TestCase):
    def test_set_and_clear_mesh_repair_candidates(self) -> None:
        session = PipelineSession()
        self.assertFalse(session.mesh_repair_ready)

        candidates = [
            {
                "loop_id": 3,
                "points": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]],
            }
        ]
        analysis = {"loop_count": 1, "bbox_center": [0.0, 0.0, 0.0]}
        session.set_mesh_repair_candidates("/tmp/object_mesh_wrapped.ply", candidates, analysis=analysis)

        self.assertTrue(session.mesh_repair_ready)
        self.assertEqual(session.mesh_repair_source_mesh_path, "/tmp/object_mesh_wrapped.ply")
        self.assertEqual(len(session.mesh_repair_candidates), 1)
        self.assertEqual(session.mesh_repair_analysis.get("loop_count"), 1)

        session.mesh_repair_selected_loop_ids = [3]
        status = session.to_status_dict()
        self.assertTrue(status["mesh_repair"]["ready"])
        self.assertEqual(status["mesh_repair"]["candidate_count"], 1)
        self.assertEqual(status["mesh_repair"]["selected_loop_count"], 1)
        self.assertEqual(status["mesh_repair"]["analysis"]["loop_count"], 1)

        session.clear_mesh_repair_candidates()
        self.assertFalse(session.mesh_repair_ready)
        self.assertEqual(session.mesh_repair_candidates, [])
        self.assertEqual(session.mesh_repair_selected_loop_ids, [])
        self.assertEqual(session.mesh_repair_analysis, {})

    def test_hydrate_clears_interactive_mesh_repair_state(self) -> None:
        session = PipelineSession()
        session.set_mesh_repair_candidates(
            "/tmp/object_mesh_wrapped.ply",
            [{"loop_id": 1, "points": [[0.0, 0.0, 0.0]]}],
            analysis={"loop_count": 1},
        )

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh.ply").write_text("ply\n", encoding="utf-8")
            session.hydrate_from_output_dir(out)

        self.assertFalse(session.mesh_repair_ready)
        self.assertEqual(session.mesh_repair_candidates, [])
        self.assertEqual(session.mesh_repair_selected_loop_ids, [])
        self.assertEqual(session.mesh_repair_analysis, {})


if __name__ == "__main__":
    unittest.main()
