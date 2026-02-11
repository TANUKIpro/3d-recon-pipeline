from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.dashboard.configuration import build_pipeline_config
from scripts.dashboard.state import detect_stage_outputs


class BuildPipelineConfigTests(unittest.TestCase):
    def test_respects_raw_diffcd_values_without_legacy_upgrade(self) -> None:
        cfg = build_pipeline_config(
            {
                "diffcd_batch_size": 3000,
                "diffcd_n_batches": 2500,
                "diffcd_resolution": 384,
            },
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={
                "DIFFCD_BATCH_SIZE": "5000",
                "DIFFCD_N_BATCHES": "30000",
                "DIFFCD_RESOLUTION": "512",
            },
        )

        self.assertEqual(cfg.diffcd_batch_size, 3000)
        self.assertEqual(cfg.diffcd_n_batches, 2500)
        self.assertEqual(cfg.diffcd_resolution, 384)

    def test_clamps_and_normalizes_numeric_fields(self) -> None:
        cfg = build_pipeline_config(
            {
                "max_frames": 1,
                "pi3x_frame_target": 999,
                "denoise_dbscan_eps": -1,
                "denoise_dbscan_eps_ratio": -0.1,
                "denoise_dbscan_min_samples": 0,
                "denoise_dbscan_max_points": 10,
                "denoise_sor_neighbors": 1,
                "denoise_sor_std_ratio": 0,
                "denoise_radius_neighbors": 0,
                "denoise_radius_radius_ratio": 0,
            },
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )

        self.assertEqual(cfg.max_frames, 2)
        self.assertEqual(cfg.pi3x_frame_target, 2)
        self.assertEqual(cfg.denoise_dbscan_eps, 0.0)
        self.assertEqual(cfg.denoise_dbscan_eps_ratio, 0.0001)
        self.assertEqual(cfg.denoise_dbscan_min_samples, 1)
        self.assertEqual(cfg.denoise_dbscan_max_points, 1000)
        self.assertEqual(cfg.denoise_sor_neighbors, 2)
        self.assertEqual(cfg.denoise_sor_std_ratio, 0.1)
        self.assertEqual(cfg.denoise_radius_neighbors, 1)
        self.assertEqual(cfg.denoise_radius_radius_ratio, 0.0001)

    def test_texture_size_zero_is_preserved_for_auto_mode(self) -> None:
        cfg = build_pipeline_config(
            {
                "texture_size": 0,
            },
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={"TEXTURE_SIZE": "2048"},
        )

        self.assertEqual(cfg.texture_size, 0)

    def test_mesh_repair_fields_are_parsed_and_clamped(self) -> None:
        cfg = build_pipeline_config(
            {
                "mesh_repair_enabled": False,
                "mesh_repair_max_diameter_ratio": -1.0,
                "mesh_repair_y_band_ratio": 9.9,
                "mesh_repair_smooth_iters": 999,
            },
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )

        self.assertFalse(cfg.mesh_repair_enabled)
        self.assertEqual(cfg.mesh_repair_max_diameter_ratio, 0.005)
        self.assertEqual(cfg.mesh_repair_y_band_ratio, 0.5)
        self.assertEqual(cfg.mesh_repair_smooth_iters, 12)


class DetectStageOutputsTests(unittest.TestCase):
    def test_stage6_and_stage7_require_mesh_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "textured_mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")
            stages, _, _ = detect_stage_outputs(out)

            self.assertFalse(stages[6])
            self.assertFalse(stages[7])
            self.assertTrue(stages[8])

    def test_stage6_complete_when_wrapped_mesh_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh_wrapped.ply").write_text("ply\n", encoding="utf-8")
            stages, _, _ = detect_stage_outputs(out)

            self.assertTrue(stages[6])

    def test_stage7_complete_when_repaired_mesh_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh_repaired.ply").write_text("ply\n", encoding="utf-8")
            stages, _, _ = detect_stage_outputs(out)

            self.assertFalse(stages[6])
            self.assertTrue(stages[7])


if __name__ == "__main__":
    unittest.main()
