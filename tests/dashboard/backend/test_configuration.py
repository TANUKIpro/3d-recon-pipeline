from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.dashboard.object_store import STAGE_RESET_PATHS
from scripts.dashboard.configuration import build_pipeline_config
from scripts.dashboard.state import detect_stage_outputs


class BuildPipelineConfigTests(unittest.TestCase):
    def test_colmap_matcher_defaults_to_sequential(self) -> None:
        cfg = build_pipeline_config(
            {},
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )
        self.assertEqual(cfg.colmap_matcher, "sequential")

    def test_colmap_max_features_clamp(self) -> None:
        cfg = build_pipeline_config(
            {"colmap_max_features": 500},
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )
        self.assertEqual(cfg.colmap_max_features, 1000)

    def test_colmap_image_size_clamp(self) -> None:
        cfg = build_pipeline_config(
            {"colmap_image_size": 100},
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )
        self.assertEqual(cfg.colmap_image_size, 256)

    def test_max_frames_clamp(self) -> None:
        cfg = build_pipeline_config(
            {"max_frames": 1},
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )
        self.assertEqual(cfg.max_frames, 2)

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

    def test_texture_view_assign_mode_defaults_to_region_gc(self) -> None:
        cfg = build_pipeline_config(
            {},
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )

        self.assertEqual(cfg.texture_view_assign_mode, "region_gc")

    def test_texture_view_assign_mode_accepts_region_gc(self) -> None:
        cfg = build_pipeline_config(
            {
                "texture_view_assign_mode": "region_gc",
            },
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={"TEXTURE_VIEW_ASSIGN_MODE": "legacy"},
        )

        self.assertEqual(cfg.texture_view_assign_mode, "region_gc")

    def test_texture_quality_boost_defaults_to_true(self) -> None:
        cfg = build_pipeline_config(
            {},
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )

        self.assertTrue(cfg.texture_quality_boost)

    def test_texture_quality_boost_accepts_true(self) -> None:
        cfg = build_pipeline_config(
            {
                "texture_quality_boost": True,
            },
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={"TEXTURE_QUALITY_BOOST": "false"},
        )

        self.assertTrue(cfg.texture_quality_boost)

    def test_gs2mesh_defaults(self) -> None:
        cfg = build_pipeline_config(
            {},
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )

        self.assertEqual(cfg.gs2mesh_gs_iterations, 30000)
        self.assertEqual(cfg.gs2mesh_runtime_profile, "auto")
        self.assertEqual(cfg.gs2mesh_stereo_model, "DLNR_Middlebury")
        self.assertEqual(cfg.gs2mesh_tsdf_voxel_size, 0.005)
        self.assertEqual(cfg.gs2mesh_tsdf_depth_trunc, 0.04)
        self.assertTrue(cfg.gs2mesh_use_masks)

    def test_gs2mesh_fields_parsed_and_clamped(self) -> None:
        cfg = build_pipeline_config(
            {
                "gs2mesh_gs_iterations": 500,
                "gs2mesh_runtime_profile": "compat",
                "gs2mesh_tsdf_voxel_size": 0.0001,
                "gs2mesh_tsdf_depth_trunc": 0.001,
                "gs2mesh_use_masks": False,
            },
            video_path="input.mp4",
            object_name="sample",
            output_dir=Path("/tmp/sample"),
            env={},
        )

        self.assertEqual(cfg.gs2mesh_gs_iterations, 1000)
        self.assertEqual(cfg.gs2mesh_runtime_profile, "compat")
        self.assertEqual(cfg.gs2mesh_tsdf_voxel_size, 0.001)
        self.assertEqual(cfg.gs2mesh_tsdf_depth_trunc, 0.005)
        self.assertFalse(cfg.gs2mesh_use_masks)


class DetectStageOutputsTests(unittest.TestCase):
    def test_stage4_reset_includes_mesh(self) -> None:
        self.assertIn("object_mesh.ply", STAGE_RESET_PATHS[4]["files"])

    def test_stage5_detected_when_obj_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "textured_mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")
            stages, _, _ = detect_stage_outputs(out)
            self.assertTrue(stages[5])

    def test_stage4_detected_when_mesh_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh.ply").write_text("ply\n", encoding="utf-8")
            stages, _, _ = detect_stage_outputs(out)
            self.assertTrue(stages[4])


if __name__ == "__main__":
    unittest.main()
