"""Tests for scripts/stage_gwrapping_reconstruct.py."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from scripts.gwrapping_config import GWrappingSettings
from scripts.stage_gwrapping_reconstruct import (
    _GW_OURS_REGULARIZATION_FROM_ITER,
    _GW_RADEGS_REGULARIZATION_FROM_ITER,
    _build_extract_args,
    _build_train_args,
    _find_extracted_mesh,
    _reset_gwrapping_outputs,
    run_gwrapping,
)


def _settings(**overrides) -> GWrappingSettings:
    base = {
        "iterations": 30000,
        "rasterizer": "ours",
        "resolution": 2,
        "use_masks": True,
        "extraction_method": "pivot",
        "use_depth": True,
    }
    base.update(overrides)
    return GWrappingSettings(**base)


class TestGWrappingSettings(unittest.TestCase):
    def test_high_preset_iterations_match_config(self) -> None:
        settings = GWrappingSettings.from_preset("high")
        self.assertEqual(settings.iterations, 40000)
        self.assertEqual(settings.resolution, 1)

    def test_from_config_mapping_overrides_selected_fields(self) -> None:
        settings = GWrappingSettings.from_config_mapping(
            {
                "gwrapping_iterations": 12345,
                "gwrapping_use_depth": False,
            },
            preset="default",
        )
        self.assertEqual(settings.iterations, 12345)
        self.assertFalse(settings.use_depth)
        self.assertEqual(settings.rasterizer, "ours")


class TestBuildTrainArgs(unittest.TestCase):
    def test_train_args_include_iterations_and_ours_flags(self) -> None:
        args = _build_train_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            _settings(iterations=12345, rasterizer="ours", resolution=4),
        )
        self.assertIn("--iterations", args)
        self.assertIn("12345", args)
        self.assertIn("--rasterizer", args)
        self.assertIn("ours", args)
        self.assertIn("--feature_dc_lr", args)
        self.assertIn("--feature_rest_lr", args)
        self.assertIn("--exposure_compensation", args)
        reg_index = args.index("--regularization_from_iter")
        self.assertEqual(
            args[reg_index + 1],
            str(_GW_OURS_REGULARIZATION_FROM_ITER),
        )

    def test_train_args_disable_regularization_when_depth_is_off(self) -> None:
        args = _build_train_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            _settings(iterations=15000, use_depth=False),
        )
        reg_index = args.index("--regularization_from_iter")
        self.assertEqual(args[reg_index + 1], "15001")

    def test_train_args_include_radegs_multiview_flags(self) -> None:
        args = _build_train_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            _settings(rasterizer="radegs"),
        )
        self.assertIn("--multiview_config", args)
        self.assertIn("--multiview_factor", args)
        self.assertIn("--use_max_size_threshold", args)
        reg_index = args.index("--regularization_from_iter")
        self.assertEqual(
            args[reg_index + 1],
            str(_GW_RADEGS_REGULARIZATION_FROM_ITER),
        )


class TestBuildExtractArgs(unittest.TestCase):
    def test_extract_args_include_mask_flag_only_when_enabled(self) -> None:
        with_masks = _build_extract_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            _settings(use_masks=True),
        )
        without_masks = _build_extract_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            _settings(use_masks=False),
        )
        self.assertIn("--use_valid_mask", with_masks)
        self.assertNotIn("--use_valid_mask", without_masks)

    def test_extract_args_include_selected_iteration(self) -> None:
        args = _build_extract_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            _settings(iterations=10000, rasterizer="radegs"),
        )
        iter_index = args.index("--iteration")
        self.assertEqual(args[iter_index + 1], "10000")
        self.assertIn("--sdf_mode", args)
        self.assertIn("exact_computation", args)

    def test_unsupported_extraction_method_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            _build_extract_args(
                Path("/tmp/source"),
                Path("/tmp/model"),
                _settings(extraction_method="primal"),
            )


class TestFindExtractedMesh(unittest.TestCase):
    def test_prefers_postprocessed_mesh(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            raw = model_dir / "mesh_ours_2pivots.ply"
            refined = model_dir / "mesh_ours_2pivots_texture_refined_999.ply"
            post = model_dir / "mesh_ours_2pivots_post.ply"
            raw.write_text("raw", encoding="utf-8")
            refined.write_text("refined", encoding="utf-8")
            post.write_text("post", encoding="utf-8")

            selected = _find_extracted_mesh(model_dir, _settings())
            self.assertEqual(selected, post)

    def test_falls_back_to_latest_mesh_like_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            other = model_dir / "point_cloud.ply"
            mesh = model_dir / "fallback_mesh_result.ply"
            other.write_text("pointcloud", encoding="utf-8")
            mesh.write_text("mesh", encoding="utf-8")
            mesh.touch()

            selected = _find_extracted_mesh(
                model_dir,
                _settings(rasterizer="radegs"),
            )
            self.assertEqual(selected, mesh)


class TestResetGWrappingOutputs(unittest.TestCase):
    def test_reset_removes_workspace_filtered_sparse_and_mesh(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "gwrapping_workspace").mkdir()
            (out / "gwrapping_workspace" / "tmp.txt").write_text("x", encoding="utf-8")
            (out / "colmap_sparse_filtered").mkdir()
            (out / "colmap_sparse_filtered" / "cameras.bin").write_bytes(b"bin")
            (out / "object_mesh.ply").write_text("ply", encoding="utf-8")

            _reset_gwrapping_outputs(out)

            self.assertFalse((out / "gwrapping_workspace").exists())
            self.assertFalse((out / "colmap_sparse_filtered").exists())
            self.assertFalse((out / "object_mesh.ply").exists())


class TestRunGWrapping(unittest.TestCase):
    def test_run_gwrapping_forwards_settings_and_copies_selected_mesh(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            frames_dir = out / "frames"
            frames_dir.mkdir()
            selected_mesh = out / "selected_mesh.ply"
            selected_mesh.write_text("ply", encoding="utf-8")
            settings = _settings(
                iterations=10000,
                rasterizer="radegs",
                use_depth=False,
            )

            stub_vram = types.ModuleType("scripts.vram_utils")
            stub_vram.log_vram_detailed = MagicMock()
            stub_vram.log_vram = MagicMock()
            stub_vram.cleanup_pytorch_vram = MagicMock()

            with patch.dict(sys.modules, {"scripts.vram_utils": stub_vram}):
                with patch(
                    "scripts.stage_gwrapping_reconstruct._prepare_sparse_model",
                    return_value=out / "colmap_sparse" / "0",
                ) as mock_prepare, patch(
                    "scripts.stage_gwrapping_reconstruct._run_colmap_cmd",
                ) as mock_colmap, patch(
                    "scripts.stage_gwrapping_reconstruct._apply_masks_to_images",
                ) as mock_masks, patch(
                    "scripts.stage_gwrapping_reconstruct._setup_source_dir",
                ) as mock_setup, patch(
                    "scripts.stage_gwrapping_reconstruct._run_gwrapping_pipeline",
                ) as mock_pipeline, patch(
                    "scripts.stage_gwrapping_reconstruct._find_extracted_mesh",
                    return_value=selected_mesh,
                ) as mock_find, patch(
                    "scripts.stage_gwrapping_reconstruct.shutil.copy2",
                ) as mock_copy:
                    result = run_gwrapping(
                        str(frames_dir),
                        str(out / "colmap_sparse"),
                        str(out / "masks"),
                        str(out),
                        settings=settings,
                    )

            self.assertEqual(result, str(out / "object_mesh.ply"))
            mock_prepare.assert_called_once()
            mock_colmap.assert_called_once()
            mock_masks.assert_called_once()
            mock_setup.assert_called_once()
            mock_pipeline.assert_called_once()
            self.assertEqual(mock_pipeline.call_args.args[2], settings)
            self.assertEqual(mock_find.call_args.args[1], settings)
            self.assertEqual(
                mock_find.call_args.args[0],
                out / "gwrapping_workspace" / "model",
            )
            mock_copy.assert_called_once_with(
                str(selected_mesh),
                str(out / "object_mesh.ply"),
            )
            stub_vram.log_vram_detailed.assert_called_once()
            stub_vram.log_vram.assert_called_once()
            stub_vram.cleanup_pytorch_vram.assert_called_once()
