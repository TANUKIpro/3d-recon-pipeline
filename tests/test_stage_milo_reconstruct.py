from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.milo_config import MiloSettings
from scripts.stage_milo_reconstruct import (
    _build_milo_env,
    _build_milo_extract_args,
    _build_milo_train_args,
    _ensure_sparse_0,
    _find_extracted_mesh,
    _find_recon_dir,
    _parse_milo_progress,
    _setup_milo_source_dir,
    run_milo,
)


class TestFindReconDir(unittest.TestCase):
    def test_finds_subdir_with_images_bin(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "0"
            sub.mkdir()
            (sub / "images.bin").write_bytes(b"data")

            result = _find_recon_dir(str(root))

            self.assertEqual(result, sub)

    def test_raises_when_no_valid_subdir(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "No valid COLMAP"):
                _find_recon_dir(tmp)


class TestEnsureSparse0(unittest.TestCase):
    def test_creates_0_subdir_when_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            sparse = Path(tmp) / "sparse"
            sparse.mkdir()
            (sparse / "cameras.bin").write_bytes(b"cam")

            _ensure_sparse_0(sparse)

            self.assertTrue((sparse / "0" / "cameras.bin").exists())

    def test_noop_when_0_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            sparse = Path(tmp) / "sparse"
            sub = sparse / "0"
            sub.mkdir(parents=True)
            (sub / "cameras.bin").write_bytes(b"cam")

            _ensure_sparse_0(sparse)

            self.assertTrue((sparse / "0" / "cameras.bin").exists())


class TestSetupMiloSourceDir(unittest.TestCase):
    def test_creates_symlinks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            sparse = root / "sparse"
            sparse.mkdir()
            milo_source = root / "milo_source"

            _setup_milo_source_dir(milo_source, images, sparse)

            self.assertTrue((milo_source / "images").is_symlink())
            self.assertTrue((milo_source / "sparse").is_symlink())


class TestBuildMiloTrainArgs(unittest.TestCase):
    def test_default_settings_produce_expected_args(self) -> None:
        settings = MiloSettings.from_preset("default")
        args = _build_milo_train_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            settings,
        )

        self.assertIn("train.py", args[2])
        self.assertEqual(args[args.index("-s") + 1], "/tmp/source")
        self.assertEqual(args[args.index("-m") + 1], "/tmp/model")
        self.assertEqual(
            args[args.index("--iterations") + 1],
            str(settings.iterations),
        )
        self.assertEqual(
            args[args.index("--imp_metric") + 1],
            settings.scene_type,
        )
        self.assertEqual(
            args[args.index("--rasterizer") + 1],
            settings.rasterizer,
        )

    def test_dense_gaussians_flag_added_when_enabled(self) -> None:
        settings = MiloSettings(
            iterations=7000,
            scene_type="indoor",
            mesh_config="default",
            extraction_method="sdf",
            rasterizer="radegs",
            use_masks=True,
            dense_gaussians=True,
            data_device="cuda",
            mesh_res=1024,
        )
        args = _build_milo_train_args(
            Path("/tmp/source"), Path("/tmp/model"), settings,
        )

        self.assertIn("--dense_gaussians", args)


class TestBuildMiloExtractArgs(unittest.TestCase):
    def test_sdf_extraction_method(self) -> None:
        settings = MiloSettings.from_preset("default")
        args = _build_milo_extract_args(
            Path("/tmp/source"), Path("/tmp/model"), settings,
        )

        self.assertIn("mesh_extract_sdf.py", args[2])

    def test_integration_extraction_method(self) -> None:
        settings = MiloSettings(
            iterations=18000, scene_type="indoor", mesh_config="default",
            extraction_method="integration", rasterizer="radegs",
            use_masks=True, dense_gaussians=False,
            data_device="cuda", mesh_res=1024,
        )
        args = _build_milo_extract_args(
            Path("/tmp/source"), Path("/tmp/model"), settings,
        )

        self.assertIn("mesh_extract_integration.py", args[2])

    def test_regular_tsdf_extraction_method_includes_mesh_res(self) -> None:
        settings = MiloSettings(
            iterations=18000, scene_type="indoor", mesh_config="default",
            extraction_method="regular_tsdf", rasterizer="radegs",
            use_masks=True, dense_gaussians=False,
            data_device="cuda", mesh_res=512,
        )
        args = _build_milo_extract_args(
            Path("/tmp/source"), Path("/tmp/model"), settings,
        )

        self.assertIn("mesh_extract_regular_tsdf.py", args[2])
        self.assertEqual(args[args.index("--mesh_res") + 1], "512")


class TestBuildMiloEnv(unittest.TestCase):
    def test_adds_milo_base_to_pythonpath(self) -> None:
        env = _build_milo_env()

        self.assertIn("/opt/MILo", env["PYTHONPATH"])


class TestParseMiloProgress(unittest.TestCase):
    def test_parses_iteration_line(self) -> None:
        calls = []
        _parse_milo_progress(
            "[ITER 9000] Loss: 0.123",
            18000,
            lambda pct, msg: calls.append((pct, msg)),
        )

        self.assertEqual(len(calls), 1)
        pct, msg = calls[0]
        # 9000/18000 = 0.5, mapped to 5.0 + 0.5*80 = 45.0
        self.assertAlmostEqual(pct, 45.0)
        self.assertIn("9000", msg)

    def test_ignores_non_iter_line(self) -> None:
        calls = []
        _parse_milo_progress(
            "Loading model...",
            18000,
            lambda pct, msg: calls.append((pct, msg)),
        )

        self.assertEqual(len(calls), 0)


class TestFindExtractedMesh(unittest.TestCase):
    def test_sdf_method_returns_expected_path(self) -> None:
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            result = _find_extracted_mesh(model_dir, "sdf")

            self.assertEqual(result, model_dir / "mesh_learnable_sdf.ply")

    def test_integration_method_returns_expected_path(self) -> None:
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            result = _find_extracted_mesh(model_dir, "integration")

            self.assertEqual(result, model_dir / "mesh_integration_sdf.ply")

    def test_regular_tsdf_method_finds_glob(self) -> None:
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            mesh_file = model_dir / "mesh_regular_tsdf_res512.ply"
            mesh_file.write_bytes(b"ply")

            result = _find_extracted_mesh(model_dir, "regular_tsdf")

            self.assertEqual(result, mesh_file)


class TestRunMiloIntegration(unittest.TestCase):
    @patch("scripts.stage_milo_reconstruct.shutil.copy2")
    @patch("scripts.stage_milo_reconstruct._find_extracted_mesh")
    @patch("scripts.stage_milo_reconstruct._run_subprocess")
    @patch("scripts.stage_milo_reconstruct._setup_milo_source_dir")
    @patch("scripts.stage_milo_reconstruct._apply_masks_to_images")
    @patch("scripts.stage_milo_reconstruct._ensure_sparse_0")
    @patch("scripts.stage_milo_reconstruct._run_colmap_cmd")
    @patch("scripts.stage_milo_reconstruct._find_recon_dir")
    def test_run_milo_completes_pipeline(
        self,
        mock_find_recon,
        mock_colmap_cmd,
        mock_ensure_sparse,
        mock_apply_masks,
        mock_setup_source,
        mock_run_subprocess,
        mock_find_mesh,
        mock_copy2,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            frames_dir = Path(tmp) / "frames"
            colmap_sparse = Path(tmp) / "colmap_sparse"
            frames_dir.mkdir(parents=True)
            colmap_sparse.mkdir(parents=True)

            mock_find_recon.return_value = colmap_sparse / "0"

            mesh_file = Path(tmp) / "extracted.ply"
            mesh_file.write_bytes(b"ply")
            mock_find_mesh.return_value = mesh_file

            settings = MiloSettings.from_preset("default")

            with patch("scripts.vram_utils.cleanup_pytorch_vram"), \
                 patch("scripts.vram_utils.log_vram"), \
                 patch("scripts.vram_utils.log_vram_detailed"):
                result = run_milo(
                    str(frames_dir),
                    str(colmap_sparse),
                    str(Path(tmp) / "masks"),
                    str(out),
                    settings=settings,
                )

            mock_colmap_cmd.assert_called_once()
            self.assertEqual(mock_run_subprocess.call_count, 2)
            mock_copy2.assert_called_once()

    @patch("scripts.stage_milo_reconstruct.shutil.copy2")
    @patch("scripts.stage_milo_reconstruct._find_extracted_mesh")
    @patch("scripts.stage_milo_reconstruct._run_subprocess")
    @patch("scripts.stage_milo_reconstruct._setup_milo_source_dir")
    @patch("scripts.stage_milo_reconstruct._apply_masks_to_images")
    @patch("scripts.stage_milo_reconstruct._ensure_sparse_0")
    @patch("scripts.stage_milo_reconstruct._run_colmap_cmd")
    @patch("scripts.stage_milo_reconstruct._find_recon_dir")
    def test_run_milo_applies_masks_when_enabled(
        self,
        mock_find_recon,
        mock_colmap_cmd,
        mock_ensure_sparse,
        mock_apply_masks,
        mock_setup_source,
        mock_run_subprocess,
        mock_find_mesh,
        mock_copy2,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            frames_dir = Path(tmp) / "frames"
            colmap_sparse = Path(tmp) / "colmap_sparse"
            masks_dir = Path(tmp) / "masks"
            frames_dir.mkdir(parents=True)
            colmap_sparse.mkdir(parents=True)
            masks_dir.mkdir(parents=True)

            mock_find_recon.return_value = colmap_sparse / "0"
            mesh_file = Path(tmp) / "extracted.ply"
            mesh_file.write_bytes(b"ply")
            mock_find_mesh.return_value = mesh_file

            settings = MiloSettings.from_preset("default")
            assert settings.use_masks is True

            with patch("scripts.vram_utils.cleanup_pytorch_vram"), \
                 patch("scripts.vram_utils.log_vram"), \
                 patch("scripts.vram_utils.log_vram_detailed"):
                run_milo(
                    str(frames_dir),
                    str(colmap_sparse),
                    str(masks_dir),
                    str(out),
                    settings=settings,
                )

            mock_apply_masks.assert_called_once()

    @patch("scripts.stage_milo_reconstruct.shutil.copy2")
    @patch("scripts.stage_milo_reconstruct._find_extracted_mesh")
    @patch("scripts.stage_milo_reconstruct._run_subprocess")
    @patch("scripts.stage_milo_reconstruct._setup_milo_source_dir")
    @patch("scripts.stage_milo_reconstruct._apply_masks_to_images")
    @patch("scripts.stage_milo_reconstruct._ensure_sparse_0")
    @patch("scripts.stage_milo_reconstruct._run_colmap_cmd")
    @patch("scripts.stage_milo_reconstruct._find_recon_dir")
    def test_run_milo_skips_masks_when_disabled(
        self,
        mock_find_recon,
        mock_colmap_cmd,
        mock_ensure_sparse,
        mock_apply_masks,
        mock_setup_source,
        mock_run_subprocess,
        mock_find_mesh,
        mock_copy2,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            frames_dir = Path(tmp) / "frames"
            colmap_sparse = Path(tmp) / "colmap_sparse"
            frames_dir.mkdir(parents=True)
            colmap_sparse.mkdir(parents=True)

            mock_find_recon.return_value = colmap_sparse / "0"
            mesh_file = Path(tmp) / "extracted.ply"
            mesh_file.write_bytes(b"ply")
            mock_find_mesh.return_value = mesh_file

            settings = MiloSettings(
                iterations=18000, scene_type="indoor",
                mesh_config="default", extraction_method="sdf",
                rasterizer="radegs", use_masks=False,
                dense_gaussians=False, data_device="cuda",
                mesh_res=1024,
            )

            with patch("scripts.vram_utils.cleanup_pytorch_vram"), \
                 patch("scripts.vram_utils.log_vram"), \
                 patch("scripts.vram_utils.log_vram_detailed"):
                run_milo(
                    str(frames_dir),
                    str(colmap_sparse),
                    None,
                    str(out),
                    settings=settings,
                )

            mock_apply_masks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
