from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from scripts.gwrapping_config import GWrappingSettings
from scripts.stage_gwrapping_reconstruct import run_gwrapping


class TestRunGWrapping(unittest.TestCase):
    """Tests for run_gwrapping with GWrappingSettings."""

    @patch("scripts.stage_gwrapping_reconstruct.shutil.copy2")
    @patch("scripts.stage_gwrapping_reconstruct._run_gwrapping_pipeline")
    def test_run_gwrapping_forwards_settings(
        self,
        mock_run_pipeline,
        mock_copy2,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            frames_dir = Path(tmp) / "frames"
            colmap_sparse = Path(tmp) / "colmap_sparse"
            mask_dir = Path(tmp) / "masks"
            frames_dir.mkdir(parents=True)
            colmap_sparse.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            settings = GWrappingSettings(
                iterations=30000,
                rasterizer="ours",
                resolution=2,
                use_masks=True,
                use_depth=True,
                extraction_method="pivot",
            )

            run_gwrapping(
                str(frames_dir),
                str(colmap_sparse),
                str(mask_dir),
                str(out),
                settings=settings,
                progress_cb=None,
                cancel_cb=None,
                register_process=None,
                unregister_process=None,
            )

            mock_run_pipeline.assert_called_once()

    @patch("scripts.stage_gwrapping_reconstruct.shutil.copy2")
    @patch("scripts.stage_gwrapping_reconstruct._run_gwrapping_pipeline")
    def test_run_gwrapping_default_settings(
        self,
        mock_run_pipeline,
        mock_copy2,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            frames_dir = Path(tmp) / "frames"
            colmap_sparse = Path(tmp) / "colmap_sparse"
            mask_dir = Path(tmp) / "masks"
            frames_dir.mkdir(parents=True)
            colmap_sparse.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            run_gwrapping(
                str(frames_dir),
                str(colmap_sparse),
                str(mask_dir),
                str(out),
            )

            mock_run_pipeline.assert_called_once()

    @patch("scripts.stage_gwrapping_reconstruct.shutil.copy2")
    @patch("scripts.stage_gwrapping_reconstruct._run_gwrapping_pipeline")
    def test_run_gwrapping_with_callbacks(
        self,
        mock_run_pipeline,
        mock_copy2,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            frames_dir = Path(tmp) / "frames"
            colmap_sparse = Path(tmp) / "colmap_sparse"
            mask_dir = Path(tmp) / "masks"
            frames_dir.mkdir(parents=True)
            colmap_sparse.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            progress_cb = MagicMock()
            cancel_cb = MagicMock()
            register_process = MagicMock()
            unregister_process = MagicMock()

            run_gwrapping(
                str(frames_dir),
                str(colmap_sparse),
                str(mask_dir),
                str(out),
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                register_process=register_process,
                unregister_process=unregister_process,
            )

            mock_run_pipeline.assert_called_once()


class TestGWrappingSettings(unittest.TestCase):
    """Tests for GWrappingSettings dataclass."""

    def test_default_settings(self) -> None:
        settings = GWrappingSettings()
        self.assertEqual(settings.iterations, 30000)
        self.assertEqual(settings.rasterizer, "ours")
        self.assertEqual(settings.resolution, 2)
        self.assertTrue(settings.use_masks)
        self.assertTrue(settings.use_depth)
        self.assertEqual(settings.extraction_method, "pivot")

    def test_custom_settings(self) -> None:
        settings = GWrappingSettings(
            iterations=10000,
            rasterizer="radegs",
            resolution=4,
            use_masks=False,
            use_depth=False,
            extraction_method="primal",
        )
        self.assertEqual(settings.iterations, 10000)
        self.assertEqual(settings.rasterizer, "radegs")
        self.assertEqual(settings.resolution, 4)
        self.assertFalse(settings.use_masks)
        self.assertFalse(settings.use_depth)
        self.assertEqual(settings.extraction_method, "primal")

    def test_from_preset_default(self) -> None:
        settings = GWrappingSettings.from_preset("default")
        self.assertEqual(settings.iterations, 30000)
        self.assertEqual(settings.rasterizer, "ours")
        self.assertEqual(settings.resolution, 2)
        self.assertTrue(settings.use_masks)
        self.assertTrue(settings.use_depth)
        self.assertEqual(settings.extraction_method, "pivot")

    def test_from_preset_high(self) -> None:
        settings = GWrappingSettings.from_preset("high")
        self.assertEqual(settings.iterations, 30000)

    def test_from_preset_fast(self) -> None:
        settings = GWrappingSettings.from_preset("fast")
        self.assertIsNotNone(settings)


if __name__ == "__main__":
    unittest.main()
