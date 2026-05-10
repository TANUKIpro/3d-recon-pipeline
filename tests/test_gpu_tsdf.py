from __future__ import annotations

import os
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.gpu_tsdf import (
    _apply_silhouette_depth_fill,
    _carve_visual_hull_occupancy,
    _load_visual_hull_extra_views,
    _resolve_silhouette_fill_config,
)


class TestSilhouetteDepthFill(unittest.TestCase):
    def test_fills_only_invalid_masked_pixels(self) -> None:
        depth = np.array(
            [
                [0.0, 0.70, 0.0],
                [1.20, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        object_mask = np.array(
            [
                [True, True, False],
                [True, False, True],
            ],
            dtype=bool,
        )
        hull_depth = np.array(
            [
                [0.50, 0.60, 0.55],
                [0.80, 0.90, np.inf],
            ],
            dtype=np.float32,
        )

        filled, count = _apply_silhouette_depth_fill(
            depth,
            object_mask,
            hull_depth,
            depth_min=0.1,
            depth_max=1.0,
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(filled[0, 0]), 0.50)
        self.assertAlmostEqual(float(filled[1, 0]), 0.80)
        self.assertAlmostEqual(float(filled[0, 1]), 0.70)
        self.assertEqual(float(filled[0, 2]), 0.0)
        self.assertEqual(float(filled[1, 1]), 0.0)
        self.assertEqual(float(filled[1, 2]), 0.0)

    def test_replace_mode_overwrites_all_masked_pixels(self) -> None:
        depth = np.array(
            [
                [0.40, 0.70, 0.30],
                [1.20, 0.45, 0.20],
            ],
            dtype=np.float32,
        )
        object_mask = np.array(
            [
                [True, True, False],
                [True, True, True],
            ],
            dtype=bool,
        )
        hull_depth = np.array(
            [
                [0.50, np.inf, 0.55],
                [0.80, 0.90, np.inf],
            ],
            dtype=np.float32,
        )

        filled, count = _apply_silhouette_depth_fill(
            depth,
            object_mask,
            hull_depth,
            depth_min=0.1,
            depth_max=1.0,
            mode="replace",
        )

        self.assertEqual(count, 3)
        self.assertAlmostEqual(float(filled[0, 0]), 0.50)
        self.assertAlmostEqual(float(filled[1, 0]), 0.80)
        self.assertAlmostEqual(float(filled[1, 1]), 0.90)
        self.assertEqual(float(filled[0, 1]), 0.0)
        self.assertEqual(float(filled[1, 2]), 0.0)
        self.assertAlmostEqual(float(filled[0, 2]), 0.30)

    def test_default_mask_depth_mode_keeps_crop_behavior(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = _resolve_silhouette_fill_config()

        self.assertFalse(config.enabled)
        self.assertEqual(config.depth_mode, "crop")

    def test_explicit_replace_mode_enables_sam2_primary_depth(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = _resolve_silhouette_fill_config(
                mask_depth_mode="replace",
            )

        self.assertTrue(config.enabled)
        self.assertEqual(config.depth_mode, "replace")

    def test_mask_depth_mode_can_use_hybrid_fill_behavior(self) -> None:
        with patch.dict(
            os.environ,
            {"GS2MESH_MASK_DEPTH_MODE": "fill"},
            clear=True,
        ):
            config = _resolve_silhouette_fill_config()

        self.assertTrue(config.enabled)
        self.assertEqual(config.depth_mode, "fill")

    def test_legacy_silhouette_fill_env_enables_hybrid_fill(self) -> None:
        with patch.dict(
            os.environ,
            {"GS2MESH_SILHOUETTE_FILL": "true"},
            clear=True,
        ):
            config = _resolve_silhouette_fill_config()

        self.assertEqual(config.depth_mode, "fill")


class TestVisualHullOccupancy(unittest.TestCase):
    def test_loads_extra_visual_hull_views_from_manifest(self) -> None:
        from PIL import Image

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "mask.png"
            Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(
                mask_path
            )
            manifest_path = root / "extra_views.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "views": [
                            {
                                "mask_path": str(mask_path),
                                "camera": {
                                    "extrinsic": np.eye(4).tolist(),
                                    "fx": 1.0,
                                    "fy": 1.0,
                                    "cx": 0.5,
                                    "cy": 0.5,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            views = _load_visual_hull_extra_views(
                manifest_path,
                mask_dilate_px=0,
            )

            self.assertEqual(len(views), 1)
            camera, mask = views[0]
            self.assertEqual(camera["fx"], 1.0)
            self.assertTrue(mask[0, 1])

    def test_carves_occupancy_from_mask_projection(self) -> None:
        camera = {
            "extrinsic": np.eye(4).tolist(),
            "fx": 20.0,
            "fy": 20.0,
            "cx": 10.0,
            "cy": 10.0,
        }
        mask = np.zeros((20, 20), dtype=bool)
        mask[7:13, 7:13] = True

        occupancy, voxel_size = _carve_visual_hull_occupancy(
            [camera],
            {0: mask},
            bbox_min=np.array([-0.25, -0.25, 1.0], dtype=np.float64),
            bbox_max=np.array([0.25, 0.25, 1.4], dtype=np.float64),
            voxel_resolution=10,
            min_views=1,
            consensus=1.0,
            depth_min=0.5,
            depth_max=2.0,
            tsdf_scale=1.0,
        )

        self.assertGreater(voxel_size, 0.0)
        self.assertTrue(bool(occupancy.any()))
        self.assertFalse(bool(occupancy.all()))

    def test_carve_occupancy_reports_chunk_progress(self) -> None:
        camera = {
            "extrinsic": np.eye(4).tolist(),
            "fx": 20.0,
            "fy": 20.0,
            "cx": 10.0,
            "cy": 10.0,
        }
        mask = np.zeros((20, 20), dtype=bool)
        mask[7:13, 7:13] = True
        progress: list[tuple[int, int]] = []

        _carve_visual_hull_occupancy(
            [camera],
            {0: mask},
            bbox_min=np.array([-0.25, -0.25, 1.0], dtype=np.float64),
            bbox_max=np.array([0.25, 0.25, 1.4], dtype=np.float64),
            voxel_resolution=10,
            min_views=1,
            consensus=1.0,
            depth_min=0.5,
            depth_max=2.0,
            tsdf_scale=1.0,
            progress_cb=lambda current, total: progress.append(
                (current, total)
            ),
        )

        self.assertTrue(progress)
        self.assertEqual(progress[-1][0], progress[-1][1])
        self.assertGreaterEqual(progress[-1][1], 1)


if __name__ == "__main__":
    unittest.main()
