from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from scripts.colmap_sparse_filter import (
    ColmapCamera,
    ColmapImage,
    ColmapModel,
    ColmapPoint3D,
    SparseFilterSettings,
    filter_colmap_sparse_model,
    read_colmap_model,
    write_colmap_model,
)


def _make_test_model() -> ColmapModel:
    camera = ColmapCamera(
        camera_id=1,
        model_id=1,
        width=8,
        height=8,
        params=(10.0, 10.0, 4.0, 4.0),
    )
    xys = np.array([[2.0, 2.0], [6.0, 6.0]], dtype=np.float64)
    point_ids = np.array([1, 2], dtype=np.int64)
    image0 = ColmapImage(
        image_id=1,
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
        camera_id=1,
        name="00000.jpg",
        xys=xys,
        point3D_ids=point_ids,
    )
    image1 = ColmapImage(
        image_id=2,
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
        camera_id=1,
        name="00001.jpg",
        xys=np.array(xys, copy=True),
        point3D_ids=np.array(point_ids, copy=True),
    )
    point1 = ColmapPoint3D(
        point3D_id=1,
        xyz=(0.0, 0.0, 1.0),
        rgb=(255, 0, 0),
        error=0.1,
        track=((1, 0), (2, 0)),
    )
    point2 = ColmapPoint3D(
        point3D_id=2,
        xyz=(0.0, 0.0, 2.0),
        rgb=(0, 255, 0),
        error=0.2,
        track=((1, 1), (2, 1)),
    )
    return ColmapModel(
        cameras={1: camera},
        images={1: image0, 2: image1},
        points3D={1: point1, 2: point2},
    )


def _write_mask(path: Path, inside_pixels: list[tuple[int, int]]) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    for row, col in inside_pixels:
        mask[row, col] = 255
    Image.fromarray(mask, mode="L").save(path)


class TestColmapModelRoundTrip(unittest.TestCase):
    def test_binary_round_trip_preserves_model(self) -> None:
        model = _make_test_model()
        with TemporaryDirectory() as tmp:
            recon_dir = Path(tmp) / "recon"
            write_colmap_model(model, recon_dir)

            loaded = read_colmap_model(recon_dir)

            self.assertEqual(set(loaded.cameras), {1})
            self.assertEqual(set(loaded.images), {1, 2})
            self.assertEqual(set(loaded.points3D), {1, 2})
            np.testing.assert_array_equal(
                loaded.images[1].point3D_ids,
                np.array([1, 2], dtype=np.int64),
            )


class TestFilterColmapSparseModel(unittest.TestCase):
    def test_keeps_points_supported_by_masks_and_updates_images(self) -> None:
        model = _make_test_model()
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            recon_dir = base / "colmap_sparse" / "0"
            mask_dir = base / "masks"
            filtered_root = base / "colmap_sparse_filtered"
            mask_dir.mkdir(parents=True)
            write_colmap_model(model, recon_dir)
            _write_mask(mask_dir / "00000.png", [(2, 2)])
            _write_mask(mask_dir / "00001.png", [(2, 2)])

            result = filter_colmap_sparse_model(
                recon_dir,
                mask_dir,
                filtered_root,
                SparseFilterSettings(
                    enabled=True,
                    min_inside_views=2,
                    min_inside_ratio=0.6,
                    mask_dilate=0,
                    min_points=1,
                ),
            )

            self.assertTrue(result.used_filtered)
            self.assertEqual(result.kept_points, 1)
            filtered = read_colmap_model(filtered_root / "0")
            self.assertEqual(set(filtered.points3D), {1})
            np.testing.assert_array_equal(
                filtered.images[1].point3D_ids,
                np.array([1, -1], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                filtered.images[2].point3D_ids,
                np.array([1, -1], dtype=np.int64),
            )

    def test_falls_back_when_surviving_points_below_threshold(self) -> None:
        model = _make_test_model()
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            recon_dir = base / "colmap_sparse" / "0"
            mask_dir = base / "masks"
            filtered_root = base / "colmap_sparse_filtered"
            mask_dir.mkdir(parents=True)
            write_colmap_model(model, recon_dir)
            _write_mask(mask_dir / "00000.png", [(2, 2)])
            _write_mask(mask_dir / "00001.png", [(2, 2)])

            result = filter_colmap_sparse_model(
                recon_dir,
                mask_dir,
                filtered_root,
                SparseFilterSettings(
                    enabled=True,
                    min_inside_views=2,
                    min_inside_ratio=0.6,
                    mask_dilate=0,
                    min_points=2,
                ),
            )

            self.assertFalse(result.used_filtered)
            self.assertEqual(result.selected_recon_dir, recon_dir)
            self.assertFalse((filtered_root / "0").exists())


if __name__ == "__main__":
    unittest.main()
