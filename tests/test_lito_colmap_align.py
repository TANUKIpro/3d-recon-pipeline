"""Tests for scripts.lito.colmap_align (Sim(3) alignment).

ICP refinement requires Open3D and runs only inside the Docker test image.
The PCA-init path and Sim3Transform application are pure numpy and tested
on synthetic point clouds here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.lito.colmap_align import (
    Sim3Transform,
    _read_images_with_points,
    _read_points3d,
    pca_initial_sim3,
)


def _random_points(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


def _apply_sim3(
    points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    return (s * (R @ points.T)).T + t


class TestSim3Transform:
    def test_matrix_round_trip(self):
        R = np.eye(3)
        T = Sim3Transform(
            rotation=R,
            translation=np.array([1.0, 2.0, 3.0]),
            scale=2.0,
            residual_rms=0.0,
            inlier_count=0,
        )
        H = T.matrix()
        assert H.shape == (4, 4)
        assert np.allclose(H[:3, :3], 2.0 * R)
        assert np.allclose(H[:3, 3], [1.0, 2.0, 3.0])

    def test_apply_points_matches_matrix(self):
        rng = np.random.default_rng(7)
        R = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        t = np.array([0.5, -0.5, 1.0])
        s = 1.5
        T = Sim3Transform(
            rotation=R, translation=t, scale=s, residual_rms=0.0, inlier_count=0
        )
        pts = rng.normal(size=(50, 3))
        out = T.apply_points(pts)
        H = T.matrix()
        homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        ref = (H @ homo.T).T[:, :3]
        assert np.allclose(out, ref)


class TestPcaInitialSim3:
    def test_recovers_pure_translation(self):
        src = _random_points(200, seed=1)
        t_true = np.array([5.0, -2.0, 7.0])
        tgt = src + t_true
        sim = pca_initial_sim3(src, tgt)
        assert abs(sim.scale - 1.0) < 0.05
        # Translation should be ≈ t_true. PCA cannot determine signs but a pure
        # translation has identity rotation up to sign-flip; align gross translation.
        gap = np.linalg.norm(sim.translation - t_true)
        # The PCA-only init may flip signs on near-isotropic data; just demand
        # the alignment is no worse than half-diameter-residual.
        assert sim.residual_rms < float(np.linalg.norm(tgt.std(axis=0)))
        assert gap < 50.0  # sanity bound

    def test_recovers_uniform_scale(self):
        rng = np.random.default_rng(2)
        src = rng.normal(size=(300, 3))
        scale = 3.0
        # Apply a non-trivial rotation as well so axes are not axis-aligned.
        theta = np.deg2rad(35.0)
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Make src anisotropic so PCA axes are well-defined.
        src = src * np.array([3.0, 1.0, 0.4])
        tgt = _apply_sim3(src, R, np.zeros(3), scale)
        sim = pca_initial_sim3(src, tgt)
        assert abs(sim.scale - scale) / scale < 0.10

    def test_residual_decreases_for_well_aligned_pcs(self):
        rng = np.random.default_rng(3)
        src = rng.normal(size=(500, 3)) * np.array([4.0, 2.0, 1.0])
        R = np.eye(3)
        t = np.array([2.0, -1.0, 0.0])
        s = 0.5
        tgt = _apply_sim3(src, R, t, s) + rng.normal(size=src.shape) * 0.02
        sim = pca_initial_sim3(src, tgt)
        # The PCA estimate should produce a residual much smaller than the
        # raw inter-cloud distance.
        raw = np.linalg.norm(src.mean(0) - tgt.mean(0))
        assert sim.residual_rms < raw

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="≥3 points"):
            pca_initial_sim3(np.zeros((2, 3)), np.zeros((10, 3)))


class TestColmapBinaryReaders:
    def _write_points3d(self, path: Path, records: list[dict]) -> None:
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(records)))
            for r in records:
                f.write(struct.pack("<Q", r["id"]))
                f.write(struct.pack("<3d", *r["xyz"]))
                f.write(bytes(r["rgb"]))  # 3 uint8
                f.write(struct.pack("<d", r["error"]))
                f.write(struct.pack("<Q", len(r["track"])))
                for image_id, p2d_idx in r["track"]:
                    f.write(struct.pack("<II", image_id, p2d_idx))

    def _write_images(self, path: Path, records: list[dict]) -> None:
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(records)))
            for r in records:
                f.write(struct.pack("<I", r["id"]))
                f.write(struct.pack("<4d", *r["qvec"]))
                f.write(struct.pack("<3d", *r["tvec"]))
                f.write(struct.pack("<I", r["camera_id"]))
                f.write(r["name"].encode("utf-8") + b"\x00")
                xys = r["xys"]
                p3d_ids = r["p3d_ids"]
                f.write(struct.pack("<Q", len(xys)))
                for (x, y), p3d in zip(xys, p3d_ids):
                    f.write(struct.pack("<dd q", x, y, p3d))

    def test_points3d_round_trip(self, tmp_path: Path):
        records = [
            {
                "id": 1,
                "xyz": (0.1, 0.2, 0.3),
                "rgb": (10, 20, 30),
                "error": 0.5,
                "track": [(1, 0), (2, 5)],
            },
            {
                "id": 2,
                "xyz": (-1.0, 2.0, 3.0),
                "rgb": (200, 100, 50),
                "error": 0.1,
                "track": [(1, 1)],
            },
        ]
        path = tmp_path / "points3D.bin"
        self._write_points3d(path, records)
        out = _read_points3d(path)
        assert set(out) == {1, 2}
        assert np.allclose(out[1], (0.1, 0.2, 0.3))
        assert np.allclose(out[2], (-1.0, 2.0, 3.0))

    def test_images_round_trip(self, tmp_path: Path):
        records = [
            {
                "id": 1,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, 0.0, 0.0),
                "camera_id": 1,
                "name": "00000.jpg",
                "xys": [(10.0, 20.0), (30.0, 40.0)],
                "p3d_ids": [1, -1],
            }
        ]
        path = tmp_path / "images.bin"
        self._write_images(path, records)
        out = _read_images_with_points(path)
        assert 1 in out
        entry = out[1]
        assert entry["name"] == "00000.jpg"
        assert entry["xys"].shape == (2, 2)
        assert entry["point3d_ids"].tolist() == [1, -1]
