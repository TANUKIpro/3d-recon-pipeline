"""Tests for scripts.lito.tsdf_core (pure TSDF fusion core).

Phase 1.5 Step 1: empty/all-invalid input → None. Step 2-4 add synthetic
cube fusion, cleaning, decimation, smoothing, and confidence tests.
"""

from __future__ import annotations

import numpy as np

from scripts.lito.tsdf_core import TsdfFusionParams, TsdfView, fuse_tsdf


def _make_params(**overrides) -> TsdfFusionParams:
    base = dict(
        voxel_size=0.005 / 512.0,
        sdf_trunc=0.04,
        depth_min=0.1,
        depth_max=2.0,
        device="CPU:0",
    )
    base.update(overrides)
    return TsdfFusionParams(**base)


def _make_view(depth: np.ndarray, *, K=None, T_cw=None) -> TsdfView:
    h, w = depth.shape
    if K is None:
        K = np.array([[500.0, 0.0, w / 2], [0.0, 500.0, h / 2], [0.0, 0.0, 1.0]])
    if T_cw is None:
        T_cw = np.eye(4)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    return TsdfView(rgb=rgb, depth=depth.astype(np.float32), K=K, T_cw=T_cw)


class TestFuseTsdfSkeleton:
    """Step 1 (skeleton) tests: empty / no-valid-depth → None."""

    def test_empty_views_returns_none(self):
        assert fuse_tsdf([], _make_params()) is None

    def test_all_zero_depth_returns_none(self):
        depth = np.zeros((32, 32), dtype=np.float32)
        view = _make_view(depth)
        assert fuse_tsdf([view], _make_params()) is None

    def test_depth_below_min_returns_none(self):
        depth = np.full((32, 32), 0.05, dtype=np.float32)  # below depth_min=0.1
        view = _make_view(depth)
        assert fuse_tsdf([view], _make_params()) is None

    def test_depth_above_max_returns_none(self):
        depth = np.full((32, 32), 5.0, dtype=np.float32)  # above depth_max=2.0
        view = _make_view(depth)
        assert fuse_tsdf([view], _make_params()) is None

    def test_confidence_below_min_zeros_pixels_returns_none(self):
        depth = np.full((32, 32), 1.0, dtype=np.float32)
        confidence = np.full((32, 32), 0.1, dtype=np.float32)
        view = TsdfView(
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            depth=depth,
            K=np.array([[500.0, 0, 16.0], [0, 500.0, 16.0], [0, 0, 1.0]]),
            T_cw=np.eye(4),
            confidence=confidence,
            confidence_min=0.5,
        )
        # All confident pixels filtered out → no valid depth → None
        assert fuse_tsdf([view], _make_params()) is None


# Real-fusion tests (synthetic cube, decimation, smoothing, confidence)
# require Open3D and run only inside the Docker test image. Tracked as
# Phase 1.5 Step 4 (see .claude/plans/phase15_tsdf_refactor.md).
