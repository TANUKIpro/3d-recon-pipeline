import os
import unittest
from unittest.mock import patch

import numpy as np

from scripts.stage_texture_bake import (
    _build_face_charts,
    _kdtree_color_interpolation,
    _rank_chart_view_candidates,
    _resolve_texture_device,
    _resolve_texture_size,
    _secondary_min_cos_levels,
)


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class _FakeTorch:
    cuda = _FakeCuda()


class _FakeCudaUnavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorchUnavailable:
    cuda = _FakeCudaUnavailable()


class ResolveTextureDeviceTests(unittest.TestCase):
    def test_default_prefers_cuda(self) -> None:
        old = os.environ.pop("TEXTURE_DEVICE", None)
        try:
            with patch("scripts.stage_texture_bake.torch", _FakeTorch()):
                self.assertEqual(_resolve_texture_device(), "cuda")
        finally:
            if old is not None:
                os.environ["TEXTURE_DEVICE"] = old

    def test_default_falls_back_to_cpu_without_cuda(self) -> None:
        old = os.environ.pop("TEXTURE_DEVICE", None)
        try:
            with patch("scripts.stage_texture_bake.torch", _FakeTorchUnavailable()):
                self.assertEqual(_resolve_texture_device(), "cpu")
        finally:
            if old is not None:
                os.environ["TEXTURE_DEVICE"] = old

    def test_auto_without_torch_falls_back_to_cpu(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "auto"}, clear=False):
            with patch("scripts.stage_texture_bake.torch", None):
                self.assertEqual(_resolve_texture_device(), "cpu")

    def test_cuda_without_torch_falls_back_to_cpu(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "cuda"}, clear=False):
            with patch("scripts.stage_texture_bake.torch", None):
                self.assertEqual(_resolve_texture_device(), "cpu")

    def test_cpu_mode_is_always_cpu(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "cpu"}, clear=False):
            with patch("scripts.stage_texture_bake.torch", _FakeTorch()):
                self.assertEqual(_resolve_texture_device(), "cpu")

    def test_auto_with_cuda_available_uses_cuda(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "auto"}, clear=False):
            with patch("scripts.stage_texture_bake.torch", _FakeTorch()):
                self.assertEqual(_resolve_texture_device(), "cuda")

    def test_gpu_alias_maps_to_cuda(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "gpu"}, clear=False):
            with patch("scripts.stage_texture_bake.torch", _FakeTorch()):
                self.assertEqual(_resolve_texture_device(), "cuda")


class BuildFaceChartsTests(unittest.TestCase):
    def test_faces_connected_by_edge_share_chart(self) -> None:
        faces = np.array(
            [
                [0, 1, 2],
                [2, 1, 3],  # shares edge (1,2) with face 0
                [4, 5, 6],  # separate island
            ],
            dtype=np.int32,
        )
        chart_ids, n_charts = _build_face_charts(faces)

        self.assertEqual(n_charts, 2)
        self.assertEqual(int(chart_ids[0]), int(chart_ids[1]))
        self.assertNotEqual(int(chart_ids[0]), int(chart_ids[2]))


class RankChartViewCandidatesTests(unittest.TestCase):
    def test_ranking_prioritizes_coverage_then_score(self) -> None:
        chart_candidates = [
            [(0.40, 10.0, 0), (0.60, 1.0, 1), (0.60, 0.5, 2)],
            [],
        ]
        primary, orders = _rank_chart_view_candidates(chart_candidates, n_views=4)

        self.assertEqual(int(primary[0]), 1)
        self.assertEqual(orders[0][:3], [1, 2, 0])
        self.assertEqual(int(primary[1]), -1)
        self.assertEqual(orders[1][:4], [0, 1, 2, 3])


class SecondaryMinCosLevelsTests(unittest.TestCase):
    def test_relaxed_levels_are_unique_and_strictly_lower(self) -> None:
        levels = _secondary_min_cos_levels(0.2)

        self.assertGreater(len(levels), 0)
        self.assertEqual(levels, sorted(levels, reverse=True))
        for level in levels:
            self.assertLess(level, 0.2)
        self.assertEqual(len(levels), len(set(levels)))


class ResolveTextureSizeTests(unittest.TestCase):
    def test_manual_size_kept(self) -> None:
        size, is_auto = _resolve_texture_size(2048, 1920, 1080)
        self.assertEqual(size, 2048)
        self.assertFalse(is_auto)

    def test_auto_size_from_video_pixels(self) -> None:
        size, is_auto = _resolve_texture_size(0, 1920, 1080)
        self.assertEqual(size, 1440)
        self.assertTrue(is_auto)

    def test_none_uses_env_auto_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            old = os.environ.pop("TEXTURE_SIZE", None)
            try:
                size, is_auto = _resolve_texture_size(None, 1280, 720)
                self.assertEqual(size, 960)
                self.assertTrue(is_auto)
            finally:
                if old is not None:
                    os.environ["TEXTURE_SIZE"] = old

    def test_invalid_env_value_falls_back_to_auto(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_SIZE": "invalid"}, clear=False):
            size, is_auto = _resolve_texture_size(None, 640, 360)
            self.assertEqual(size, 480)
            self.assertTrue(is_auto)


class TestKDTreeColorInterpolation(unittest.TestCase):
    """Tests for _kdtree_color_interpolation."""

    def _make_cube_pc(self):
        """4 points at cube corners with distinct colours."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)
        return points, colors

    def test_basic_interpolation(self) -> None:
        """Texel near a known point should get a colour close to that point."""
        pc_points, pc_colors = self._make_cube_pc()
        texels = np.array([[0.01, 0.01, 0.01]], dtype=np.float64)

        colors, covered = _kdtree_color_interpolation(
            texels, pc_points, pc_colors, k=4, max_distance=5.0,
        )
        self.assertTrue(covered[0])
        # Closest point is [0,0,0] with colour [1,0,0]; IDW should favour it.
        self.assertGreater(colors[0, 0], 0.5)  # red channel dominant

    def test_uncovered_texels(self) -> None:
        """Texels beyond max_distance should be uncovered."""
        pc_points, pc_colors = self._make_cube_pc()
        texels = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)

        colors, covered = _kdtree_color_interpolation(
            texels, pc_points, pc_colors, k=4, max_distance=1.0,
        )
        self.assertFalse(covered[0])
        np.testing.assert_array_equal(colors[0], [0.0, 0.0, 0.0])

    def test_auto_max_distance(self) -> None:
        """max_distance=0 should auto-compute a positive value and cover nearby texels."""
        pc_points, pc_colors = self._make_cube_pc()
        texels = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)

        colors, covered = _kdtree_color_interpolation(
            texels, pc_points, pc_colors, k=4, max_distance=0.0,
        )
        self.assertTrue(covered[0])
        # Should be a blend, sum of channels roughly > 0
        self.assertGreater(np.sum(colors[0]), 0.1)

    def test_k_equals_one(self) -> None:
        """k=1 should return the exact nearest neighbour colour."""
        pc_points, pc_colors = self._make_cube_pc()
        texels = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        colors, covered = _kdtree_color_interpolation(
            texels, pc_points, pc_colors, k=1, max_distance=5.0,
        )
        self.assertTrue(covered[0])
        np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=1e-5)

    def test_batch_boundary(self) -> None:
        """Results should be consistent regardless of batch size."""
        pc_points, pc_colors = self._make_cube_pc()
        texels = np.tile([[0.5, 0.5, 0.0]], (120, 1)).astype(np.float64)

        c1, cov1 = _kdtree_color_interpolation(
            texels, pc_points, pc_colors, k=4, max_distance=5.0, batch_size=50,
        )
        c2, cov2 = _kdtree_color_interpolation(
            texels, pc_points, pc_colors, k=4, max_distance=5.0, batch_size=200,
        )
        np.testing.assert_array_equal(cov1, cov2)
        np.testing.assert_allclose(c1, c2, atol=1e-6)
