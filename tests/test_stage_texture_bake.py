import os
import unittest
from unittest.mock import patch

import numpy as np

from scripts.stage_texture_bake import (
    _apply_view_hardening,
    _compute_conflict_texels,
    _compute_face_locked_views,
    _resolve_texture_device,
    _resolve_texture_size,
    _update_topk_scores,
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


class UpdateTopKScoresTests(unittest.TestCase):
    def test_initial_insert_into_empty_slots(self) -> None:
        K = 3
        best_scores = np.full((5, K), -1.0, dtype=np.float32)
        best_views = np.full((5, K), -1, dtype=np.int32)
        valid = np.array([True, True, False, True, False])
        score = np.array([0.8, 0.5, 0.9, 0.3, 0.7], dtype=np.float64)

        _update_topk_scores(best_scores, best_views, valid, score, vidx=0)

        # Valid texels should have score in slot 0
        self.assertAlmostEqual(float(best_scores[0, 0]), 0.8, places=5)
        self.assertEqual(int(best_views[0, 0]), 0)
        self.assertAlmostEqual(float(best_scores[1, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(best_scores[3, 0]), 0.3, places=5)
        # Invalid texels should remain empty
        self.assertAlmostEqual(float(best_scores[2, 0]), -1.0, places=5)
        self.assertEqual(int(best_views[2, 0]), -1)

    def test_higher_score_replaces_lower(self) -> None:
        K = 3
        best_scores = np.array([[0.8, 0.5, 0.2]], dtype=np.float32)
        best_views = np.array([[0, 1, 2]], dtype=np.int32)
        valid = np.array([True])
        score = np.array([0.6], dtype=np.float64)

        _update_topk_scores(best_scores, best_views, valid, score, vidx=3)

        # 0.6 > 0.2 (worst), so replaces slot 2 and re-sorts
        np.testing.assert_array_almost_equal(
            best_scores[0], [0.8, 0.6, 0.5], decimal=5
        )
        np.testing.assert_array_equal(best_views[0], [0, 3, 1])

    def test_lower_score_does_not_replace(self) -> None:
        K = 3
        best_scores = np.array([[0.8, 0.5, 0.3]], dtype=np.float32)
        best_views = np.array([[0, 1, 2]], dtype=np.int32)
        valid = np.array([True])
        score = np.array([0.1], dtype=np.float64)

        _update_topk_scores(best_scores, best_views, valid, score, vidx=3)

        # 0.1 < 0.3 (worst), no change
        np.testing.assert_array_almost_equal(
            best_scores[0], [0.8, 0.5, 0.3], decimal=5
        )
        np.testing.assert_array_equal(best_views[0], [0, 1, 2])

    def test_invalid_texels_excluded(self) -> None:
        K = 2
        best_scores = np.full((3, K), -1.0, dtype=np.float32)
        best_views = np.full((3, K), -1, dtype=np.int32)
        valid = np.array([False, True, False])
        score = np.array([0.9, 0.4, 0.7], dtype=np.float64)

        _update_topk_scores(best_scores, best_views, valid, score, vidx=0)

        # Only texel 1 should be updated
        self.assertAlmostEqual(float(best_scores[0, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(best_scores[1, 0]), 0.4, places=5)
        self.assertAlmostEqual(float(best_scores[2, 0]), -1.0, places=5)

    def test_k1_single_view(self) -> None:
        K = 1
        best_scores = np.full((2, K), -1.0, dtype=np.float32)
        best_views = np.full((2, K), -1, dtype=np.int32)
        valid = np.array([True, True])
        score = np.array([0.5, 0.3], dtype=np.float64)

        _update_topk_scores(best_scores, best_views, valid, score, vidx=0)

        np.testing.assert_array_almost_equal(
            best_scores[:, 0], [0.5, 0.3], decimal=5
        )

        # Second view with higher score for texel 1
        score2 = np.array([0.4, 0.7], dtype=np.float64)
        _update_topk_scores(best_scores, best_views, valid, score2, vidx=1)

        np.testing.assert_array_almost_equal(
            best_scores[:, 0], [0.5, 0.7], decimal=5
        )
        np.testing.assert_array_equal(best_views[:, 0], [0, 1])


class BlendNormalizationTests(unittest.TestCase):
    def test_equal_scores_average_colors(self) -> None:
        """Two views with equal scores should produce average color."""
        K = 2
        n_texels = 4
        best_scores = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.8, -1.0],
            [0.5, 0.5],
        ], dtype=np.float32)

        # Simulate weight accumulation and normalization
        color_v0 = np.array([1.0, 0.0, 0.0])  # red
        color_v1 = np.array([0.0, 0.0, 1.0])  # blue

        texture = np.zeros((n_texels, 3), dtype=np.float64)
        weight_sum = np.zeros(n_texels, dtype=np.float64)

        # Texels 0,1,3 have both views; texel 2 has only view 0
        for i in [0, 1, 3]:
            texture[i] += 0.5 * color_v0 + 0.5 * color_v1
            weight_sum[i] += 1.0
        texture[2] += 0.8 * color_v0
        weight_sum[2] += 0.8

        colored = weight_sum > 0
        texture[colored] /= weight_sum[colored, None]

        # Equal weight blend → average of red and blue = purple
        expected_blend = np.array([0.5, 0.0, 0.5])
        np.testing.assert_array_almost_equal(texture[0], expected_blend, decimal=5)
        np.testing.assert_array_almost_equal(texture[1], expected_blend, decimal=5)
        np.testing.assert_array_almost_equal(texture[3], expected_blend, decimal=5)
        # Single view → just that color
        np.testing.assert_array_almost_equal(texture[2], color_v0, decimal=5)


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


class ViewHardeningTests(unittest.TestCase):
    def test_dominant_texel_zeroed(self) -> None:
        """Top-1 >> top-2 should zero out non-dominant slots."""
        best_scores = np.array([[0.9, 0.3, 0.1]], dtype=np.float32)
        best_views = np.array([[2, 5, 7]], dtype=np.int32)
        n = _apply_view_hardening(best_scores, best_views, hard_ratio=2.0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(float(best_scores[0, 0]), 0.9, places=5)
        self.assertAlmostEqual(float(best_scores[0, 1]), -1.0, places=5)
        self.assertAlmostEqual(float(best_scores[0, 2]), -1.0, places=5)
        self.assertEqual(int(best_views[0, 0]), 2)
        self.assertEqual(int(best_views[0, 1]), -1)
        self.assertEqual(int(best_views[0, 2]), -1)

    def test_competitive_texel_unchanged(self) -> None:
        """Close scores should not be hardened."""
        best_scores = np.array([[0.5, 0.4, 0.1]], dtype=np.float32)
        best_views = np.array([[1, 3, 6]], dtype=np.int32)
        scores_orig = best_scores.copy()
        views_orig = best_views.copy()
        n = _apply_view_hardening(best_scores, best_views, hard_ratio=2.0)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(best_scores, scores_orig)
        np.testing.assert_array_equal(best_views, views_orig)

    def test_disabled_when_ratio_zero(self) -> None:
        """hard_ratio=0 should disable hardening entirely."""
        best_scores = np.array([[0.9, 0.1]], dtype=np.float32)
        best_views = np.array([[0, 1]], dtype=np.int32)
        scores_orig = best_scores.copy()
        views_orig = best_views.copy()
        n = _apply_view_hardening(best_scores, best_views, hard_ratio=0.0)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(best_scores, scores_orig)
        np.testing.assert_array_equal(best_views, views_orig)

    def test_single_view_k1_noop(self) -> None:
        """K=1 should always be a no-op."""
        best_scores = np.array([[0.9]], dtype=np.float32)
        best_views = np.array([[2]], dtype=np.int32)
        n = _apply_view_hardening(best_scores, best_views, hard_ratio=2.0)
        self.assertEqual(n, 0)


class ConflictDetectionTests(unittest.TestCase):
    def test_marks_close_scores_with_large_view_separation(self) -> None:
        pos3d = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 3, axis=0)
        poses[0, :3, 3] = np.array([1.0, 0.0, 0.0])
        poses[1, :3, 3] = np.array([0.0, 0.0, 1.0])
        poses[2, :3, 3] = np.array([0.98, 0.0, 0.17])

        best_scores = np.array([
            [0.52, 0.47],
            [0.80, 0.20],
            [0.51, 0.49],
        ], dtype=np.float32)
        best_views = np.array([
            [0, 1],
            [0, 1],
            [0, 2],
        ], dtype=np.int32)

        conflict = _compute_conflict_texels(
            pos3d=pos3d,
            poses=poses,
            best_scores=best_scores,
            best_views=best_views,
            conflict_ratio=1.35,
            min_view_angle_deg=20.0,
        )

        np.testing.assert_array_equal(conflict, np.array([True, False, False]))


class FaceLockingTests(unittest.TestCase):
    def test_locks_conflict_face_to_dominant_view(self) -> None:
        fids = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
        best_scores = np.array([
            [0.50, 0.45],
            [0.48, 0.46],
            [0.49, 0.44],
            [0.51, 0.43],
            [0.80, -1.0],
            [0.75, -1.0],
        ], dtype=np.float32)
        best_views = np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, -1],
            [1, -1],
        ], dtype=np.int32)
        conflict_texels = np.array([True, True, True, True, False, False], dtype=bool)
        face_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [2, 1, 3],
        ], dtype=np.int32)

        face_locked_view, face_support = _compute_face_locked_views(
            fids=fids,
            n_faces=2,
            n_views=2,
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=faces,
        )

        np.testing.assert_array_equal(face_locked_view, np.array([0, -1], dtype=np.int32))
        self.assertGreater(float(face_support[0]), 0.0)
        self.assertAlmostEqual(float(face_support[1]), 0.0, places=5)

    def test_does_not_lock_when_dominant_view_coverage_is_too_low(self) -> None:
        fids = np.array([0, 0, 0, 0], dtype=np.int32)
        best_scores = np.array([
            [0.90, 0.10],
            [0.40, 0.30],
            [0.90, 0.10],
            [0.40, 0.30],
        ], dtype=np.float32)
        best_views = np.array([
            [0, 1],
            [1, 2],
            [0, 1],
            [1, 2],
        ], dtype=np.int32)
        conflict_texels = np.ones(4, dtype=bool)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        face_locked_view, face_support = _compute_face_locked_views(
            fids=fids,
            n_faces=1,
            n_views=3,
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=faces,
        )

        np.testing.assert_array_equal(face_locked_view, np.array([-1], dtype=np.int32))
        self.assertAlmostEqual(float(face_support[0]), 0.0, places=5)

    def test_neighbor_majority_smooths_small_face_island(self) -> None:
        fids = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
        ], dtype=np.int32)
        best_scores = np.array([
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.26, 0.25],
            [0.26, 0.25],
            [0.26, 0.25],
            [0.26, 0.25],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
        ], dtype=np.float32)
        best_views = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ], dtype=np.int32)
        conflict_texels = np.ones(12, dtype=bool)
        face_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
        ], dtype=np.int32)

        face_locked_view, _ = _compute_face_locked_views(
            fids=fids,
            n_faces=3,
            n_views=2,
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=faces,
        )

        np.testing.assert_array_equal(face_locked_view, np.array([1, 1, 1], dtype=np.int32))

    def test_smoothing_does_not_force_view_missing_from_face(self) -> None:
        fids = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
        ], dtype=np.int32)
        best_scores = np.array([
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.31, -1.0],
            [0.31, -1.0],
            [0.31, -1.0],
            [0.31, -1.0],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
            [0.49, 0.48],
        ], dtype=np.float32)
        best_views = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ], dtype=np.int32)
        conflict_texels = np.ones(12, dtype=bool)
        face_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
        ], dtype=np.int32)

        face_locked_view, _ = _compute_face_locked_views(
            fids=fids,
            n_faces=3,
            n_views=2,
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=faces,
        )

        np.testing.assert_array_equal(face_locked_view, np.array([1, 0, 1], dtype=np.int32))
