import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from scripts.stage_texture_bake import (
    _apply_narrow_seam_leveling,
    _apply_view_hardening,
    _compute_conflict_texels,
    _compute_face_locked_views,
    _compute_region_gc_locked_views,
    _estimate_detail_band_shift,
    _normalize_candidate_patch_colors,
    _refine_detail_band_shift,
    _resolve_texture_device,
    _resolve_texture_quality_boost,
    _resolve_texture_size,
    _resolve_texture_view_assign_mode,
    _load_poses,
    _select_detail_companion_view_candidates,
    _select_detail_companion_views,
    _translate_patch,
    _update_topk_scores,
)
from scripts.texture.bake import (
    _maybe_simplify_mesh_for_uv,
    _resolve_intrinsics_point_cloud,
)
from scripts.texture.config import _resolve_uv_face_budget


def _make_mock_torch(cuda_available: bool = True) -> MagicMock:
    """Return a ``MagicMock`` that behaves like the ``torch`` module."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = cuda_available
    return mock


class ResolveTextureDeviceTests(unittest.TestCase):
    def test_default_prefers_cuda(self) -> None:
        old = os.environ.pop("TEXTURE_DEVICE", None)
        try:
            with patch("scripts.texture.config.torch", _make_mock_torch(cuda_available=True)):
                self.assertEqual(_resolve_texture_device(), "cuda")
        finally:
            if old is not None:
                os.environ["TEXTURE_DEVICE"] = old

    def test_default_falls_back_to_cpu_without_cuda(self) -> None:
        old = os.environ.pop("TEXTURE_DEVICE", None)
        try:
            with patch("scripts.texture.config.torch", _make_mock_torch(cuda_available=False)):
                self.assertEqual(_resolve_texture_device(), "cpu")
        finally:
            if old is not None:
                os.environ["TEXTURE_DEVICE"] = old

    def test_auto_without_torch_falls_back_to_cpu(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "auto"}, clear=False):
            with patch("scripts.texture.config.torch", None):
                self.assertEqual(_resolve_texture_device(), "cpu")

    def test_cuda_without_torch_falls_back_to_cpu(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "cuda"}, clear=False):
            with patch("scripts.texture.config.torch", None):
                self.assertEqual(_resolve_texture_device(), "cpu")

    def test_cpu_mode_is_always_cpu(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "cpu"}, clear=False):
            with patch("scripts.texture.config.torch", _make_mock_torch(cuda_available=True)):
                self.assertEqual(_resolve_texture_device(), "cpu")

    def test_auto_with_cuda_available_uses_cuda(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "auto"}, clear=False):
            with patch("scripts.texture.config.torch", _make_mock_torch(cuda_available=True)):
                self.assertEqual(_resolve_texture_device(), "cuda")

    def test_gpu_alias_maps_to_cuda(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_DEVICE": "gpu"}, clear=False):
            with patch("scripts.texture.config.torch", _make_mock_torch(cuda_available=True)):
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


class PoseLoadingTests(unittest.TestCase):
    def test_load_poses_accepts_stage_two_list_format(self) -> None:
        payload = [
            {
                "frame_name": "00007.jpg",
                "transform_matrix": np.eye(4).tolist(),
            },
            {
                "frame_name": "00011.jpg",
                "transform_matrix": (np.eye(4) * 2.0).tolist(),
            },
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_poses.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            poses, frame_indices = _load_poses(str(path))

        self.assertEqual(poses.shape, (2, 4, 4))
        self.assertEqual(frame_indices, [7, 11])

    def test_load_poses_uses_positional_indices_when_names_missing(self) -> None:
        payload = [
            {"transform_matrix": np.eye(4).tolist()},
            {"transform_matrix": np.eye(4).tolist()},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_poses.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            _, frame_indices = _load_poses(str(path))

        self.assertEqual(frame_indices, [0, 1])


class IntrinsicsPointCloudFallbackTests(unittest.TestCase):
    def test_falls_back_to_mesh_vertex_colors(self) -> None:
        mesh_vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        mesh_colors = np.array([[0.2, 0.4, 0.6]], dtype=np.float64)

        with TemporaryDirectory() as tmp:
            points, colors = _resolve_intrinsics_point_cloud(
                Path(tmp),
                mesh_vertices,
                mesh_colors,
            )

        np.testing.assert_array_equal(points, mesh_vertices)
        np.testing.assert_array_equal(colors, mesh_colors)

    def test_errors_when_no_colored_source_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Missing colored point cloud",
            ):
                _resolve_intrinsics_point_cloud(
                    Path(tmp),
                    np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
                    None,
                )


class TextureUvProxyTests(unittest.TestCase):
    @patch("scripts.texture.config._get_available_memory_mb", return_value=8192.0)
    def test_auto_small_mesh_uses_min_floor(self, _mock_mem) -> None:
        """Mesh ≤ 50K → budget stays at 50K floor."""
        old = os.environ.pop("TEXTURE_UV_MAX_FACES", None)
        try:
            budget, source = _resolve_uv_face_budget(30_000)
            self.assertEqual(source, "auto")
            self.assertEqual(budget, 50_000)
        finally:
            if old is not None:
                os.environ["TEXTURE_UV_MAX_FACES"] = old

    @patch("scripts.texture.config._get_available_memory_mb", return_value=8192.0)
    def test_auto_mesh_within_cap(self, _mock_mem) -> None:
        """Mesh 55K + enough RAM → budget = input faces (no simplification)."""
        old = os.environ.pop("TEXTURE_UV_MAX_FACES", None)
        try:
            budget, source = _resolve_uv_face_budget(55_000)
            self.assertEqual(source, "auto")
            self.assertEqual(budget, 55_000)
        finally:
            if old is not None:
                os.environ["TEXTURE_UV_MAX_FACES"] = old

    @patch("scripts.texture.config._get_available_memory_mb", return_value=8192.0)
    def test_auto_large_mesh_clamped_to_max(self, _mock_mem) -> None:
        """Mesh 300K + enough RAM → budget capped at 60K."""
        old = os.environ.pop("TEXTURE_UV_MAX_FACES", None)
        try:
            budget, source = _resolve_uv_face_budget(300_000)
            self.assertEqual(source, "auto")
            self.assertEqual(budget, 60_000)
        finally:
            if old is not None:
                os.environ["TEXTURE_UV_MAX_FACES"] = old

    @patch("scripts.texture.config._get_available_memory_mb", return_value=3072.0)
    def test_auto_ram_limits_budget(self, _mock_mem) -> None:
        """Mesh 200K but RAM only allows ~107K → budget clamped by RAM."""
        old = os.environ.pop("TEXTURE_UV_MAX_FACES", None)
        try:
            budget, source = _resolve_uv_face_budget(200_000)
            self.assertEqual(source, "auto")
            # (3072 - 2048) * 1024² / 10000 ≈ 107374
            self.assertGreater(budget, 50_000)
            self.assertLess(budget, 200_000)
        finally:
            if old is not None:
                os.environ["TEXTURE_UV_MAX_FACES"] = old

    @patch("scripts.texture.config._get_available_memory_mb", return_value=2048.0)
    def test_auto_very_low_ram_clamps_to_min(self, _mock_mem) -> None:
        """Zero usable RAM → budget = 50K floor."""
        old = os.environ.pop("TEXTURE_UV_MAX_FACES", None)
        try:
            budget, source = _resolve_uv_face_budget(200_000)
            self.assertEqual(source, "auto")
            self.assertEqual(budget, 50_000)
        finally:
            if old is not None:
                os.environ["TEXTURE_UV_MAX_FACES"] = old

    @patch("scripts.texture.config._get_available_memory_mb", return_value=4096.0)
    def test_env_positive_value_used_directly(self, _mock_mem) -> None:
        with patch.dict(os.environ, {"TEXTURE_UV_MAX_FACES": "123456"}, clear=False):
            budget, source = _resolve_uv_face_budget(200_000)
            self.assertEqual(budget, 123456)
            self.assertEqual(source, "env")

    @patch("scripts.texture.config._get_available_memory_mb", return_value=4096.0)
    def test_env_zero_means_unlimited(self, _mock_mem) -> None:
        with patch.dict(os.environ, {"TEXTURE_UV_MAX_FACES": "0"}, clear=False):
            budget, source = _resolve_uv_face_budget(200_000)
            self.assertEqual(budget, 0)
            self.assertEqual(source, "env")

    @patch("scripts.texture.config._get_available_memory_mb", return_value=8192.0)
    def test_env_negative_falls_through_to_auto(self, _mock_mem) -> None:
        with patch.dict(os.environ, {"TEXTURE_UV_MAX_FACES": "-1"}, clear=False):
            budget, source = _resolve_uv_face_budget(55_000)
            self.assertEqual(source, "auto")
            self.assertEqual(budget, 55_000)

    @patch("scripts.texture.config._get_available_memory_mb", return_value=8192.0)
    def test_env_unparseable_falls_through_to_auto(self, _mock_mem) -> None:
        with patch.dict(os.environ, {"TEXTURE_UV_MAX_FACES": "bad"}, clear=False):
            budget, source = _resolve_uv_face_budget(55_000)
            self.assertEqual(source, "auto")
            self.assertEqual(budget, 55_000)

    @patch("scripts.texture.bake._simplify_mesh_for_uv")
    def test_skips_simplification_when_under_budget(self, mock_simplify) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        uv_vertices, uv_faces, simplified = _maybe_simplify_mesh_for_uv(
            vertices,
            faces,
            face_budget=10,
        )

        self.assertFalse(simplified)
        np.testing.assert_array_equal(uv_vertices, vertices)
        np.testing.assert_array_equal(uv_faces, faces)
        mock_simplify.assert_not_called()

    @patch("scripts.texture.bake._simplify_mesh_for_uv")
    def test_uses_proxy_when_over_budget(self, mock_simplify) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32)
        proxy_vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        proxy_faces = np.array([[0, 1, 2]], dtype=np.int32)
        mock_simplify.return_value = (proxy_vertices, proxy_faces)

        uv_vertices, uv_faces, simplified = _maybe_simplify_mesh_for_uv(
            vertices,
            faces,
            face_budget=1,
        )

        self.assertTrue(simplified)
        np.testing.assert_array_equal(uv_vertices, proxy_vertices)
        np.testing.assert_array_equal(uv_faces, proxy_faces)
        mock_simplify.assert_called_once()


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


class ResolveTextureViewAssignModeTests(unittest.TestCase):
    def test_default_mode_is_region_gc(self) -> None:
        old = os.environ.pop("TEXTURE_VIEW_ASSIGN_MODE", None)
        try:
            self.assertEqual(_resolve_texture_view_assign_mode(), "region_gc")
        finally:
            if old is not None:
                os.environ["TEXTURE_VIEW_ASSIGN_MODE"] = old

    def test_explicit_region_gc_mode_is_kept(self) -> None:
        self.assertEqual(_resolve_texture_view_assign_mode("region_gc"), "region_gc")

    def test_invalid_mode_falls_back_to_default(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_VIEW_ASSIGN_MODE": "broken"}, clear=False):
            self.assertEqual(_resolve_texture_view_assign_mode(), "region_gc")


class ResolveTextureQualityBoostTests(unittest.TestCase):
    def test_default_mode_is_disabled(self) -> None:
        old = os.environ.pop("TEXTURE_QUALITY_BOOST", None)
        try:
            self.assertFalse(_resolve_texture_quality_boost())
        finally:
            if old is not None:
                os.environ["TEXTURE_QUALITY_BOOST"] = old

    def test_env_truthy_value_enables_boost(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_QUALITY_BOOST": "true"}, clear=False):
            self.assertTrue(_resolve_texture_quality_boost())

    def test_explicit_argument_wins(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_QUALITY_BOOST": "false"}, clear=False):
            self.assertTrue(_resolve_texture_quality_boost(True))


class DetailCompanionViewSelectionTests(unittest.TestCase):
    def test_picks_first_non_primary_candidate(self) -> None:
        primary = np.array([2, 1, 4], dtype=np.int32)
        best_views = np.array([
            [2, 5, 7],
            [1, 1, 6],
            [4, -1, -1],
        ], dtype=np.int32)
        best_scores = np.array([
            [0.9, 0.6, 0.2],
            [0.8, 0.7, 0.4],
            [0.5, -1.0, -1.0],
        ], dtype=np.float32)

        companions = _select_detail_companion_views(primary, best_views, best_scores)

        np.testing.assert_array_equal(companions, np.array([5, 6, -1], dtype=np.int32))

    def test_collects_multiple_unique_candidates_in_rank_order(self) -> None:
        primary = np.array([2, 1, 4], dtype=np.int32)
        best_views = np.array([
            [2, 5, 7, 8],
            [1, 1, 6, 9],
            [4, 8, 9, -1],
        ], dtype=np.int32)
        best_scores = np.array([
            [0.9, 0.6, 0.3, 0.2],
            [0.8, 0.7, 0.4, 0.1],
            [0.5, 0.4, 0.2, -1.0],
        ], dtype=np.float32)

        companions = _select_detail_companion_view_candidates(
            primary,
            best_views,
            best_scores,
            max_candidates=3,
        )

        self.assertEqual(len(companions), 3)
        np.testing.assert_array_equal(companions[0], np.array([5, 6, 8], dtype=np.int32))
        np.testing.assert_array_equal(companions[1], np.array([7, 9, 9], dtype=np.int32))
        np.testing.assert_array_equal(companions[2], np.array([8, -1, -1], dtype=np.int32))


class DetailPatchNormalizationTests(unittest.TestCase):
    def test_normalizes_local_mean_toward_reference(self) -> None:
        reference = np.full((6, 6, 3), 0.40, dtype=np.float32)
        reference[2:4, 2:4, 1] = 0.75
        candidate = np.clip(reference * 1.20 + 0.10, 0.0, 1.0)
        mask = np.ones((6, 6), dtype=bool)

        normalized = _normalize_candidate_patch_colors(reference, candidate, mask)

        before = float(np.mean(np.abs(reference - candidate)))
        after = float(np.mean(np.abs(reference - normalized)))
        self.assertLess(after, before)


class DetailBandShiftRefinementTests(unittest.TestCase):
    def test_ecc_refinement_reduces_alignment_error(self) -> None:
        ref = np.zeros((32, 32, 3), dtype=np.float32)
        ref[8:24, 8:24, :] = 0.75
        ref[10:22, 10:22, 1] = 0.25
        cand = cv2.warpAffine(
            ref,
            np.array([[1.0, 0.0, 1.2], [0.0, 1.0, -0.7]], dtype=np.float32),
            (32, 32),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        cand = np.clip(cand * 1.10 + 0.04, 0.0, 1.0)
        mask = np.zeros((32, 32), dtype=bool)
        mask[5:27, 5:27] = True

        normalized = _normalize_candidate_patch_colors(ref, cand, mask)
        phase_dx, phase_dy, _phase_response = _estimate_detail_band_shift(ref, normalized, mask)
        refined_dx, refined_dy, refined_response = _refine_detail_band_shift(
            ref,
            normalized,
            mask,
            init_dx=phase_dx,
            init_dy=phase_dy,
        )

        phase_aligned = _translate_patch(normalized, phase_dx, phase_dy)
        refined_aligned = _translate_patch(normalized, refined_dx, refined_dy)
        phase_error = float(np.mean(np.abs(ref[mask] - phase_aligned[mask])))
        refined_error = float(np.mean(np.abs(ref[mask] - refined_aligned[mask])))

        self.assertLessEqual(refined_error, phase_error + 1e-4)
        self.assertGreater(refined_response, 0.0)


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


class RegionGraphLockingTests(unittest.TestCase):
    def test_region_labeling_smooths_ambiguous_curved_strip(self) -> None:
        fids = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
        ], dtype=np.int32)
        best_scores = np.array([
            [0.52, 0.45],
            [0.52, 0.45],
            [0.52, 0.45],
            [0.52, 0.45],
            [0.52, 0.50],
            [0.52, 0.50],
            [0.52, 0.50],
            [0.52, 0.50],
            [0.52, 0.45],
            [0.52, 0.45],
            [0.52, 0.45],
            [0.52, 0.45],
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
        base_locked = np.array([1, 0, 1], dtype=np.int32)

        face_locked_view, face_support, region_ids = _compute_region_gc_locked_views(
            fids=fids,
            n_faces=3,
            n_views=2,
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=faces,
            base_locked_view=base_locked,
        )

        np.testing.assert_array_equal(face_locked_view, np.array([1, 1, 1], dtype=np.int32))
        self.assertTrue(np.all(face_support > 0.0))
        np.testing.assert_array_equal(region_ids, np.array([0, 0, 0], dtype=np.int32))

    def test_region_labeling_respects_sharp_edge_split(self) -> None:
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
            [0.52, 0.50],
            [0.52, 0.50],
            [0.52, 0.50],
            [0.52, 0.50],
            [0.53, 0.47],
            [0.53, 0.47],
            [0.53, 0.47],
            [0.53, 0.47],
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
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ], dtype=np.int32)
        conflict_texels = np.ones(12, dtype=bool)
        face_normals = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
        ], dtype=np.int32)
        base_locked = np.array([1, 0, 0], dtype=np.int32)

        face_locked_view, _face_support, region_ids = _compute_region_gc_locked_views(
            fids=fids,
            n_faces=3,
            n_views=2,
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=faces,
            base_locked_view=base_locked,
        )

        np.testing.assert_array_equal(face_locked_view, np.array([1, 0, 0], dtype=np.int32))
        np.testing.assert_array_equal(region_ids, np.array([-1, -1, -1], dtype=np.int32))

    def test_region_labeling_expands_to_adjacent_smooth_faces(self) -> None:
        fids = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
        ], dtype=np.int32)
        best_scores = np.array([
            [0.53, 0.48],
            [0.53, 0.48],
            [0.53, 0.48],
            [0.53, 0.48],
            [0.52, 0.49],
            [0.52, 0.49],
            [0.52, 0.49],
            [0.52, 0.49],
            [0.51, 0.46],
            [0.51, 0.46],
            [0.51, 0.46],
            [0.51, 0.46],
            [0.50, 0.40],
            [0.50, 0.40],
            [0.50, 0.40],
            [0.50, 0.40],
        ], dtype=np.float32)
        best_views = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ], dtype=np.int32)
        conflict_texels = np.array([
            True, True, True, True,
            True, True, True, True,
            False, False, False, False,
            False, False, False, False,
        ], dtype=bool)
        face_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
            [4, 3, 5],
        ], dtype=np.int32)
        base_locked = np.array([1, 1, -1, -1], dtype=np.int32)

        face_locked_view, _face_support, region_ids = _compute_region_gc_locked_views(
            fids=fids,
            n_faces=4,
            n_views=2,
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=faces,
            base_locked_view=base_locked,
        )

        np.testing.assert_array_equal(face_locked_view, np.array([1, 1, 1, 1], dtype=np.int32))
        np.testing.assert_array_equal(region_ids, np.array([0, 0, 0, 0], dtype=np.int32))


class NarrowSeamLevelingTests(unittest.TestCase):
    def test_only_boundary_band_is_softened(self) -> None:
        texture = np.zeros((9, 9, 3), dtype=np.float32)
        texture[:, :4, 0] = 1.0
        texture[:, 4:, 2] = 1.0
        valid_mask = np.ones((9, 9), dtype=bool)
        label_buffer = np.full((9, 9), -1, dtype=np.int32)
        label_buffer[:, :4] = 0
        label_buffer[:, 4:] = 1

        leveled, seam_texels = _apply_narrow_seam_leveling(texture, valid_mask, label_buffer)

        self.assertGreater(seam_texels, 0)
        np.testing.assert_array_almost_equal(leveled[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(leveled[0, 8], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        self.assertGreater(float(leveled[4, 4, 0]), 0.0)
        self.assertGreater(float(leveled[4, 3, 2]), 0.0)

    def test_invalid_uv_space_does_not_darken_boundary(self) -> None:
        texture = np.zeros((9, 9, 3), dtype=np.float32)
        valid_mask = np.zeros((9, 9), dtype=bool)
        valid_mask[2:7, 2:7] = True
        texture[2:7, 2:4, 0] = 1.0
        texture[2:7, 4:7, 2] = 1.0
        label_buffer = np.full((9, 9), -1, dtype=np.int32)
        label_buffer[2:7, 2:4] = 0
        label_buffer[2:7, 4:7] = 1

        leveled, seam_texels = _apply_narrow_seam_leveling(texture, valid_mask, label_buffer)

        self.assertGreater(seam_texels, 0)
        self.assertGreater(float(leveled[4, 4, 0] + leveled[4, 4, 2]), 0.95)
        self.assertGreater(float(leveled[4, 2, 0]), 0.95)
        self.assertGreater(float(leveled[4, 6, 2]), 0.95)

    def test_detail_band_shift_detects_small_translation(self) -> None:
        ref = np.zeros((12, 12, 3), dtype=np.float32)
        ref[3:9, 4:8, :] = 1.0
        cand = np.roll(ref, shift=1, axis=1)
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 2:10] = True

        dx, dy, response = _estimate_detail_band_shift(ref, cand, mask)

        self.assertGreater(abs(dx), 0.2)
        self.assertAlmostEqual(dy, 0.0, places=1)
        self.assertGreaterEqual(response, 0.0)

    def test_quality_boost_preserves_stronger_companion_detail(self) -> None:
        texture = np.full((11, 11, 3), 0.5, dtype=np.float32)
        texture[:, :5, 0] += 0.15
        texture[:, 6:, 2] += 0.15
        valid_mask = np.ones((11, 11), dtype=bool)
        label_buffer = np.full((11, 11), -1, dtype=np.int32)
        label_buffer[:, :5] = 0
        label_buffer[:, 6:] = 1
        label_buffer[:, 5] = 1

        companion = texture.copy()
        companion[3:8, 4:7, 1] = np.array(
            [
                [0.0, 0.8, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.9, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        companion_valid = np.zeros((11, 11), dtype=bool)
        companion_valid[2:9, 3:8] = True

        leveled, seam_texels = _apply_narrow_seam_leveling(
            texture,
            valid_mask,
            label_buffer,
            detail_texture=companion,
            detail_valid_mask=companion_valid,
            quality_boost=True,
        )

        self.assertGreater(seam_texels, 0)
        np.testing.assert_array_almost_equal(leveled[0, 0], texture[0, 0], decimal=5)
        self.assertGreater(float(leveled[5, 5, 1]), float(texture[5, 5, 1]) + 0.15)
