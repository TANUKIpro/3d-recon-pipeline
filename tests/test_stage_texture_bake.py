import json
import os
import hashlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

import scripts.texture.gpu_raster as gpu_raster
from scripts.stage_texture_bake import (
    _apply_narrow_seam_leveling,
    _apply_view_hardening,
    _compute_conflict_texels,
    _evaluate_view_packet,
    _evaluate_view_samples,
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
    _score_view_packet,
    _select_detail_companion_view_candidates,
    _select_detail_companion_views,
    _translate_patch,
    _update_topk_scores,
    ViewEvalPacket,
    ViewPacketShapeError,
    bake_texture,
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


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_fake_xatlas(result):
    class FakeAtlas:
        def add_mesh(self, *args, **kwargs):
            return None

        def generate(self, chart_options=None, pack_options=None):
            del chart_options, pack_options
            return None

        def __getitem__(self, index):
            del index
            return result

    return types.SimpleNamespace(
        Atlas=FakeAtlas,
        ChartOptions=type("ChartOptions", (), {}),
        PackOptions=type("PackOptions", (), {}),
    )


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


class ViewEvalPacketTests(unittest.TestCase):
    def _make_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos3d = np.array([
            [0.0, 0.0, 1.0],
            [0.2, 0.1, 1.4],
            [-0.3, 0.2, 1.8],
            [1.9, 0.0, 0.8],
        ], dtype=np.float64)
        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        K = np.array([
            [4.0, 0.0, 4.0],
            [0.0, 4.0, 4.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        mask = np.ones((8, 8), dtype=bool)
        mask[4, 4] = False
        depth_buffer = np.full((8, 8), np.inf, dtype=np.float32)
        depth_buffer[4, 4] = 0.95
        depth_buffer[4, 5] = 1.5
        depth_buffer[4, 3] = 1.9
        return pos3d, normals, c2w, K, mask, depth_buffer

    def test_packet_matches_public_view_scoring(self) -> None:
        pos3d, normals, c2w, K, mask, depth_buffer = self._make_inputs()

        packet, valid_packet, score_packet = _evaluate_view_packet(
            pos3d=pos3d,
            normals=normals,
            c2w=c2w,
            K=K,
            img_w=8,
            img_h=8,
            mask_bool=mask,
            depth_buffer=depth_buffer,
            min_cos=0.2,
            angle_exp=4.0,
            dist_pow=1.0,
            device="cpu",
        )
        valid_public, score_public, px_public, py_public = _evaluate_view_samples(
            pos3d=pos3d,
            normals=normals,
            c2w=c2w,
            K=K,
            img_w=8,
            img_h=8,
            mask_bool=mask,
            depth_buffer=depth_buffer,
            min_cos=0.2,
            angle_exp=4.0,
            dist_pow=1.0,
            device="cpu",
        )
        valid_rescored, score_rescored = _score_view_packet(
            packet,
            min_cos=0.2,
            angle_exp=4.0,
            dist_pow=1.0,
        )

        np.testing.assert_array_equal(valid_packet, valid_public)
        np.testing.assert_array_equal(valid_rescored, valid_public)
        np.testing.assert_array_equal(score_packet, score_public)
        np.testing.assert_array_equal(score_rescored, score_public)
        np.testing.assert_array_equal(packet.px_proj, px_public)
        np.testing.assert_array_equal(packet.py_proj, py_public)

    def test_packet_hash_regression(self) -> None:
        pos3d, normals, c2w, K, mask, depth_buffer = self._make_inputs()

        packet, valid, score = _evaluate_view_packet(
            pos3d=pos3d,
            normals=normals,
            c2w=c2w,
            K=K,
            img_w=8,
            img_h=8,
            mask_bool=mask,
            depth_buffer=depth_buffer,
            min_cos=0.2,
            angle_exp=4.0,
            dist_pow=1.0,
            device="cpu",
        )

        self.assertEqual(
            _hash_arrays(
                packet.px_proj,
                packet.py_proj,
                packet.sample_valid,
                packet.cos_angle,
                packet.distances,
                valid,
                score,
            ),
            "77827b06d2dbd09345faf180e598156cfc8bf2cbc7f6d8528fe0e772ce43f572",
        )

    def test_gpu_packet_length_mismatch_raises_targeted_error(self) -> None:
        pos3d, normals, c2w, K, mask, depth_buffer = self._make_inputs()
        pos3d = np.vstack([
            pos3d,
            np.array([[0.1, -0.2, 1.2]], dtype=np.float64),
        ])
        normals = np.vstack([
            normals,
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        ])
        bad_packet = ViewEvalPacket(
            px_proj=np.zeros(3, dtype=np.float64),
            py_proj=np.zeros(3, dtype=np.float64),
            sample_valid=np.array([True, False, True]),
            cos_angle=np.ones(3, dtype=np.float64),
            distances=np.ones(3, dtype=np.float64),
        )

        with patch(
            "scripts.texture.gpu_raster.gpu_evaluate_view_packet",
            return_value=bad_packet,
        ):
            with self.assertRaisesRegex(
                ViewPacketShapeError,
                "expected 5 samples",
            ):
                _evaluate_view_packet(
                    pos3d=pos3d,
                    normals=normals,
                    c2w=c2w,
                    K=K,
                    img_w=8,
                    img_h=8,
                    mask_bool=mask,
                    depth_buffer=depth_buffer,
                    min_cos=0.2,
                    angle_exp=4.0,
                    dist_pow=1.0,
                    device="cuda",
                )


class GpuTexelCacheTests(unittest.TestCase):
    def test_subset_upload_does_not_replace_cached_canonical_texels(self) -> None:
        full_pos = np.arange(18, dtype=np.float64).reshape(6, 3)
        full_norm = np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            (6, 1),
        )
        subset_pos = full_pos[:3].copy()
        subset_norm = full_norm[:3].copy()
        upload_count = {"count": 0}

        def fake_as_tensor(array, device=None):
            upload_count["count"] += 1
            return {
                "upload": upload_count["count"],
                "shape": tuple(array.shape),
                "device": device,
            }

        fake_torch = types.SimpleNamespace(as_tensor=fake_as_tensor)

        with patch.object(gpu_raster, "_torch", fake_torch):
            gpu_raster.clear_cache()
            full_pos_gpu_1, full_norm_gpu_1 = gpu_raster._cache_texel_tensors(
                full_pos, full_norm
            )
            subset_pos_gpu, subset_norm_gpu = gpu_raster._cache_texel_tensors(
                subset_pos, subset_norm
            )
            full_pos_gpu_2, full_norm_gpu_2 = gpu_raster._cache_texel_tensors(
                full_pos, full_norm
            )

            self.assertIs(gpu_raster._cached_pos3d_owner, full_pos)
            self.assertIs(gpu_raster._cached_normals_owner, full_norm)
            self.assertEqual(gpu_raster._cached_texel_count, 6)
            gpu_raster.clear_cache()

        self.assertIs(full_pos_gpu_1, full_pos_gpu_2)
        self.assertIs(full_norm_gpu_1, full_norm_gpu_2)
        self.assertIsNot(full_pos_gpu_1, subset_pos_gpu)
        self.assertIsNot(full_norm_gpu_1, subset_norm_gpu)
        self.assertEqual(upload_count["count"], 4)


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
    def test_auto_large_mesh_uncapped_with_enough_ram(self, _mock_mem) -> None:
        """Mesh 300K + enough RAM → budget = input faces (no time-based cap)."""
        old = os.environ.pop("TEXTURE_UV_MAX_FACES", None)
        try:
            budget, source = _resolve_uv_face_budget(300_000)
            self.assertEqual(source, "auto")
            self.assertEqual(budget, 300_000)
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


class BakeTextureRegressionTests(unittest.TestCase):
    @staticmethod
    def _write_mesh(path: Path) -> None:
        vertex_dtype = np.dtype([
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ])
        face_dtype = np.dtype([("vertex_indices", "i4", (3,))])

        vertices = np.array([
            (-0.5, -0.5, 1.0, 255, 80, 80),
            (0.5, -0.5, 1.0, 80, 255, 80),
            (0.0, 0.5, 1.0, 80, 80, 255),
        ], dtype=vertex_dtype)
        faces = np.array([([0, 2, 1],)], dtype=face_dtype)
        PlyData([
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(faces, "face"),
        ], text=True).write(str(path))

    def test_mocked_single_triangle_output_hashes_are_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh_path = tmp_path / "mesh.ply"
            self._write_mesh(mesh_path)

            frames_dir = tmp_path / "frames"
            masks_dir = tmp_path / "masks"
            output_dir = tmp_path / "out"
            frames_dir.mkdir()
            masks_dir.mkdir()
            output_dir.mkdir()

            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[..., 0] = 32
            frame[..., 1] = np.array([
                [16, 24, 32, 40, 48, 56, 64, 72],
                [16, 24, 32, 40, 48, 56, 64, 72],
                [16, 24, 32, 40, 48, 56, 64, 72],
                [16, 24, 32, 40, 48, 56, 64, 72],
                [16, 24, 32, 40, 48, 56, 64, 72],
                [16, 24, 32, 40, 48, 56, 64, 72],
                [16, 24, 32, 40, 48, 56, 64, 72],
                [16, 24, 32, 40, 48, 56, 64, 72],
            ], dtype=np.uint8)
            frame[..., 2] = np.array([
                [80, 80, 80, 80, 80, 80, 80, 80],
                [88, 88, 88, 88, 88, 88, 88, 88],
                [96, 96, 96, 96, 96, 96, 96, 96],
                [104, 104, 104, 104, 104, 104, 104, 104],
                [112, 112, 112, 112, 112, 112, 112, 112],
                [120, 120, 120, 120, 120, 120, 120, 120],
                [128, 128, 128, 128, 128, 128, 128, 128],
                [136, 136, 136, 136, 136, 136, 136, 136],
            ], dtype=np.uint8)
            cv2.imwrite(str(frames_dir / "00000.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(masks_dir / "00000.png"), np.full((8, 8), 255, dtype=np.uint8))

            poses_path = tmp_path / "camera_poses.json"
            poses_path.write_text(json.dumps([
                {
                    "frame_name": "00000.jpg",
                    "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
                }
            ]), encoding="utf-8")

            parallel_result = (
                np.array([0, 1, 2], dtype=np.uint32),
                np.array([[0, 2, 1]], dtype=np.uint32),
                np.array([[0.1, 0.1], [0.5, 0.9], [0.9, 0.1]], dtype=np.float32),
            )
            fake_xatlas = _make_fake_xatlas(parallel_result)
            intrinsics = {"K": [[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]}

            with patch.dict(sys.modules, {"xatlas": fake_xatlas}):
                with patch.dict(
                    os.environ,
                    {
                        "TEXTURE_DEVICE": "cpu",
                        "TEXTURE_OVERSAMPLE": "1",
                        "TEXTURE_BLEND_TOPK": "1",
                        "TEXTURE_SHARPEN": "0",
                        "TEXTURE_QUALITY_BOOST": "false",
                    },
                    clear=False,
                ):
                    with patch(
                        "scripts.texture.parallel_uv.parallel_xatlas_generate",
                        return_value=parallel_result,
                    ):
                        with patch(
                            "scripts.texture.bake._estimate_intrinsics",
                            return_value=intrinsics,
                        ):
                            with patch(
                                "scripts.texture.bake.orient_mesh_outward",
                                side_effect=lambda vertices, faces: (faces, False, 1.0, 1.0),
                            ):
                                result = bake_texture(
                                    str(mesh_path),
                                    str(poses_path),
                                    str(frames_dir),
                                    str(masks_dir),
                                    str(output_dir),
                                    tex_size=8,
                                    quality_boost=False,
                                )

            self.assertEqual(result, output_dir / "textured_mesh.obj")
            self.assertEqual(_hash_file(output_dir / "textured_mesh.obj"), "ed5e5443d4cab796bc64dfa73d88299135b36fd2ceed6038b14fa8c6bb3b9f40")
            self.assertEqual(_hash_file(output_dir / "textured_mesh.mtl"), "b2afa1559c9a36be27e591d98f7384c7f1c99e7bf33349e9e78c786c922efced")
            self.assertEqual(_hash_file(output_dir / "texture.png"), "7ace89584d155af0638ea9e91efa2a856e74d444e02f36b43adca47afdb90719")

    def test_fallback_subset_reuse_does_not_boolean_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh_path = tmp_path / "mesh.ply"
            self._write_mesh(mesh_path)

            frames_dir = tmp_path / "frames"
            masks_dir = tmp_path / "masks"
            output_dir = tmp_path / "out"
            frames_dir.mkdir()
            masks_dir.mkdir()
            output_dir.mkdir()

            frame0 = np.full((8, 8, 3), 96, dtype=np.uint8)
            frame1 = np.full((8, 8, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(frames_dir / "00000.jpg"), cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(frames_dir / "00001.jpg"), cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(masks_dir / "00000.png"), np.full((8, 8), 255, dtype=np.uint8))
            cv2.imwrite(str(masks_dir / "00001.png"), np.full((8, 8), 255, dtype=np.uint8))

            poses_path = tmp_path / "camera_poses.json"
            poses_path.write_text(json.dumps([
                {
                    "frame_name": "00000.jpg",
                    "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
                },
                {
                    "frame_name": "00001.jpg",
                    "transform_matrix": np.array(
                        [[1.0, 0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    ).tolist(),
                }
            ]), encoding="utf-8")

            parallel_result = (
                np.array([0, 1, 2], dtype=np.uint32),
                np.array([[0, 2, 1]], dtype=np.uint32),
                np.array([[0.1, 0.1], [0.5, 0.9], [0.9, 0.1]], dtype=np.float32),
            )
            fake_xatlas = _make_fake_xatlas(parallel_result)
            intrinsics = {"K": [[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]}

            def fake_eval_samples(
                pos3d, normals, c2w, K, img_w, img_h, mask_bool, depth_buffer,
                min_cos, angle_exp, dist_pow, device="cpu",
            ):
                del normals, K, img_w, img_h, mask_bool, depth_buffer, device
                n = len(pos3d)
                px = np.linspace(1.0, 6.0, n, dtype=np.float64)
                py = np.linspace(6.0, 1.0, n, dtype=np.float64)
                valid = np.zeros(n, dtype=bool)
                if min_cos <= 0.0:
                    if float(c2w[0, 3]) < 0.5:
                        valid[:3] = True
                    else:
                        valid[3:] = True
                score = np.zeros(n, dtype=np.float64)
                if np.any(valid):
                    score[valid] = 1.0
                return valid, score, px, py

            with patch.dict(sys.modules, {"xatlas": fake_xatlas}):
                with patch.dict(
                    os.environ,
                    {
                        "TEXTURE_DEVICE": "cpu",
                        "TEXTURE_OVERSAMPLE": "1",
                        "TEXTURE_BLEND_TOPK": "1",
                        "TEXTURE_SHARPEN": "0",
                        "TEXTURE_QUALITY_BOOST": "false",
                    },
                    clear=False,
                ):
                    with patch(
                        "scripts.texture.parallel_uv.parallel_xatlas_generate",
                        return_value=parallel_result,
                    ):
                        with patch(
                            "scripts.texture.bake._estimate_intrinsics",
                            return_value=intrinsics,
                        ):
                            with patch(
                                "scripts.texture.bake.orient_mesh_outward",
                                side_effect=lambda vertices, faces: (faces, False, 1.0, 1.0),
                            ):
                                with patch(
                                    "scripts.texture.bake._evaluate_view_samples",
                                    side_effect=fake_eval_samples,
                                ):
                                    with patch(
                                        "scripts.texture.bake._identify_cap_texels",
                                        return_value=(None, None),
                                    ):
                                        result = bake_texture(
                                            str(mesh_path),
                                            str(poses_path),
                                            str(frames_dir),
                                            str(masks_dir),
                                            str(output_dir),
                                            tex_size=8,
                                            quality_boost=False,
                                        )

            self.assertEqual(result, output_dir / "textured_mesh.obj")
            self.assertTrue((output_dir / "texture.png").is_file())
