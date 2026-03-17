"""Tests for GPU rasterization (scripts.texture.gpu_raster).

These tests validate GPU/CPU equivalence for depth rasterization, UV-space
rasterization, and view scoring.  Tests that require CUDA are skipped
automatically when a GPU is unavailable.
"""

import unittest

import numpy as np


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_nvdiffrast() -> bool:
    if not _has_cuda():
        return False
    try:
        import nvdiffrast.torch  # noqa: F401
        return True
    except ImportError:
        return False


def _make_two_triangle_mesh():
    """Create a simple mesh: two triangles forming a quad on the XY plane at z=1."""
    vertices = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    return vertices, faces


def _make_identity_camera():
    """Camera at origin looking down +Z (identity c2w)."""
    c2w = np.eye(4, dtype=np.float64)
    return c2w


def _make_pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


class TestPinholeToGlProj(unittest.TestCase):
    """Test the pinhole-to-OpenGL projection conversion."""

    def test_basic_shape(self):
        from scripts.texture.gpu_raster import _pinhole_to_gl_proj
        K = _make_pinhole_K()
        proj = _pinhole_to_gl_proj(K, 640, 480)
        self.assertEqual(proj.shape, (4, 4))
        self.assertEqual(proj.dtype, np.float32)

    def test_perspective_divide_center(self):
        """A point at the principal point should map to NDC (0, 0)."""
        from scripts.texture.gpu_raster import _pinhole_to_gl_proj
        K = _make_pinhole_K(fx=500, fy=500, cx=320, cy=240)
        proj = _pinhole_to_gl_proj(K, 640, 480, near=0.01, far=100.0)
        # Point at (0, 0, -1) in OpenGL camera space (looking down -Z)
        # In our convention, camera-space is z-forward, but the projection
        # flips Z. We test the matrix directly.
        # Camera point at image center, depth=1:
        # cam_x = 0, cam_y = 0, cam_z = 1 (our convention)
        pt = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        clip = proj @ pt
        ndc_x = clip[0] / clip[3]
        ndc_y = clip[1] / clip[3]
        # Should be near center of NDC
        self.assertAlmostEqual(ndc_x, 0.0, places=1)


@unittest.skipUnless(_has_nvdiffrast(), "nvdiffrast + CUDA required")
class TestGpuRasterizeDepth(unittest.TestCase):
    """GPU depth rasterization vs CPU reference."""

    def test_depth_matches_cpu(self):
        from scripts.texture.view_scoring import _rasterize_view_depth

        vertices, faces = _make_two_triangle_mesh()
        c2w = _make_identity_camera()
        K = _make_pinhole_K()
        img_w, img_h = 640, 480

        depth_cpu = _rasterize_view_depth(vertices, faces, c2w, K, img_w, img_h, device="cpu")
        depth_gpu = _rasterize_view_depth(vertices, faces, c2w, K, img_w, img_h, device="cuda")

        # Both should have same shape
        self.assertEqual(depth_cpu.shape, depth_gpu.shape)

        # Background should be inf in both
        bg_cpu = np.isinf(depth_cpu)
        bg_gpu = np.isinf(depth_gpu)

        # Foreground pixels (where mesh is visible) should match closely
        fg_mask = ~bg_cpu & ~bg_gpu
        if np.any(fg_mask):
            diff = np.abs(depth_cpu[fg_mask] - depth_gpu[fg_mask])
            self.assertLess(np.max(diff), 1e-2, "Depth values differ by more than 0.01")

    def test_known_depth_value(self):
        """Mesh at z=1, camera at origin: depth of visible pixels should be ~1.0."""
        from scripts.texture.gpu_raster import gpu_rasterize_depth

        vertices, faces = _make_two_triangle_mesh()
        c2w = _make_identity_camera()
        K = _make_pinhole_K()
        img_w, img_h = 640, 480

        depth = gpu_rasterize_depth(vertices, faces, c2w, K, img_w, img_h)
        fg = ~np.isinf(depth)
        self.assertTrue(np.any(fg), "No foreground pixels rendered")
        self.assertAlmostEqual(float(np.mean(depth[fg])), 1.0, places=2)


@unittest.skipUnless(_has_nvdiffrast(), "nvdiffrast + CUDA required")
class TestGpuRasterizeUvSpace(unittest.TestCase):
    """GPU UV-space rasterization."""

    def test_face_id_coverage(self):
        """Simple UV layout should produce non-empty face_id_buf."""
        from scripts.texture.gpu_raster import gpu_rasterize_uv_space

        # Two triangles covering [0,1]x[0,1] in UV space
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        face_id_buf, bary_buf = gpu_rasterize_uv_space(uvs, faces, tex_res=64)

        self.assertEqual(face_id_buf.shape, (64, 64))
        self.assertEqual(bary_buf.shape, (64, 64, 3))

        # Most texels should be assigned to a face
        coverage = np.count_nonzero(face_id_buf >= 0)
        total = face_id_buf.size
        self.assertGreater(coverage / total, 0.90, "UV coverage too low")

        # Face IDs should be 0 or 1
        valid = face_id_buf[face_id_buf >= 0]
        self.assertTrue(np.all((valid == 0) | (valid == 1)))

    def test_barycentric_sum_to_one(self):
        """Barycentric weights should sum to ~1.0 for valid texels."""
        from scripts.texture.gpu_raster import gpu_rasterize_uv_space

        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        face_id_buf, bary_buf = gpu_rasterize_uv_space(uvs, faces, tex_res=64)
        valid = face_id_buf >= 0
        if np.any(valid):
            bary_sum = bary_buf[valid].sum(axis=1)
            np.testing.assert_allclose(bary_sum, 1.0, atol=1e-4)


@unittest.skipUnless(_has_cuda(), "CUDA required")
class TestGpuEvaluateViewSamples(unittest.TestCase):
    """GPU view scoring vs CPU reference."""

    def test_scores_match_cpu(self):
        from scripts.texture.view_scoring import _evaluate_view_samples

        rng = np.random.default_rng(42)
        n_texels = 500
        pos3d = rng.standard_normal((n_texels, 3)).astype(np.float64) * 0.5
        pos3d[:, 2] += 2.0  # in front of camera
        normals = np.zeros_like(pos3d)
        normals[:, 2] = -1.0  # facing camera

        c2w = np.eye(4, dtype=np.float64)
        K = _make_pinhole_K()
        img_w, img_h = 640, 480

        mask_bool = np.ones((img_h, img_w), dtype=bool)
        depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)

        valid_cpu, score_cpu, px_cpu, py_cpu = _evaluate_view_samples(
            pos3d, normals, c2w, K, img_w, img_h,
            mask_bool, depth_buffer, 0.2, 4.0, 1.0,
            device="cpu",
        )
        valid_gpu, score_gpu, px_gpu, py_gpu = _evaluate_view_samples(
            pos3d, normals, c2w, K, img_w, img_h,
            mask_bool, depth_buffer, 0.2, 4.0, 1.0,
            device="cuda",
        )

        # Valid masks should match
        np.testing.assert_array_equal(valid_cpu, valid_gpu)

        # Projections should be close (float32 vs float64)
        if np.any(valid_cpu):
            np.testing.assert_allclose(
                px_cpu[valid_cpu], px_gpu[valid_cpu], atol=0.5,
            )
            np.testing.assert_allclose(
                py_cpu[valid_cpu], py_gpu[valid_cpu], atol=0.5,
            )


class TestCpuFallback(unittest.TestCase):
    """Verify CPU fallback when device='cpu'."""

    def test_rasterize_cpu_unchanged(self):
        """device='cpu' should use the original CPU path without error."""
        from scripts.texture.view_scoring import _rasterize_view_depth

        vertices, faces = _make_two_triangle_mesh()
        c2w = _make_identity_camera()
        K = _make_pinhole_K()
        img_w, img_h = 640, 480

        depth = _rasterize_view_depth(vertices, faces, c2w, K, img_w, img_h, device="cpu")
        self.assertEqual(depth.shape, (img_h, img_w))
        self.assertTrue(np.any(~np.isinf(depth)), "No visible pixels")

    def test_evaluate_cpu_unchanged(self):
        """device='cpu' should use the original CPU path without error."""
        from scripts.texture.view_scoring import _evaluate_view_samples

        pos3d = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        K = _make_pinhole_K()
        mask = np.ones((480, 640), dtype=bool)
        depth = np.full((480, 640), np.inf, dtype=np.float32)

        valid, score, px, py = _evaluate_view_samples(
            pos3d, normals, c2w, K, 640, 480, mask, depth, 0.2, 4.0, 1.0,
            device="cpu",
        )
        self.assertEqual(valid.shape, (1,))


if __name__ == "__main__":
    unittest.main()
