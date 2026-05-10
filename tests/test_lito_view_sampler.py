"""Tests for scripts.lito.view_sampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.lito.view_sampler import (
    fibonacci_sphere,
    load_views_json,
    sample_canonical_views,
    save_views_json,
)


class TestFibonacciSphere:
    def test_zero_returns_empty(self):
        assert fibonacci_sphere(0, 1.0) == []

    def test_n_points_on_sphere(self):
        pts = fibonacci_sphere(64, 1.5)
        assert len(pts) == 64
        for x, y, z in pts:
            r = (x * x + y * y + z * z) ** 0.5
            assert abs(r - 1.5) < 1e-6


class TestSampleCanonicalViews:
    def test_input_view_first(self):
        views = sample_canonical_views(
            n=10, input_view_position=(0.0, 0.0, 2.0)
        )
        assert len(views) == 11
        assert views[0].is_input_view
        assert views[0].label == "input"
        # All others on the Fibonacci sphere
        for v in views[1:]:
            assert not v.is_input_view

    def test_camera_to_world_looks_at_origin(self):
        views = sample_canonical_views(n=8, radius=2.0)
        for v in views:
            eye = v.H_c2w[:3, 3]
            forward = v.H_c2w[:3, 2]  # camera +z column
            target_dir = -eye / max(np.linalg.norm(eye), 1e-12)
            # In the look_at convention forward = (target - eye)/||·||,
            # where target=origin → forward = -eye/||eye||.
            cos = float(forward @ target_dir)
            assert cos > 0.999, f"view {v.index} forward not pointing at origin"

    def test_intrinsic_consistent_across_views(self):
        views = sample_canonical_views(n=4, fov_y_deg=40.0, image_size=(256, 256))
        K0 = views[0].K
        for v in views[1:]:
            assert np.allclose(v.K, K0)
        # Sanity: cx, cy at image centre
        assert abs(K0[0, 2] - 128.0) < 1e-9
        assert abs(K0[1, 2] - 128.0) < 1e-9


class TestSerialisation:
    def test_round_trip(self, tmp_path: Path):
        views = sample_canonical_views(n=6, radius=2.0)
        path = tmp_path / "views.json"
        save_views_json(views, path)
        restored = load_views_json(path)
        assert len(restored) == len(views)
        for a, b in zip(views, restored):
            assert a.index == b.index
            assert a.is_input_view == b.is_input_view
            assert a.label == b.label
            assert np.allclose(a.H_c2w, b.H_c2w)
            assert np.allclose(a.K, b.K)


class TestAngularConfidence:
    """`gaussian_to_mesh._angular_confidence` is a private helper but we
    pin its monotonicity here so future tweaks don't silently invert the
    confidence direction."""

    def test_input_view_has_max_confidence(self):
        from scripts.lito.gaussian_to_mesh import _angular_confidence

        ip = np.array([0.0, 0.0, 2.0])
        same = _angular_confidence(ip, ip)
        opposite = _angular_confidence(-ip, ip)
        side = _angular_confidence(np.array([2.0, 0.0, 0.0]), ip)
        assert same > side > opposite
        assert opposite < 0.05
        assert same > 0.95
