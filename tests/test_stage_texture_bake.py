import os
import unittest
from unittest.mock import patch

import numpy as np

from scripts.stage_texture_bake import _normalize_weighted_colors, _resolve_texture_device


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


class NormalizeWeightedColorsTests(unittest.TestCase):
    def test_weighted_blend_normalizes_once(self) -> None:
        # Simulate post-accumulation weighted sums from one texel pass.
        color_sum = np.array([[0.4, 0.2, 0.6], [0.9, 0.3, 0.6]], dtype=np.float64)
        weight_sum = np.array([0.5, 2.0], dtype=np.float64)

        normalized = _normalize_weighted_colors(color_sum.copy(), weight_sum, use_max_blend=False)
        expected = np.array([[0.8, 0.4, 1.2], [0.45, 0.15, 0.3]], dtype=np.float64)
        np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=1e-12)

    def test_max_blend_leaves_values_unchanged(self) -> None:
        color_sum = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
        weight_sum = np.array([3.0], dtype=np.float64)

        normalized = _normalize_weighted_colors(color_sum.copy(), weight_sum, use_max_blend=True)
        np.testing.assert_allclose(normalized, color_sum, rtol=0.0, atol=0.0)
