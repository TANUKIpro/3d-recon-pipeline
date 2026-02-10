import os
import unittest
from unittest.mock import patch

from scripts.stage_texture_bake import _resolve_texture_device


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
