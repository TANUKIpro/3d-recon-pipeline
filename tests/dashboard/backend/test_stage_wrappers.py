"""Tests for scripts/dashboard/stage_wrappers.py.

Covers lazy import verification, argument forwarding, VRAM cleanup,
process registration passthrough, and unused callback deletion.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock heavy modules that wrappers import inside function bodies.
# ---------------------------------------------------------------------------

_mock_extract = types.ModuleType("stage_extract_frames")
_mock_extract.extract_frames = MagicMock(name="extract_frames")

_mock_pi3x = types.ModuleType("stage_pi3x_reconstruct")
_mock_pi3x.run_pi3x_inference = MagicMock(name="run_pi3x_inference")
_mock_pi3x.apply_sam2_masks = MagicMock(name="apply_sam2_masks")
_mock_pi3x.extract_ground_plane = MagicMock(name="extract_ground_plane")

_mock_vram = types.ModuleType("vram_utils")
_mock_vram.cleanup_pytorch_vram = MagicMock(name="cleanup_pytorch_vram")
_mock_vram.ensure_vram_available = MagicMock(name="ensure_vram_available")

_mock_denoise = types.ModuleType("stage_denoise")
_mock_denoise.denoise = MagicMock(name="denoise")

_mock_diffcd = types.ModuleType("stage_diffcd_mesh")
_mock_diffcd.run_diffcd = MagicMock(name="run_diffcd")

_mock_classical = types.ModuleType("stage_classical_mesh")
_mock_classical.run_classical_mesh = MagicMock(name="run_classical_mesh")

_mock_mesh_wrap = types.ModuleType("stage_mesh_wrap")
_mock_mesh_wrap.run_mesh_wrap = MagicMock(name="run_mesh_wrap")

_mock_contact_hole = types.ModuleType("stage_contact_hole_repair")
_mock_contact_hole.run_contact_hole_repair = MagicMock(name="run_contact_hole_repair")
_mock_contact_hole.analyze_contact_hole_candidates = MagicMock(name="analyze")
_mock_contact_hole.run_selected_contact_hole_repair = MagicMock(name="run_selected")

_mock_texture = types.ModuleType("stage_texture_bake")
_mock_texture.bake_texture = MagicMock(name="bake_texture")

_MODULE_PATCHES = {
    "stage_extract_frames": _mock_extract,
    "stage_pi3x_reconstruct": _mock_pi3x,
    "vram_utils": _mock_vram,
    "stage_denoise": _mock_denoise,
    "stage_diffcd_mesh": _mock_diffcd,
    "stage_classical_mesh": _mock_classical,
    "stage_mesh_wrap": _mock_mesh_wrap,
    "stage_contact_hole_repair": _mock_contact_hole,
    "stage_texture_bake": _mock_texture,
}

_saved_modules: dict[str, types.ModuleType | None] = {}
for _name, _mod in _MODULE_PATCHES.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mod

from scripts.dashboard.stage_wrappers import (  # noqa: E402
    _stage_diffcd,
    _stage_extract_frames,
    _stage_pi3x_inference,
)


def _reset_all_mocks() -> None:
    for mod in _MODULE_PATCHES.values():
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, MagicMock):
                obj.reset_mock()


class TestStageExtractFramesArgs(unittest.TestCase):
    """9.2 -- _stage_extract_frames forwards correct args."""

    def setUp(self) -> None:
        _reset_all_mocks()

    def test_forwards_args_to_extract_frames(self) -> None:
        cb = MagicMock()
        _stage_extract_frames(
            "/data/input/video.mp4",
            "/data/output",
            frame_interval=5,
            max_frames=30,
            progress_cb=cb,
            cancel_cb=None,
        )
        _mock_extract.extract_frames.assert_called_once_with(
            "/data/input/video.mp4",
            "/data/output",
            frame_interval=5,
            max_frames=30,
            progress_cb=cb,
            cancel_cb=None,
        )


class TestStagePi3xVRAMCleanup(unittest.TestCase):
    """9.3 -- _stage_pi3x_inference calls cleanup_pytorch_vram after inference."""

    def setUp(self) -> None:
        _reset_all_mocks()

    def test_cleanup_called_after_inference(self) -> None:
        _stage_pi3x_inference(
            "/data/output/frames", "/data/output",
            pixel_limit=255000, pi3x_frame_target=50,
            conf_threshold=0.2, edge_rtol=0.03,
        )
        _mock_pi3x.run_pi3x_inference.assert_called_once()
        _mock_vram.cleanup_pytorch_vram.assert_called_once()


class TestStageDiffcdProcessRegistration(unittest.TestCase):
    """9.4 -- _stage_diffcd passes through register/unregister process."""

    def setUp(self) -> None:
        _reset_all_mocks()

    def test_process_registration_passed_through(self) -> None:
        reg = MagicMock(name="register")
        unreg = MagicMock(name="unregister")
        _stage_diffcd(
            "/data/output/object_denoised.ply",
            "/data/output",
            register_process=reg,
            unregister_process=unreg,
        )
        call_kwargs = _mock_diffcd.run_diffcd.call_args
        self.assertEqual(call_kwargs.kwargs.get("register_process"), reg)
        self.assertEqual(call_kwargs.kwargs.get("unregister_process"), unreg)


class TestUnusedCallbacksDeleted(unittest.TestCase):
    """9.5 -- Wrappers that delete callbacks run without error."""

    def setUp(self) -> None:
        _reset_all_mocks()

    def test_extract_frames_ignores_process_callbacks(self) -> None:
        """Calling with register_process/unregister_process doesn't error."""
        _stage_extract_frames(
            "/data/input/v.mp4", "/data/output",
            frame_interval=10, max_frames=50,
            register_process=MagicMock(),
            unregister_process=MagicMock(),
        )
        _mock_extract.extract_frames.assert_called_once()
        # Verify register_process/unregister_process are NOT in the call kwargs
        call_kwargs = _mock_extract.extract_frames.call_args
        self.assertNotIn("register_process", (call_kwargs.kwargs or {}))
        self.assertNotIn("unregister_process", (call_kwargs.kwargs or {}))


class TestLazyImportVerification(unittest.TestCase):
    """9.1 -- Wrapper functions call the underlying stage function."""

    def setUp(self) -> None:
        _reset_all_mocks()

    def test_extract_frames_calls_underlying(self) -> None:
        _stage_extract_frames(
            "/v.mp4", "/out", frame_interval=10, max_frames=50,
        )
        self.assertTrue(_mock_extract.extract_frames.called)

    def test_pi3x_calls_underlying(self) -> None:
        _stage_pi3x_inference(
            "/frames", "/out",
            pixel_limit=255000, pi3x_frame_target=50,
            conf_threshold=0.2, edge_rtol=0.03,
        )
        self.assertTrue(_mock_pi3x.run_pi3x_inference.called)

    def test_diffcd_calls_underlying(self) -> None:
        _stage_diffcd("/dn.ply", "/out")
        self.assertTrue(_mock_diffcd.run_diffcd.called)


if __name__ == "__main__":
    unittest.main()
