"""Tests for scripts/dashboard/stage_wrappers.py.

Covers lazy import verification, argument forwarding, VRAM cleanup,
process registration passthrough, and unused callback deletion.

The wrapper functions use deferred bare imports (e.g.
``from stage_extract_frames import extract_frames``) that resolve via
PYTHONPATH inside Docker.  On the host those modules don't exist, so we
inject lightweight stubs into ``sys.modules`` via ``patch.dict`` and then
replace the target function with a ``MagicMock``.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

from scripts.dashboard.stage_wrappers import (
    _stage_diffcd,
    _stage_extract_frames,
    _stage_pi3x_inference,
)

# ---------------------------------------------------------------------------
# Stub modules for the bare imports that stage_wrappers does at call time.
# ---------------------------------------------------------------------------

_stub_extract = types.ModuleType("scripts.stage_extract_frames")
_stub_pi3x = types.ModuleType("scripts.stage_pi3x_reconstruct")
_stub_vram = types.ModuleType("scripts.vram_utils")
_stub_diffcd = types.ModuleType("scripts.stage_diffcd_mesh")

_STUBS = {
    "scripts.stage_extract_frames": _stub_extract,
    "scripts.stage_pi3x_reconstruct": _stub_pi3x,
    "scripts.vram_utils": _stub_vram,
    "scripts.stage_diffcd_mesh": _stub_diffcd,
}


class _WrapperTestBase(unittest.TestCase):
    """Install stub modules and assign fresh MagicMocks before each test."""

    def setUp(self) -> None:
        # Install stubs into sys.modules for the duration of the test.
        self._modules_patch = patch.dict("sys.modules", _STUBS)
        self._modules_patch.start()

        # Assign fresh mocks so each test gets a clean slate.
        self.mock_extract = MagicMock(name="extract_frames")
        _stub_extract.extract_frames = self.mock_extract  # type: ignore[attr-defined]

        self.mock_run_pi3x = MagicMock(name="run_pi3x_inference")
        _stub_pi3x.run_pi3x_inference = self.mock_run_pi3x  # type: ignore[attr-defined]

        self.mock_cleanup = MagicMock(name="cleanup_pytorch_vram")
        _stub_vram.cleanup_pytorch_vram = self.mock_cleanup  # type: ignore[attr-defined]

        self.mock_diffcd = MagicMock(name="run_diffcd")
        _stub_diffcd.run_diffcd = self.mock_diffcd  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        self._modules_patch.stop()


class TestStageExtractFramesArgs(_WrapperTestBase):
    """9.2 -- _stage_extract_frames forwards correct args."""

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
        self.mock_extract.assert_called_once_with(
            "/data/input/video.mp4",
            "/data/output",
            frame_interval=5,
            max_frames=30,
            progress_cb=cb,
            cancel_cb=None,
        )


class TestStagePi3xVRAMCleanup(_WrapperTestBase):
    """9.3 -- _stage_pi3x_inference calls cleanup_pytorch_vram after inference."""

    def test_cleanup_called_after_inference(self) -> None:
        _stage_pi3x_inference(
            "/data/output/frames", "/data/output",
            pixel_limit=255000, pi3x_frame_target=50,
            conf_threshold=0.2, edge_rtol=0.03,
        )
        self.mock_run_pi3x.assert_called_once()
        self.mock_cleanup.assert_called_once()


class TestStageDiffcdProcessRegistration(_WrapperTestBase):
    """9.4 -- _stage_diffcd passes through register/unregister process."""

    def test_process_registration_passed_through(self) -> None:
        reg = MagicMock(name="register")
        unreg = MagicMock(name="unregister")
        _stage_diffcd(
            "/data/output/object_denoised.ply",
            "/data/output",
            register_process=reg,
            unregister_process=unreg,
        )
        call_kwargs = self.mock_diffcd.call_args
        self.assertEqual(call_kwargs.kwargs.get("register_process"), reg)
        self.assertEqual(call_kwargs.kwargs.get("unregister_process"), unreg)


class TestUnusedCallbacksDeleted(_WrapperTestBase):
    """9.5 -- Wrappers that delete callbacks run without error."""

    def test_extract_frames_ignores_process_callbacks(self) -> None:
        """Calling with register_process/unregister_process doesn't error."""
        _stage_extract_frames(
            "/data/input/v.mp4", "/data/output",
            frame_interval=10, max_frames=50,
            register_process=MagicMock(),
            unregister_process=MagicMock(),
        )
        self.mock_extract.assert_called_once()
        # Verify register_process/unregister_process are NOT in the call kwargs
        call_kwargs = self.mock_extract.call_args
        self.assertNotIn("register_process", (call_kwargs.kwargs or {}))
        self.assertNotIn("unregister_process", (call_kwargs.kwargs or {}))


class TestLazyImportVerification(_WrapperTestBase):
    """9.1 -- Wrapper functions call the underlying stage function."""

    def test_extract_frames_calls_underlying(self) -> None:
        _stage_extract_frames(
            "/v.mp4", "/out", frame_interval=10, max_frames=50,
        )
        self.assertTrue(self.mock_extract.called)

    def test_pi3x_calls_underlying(self) -> None:
        _stage_pi3x_inference(
            "/frames", "/out",
            pixel_limit=255000, pi3x_frame_target=50,
            conf_threshold=0.2, edge_rtol=0.03,
        )
        self.assertTrue(self.mock_run_pi3x.called)

    def test_diffcd_calls_underlying(self) -> None:
        _stage_diffcd("/dn.ply", "/out")
        self.assertTrue(self.mock_diffcd.called)


if __name__ == "__main__":
    unittest.main()
