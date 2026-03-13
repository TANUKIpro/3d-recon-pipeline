"""Tests for scripts/dashboard/stage_wrappers.py.

Covers lazy import verification, argument forwarding, VRAM cleanup,
process registration passthrough, and unused callback deletion.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from scripts.dashboard.stage_wrappers import (
    _stage_diffcd,
    _stage_extract_frames,
    _stage_pi3x_inference,
)


class TestStageExtractFramesArgs(unittest.TestCase):
    """9.2 -- _stage_extract_frames forwards correct args."""

    def setUp(self) -> None:
        self.p_extract = patch("stage_extract_frames.extract_frames")
        self.mock_extract = self.p_extract.start()

    def tearDown(self) -> None:
        self.p_extract.stop()

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


class TestStagePi3xVRAMCleanup(unittest.TestCase):
    """9.3 -- _stage_pi3x_inference calls cleanup_pytorch_vram after inference."""

    def setUp(self) -> None:
        self.p_run = patch("stage_pi3x_reconstruct.run_pi3x_inference")
        self.p_cleanup = patch("vram_utils.cleanup_pytorch_vram")
        self.mock_run = self.p_run.start()
        self.mock_cleanup = self.p_cleanup.start()

    def tearDown(self) -> None:
        self.p_run.stop()
        self.p_cleanup.stop()

    def test_cleanup_called_after_inference(self) -> None:
        _stage_pi3x_inference(
            "/data/output/frames", "/data/output",
            pixel_limit=255000, pi3x_frame_target=50,
            conf_threshold=0.2, edge_rtol=0.03,
        )
        self.mock_run.assert_called_once()
        self.mock_cleanup.assert_called_once()


class TestStageDiffcdProcessRegistration(unittest.TestCase):
    """9.4 -- _stage_diffcd passes through register/unregister process."""

    def setUp(self) -> None:
        self.p_diffcd = patch("stage_diffcd_mesh.run_diffcd")
        self.mock_diffcd = self.p_diffcd.start()

    def tearDown(self) -> None:
        self.p_diffcd.stop()

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


class TestUnusedCallbacksDeleted(unittest.TestCase):
    """9.5 -- Wrappers that delete callbacks run without error."""

    def setUp(self) -> None:
        self.p_extract = patch("stage_extract_frames.extract_frames")
        self.mock_extract = self.p_extract.start()

    def tearDown(self) -> None:
        self.p_extract.stop()

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


class TestLazyImportVerification(unittest.TestCase):
    """9.1 -- Wrapper functions call the underlying stage function."""

    def setUp(self) -> None:
        self.p_extract = patch("stage_extract_frames.extract_frames")
        self.p_run = patch("stage_pi3x_reconstruct.run_pi3x_inference")
        self.p_cleanup = patch("vram_utils.cleanup_pytorch_vram")
        self.p_diffcd = patch("stage_diffcd_mesh.run_diffcd")
        self.mock_extract = self.p_extract.start()
        self.mock_run = self.p_run.start()
        self.mock_cleanup = self.p_cleanup.start()
        self.mock_diffcd = self.p_diffcd.start()

    def tearDown(self) -> None:
        self.p_extract.stop()
        self.p_run.stop()
        self.p_cleanup.stop()
        self.p_diffcd.stop()

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
        self.assertTrue(self.mock_run.called)

    def test_diffcd_calls_underlying(self) -> None:
        _stage_diffcd("/dn.ply", "/out")
        self.assertTrue(self.mock_diffcd.called)


if __name__ == "__main__":
    unittest.main()
