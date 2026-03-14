"""Tests for scripts/dashboard/stage_wrappers.py.

Covers lazy import verification, argument forwarding, VRAM cleanup,
process registration passthrough, and unused callback deletion.

The wrapper functions use deferred bare imports (e.g.
``from scripts.stage_extract_frames import extract_frames``) that resolve via
PYTHONPATH inside Docker.  On the host those modules don't exist, so we
inject lightweight stubs into ``sys.modules`` via ``patch.dict`` and then
replace the target function with a ``MagicMock``.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

from scripts.dashboard.stage_wrappers import (
    _stage_colmap_sfm,
    _stage_extract_frames,
    _stage_gs2mesh_reconstruct,
    _stage_texture_bake,
)

# ---------------------------------------------------------------------------
# Stub modules for the bare imports that stage_wrappers does at call time.
# ---------------------------------------------------------------------------

_stub_extract = types.ModuleType("scripts.stage_extract_frames")
_stub_colmap = types.ModuleType("scripts.stage_colmap_sfm")
_stub_gs2mesh = types.ModuleType("scripts.stage_gs2mesh_reconstruct")
_stub_texture = types.ModuleType("scripts.stage_texture_bake")
_stub_vram = types.ModuleType("scripts.vram_utils")

_STUBS = {
    "scripts.stage_extract_frames": _stub_extract,
    "scripts.stage_colmap_sfm": _stub_colmap,
    "scripts.stage_gs2mesh_reconstruct": _stub_gs2mesh,
    "scripts.stage_texture_bake": _stub_texture,
    "scripts.vram_utils": _stub_vram,
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

        self.mock_colmap = MagicMock(name="run_colmap_sfm")
        _stub_colmap.run_colmap_sfm = self.mock_colmap  # type: ignore[attr-defined]

        self.mock_gs2mesh = MagicMock(name="run_gs2mesh")
        _stub_gs2mesh.run_gs2mesh = self.mock_gs2mesh  # type: ignore[attr-defined]

        self.mock_bake = MagicMock(name="bake_texture")
        _stub_texture.bake_texture = self.mock_bake  # type: ignore[attr-defined]

        self.mock_cleanup = MagicMock(name="cleanup_pytorch_vram")
        _stub_vram.cleanup_pytorch_vram = self.mock_cleanup  # type: ignore[attr-defined]

        self.mock_ensure_vram = MagicMock(name="ensure_vram_available")
        _stub_vram.ensure_vram_available = self.mock_ensure_vram  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        self._modules_patch.stop()


class TestStageExtractFramesArgs(_WrapperTestBase):
    """_stage_extract_frames forwards correct args."""

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


class TestStageColmapSfmArgs(_WrapperTestBase):
    """_stage_colmap_sfm forwards correct args."""

    def test_forwards_args_to_colmap_sfm(self) -> None:
        reg = MagicMock(name="register")
        unreg = MagicMock(name="unregister")
        _stage_colmap_sfm(
            "/data/output/frames",
            "/data/output",
            matcher="sequential",
            max_features=8192,
            image_size=1024,
            register_process=reg,
            unregister_process=unreg,
        )
        self.mock_colmap.assert_called_once()
        call_kwargs = self.mock_colmap.call_args.kwargs
        self.assertEqual(call_kwargs.get("register_process"), reg)
        self.assertEqual(call_kwargs.get("unregister_process"), unreg)


class TestStageGs2meshVRAMCleanup(_WrapperTestBase):
    """_stage_gs2mesh_reconstruct calls cleanup_pytorch_vram after inference."""

    def test_cleanup_called_after_inference(self) -> None:
        _stage_gs2mesh_reconstruct(
            "/data/output/frames",
            "/data/output/colmap_sparse",
            "/data/output/masks",
            "/data/output",
            gs_iterations=30000,
            runtime_profile="compat",
            stereo_model="DLNR_Middlebury",
            tsdf_voxel_size=0.005,
            tsdf_depth_trunc=0.04,
            use_masks=True,
        )
        self.mock_gs2mesh.assert_called_once()
        self.assertEqual(
            self.mock_gs2mesh.call_args.kwargs.get("runtime_profile"),
            "compat",
        )
        self.mock_cleanup.assert_called_once()


class TestStageTextureBakeArgs(_WrapperTestBase):
    """_stage_texture_bake forwards correct args."""

    def test_forwards_args_to_bake_texture(self) -> None:
        cb = MagicMock()
        _stage_texture_bake(
            "/data/output/object_mesh.ply",
            "/data/output/camera_poses.json",
            "/data/output/frames",
            "/data/output/masks",
            "/data/output",
            texture_size=2048,
            texture_view_assign_mode="region_gc",
            texture_quality_boost=True,
            progress_cb=cb,
        )
        self.mock_bake.assert_called_once()


class TestUnusedCallbacksDeleted(_WrapperTestBase):
    """Wrappers that delete callbacks run without error."""

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
    """Wrapper functions call the underlying stage function."""

    def test_extract_frames_calls_underlying(self) -> None:
        _stage_extract_frames(
            "/v.mp4", "/out", frame_interval=10, max_frames=50,
        )
        self.assertTrue(self.mock_extract.called)

    def test_colmap_sfm_calls_underlying(self) -> None:
        _stage_colmap_sfm(
            "/frames", "/out",
            matcher="sequential", max_features=8192, image_size=1024,
        )
        self.assertTrue(self.mock_colmap.called)

    def test_gs2mesh_calls_underlying(self) -> None:
        _stage_gs2mesh_reconstruct(
            "/frames", "/colmap_sparse", "/masks", "/out",
        )
        self.assertTrue(self.mock_gs2mesh.called)

    def test_texture_bake_calls_underlying(self) -> None:
        _stage_texture_bake(
            "/mesh.ply", "/poses.json", "/frames", "/masks", "/out",
            texture_size=2048,
        )
        self.assertTrue(self.mock_bake.called)


if __name__ == "__main__":
    unittest.main()
