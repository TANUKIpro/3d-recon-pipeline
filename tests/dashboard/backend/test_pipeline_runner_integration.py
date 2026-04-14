"""Integration tests for pipeline_runner: run_pipeline, _run_stage, callbacks.

Covers the 5-stage pipeline: ExtractFrames, COLMAP SfM, SAM2 Segment,
MILo Reconstruct, Texture Bake.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.dashboard.pipeline_runner import (
    _CancelledError,
    _make_cancel_cb,
    _make_progress_cb,
    _run_stage,
    run_pipeline,
)
from scripts.dashboard.state import PipelineSession, PipelineStage, StageStatus


# ── Helpers ────────────────────────────────────────────────────────────────

_MODULE = "scripts.dashboard.pipeline_runner"


@contextmanager
def _noop_scope(*_args, **_kwargs):
    yield


class _FakeSAM2Service:
    """Fake SAM2 service that creates mask files on propagate."""

    def __init__(self, output_dir: str, *, with_ground: bool = False) -> None:
        self._output_dir = output_dir
        self._with_ground = with_ground
        self.initialize_calls = 0
        self.propagate_calls = 0
        self.release_calls = 0
        self.modes: list[str] = []

    def initialize(self, frames_dir, output_dir, model):
        self.initialize_calls += 1
        return {"frame_count": 10, "width": 640, "height": 480}

    def propagate_and_save(self, cb=None):
        self.propagate_calls += 1
        mask_dir = str(Path(self._output_dir) / "masks")
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (Path(mask_dir) / f"mask_{i:04d}.png").write_bytes(b"\x89PNG")
        ground_mask_dir = None
        if self._with_ground:
            ground_mask_dir = str(Path(self._output_dir) / "masks_ground")
            Path(ground_mask_dir).mkdir(parents=True, exist_ok=True)
            for i in range(3):
                (Path(ground_mask_dir) / f"mask_{i:04d}.png").write_bytes(b"\x89PNG")
        return mask_dir, ground_mask_dir

    def release(self):
        self.release_calls += 1

    def set_mode(self, mode):
        self.modes.append(mode)


# ── Sentinel file helpers ──────────────────────────────────────────────────


def _populate_frames(d: str, count: int = 5) -> None:
    p = Path(d) / "frames"
    p.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (p / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8")


def _populate_colmap(d: str) -> None:
    p = Path(d)
    (p / "camera_poses.json").write_text("{}", encoding="utf-8")
    colmap_dir = p / "colmap_sparse"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    (colmap_dir / "cameras.bin").write_bytes(b"cam")


def _populate_masks(d: str) -> None:
    masks_dir = Path(d) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (masks_dir / f"mask_{i:04d}.png").write_bytes(b"\x89PNG")


def _populate_mesh(d: str) -> None:
    (Path(d) / "object_mesh.ply").write_bytes(b"ply")


def _populate_cleanup_outputs(d: str) -> None:
    out = Path(d)
    (out / "post_texture_contact_cleanup").mkdir(parents=True, exist_ok=True)
    (out / "post_texture_contact_cleanup" / "proposal.json").write_text(
        '{"status":"proposal_ready","requires_review":false,"recommended_decision":"skip"}',
        encoding="utf-8",
    )
    final_dir = out / out.name
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "textured_mesh_cleaned.obj").write_bytes(b"obj")


# ── Common base ────────────────────────────────────────────────────────────


class _PipelineIntegrationBase(unittest.IsolatedAsyncioTestCase):
    """Shared setUp / tearDown for pipeline integration tests."""

    auto_accept: bool = True
    ground_plane_enabled: bool = False
    with_ground_masks: bool = False

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="test_pipeline_")
        self.session = PipelineSession()
        self.session.config.output_dir = self.tmpdir
        self.session.config.video_path = str(Path(self.tmpdir) / "input.mp4")
        self.session.config.auto_accept = self.auto_accept
        self.session.config.ground_plane_enabled = self.ground_plane_enabled
        Path(self.session.config.video_path).write_bytes(b"MP4")
        self.sam2 = _FakeSAM2Service(
            self.tmpdir, with_ground=self.with_ground_masks,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- Sentinel side-effects --

    def _make_stage_side_effects(self) -> dict:
        d = self.tmpdir
        return {
            "extract_frames": lambda *a, **kw: _populate_frames(d),
            "colmap_sfm": lambda *a, **kw: _populate_colmap(d),
            "milo_reconstruct": lambda *a, **kw: _populate_mesh(d),
            "texture_bake": lambda *a, **kw: (
                Path(d) / "textured_mesh.obj"
            ).write_bytes(b"obj"),
            "post_texture_prepare": lambda *a, **kw: (
                (Path(d) / "post_texture_contact_cleanup").mkdir(parents=True, exist_ok=True),
                (Path(d) / "post_texture_contact_cleanup" / "proposal.json").write_text(
                    '{"status":"proposal_ready","requires_review":false,"recommended_decision":"skip"}',
                    encoding="utf-8",
                ),
            )[1] and {"requires_review": False, "recommended_decision": "skip"},
            "post_texture_apply": lambda *a, **kw: (
                _populate_cleanup_outputs(d),
                str(Path(d) / Path(d).name / "textured_mesh_cleaned.obj"),
            )[1],
            "post_texture_skip": lambda *a, **kw: (
                _populate_cleanup_outputs(d),
                str(Path(d) / Path(d).name / "textured_mesh_cleaned.obj"),
            )[1],
        }

    # -- Broadcast helpers --

    @staticmethod
    def _broadcasts_of_type(mock_broadcast: AsyncMock, msg_type: str) -> list[dict]:
        return [
            call.args[1]
            for call in mock_broadcast.call_args_list
            if isinstance(call.args[1], dict) and call.args[1].get("type") == msg_type
        ]

    # -- Event signalers --

    async def _auto_signal_sam2(self, delay: float = 0.02) -> None:
        """Signal SAM2 confirm + approve so stage 3 completes."""
        await asyncio.sleep(delay)
        self.session.sam2_confirm_event.set()
        await asyncio.sleep(delay)
        self.session.sam2_approved = True
        self.session.sam2_approve_event.set()

    def _start_sam2_signaler(self, delay: float = 0.02) -> asyncio.Task:
        return asyncio.create_task(self._auto_signal_sam2(delay))

    # -- Patch helper --

    def _enter_all_patches(
        self,
        stack: ExitStack,
        *,
        extra_overrides: dict | None = None,
    ) -> dict[str, MagicMock | AsyncMock]:
        """Activate patches for all stage wrappers + broadcast. Returns mocks dict."""
        effects = self._make_stage_side_effects()
        targets: list[tuple[str, MagicMock | AsyncMock]] = [
            (f"{_MODULE}.broadcast", AsyncMock()),
            (f"{_MODULE}.stage_log_scope", MagicMock(side_effect=_noop_scope)),
            (f"{_MODULE}._stage_extract_frames", MagicMock(side_effect=effects["extract_frames"])),
            (f"{_MODULE}._stage_colmap_sfm", MagicMock(side_effect=effects["colmap_sfm"])),
            (f"{_MODULE}._stage_milo_reconstruct", MagicMock(side_effect=effects["milo_reconstruct"])),
            (f"{_MODULE}._stage_texture_bake", MagicMock(side_effect=effects["texture_bake"])),
            (f"{_MODULE}._stage_post_texture_contact_cleanup_prepare", MagicMock(side_effect=effects["post_texture_prepare"])),
            (f"{_MODULE}._stage_post_texture_contact_cleanup_apply", MagicMock(side_effect=effects["post_texture_apply"])),
            (f"{_MODULE}._stage_post_texture_contact_cleanup_skip", MagicMock(side_effect=effects["post_texture_skip"])),
            (f"{_MODULE}._vram_gate", MagicMock()),
        ]
        overrides = extra_overrides or {}
        for i, (target, default_mock) in enumerate(targets):
            short = target.rsplit(".", 1)[-1]
            if short in overrides:
                targets[i] = (target, overrides[short])

        mocks: dict[str, MagicMock | AsyncMock] = {}
        for target, mock_val in targets:
            short = target.rsplit(".", 1)[-1]
            mocks[short] = stack.enter_context(patch(target, mock_val))
        return mocks


# ═══════════════════════════════════════════════════════════════════════════
# _run_stage tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunStage(unittest.IsolatedAsyncioTestCase):
    """Direct tests for _run_stage lifecycle, errors, callbacks, scope."""

    def setUp(self) -> None:
        self.session = PipelineSession()

    # lifecycle broadcast order
    async def test_lifecycle_order(self) -> None:
        mock_fn = MagicMock()

        with patch(f"{_MODULE}.broadcast", new_callable=AsyncMock) as bc, \
             patch(f"{_MODULE}.stage_log_scope", MagicMock(side_effect=_noop_scope)):
            await _run_stage(self.session, PipelineStage.EXTRACT_FRAMES, mock_fn)

        # Session status should be COMPLETE
        info = self.session.stages[int(PipelineStage.EXTRACT_FRAMES)]
        self.assertEqual(info.status, StageStatus.COMPLETE)

        types = [
            c.args[1]["type"]
            for c in bc.call_args_list
            if isinstance(c.args[1], dict)
        ]
        self.assertEqual(types[0], "stage_start")
        self.assertEqual(types[1], "stage_progress")
        self.assertEqual(bc.call_args_list[1].args[1]["progress"], 0.0)
        self.assertEqual(types[-1], "stage_complete")

    # exception marks stage FAILED
    async def test_fn_exception_marks_failed(self) -> None:
        mock_fn = MagicMock(side_effect=ValueError("boom"))

        with patch(f"{_MODULE}.broadcast", new_callable=AsyncMock) as bc, \
             patch(f"{_MODULE}.stage_log_scope", MagicMock(side_effect=_noop_scope)):
            with self.assertRaises(ValueError):
                await _run_stage(
                    self.session, PipelineStage.EXTRACT_FRAMES, mock_fn,
                )

        info = self.session.stages[int(PipelineStage.EXTRACT_FRAMES)]
        self.assertEqual(info.status, StageStatus.FAILED)
        self.assertEqual(info.error, "boom")

        # Last broadcast should be stage_complete with error
        last = bc.call_args_list[-1].args[1]
        self.assertEqual(last["type"], "stage_complete")
        self.assertEqual(last["error"], "boom")

    # callbacks passed to fn
    async def test_callbacks_passed_to_fn(self) -> None:
        mock_fn = MagicMock()

        with patch(f"{_MODULE}.broadcast", new_callable=AsyncMock), \
             patch(f"{_MODULE}.stage_log_scope", MagicMock(side_effect=_noop_scope)):
            await _run_stage(self.session, PipelineStage.MILO_RECONSTRUCT, mock_fn)

        mock_fn.assert_called_once()
        kw = mock_fn.call_args.kwargs
        for key in ("progress_cb", "cancel_cb", "register_process", "unregister_process"):
            self.assertIn(key, kw, f"Missing keyword arg: {key}")
            self.assertTrue(callable(kw[key]), f"{key} should be callable")

    # stage_log_scope wrapping
    async def test_stage_log_scope_wrapping(self) -> None:
        mock_fn = MagicMock()
        mock_scope = MagicMock(side_effect=_noop_scope)

        with patch(f"{_MODULE}.broadcast", new_callable=AsyncMock), \
             patch(f"{_MODULE}.stage_log_scope", mock_scope):
            await _run_stage(self.session, PipelineStage.MILO_RECONSTRUCT, mock_fn)

        mock_scope.assert_called_once_with(int(PipelineStage.MILO_RECONSTRUCT))
        mock_fn.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# _make_progress_cb / _make_cancel_cb tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMakeProgressCb(unittest.TestCase):
    """_make_progress_cb uses loop.call_soon_threadsafe."""

    def test_uses_call_soon_threadsafe(self) -> None:
        session = PipelineSession()
        session.stage_start(PipelineStage.EXTRACT_FRAMES)
        mock_loop = MagicMock()

        cb = _make_progress_cb(session, PipelineStage.EXTRACT_FRAMES, mock_loop)
        cb(50.0, "Half done")

        mock_loop.call_soon_threadsafe.assert_called_once()


class TestMakeCancelCb(unittest.TestCase):
    """_make_cancel_cb raises on cancellation."""

    def test_cancel_cb_raises(self) -> None:
        session = PipelineSession()

        cb = _make_cancel_cb(session)

        # Not cancelled -> no error
        cb()

        # Cancelled -> _CancelledError
        session.cancelled = True
        with self.assertRaises(_CancelledError):
            cb()


# ═══════════════════════════════════════════════════════════════════════════
# run_pipeline integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunPipelineFullFlow(_PipelineIntegrationBase):
    """Full 5-stage pipeline with auto_accept=True."""

    auto_accept = True

    async def test_full_stage_flow(self) -> None:
        with ExitStack() as stack:
            mocks = self._enter_all_patches(stack)
            signaler = self._start_sam2_signaler()
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not signaler.done():
                    signaler.cancel()

        # All stage functions called once
        mocks["_stage_extract_frames"].assert_called_once()
        mocks["_stage_colmap_sfm"].assert_called_once()
        mocks["_stage_milo_reconstruct"].assert_called_once()
        mocks["_stage_texture_bake"].assert_called_once()

        # Broadcast checks: 6 stage_starts (non-SAM2 via _run_stage, SAM2 inline, stage 6 inline)
        bc = mocks["broadcast"]
        starts = self._broadcasts_of_type(bc, "stage_start")
        # Stages 1,2,4,5 via _run_stage (4) + stage 3 inline (1) + stage 6 inline (1) = 6
        self.assertEqual(len(starts), 6)

        completes = self._broadcasts_of_type(bc, "stage_complete")
        self.assertEqual(len(completes), 6)

        pipeline_complete = self._broadcasts_of_type(bc, "pipeline_complete")
        self.assertEqual(len(pipeline_complete), 1)

        # Session no longer running
        self.assertFalse(self.session.running)


class TestRunPipelineResumeFromStage(_PipelineIntegrationBase):
    """Resume from MILO_RECONSTRUCT (stage 4)."""

    auto_accept = True

    async def test_resume_from_stage(self) -> None:
        # Pre-create outputs for stages 1-3
        _populate_frames(self.tmpdir)
        _populate_colmap(self.tmpdir)
        _populate_masks(self.tmpdir)

        # Set session paths that stages 1-3 would have set
        self.session.resume_from_stage = PipelineStage.MILO_RECONSTRUCT
        self.session.frames_dir = str(Path(self.tmpdir) / "frames")
        self.session.colmap_sparse_path = str(Path(self.tmpdir) / "colmap_sparse")
        self.session.poses_path = str(Path(self.tmpdir) / "camera_poses.json")
        self.session.mask_dir = str(Path(self.tmpdir) / "masks")

        with ExitStack() as stack:
            mocks = self._enter_all_patches(stack)
            # No SAM2 signaler needed since we skip stage 3
            await run_pipeline(self.session, self.sam2)

        # Stages 1-3 NOT called
        mocks["_stage_extract_frames"].assert_not_called()
        mocks["_stage_colmap_sfm"].assert_not_called()

        # Stages 4-5 called
        mocks["_stage_milo_reconstruct"].assert_called_once()
        mocks["_stage_texture_bake"].assert_called_once()

        # Broadcast stage_starts are only for stage 4+
        bc = mocks["broadcast"]
        starts = self._broadcasts_of_type(bc, "stage_start")
        start_stages = {s["stage"] for s in starts}
        self.assertTrue(start_stages.issubset({4, 5, 6}))
        self.assertNotIn(1, start_stages)
        self.assertNotIn(2, start_stages)
        self.assertNotIn(3, start_stages)


class TestRunPipelineCancelBroadcastsError(_PipelineIntegrationBase):
    """Cancel during stage 1 broadcasts pipeline_error."""

    auto_accept = True

    async def test_cancel_broadcasts_error(self) -> None:
        def _cancel_in_stage1(*args, **kwargs):
            _populate_frames(self.tmpdir)
            self.session.request_cancel()

        with ExitStack() as stack:
            mocks = self._enter_all_patches(
                stack,
                extra_overrides={
                    "_stage_extract_frames": MagicMock(side_effect=_cancel_in_stage1),
                },
            )
            signaler = self._start_sam2_signaler()
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not signaler.done():
                    signaler.cancel()

        bc = mocks["broadcast"]
        errors = self._broadcasts_of_type(bc, "pipeline_error")
        self.assertTrue(len(errors) >= 1)
        self.assertIn(errors[0]["reason_code"], ("cancelled", "cancelled_force"))


class TestRunPipelineCancelHydrates(_PipelineIntegrationBase):
    """Cancel triggers hydrate_from_output_dir."""

    auto_accept = True

    async def test_cancel_hydrates(self) -> None:
        def _cancel_in_stage1(*args, **kwargs):
            _populate_frames(self.tmpdir)
            self.session.request_cancel()

        with ExitStack() as stack, \
             patch.object(self.session, "hydrate_from_output_dir", wraps=self.session.hydrate_from_output_dir) as mock_hydrate:
            mocks = self._enter_all_patches(
                stack,
                extra_overrides={
                    "_stage_extract_frames": MagicMock(side_effect=_cancel_in_stage1),
                },
            )
            signaler = self._start_sam2_signaler()
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not signaler.done():
                    signaler.cancel()

        mock_hydrate.assert_called_once_with(self.tmpdir)


class TestRunPipelineExceptionBroadcastsError(_PipelineIntegrationBase):
    """Stage 1 exception broadcasts pipeline_error."""

    auto_accept = True

    async def test_exception_broadcasts_error(self) -> None:
        with ExitStack() as stack:
            mocks = self._enter_all_patches(
                stack,
                extra_overrides={
                    "_stage_extract_frames": MagicMock(
                        side_effect=RuntimeError("boom"),
                    ),
                },
            )
            signaler = self._start_sam2_signaler()
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not signaler.done():
                    signaler.cancel()

        bc = mocks["broadcast"]
        errors = self._broadcasts_of_type(bc, "pipeline_error")
        self.assertTrue(len(errors) >= 1)
        self.assertIn("boom", errors[0]["error"])

        self.assertEqual(
            self.session.stages[int(PipelineStage.EXTRACT_FRAMES)].status,
            StageStatus.FAILED,
        )


class TestRunPipelineFinallyReleasesSAM2(_PipelineIntegrationBase):
    """SAM2 service released in finally block."""

    auto_accept = True

    async def test_finally_releases_sam2(self) -> None:
        with ExitStack() as stack:
            self._enter_all_patches(stack)
            signaler = self._start_sam2_signaler()
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not signaler.done():
                    signaler.cancel()

        self.assertGreaterEqual(self.sam2.release_calls, 1)
        self.assertFalse(self.session.running)

    async def test_finally_releases_sam2_on_error(self) -> None:
        with ExitStack() as stack:
            self._enter_all_patches(
                stack,
                extra_overrides={
                    "_stage_extract_frames": MagicMock(
                        side_effect=RuntimeError("error"),
                    ),
                },
            )
            await run_pipeline(self.session, self.sam2)

        self.assertGreaterEqual(self.sam2.release_calls, 1)
        self.assertFalse(self.session.running)


class TestRunPipelineSAM2Redo(_PipelineIntegrationBase):
    """SAM2 redo loop re-initializes and re-propagates."""

    auto_accept = True

    async def test_sam2_redo_loop(self) -> None:
        async def _redo_then_approve(delay=0.02):
            # Round 1: confirm, then reject (redo)
            await asyncio.sleep(delay)
            self.session.sam2_confirm_event.set()
            await asyncio.sleep(delay)
            self.session.sam2_approved = False
            self.session.sam2_approve_event.set()
            # Round 2: wait for re-init, then confirm + approve
            while self.sam2.initialize_calls < 2:
                await asyncio.sleep(0.005)
            await asyncio.sleep(delay)
            self.session.sam2_confirm_event.set()
            await asyncio.sleep(delay)
            self.session.sam2_approved = True
            self.session.sam2_approve_event.set()

        with ExitStack() as stack:
            self._enter_all_patches(stack)
            task = asyncio.create_task(_redo_then_approve())
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not task.done():
                    task.cancel()

        self.assertEqual(self.sam2.initialize_calls, 2)
        self.assertEqual(self.sam2.propagate_calls, 2)


class TestRunPipelineAutoAcceptPassesAll(_PipelineIntegrationBase):
    """auto_accept=True auto-passes confirmations."""

    auto_accept = True

    async def test_auto_accept_passes_all(self) -> None:
        with ExitStack() as stack:
            mocks = self._enter_all_patches(stack)
            signaler = self._start_sam2_signaler()
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not signaler.done():
                    signaler.cancel()

        # All next_stage_confirmation_required broadcasts have auto_accepted=True
        bc = mocks["broadcast"]
        confirms = self._broadcasts_of_type(bc, "next_stage_confirmation_required")
        for c in confirms:
            self.assertTrue(c["auto_accepted"], f"Confirmation not auto-accepted: {c}")

        # Pipeline completed successfully
        pipeline_complete = self._broadcasts_of_type(bc, "pipeline_complete")
        self.assertEqual(len(pipeline_complete), 1)


class TestRunPipelineGroundPlaneFlow(_PipelineIntegrationBase):
    """Ground plane segmentation with confirm."""

    auto_accept = False
    ground_plane_enabled = True
    with_ground_masks = True

    async def test_ground_plane_flow(self) -> None:
        async def _signal_events(delay=0.02):
            # SAM2 object confirm
            await asyncio.sleep(delay)
            self.session.sam2_confirm_event.set()
            # Ground plane confirm
            await asyncio.sleep(delay)
            self.session.sam2_confirm_event.set()
            # Approve masks
            await asyncio.sleep(delay)
            self.session.sam2_approved = True
            self.session.sam2_approve_event.set()

        async def _auto_confirm_noop(session, from_stage, to_stage, message):
            pass

        with ExitStack() as stack:
            mocks = self._enter_all_patches(stack)
            stack.enter_context(
                patch(f"{_MODULE}._wait_for_next_stage_confirmation", _auto_confirm_noop),
            )
            task = asyncio.create_task(_signal_events())
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not task.done():
                    task.cancel()

        bc = mocks["broadcast"]

        # Ground phase broadcast
        ground_phase = self._broadcasts_of_type(bc, "sam2_ground_phase")
        self.assertTrue(len(ground_phase) >= 1)

        # SAM2 service mode sequence: ground -> object
        self.assertIn("ground", self.sam2.modes)
        self.assertIn("object", self.sam2.modes)
        ground_idx = self.sam2.modes.index("ground")
        object_idx = self.sam2.modes.index("object")
        self.assertLess(ground_idx, object_idx)


class TestRunPipelineGroundSkip(_PipelineIntegrationBase):
    """Ground plane segmentation skipped."""

    auto_accept = False
    ground_plane_enabled = True
    with_ground_masks = False

    async def test_ground_skip(self) -> None:
        async def _signal_events(delay=0.02):
            # SAM2 object confirm
            await asyncio.sleep(delay)
            self.session.sam2_confirm_event.set()
            # Skip ground plane
            await asyncio.sleep(delay)
            self.session.sam2_ground_skip_event.set()
            # Approve masks
            await asyncio.sleep(delay)
            self.session.sam2_approved = True
            self.session.sam2_approve_event.set()

        async def _auto_confirm_noop(session, from_stage, to_stage, message):
            pass

        with ExitStack() as stack:
            mocks = self._enter_all_patches(stack)
            stack.enter_context(
                patch(f"{_MODULE}._wait_for_next_stage_confirmation", _auto_confirm_noop),
            )
            task = asyncio.create_task(_signal_events())
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not task.done():
                    task.cancel()

        bc = mocks["broadcast"]
        skipped = self._broadcasts_of_type(bc, "sam2_ground_skipped")
        self.assertTrue(len(skipped) >= 1)


class TestRunPipelineGroundPlaneAutoAccept(_PipelineIntegrationBase):
    """Ground plane phase runs even with auto_accept=True."""

    auto_accept = True
    ground_plane_enabled = True
    with_ground_masks = True

    async def test_ground_phase_runs_with_auto_accept(self) -> None:
        async def _signal_events(delay=0.02):
            # SAM2 object confirm
            await asyncio.sleep(delay)
            self.session.sam2_confirm_event.set()
            # Ground plane confirm
            await asyncio.sleep(delay)
            self.session.sam2_confirm_event.set()
            # Approve masks
            await asyncio.sleep(delay)
            self.session.sam2_approved = True
            self.session.sam2_approve_event.set()

        async def _auto_confirm_noop(session, from_stage, to_stage, message):
            pass

        with ExitStack() as stack:
            mocks = self._enter_all_patches(stack)
            stack.enter_context(
                patch(f"{_MODULE}._wait_for_next_stage_confirmation", _auto_confirm_noop),
            )
            task = asyncio.create_task(_signal_events())
            try:
                await run_pipeline(self.session, self.sam2)
            finally:
                if not task.done():
                    task.cancel()

        bc = mocks["broadcast"]

        # Ground phase broadcast must have been sent
        ground_phase = self._broadcasts_of_type(bc, "sam2_ground_phase")
        self.assertTrue(len(ground_phase) >= 1)

        # SAM2 service mode sequence: ground -> object
        self.assertIn("ground", self.sam2.modes)
        self.assertIn("object", self.sam2.modes)
        ground_idx = self.sam2.modes.index("ground")
        object_idx = self.sam2.modes.index("object")
        self.assertLess(ground_idx, object_idx)


if __name__ == "__main__":
    unittest.main()
