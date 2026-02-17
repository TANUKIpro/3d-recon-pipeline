"""Comprehensive tests for scripts/dashboard/state.py.

Covers enums, dataclasses, session lifecycle, cancellation, process management,
progress computation, status serialisation, stage output detection, and hydration.
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from dataclasses import fields as dataclass_fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

from scripts.dashboard.state import (
    STAGE_LABELS,
    STAGE_OUTPUT_FILES,
    PipelineConfig,
    PipelineSession,
    PipelineStage,
    StageInfo,
    StageStatus,
    detect_stage_outputs,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "coffee01"


# ------------------------------------------------------------------ helpers ---


class _FakeProcess:
    """Minimal subprocess-like stub for process management tests."""

    def __init__(
        self,
        *,
        pid: int = 12345,
        poll_return: int | None = None,
    ) -> None:
        self.pid = pid
        self._poll_return = poll_return
        self.terminate = MagicMock()
        self.kill = MagicMock()
        self.wait = MagicMock()

    def poll(self) -> int | None:
        return self._poll_return


# ------------------------------------------------------------------ enums ---


class TestStageStatusEnum(unittest.TestCase):
    """StageStatus values match the JSON contract and support str serialisation."""

    def test_enum_values_match_json_contract(self) -> None:
        expected = {
            "pending": StageStatus.PENDING,
            "running": StageStatus.RUNNING,
            "complete": StageStatus.COMPLETE,
            "failed": StageStatus.FAILED,
            "interactive": StageStatus.INTERACTIVE,
        }
        for value, member in expected.items():
            self.assertEqual(member.value, value)

    def test_str_serialisation(self) -> None:
        for member in StageStatus:
            self.assertIsInstance(member.value, str)
            # str(Enum) should be usable in JSON payloads
            self.assertEqual(str(member.value), member.value)


class TestPipelineStageEnum(unittest.TestCase):
    """PipelineStage IntEnum ordering and label/output coverage."""

    def test_int_enum_ordering(self) -> None:
        ordered = list(PipelineStage)
        for i in range(len(ordered) - 1):
            self.assertLess(ordered[i], ordered[i + 1])

    def test_stage_labels_cover_stages_1_to_8(self) -> None:
        for stage_id in range(1, 9):
            self.assertIn(stage_id, STAGE_LABELS, f"Stage {stage_id} missing from STAGE_LABELS")

    def test_stage_output_files_cover_stages_2_to_8(self) -> None:
        for stage_id in range(2, 9):
            self.assertIn(stage_id, STAGE_OUTPUT_FILES, f"Stage {stage_id} missing from STAGE_OUTPUT_FILES")
            self.assertGreater(len(STAGE_OUTPUT_FILES[stage_id]), 0)


# -------------------------------------------------------------- dataclasses ---


class TestPipelineConfig(unittest.TestCase):
    """PipelineConfig default construction and serialisation."""

    def test_default_construction(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.video_path, "")
        self.assertEqual(cfg.output_dir, "/data/output")
        self.assertEqual(cfg.object_name, "")

    def test_from_dict_roundtrip(self) -> None:
        original = PipelineConfig(video_path="/tmp/v.mp4", max_frames=99)
        d = original.to_dict()
        restored = PipelineConfig.from_dict(d)
        self.assertEqual(restored.video_path, "/tmp/v.mp4")
        self.assertEqual(restored.max_frames, 99)

    def test_to_dict_contains_all_fields(self) -> None:
        cfg = PipelineConfig()
        d = cfg.to_dict()
        field_names = {f.name for f in dataclass_fields(PipelineConfig)}
        self.assertEqual(set(d.keys()), field_names)

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"video_path": "/tmp/v.mp4", "unknown_key_xyz": 42}
        cfg = PipelineConfig.from_dict(d)
        self.assertEqual(cfg.video_path, "/tmp/v.mp4")
        self.assertFalse(hasattr(cfg, "unknown_key_xyz"))

    def test_from_dict_to_dict_roundtrip_preserves_all_values(self) -> None:
        cfg = PipelineConfig(
            video_path="/v.mp4",
            frame_interval=5,
            max_frames=30,
            pixel_limit=100_000,
            confidence_threshold=0.5,
            auto_accept=True,
        )
        restored = PipelineConfig.from_dict(cfg.to_dict())
        for f in dataclass_fields(PipelineConfig):
            self.assertEqual(
                getattr(cfg, f.name),
                getattr(restored, f.name),
                f"Mismatch on field {f.name}",
            )


class TestStageInfo(unittest.TestCase):
    """StageInfo default values."""

    def test_default_values(self) -> None:
        info = StageInfo()
        self.assertEqual(info.status, StageStatus.PENDING)
        self.assertIsNone(info.start_time)
        self.assertIsNone(info.elapsed)
        self.assertIsNone(info.error)
        self.assertAlmostEqual(info.progress, 0.0)
        self.assertIsNone(info.detail)
        self.assertIsNone(info.checkpoint_id)


# ---------------------------------------------------- PipelineSession init ---


class TestPipelineSessionInit(unittest.TestCase):
    """Post-init creates stages 1-8 and preserves preconditions."""

    def test_stages_1_to_8_initialised(self) -> None:
        session = PipelineSession()
        for s in range(1, 9):
            self.assertIn(s, session.stages)
            self.assertEqual(session.stages[s].status, StageStatus.PENDING)

    def test_preconditions_preserved(self) -> None:
        session = PipelineSession()
        self.assertEqual(session.current_stage, PipelineStage.IDLE)
        self.assertFalse(session.running)
        self.assertFalse(session.cancelled)
        self.assertIsNone(session.pipeline_start_time)


# --------------------------------------------------- PipelineSession reset ---


class TestPipelineSessionReset(unittest.TestCase):
    """reset() clears all mutable state."""

    def setUp(self) -> None:
        self.session = PipelineSession()
        # Dirty every field that reset() should clear
        self.session.current_stage = PipelineStage.DENOISE
        self.session.running = True
        self.session.cancelled = True
        self.session.cancel_requested = True
        self.session.cancel_force = True
        self.session.pipeline_start_time = 1234.0
        self.session.current_checkpoint_id = "ckpt-1"
        self.session.sam2_approved = True
        self.session.sam2_frame_count = 5
        self.session.next_stage_confirmation_required = True
        self.session.next_stage_confirmation_from = 3
        self.session.next_stage_confirmation_to = 4
        self.session.next_stage_confirmation_message = "continue?"
        self.session.mesh_repair_ready = True
        self.session.mesh_repair_candidates = [{"loop_id": 1}]
        self.session.mesh_repair_selected_loop_ids = [1]
        self.session.mesh_repair_source_mesh_path = "/tmp/mesh.ply"
        self.session.mesh_repair_analysis = {"k": "v"}
        self.session.frames_dir = "/tmp/frames"
        self.session.mask_dir = "/tmp/masks"
        self.session.ply_path = "/tmp/object.ply"
        self.session.denoised_ply = "/tmp/dn.ply"
        self.session.mesh_ply = "/tmp/mesh.ply"
        self.session.obj_path = "/tmp/obj.obj"
        self.session.frame_count = 50
        self.session.mask_count = 50
        self.session.stages[1].status = StageStatus.COMPLETE
        self.session.stages[1].progress = 100.0
        self.session.reset()

    def test_runtime_fields_cleared(self) -> None:
        s = self.session
        self.assertEqual(s.current_stage, PipelineStage.IDLE)
        self.assertFalse(s.running)
        self.assertFalse(s.cancelled)
        self.assertFalse(s.cancel_requested)
        self.assertFalse(s.cancel_force)
        self.assertIsNone(s.pipeline_start_time)
        self.assertIsNone(s.current_checkpoint_id)

    def test_all_stages_pending_after_reset(self) -> None:
        for stage_id, info in self.session.stages.items():
            self.assertEqual(info.status, StageStatus.PENDING, f"Stage {stage_id}")
            self.assertAlmostEqual(info.progress, 0.0, msg=f"Stage {stage_id}")

    def test_artifact_paths_cleared(self) -> None:
        s = self.session
        self.assertIsNone(s.frames_dir)
        self.assertIsNone(s.mask_dir)
        self.assertIsNone(s.ply_path)
        self.assertIsNone(s.denoised_ply)
        self.assertIsNone(s.mesh_ply)
        self.assertIsNone(s.obj_path)

    def test_cancel_state_cleared(self) -> None:
        s = self.session
        self.assertFalse(s.cancel_event.is_set())

    def test_mesh_repair_state_cleared(self) -> None:
        s = self.session
        self.assertFalse(s.mesh_repair_ready)
        self.assertEqual(s.mesh_repair_candidates, [])
        self.assertEqual(s.mesh_repair_selected_loop_ids, [])
        self.assertIsNone(s.mesh_repair_source_mesh_path)
        self.assertEqual(s.mesh_repair_analysis, {})

    def test_sam2_state_cleared(self) -> None:
        s = self.session
        self.assertFalse(s.sam2_approved)
        self.assertEqual(s.sam2_frame_count, 0)


# ----------------------------------------- PipelineSession stage lifecycle ---


class TestPipelineSessionStageLifecycle(unittest.TestCase):
    """Stage start / complete / failed / interactive / progress transitions."""

    def setUp(self) -> None:
        self.session = PipelineSession()

    @patch("scripts.dashboard.state.time")
    def test_stage_start_sets_running(self, mock_time: MagicMock) -> None:
        mock_time.time.return_value = 1000.0
        self.session.stage_start(PipelineStage.EXTRACT_FRAMES)
        info = self.session.stages[1]
        self.assertEqual(info.status, StageStatus.RUNNING)
        self.assertEqual(info.start_time, 1000.0)
        self.assertAlmostEqual(info.progress, 0.0)
        self.assertEqual(self.session.current_stage, PipelineStage.EXTRACT_FRAMES)

    @patch("scripts.dashboard.state.time")
    def test_stage_complete_sets_100(self, mock_time: MagicMock) -> None:
        mock_time.time.side_effect = [1000.0, 1005.0]
        self.session.stage_start(PipelineStage.EXTRACT_FRAMES)
        self.session.stage_complete(PipelineStage.EXTRACT_FRAMES)
        info = self.session.stages[1]
        self.assertEqual(info.status, StageStatus.COMPLETE)
        self.assertAlmostEqual(info.progress, 100.0)
        self.assertAlmostEqual(info.elapsed, 5.0)

    @patch("scripts.dashboard.state.time")
    def test_stage_failed_records_error(self, mock_time: MagicMock) -> None:
        mock_time.time.side_effect = [1000.0, 1003.0]
        self.session.stage_start(PipelineStage.PI3X_RECONSTRUCT)
        self.session.stage_failed(PipelineStage.PI3X_RECONSTRUCT, "OOM")
        info = self.session.stages[2]
        self.assertEqual(info.status, StageStatus.FAILED)
        self.assertEqual(info.error, "OOM")
        self.assertAlmostEqual(info.elapsed, 3.0)

    def test_stage_interactive(self) -> None:
        self.session.stage_interactive(PipelineStage.SAM2_SEGMENT)
        info = self.session.stages[3]
        self.assertEqual(info.status, StageStatus.INTERACTIVE)

    def test_progress_clamped_to_0_100(self) -> None:
        self.session.stage_progress(PipelineStage.EXTRACT_FRAMES, progress=-50.0)
        self.assertAlmostEqual(self.session.stages[1].progress, 0.0)
        self.session.stage_progress(PipelineStage.EXTRACT_FRAMES, progress=200.0)
        self.assertAlmostEqual(self.session.stages[1].progress, 100.0)

    def test_progress_checkpoint_update(self) -> None:
        self.session.stage_progress(
            PipelineStage.DENOISE, progress=50.0, checkpoint_id="ckpt-42"
        )
        info = self.session.stages[4]
        self.assertAlmostEqual(info.progress, 50.0)
        self.assertEqual(info.checkpoint_id, "ckpt-42")
        self.assertEqual(self.session.current_checkpoint_id, "ckpt-42")

    def test_progress_none_does_not_overwrite(self) -> None:
        self.session.stage_progress(PipelineStage.EXTRACT_FRAMES, progress=60.0)
        self.session.stage_progress(
            PipelineStage.EXTRACT_FRAMES, progress=None, detail="step 2"
        )
        info = self.session.stages[1]
        self.assertAlmostEqual(info.progress, 60.0)
        self.assertEqual(info.detail, "step 2")


# ----------------------------------------- PipelineSession confirmation ---


class TestPipelineSessionConfirmation(unittest.TestCase):
    """Stage-to-stage confirmation gate."""

    def setUp(self) -> None:
        self.session = PipelineSession()

    def test_require_next_stage_confirmation(self) -> None:
        self.session.require_next_stage_confirmation(
            PipelineStage.DENOISE, PipelineStage.DIFFCD_MESH, "proceed?"
        )
        self.assertTrue(self.session.next_stage_confirmation_required)
        self.assertEqual(self.session.next_stage_confirmation_from, 4)
        self.assertEqual(self.session.next_stage_confirmation_to, 5)
        self.assertEqual(self.session.next_stage_confirmation_message, "proceed?")

    def test_clear_next_stage_confirmation(self) -> None:
        self.session.require_next_stage_confirmation(
            PipelineStage.DENOISE, PipelineStage.DIFFCD_MESH, "proceed?"
        )
        self.session.clear_next_stage_confirmation()
        self.assertFalse(self.session.next_stage_confirmation_required)
        self.assertIsNone(self.session.next_stage_confirmation_from)
        self.assertIsNone(self.session.next_stage_confirmation_to)
        self.assertIsNone(self.session.next_stage_confirmation_message)


# ------------------------------------------------ PipelineSession cancel ---


class TestPipelineSessionCancel(unittest.TestCase):
    """Cancellation request / force / clear."""

    def setUp(self) -> None:
        self.session = PipelineSession()

    def test_request_cancel_normal(self) -> None:
        self.session.request_cancel(force=False)
        self.assertTrue(self.session.cancelled)
        self.assertTrue(self.session.cancel_requested)
        self.assertFalse(self.session.cancel_force)
        self.assertTrue(self.session.cancel_event.is_set())

    def test_request_cancel_force(self) -> None:
        self.session.request_cancel(force=True)
        self.assertTrue(self.session.cancel_force)

    def test_clear_cancel(self) -> None:
        self.session.request_cancel(force=True)
        self.session.clear_cancel()
        self.assertFalse(self.session.cancelled)
        self.assertFalse(self.session.cancel_requested)
        self.assertFalse(self.session.cancel_force)
        self.assertFalse(self.session.cancel_event.is_set())


# ---------------------------------------- PipelineSession process mgmt ---


class TestPipelineSessionProcessManagement(unittest.TestCase):
    """Active process register / unregister / terminate."""

    def setUp(self) -> None:
        self.session = PipelineSession()

    def test_register_active_process(self) -> None:
        proc = _FakeProcess()
        self.session.register_active_process(proc)
        self.assertIn(proc, self.session._active_processes)

    def test_unregister_active_process(self) -> None:
        proc = _FakeProcess()
        self.session.register_active_process(proc)
        self.session.unregister_active_process(proc)
        self.assertNotIn(proc, self.session._active_processes)

    @patch("scripts.dashboard.state._terminate_process_tree", return_value=True)
    def test_terminate_running_process(self, mock_term: MagicMock) -> None:
        proc = _FakeProcess(poll_return=None)  # still running
        self.session.register_active_process(proc)
        count = self.session.terminate_active_processes(grace_seconds=0.5)
        self.assertEqual(count, 1)
        mock_term.assert_called_once_with(proc, grace_seconds=0.5)

    @patch("scripts.dashboard.state._terminate_process_tree", return_value=True)
    def test_terminate_already_exited_skipped(self, mock_term: MagicMock) -> None:
        proc = _FakeProcess(poll_return=0)  # already exited
        self.session.register_active_process(proc)
        count = self.session.terminate_active_processes()
        self.assertEqual(count, 0)
        mock_term.assert_not_called()


# ---------------------------------------- PipelineSession overall progress ---


class TestPipelineSessionOverallProgress(unittest.TestCase):
    """Average progress across 8 stages."""

    def setUp(self) -> None:
        self.session = PipelineSession()

    def test_all_zero(self) -> None:
        self.assertAlmostEqual(self.session.overall_progress(), 0.0)

    def test_all_complete(self) -> None:
        for s in range(1, 9):
            self.session.stages[s].progress = 100.0
        self.assertAlmostEqual(self.session.overall_progress(), 100.0)

    def test_mixed_progress(self) -> None:
        # Stages 1-4 at 100%, 5-8 at 0% => 400/8 = 50.0
        for s in range(1, 5):
            self.session.stages[s].progress = 100.0
        self.assertAlmostEqual(self.session.overall_progress(), 50.0)


# ---------------------------------------- PipelineSession status dict ---


class TestPipelineSessionStatusDict(unittest.TestCase):
    """to_status_dict() serialisation contract."""

    def setUp(self) -> None:
        self.session = PipelineSession()

    def test_required_top_level_keys(self) -> None:
        d = self.session.to_status_dict()
        required = {
            "current_stage",
            "running",
            "cancelled",
            "cancel_requested",
            "cancel_force",
            "resume_from_stage",
            "object_name",
            "mesh_method",
            "video_path",
            "output_dir",
            "current_checkpoint_id",
            "frame_count",
            "mask_count",
            "elapsed",
            "next_stage_confirmation",
            "mesh_repair",
            "stages",
            "auto_accept",
            "overall_progress",
        }
        self.assertTrue(required.issubset(set(d.keys())), f"Missing keys: {required - set(d.keys())}")

    def test_stages_keys_are_strings(self) -> None:
        d = self.session.to_status_dict()
        for key in d["stages"]:
            self.assertIsInstance(key, str)

    @patch("scripts.dashboard.state.time")
    def test_elapsed_while_running(self, mock_time: MagicMock) -> None:
        mock_time.time.return_value = 2000.0
        self.session.pipeline_start_time = 1990.0
        d = self.session.to_status_dict()
        self.assertAlmostEqual(d["elapsed"], 10.0)

    def test_elapsed_none_when_idle(self) -> None:
        self.session.pipeline_start_time = None
        d = self.session.to_status_dict()
        self.assertIsNone(d["elapsed"])


# ---------------------------------------- detect_stage_outputs ---


class TestDetectStageOutputs(unittest.TestCase):
    """detect_stage_outputs() file-system scanning."""

    def test_empty_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            stages, fc, mc = detect_stage_outputs(tmp)
            for stage_id in range(1, 9):
                self.assertFalse(stages[stage_id], f"Stage {stage_id} should be False")
            self.assertEqual(fc, 0)
            self.assertEqual(mc, 0)

    def test_stage1_detection_frames_only(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "frames").mkdir()
            (out / "frames" / "00000.jpg").write_bytes(b"\xff\xd8")
            (out / "frames" / "00001.jpg").write_bytes(b"\xff\xd8")
            stages, fc, mc = detect_stage_outputs(out)
            self.assertTrue(stages[1])
            self.assertEqual(fc, 2)
            self.assertEqual(mc, 0)

    def test_stage3_requires_object_ply_and_masks(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # object.ply alone is not enough
            (out / "object.ply").write_text("ply\n", encoding="utf-8")
            stages, _, mc = detect_stage_outputs(out)
            self.assertFalse(stages[3], "stage 3 needs masks too")
            self.assertEqual(mc, 0)

            # add masks
            (out / "masks").mkdir()
            (out / "masks" / "00000.png").write_bytes(b"\x89PNG")
            stages, _, mc = detect_stage_outputs(out)
            self.assertTrue(stages[3])
            self.assertEqual(mc, 1)

    def test_fixtures_all_stages_complete(self) -> None:
        stages, fc, mc = detect_stage_outputs(FIXTURE_DIR)
        for stage_id in range(1, 9):
            self.assertTrue(stages[stage_id], f"Fixture stage {stage_id} not detected")
        self.assertGreater(fc, 0)
        self.assertGreater(mc, 0)


# ---------------------------------------- hydrate_from_output_dir ---


class TestHydrateFromOutputDir(unittest.TestCase):
    """hydrate_from_output_dir() path setting, mesh priority, current_stage."""

    def _make_stage1_dir(self, base: Path) -> None:
        (base / "frames").mkdir(parents=True, exist_ok=True)
        (base / "frames" / "00000.jpg").write_bytes(b"\xff\xd8")

    def _make_stage2_files(self, base: Path) -> None:
        for name in STAGE_OUTPUT_FILES[2]:
            (base / name).write_text("data", encoding="utf-8")

    def _make_stage3_files(self, base: Path) -> None:
        (base / "object.ply").write_text("ply\n", encoding="utf-8")
        (base / "masks").mkdir(parents=True, exist_ok=True)
        (base / "masks" / "00000.png").write_bytes(b"\x89PNG")

    def test_path_setting_frames_dir(self) -> None:
        session = PipelineSession()
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._make_stage1_dir(out)
            session.hydrate_from_output_dir(out)
            self.assertEqual(session.frames_dir, str(out / "frames"))
            self.assertGreater(session.frame_count, 0)

    def test_path_setting_no_frames(self) -> None:
        session = PipelineSession()
        with TemporaryDirectory() as tmp:
            session.hydrate_from_output_dir(tmp)
            self.assertIsNone(session.frames_dir)
            self.assertEqual(session.frame_count, 0)

    def test_mesh_priority_repaired_over_wrapped(self) -> None:
        session = PipelineSession()
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh.ply").write_text("ply\n", encoding="utf-8")
            (out / "object_mesh_wrapped.ply").write_text("ply\n", encoding="utf-8")
            (out / "object_mesh_repaired.ply").write_text("ply\n", encoding="utf-8")
            session.hydrate_from_output_dir(out)
            self.assertEqual(session.mesh_ply, str(out / "object_mesh_repaired.ply"))

    def test_mesh_priority_wrapped_over_base(self) -> None:
        session = PipelineSession()
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh.ply").write_text("ply\n", encoding="utf-8")
            (out / "object_mesh_wrapped.ply").write_text("ply\n", encoding="utf-8")
            session.hydrate_from_output_dir(out)
            self.assertEqual(session.mesh_ply, str(out / "object_mesh_wrapped.ply"))

    def test_mesh_priority_base_only(self) -> None:
        session = PipelineSession()
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh.ply").write_text("ply\n", encoding="utf-8")
            session.hydrate_from_output_dir(out)
            self.assertEqual(session.mesh_ply, str(out / "object_mesh.ply"))

    def test_current_stage_inferred_from_completed(self) -> None:
        session = PipelineSession()
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._make_stage1_dir(out)
            self._make_stage2_files(out)
            self._make_stage3_files(out)
            session.hydrate_from_output_dir(out)
            # Stages 1-3 complete -> current_stage = 3 (max completed)
            self.assertEqual(session.current_stage, PipelineStage.SAM2_SEGMENT)

    def test_return_value_contains_expected_keys(self) -> None:
        session = PipelineSession()
        with TemporaryDirectory() as tmp:
            result = session.hydrate_from_output_dir(tmp)
            self.assertIn("stage_complete", result)
            self.assertIn("frame_count", result)
            self.assertIn("mask_count", result)


if __name__ == "__main__":
    unittest.main()
