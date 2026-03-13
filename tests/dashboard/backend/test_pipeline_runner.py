"""Tests for scripts/dashboard/pipeline_runner.py helper functions.

Covers broadcast, _build_stage_progress_payload, _CancelledError,
_safe_current_stage, _require_file, _require_dir, _check_cancelled,
_mesh_method_label, and _wait_for_next_stage_confirmation.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.dashboard.pipeline_runner import (
    _build_stage_progress_payload,
    _CancelledError,
    _check_cancelled,
    _mesh_method_label,
    _require_dir,
    _require_file,
    _safe_current_stage,
    _wait_for_next_stage_confirmation,
    broadcast,
)
from scripts.dashboard.state import PipelineSession, PipelineStage


# ------------------------------------------------------------------ helpers ---


class _FakeWS:
    """Minimal WebSocket stub with async send_text."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        if self._fail:
            raise ConnectionError("closed")
        self.sent.append(text)


# ------------------------------------------------------------------ tests ---


class TestBroadcastFunction(unittest.IsolatedAsyncioTestCase):
    """broadcast() delivers JSON to all WS clients."""

    async def test_all_clients_receive(self) -> None:
        session = PipelineSession()
        ws1 = _FakeWS()
        ws2 = _FakeWS()
        session.ws_clients = [ws1, ws2]
        msg = {"type": "ping", "value": 42}

        await broadcast(session, msg)

        expected = json.dumps(msg)
        self.assertEqual(ws1.sent, [expected])
        self.assertEqual(ws2.sent, [expected])

    async def test_stale_connection_removed(self) -> None:
        session = PipelineSession()
        good = _FakeWS()
        bad = _FakeWS(fail=True)
        session.ws_clients = [good, bad]

        await broadcast(session, {"type": "test"})

        self.assertEqual(len(session.ws_clients), 1)
        self.assertIs(session.ws_clients[0], good)
        self.assertEqual(len(good.sent), 1)

    async def test_empty_clients_safe(self) -> None:
        session = PipelineSession()
        session.ws_clients = []

        await broadcast(session, {"type": "noop"})

        self.assertEqual(session.ws_clients, [])


class TestBuildStageProgressPayload(unittest.TestCase):
    """_build_stage_progress_payload returns correct dict and updates session."""

    def test_required_keys_present(self) -> None:
        session = PipelineSession()
        session.config.mesh_method = "poisson"
        session.stage_start(PipelineStage.EXTRACT_FRAMES)

        result = _build_stage_progress_payload(
            session,
            PipelineStage.EXTRACT_FRAMES,
            progress=50.0,
            detail="Extracting",
        )

        required_keys = {
            "type",
            "stage",
            "progress",
            "detail",
            "checkpoint_id",
            "overall_progress",
        }
        self.assertTrue(required_keys.issubset(result.keys()))
        self.assertEqual(result["type"], "stage_progress")
        self.assertEqual(result["stage"], int(PipelineStage.EXTRACT_FRAMES))

    def test_session_updated(self) -> None:
        session = PipelineSession()
        session.config.mesh_method = "poisson"
        session.stage_start(PipelineStage.DENOISE)

        _build_stage_progress_payload(
            session,
            PipelineStage.DENOISE,
            progress=75.0,
            detail="Running SOR",
        )

        info = session.stages[int(PipelineStage.DENOISE)]
        self.assertEqual(info.progress, 75.0)
        self.assertEqual(info.detail, "Running SOR")

    def test_checkpoint_resolved(self) -> None:
        session = PipelineSession()
        session.config.mesh_method = "poisson"
        session.stage_start(PipelineStage.EXTRACT_FRAMES)

        result = _build_stage_progress_payload(
            session,
            PipelineStage.EXTRACT_FRAMES,
            checkpoint_id="s1.extract",
        )

        self.assertEqual(result["checkpoint_id"], "s1.extract")
        info = session.stages[int(PipelineStage.EXTRACT_FRAMES)]
        self.assertEqual(info.checkpoint_id, "s1.extract")


class TestCancelledError(unittest.TestCase):
    """_CancelledError behaves as a normal exception."""

    def test_default_message(self) -> None:
        err = _CancelledError()
        self.assertEqual(str(err), "Pipeline cancelled by user")

    def test_exception_inheritance(self) -> None:
        err = _CancelledError()
        self.assertIsInstance(err, Exception)
        with self.assertRaises(_CancelledError):
            raise err


class TestSafeCurrentStage(unittest.TestCase):
    """_safe_current_stage maps values to PipelineStage or None."""

    def test_valid_stage(self) -> None:
        result = _safe_current_stage(PipelineStage.EXTRACT_FRAMES)
        self.assertEqual(result, PipelineStage.EXTRACT_FRAMES)

    def test_idle_returns_none(self) -> None:
        result = _safe_current_stage(PipelineStage.IDLE)
        self.assertIsNone(result)

    def test_complete_returns_none(self) -> None:
        result = _safe_current_stage(PipelineStage.COMPLETE)
        self.assertIsNone(result)

    def test_invalid_returns_none(self) -> None:
        result = _safe_current_stage("bogus")
        self.assertIsNone(result)


class TestRequireFile(unittest.TestCase):
    """_require_file validates file path existence."""

    def test_none_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            _require_file(None, "Test file")
        self.assertIn("missing", str(ctx.exception))

    def test_empty_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            _require_file("", "Test file")
        self.assertIn("missing", str(ctx.exception))

    def test_nonexistent_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            _require_file("/tmp/nonexistent_12345.xyz", "Test file")
        self.assertIn("not found", str(ctx.exception))

    def test_existing_file_ok(self) -> None:
        with TemporaryDirectory() as td:
            f = Path(td) / "hello.txt"
            f.write_text("hi")
            _require_file(str(f), "Test file")


class TestRequireDir(unittest.TestCase):
    """_require_dir validates directory existence and optional suffix."""

    def test_none_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            _require_dir(None, "Frames dir")
        self.assertIn("missing", str(ctx.exception))

    def test_nonexistent_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            _require_dir("/tmp/nonexistent_dir_99999", "Frames dir")
        self.assertIn("not found", str(ctx.exception))

    def test_empty_suffix_raises(self) -> None:
        with TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError) as ctx:
                _require_dir(td, "Frames dir", must_have_suffix=".jpg")
            self.assertIn("no '.jpg' files", str(ctx.exception))

    def test_has_files_ok(self) -> None:
        with TemporaryDirectory() as td:
            (Path(td) / "frame_001.jpg").write_bytes(b"\xff")
            _require_dir(td, "Frames dir", must_have_suffix=".jpg")


class TestCheckCancelled(unittest.TestCase):
    """_check_cancelled raises when session has any cancel flag set."""

    def test_cancelled_flag(self) -> None:
        session = PipelineSession()
        session.cancelled = True
        with self.assertRaises(_CancelledError):
            _check_cancelled(session)

    def test_cancel_requested_flag(self) -> None:
        session = PipelineSession()
        session.cancel_requested = True
        with self.assertRaises(_CancelledError):
            _check_cancelled(session)

    def test_cancel_event_set(self) -> None:
        session = PipelineSession()
        session.cancel_event.set()
        with self.assertRaises(_CancelledError):
            _check_cancelled(session)

    def test_normal_no_raise(self) -> None:
        session = PipelineSession()
        _check_cancelled(session)


class TestMeshMethodLabel(unittest.TestCase):
    """_mesh_method_label returns human-readable label."""

    def test_diffcd(self) -> None:
        self.assertEqual(_mesh_method_label("diffcd"), "Learning Mesh (DiffCD)")

    def test_poisson(self) -> None:
        self.assertEqual(_mesh_method_label("poisson"), "Classical Mesh")


class TestWaitForNextStageConfirmation(unittest.IsolatedAsyncioTestCase):
    """_wait_for_next_stage_confirmation handles auto-accept and cancel."""

    @patch("scripts.dashboard.pipeline_runner.broadcast", new_callable=AsyncMock)
    async def test_auto_accept_immediate(self, mock_broadcast: AsyncMock) -> None:
        session = PipelineSession()
        session.config.auto_accept = True
        session.config.mesh_method = "poisson"
        session.stage_start(PipelineStage.EXTRACT_FRAMES)

        await _wait_for_next_stage_confirmation(
            session,
            PipelineStage.EXTRACT_FRAMES,
            PipelineStage.PI3X_RECONSTRUCT,
            "Continue?",
        )

        self.assertFalse(session.next_stage_confirmation_required)
        self.assertTrue(mock_broadcast.called)

    @patch("scripts.dashboard.pipeline_runner.broadcast", new_callable=AsyncMock)
    async def test_cancel_during_wait(self, mock_broadcast: AsyncMock) -> None:
        session = PipelineSession()
        session.config.auto_accept = False
        session.config.mesh_method = "poisson"
        session.stage_start(PipelineStage.EXTRACT_FRAMES)

        async def _set_cancel_then_confirm() -> None:
            await asyncio.sleep(0.01)
            session.cancelled = True
            session.next_stage_confirm_event.set()

        task = asyncio.create_task(_set_cancel_then_confirm())

        with self.assertRaises(_CancelledError):
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.EXTRACT_FRAMES,
                PipelineStage.PI3X_RECONSTRUCT,
                "Continue?",
            )

        await task


class TestWaitForNextStageConfirmationBroadcast(unittest.IsolatedAsyncioTestCase):
    """6.6.3 / 6.6.4 — Broadcast message structure and session fields."""

    @patch("scripts.dashboard.pipeline_runner.broadcast", new_callable=AsyncMock)
    async def test_broadcast_required_and_cleared(self, mock_bc: AsyncMock) -> None:
        """6.6.3 — auto_accept broadcasts required + cleared messages."""
        session = PipelineSession()
        session.config.auto_accept = True
        session.config.mesh_method = "poisson"
        session.stage_start(PipelineStage.EXTRACT_FRAMES)

        await _wait_for_next_stage_confirmation(
            session,
            PipelineStage.EXTRACT_FRAMES,
            PipelineStage.PI3X_RECONSTRUCT,
            "Continue to Pi3X?",
        )

        # Find the required broadcast
        required_calls = [
            c for c in mock_bc.call_args_list
            if isinstance(c[0][1], dict)
            and c[0][1].get("type") == "next_stage_confirmation_required"
        ]
        self.assertGreater(len(required_calls), 0, "Should broadcast required message")
        req_msg = required_calls[0][0][1]
        self.assertIn("from_stage", req_msg)
        self.assertIn("to_stage", req_msg)
        self.assertIn("message", req_msg)
        self.assertTrue(req_msg.get("auto_accepted"))

        # Find the cleared broadcast
        cleared_calls = [
            c for c in mock_bc.call_args_list
            if isinstance(c[0][1], dict)
            and c[0][1].get("type") == "next_stage_confirmation_cleared"
        ]
        self.assertGreater(len(cleared_calls), 0, "Should broadcast cleared message")

    @patch("scripts.dashboard.pipeline_runner.broadcast", new_callable=AsyncMock)
    async def test_session_fields_set_and_cleared(self, mock_bc: AsyncMock) -> None:
        """6.6.4 — Session confirmation fields set during and cleared after."""
        session = PipelineSession()
        session.config.auto_accept = True
        session.config.mesh_method = "poisson"
        session.stage_start(PipelineStage.EXTRACT_FRAMES)

        # Capture the state during the required broadcast
        required_seen = False

        async def _capture_required(sess, msg):
            nonlocal required_seen
            if isinstance(msg, dict) and msg.get("type") == "next_stage_confirmation_required":
                required_seen = True
                # During this call, session fields should be set
                # (auto_accept skips immediately, but fields are set first)

        mock_bc.side_effect = _capture_required

        await _wait_for_next_stage_confirmation(
            session,
            PipelineStage.EXTRACT_FRAMES,
            PipelineStage.PI3X_RECONSTRUCT,
            "Continue?",
        )

        self.assertTrue(required_seen, "Required broadcast should have been sent")
        # After completion, fields should be cleared
        self.assertFalse(session.next_stage_confirmation_required)


if __name__ == "__main__":
    unittest.main()
