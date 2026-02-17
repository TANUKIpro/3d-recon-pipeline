"""Tests for scripts/dashboard/app.py helper functions and REST API endpoints."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.dashboard.object_store import (
    _safe_json_load,
    _sanitize_object_name,
    _suggest_object_name,
    _validate_object_name,
)
from scripts.dashboard.app import (
    mesh_repair_candidates,
    pipeline_cancel,
    pipeline_confirm_next,
    pipeline_status,
    sam2_approve,
    sam2_click,
    sam2_confirm,
    sam2_redo,
    sam2_service,
    session,
)


# ── Helpers ────────────────────────────────────────────────────────


def _json_payload(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


# ── _sanitize_object_name ─────────────────────────────────────────


class TestSanitizeObjectName(unittest.TestCase):
    def test_normal_name_unchanged(self) -> None:
        self.assertEqual(_sanitize_object_name("coffee-mug"), "coffee-mug")

    def test_spaces_become_hyphens(self) -> None:
        self.assertEqual(_sanitize_object_name("my cool object"), "my-cool-object")

    def test_slashes_become_hyphens(self) -> None:
        self.assertEqual(_sanitize_object_name("a/b/c"), "a-b-c")

    def test_special_chars_replaced(self) -> None:
        result = _sanitize_object_name("hello@world#2024!")
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)
        self.assertNotIn("!", result)
        # Should still contain the word parts
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("2024", result)

    def test_consecutive_hyphens_collapsed(self) -> None:
        result = _sanitize_object_name("a---b----c")
        self.assertNotIn("--", result)
        self.assertEqual(result, "a-b-c")

    def test_empty_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            _sanitize_object_name("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            _sanitize_object_name("   ")

    def test_truncation_at_80_chars(self) -> None:
        long_name = "a" * 120
        result = _sanitize_object_name(long_name)
        self.assertEqual(len(result), 80)

    def test_leading_trailing_dots_stripped(self) -> None:
        result = _sanitize_object_name("..hello..")
        self.assertFalse(result.startswith("."))
        self.assertFalse(result.endswith("."))
        self.assertIn("hello", result)


# ── _validate_object_name ─────────────────────────────────────────


class TestValidateObjectName(unittest.TestCase):
    def test_normal_name_passes(self) -> None:
        self.assertEqual(_validate_object_name("coffee-mug"), "coffee-mug")

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_object_name("")

    def test_dot_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_object_name(".")

    def test_dotdot_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_object_name("..")

    def test_slash_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_object_name("a/b")

    def test_backslash_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_object_name("a\\b")


# ── _suggest_object_name ──────────────────────────────────────────


class TestSuggestObjectName(unittest.TestCase):
    def test_stem_extracted_from_video_path(self) -> None:
        result = _suggest_object_name("/data/input/coffee01.mp4")
        self.assertEqual(result, "coffee01")

    def test_empty_path_returns_object(self) -> None:
        result = _suggest_object_name("")
        self.assertEqual(result, "object")

    def test_special_chars_sanitized(self) -> None:
        result = _suggest_object_name("/data/input/hello world!.mp4")
        self.assertNotIn(" ", result)
        self.assertNotIn("!", result)
        self.assertTrue(len(result) > 0)


# ── _safe_json_load ───────────────────────────────────────────────


class TestSafeJsonLoad(unittest.TestCase):
    def test_valid_json_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.json"
            p.write_text('{"key": "value"}', encoding="utf-8")
            result = _safe_json_load(p)
            self.assertEqual(result, {"key": "value"})

    def test_nonexistent_returns_empty_dict(self) -> None:
        result = _safe_json_load(Path("/nonexistent/path/data.json"))
        self.assertEqual(result, {})

    def test_broken_json_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad.json"
            p.write_text("{not valid json", encoding="utf-8")
            result = _safe_json_load(p)
            self.assertEqual(result, {})

    def test_non_dict_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "list.json"
            p.write_text("[1, 2, 3]", encoding="utf-8")
            result = _safe_json_load(p)
            self.assertEqual(result, {})


# ── Pipeline status endpoint ──────────────────────────────────────


class TestPipelineStatusEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_returns_status_dict_format(self) -> None:
        response = await pipeline_status()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        # to_status_dict() must contain these keys
        self.assertIn("running", payload)
        self.assertIn("current_stage", payload)
        self.assertIn("stages", payload)
        self.assertIn("cancelled", payload)


# ── Pipeline cancel endpoint ──────────────────────────────────────


class TestPipelineCancelEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "running": session.running,
            "cancelled": session.cancelled,
            "cancel_requested": session.cancel_requested,
            "cancel_force": session.cancel_force,
            "current_stage": session.current_stage,
            "current_checkpoint_id": session.current_checkpoint_id,
            "sam2_confirm_event": session.sam2_confirm_event,
            "sam2_approve_event": session.sam2_approve_event,
            "next_stage_confirm_event": session.next_stage_confirm_event,
            "mesh_repair_confirm_event": session.mesh_repair_confirm_event,
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.cancelled = self._snapshot["cancelled"]
        session.cancel_requested = self._snapshot["cancel_requested"]
        session.cancel_force = self._snapshot["cancel_force"]
        session.current_stage = self._snapshot["current_stage"]
        session.current_checkpoint_id = self._snapshot["current_checkpoint_id"]
        session.sam2_confirm_event = self._snapshot["sam2_confirm_event"]
        session.sam2_approve_event = self._snapshot["sam2_approve_event"]
        session.next_stage_confirm_event = self._snapshot["next_stage_confirm_event"]
        session.mesh_repair_confirm_event = self._snapshot["mesh_repair_confirm_event"]

    async def test_not_running_returns_409(self) -> None:
        session.running = False
        response = await pipeline_cancel()
        self.assertEqual(response.status_code, 409)
        payload = _json_payload(response)
        self.assertIn("error", payload)

    @patch("scripts.dashboard.app.broadcast", new_callable=AsyncMock)
    @patch("asyncio.to_thread", new_callable=AsyncMock, return_value=0)
    async def test_running_sets_cancel_flags(self, mock_to_thread, mock_broadcast) -> None:
        session.running = True
        session.cancelled = False
        session.cancel_force = False
        session.current_stage = 1  # type: ignore[assignment]
        session.current_checkpoint_id = None
        session.sam2_confirm_event = asyncio.Event()
        session.sam2_approve_event = asyncio.Event()
        session.next_stage_confirm_event = asyncio.Event()
        session.mesh_repair_confirm_event = asyncio.Event()

        response = await pipeline_cancel()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "cancelling")
        # Cancel flags should be set via request_cancel(force=True)
        self.assertTrue(session.cancelled)
        self.assertTrue(session.cancel_force)
        # Blocking events should be unblocked
        self.assertTrue(session.sam2_confirm_event.is_set())
        self.assertTrue(session.sam2_approve_event.is_set())
        self.assertTrue(session.next_stage_confirm_event.is_set())
        self.assertTrue(session.mesh_repair_confirm_event.is_set())


# ── Pipeline confirm-next endpoint ────────────────────────────────


class TestConfirmNextEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "running": session.running,
            "next_stage_confirmation_required": session.next_stage_confirmation_required,
            "next_stage_confirm_event": session.next_stage_confirm_event,
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.next_stage_confirmation_required = self._snapshot[
            "next_stage_confirmation_required"
        ]
        session.next_stage_confirm_event = self._snapshot["next_stage_confirm_event"]

    async def test_not_running_returns_409(self) -> None:
        session.running = False
        response = await pipeline_confirm_next()
        self.assertEqual(response.status_code, 409)
        payload = _json_payload(response)
        self.assertIn("error", payload)

    async def test_no_waiting_confirmation(self) -> None:
        session.running = True
        session.next_stage_confirmation_required = False
        response = await pipeline_confirm_next()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "no_waiting_confirmation")

    async def test_confirm_sets_event(self) -> None:
        session.running = True
        session.next_stage_confirmation_required = True
        session.next_stage_confirm_event = asyncio.Event()
        response = await pipeline_confirm_next()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "confirmed")
        self.assertTrue(session.next_stage_confirm_event.is_set())


# ── SAM2 click endpoint ──────────────────────────────────────────


class TestSam2ClickEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session

    async def test_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_click({"x": 0.5, "y": 0.5, "label": 1})
        self.assertEqual(response.status_code, 409)
        payload = _json_payload(response)
        self.assertIn("SAM2 not ready", payload["error"])


# ── SAM2 confirm endpoint ─────────────────────────────────────────


class TestSam2ConfirmEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session
        self._orig_event = session.sam2_confirm_event

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session
        session.sam2_confirm_event = self._orig_event

    async def test_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_confirm()
        self.assertEqual(response.status_code, 409)
        payload = _json_payload(response)
        self.assertIn("SAM2 not ready", payload["error"])

    async def test_confirm_sets_event(self) -> None:
        sam2_service._session = object()  # truthy sentinel
        session.sam2_confirm_event = asyncio.Event()
        response = await sam2_confirm()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "confirming")
        self.assertTrue(session.sam2_confirm_event.is_set())


# ── SAM2 approve endpoint ─────────────────────────────────────────


class TestSam2ApproveEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_approved = session.sam2_approved
        self._orig_event = session.sam2_approve_event

    def tearDown(self) -> None:
        session.sam2_approved = self._orig_approved
        session.sam2_approve_event = self._orig_event

    async def test_approve_sets_flag_and_event(self) -> None:
        session.sam2_approved = False
        session.sam2_approve_event = asyncio.Event()
        response = await sam2_approve()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "approved")
        self.assertTrue(session.sam2_approved)
        self.assertTrue(session.sam2_approve_event.is_set())


# ── SAM2 redo endpoint ────────────────────────────────────────────


class TestSam2RedoEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_approved = session.sam2_approved
        self._orig_event = session.sam2_approve_event

    def tearDown(self) -> None:
        session.sam2_approved = self._orig_approved
        session.sam2_approve_event = self._orig_event

    async def test_redo_clears_flag_and_sets_event(self) -> None:
        session.sam2_approved = True
        session.sam2_approve_event = asyncio.Event()
        response = await sam2_redo()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "redo")
        self.assertFalse(session.sam2_approved)
        self.assertTrue(session.sam2_approve_event.is_set())


# ── Mesh repair candidates endpoint ───────────────────────────────


class TestMeshRepairCandidatesEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "running": session.running,
            "mesh_repair_ready": session.mesh_repair_ready,
            "mesh_repair_candidates": list(session.mesh_repair_candidates),
            "mesh_repair_source_mesh_path": session.mesh_repair_source_mesh_path,
            "mesh_repair_analysis": dict(session.mesh_repair_analysis),
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.mesh_repair_ready = self._snapshot["mesh_repair_ready"]
        session.mesh_repair_candidates = self._snapshot["mesh_repair_candidates"]
        session.mesh_repair_source_mesh_path = self._snapshot[
            "mesh_repair_source_mesh_path"
        ]
        session.mesh_repair_analysis = self._snapshot["mesh_repair_analysis"]

    async def test_not_running_returns_409(self) -> None:
        session.running = False
        response = await mesh_repair_candidates()
        self.assertEqual(response.status_code, 409)
        payload = _json_payload(response)
        self.assertIn("not running", payload["error"])

    async def test_not_ready_returns_409(self) -> None:
        session.running = True
        session.mesh_repair_ready = False
        response = await mesh_repair_candidates()
        self.assertEqual(response.status_code, 409)
        payload = _json_payload(response)
        self.assertIn("not ready", payload["error"])

    async def test_ready_returns_candidates(self) -> None:
        session.running = True
        session.mesh_repair_ready = True
        session.mesh_repair_candidates = [
            {"loop_id": 1, "points": [[0.0, 0.0, 0.0]]},
            {"loop_id": 2, "points": [[0.1, 0.0, 0.0]]},
        ]
        session.mesh_repair_source_mesh_path = None
        session.mesh_repair_analysis = {"total_loops": 2}

        response = await mesh_repair_candidates()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["loop_count"], 2)
        self.assertEqual(len(payload["loops"]), 2)
        self.assertEqual(payload["analysis"], {"total_loops": 2})
        # color_scheme should be present
        self.assertIn("color_scheme", payload)
        self.assertIn("candidate", payload["color_scheme"])
        self.assertIn("selected", payload["color_scheme"])
        self.assertIn("confirmed", payload["color_scheme"])


if __name__ == "__main__":
    unittest.main()
