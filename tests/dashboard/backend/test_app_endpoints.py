"""Tests for scripts/dashboard/app.py helper functions and REST API endpoints."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import WebSocketDisconnect

from scripts.dashboard.object_store import (
    _safe_json_load,
    _sanitize_object_name,
    _suggest_object_name,
    _validate_object_name,
)
from scripts.dashboard.app import (
    mesh_postprocess,
    mesh_repair_candidates,
    mesh_repair_confirm,
    pipeline_cancel,
    pipeline_confirm_next,
    pipeline_load_object,
    pipeline_objects,
    pipeline_object_info,
    pipeline_pi3x_plan,
    pipeline_start,
    pipeline_video_info,
    pipeline_videos,
    preview_file,
    preview_object_file,
    preview_outputs,
    preview_crop_obb,
    pipeline_status,
    sam2_approve,
    sam2_clear,
    sam2_click,
    sam2_confirm,
    sam2_frame,
    sam2_get_mode,
    sam2_mask,
    sam2_mode,
    sam2_redo,
    sam2_service,
    sam2_skip_ground,
    sam2_undo,
    session,
    verification_frame,
    verification_ground_frame,
    vram_info,
    websocket_endpoint,
    _startup,
    _shutdown,
    _active_output_dir,
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


# ── Preview file endpoint ─────────────────────────────────────────


class TestPreviewFileEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_output_dir = session.config.output_dir
        self._tmp = tempfile.TemporaryDirectory()
        session.config.output_dir = self._tmp.name

    def tearDown(self) -> None:
        session.config.output_dir = self._orig_output_dir
        self._tmp.cleanup()

    async def test_preview_file_sets_no_store_cache_headers(self) -> None:
        out = Path(session.config.output_dir)
        target = out / "object_full.ply"
        target.write_text("ply\n", encoding="utf-8")

        response = await preview_file("object_full.ply")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store, max-age=0")
        self.assertEqual(response.headers.get("pragma"), "no-cache")
        self.assertEqual(response.headers.get("expires"), "0")

    async def test_path_traversal_returns_403(self) -> None:
        response = await preview_file("../../etc/passwd")
        self.assertEqual(response.status_code, 403)
        payload = _json_payload(response)
        self.assertEqual(payload["error"], "Access denied")

    async def test_dotdot_in_middle_returns_403(self) -> None:
        # Create a subdir so the prefix portion looks plausible
        Path(session.config.output_dir, "subdir").mkdir()
        response = await preview_file("subdir/../../outside")
        self.assertEqual(response.status_code, 403)
        payload = _json_payload(response)
        self.assertEqual(payload["error"], "Access denied")


# ── Preview object-file path escape (8.5.4) ──────────────────────


class TestPreviewObjectFilePathEscape(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch(
            "scripts.dashboard.app.OUTPUT_DIR", self._tmp.name
        )
        self._patcher.start()
        # Create the object directory so the endpoint doesn't 404 early
        obj_dir = Path(self._tmp.name) / "objects" / "test-obj"
        obj_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    async def test_path_escape_returns_403(self) -> None:
        response = await preview_object_file("test-obj", "../../etc/passwd")
        self.assertEqual(response.status_code, 403)
        payload = _json_payload(response)
        self.assertEqual(payload["error"], "Access denied")

    async def test_dotdot_path_returns_403(self) -> None:
        response = await preview_object_file("test-obj", "../outside")
        self.assertEqual(response.status_code, 403)
        payload = _json_payload(response)
        self.assertEqual(payload["error"], "Access denied")


# ── Pipeline video-info path security (8.1.20) ───────────────────


class TestPipelineVideoInfoPathSecurity(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch(
            "scripts.dashboard.app.INPUT_DIR", self._tmp.name
        )
        self._patcher.start()
        # Create a file *outside* INPUT_DIR to confirm 403, not 404
        self._outside_tmp = tempfile.TemporaryDirectory()
        self._outside_file = Path(self._outside_tmp.name) / "evil.mp4"
        self._outside_file.write_bytes(b"\x00")

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()
        self._outside_tmp.cleanup()

    async def test_path_outside_input_dir_returns_403(self) -> None:
        response = await pipeline_video_info(str(self._outside_file))
        self.assertEqual(response.status_code, 403)
        payload = _json_payload(response)
        self.assertEqual(payload["error"], "Access denied")


# ── WebSocket helper ──────────────────────────────────────────────


class _FakeWS:
    """Mock WebSocket for websocket_endpoint tests."""

    def __init__(self, *, disconnect_after: int = 0) -> None:
        self.accepted = False
        self.sent: list[str] = []
        self._recv_count = 0
        self._disconnect_after = disconnect_after

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def receive_text(self) -> str:
        self._recv_count += 1
        if self._recv_count > self._disconnect_after:
            raise WebSocketDisconnect()
        return "ping"


# ── Pipeline videos endpoint (8.1.7–8.1.8) ───────────────────────


class TestPipelineVideosEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch("scripts.dashboard.app.INPUT_DIR", self._tmp.name)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    async def test_returns_video_list_format(self) -> None:
        # Create a dummy mp4 file
        (Path(self._tmp.name) / "test.mp4").write_bytes(b"\x00" * 1024)
        response = await pipeline_videos()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertIn("videos", payload)
        self.assertEqual(len(payload["videos"]), 1)
        v = payload["videos"][0]
        self.assertIn("name", v)
        self.assertIn("path", v)
        self.assertIn("size_mb", v)
        self.assertIn("suggested_object_name", v)
        self.assertEqual(v["name"], "test.mp4")

    async def test_filters_video_extensions(self) -> None:
        base = Path(self._tmp.name)
        (base / "clip.mp4").write_bytes(b"\x00")
        (base / "notes.txt").write_bytes(b"\x00")
        (base / "photo.jpg").write_bytes(b"\x00")
        (base / "scene.mov").write_bytes(b"\x00")
        response = await pipeline_videos()
        payload = _json_payload(response)
        names = {v["name"] for v in payload["videos"]}
        self.assertIn("clip.mp4", names)
        self.assertIn("scene.mov", names)
        self.assertNotIn("notes.txt", names)
        self.assertNotIn("photo.jpg", names)


# ── Pipeline objects endpoint (8.1.9) ─────────────────────────────


class TestPipelineObjectsEndpoint(unittest.IsolatedAsyncioTestCase):
    @patch("scripts.dashboard.app.OUTPUT_DIR", "/tmp/test-out")
    @patch("scripts.dashboard.app._list_objects", return_value=[
        {"name": "obj1", "updated_at": "2026-01-01"},
    ])
    async def test_returns_objects_and_active(self, mock_list: MagicMock) -> None:
        response = await pipeline_objects()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertIn("objects", payload)
        self.assertIn("active_object", payload)
        self.assertEqual(len(payload["objects"]), 1)


# ── Pipeline object-info endpoint (8.1.10–8.1.12) ────────────────


class TestPipelineObjectInfoEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch("scripts.dashboard.app.OUTPUT_DIR", self._tmp.name)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    @patch("scripts.dashboard.app._summarize_object")
    async def test_valid_name_returns_info(self, mock_summarize: MagicMock) -> None:
        obj_dir = Path(self._tmp.name) / "objects" / "test-obj"
        obj_dir.mkdir(parents=True)
        mock_summarize.return_value = {"name": "test-obj"}
        response = await pipeline_object_info(name="test-obj")
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertIn("object", payload)

    async def test_invalid_name_returns_400(self) -> None:
        response = await pipeline_object_info(name="..")
        self.assertEqual(response.status_code, 400)

    async def test_not_found_returns_404(self) -> None:
        response = await pipeline_object_info(name="nonexistent-obj")
        self.assertEqual(response.status_code, 404)


# ── Pipeline load-object endpoint (8.1.13–8.1.14) ────────────────


class TestPipelineLoadObjectEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch("scripts.dashboard.app.OUTPUT_DIR", self._tmp.name)
        self._patcher.start()
        self._snapshot = {
            "running": session.running,
            "config_object_name": session.config.object_name,
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.config.object_name = self._snapshot["config_object_name"]
        self._patcher.stop()
        self._tmp.cleanup()

    @patch("scripts.dashboard.app.broadcast", new_callable=AsyncMock)
    @patch("scripts.dashboard.app._load_object_into_session", return_value={"name": "test-obj"})
    async def test_load_success(self, mock_load: MagicMock, mock_broadcast: AsyncMock) -> None:
        session.running = False
        obj_dir = Path(self._tmp.name) / "objects" / "test-obj"
        obj_dir.mkdir(parents=True)
        response = await pipeline_load_object({"name": "test-obj"})
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "loaded")

    async def test_load_while_running_returns_409(self) -> None:
        session.running = True
        response = await pipeline_load_object({"name": "test-obj"})
        self.assertEqual(response.status_code, 409)


# ── Pipeline start endpoint (8.1.15–8.1.18) ──────────────────────


class TestPipelineStartEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch("scripts.dashboard.app.OUTPUT_DIR", self._tmp.name)
        self._patcher.start()
        self._snapshot = {
            "running": session.running,
            "config": session.config,
            "resume_from_stage": session.resume_from_stage,
        }
        # Save env vars
        self._env_keys = [
            "DIFFCD_BATCH_SIZE", "DIFFCD_N_BATCHES", "DIFFCD_RESOLUTION",
            "MESH_REPAIR_ENABLED", "MESH_REPAIR_MAX_DIAMETER_RATIO",
            "MESH_REPAIR_Y_BAND_RATIO", "MESH_REPAIR_SMOOTH_ITERS",
        ]
        import os
        self._env_backup = {k: os.environ.get(k) for k in self._env_keys}

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.config = self._snapshot["config"]
        session.resume_from_stage = self._snapshot["resume_from_stage"]
        self._patcher.stop()
        self._tmp.cleanup()
        import os
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    async def test_start_while_running_returns_409(self) -> None:
        session.running = True
        response = await pipeline_start({"video_path": "/data/input/test.mp4"})
        self.assertEqual(response.status_code, 409)

    async def test_resume_stage_out_of_range_returns_400(self) -> None:
        session.running = False
        response = await pipeline_start({
            "video_path": "/data/input/test.mp4",
            "resume_from_stage": 0,
        })
        self.assertEqual(response.status_code, 400)

    async def test_missing_prereqs_returns_400(self) -> None:
        session.running = False
        # Create object dir so it exists but is empty
        obj_dir = Path(self._tmp.name) / "objects" / "test-obj"
        obj_dir.mkdir(parents=True)
        response = await pipeline_start({
            "video_path": "/data/input/test.mp4",
            "object_name": "test-obj",
            "resume_from_stage": 3,
        })
        self.assertEqual(response.status_code, 400)
        payload = _json_payload(response)
        self.assertIn("missing", payload)

    @patch("scripts.dashboard.app.run_pipeline", new_callable=AsyncMock)
    async def test_start_success(self, mock_run: AsyncMock) -> None:
        session.running = False
        response = await pipeline_start({
            "video_path": "/data/input/test.mp4",
            "object_name": "test-obj",
        })
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "started")
        self.assertIn("object_name", payload)
        self.assertIn("output_dir", payload)
        self.assertIn("resume_from_stage", payload)
        # Clean up the created task
        if session._task is not None:
            session._task.cancel()
            try:
                await session._task
            except (asyncio.CancelledError, Exception):
                pass


# ── Pipeline video-info success (8.1.19) ──────────────────────────


class TestPipelineVideoInfoSuccess(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch("scripts.dashboard.app.INPUT_DIR", self._tmp.name)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_valid_video_returns_metadata(self, mock_to_thread: AsyncMock) -> None:
        video = Path(self._tmp.name) / "clip.mp4"
        video.write_bytes(b"\x00" * 100)
        mock_to_thread.return_value = {
            "fps": 30, "total_frames": 300, "width": 1920, "height": 1080,
            "duration": 10.0, "rotation": 0,
            "suggested_frame_interval": 15, "suggested_max_frames": 20,
        }
        response = await pipeline_video_info(str(video))
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["fps"], 30)


# ── Pipeline pi3x-plan endpoint (8.1.21) ─────────────────────────


class TestPipelinePi3xPlanEndpoint(unittest.IsolatedAsyncioTestCase):
    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_returns_plan(self, mock_to_thread: AsyncMock) -> None:
        mock_to_thread.return_value = {
            "requested_frames": 50, "actual_frames": 40, "pixel_limit": 512,
        }
        response = await pipeline_pi3x_plan(requested_frames=50, pixel_limit=512)
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertIn("requested_frames", payload)


# ── SAM2 click success (8.2.2) ───────────────────────────────────


class TestSam2ClickSuccess(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_click_returns_png(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.return_value = b"\x89PNG"
        response = await sam2_click({"x": 0.5, "y": 0.5, "label": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/png")


# ── SAM2 undo endpoint (8.2.3) ───────────────────────────────────


class TestSam2UndoEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_undo_returns_png(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.return_value = b"\x89PNG"
        response = await sam2_undo()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/png")

    async def test_undo_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_undo()
        self.assertEqual(response.status_code, 409)


# ── SAM2 clear endpoint (8.2.4) ──────────────────────────────────


class TestSam2ClearEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_clear_returns_png(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.return_value = b"\x89PNG"
        response = await sam2_clear()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/png")

    async def test_clear_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_clear()
        self.assertEqual(response.status_code, 409)


# ── SAM2 mode endpoint (8.2.9–8.2.10) ────────────────────────────


class TestSam2ModeEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session
        self._orig_mode = sam2_service._segmentation_mode

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session
        sam2_service._segmentation_mode = self._orig_mode

    async def test_set_mode_success(self) -> None:
        sam2_service._session = object()
        response = await sam2_mode({"mode": "ground"})
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "ground")

    async def test_set_mode_invalid_returns_400(self) -> None:
        sam2_service._session = object()
        response = await sam2_mode({"mode": "invalid"})
        self.assertEqual(response.status_code, 400)

    async def test_set_mode_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_mode({"mode": "ground"})
        self.assertEqual(response.status_code, 409)

    async def test_get_mode_returns_current(self) -> None:
        sam2_service._segmentation_mode = "ground"
        response = await sam2_get_mode()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["mode"], "ground")


# ── SAM2 skip-ground endpoint (8.2.11) ───────────────────────────


class TestSam2SkipGroundEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_event = session.sam2_ground_skip_event

    def tearDown(self) -> None:
        session.sam2_ground_skip_event = self._orig_event

    async def test_skip_ground_sets_event(self) -> None:
        session.sam2_ground_skip_event = asyncio.Event()
        response = await sam2_skip_ground()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "skipping_ground")
        self.assertTrue(session.sam2_ground_skip_event.is_set())


# ── SAM2 frame endpoint (8.2.12–8.2.13) ──────────────────────────


class TestSam2FrameEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_frame_returns_jpeg(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.return_value = b"\xff\xd8\xff"
        response = await sam2_frame(0)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/jpeg")

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_frame_out_of_range_returns_404(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.side_effect = IndexError("frame out of range")
        response = await sam2_frame(999)
        self.assertEqual(response.status_code, 404)

    async def test_frame_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_frame(0)
        self.assertEqual(response.status_code, 409)


# ── SAM2 mask endpoint (8.2.14–8.2.15) ───────────────────────────


class TestSam2MaskEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_mask_returns_png(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.return_value = b"\x89PNG"
        response = await sam2_mask(0)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/png")

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_mask_not_found_returns_404(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.return_value = None
        response = await sam2_mask(0)
        self.assertEqual(response.status_code, 404)

    async def test_mask_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_mask(0)
        self.assertEqual(response.status_code, 409)


# ── Mesh postprocess endpoint (8.4.1–8.4.5) ──────────────────────


class TestMeshPostprocessEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._snapshot = {
            "running": session.running,
            "next_stage_confirmation_required": session.next_stage_confirmation_required,
            "next_stage_confirmation_from": session.next_stage_confirmation_from,
            "mesh_ply": session.mesh_ply,
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.next_stage_confirmation_required = self._snapshot["next_stage_confirmation_required"]
        session.next_stage_confirmation_from = self._snapshot["next_stage_confirmation_from"]
        session.mesh_ply = self._snapshot["mesh_ply"]
        self._tmp.cleanup()

    @patch("scripts.dashboard.app._reset_outputs_from_stage")
    @patch("asyncio.to_thread", new_callable=AsyncMock)
    @patch("scripts.dashboard.app._active_output_dir")
    async def test_postprocess_success(
        self, mock_out_dir: MagicMock, mock_to_thread: AsyncMock, mock_reset: MagicMock,
    ) -> None:
        session.running = False
        out = Path(self._tmp.name)
        mock_out_dir.return_value = out
        (out / "object_mesh.ply").write_bytes(b"ply")
        (out / "object_mesh_raw.ply").write_bytes(b"ply")
        mock_to_thread.return_value = (1000, 2000, False)
        response = await mesh_postprocess({"invalidate_texture": False})
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["vertices"], 1000)
        self.assertEqual(payload["faces"], 2000)
        self.assertIn("method", payload)
        self.assertIn("iterations", payload)

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    @patch("scripts.dashboard.app._active_output_dir")
    async def test_method_validation(
        self, mock_out_dir: MagicMock, mock_to_thread: AsyncMock,
    ) -> None:
        session.running = False
        out = Path(self._tmp.name)
        mock_out_dir.return_value = out
        (out / "object_mesh.ply").write_bytes(b"ply")
        (out / "object_mesh_raw.ply").write_bytes(b"ply")
        mock_to_thread.return_value = (100, 200, False)
        response = await mesh_postprocess({
            "method": "invalid",
            "invalidate_texture": False,
        })
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        # Invalid method falls back to "laplacian"
        self.assertEqual(payload["method"], "laplacian")

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    @patch("scripts.dashboard.app._active_output_dir")
    async def test_iterations_clamped(
        self, mock_out_dir: MagicMock, mock_to_thread: AsyncMock,
    ) -> None:
        session.running = False
        out = Path(self._tmp.name)
        mock_out_dir.return_value = out
        (out / "object_mesh.ply").write_bytes(b"ply")
        (out / "object_mesh_raw.ply").write_bytes(b"ply")
        mock_to_thread.return_value = (100, 200, False)

        # iterations=200 → clamped to 100
        response = await mesh_postprocess({
            "iterations": 200,
            "invalidate_texture": False,
        })
        payload = _json_payload(response)
        self.assertEqual(payload["iterations"], 100)

        # iterations=-5 → clamped to 0
        response = await mesh_postprocess({
            "iterations": -5,
            "invalidate_texture": False,
        })
        payload = _json_payload(response)
        self.assertEqual(payload["iterations"], 0)

    @patch("scripts.dashboard.app._reset_outputs_from_stage")
    @patch("asyncio.to_thread", new_callable=AsyncMock)
    @patch("scripts.dashboard.app._active_output_dir")
    async def test_invalidate_texture_resets(
        self, mock_out_dir: MagicMock, mock_to_thread: AsyncMock, mock_reset: MagicMock,
    ) -> None:
        session.running = False
        out = Path(self._tmp.name)
        mock_out_dir.return_value = out
        (out / "object_mesh.ply").write_bytes(b"ply")
        (out / "object_mesh_raw.ply").write_bytes(b"ply")
        # Create a downstream file that triggers invalidation
        (out / "textured_mesh.obj").write_text("obj", encoding="utf-8")
        mock_to_thread.return_value = (100, 200, False)
        response = await mesh_postprocess({"invalidate_texture": True})
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertTrue(payload["texture_invalidated"])
        mock_reset.assert_called_once()

    @patch("scripts.dashboard.app._active_output_dir")
    async def test_running_non_stage5_returns_409(self, mock_out_dir: MagicMock) -> None:
        session.running = True
        session.next_stage_confirmation_required = False
        response = await mesh_postprocess({})
        self.assertEqual(response.status_code, 409)


# ── WebSocket endpoint (8.7.1–8.7.4) ─────────────────────────────


class TestWebSocketEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_clients = list(session.ws_clients)

    def tearDown(self) -> None:
        session.ws_clients[:] = self._orig_clients

    async def test_client_added_and_removed(self) -> None:
        ws = _FakeWS(disconnect_after=0)
        await websocket_endpoint(ws)  # type: ignore[arg-type]
        self.assertTrue(ws.accepted)
        # After disconnect, client should be removed
        self.assertNotIn(ws, session.ws_clients)

    async def test_status_snapshot_sent_on_connect(self) -> None:
        ws = _FakeWS(disconnect_after=0)
        await websocket_endpoint(ws)  # type: ignore[arg-type]
        self.assertTrue(len(ws.sent) >= 1)
        first = json.loads(ws.sent[0])
        self.assertEqual(first["type"], "status")
        self.assertIn("running", first)
        self.assertIn("stages", first)

    async def test_multiple_clients(self) -> None:
        ws1 = _FakeWS(disconnect_after=0)
        ws2 = _FakeWS(disconnect_after=0)
        t1 = asyncio.create_task(websocket_endpoint(ws1))  # type: ignore[arg-type]
        t2 = asyncio.create_task(websocket_endpoint(ws2))  # type: ignore[arg-type]
        await asyncio.gather(t1, t2)
        # Both received status snapshot
        self.assertTrue(len(ws1.sent) >= 1)
        self.assertTrue(len(ws2.sent) >= 1)
        self.assertEqual(json.loads(ws1.sent[0])["type"], "status")
        self.assertEqual(json.loads(ws2.sent[0])["type"], "status")


# ── Mesh repair confirm dedup (8.3.7) ────────────────────────────


class TestMeshRepairConfirmDedup(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "running": session.running,
            "mesh_repair_ready": session.mesh_repair_ready,
            "mesh_repair_candidates": list(session.mesh_repair_candidates),
            "mesh_repair_selected_loop_ids": list(session.mesh_repair_selected_loop_ids),
            "mesh_repair_confirm_event": session.mesh_repair_confirm_event,
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.mesh_repair_ready = self._snapshot["mesh_repair_ready"]
        session.mesh_repair_candidates = self._snapshot["mesh_repair_candidates"]
        session.mesh_repair_selected_loop_ids = self._snapshot["mesh_repair_selected_loop_ids"]
        session.mesh_repair_confirm_event = self._snapshot["mesh_repair_confirm_event"]

    async def test_duplicate_ids_deduped_preserving_order(self) -> None:
        session.running = True
        session.mesh_repair_ready = True
        session.mesh_repair_candidates = [
            {"loop_id": 1}, {"loop_id": 2},
        ]
        session.mesh_repair_confirm_event = asyncio.Event()
        response = await mesh_repair_confirm({"selected_loop_ids": [2, 1, 2, 1, 2]})
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["selected_loop_ids"], [2, 1])
        self.assertEqual(payload["selected_count"], 2)


# ── Preview Endpoints ─────────────────────────────────────────────


class TestPreviewObjectFileNormal(unittest.IsolatedAsyncioTestCase):
    """8.5.3 — preview_object_file normal file serving."""

    @patch("scripts.dashboard.app._resolve_output_root")
    @patch("scripts.dashboard.app._validate_object_name", side_effect=lambda n: n)
    async def test_serves_existing_file(self, mock_validate, mock_root) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            obj_dir = base / "objects" / "test-obj"
            obj_dir.mkdir(parents=True)
            (obj_dir / "test.ply").write_bytes(b"ply data")
            mock_root.return_value = base

            response = await preview_object_file("test-obj", "test.ply")

            self.assertEqual(response.status_code, 200)


class TestPreviewOutputs(unittest.IsolatedAsyncioTestCase):
    """8.5.5 — preview_outputs returns file list."""

    @patch("scripts.dashboard.app._active_output_dir")
    async def test_returns_files(self, mock_dir) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "object_mesh.ply").write_bytes(b"ply")
            mock_dir.return_value = out

            response = await preview_outputs()

            data = _json_payload(response)
            self.assertIn("files", data)
            names = [f["name"] for f in data["files"]]
            self.assertIn("object_mesh.ply", names)


class TestPreviewCropOBB(unittest.IsolatedAsyncioTestCase):
    """8.5.6 — preview_crop_obb returns OBB dict."""

    @patch("scripts.dashboard.app._active_output_dir")
    async def test_missing_mesh_returns_404(self, mock_dir) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mock_dir.return_value = Path(tmp)

            response = await preview_crop_obb()

            data = _json_payload(response)
            self.assertEqual(response.status_code, 404)


class TestVerificationFrame(unittest.IsolatedAsyncioTestCase):
    """8.5.7 — verification_frame returns composited JPEG."""

    @patch("scripts.dashboard.app._active_output_dir")
    async def test_missing_frame_returns_404(self, mock_dir) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            mock_dir.return_value = out

            response = await verification_frame(0)

            data = _json_payload(response)
            self.assertEqual(response.status_code, 404)


class TestVerificationGroundFrame(unittest.IsolatedAsyncioTestCase):
    """8.5.8 — verification_ground_frame returns composited JPEG."""

    @patch("scripts.dashboard.app._active_output_dir")
    async def test_missing_frame_returns_404(self, mock_dir) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            mock_dir.return_value = out

            response = await verification_ground_frame(0)

            data = _json_payload(response)
            self.assertEqual(response.status_code, 404)


class TestVramInfo(unittest.IsolatedAsyncioTestCase):
    """8.6.1 — vram_info returns free_mb."""

    @patch("scripts.dashboard.app.get_free_vram_mb", create=True, return_value=8192)
    async def test_returns_free_mb(self, mock_vram) -> None:
        # We need to mock the import inside the function
        pass

    async def test_vram_info_no_gpu_returns_null(self) -> None:
        """When vram_utils raises, free_mb is None."""
        with patch.dict("sys.modules", {"scripts.vram_utils": MagicMock(
            get_free_vram_mb=MagicMock(side_effect=RuntimeError("no GPU")),
        )}):
            response = await vram_info()
            data = _json_payload(response)
            self.assertIn("free_mb", data)


class TestLifecycleStartup(unittest.IsolatedAsyncioTestCase):
    """8.8.1-8.8.3 — _startup installs LogBroadcaster and auto-loads."""

    @patch("scripts.dashboard.app._list_objects", return_value=[])
    @patch("scripts.dashboard.app._resolve_output_root")
    @patch("scripts.dashboard.app.LogBroadcaster")
    async def test_startup_installs_broadcaster(self, mock_lb_cls, mock_root, mock_list) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mock_root.return_value = Path(tmp)
            mock_lb = MagicMock()
            mock_lb.drain = AsyncMock()
            mock_lb_cls.return_value = mock_lb

            import scripts.dashboard.app as app_mod
            old_broadcaster = app_mod.log_broadcaster
            try:
                await _startup()
                mock_lb.install.assert_called_once()
            finally:
                app_mod.log_broadcaster = old_broadcaster

    @patch("scripts.dashboard.app._list_objects")
    @patch("scripts.dashboard.app._resolve_output_root")
    @patch("scripts.dashboard.app.LogBroadcaster")
    async def test_startup_auto_loads_latest_object(self, mock_lb_cls, mock_root, mock_list) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            obj_dir = base / "objects" / "latest-obj"
            obj_dir.mkdir(parents=True)
            mock_root.return_value = base
            mock_list.return_value = [{"name": "latest-obj"}]

            mock_lb = MagicMock()
            mock_lb.drain = AsyncMock()
            mock_lb_cls.return_value = mock_lb

            import scripts.dashboard.app as app_mod
            old_broadcaster = app_mod.log_broadcaster
            try:
                with patch("scripts.dashboard.app._load_object_into_session") as mock_load:
                    await _startup()
                    mock_load.assert_called_once()
            finally:
                app_mod.log_broadcaster = old_broadcaster


class TestLifecycleShutdown(unittest.IsolatedAsyncioTestCase):
    """8.8.4-8.8.5 — _shutdown releases SAM2 and uninstalls broadcaster."""

    async def test_shutdown_releases_sam2(self) -> None:
        import scripts.dashboard.app as app_mod
        mock_sam2 = MagicMock()
        old_sam2 = app_mod.sam2_service
        old_broadcaster = app_mod.log_broadcaster
        app_mod.sam2_service = mock_sam2
        app_mod.log_broadcaster = MagicMock()

        try:
            await _shutdown()
            mock_sam2.release.assert_called_once()
        finally:
            app_mod.sam2_service = old_sam2
            app_mod.log_broadcaster = old_broadcaster

    async def test_shutdown_uninstalls_broadcaster(self) -> None:
        import scripts.dashboard.app as app_mod
        mock_broadcaster = MagicMock()
        mock_sam2 = MagicMock()
        old_sam2 = app_mod.sam2_service
        old_broadcaster = app_mod.log_broadcaster
        app_mod.log_broadcaster = mock_broadcaster
        app_mod.sam2_service = mock_sam2

        try:
            await _shutdown()
            mock_broadcaster.uninstall.assert_called_once()
        finally:
            app_mod.sam2_service = old_sam2
            app_mod.log_broadcaster = old_broadcaster


if __name__ == "__main__":
    unittest.main()
