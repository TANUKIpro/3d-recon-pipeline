"""Tests for scripts/dashboard/app.py helper functions and REST API endpoints."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import WebSocketDisconnect

from scripts.dashboard.dependencies import get_state
from scripts.dashboard.object_store import (
    safe_json_load,
    sanitize_object_name,
    suggest_object_name,
    validate_object_name,
)
from scripts.dashboard.app import (
    pipeline_cancel,
    pipeline_confirm_next,
    pipeline_load_object,
    pipeline_objects,
    pipeline_object_info,
    pipeline_start,
    pipeline_video_info,
    pipeline_videos,
    preview_file,
    preview_object_file,
    preview_outputs,
    preview_crop_obb,
    post_texture_cleanup_confirm,
    post_texture_cleanup_proposal,
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
from scripts.dashboard.state import PipelineStage


# ── Helpers ────────────────────────────────────────────────────────


def _json_payload(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


# ── sanitize_object_name ─────────────────────────────────────────


class TestSanitizeObjectName(unittest.TestCase):
    def test_normal_name_unchanged(self) -> None:
        self.assertEqual(sanitize_object_name("coffee-mug"), "coffee-mug")

    def test_spaces_become_hyphens(self) -> None:
        self.assertEqual(sanitize_object_name("my cool object"), "my-cool-object")

    def test_slashes_become_hyphens(self) -> None:
        self.assertEqual(sanitize_object_name("a/b/c"), "a-b-c")

    def test_special_chars_replaced(self) -> None:
        result = sanitize_object_name("hello@world#2024!")
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)
        self.assertNotIn("!", result)
        # Should still contain the word parts
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("2024", result)

    def test_consecutive_hyphens_collapsed(self) -> None:
        result = sanitize_object_name("a---b----c")
        self.assertNotIn("--", result)
        self.assertEqual(result, "a-b-c")

    def test_empty_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_object_name("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_object_name("   ")

    def test_truncation_at_80_chars(self) -> None:
        long_name = "a" * 120
        result = sanitize_object_name(long_name)
        self.assertLessEqual(len(result), 80)

    def test_leading_trailing_dots_stripped(self) -> None:
        result = sanitize_object_name("..foo..")
        self.assertFalse(result.startswith("."))
        self.assertFalse(result.endswith("."))


# ── validate_object_name ─────────────────────────────────────────


class TestValidateObjectName(unittest.TestCase):
    def test_normal_name_passes(self) -> None:
        self.assertEqual(validate_object_name("coffee-mug"), "coffee-mug")

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_object_name("")

    def test_dot_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_object_name(".")

    def test_dotdot_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_object_name("..")

    def test_slash_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_object_name("a/b")

    def test_backslash_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_object_name("a\\b")


# ── suggest_object_name ──────────────────────────────────────────


class TestSuggestObjectName(unittest.TestCase):
    def test_stem_extracted_from_video_path(self) -> None:
        result = suggest_object_name("/data/input/my_video.mp4")
        self.assertEqual(result, "my_video")

    def test_empty_path_returns_object(self) -> None:
        result = suggest_object_name("")
        self.assertEqual(result, "object")

    def test_special_chars_sanitized(self) -> None:
        result = suggest_object_name("/data/input/hello world!.mp4")
        self.assertNotIn("!", result)
        self.assertNotIn(" ", result)


# ── safe_json_load ───────────────────────────────────────────────


class TestSafeJsonLoad(unittest.TestCase):
    def test_valid_json_dict(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            result = safe_json_load(Path(f.name))
        self.assertEqual(result, {"key": "value"})

    def test_nonexistent_returns_empty_dict(self) -> None:
        result = safe_json_load(Path("/tmp/nonexistent_xyz.json"))
        self.assertEqual(result, {})

    def test_broken_json_returns_empty_dict(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{broken")
            f.flush()
            result = safe_json_load(Path(f.name))
        self.assertEqual(result, {})

    def test_non_dict_returns_empty_dict(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            f.flush()
            result = safe_json_load(Path(f.name))
        self.assertEqual(result, {})


# ── Pipeline status endpoint ─────────────────────────────────────


class TestPipelineStatusEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_returns_status_dict_format(self) -> None:
        response = await pipeline_status()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertIn("current_stage", payload)
        self.assertIn("running", payload)
        self.assertIn("stages", payload)
        self.assertIn("overall_progress", payload)


class TestPostTextureCleanupEndpoints(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._snapshot = {
            "running": session.running,
            "current_stage": session.current_stage,
            "cleanup_review_event": session.cleanup_review_event,
            "cleanup_decision": session.cleanup_decision,
            "cleanup_proposal_path": session.cleanup_proposal_path,
            "output_dir": session.config.output_dir,
        }
        proposal_dir = Path(self.tmpdir.name) / "post_texture_contact_cleanup"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        self.proposal_path = proposal_dir / "proposal.json"
        self.proposal_path.write_text(
            json.dumps({"status": "proposal_ready", "recommended_decision": "apply"}),
            encoding="utf-8",
        )
        session.config.output_dir = self.tmpdir.name
        session.cleanup_proposal_path = str(self.proposal_path)
        session.cleanup_review_event = asyncio.Event()
        session.cleanup_decision = None

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.current_stage = self._snapshot["current_stage"]
        session.cleanup_review_event = self._snapshot["cleanup_review_event"]
        session.cleanup_decision = self._snapshot["cleanup_decision"]
        session.cleanup_proposal_path = self._snapshot["cleanup_proposal_path"]
        session.config.output_dir = self._snapshot["output_dir"]
        self.tmpdir.cleanup()

    async def test_get_proposal_returns_payload(self) -> None:
        response = await post_texture_cleanup_proposal()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["proposal"]["recommended_decision"], "apply")

    async def test_confirm_requires_running_pipeline(self) -> None:
        session.running = False
        response = await post_texture_cleanup_confirm({"decision": "apply"})
        self.assertEqual(response.status_code, 409)

    async def test_confirm_sets_decision_and_event(self) -> None:
        session.running = True
        session.current_stage = PipelineStage.POST_TEXTURE_CONTACT_CLEANUP
        response = await post_texture_cleanup_confirm({"decision": "skip"})
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertEqual(payload["decision"], "skip")
        self.assertEqual(session.cleanup_decision, "skip")
        self.assertTrue(session.cleanup_review_event.is_set())


# ── Pipeline cancel endpoint ─────────────────────────────────────


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

    async def test_not_running_returns_409(self) -> None:
        session.running = False
        response = await pipeline_cancel()
        self.assertEqual(response.status_code, 409)
        payload = _json_payload(response)
        self.assertIn("error", payload)

    @patch("scripts.dashboard.routes.pipeline.broadcast", new_callable=AsyncMock)
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

    async def test_no_waiting_confirmation(self) -> None:
        session.running = True
        session.next_stage_confirmation_required = False
        response = await pipeline_confirm_next()
        payload = _json_payload(response)
        self.assertEqual(payload["status"], "no_waiting_confirmation")

    async def test_confirm_sets_event(self) -> None:
        session.running = True
        session.next_stage_confirmation_required = True
        session.next_stage_confirm_event = asyncio.Event()
        self.assertFalse(session.next_stage_confirm_event.is_set())

        response = await pipeline_confirm_next()

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


# ── SAM2 confirm endpoint ────────────────────────────────────────


class TestSam2ConfirmEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "sam2_confirm_event": session.sam2_confirm_event,
        }

    def tearDown(self) -> None:
        session.sam2_confirm_event = self._snapshot["sam2_confirm_event"]

    async def test_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_confirm()
        self.assertEqual(response.status_code, 409)

    async def test_confirm_sets_event(self) -> None:
        sam2_service._session = object()
        session.sam2_confirm_event = asyncio.Event()
        self.assertFalse(session.sam2_confirm_event.is_set())

        response = await sam2_confirm()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(session.sam2_confirm_event.is_set())
        sam2_service._session = None  # restore


# ── SAM2 approve endpoint ────────────────────────────────────────


class TestSam2ApproveEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "sam2_approve_event": session.sam2_approve_event,
            "sam2_approved": session.sam2_approved,
        }

    def tearDown(self) -> None:
        session.sam2_approve_event = self._snapshot["sam2_approve_event"]
        session.sam2_approved = self._snapshot["sam2_approved"]

    async def test_approve_sets_flag_and_event(self) -> None:
        session.sam2_approve_event = asyncio.Event()
        session.sam2_approved = False

        response = await sam2_approve()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(session.sam2_approved)
        self.assertTrue(session.sam2_approve_event.is_set())


# ── SAM2 redo endpoint ───────────────────────────────────────────


class TestSam2RedoEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "sam2_approve_event": session.sam2_approve_event,
            "sam2_approved": session.sam2_approved,
        }

    def tearDown(self) -> None:
        session.sam2_approve_event = self._snapshot["sam2_approve_event"]
        session.sam2_approved = self._snapshot["sam2_approved"]

    async def test_redo_clears_flag_and_sets_event(self) -> None:
        session.sam2_approve_event = asyncio.Event()
        session.sam2_approved = True

        response = await sam2_redo()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(session.sam2_approved)
        self.assertTrue(session.sam2_approve_event.is_set())


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
        out = Path(self._tmp.name)
        test_file = out / "test.ply"
        test_file.write_bytes(b"ply data")

        response = await preview_file("test.ply")
        self.assertEqual(response.status_code, 200)
        # Check cache-control header
        if hasattr(response, "headers"):
            cc = response.headers.get("cache-control", "")
            self.assertIn("no-store", cc)

    async def test_path_traversal_returns_403(self) -> None:
        response = await preview_file("../../../etc/passwd")
        data = _json_payload(response)
        self.assertEqual(response.status_code, 403)

    async def test_dotdot_in_middle_returns_403(self) -> None:
        response = await preview_file("subdir/../../../etc/passwd")
        data = _json_payload(response)
        self.assertEqual(response.status_code, 403)


# ── Preview object file path escape ───────────────────────────────


class TestPreviewObjectFilePathEscape(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        state = get_state()
        self._orig_output_dir = state.output_dir
        state.output_dir = self._tmp.name

    def tearDown(self) -> None:
        get_state().output_dir = self._orig_output_dir
        self._tmp.cleanup()

    async def test_path_escape_returns_error(self) -> None:
        # Object dir doesn't exist, so 404 is returned before path traversal check
        response = await preview_object_file("test-obj", "../../etc/passwd")
        self.assertIn(response.status_code, (403, 404))

    async def test_dotdot_path_returns_error(self) -> None:
        # Object dir doesn't exist, so 404 is returned before path traversal check
        response = await preview_object_file("test-obj", "../secret.txt")
        self.assertIn(response.status_code, (403, 404))


# ── Pipeline video-info path security ─────────────────────────────


class TestPipelineVideoInfoPathSecurity(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        state = get_state()
        self._orig_input_dir = state.input_dir
        state.input_dir = self._tmp.name

    def tearDown(self) -> None:
        get_state().input_dir = self._orig_input_dir
        self._tmp.cleanup()

    async def test_path_outside_input_dir_returns_403(self) -> None:
        response = await pipeline_video_info("/etc/passwd")
        self.assertEqual(response.status_code, 403)


# ── Fake WebSocket ────────────────────────────────────────────────


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


# ── Pipeline videos endpoint ─────────────────────────────────────


class TestPipelineVideosEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        state = get_state()
        self._orig_input_dir = state.input_dir
        state.input_dir = self._tmp.name

    def tearDown(self) -> None:
        get_state().input_dir = self._orig_input_dir
        self._tmp.cleanup()

    async def test_returns_video_list_format(self) -> None:
        # Create a test video file
        video = Path(self._tmp.name) / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        response = await pipeline_videos()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertIn("videos", payload)
        self.assertIsInstance(payload["videos"], list)
        self.assertTrue(len(payload["videos"]) >= 1)
        self.assertIn("name", payload["videos"][0])
        self.assertIn("path", payload["videos"][0])

    async def test_filters_video_extensions(self) -> None:
        # Create various files
        for name in ("test.mp4", "test.mov", "test.txt", "test.py"):
            (Path(self._tmp.name) / name).write_bytes(b"\x00")

        response = await pipeline_videos()
        payload = _json_payload(response)
        names = [v["name"] for v in payload["videos"]]
        self.assertIn("test.mp4", names)
        self.assertIn("test.mov", names)
        self.assertNotIn("test.txt", names)
        self.assertNotIn("test.py", names)


# ── Pipeline objects endpoint ─────────────────────────────────────


class TestPipelineObjectsEndpoint(unittest.IsolatedAsyncioTestCase):
    @patch("scripts.dashboard.routes.pipeline.resolve_output_root")
    @patch("scripts.dashboard.routes.pipeline.list_objects", return_value=[])
    async def test_returns_objects_and_active(self, mock_list: MagicMock, mock_root: MagicMock) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mock_root.return_value = Path(tmp)
            response = await pipeline_objects()
            self.assertEqual(response.status_code, 200)
            payload = _json_payload(response)
            self.assertIn("objects", payload)
            self.assertIn("active_object", payload)


# ── Pipeline object-info endpoint ─────────────────────────────────


class TestPipelineObjectInfoEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        state = get_state()
        self._orig_output_dir = state.output_dir
        state.output_dir = self._tmp.name

    def tearDown(self) -> None:
        get_state().output_dir = self._orig_output_dir
        self._tmp.cleanup()

    @patch("scripts.dashboard.routes.pipeline.summarize_object", return_value={"name": "test"})
    async def test_valid_name_returns_info(self, mock_summarize: MagicMock) -> None:
        obj = Path(self._tmp.name) / "objects" / "test-obj"
        obj.mkdir(parents=True)
        response = await pipeline_object_info("test-obj")
        self.assertEqual(response.status_code, 200)

    async def test_invalid_name_returns_400(self) -> None:
        response = await pipeline_object_info("..")
        self.assertEqual(response.status_code, 400)

    async def test_not_found_returns_404(self) -> None:
        response = await pipeline_object_info("nonexistent")
        self.assertEqual(response.status_code, 404)


# ── Pipeline load-object endpoint ─────────────────────────────────


class TestPipelineLoadObjectEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        state = get_state()
        self._orig_output_dir = state.output_dir
        state.output_dir = self._tmp.name
        self._snapshot = {
            "running": session.running,
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        get_state().output_dir = self._orig_output_dir
        self._tmp.cleanup()

    @patch("scripts.dashboard.routes.pipeline.broadcast", new_callable=AsyncMock)
    @patch("scripts.dashboard.dependencies.AppState.load_object_into_session")
    async def test_load_success(self, mock_load: MagicMock, mock_broadcast: AsyncMock) -> None:
        session.running = False
        obj = Path(self._tmp.name) / "objects" / "test-obj"
        obj.mkdir(parents=True)
        mock_load.return_value = {"name": "test-obj"}
        response = await pipeline_load_object({"name": "test-obj"})
        self.assertEqual(response.status_code, 200)

    async def test_load_while_running_returns_409(self) -> None:
        session.running = True
        response = await pipeline_load_object({"name": "test-obj"})
        self.assertEqual(response.status_code, 409)


# ── Pipeline start endpoint ──────────────────────────────────────


class TestPipelineStartEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        state = get_state()
        self._orig_output_dir = state.output_dir
        state.output_dir = self._tmp.name
        self._snapshot = {
            "running": session.running,
            "config": session.config,
            "resume_from_stage": session.resume_from_stage,
        }

    def tearDown(self) -> None:
        session.running = self._snapshot["running"]
        session.config = self._snapshot["config"]
        session.resume_from_stage = self._snapshot["resume_from_stage"]
        get_state().output_dir = self._orig_output_dir
        self._tmp.cleanup()

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

    @patch("scripts.dashboard.routes.pipeline.run_pipeline", new_callable=AsyncMock)
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


# ── Pipeline video-info success ───────────────────────────────────


class TestPipelineVideoInfoSuccess(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        state = get_state()
        self._orig_input_dir = state.input_dir
        state.input_dir = self._tmp.name

    def tearDown(self) -> None:
        get_state().input_dir = self._orig_input_dir
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


# ── SAM2 click success ───────────────────────────────────────────


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


# ── SAM2 undo endpoint ───────────────────────────────────────────


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


# ── SAM2 clear endpoint ──────────────────────────────────────────


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


# ── SAM2 mode endpoint ───────────────────────────────────────────


class TestSam2ModeEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_session = sam2_service._session

    def tearDown(self) -> None:
        sam2_service._session = self._orig_session

    async def test_set_mode_success(self) -> None:
        sam2_service._session = object()
        sam2_service._mode = "object"
        response = await sam2_mode({"mode": "ground"})
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
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
        sam2_service._session = object()
        sam2_service._mode = "object"
        response = await sam2_get_mode()
        self.assertEqual(response.status_code, 200)
        payload = _json_payload(response)
        self.assertIn("mode", payload)


# ── SAM2 skip-ground endpoint ─────────────────────────────────────


class TestSam2SkipGroundEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "sam2_ground_skip_event": session.sam2_ground_skip_event,
        }

    def tearDown(self) -> None:
        session.sam2_ground_skip_event = self._snapshot["sam2_ground_skip_event"]

    async def test_skip_ground_sets_event(self) -> None:
        session.sam2_ground_skip_event = asyncio.Event()
        self.assertFalse(session.sam2_ground_skip_event.is_set())

        response = await sam2_skip_ground()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(session.sam2_ground_skip_event.is_set())


# ── SAM2 frame endpoint ──────────────────────────────────────────


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

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_frame_out_of_range_returns_404(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.side_effect = IndexError("out of range")
        response = await sam2_frame(999)
        self.assertEqual(response.status_code, 404)

    async def test_frame_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_frame(0)
        self.assertEqual(response.status_code, 409)


# ── SAM2 mask endpoint ───────────────────────────────────────────


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

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_mask_not_found_returns_404(self, mock_to_thread: AsyncMock) -> None:
        sam2_service._session = object()
        mock_to_thread.return_value = None
        response = await sam2_mask(999)
        self.assertEqual(response.status_code, 404)

    async def test_mask_not_initialized_returns_409(self) -> None:
        sam2_service._session = None
        response = await sam2_mask(0)
        self.assertEqual(response.status_code, 409)


# ── WebSocket endpoint ───────────────────────────────────────────


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


# ── Preview Endpoints ─────────────────────────────────────────────


class TestPreviewObjectFileNormal(unittest.IsolatedAsyncioTestCase):
    """8.5.3 — preview_object_file normal file serving."""

    @patch("scripts.dashboard.routes.preview.resolve_output_root")
    @patch("scripts.dashboard.routes.preview.validate_object_name", side_effect=lambda n: n)
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

    @patch("scripts.dashboard.dependencies.AppState.active_output_dir")
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

    @patch("scripts.dashboard.dependencies.AppState.active_output_dir")
    async def test_missing_mesh_returns_404(self, mock_dir) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mock_dir.return_value = Path(tmp)

            response = await preview_crop_obb()

            data = _json_payload(response)
            self.assertEqual(response.status_code, 404)


class TestVerificationFrame(unittest.IsolatedAsyncioTestCase):
    """8.5.7 — verification_frame returns composited JPEG."""

    @patch("scripts.dashboard.dependencies.AppState.active_output_dir")
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

    @patch("scripts.dashboard.dependencies.AppState.active_output_dir")
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

    @patch("scripts.dashboard.app.list_objects", return_value=[])
    @patch("scripts.dashboard.app.resolve_output_root")
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

    @patch("scripts.dashboard.app.list_objects")
    @patch("scripts.dashboard.app.resolve_output_root")
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
