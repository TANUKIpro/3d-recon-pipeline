"""Tests for scripts/dashboard/sam2_service.py.

Covers initialization, click management, undo/clear, propagation,
release, frame retrieval, and thread safety.  All SAM2/GPU dependencies
are mocked so the suite runs on CPU-only hosts.
"""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_FAKE_PNG = b"\x89PNG\r\n\x1a\nfakedata"


# ---------------------------------------------------------------------------
# Base class — patches deferred imports used by SAM2Service
# ---------------------------------------------------------------------------


class _SAM2PatchBase(unittest.TestCase):
    """Base class that patches deferred imports used by SAM2Service.

    Uses ``patch.dict("sys.modules", ...)`` so that every deferred
    ``import cv2`` / ``import torch`` / ``from stage_sam2_ui import …``
    inside SAM2Service methods resolves to fresh MagicMock stubs.
    """

    def setUp(self) -> None:
        # -- build mock modules ------------------------------------------------
        self.mock_torch = MagicMock(name="torch")

        self.mock_cv2 = MagicMock(name="cv2")
        self.mock_cv2.COLOR_RGB2BGR = 4
        self.mock_cv2.IMREAD_GRAYSCALE = 0
        self.mock_cv2.IMWRITE_JPEG_QUALITY = 1

        self.mock_sam2_ui = MagicMock(name="stage_sam2_ui")

        # -- install via patch.dict (auto-reverted in tearDown) ----------------
        self._modules_patch = patch.dict(
            "sys.modules",
            {
                "torch": self.mock_torch,
                "cv2": self.mock_cv2,
                "scripts.stage_sam2_ui": self.mock_sam2_ui,
            },
        )
        self._modules_patch.start()

        # Import SAM2Service *after* patching so the module is importable
        # (it has no top-level heavy imports, but we need sys.modules ready
        # for the deferred imports that fire during tests).
        from scripts.dashboard.sam2_service import SAM2Service

        self.SAM2Service = SAM2Service

    def tearDown(self) -> None:
        self._modules_patch.stop()

    # -- shared helpers --------------------------------------------------------

    def _make_fake_session(self, **overrides) -> MagicMock:
        """Return a MagicMock that behaves like a minimal SAM2Session."""
        s = MagicMock(name="FakeSession")
        s.frame_files = [Path("frame0.jpg"), Path("frame1.jpg")]
        s.img_w = 640
        s.img_h = 480
        s.click_points = []
        s.click_labels = []
        s.ground_click_points = []
        s.ground_click_labels = []
        s.predictor = MagicMock(name="predictor")
        s.inference_state = {"num_frames": 2}
        # Use a MagicMock for mask_dir so .glob() is mockable, but str() works.
        mask_dir = MagicMock(name="mask_dir")
        mask_dir.__str__ = lambda self: "/tmp/test_masks"
        mask_dir.__truediv__ = lambda self, other: Path(f"/tmp/test_masks/{other}")
        mask_dir.glob.return_value = []
        s.mask_dir = mask_dir
        ground_mask_dir = MagicMock(name="ground_mask_dir")
        ground_mask_dir.__str__ = lambda self: "/tmp/test_ground_masks"
        ground_mask_dir.__truediv__ = lambda self, other: Path(
            f"/tmp/test_ground_masks/{other}"
        )
        ground_mask_dir.glob.return_value = []
        s.ground_mask_dir = ground_mask_dir
        s.first_frame = MagicMock(name="first_frame_array")
        s.release_model = MagicMock(name="release_model")
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def _setup_render_mocks(self) -> None:
        """Configure cv2/stage_sam2_ui mocks so _render_overlay_png returns bytes."""
        self.mock_sam2_ui._create_mask_overlay.return_value = MagicMock(
            name="overlay_array"
        )
        self.mock_cv2.cvtColor.return_value = MagicMock(name="vis_bgr")
        buf_mock = MagicMock()
        buf_mock.tobytes.return_value = _FAKE_PNG
        self.mock_cv2.imencode.return_value = (True, buf_mock)


# ====================================================================
# TestSAM2ServiceInitialization
# ====================================================================


class TestSAM2ServiceInitialization(_SAM2PatchBase):
    """Initial state, successful initialization, and double-init behaviour."""

    def setUp(self) -> None:
        super().setUp()
        self.service = self.SAM2Service()

    def test_initial_state_not_initialized(self) -> None:
        """A fresh service reports initialized=False."""
        self.assertFalse(self.service.initialized)

    def test_initialize_returns_metadata_and_sets_initialized(self) -> None:
        """initialize() returns frame metadata and flips initialized to True."""
        fake = self._make_fake_session()
        self.mock_sam2_ui.SAM2Session.return_value = fake
        fake._load_model = MagicMock()
        fake._init_inference_state = MagicMock()

        result = self.service.initialize("/frames", "/output", "large")

        self.assertTrue(self.service.initialized)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["height"], 480)
        fake._load_model.assert_called_once()
        fake._init_inference_state.assert_called_once()

    def test_double_initialize_replaces_session(self) -> None:
        """Calling initialize() twice replaces the old session."""
        first = self._make_fake_session()
        second = self._make_fake_session(img_w=1280, img_h=720)

        self.mock_sam2_ui.SAM2Session.side_effect = [first, second]
        self.service.initialize("/frames", "/output", "large")
        result = self.service.initialize("/frames2", "/output2", "large")

        self.assertEqual(result["width"], 1280)
        self.assertEqual(result["height"], 720)
        # Session pointer should be the second one
        self.assertIs(self.service._session, second)


# ====================================================================
# TestSAM2ServiceAddClick
# ====================================================================


class TestSAM2ServiceAddClick(_SAM2PatchBase):
    """add_click: error when uninitialized, point appended, PNG returned."""

    def setUp(self) -> None:
        super().setUp()
        self._setup_render_mocks()
        self.service = self.SAM2Service()

    def test_add_click_uninitialized_raises(self) -> None:
        """add_click raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.add_click(0.5, 0.5, 1)

    def test_add_click_appends_point_and_label(self) -> None:
        """Click coordinates and label are stored on the session."""
        fake = self._make_fake_session()
        self.service._session = fake
        self.mock_sam2_ui._sanitize_normalized_point.return_value = (0.5, 0.3)
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.add_click(0.5, 0.3, 1)

        self.assertEqual(fake.click_points, [(0.5, 0.3)])
        self.assertEqual(fake.click_labels, [1])
        self.mock_sam2_ui._sanitize_normalized_point.assert_called_with(0.5, 0.3)
        self.mock_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake,
            obj_id=1,
            click_points=fake.click_points,
            click_labels=fake.click_labels,
        )

    def test_add_click_returns_png_bytes(self) -> None:
        """Return value is bytes (the PNG overlay)."""
        fake = self._make_fake_session()
        self.service._session = fake
        self.mock_sam2_ui._sanitize_normalized_point.return_value = (0.1, 0.2)
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        result = self.service.add_click(0.1, 0.2, 0)

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, _FAKE_PNG)

    def test_add_click_ground_mode_stores_on_ground_lists(self) -> None:
        """In ground mode, clicks go to ground_click_points/labels."""
        fake = self._make_fake_session()
        self.service._session = fake
        self.service._segmentation_mode = "ground"
        self.mock_sam2_ui._sanitize_normalized_point.return_value = (0.6, 0.7)
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.add_click(0.6, 0.7, 1)

        self.assertEqual(fake.ground_click_points, [(0.6, 0.7)])
        self.assertEqual(fake.ground_click_labels, [1])
        # Object lists untouched
        self.assertEqual(fake.click_points, [])
        self.assertEqual(fake.click_labels, [])
        self.mock_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake,
            obj_id=2,
            click_points=fake.ground_click_points,
            click_labels=fake.ground_click_labels,
        )


# ====================================================================
# TestSAM2ServiceUndoClick
# ====================================================================


class TestSAM2ServiceUndoClick(_SAM2PatchBase):
    """undo_click: error when uninitialized, pops last, safe on empty."""

    def setUp(self) -> None:
        super().setUp()
        self._setup_render_mocks()
        self.service = self.SAM2Service()

    def test_undo_click_uninitialized_raises(self) -> None:
        """undo_click raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.undo_click()

    def test_undo_click_removes_last(self) -> None:
        """After undo, the most recent point/label pair is removed."""
        fake = self._make_fake_session()
        fake.click_points = [(0.1, 0.2), (0.3, 0.4)]
        fake.click_labels = [1, 0]
        self.service._session = fake
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.undo_click()

        self.assertEqual(fake.click_points, [(0.1, 0.2)])
        self.assertEqual(fake.click_labels, [1])
        self.mock_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake,
            obj_id=1,
            click_points=fake.click_points,
            click_labels=fake.click_labels,
        )

    def test_undo_click_empty_is_safe(self) -> None:
        """Undoing when no clicks exist does not raise; mask becomes None."""
        fake = self._make_fake_session()
        fake.click_points = []
        fake.click_labels = []
        self.service._session = fake

        result = self.service.undo_click()

        self.assertIsNone(self.service._current_mask)
        self.assertIsInstance(result, bytes)


# ====================================================================
# TestSAM2ServiceClearClicks
# ====================================================================


class TestSAM2ServiceClearClicks(_SAM2PatchBase):
    """clear_clicks: error when uninitialized, all points removed."""

    def setUp(self) -> None:
        super().setUp()
        self._setup_render_mocks()
        self.service = self.SAM2Service()

    def test_clear_clicks_uninitialized_raises(self) -> None:
        """clear_clicks raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.clear_clicks()

    def test_clear_clicks_removes_all(self) -> None:
        """After clear, click_points and click_labels are empty."""
        fake = self._make_fake_session()
        fake.click_points = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        fake.click_labels = [1, 0, 1]
        self.service._session = fake

        result = self.service.clear_clicks()

        self.assertEqual(fake.click_points, [])
        self.assertEqual(fake.click_labels, [])
        self.assertIsNone(self.service._current_mask)
        fake.predictor.reset_state.assert_called_once_with(fake.inference_state)
        self.assertIsInstance(result, bytes)

    def test_clear_ground_preserves_object_clicks(self) -> None:
        """Clearing in ground mode removes ground clicks but keeps object clicks."""
        fake = self._make_fake_session()
        fake.click_points = [(0.1, 0.2)]
        fake.click_labels = [1]
        fake.ground_click_points = [(0.5, 0.6), (0.7, 0.8)]
        fake.ground_click_labels = [1, 1]
        self.service._session = fake
        self.service._segmentation_mode = "ground"
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        result = self.service.clear_clicks()

        # Ground clicks cleared
        self.assertEqual(fake.ground_click_points, [])
        self.assertEqual(fake.ground_click_labels, [])
        self.assertIsNone(self.service._current_ground_mask)
        # Object clicks preserved and re-added after reset
        self.assertEqual(fake.click_points, [(0.1, 0.2)])
        self.assertEqual(fake.click_labels, [1])
        fake.predictor.reset_state.assert_called_once_with(fake.inference_state)
        self.mock_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake,
            obj_id=1,
            click_points=fake.click_points,
            click_labels=fake.click_labels,
        )
        self.assertIsInstance(result, bytes)


# ====================================================================
# TestSAM2ServicePropagateAndSave
# ====================================================================


class TestSAM2ServicePropagateAndSave(_SAM2PatchBase):
    """propagate_and_save: uninit, no clicks, and normal flow."""

    def setUp(self) -> None:
        super().setUp()
        self.service = self.SAM2Service()

    def test_propagate_uninitialized_raises(self) -> None:
        """propagate_and_save raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.propagate_and_save()

    def test_propagate_no_clicks_raises(self) -> None:
        """propagate_and_save raises ValueError with no click points."""
        fake = self._make_fake_session()
        fake.click_points = []
        self.service._session = fake

        with self.assertRaises(ValueError):
            self.service.propagate_and_save()

    def test_propagate_normal_returns_mask_dir_tuple(self) -> None:
        """Normal propagation writes masks and returns (mask_dir, None) tuple."""
        fake = self._make_fake_session()
        fake.click_points = [(0.5, 0.5)]
        fake.click_labels = [1]
        fake.inference_state = {"num_frames": 2}
        self.service._session = fake

        # _current_mask is not None: skip re-inference
        mask_array = MagicMock()
        mask_array.__mul__ = MagicMock(return_value=MagicMock())
        self.service._current_mask = mask_array

        # Set up propagate_in_video to yield frames (obj_id=1 only)
        mask_tensor = MagicMock()
        mask_tensor.__getitem__ = MagicMock(
            return_value=MagicMock(
                __gt__=MagicMock(
                    return_value=MagicMock(
                        cpu=MagicMock(
                            return_value=MagicMock(
                                numpy=MagicMock(
                                    return_value=MagicMock(
                                        squeeze=MagicMock(return_value=MagicMock())
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        fake.predictor.propagate_in_video.return_value = [
            (0, [1], mask_tensor),
            (1, [1], mask_tensor),
        ]

        # torch.inference_mode() must work as a context manager
        self.mock_torch.inference_mode.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )

        result = self.service.propagate_and_save()

        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], str(fake.mask_dir))
        self.assertIsNone(result[1])  # no ground clicks -> ground_dir is None
        self.assertTrue(self.mock_cv2.imwrite.called)


# ====================================================================
# TestSAM2ServiceRelease
# ====================================================================


class TestSAM2ServiceRelease(_SAM2PatchBase):
    """release: calls release_model, clears session, safe when uninit."""

    def setUp(self) -> None:
        super().setUp()
        self.service = self.SAM2Service()

    def test_release_calls_release_model(self) -> None:
        """release_model is called on the session."""
        fake = self._make_fake_session()
        self.service._session = fake

        self.service.release()

        fake.release_model.assert_called_once()

    def test_release_sets_session_to_none(self) -> None:
        """After release, session is None and initialized is False."""
        fake = self._make_fake_session()
        self.service._session = fake

        self.service.release()

        self.assertIsNone(self.service._session)
        self.assertFalse(self.service.initialized)

    def test_release_uninitialized_is_noop(self) -> None:
        """Releasing when not initialized does not raise."""
        self.service.release()  # should not raise
        self.assertFalse(self.service.initialized)

    def test_release_clears_current_mask(self) -> None:
        """After release, _current_mask is None."""
        fake = self._make_fake_session()
        self.service._session = fake
        self.service._current_mask = MagicMock(name="some_mask")

        self.service.release()

        self.assertIsNone(self.service._current_mask)


# ====================================================================
# TestSAM2ServiceGetFrame
# ====================================================================


class TestSAM2ServiceGetFrame(_SAM2PatchBase):
    """get_frame_jpeg: uninit error, out-of-range, and normal return."""

    def setUp(self) -> None:
        super().setUp()
        self.service = self.SAM2Service()

    def test_get_frame_uninitialized_raises(self) -> None:
        """get_frame_jpeg raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.get_frame_jpeg(0)

    def test_get_frame_out_of_range_raises(self) -> None:
        """get_frame_jpeg raises IndexError for invalid index."""
        fake = self._make_fake_session()
        self.service._session = fake

        with self.assertRaises(IndexError):
            self.service.get_frame_jpeg(99)

        with self.assertRaises(IndexError):
            self.service.get_frame_jpeg(-1)

    def test_get_frame_returns_jpeg_bytes(self) -> None:
        """get_frame_jpeg reads the file and returns encoded bytes."""
        fake = self._make_fake_session()
        self.service._session = fake
        self.mock_cv2.imread.return_value = MagicMock(name="img_array")
        jpeg_bytes = b"\xff\xd8\xff\xe0jpegdata"
        buf_mock = MagicMock()
        buf_mock.tobytes.return_value = jpeg_bytes
        self.mock_cv2.imencode.return_value = (True, buf_mock)

        result = self.service.get_frame_jpeg(0)

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, jpeg_bytes)
        self.mock_cv2.imread.assert_called_with(str(fake.frame_files[0]))


# ====================================================================
# TestSAM2ServiceThreadSafety
# ====================================================================


class TestSAM2ServiceThreadSafety(_SAM2PatchBase):
    """Concurrent add_click calls must not deadlock or corrupt state."""

    def test_concurrent_add_click_no_deadlock(self) -> None:
        """Multiple threads calling add_click complete without deadlock."""
        self._setup_render_mocks()
        service = self.SAM2Service()
        fake = self._make_fake_session()
        service._session = fake
        self.mock_sam2_ui._sanitize_normalized_point.return_value = (0.5, 0.5)
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                service.add_click(0.5, 0.5, 1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        alive = [t for t in threads if t.is_alive()]
        self.assertEqual(len(alive), 0, "Some threads are still alive (deadlock?)")
        self.assertEqual(errors, [], f"Unexpected errors in threads: {errors}")
        # All 10 clicks should have been appended (serialised by lock)
        self.assertEqual(len(fake.click_points), 10)
        self.assertEqual(len(fake.click_labels), 10)


# ====================================================================
# TestSAM2ServiceCoordinateNormalization
# ====================================================================


class TestSAM2ServiceCoordinateNormalization(_SAM2PatchBase):
    """5.2.5 — Verify _sanitize_normalized_point called during add_click."""

    def setUp(self) -> None:
        super().setUp()
        self._setup_render_mocks()
        self.service = self.SAM2Service()

    def test_sanitize_called_with_coordinates(self) -> None:
        fake = self._make_fake_session()
        self.service._session = fake
        self.mock_sam2_ui._sanitize_normalized_point.return_value = (0.42, 0.58)
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.add_click(0.42, 0.58, 1)

        self.mock_sam2_ui._sanitize_normalized_point.assert_called_with(0.42, 0.58)
        # Verify the sanitized values were used
        self.assertEqual(fake.click_points[-1], (0.42, 0.58))


# ====================================================================
# TestSAM2ServiceGroundUndoIsolation
# ====================================================================


class TestSAM2ServiceGroundUndoIsolation(_SAM2PatchBase):
    """5.3.4 — Ground undo only affects ground list."""

    def setUp(self) -> None:
        super().setUp()
        self._setup_render_mocks()
        self.service = self.SAM2Service()

    def test_ground_undo_preserves_object_clicks(self) -> None:
        fake = self._make_fake_session()
        fake.click_points = [(0.1, 0.2), (0.3, 0.4)]
        fake.click_labels = [1, 1]
        fake.ground_click_points = [(0.5, 0.6), (0.7, 0.8)]
        fake.ground_click_labels = [1, 0]
        self.service._session = fake
        self.service._segmentation_mode = "ground"
        self.mock_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.undo_click()

        # Ground list shrunk
        self.assertEqual(len(fake.ground_click_points), 1)
        self.assertEqual(fake.ground_click_points, [(0.5, 0.6)])
        # Object list unchanged
        self.assertEqual(len(fake.click_points), 2)
        self.assertEqual(fake.click_points, [(0.1, 0.2), (0.3, 0.4)])


# ====================================================================
# TestSAM2ServiceClearResetState
# ====================================================================


class TestSAM2ServiceClearResetState(_SAM2PatchBase):
    """5.4.4 — clear_clicks() calls predictor.reset_state."""

    def setUp(self) -> None:
        super().setUp()
        self._setup_render_mocks()
        self.service = self.SAM2Service()

    def test_clear_calls_predictor_reset_state(self) -> None:
        fake = self._make_fake_session()
        fake.click_points = [(0.5, 0.5)]
        fake.click_labels = [1]
        self.service._session = fake

        self.service.clear_clicks()

        fake.predictor.reset_state.assert_called_once_with(fake.inference_state)


# ====================================================================
# TestSAM2ServicePropagateProgress
# ====================================================================


class TestSAM2ServicePropagateProgress(_SAM2PatchBase):
    """5.5.4 / 5.5.6 — propagate_and_save progress callback + stale mask cleanup."""

    def setUp(self) -> None:
        super().setUp()
        self.service = self.SAM2Service()

    def _setup_propagation(self, fake):
        """Common propagation setup."""
        mask_array = MagicMock()
        mask_array.__mul__ = MagicMock(return_value=MagicMock())
        self.service._current_mask = mask_array

        mask_tensor = MagicMock()
        mask_tensor.__getitem__ = MagicMock(
            return_value=MagicMock(
                __gt__=MagicMock(
                    return_value=MagicMock(
                        cpu=MagicMock(
                            return_value=MagicMock(
                                numpy=MagicMock(
                                    return_value=MagicMock(
                                        squeeze=MagicMock(return_value=MagicMock())
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        fake.predictor.propagate_in_video.return_value = [
            (0, [1], mask_tensor),
            (1, [1], mask_tensor),
        ]
        self.mock_torch.inference_mode.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )

    def test_progress_callback_called(self) -> None:
        """5.5.4 — progress callback receives (frame_idx, total) args."""
        fake = self._make_fake_session()
        fake.click_points = [(0.5, 0.5)]
        fake.click_labels = [1]
        self.service._session = fake
        self._setup_propagation(fake)

        progress = MagicMock()
        self.service.propagate_and_save(progress_callback=progress)

        self.assertTrue(progress.called)
        # Called with (frame_idx, num_frames)
        for call in progress.call_args_list:
            args = call[0]
            self.assertEqual(len(args), 2)

    def test_stale_masks_cleared(self) -> None:
        """5.5.6 — Existing mask files are cleaned before saving."""
        fake = self._make_fake_session()
        fake.click_points = [(0.5, 0.5)]
        fake.click_labels = [1]
        self.service._session = fake
        self._setup_propagation(fake)

        # Pre-create stale mock files
        stale_mask = MagicMock()
        stale_mask.unlink = MagicMock()
        fake.mask_dir.glob.return_value = [stale_mask]
        fake.ground_mask_dir.glob.return_value = []

        self.service.propagate_and_save()

        # Stale masks should have been unlinked
        stale_mask.unlink.assert_called_once()


# ====================================================================
# TestSAM2ServiceGetMaskPng
# ====================================================================


class TestSAM2ServiceGetMaskPng(_SAM2PatchBase):
    """5.7.4 / 5.7.5 — get_mask_png returns bytes or None."""

    def setUp(self) -> None:
        super().setUp()
        self.service = self.SAM2Service()

    def test_existing_mask_returns_bytes(self) -> None:
        """5.7.4 — get_mask_png returns PNG bytes for existing mask."""
        fake = self._make_fake_session()
        self.service._session = fake

        # Make the mask path exist
        mask_path = MagicMock()
        mask_path.exists.return_value = True
        fake.mask_dir.__truediv__ = lambda self, other: mask_path

        self.mock_cv2.imread.return_value = MagicMock(name="mask_img")
        buf_mock = MagicMock()
        buf_mock.tobytes.return_value = b"\x89PNGdata"
        self.mock_cv2.imencode.return_value = (True, buf_mock)

        result = self.service.get_mask_png(0)

        self.assertEqual(result, b"\x89PNGdata")

    def test_missing_mask_returns_none(self) -> None:
        """5.7.5 — get_mask_png returns None for missing mask."""
        fake = self._make_fake_session()
        self.service._session = fake

        # Make the mask path not exist
        mask_path = MagicMock()
        mask_path.exists.return_value = False
        fake.mask_dir.__truediv__ = lambda self, other: mask_path

        result = self.service.get_mask_png(5)

        self.assertIsNone(result)


# ====================================================================
# TestSAM2ServiceSegmentationMode
# ====================================================================


class TestSAM2ServiceSegmentationMode(_SAM2PatchBase):
    """5.9.1 / 5.9.2 / 5.9.3 — Segmentation mode and has_ground_clicks."""

    def setUp(self) -> None:
        super().setUp()
        self.service = self.SAM2Service()

    def test_default_mode_is_object(self) -> None:
        """5.9.1 — Default segmentation mode is 'object'."""
        self.assertEqual(self.service.segmentation_mode, "object")

    def test_set_mode_updates_property(self) -> None:
        """5.9.2 — set_mode changes segmentation_mode."""
        self.service.set_mode("ground")
        self.assertEqual(self.service.segmentation_mode, "ground")

    def test_set_mode_invalid_raises(self) -> None:
        """5.9.2 — Invalid mode raises ValueError."""
        with self.assertRaises(ValueError):
            self.service.set_mode("invalid")

    def test_has_ground_clicks_no_session(self) -> None:
        """5.9.3 — has_ground_clicks False when no session."""
        self.assertFalse(self.service.has_ground_clicks)

    def test_has_ground_clicks_empty(self) -> None:
        """5.9.3 — has_ground_clicks False when ground list empty."""
        fake = self._make_fake_session()
        fake.ground_click_points = []
        self.service._session = fake
        self.assertFalse(self.service.has_ground_clicks)

    def test_has_ground_clicks_nonempty(self) -> None:
        """5.9.3 — has_ground_clicks True when ground list non-empty."""
        fake = self._make_fake_session()
        fake.ground_click_points = [(0.5, 0.5)]
        self.service._session = fake
        self.assertTrue(self.service.has_ground_clicks)


if __name__ == "__main__":
    unittest.main()
