"""Tests for scripts/dashboard/sam2_service.py.

Covers initialization, click management, undo/clear, propagation,
release, frame retrieval, and thread safety.  All SAM2/GPU dependencies
are mocked so the suite runs on CPU-only hosts.
"""

from __future__ import annotations

import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy modules that are imported *inside* SAM2Service methods.
# Because the imports are deferred (`from stage_sam2_ui import ...`), we
# install lightweight stubs into sys.modules before importing the module
# under test.  Each test resets relevant return values in setUp.
# ---------------------------------------------------------------------------

_mock_stage_sam2_ui = types.ModuleType("stage_sam2_ui")
_mock_stage_sam2_ui.SAM2Session = MagicMock(name="SAM2Session")
_mock_stage_sam2_ui._run_single_frame_inference = MagicMock(
    name="_run_single_frame_inference"
)
_mock_stage_sam2_ui._run_single_frame_inference_obj = MagicMock(
    name="_run_single_frame_inference_obj"
)
_mock_stage_sam2_ui._sanitize_normalized_point = MagicMock(
    name="_sanitize_normalized_point"
)
_mock_stage_sam2_ui._create_mask_overlay = MagicMock(name="_create_mask_overlay")

_mock_cv2 = types.ModuleType("cv2")
_mock_cv2.cvtColor = MagicMock(name="cv2.cvtColor")
_mock_cv2.imencode = MagicMock(name="cv2.imencode")
_mock_cv2.imread = MagicMock(name="cv2.imread")
_mock_cv2.imwrite = MagicMock(name="cv2.imwrite")
_mock_cv2.COLOR_RGB2BGR = 4
_mock_cv2.IMREAD_GRAYSCALE = 0
_mock_cv2.IMWRITE_JPEG_QUALITY = 1

_mock_torch = types.ModuleType("torch")
_mock_torch.inference_mode = MagicMock(name="torch.inference_mode")

_mock_numpy = types.ModuleType("numpy")
_mock_numpy.uint8 = "uint8"

# Patch sys.modules so deferred imports resolve to our stubs.
_MODULE_PATCHES = {
    "stage_sam2_ui": _mock_stage_sam2_ui,
    "cv2": _mock_cv2,
    "torch": _mock_torch,
    "numpy": _mock_numpy,
}

for _name, _mod in _MODULE_PATCHES.items():
    sys.modules.setdefault(_name, _mod)

from scripts.dashboard.sam2_service import SAM2Service  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_PNG = b"\x89PNG\r\n\x1a\nfakedata"


def _reset_all_mocks() -> None:
    """Reset all module-level mocks to avoid cross-test contamination."""
    _mock_stage_sam2_ui.SAM2Session.reset_mock()
    _mock_stage_sam2_ui.SAM2Session.side_effect = None
    _mock_stage_sam2_ui._run_single_frame_inference.reset_mock()
    _mock_stage_sam2_ui._run_single_frame_inference_obj.reset_mock()
    _mock_stage_sam2_ui._sanitize_normalized_point.reset_mock()
    _mock_stage_sam2_ui._create_mask_overlay.reset_mock()
    _mock_cv2.cvtColor.reset_mock()
    _mock_cv2.imencode.reset_mock()
    _mock_cv2.imread.reset_mock()
    _mock_cv2.imwrite.reset_mock()
    _mock_torch.inference_mode.reset_mock()


def _make_fake_session(**overrides) -> MagicMock:
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
    ground_mask_dir.__truediv__ = lambda self, other: Path(f"/tmp/test_ground_masks/{other}")
    ground_mask_dir.glob.return_value = []
    s.ground_mask_dir = ground_mask_dir
    s.first_frame = MagicMock(name="first_frame_array")
    s.release_model = MagicMock(name="release_model")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _setup_render_mocks() -> None:
    """Configure cv2/stage_sam2_ui mocks so _render_overlay_png returns bytes."""
    _mock_stage_sam2_ui._create_mask_overlay.return_value = MagicMock(
        name="overlay_array"
    )
    _mock_cv2.cvtColor.return_value = MagicMock(name="vis_bgr")
    buf_mock = MagicMock()
    buf_mock.tobytes.return_value = _FAKE_PNG
    _mock_cv2.imencode.return_value = (True, buf_mock)


# ====================================================================
# TestSAM2ServiceInitialization
# ====================================================================


class TestSAM2ServiceInitialization(unittest.TestCase):
    """Initial state, successful initialization, and double-init behaviour."""

    def setUp(self) -> None:
        _reset_all_mocks()
        self.service = SAM2Service()

    def test_initial_state_not_initialized(self) -> None:
        """A fresh service reports initialized=False."""
        self.assertFalse(self.service.initialized)

    def test_initialize_returns_metadata_and_sets_initialized(self) -> None:
        """initialize() returns frame metadata and flips initialized to True."""
        fake = _make_fake_session()
        _mock_stage_sam2_ui.SAM2Session.return_value = fake
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
        first = _make_fake_session()
        second = _make_fake_session(img_w=1280, img_h=720)

        _mock_stage_sam2_ui.SAM2Session.side_effect = [first, second]
        self.service.initialize("/frames", "/output", "large")
        result = self.service.initialize("/frames2", "/output2", "large")

        self.assertEqual(result["width"], 1280)
        self.assertEqual(result["height"], 720)
        # Session pointer should be the second one
        self.assertIs(self.service._session, second)


# ====================================================================
# TestSAM2ServiceAddClick
# ====================================================================


class TestSAM2ServiceAddClick(unittest.TestCase):
    """add_click: error when uninitialized, point appended, PNG returned."""

    def setUp(self) -> None:
        _reset_all_mocks()
        _setup_render_mocks()
        self.service = SAM2Service()

    def test_add_click_uninitialized_raises(self) -> None:
        """add_click raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.add_click(0.5, 0.5, 1)

    def test_add_click_appends_point_and_label(self) -> None:
        """Click coordinates and label are stored on the session."""
        fake = _make_fake_session()
        self.service._session = fake
        _mock_stage_sam2_ui._sanitize_normalized_point.return_value = (0.5, 0.3)
        _mock_stage_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.add_click(0.5, 0.3, 1)

        self.assertEqual(fake.click_points, [(0.5, 0.3)])
        self.assertEqual(fake.click_labels, [1])
        _mock_stage_sam2_ui._sanitize_normalized_point.assert_called_with(0.5, 0.3)
        _mock_stage_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake, obj_id=1,
            click_points=fake.click_points,
            click_labels=fake.click_labels,
        )

    def test_add_click_returns_png_bytes(self) -> None:
        """Return value is bytes (the PNG overlay)."""
        fake = _make_fake_session()
        self.service._session = fake
        _mock_stage_sam2_ui._sanitize_normalized_point.return_value = (0.1, 0.2)
        _mock_stage_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        result = self.service.add_click(0.1, 0.2, 0)

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, _FAKE_PNG)

    def test_add_click_ground_mode_stores_on_ground_lists(self) -> None:
        """In ground mode, clicks go to ground_click_points/labels."""
        fake = _make_fake_session()
        self.service._session = fake
        self.service._segmentation_mode = "ground"
        _mock_stage_sam2_ui._sanitize_normalized_point.return_value = (0.6, 0.7)
        _mock_stage_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.add_click(0.6, 0.7, 1)

        self.assertEqual(fake.ground_click_points, [(0.6, 0.7)])
        self.assertEqual(fake.ground_click_labels, [1])
        # Object lists untouched
        self.assertEqual(fake.click_points, [])
        self.assertEqual(fake.click_labels, [])
        _mock_stage_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake, obj_id=2,
            click_points=fake.ground_click_points,
            click_labels=fake.ground_click_labels,
        )


# ====================================================================
# TestSAM2ServiceUndoClick
# ====================================================================


class TestSAM2ServiceUndoClick(unittest.TestCase):
    """undo_click: error when uninitialized, pops last, safe on empty."""

    def setUp(self) -> None:
        _reset_all_mocks()
        _setup_render_mocks()
        self.service = SAM2Service()

    def test_undo_click_uninitialized_raises(self) -> None:
        """undo_click raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.undo_click()

    def test_undo_click_removes_last(self) -> None:
        """After undo, the most recent point/label pair is removed."""
        fake = _make_fake_session()
        fake.click_points = [(0.1, 0.2), (0.3, 0.4)]
        fake.click_labels = [1, 0]
        self.service._session = fake
        _mock_stage_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        self.service.undo_click()

        self.assertEqual(fake.click_points, [(0.1, 0.2)])
        self.assertEqual(fake.click_labels, [1])
        _mock_stage_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake, obj_id=1,
            click_points=fake.click_points,
            click_labels=fake.click_labels,
        )

    def test_undo_click_empty_is_safe(self) -> None:
        """Undoing when no clicks exist does not raise; mask becomes None."""
        fake = _make_fake_session()
        fake.click_points = []
        fake.click_labels = []
        self.service._session = fake

        result = self.service.undo_click()

        self.assertIsNone(self.service._current_mask)
        self.assertIsInstance(result, bytes)


# ====================================================================
# TestSAM2ServiceClearClicks
# ====================================================================


class TestSAM2ServiceClearClicks(unittest.TestCase):
    """clear_clicks: error when uninitialized, all points removed."""

    def setUp(self) -> None:
        _reset_all_mocks()
        _setup_render_mocks()
        self.service = SAM2Service()

    def test_clear_clicks_uninitialized_raises(self) -> None:
        """clear_clicks raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.clear_clicks()

    def test_clear_clicks_removes_all(self) -> None:
        """After clear, click_points and click_labels are empty."""
        fake = _make_fake_session()
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
        fake = _make_fake_session()
        fake.click_points = [(0.1, 0.2)]
        fake.click_labels = [1]
        fake.ground_click_points = [(0.5, 0.6), (0.7, 0.8)]
        fake.ground_click_labels = [1, 1]
        self.service._session = fake
        self.service._segmentation_mode = "ground"
        _mock_stage_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

        result = self.service.clear_clicks()

        # Ground clicks cleared
        self.assertEqual(fake.ground_click_points, [])
        self.assertEqual(fake.ground_click_labels, [])
        self.assertIsNone(self.service._current_ground_mask)
        # Object clicks preserved and re-added after reset
        self.assertEqual(fake.click_points, [(0.1, 0.2)])
        self.assertEqual(fake.click_labels, [1])
        fake.predictor.reset_state.assert_called_once_with(fake.inference_state)
        _mock_stage_sam2_ui._run_single_frame_inference_obj.assert_called_once_with(
            fake, obj_id=1,
            click_points=fake.click_points,
            click_labels=fake.click_labels,
        )
        self.assertIsInstance(result, bytes)


# ====================================================================
# TestSAM2ServicePropagateAndSave
# ====================================================================


class TestSAM2ServicePropagateAndSave(unittest.TestCase):
    """propagate_and_save: uninit, no clicks, and normal flow."""

    def setUp(self) -> None:
        _reset_all_mocks()
        self.service = SAM2Service()

    def test_propagate_uninitialized_raises(self) -> None:
        """propagate_and_save raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.propagate_and_save()

    def test_propagate_no_clicks_raises(self) -> None:
        """propagate_and_save raises ValueError with no click points."""
        fake = _make_fake_session()
        fake.click_points = []
        self.service._session = fake

        with self.assertRaises(ValueError):
            self.service.propagate_and_save()

    def test_propagate_normal_returns_mask_dir_tuple(self) -> None:
        """Normal propagation writes masks and returns (mask_dir, None) tuple."""
        fake = _make_fake_session()
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
        _mock_torch.inference_mode.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )

        result = self.service.propagate_and_save()

        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], str(fake.mask_dir))
        self.assertIsNone(result[1])  # no ground clicks → ground_dir is None
        self.assertTrue(_mock_cv2.imwrite.called)


# ====================================================================
# TestSAM2ServiceRelease
# ====================================================================


class TestSAM2ServiceRelease(unittest.TestCase):
    """release: calls release_model, clears session, safe when uninit."""

    def setUp(self) -> None:
        _reset_all_mocks()
        self.service = SAM2Service()

    def test_release_calls_release_model(self) -> None:
        """release_model is called on the session."""
        fake = _make_fake_session()
        self.service._session = fake

        self.service.release()

        fake.release_model.assert_called_once()

    def test_release_sets_session_to_none(self) -> None:
        """After release, session is None and initialized is False."""
        fake = _make_fake_session()
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
        fake = _make_fake_session()
        self.service._session = fake
        self.service._current_mask = MagicMock(name="some_mask")

        self.service.release()

        self.assertIsNone(self.service._current_mask)


# ====================================================================
# TestSAM2ServiceGetFrame
# ====================================================================


class TestSAM2ServiceGetFrame(unittest.TestCase):
    """get_frame_jpeg: uninit error, out-of-range, and normal return."""

    def setUp(self) -> None:
        _reset_all_mocks()
        self.service = SAM2Service()

    def test_get_frame_uninitialized_raises(self) -> None:
        """get_frame_jpeg raises RuntimeError when session is None."""
        with self.assertRaises(RuntimeError):
            self.service.get_frame_jpeg(0)

    def test_get_frame_out_of_range_raises(self) -> None:
        """get_frame_jpeg raises IndexError for invalid index."""
        fake = _make_fake_session()
        self.service._session = fake

        with self.assertRaises(IndexError):
            self.service.get_frame_jpeg(99)

        with self.assertRaises(IndexError):
            self.service.get_frame_jpeg(-1)

    def test_get_frame_returns_jpeg_bytes(self) -> None:
        """get_frame_jpeg reads the file and returns encoded bytes."""
        fake = _make_fake_session()
        self.service._session = fake
        _mock_cv2.imread.return_value = MagicMock(name="img_array")
        jpeg_bytes = b"\xff\xd8\xff\xe0jpegdata"
        buf_mock = MagicMock()
        buf_mock.tobytes.return_value = jpeg_bytes
        _mock_cv2.imencode.return_value = (True, buf_mock)

        result = self.service.get_frame_jpeg(0)

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, jpeg_bytes)
        _mock_cv2.imread.assert_called_with(str(fake.frame_files[0]))


# ====================================================================
# TestSAM2ServiceThreadSafety
# ====================================================================


class TestSAM2ServiceThreadSafety(unittest.TestCase):
    """Concurrent add_click calls must not deadlock or corrupt state."""

    def test_concurrent_add_click_no_deadlock(self) -> None:
        """Multiple threads calling add_click complete without deadlock."""
        _reset_all_mocks()
        _setup_render_mocks()
        service = SAM2Service()
        fake = _make_fake_session()
        service._session = fake
        _mock_stage_sam2_ui._sanitize_normalized_point.return_value = (0.5, 0.5)
        _mock_stage_sam2_ui._run_single_frame_inference_obj.return_value = MagicMock()

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


if __name__ == "__main__":
    unittest.main()
