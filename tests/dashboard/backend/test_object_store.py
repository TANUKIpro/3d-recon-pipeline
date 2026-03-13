"""Tests for scripts/dashboard/object_store.py utility functions.

Covers stage reset, resume inference, prerequisite validation,
metadata persistence, object summarisation, and listing.
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.dashboard.object_store import (
    OBJECT_META_FILE,
    STAGE_RESET_PATHS,
    _infer_resume_stage,
    _list_objects,
    _reset_outputs_from_stage,
    _summarize_object,
    _validate_resume_prerequisites,
    _write_object_meta,
)


class TestResetOutputsFromStage(unittest.TestCase):
    """7.2.3 — _reset_outputs_from_stage keeps earlier stages intact."""

    def test_reset_from_stage_3_preserves_1_and_2(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Create stage 1 output
            (out / "frames").mkdir()
            (out / "frames" / "00000.jpg").write_bytes(b"\xff\xd8")
            # Create stage 2 outputs
            (out / "object_full.ply").write_text("ply\n")
            (out / "pi3x_cache.npz").write_bytes(b"npz")
            (out / "camera_poses.json").write_text("{}")
            # Create stage 3 outputs
            (out / "masks").mkdir()
            (out / "masks" / "00000.png").write_bytes(b"png")
            (out / "object.ply").write_text("ply\n")
            # Create stage 4 output
            (out / "object_denoised.ply").write_text("ply\n")

            _reset_outputs_from_stage(out, 3)

            # Stages 1-2 intact
            self.assertTrue((out / "frames" / "00000.jpg").is_file())
            self.assertTrue((out / "object_full.ply").is_file())
            self.assertTrue((out / "pi3x_cache.npz").is_file())
            self.assertTrue((out / "camera_poses.json").is_file())
            # Stages 3+ deleted
            self.assertFalse((out / "masks").is_dir())
            self.assertFalse((out / "object.ply").is_file())
            self.assertFalse((out / "object_denoised.ply").is_file())


class TestInferResumeStage(unittest.TestCase):
    """7.2.4 — _infer_resume_stage returns the first incomplete stage."""

    def test_stages_1_2_complete_returns_3(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Stage 1: frames dir with at least one .jpg
            (out / "frames").mkdir()
            (out / "frames" / "00000.jpg").write_bytes(b"\xff\xd8")
            # Stage 2: Pi3X outputs
            (out / "object_full.ply").write_text("ply\n")
            (out / "pi3x_cache.npz").write_bytes(b"npz")
            (out / "camera_poses.json").write_text("{}")

            result = _infer_resume_stage(out)
            self.assertEqual(result, 3)

    def test_empty_dir_returns_1(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _infer_resume_stage(Path(tmp))
            self.assertEqual(result, 1)


class TestValidateResumePrerequisites(unittest.TestCase):
    """7.2.5 — _validate_resume_prerequisites detects missing files."""

    def test_missing_files_returns_issues(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Stage 3 requires frames dir, object_full.ply, camera_poses.json, pi3x_cache.npz
            issues = _validate_resume_prerequisites(out, 3)
            self.assertGreater(len(issues), 0)

    def test_all_present_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Set up prerequisites for stage 4 (needs object.ply)
            (out / "object.ply").write_text("ply\n")

            issues = _validate_resume_prerequisites(out, 4)
            self.assertEqual(issues, [])

    def test_stage_1_no_prerequisites(self) -> None:
        with TemporaryDirectory() as tmp:
            issues = _validate_resume_prerequisites(Path(tmp), 1)
            self.assertEqual(issues, [])


class TestWriteObjectMeta(unittest.TestCase):
    """7.2.6 — _write_object_meta creates/updates metadata file."""

    def test_first_write_creates_meta(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_object_meta("test-obj", out, "/data/input/video.mp4")

            meta_path = out / OBJECT_META_FILE
            self.assertTrue(meta_path.is_file())
            meta = json.loads(meta_path.read_text())
            self.assertEqual(meta["object_name"], "test-obj")
            self.assertEqual(meta["video_path"], "/data/input/video.mp4")
            self.assertIn("created_at", meta)
            self.assertIn("updated_at", meta)

    def test_second_write_preserves_created_at(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_object_meta("test-obj", out, "/data/input/video.mp4")
            meta1 = json.loads((out / OBJECT_META_FILE).read_text())
            created_at = meta1["created_at"]

            time.sleep(0.05)
            _write_object_meta("test-obj", out, "/data/input/video2.mp4")
            meta2 = json.loads((out / OBJECT_META_FILE).read_text())

            self.assertEqual(meta2["created_at"], created_at)
            self.assertEqual(meta2["video_path"], "/data/input/video2.mp4")


class TestSummarizeObject(unittest.TestCase):
    """7.2.7 — _summarize_object returns dict with expected keys."""

    def test_minimal_object_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_object_meta("test-obj", out, "/data/input/v.mp4")

            result = _summarize_object("test-obj", out)

            self.assertEqual(result["name"], "test-obj")
            self.assertIn("video_path", result)
            self.assertIn("output_dir", result)
            self.assertIn("stages", result)
            self.assertIn("frame_count", result)
            self.assertIn("size_mb", result)
            self.assertIn("resume_from_stage", result)


class TestListObjects(unittest.TestCase):
    """7.2.8 — _list_objects returns sorted list."""

    def test_multiple_objects_sorted_by_updated_at_desc(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            objects_dir = base / "objects"
            objects_dir.mkdir()

            # Create 3 objects with staggered timestamps via os.utime so
            # _latest_update_ts (which reads file mtime) produces a
            # deterministic ordering regardless of filesystem granularity.
            import os

            names = ["alpha", "beta", "gamma"]
            for i, name in enumerate(names):
                obj_dir = objects_dir / name
                obj_dir.mkdir()
                _write_object_meta(name, obj_dir, f"/data/input/{name}.mp4")
                # Force mtime: alpha=1000, beta=2000, gamma=3000
                ts = 1000.0 + i * 1000.0
                meta_path = obj_dir / OBJECT_META_FILE
                os.utime(meta_path, (ts, ts))
                os.utime(obj_dir, (ts, ts))

            result = _list_objects(base)

            self.assertEqual(len(result), 3)
            # Most recently updated first
            self.assertEqual(result[0]["name"], "gamma")
            self.assertEqual(result[2]["name"], "alpha")

    def test_empty_objects_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _list_objects(Path(tmp))
            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
