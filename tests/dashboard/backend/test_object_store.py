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
    infer_resume_stage,
    list_objects,
    reset_outputs_from_stage,
    summarize_object,
    validate_resume_prerequisites,
    write_object_meta,
)
from scripts.output_layout import (
    camera_poses_path,
    colmap_sparse_dir,
    frames_dir,
    intrinsics_path,
    masks_dir,
    object_mesh_path,
    texture_png_path,
    textured_mesh_obj_path,
)


class TestResetOutputsFromStage(unittest.TestCase):
    """reset_outputs_from_stage keeps earlier stages intact."""

    def test_reset_from_stage_5_preserves_intrinsics(self) -> None:
        """Stage 5 reset must NOT wipe intrinsics.json.

        Mirrors the 6d31fba fix to ``checkpoints.py:_STAGE_FALLBACK_RESET[5]``
        for the second reset code path. COLMAP's bundle-adjusted pinhole
        values must survive a stage-5 restart so the bake doesn't fall onto
        its grid-search estimator and destroy fine texture detail on
        cylindrical surfaces.
        """
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Stage 2: COLMAP intrinsics + sparse + poses
            intr_file = intrinsics_path(out)
            intr_file.parent.mkdir(parents=True, exist_ok=True)
            intr_file.write_text(json.dumps({"source": "colmap:SIMPLE_RADIAL"}))
            colmap_sparse_dir(out).mkdir(parents=True, exist_ok=True)
            # Stage 5: textured outputs
            tex_obj = textured_mesh_obj_path(out)
            tex_obj.parent.mkdir(parents=True, exist_ok=True)
            tex_obj.write_text("o\n")
            tex_png = texture_png_path(out)
            tex_png.write_bytes(b"\x89PNG")

            reset_outputs_from_stage(out, 5)

            # Stage 5 outputs deleted
            self.assertFalse(textured_mesh_obj_path(out).is_file())
            self.assertFalse(texture_png_path(out).is_file())
            # Stage 2 intrinsics preserved (the regression guard)
            self.assertTrue(intrinsics_path(out).is_file())
            # COLMAP sparse model preserved
            self.assertTrue(colmap_sparse_dir(out).is_dir())

    def test_reset_from_stage_3_preserves_1_and_2(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Create stage 1 output
            frames_dir(out).mkdir(parents=True)
            (frames_dir(out) / "00000.jpg").write_bytes(b"\xff\xd8")
            # Create stage 2 outputs
            colmap_sparse_dir(out).mkdir(parents=True)
            poses_file = camera_poses_path(out)
            poses_file.parent.mkdir(parents=True, exist_ok=True)
            poses_file.write_text("{}")
            # Create stage 3 outputs
            masks_dir(out).mkdir(parents=True)
            (masks_dir(out) / "00000.png").write_bytes(b"png")
            # Create stage 4 output
            mesh_file = object_mesh_path(out)
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            mesh_file.write_text("ply\n")

            reset_outputs_from_stage(out, 3)

            # Stages 1-2 intact
            self.assertTrue((frames_dir(out) / "00000.jpg").is_file())
            self.assertTrue(colmap_sparse_dir(out).is_dir())
            self.assertTrue(camera_poses_path(out).is_file())
            # Stages 3+ deleted
            self.assertFalse(masks_dir(out).is_dir())
            self.assertFalse(object_mesh_path(out).is_file())


class TestInferResumeStage(unittest.TestCase):
    """infer_resume_stage returns the first incomplete stage."""

    def test_stage_1_complete_returns_2(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Stage 1: frames dir with at least one .jpg
            frames_dir(out).mkdir(parents=True)
            (frames_dir(out) / "00000.jpg").write_bytes(b"\xff\xd8")

            result = infer_resume_stage(out)
            self.assertEqual(result, 2)

    def test_empty_dir_returns_1(self) -> None:
        with TemporaryDirectory() as tmp:
            result = infer_resume_stage(Path(tmp))
            self.assertEqual(result, 1)


class TestValidateResumePrerequisites(unittest.TestCase):
    """validate_resume_prerequisites detects missing files."""

    def test_missing_files_returns_issues(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Stage 3 requires frames dir and camera_poses.json
            issues = validate_resume_prerequisites(out, 3)
            self.assertGreater(len(issues), 0)

    def test_all_present_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Set up prerequisites for stage 4 (needs frames dir, camera_poses.json)
            frames_dir(out).mkdir(parents=True)
            (frames_dir(out) / "00000.jpg").write_bytes(b"\xff\xd8")
            poses_file = camera_poses_path(out)
            poses_file.parent.mkdir(parents=True, exist_ok=True)
            poses_file.write_text("{}")

            issues = validate_resume_prerequisites(out, 4)
            self.assertEqual(issues, [])

    def test_stage_1_no_prerequisites(self) -> None:
        with TemporaryDirectory() as tmp:
            issues = validate_resume_prerequisites(Path(tmp), 1)
            self.assertEqual(issues, [])


class TestWriteObjectMeta(unittest.TestCase):
    """write_object_meta creates/updates metadata file."""

    def test_first_write_creates_meta(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_object_meta("test-obj", out, "/data/input/video.mp4")

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
            write_object_meta("test-obj", out, "/data/input/video.mp4")
            meta1 = json.loads((out / OBJECT_META_FILE).read_text())
            created_at = meta1["created_at"]

            time.sleep(0.05)
            write_object_meta("test-obj", out, "/data/input/video2.mp4")
            meta2 = json.loads((out / OBJECT_META_FILE).read_text())

            self.assertEqual(meta2["created_at"], created_at)
            self.assertEqual(meta2["video_path"], "/data/input/video2.mp4")


class TestSummarizeObject(unittest.TestCase):
    """summarize_object returns dict with expected keys."""

    def test_minimal_object_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_object_meta("test-obj", out, "/data/input/v.mp4")

            result = summarize_object("test-obj", out)

            self.assertEqual(result["name"], "test-obj")
            self.assertIn("video_path", result)
            self.assertIn("output_dir", result)
            self.assertIn("stages", result)
            self.assertIn("frame_count", result)
            self.assertIn("size_mb", result)
            self.assertIn("resume_from_stage", result)


class TestListObjects(unittest.TestCase):
    """list_objects returns sorted list."""

    def test_multiple_objects_sorted_by_updated_at_desc(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            branch_dir = base / "objects" / "@main"
            branch_dir.mkdir(parents=True)

            # Create 3 objects with staggered timestamps via os.utime so
            # _latest_update_ts (which reads file mtime) produces a
            # deterministic ordering regardless of filesystem granularity.
            import os

            names = ["alpha", "beta", "gamma"]
            for i, name in enumerate(names):
                obj_dir = branch_dir / name
                obj_dir.mkdir()
                write_object_meta(name, obj_dir, f"/data/input/{name}.mp4")
                # Force mtime: alpha=1000, beta=2000, gamma=3000
                ts = 1000.0 + i * 1000.0
                meta_path = obj_dir / OBJECT_META_FILE
                os.utime(meta_path, (ts, ts))
                os.utime(obj_dir, (ts, ts))

            result = list_objects(base, "main")

            self.assertEqual(len(result), 3)
            # Most recently updated first
            self.assertEqual(result[0]["name"], "gamma")
            self.assertEqual(result[2]["name"], "alpha")

    def test_empty_objects_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            result = list_objects(Path(tmp), "main")
            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
