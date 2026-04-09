"""Tests for scripts/dashboard/checkpoints.py.

Covers checkpoint_specs, first_checkpoint_id, checkpoint_index,
resolve_checkpoint_id, checkpoint_cleanup_plan, and cleanup_checkpoint_outputs.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.dashboard.checkpoints import (
    checkpoint_cleanup_plan,
    checkpoint_index,
    checkpoint_specs,
    cleanup_checkpoint_outputs,
    first_checkpoint_id,
    resolve_checkpoint_id,
)


# ------------------------------------------------------- TestCheckpointSpecs ---


class TestCheckpointSpecs(unittest.TestCase):
    """checkpoint_specs returns the correct tuple of spec dicts."""

    def test_stage1_returns_tuple(self) -> None:
        specs = checkpoint_specs(1)
        self.assertIsInstance(specs, tuple)
        self.assertTrue(len(specs) > 0)

    def test_stage1_spec_has_required_keys(self) -> None:
        specs = checkpoint_specs(1)
        required = {"id", "label", "patterns", "cleanup_dirs", "cleanup_files"}
        for spec in specs:
            self.assertTrue(required.issubset(spec.keys()), f"Missing keys in {spec}")

    def test_stage4_has_gwrapping_checkpoints(self) -> None:
        specs = checkpoint_specs(4)
        ids = [s["id"] for s in specs]
        self.assertIn("s4.undistort", ids)
        self.assertIn("s4.train_gw", ids)
        self.assertIn("s4.trim", ids)
        self.assertIn("s4.save", ids)

    def test_stage5_has_texture_checkpoints(self) -> None:
        specs = checkpoint_specs(5)
        ids = [s["id"] for s in specs]
        self.assertIn("s5.load", ids)
        self.assertIn("s5.export", ids)

    def test_invalid_stage_returns_empty_tuple(self) -> None:
        self.assertEqual(checkpoint_specs(99), ())


# ----------------------------------------------------- TestFirstCheckpointId ---


class TestFirstCheckpointId(unittest.TestCase):
    """first_checkpoint_id returns the 'id' of the first spec."""

    def test_stage1_first(self) -> None:
        self.assertEqual(first_checkpoint_id(1), "s1.inspect")

    def test_stage4_first(self) -> None:
        self.assertEqual(first_checkpoint_id(4), "s4.undistort")

    def test_stage5_first(self) -> None:
        self.assertEqual(first_checkpoint_id(5), "s5.load")

    def test_invalid_stage_returns_none(self) -> None:
        self.assertIsNone(first_checkpoint_id(99))


# ------------------------------------------------------- TestCheckpointIndex ---


class TestCheckpointIndex(unittest.TestCase):
    """checkpoint_index maps an ID to its position in the spec list."""

    def test_valid_id_returns_index(self) -> None:
        # s1.extract is the second checkpoint (index 1) in stage 1
        self.assertEqual(checkpoint_index(1, "s1.extract"), 1)

    def test_unknown_id_returns_none(self) -> None:
        self.assertIsNone(checkpoint_index(1, "s1.nonexistent"))

    def test_none_id_returns_none(self) -> None:
        self.assertIsNone(checkpoint_index(1, None))


# ------------------------------------------------- TestResolveCheckpointId ---


class TestResolveCheckpointId(unittest.TestCase):
    """resolve_checkpoint_id maps detail text to the matching checkpoint ID."""

    def test_starting_returns_first(self) -> None:
        result = resolve_checkpoint_id(1, "Starting stage 1")
        self.assertEqual(result, "s1.inspect")

    def test_complete_returns_last(self) -> None:
        result = resolve_checkpoint_id(1, "Stage 1 complete")
        self.assertEqual(result, "s1.finalize")

    def test_extracting_frames_matches_extract(self) -> None:
        result = resolve_checkpoint_id(1, "Extracting frames from video")
        self.assertEqual(result, "s1.extract")

    def test_waiting_next_stage_returns_last(self) -> None:
        result = resolve_checkpoint_id(
            1, "waiting for next-stage confirmation"
        )
        self.assertEqual(result, "s1.finalize")

    def test_empty_detail_returns_current(self) -> None:
        result = resolve_checkpoint_id(
            1, "", current_checkpoint_id="s1.extract"
        )
        self.assertEqual(result, "s1.extract")

    def test_none_detail_returns_current(self) -> None:
        result = resolve_checkpoint_id(
            1, None, current_checkpoint_id="s1.inspect"
        )
        self.assertEqual(result, "s1.inspect")

    def test_no_match_returns_current(self) -> None:
        result = resolve_checkpoint_id(
            1, "some unrecognised text", current_checkpoint_id="s1.inspect"
        )
        self.assertEqual(result, "s1.inspect")

    def test_stage4_pattern(self) -> None:
        result = resolve_checkpoint_id(
            4, "Training GaussianWrapping"
        )
        self.assertEqual(result, "s4.train_gw")

    def test_stage5_export_pattern(self) -> None:
        result = resolve_checkpoint_id(
            5, "Exporting textured mesh"
        )
        self.assertEqual(result, "s5.export")


# ------------------------------------------------ TestCheckpointCleanupPlan ---


class TestCheckpointCleanupPlan(unittest.TestCase):
    """checkpoint_cleanup_plan returns (dirs, files, used_fallback)."""

    def test_known_checkpoint_returns_spec(self) -> None:
        dirs, files, used_fallback = checkpoint_cleanup_plan(1, "s1.extract")
        self.assertIn("frames", dirs)
        self.assertFalse(used_fallback)

    def test_unknown_checkpoint_returns_fallback(self) -> None:
        dirs, files, used_fallback = checkpoint_cleanup_plan(1, "s1.bogus")
        self.assertTrue(used_fallback)
        # Fallback for stage 1 includes "frames" dir
        self.assertIn("frames", dirs)

    def test_none_checkpoint_returns_fallback(self) -> None:
        _, _, used_fallback = checkpoint_cleanup_plan(1, None)
        self.assertTrue(used_fallback)

    def test_stage4_known(self) -> None:
        dirs, files, used_fallback = checkpoint_cleanup_plan(4, "s4.save")
        self.assertFalse(used_fallback)
        self.assertIn("object_mesh.ply", files)


# -------------------------------------------- TestCleanupCheckpointOutputs ---


class TestCleanupCheckpointOutputs(unittest.TestCase):
    """cleanup_checkpoint_outputs physically removes dirs/files."""

    def test_removes_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "frames"
            target.mkdir()
            (target / "001.png").touch()
            result = cleanup_checkpoint_outputs(tmpdir, 1, "s1.extract")
            self.assertFalse(target.exists())
            self.assertIn("frames", result["removed_dirs"])

    def test_removes_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "object_mesh.ply"
            target.touch()
            result = cleanup_checkpoint_outputs(tmpdir, 4, "s4.save")
            self.assertFalse(target.exists())
            self.assertIn("object_mesh.ply", result["removed_files"])

    def test_nonexistent_is_safe(self) -> None:
        with TemporaryDirectory() as tmpdir:
            # Nothing to remove -- should not raise
            result = cleanup_checkpoint_outputs(tmpdir, 1, "s1.extract")
            self.assertEqual(result["removed_dirs"], [])
            self.assertEqual(result["removed_files"], [])

    def test_path_traversal_blocked_dotdot(self) -> None:
        """Paths containing '..' must be skipped."""
        with TemporaryDirectory() as tmpdir:
            from unittest.mock import patch

            bad_dirs = ("../escape",)
            bad_files = ("../secret.txt",)
            with patch(
                "scripts.dashboard.checkpoints.checkpoint_cleanup_plan",
                return_value=(bad_dirs, bad_files, True),
            ):
                result = cleanup_checkpoint_outputs(tmpdir, 1, None)
            self.assertEqual(result["removed_dirs"], [])
            self.assertEqual(result["removed_files"], [])

    def test_path_traversal_blocked_absolute(self) -> None:
        """Absolute paths must be skipped."""
        from unittest.mock import patch

        with TemporaryDirectory() as tmpdir:
            bad_dirs = ("/tmp/evil",)
            bad_files = ("/etc/passwd",)
            with patch(
                "scripts.dashboard.checkpoints.checkpoint_cleanup_plan",
                return_value=(bad_dirs, bad_files, True),
            ):
                result = cleanup_checkpoint_outputs(tmpdir, 1, None)
            self.assertEqual(result["removed_dirs"], [])
            self.assertEqual(result["removed_files"], [])

    def test_return_structure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            result = cleanup_checkpoint_outputs(tmpdir, 1, "s1.extract")
            self.assertIn("stage", result)
            self.assertIn("checkpoint_id", result)
            self.assertIn("used_fallback", result)
            self.assertIn("removed_dirs", result)
            self.assertIn("removed_files", result)
            self.assertEqual(result["stage"], 1)
            self.assertEqual(result["checkpoint_id"], "s1.extract")
            self.assertFalse(result["used_fallback"])


if __name__ == "__main__":
    unittest.main()
