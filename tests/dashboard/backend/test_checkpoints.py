"""Tests for scripts/dashboard/checkpoints.py.

Covers mesh_method_key, checkpoint_specs, first_checkpoint_id,
checkpoint_index, resolve_checkpoint_id, checkpoint_cleanup_plan,
and cleanup_checkpoint_outputs.
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
    mesh_method_key,
    resolve_checkpoint_id,
)


# --------------------------------------------------------- TestMeshMethodKey ---


class TestMeshMethodKey(unittest.TestCase):
    """mesh_method_key normalises a raw config value to 'poisson' or 'diffcd'."""

    def test_diffcd_lowercase(self) -> None:
        self.assertEqual(mesh_method_key("diffcd"), "diffcd")

    def test_diffcd_case_insensitive(self) -> None:
        self.assertEqual(mesh_method_key("DiffCD"), "diffcd")
        self.assertEqual(mesh_method_key("DIFFCD"), "diffcd")
        self.assertEqual(mesh_method_key(" DiffCD "), "diffcd")

    def test_poisson_explicit(self) -> None:
        self.assertEqual(mesh_method_key("poisson"), "poisson")

    def test_none_defaults_to_poisson(self) -> None:
        self.assertEqual(mesh_method_key(None), "poisson")

    def test_empty_string_defaults_to_poisson(self) -> None:
        self.assertEqual(mesh_method_key(""), "poisson")

    def test_unknown_value_defaults_to_poisson(self) -> None:
        self.assertEqual(mesh_method_key("marching_cubes"), "poisson")


# ------------------------------------------------------- TestCheckpointSpecs ---


class TestCheckpointSpecs(unittest.TestCase):
    """checkpoint_specs returns the correct tuple of spec dicts."""

    def test_stage1_returns_tuple(self) -> None:
        specs = checkpoint_specs(1, "poisson")
        self.assertIsInstance(specs, tuple)
        self.assertTrue(len(specs) > 0)

    def test_stage1_spec_has_required_keys(self) -> None:
        specs = checkpoint_specs(1, "poisson")
        required = {"id", "label", "patterns", "cleanup_dirs", "cleanup_files"}
        for spec in specs:
            self.assertTrue(required.issubset(spec.keys()), f"Missing keys in {spec}")

    def test_stage5_poisson_branch(self) -> None:
        specs = checkpoint_specs(5, "poisson")
        ids = [s["id"] for s in specs]
        self.assertIn("s5.poisson.preprocess", ids)
        self.assertIn("s5.poisson.downsample", ids)
        self.assertNotIn("s5.diffcd.prepare", ids)

    def test_stage5_diffcd_branch(self) -> None:
        specs = checkpoint_specs(5, "diffcd")
        ids = [s["id"] for s in specs]
        self.assertIn("s5.diffcd.prepare", ids)
        self.assertIn("s5.diffcd.smooth", ids)
        self.assertNotIn("s5.poisson.preprocess", ids)

    def test_invalid_stage_returns_empty_tuple(self) -> None:
        self.assertEqual(checkpoint_specs(99, "poisson"), ())


# ----------------------------------------------------- TestFirstCheckpointId ---


class TestFirstCheckpointId(unittest.TestCase):
    """first_checkpoint_id returns the 'id' of the first spec."""

    def test_stage1_first(self) -> None:
        self.assertEqual(first_checkpoint_id(1, "poisson"), "s1.inspect")

    def test_stage5_poisson_first(self) -> None:
        self.assertEqual(first_checkpoint_id(5, "poisson"), "s5.poisson.preprocess")

    def test_stage5_diffcd_first(self) -> None:
        self.assertEqual(first_checkpoint_id(5, "diffcd"), "s5.diffcd.prepare")

    def test_invalid_stage_returns_none(self) -> None:
        self.assertIsNone(first_checkpoint_id(99, "poisson"))


# ------------------------------------------------------- TestCheckpointIndex ---


class TestCheckpointIndex(unittest.TestCase):
    """checkpoint_index maps an ID to its position in the spec list."""

    def test_valid_id_returns_index(self) -> None:
        # s1.extract is the second checkpoint (index 1) in stage 1
        self.assertEqual(checkpoint_index(1, "s1.extract", "poisson"), 1)

    def test_unknown_id_returns_none(self) -> None:
        self.assertIsNone(checkpoint_index(1, "s1.nonexistent", "poisson"))

    def test_none_id_returns_none(self) -> None:
        self.assertIsNone(checkpoint_index(1, None, "poisson"))


# ------------------------------------------------- TestResolveCheckpointId ---


class TestResolveCheckpointId(unittest.TestCase):
    """resolve_checkpoint_id maps detail text to the matching checkpoint ID."""

    def test_starting_returns_first(self) -> None:
        result = resolve_checkpoint_id(1, "Starting stage 1", "poisson")
        self.assertEqual(result, "s1.inspect")

    def test_complete_returns_last(self) -> None:
        result = resolve_checkpoint_id(1, "Stage 1 complete", "poisson")
        self.assertEqual(result, "s1.finalize")

    def test_extracting_frames_matches_extract(self) -> None:
        result = resolve_checkpoint_id(1, "Extracting frames from video", "poisson")
        self.assertEqual(result, "s1.extract")

    def test_waiting_next_stage_returns_last(self) -> None:
        result = resolve_checkpoint_id(
            1, "waiting for next-stage confirmation", "poisson"
        )
        self.assertEqual(result, "s1.finalize")

    def test_empty_detail_returns_current(self) -> None:
        result = resolve_checkpoint_id(
            1, "", "poisson", current_checkpoint_id="s1.extract"
        )
        self.assertEqual(result, "s1.extract")

    def test_none_detail_returns_current(self) -> None:
        result = resolve_checkpoint_id(
            1, None, "poisson", current_checkpoint_id="s1.inspect"
        )
        self.assertEqual(result, "s1.inspect")

    def test_no_match_returns_current(self) -> None:
        result = resolve_checkpoint_id(
            1, "some unrecognised text", "poisson", current_checkpoint_id="s1.inspect"
        )
        self.assertEqual(result, "s1.inspect")

    def test_stage5_poisson_pattern(self) -> None:
        result = resolve_checkpoint_id(
            5, "classical/preprocess running", "poisson"
        )
        self.assertEqual(result, "s5.poisson.preprocess")

    def test_stage5_diffcd_pattern(self) -> None:
        result = resolve_checkpoint_id(
            5, "Running DiffCD fitting attempt 1", "diffcd"
        )
        self.assertEqual(result, "s5.diffcd.fit")


# ------------------------------------------------ TestCheckpointCleanupPlan ---


class TestCheckpointCleanupPlan(unittest.TestCase):
    """checkpoint_cleanup_plan returns (dirs, files, used_fallback)."""

    def test_known_checkpoint_returns_spec(self) -> None:
        dirs, files, used_fallback = checkpoint_cleanup_plan(1, "s1.extract", "poisson")
        self.assertIn("frames", dirs)
        self.assertFalse(used_fallback)

    def test_unknown_checkpoint_returns_fallback(self) -> None:
        dirs, files, used_fallback = checkpoint_cleanup_plan(1, "s1.bogus", "poisson")
        self.assertTrue(used_fallback)
        # Fallback for stage 1 includes "frames" dir
        self.assertIn("frames", dirs)

    def test_none_checkpoint_returns_fallback(self) -> None:
        _, _, used_fallback = checkpoint_cleanup_plan(1, None, "poisson")
        self.assertTrue(used_fallback)

    def test_stage5_poisson_known(self) -> None:
        dirs, files, used_fallback = checkpoint_cleanup_plan(
            5, "s5.poisson.reconstruct", "poisson"
        )
        self.assertFalse(used_fallback)
        self.assertIn("object_mesh_raw.ply", files)

    def test_stage5_diffcd_known(self) -> None:
        dirs, files, used_fallback = checkpoint_cleanup_plan(
            5, "s5.diffcd.fit", "diffcd"
        )
        self.assertFalse(used_fallback)
        self.assertIn("diffcd", dirs)


# -------------------------------------------- TestCleanupCheckpointOutputs ---


class TestCleanupCheckpointOutputs(unittest.TestCase):
    """cleanup_checkpoint_outputs physically removes dirs/files."""

    def test_removes_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "frames"
            target.mkdir()
            (target / "001.png").touch()
            result = cleanup_checkpoint_outputs(tmpdir, 1, "s1.extract", "poisson")
            self.assertFalse(target.exists())
            self.assertIn("frames", result["removed_dirs"])

    def test_removes_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "object_denoised.ply"
            target.touch()
            result = cleanup_checkpoint_outputs(tmpdir, 4, "s4.save", "poisson")
            self.assertFalse(target.exists())
            self.assertIn("object_denoised.ply", result["removed_files"])

    def test_nonexistent_is_safe(self) -> None:
        with TemporaryDirectory() as tmpdir:
            # Nothing to remove -- should not raise
            result = cleanup_checkpoint_outputs(tmpdir, 1, "s1.extract", "poisson")
            self.assertEqual(result["removed_dirs"], [])
            self.assertEqual(result["removed_files"], [])

    def test_path_traversal_blocked_dotdot(self) -> None:
        """Paths containing '..' must be skipped."""
        with TemporaryDirectory() as tmpdir:
            # Manually craft a spec-like scenario -- we test via fallback for a
            # stage that we know won't contain traversal paths.  Instead, patch
            # the plan to inject a traversal path and verify it is skipped.
            from unittest.mock import patch

            bad_dirs = ("../escape",)
            bad_files = ("../secret.txt",)
            with patch(
                "scripts.dashboard.checkpoints.checkpoint_cleanup_plan",
                return_value=(bad_dirs, bad_files, True),
            ):
                result = cleanup_checkpoint_outputs(tmpdir, 1, None, "poisson")
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
                result = cleanup_checkpoint_outputs(tmpdir, 1, None, "poisson")
            self.assertEqual(result["removed_dirs"], [])
            self.assertEqual(result["removed_files"], [])

    def test_return_structure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            result = cleanup_checkpoint_outputs(tmpdir, 1, "s1.extract", "poisson")
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
