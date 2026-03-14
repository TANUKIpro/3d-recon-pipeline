"""Comprehensive tests for scripts.dashboard.configuration parsing utilities."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from scripts.dashboard.configuration import (
    build_pipeline_config,
    env_bool,
    env_float,
    env_int,
    parse_bool,
    parse_choice,
    parse_float,
    parse_int,
)
from scripts.config_defaults import (
    EXTRACT_MAX_FRAMES,
    COLMAP_MATCHER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON_BUILD_KW: dict[str, Any] = {
    "video_path": "input.mp4",
    "object_name": "obj",
    "output_dir": Path("/tmp/test"),
    "env": {},
}


def _build(raw: dict[str, Any], **overrides: Any) -> Any:
    """Shorthand for build_pipeline_config with default keyword args."""
    kw = {**_COMMON_BUILD_KW, **overrides}
    return build_pipeline_config(raw, **kw)


# ===================================================================
# TestParseInt
# ===================================================================


class TestParseInt(unittest.TestCase):
    """7 tests for parse_int."""

    def test_valid_int_string(self) -> None:
        self.assertEqual(parse_int("42", 0), 42)

    def test_float_string_truncation_fails(self) -> None:
        # int("3.7") raises ValueError, so fallback is returned
        self.assertEqual(parse_int("3.7", -1), -1)

    def test_float_value_truncates(self) -> None:
        # int(3.7) succeeds and truncates to 3
        self.assertEqual(parse_int(3.7, -1), 3)

    def test_none_returns_fallback(self) -> None:
        self.assertEqual(parse_int(None, 99), 99)

    def test_empty_string_returns_fallback(self) -> None:
        self.assertEqual(parse_int("", 5), 5)

    def test_garbage_returns_fallback(self) -> None:
        self.assertEqual(parse_int("not_a_number", 10), 10)

    def test_negative_value(self) -> None:
        self.assertEqual(parse_int("-7", 0), -7)

    def test_bool_converts_to_int(self) -> None:
        # int(True) == 1, int(False) == 0
        self.assertEqual(parse_int(True, 99), 1)
        self.assertEqual(parse_int(False, 99), 0)


# ===================================================================
# TestParseFloat
# ===================================================================


class TestParseFloat(unittest.TestCase):
    """5 tests for parse_float."""

    def test_valid_float_string(self) -> None:
        self.assertAlmostEqual(parse_float("3.14", 0.0), 3.14)

    def test_int_to_float(self) -> None:
        # float(7) -> 7.0
        self.assertEqual(parse_float(7, 0.0), 7.0)

    def test_none_returns_fallback(self) -> None:
        self.assertEqual(parse_float(None, -1.5), -1.5)

    def test_garbage_returns_fallback(self) -> None:
        self.assertEqual(parse_float("xyz", 2.5), 2.5)

    def test_scientific_notation(self) -> None:
        self.assertEqual(parse_float("1e3", 0.0), 1000.0)


# ===================================================================
# TestParseChoice
# ===================================================================


class TestParseChoice(unittest.TestCase):
    """5 tests for parse_choice."""

    _choices: set[str] = {"alpha", "beta", "gamma"}

    def test_valid_choice(self) -> None:
        self.assertEqual(parse_choice("beta", self._choices, "alpha"), "beta")

    def test_invalid_choice_returns_fallback(self) -> None:
        self.assertEqual(parse_choice("delta", self._choices, "alpha"), "alpha")

    def test_none_returns_fallback(self) -> None:
        # str(None or "") -> "" which is not in choices
        self.assertEqual(parse_choice(None, self._choices, "gamma"), "gamma")

    def test_empty_string_returns_fallback(self) -> None:
        self.assertEqual(parse_choice("", self._choices, "alpha"), "alpha")

    def test_whitespace_is_trimmed(self) -> None:
        self.assertEqual(parse_choice("  beta  ", self._choices, "alpha"), "beta")


# ===================================================================
# TestParseBool
# ===================================================================


class TestParseBool(unittest.TestCase):
    """5 tests for parse_bool."""

    def test_true_variants(self) -> None:
        for val in ("1", "true", "yes", "on", "y", "TRUE", "Yes", "ON"):
            with self.subTest(val=val):
                self.assertTrue(parse_bool(val, False))

    def test_false_variants(self) -> None:
        for val in ("0", "false", "no", "off", "n", "FALSE", "No", "OFF"):
            with self.subTest(val=val):
                self.assertFalse(parse_bool(val, True))

    def test_none_returns_fallback(self) -> None:
        self.assertTrue(parse_bool(None, True))
        self.assertFalse(parse_bool(None, False))

    def test_garbage_returns_fallback(self) -> None:
        self.assertTrue(parse_bool("maybe", True))
        self.assertFalse(parse_bool("dunno", False))

    def test_bool_direct(self) -> None:
        # isinstance(value, bool) branch
        self.assertTrue(parse_bool(True, False))
        self.assertFalse(parse_bool(False, True))


# ===================================================================
# TestEnvInt
# ===================================================================


class TestEnvInt(unittest.TestCase):
    """3 tests for env_int."""

    def test_reads_from_env_dict(self) -> None:
        self.assertEqual(env_int("MY_KEY", 10, env={"MY_KEY": "42"}), 42)

    def test_missing_key_returns_fallback(self) -> None:
        self.assertEqual(env_int("ABSENT", 7, env={}), 7)

    def test_invalid_value_returns_fallback(self) -> None:
        self.assertEqual(env_int("BAD", 5, env={"BAD": "nope"}), 5)


# ===================================================================
# TestEnvFloat
# ===================================================================


class TestEnvFloat(unittest.TestCase):
    """2 tests for env_float."""

    def test_reads_from_env_dict(self) -> None:
        self.assertAlmostEqual(env_float("F", 0.0, env={"F": "2.5"}), 2.5)

    def test_missing_key_returns_fallback(self) -> None:
        self.assertAlmostEqual(env_float("MISSING", 1.1, env={}), 1.1)


# ===================================================================
# TestEnvBool
# ===================================================================


class TestEnvBool(unittest.TestCase):
    """3 tests for env_bool."""

    def test_true_value(self) -> None:
        self.assertTrue(env_bool("FLAG", False, env={"FLAG": "yes"}))

    def test_false_value(self) -> None:
        self.assertFalse(env_bool("FLAG", True, env={"FLAG": "0"}))

    def test_missing_key_returns_fallback(self) -> None:
        self.assertTrue(env_bool("ABSENT", True, env={}))
        self.assertFalse(env_bool("ABSENT", False, env={}))


# ===================================================================
# TestBuildPipelineConfigExtended
# ===================================================================


class TestBuildPipelineConfigExtended(unittest.TestCase):
    """Tests for build_pipeline_config."""

    # -- auto_accept ---

    def test_auto_accept_parsed_via_parse_bool(self) -> None:
        cfg_true = _build({"auto_accept": "yes"})
        self.assertTrue(cfg_true.auto_accept)

        cfg_false = _build({"auto_accept": "no"})
        self.assertFalse(cfg_false.auto_accept)

        cfg_default = _build({})
        self.assertFalse(cfg_default.auto_accept)

    # -- colmap_matcher ---

    def test_colmap_matcher_parse_choice(self) -> None:
        cfg_seq = _build({"colmap_matcher": "sequential"})
        self.assertEqual(cfg_seq.colmap_matcher, "sequential")

        cfg_exh = _build({"colmap_matcher": "exhaustive"})
        self.assertEqual(cfg_exh.colmap_matcher, "exhaustive")

        cfg_invalid = _build({"colmap_matcher": "invalid_matcher"})
        self.assertEqual(cfg_invalid.colmap_matcher, COLMAP_MATCHER)

    def test_ground_plane_enabled_parse_and_env(self) -> None:
        cfg_yes = _build({"ground_plane_enabled": "yes"})
        self.assertTrue(cfg_yes.ground_plane_enabled)

        cfg_env_off = _build({}, env={"GROUND_PLANE_ENABLED": "0"})
        self.assertFalse(cfg_env_off.ground_plane_enabled)

        # Default (empty env) uses config_defaults constant
        cfg_default = _build({})
        from scripts.config_defaults import GROUND_PLANE_ENABLED
        self.assertEqual(cfg_default.ground_plane_enabled, GROUND_PLANE_ENABLED)

    def test_frame_interval_max_frames_env(self) -> None:
        cfg_env = _build(
            {},
            env={
                "FRAME_INTERVAL": "7",
                "MAX_FRAMES": "30",
            },
        )
        self.assertEqual(cfg_env.frame_interval, 7)
        self.assertEqual(cfg_env.max_frames, 30)

        # Raw overrides env
        cfg_raw = _build(
            {"frame_interval": 3},
            env={"FRAME_INTERVAL": "7"},
        )
        self.assertEqual(cfg_raw.frame_interval, 3)

    def test_empty_raw_produces_valid_config(self) -> None:
        cfg = _build({})
        self.assertGreater(cfg.frame_interval, 0)
        self.assertGreaterEqual(cfg.max_frames, 2)
        self.assertEqual(cfg.video_path, "input.mp4")

    # -- gs2mesh fields ---

    def test_gs2mesh_iterations_clamped(self) -> None:
        cfg = _build({"gs2mesh_gs_iterations": 500})
        self.assertEqual(cfg.gs2mesh_gs_iterations, 1000)

    def test_gs2mesh_tsdf_voxel_size_clamped(self) -> None:
        cfg = _build({"gs2mesh_tsdf_voxel_size": 0.0001})
        self.assertEqual(cfg.gs2mesh_tsdf_voxel_size, 0.001)

    def test_gs2mesh_tsdf_depth_trunc_clamped(self) -> None:
        cfg = _build({"gs2mesh_tsdf_depth_trunc": 0.001})
        self.assertEqual(cfg.gs2mesh_tsdf_depth_trunc, 0.005)

    def test_gs2mesh_use_masks_parsed(self) -> None:
        cfg = _build({"gs2mesh_use_masks": False})
        self.assertFalse(cfg.gs2mesh_use_masks)

    def test_gs2mesh_runtime_profile_parse_and_env(self) -> None:
        cfg_raw = _build({"gs2mesh_runtime_profile": "compat"})
        self.assertEqual(cfg_raw.gs2mesh_runtime_profile, "compat")

        cfg_env = _build({}, env={"GS2MESH_RUNTIME_PROFILE": "compat"})
        self.assertEqual(cfg_env.gs2mesh_runtime_profile, "compat")

        cfg_invalid = _build({"gs2mesh_runtime_profile": "invalid"})
        self.assertEqual(cfg_invalid.gs2mesh_runtime_profile, "auto")

    # -- colmap fields ---

    def test_colmap_max_features_clamped(self) -> None:
        cfg = _build({"colmap_max_features": 500})
        self.assertEqual(cfg.colmap_max_features, 1000)

    def test_colmap_image_size_clamped(self) -> None:
        cfg = _build({"colmap_image_size": 100})
        self.assertEqual(cfg.colmap_image_size, 256)

    # -- texture fields ---

    def test_texture_size_zero_is_preserved_for_auto_mode(self) -> None:
        cfg = _build({"texture_size": 0})
        self.assertEqual(cfg.texture_size, 0)

    def test_texture_view_assign_mode_defaults_to_region_gc(self) -> None:
        cfg = _build({})
        self.assertEqual(cfg.texture_view_assign_mode, "region_gc")

    def test_texture_quality_boost_defaults_to_true(self) -> None:
        cfg = _build({})
        self.assertTrue(cfg.texture_quality_boost)


if __name__ == "__main__":
    unittest.main()
