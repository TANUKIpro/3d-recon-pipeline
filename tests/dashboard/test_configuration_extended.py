"""Comprehensive tests for scripts.dashboard.configuration parsing utilities."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from scripts.dashboard.configuration import (
    DENOISE_PRESET_DEFAULTS,
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
    CLASSICAL_PRESET_DEFAULTS,
    DENOISE_DEFAULT_PRESET,
    EXTRACT_MAX_FRAMES,
    MESH_DEFAULT_METHOD,
    MESH_METHODS,
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
        # float(7) → 7.0
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
        # str(None or "") → "" which is not in choices
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
    """12 tests for build_pipeline_config."""

    # -- Denoise preset handling ---

    def test_denoise_preset_custom_preserved(self) -> None:
        cfg = _build({"denoise_preset": "custom"})
        self.assertEqual(cfg.denoise_preset, "custom")

    def test_denoise_preset_valid_applied(self) -> None:
        cfg = _build({"denoise_preset": "aggressive_cleanup"})
        self.assertEqual(cfg.denoise_preset, "aggressive_cleanup")
        # The algorithm should be taken from the preset's default
        expected_algo = str(
            DENOISE_PRESET_DEFAULTS["aggressive_cleanup"]["denoise_algorithm"]
        )
        self.assertEqual(cfg.denoise_algorithm, expected_algo)

    def test_denoise_preset_unknown_falls_back(self) -> None:
        cfg = _build({"denoise_preset": "nonexistent"})
        self.assertEqual(cfg.denoise_preset, DENOISE_DEFAULT_PRESET)

    # -- Classical preset handling ---

    def test_classical_preset_custom_preserved(self) -> None:
        cfg = _build({"classical_preset": "custom"})
        self.assertEqual(cfg.classical_preset, "custom")

    def test_classical_preset_valid_applied(self) -> None:
        cfg = _build({"classical_preset": "trust_point_cloud"})
        self.assertEqual(cfg.classical_preset, "trust_point_cloud")
        expected_depth = int(
            CLASSICAL_PRESET_DEFAULTS["trust_point_cloud"]["classical_poisson_depth"]
        )
        self.assertEqual(cfg.classical_poisson_depth, expected_depth)

    # -- Pi3X frame target clamped to max_frames ---

    def test_pi3x_frame_target_clamped_to_max_frames(self) -> None:
        cfg = _build({"max_frames": 10, "pi3x_frame_target": 999})
        self.assertEqual(cfg.max_frames, 10)
        self.assertEqual(cfg.pi3x_frame_target, 10)

    # -- Meshwrap lower-bound clamps ---

    def test_meshwrap_poisson_depth_min_6(self) -> None:
        cfg = _build({"meshwrap_poisson_depth": 2})
        self.assertEqual(cfg.meshwrap_poisson_depth, 6)

    def test_meshwrap_poisson_scale_min_1(self) -> None:
        cfg = _build({"meshwrap_poisson_scale": 0.5})
        self.assertEqual(cfg.meshwrap_poisson_scale, 1.0)

    # -- Mesh repair clamps ---

    def test_mesh_repair_clamps(self) -> None:
        cfg = _build(
            {
                "mesh_repair_max_diameter_ratio": 0.001,  # below 0.005
                "mesh_repair_y_band_ratio": 99.0,  # above 0.50
                "mesh_repair_smooth_iters": -5,  # below 0
            }
        )
        self.assertEqual(cfg.mesh_repair_max_diameter_ratio, 0.005)
        self.assertEqual(cfg.mesh_repair_y_band_ratio, 0.50)
        self.assertEqual(cfg.mesh_repair_smooth_iters, 0)

    # -- Env override: MESH_METHOD ---

    def test_env_override_mesh_method(self) -> None:
        cfg = _build({}, env={"MESH_METHOD": "diffcd"})
        self.assertEqual(cfg.mesh_method, "diffcd")

    # -- auto_accept ---

    def test_auto_accept_parsed_via_parse_bool(self) -> None:
        cfg_true = _build({"auto_accept": "yes"})
        self.assertTrue(cfg_true.auto_accept)

        cfg_false = _build({"auto_accept": "no"})
        self.assertFalse(cfg_false.auto_accept)

        cfg_default = _build({})
        self.assertFalse(cfg_default.auto_accept)

    # -- mesh_method ---

    def test_mesh_method_parse_choice(self) -> None:
        cfg_poisson = _build({"mesh_method": "poisson"})
        self.assertEqual(cfg_poisson.mesh_method, "poisson")

        cfg_diffcd = _build({"mesh_method": "diffcd"})
        self.assertEqual(cfg_diffcd.mesh_method, "diffcd")

        cfg_invalid = _build({"mesh_method": "invalid_method"})
        self.assertEqual(cfg_invalid.mesh_method, MESH_DEFAULT_METHOD)


# ===================================================================
# TestDenoisePresetDefaults
# ===================================================================


class TestDenoisePresetDefaults(unittest.TestCase):
    """2 tests for DENOISE_PRESET_DEFAULTS structure."""

    _EXPECTED_KEYS: set[str] = {
        "denoise_algorithm",
        "denoise_dbscan_eps",
        "denoise_dbscan_eps_ratio",
        "denoise_dbscan_min_samples",
        "denoise_dbscan_max_points",
        "denoise_sor_neighbors",
        "denoise_sor_std_ratio",
        "denoise_radius_neighbors",
        "denoise_radius_radius_ratio",
    }

    def test_all_presets_have_full_keys(self) -> None:
        for name, vals in DENOISE_PRESET_DEFAULTS.items():
            with self.subTest(preset=name):
                self.assertEqual(set(vals.keys()), self._EXPECTED_KEYS)

    def test_all_keys_have_denoise_prefix(self) -> None:
        for name, vals in DENOISE_PRESET_DEFAULTS.items():
            for key in vals:
                with self.subTest(preset=name, key=key):
                    self.assertTrue(
                        key.startswith("denoise_"),
                        f"Key {key!r} in preset {name!r} missing 'denoise_' prefix",
                    )


if __name__ == "__main__":
    unittest.main()
