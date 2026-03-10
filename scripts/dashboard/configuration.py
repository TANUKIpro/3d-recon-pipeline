"""Pipeline configuration parsing and normalization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from scripts.config_defaults import (
    CLASSICAL_DEFAULT_PRESET,
    CLASSICAL_PRESET_DEFAULTS,
    DENOISE_DEFAULT_PRESET,
    DENOISE_PRESET_DEFAULTS as _RAW_DENOISE_PRESETS,
    DIFFCD_BATCH_SIZE,
    DIFFCD_N_BATCHES,
    DIFFCD_RESOLUTION,
    EXTRACT_FRAME_INTERVAL,
    EXTRACT_MAX_FRAMES,
    MESH_DEFAULT_METHOD,
    MESH_METHODS,
    MESHWRAP_ALPHA_RATIO,
    MESHWRAP_CROP_SCALE,
    MESHWRAP_DENSITY_TRIM_Q,
    MESHWRAP_ITERATIONS,
    MESHWRAP_NORMAL_RADIUS_RATIO,
    MESHWRAP_OFFSET_RATIO,
    MESHWRAP_POISSON_DEPTH,
    MESHWRAP_POISSON_SCALE,
    MESHWRAP_QUALITY_THRESHOLD,
    MESHWRAP_SAMPLE_POINTS,
    MESHWRAP_SMOOTH_ITERATIONS,
    MESHWRAP_TARGET_FACE_RATIO,
    PI3X_CONFIDENCE_THRESHOLD,
    PI3X_EDGE_RTOL,
    PI3X_PIXEL_LIMIT,
    REPAIR_ENABLED,
    REPAIR_MAX_DIAMETER_RATIO,
    REPAIR_SMOOTH_ITERS,
    REPAIR_Y_BAND_RATIO,
    SAM2_DEFAULT_MODEL,
    TEXTURE_SIZE,
    _MESHWRAP_METHOD,
    _MESHWRAP_METHODS,
)
from scripts.config_defaults import DENOISE_ALGORITHMS  # re-export
from scripts.dashboard.state import PipelineConfig

# Map short preset keys → PipelineConfig field names.
_DENOISE_KEY_MAP: dict[str, str] = {
    "algorithm": "denoise_algorithm",
    "dbscan_eps": "denoise_dbscan_eps",
    "dbscan_eps_ratio": "denoise_dbscan_eps_ratio",
    "dbscan_min_samples": "denoise_dbscan_min_samples",
    "dbscan_max_points": "denoise_dbscan_max_points",
    "sor_neighbors": "denoise_sor_neighbors",
    "sor_std_ratio": "denoise_sor_std_ratio",
    "radius_neighbors": "denoise_radius_neighbors",
    "radius_ratio": "denoise_radius_radius_ratio",
}

# Build prefixed denoise preset dicts from the short-key canonical source.
DENOISE_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    name: {_DENOISE_KEY_MAP[k]: v for k, v in vals.items()}
    for name, vals in _RAW_DENOISE_PRESETS.items()
}


def parse_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def parse_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def parse_choice(value: Any, choices: set[str], fallback: str) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in choices else fallback


def parse_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "on", "y"}:
        return True
    if candidate in {"0", "false", "no", "off", "n"}:
        return False
    return fallback


def env_int(name: str, fallback: int, env: Mapping[str, str] | None = None) -> int:
    env_map = os.environ if env is None else env
    return parse_int(env_map.get(name, fallback), fallback)


def env_float(name: str, fallback: float, env: Mapping[str, str] | None = None) -> float:
    env_map = os.environ if env is None else env
    return parse_float(env_map.get(name, fallback), fallback)


def env_bool(name: str, fallback: bool, env: Mapping[str, str] | None = None) -> bool:
    env_map = os.environ if env is None else env
    return parse_bool(env_map.get(name), fallback)


def build_pipeline_config(
    raw: dict[str, Any],
    *,
    video_path: str,
    object_name: str,
    output_dir: Path,
    env: Mapping[str, str] | None = None,
) -> PipelineConfig:
    env_map = os.environ if env is None else env

    default_mesh_method = parse_choice(env_map.get("MESH_METHOD"), MESH_METHODS, MESH_DEFAULT_METHOD)

    raw_preset = str(raw.get("denoise_preset") or "").strip()
    if raw_preset == "custom":
        preset = "custom"
        denoise_defaults = DENOISE_PRESET_DEFAULTS[DENOISE_DEFAULT_PRESET]
    else:
        preset = parse_choice(raw_preset, set(DENOISE_PRESET_DEFAULTS), DENOISE_DEFAULT_PRESET)
        denoise_defaults = DENOISE_PRESET_DEFAULTS[preset]

    denoise_algorithm = parse_choice(
        raw.get("denoise_algorithm"),
        DENOISE_ALGORITHMS,
        str(denoise_defaults["denoise_algorithm"]),
    )

    # Classical mesh preset handling (similar to denoise)
    raw_classical = str(raw.get("classical_preset") or "").strip()
    if raw_classical == "custom":
        classical_preset = "custom"
        classical_defaults = CLASSICAL_PRESET_DEFAULTS[CLASSICAL_DEFAULT_PRESET]
    else:
        classical_preset = parse_choice(raw_classical, set(CLASSICAL_PRESET_DEFAULTS), CLASSICAL_DEFAULT_PRESET)
        classical_defaults = CLASSICAL_PRESET_DEFAULTS[classical_preset]

    max_frames = max(2, parse_int(raw.get("max_frames"), env_int("MAX_FRAMES", EXTRACT_MAX_FRAMES, env_map)))
    pi3x_frame_target = max(
        2,
        min(
            parse_int(raw.get("pi3x_frame_target"), max_frames),
            max_frames,
        ),
    )

    return PipelineConfig(
        video_path=video_path,
        output_dir=str(output_dir),
        object_name=object_name,
        frame_interval=parse_int(raw.get("frame_interval"), env_int("FRAME_INTERVAL", EXTRACT_FRAME_INTERVAL, env_map)),
        max_frames=max_frames,
        pixel_limit=parse_int(raw.get("pixel_limit"), env_int("PIXEL_LIMIT", PI3X_PIXEL_LIMIT, env_map)),
        pi3x_frame_target=pi3x_frame_target,
        confidence_threshold=parse_float(
            raw.get("confidence_threshold"),
            env_float("CONFIDENCE_THRESHOLD", PI3X_CONFIDENCE_THRESHOLD, env_map),
        ),
        edge_rtol=parse_float(raw.get("edge_rtol"), env_float("EDGE_RTOL", PI3X_EDGE_RTOL, env_map)),
        sam2_model=str(raw.get("sam2_model") or env_map.get("SAM2_MODEL", SAM2_DEFAULT_MODEL)),
        denoise_preset=preset,
        denoise_algorithm=denoise_algorithm,
        denoise_dbscan_eps=max(
            0.0,
            parse_float(raw.get("denoise_dbscan_eps"), float(denoise_defaults["denoise_dbscan_eps"])),
        ),
        denoise_dbscan_eps_ratio=max(
            0.0001,
            parse_float(
                raw.get("denoise_dbscan_eps_ratio"),
                float(denoise_defaults["denoise_dbscan_eps_ratio"]),
            ),
        ),
        denoise_dbscan_min_samples=max(
            1,
            parse_int(
                raw.get("denoise_dbscan_min_samples"),
                int(denoise_defaults["denoise_dbscan_min_samples"]),
            ),
        ),
        denoise_dbscan_max_points=max(
            1000,
            parse_int(
                raw.get("denoise_dbscan_max_points"),
                int(denoise_defaults["denoise_dbscan_max_points"]),
            ),
        ),
        denoise_sor_neighbors=max(
            2,
            parse_int(
                raw.get("denoise_sor_neighbors"),
                int(denoise_defaults["denoise_sor_neighbors"]),
            ),
        ),
        denoise_sor_std_ratio=max(
            0.1,
            parse_float(
                raw.get("denoise_sor_std_ratio"),
                float(denoise_defaults["denoise_sor_std_ratio"]),
            ),
        ),
        denoise_radius_neighbors=max(
            1,
            parse_int(
                raw.get("denoise_radius_neighbors"),
                int(denoise_defaults["denoise_radius_neighbors"]),
            ),
        ),
        denoise_radius_radius_ratio=max(
            0.0001,
            parse_float(
                raw.get("denoise_radius_radius_ratio"),
                float(denoise_defaults["denoise_radius_radius_ratio"]),
            ),
        ),
        mesh_method=parse_choice(raw.get("mesh_method"), MESH_METHODS, default_mesh_method),
        diffcd_batch_size=parse_int(raw.get("diffcd_batch_size"), env_int("DIFFCD_BATCH_SIZE", DIFFCD_BATCH_SIZE, env_map)),
        diffcd_n_batches=parse_int(raw.get("diffcd_n_batches"), env_int("DIFFCD_N_BATCHES", DIFFCD_N_BATCHES, env_map)),
        diffcd_resolution=parse_int(raw.get("diffcd_resolution"), env_int("DIFFCD_RESOLUTION", DIFFCD_RESOLUTION, env_map)),
        meshwrap_poisson_depth=max(
            6,
            parse_int(raw.get("meshwrap_poisson_depth"), MESHWRAP_POISSON_DEPTH),
        ),
        meshwrap_poisson_scale=max(
            1.0,
            parse_float(raw.get("meshwrap_poisson_scale"), MESHWRAP_POISSON_SCALE),
        ),
        meshwrap_density_trim_q=min(
            0.49,
            max(0.0, parse_float(raw.get("meshwrap_density_trim_q"), MESHWRAP_DENSITY_TRIM_Q)),
        ),
        meshwrap_target_face_ratio=min(
            3.0,
            max(0.2, parse_float(raw.get("meshwrap_target_face_ratio"), MESHWRAP_TARGET_FACE_RATIO)),
        ),
        meshwrap_iterations=max(
            1,
            parse_int(raw.get("meshwrap_iterations"), MESHWRAP_ITERATIONS),
        ),
        meshwrap_crop_scale=max(
            1.0,
            parse_float(raw.get("meshwrap_crop_scale"), MESHWRAP_CROP_SCALE),
        ),
        meshwrap_sample_points=max(
            50_000,
            parse_int(raw.get("meshwrap_sample_points"), MESHWRAP_SAMPLE_POINTS),
        ),
        meshwrap_normal_radius_ratio=max(
            0.001,
            parse_float(raw.get("meshwrap_normal_radius_ratio"), MESHWRAP_NORMAL_RADIUS_RATIO),
        ),
        meshwrap_smooth_iterations=max(
            0,
            min(10, parse_int(raw.get("meshwrap_smooth_iterations"), MESHWRAP_SMOOTH_ITERATIONS)),
        ),
        meshwrap_quality_threshold=max(
            0.0,
            min(1.0, parse_float(raw.get("meshwrap_quality_threshold"), MESHWRAP_QUALITY_THRESHOLD)),
        ),
        meshwrap_method=parse_choice(
            raw.get("meshwrap_method"),
            _MESHWRAP_METHODS,
            _MESHWRAP_METHOD,
        ),
        meshwrap_alpha_ratio=max(
            0.001,
            min(0.2, parse_float(raw.get("meshwrap_alpha_ratio"), MESHWRAP_ALPHA_RATIO)),
        ),
        meshwrap_offset_ratio=max(
            0.01,
            min(2.0, parse_float(raw.get("meshwrap_offset_ratio"), MESHWRAP_OFFSET_RATIO)),
        ),
        classical_preset=classical_preset,
        classical_preprocess_enabled=parse_bool(
            raw.get("classical_preprocess_enabled"),
            bool(classical_defaults["classical_preprocess_enabled"]),
        ),
        classical_poisson_depth=max(
            4,
            parse_int(raw.get("classical_poisson_depth"), int(classical_defaults["classical_poisson_depth"])),
        ),
        classical_density_trim_q=max(
            0.0,
            parse_float(raw.get("classical_density_trim_q"), float(classical_defaults["classical_density_trim_q"])),
        ),
        classical_auto_smooth=parse_bool(
            raw.get("classical_auto_smooth"),
            bool(classical_defaults["classical_auto_smooth"]),
        ),
        classical_smooth_iterations=max(
            0,
            parse_int(raw.get("classical_smooth_iterations"), int(classical_defaults["classical_smooth_iterations"])),
        ),
        classical_downsample_enabled=parse_bool(
            raw.get("classical_downsample_enabled"),
            bool(classical_defaults["classical_downsample_enabled"]),
        ),
        classical_downsample_target_faces=max(
            10_000,
            parse_int(raw.get("classical_downsample_target_faces"), int(classical_defaults["classical_downsample_target_faces"])),
        ),
        mesh_repair_enabled=parse_bool(
            raw.get("mesh_repair_enabled"),
            env_bool("MESH_REPAIR_ENABLED", REPAIR_ENABLED, env_map),
        ),
        mesh_repair_max_diameter_ratio=max(
            0.005,
            min(
                1.50,
                parse_float(
                    raw.get("mesh_repair_max_diameter_ratio"),
                    env_float("MESH_REPAIR_MAX_DIAMETER_RATIO", REPAIR_MAX_DIAMETER_RATIO, env_map),
                ),
            ),
        ),
        mesh_repair_y_band_ratio=max(
            0.005,
            min(
                0.50,
                parse_float(
                    raw.get("mesh_repair_y_band_ratio"),
                    env_float("MESH_REPAIR_Y_BAND_RATIO", REPAIR_Y_BAND_RATIO, env_map),
                ),
            ),
        ),
        mesh_repair_smooth_iters=max(
            0,
            min(
                12,
                parse_int(
                    raw.get("mesh_repair_smooth_iters"),
                    env_int("MESH_REPAIR_SMOOTH_ITERS", REPAIR_SMOOTH_ITERS, env_map),
                ),
            ),
        ),
        texture_size=parse_int(raw.get("texture_size"), env_int("TEXTURE_SIZE", TEXTURE_SIZE, env_map)),
        auto_accept=parse_bool(raw.get("auto_accept"), False),
    )
