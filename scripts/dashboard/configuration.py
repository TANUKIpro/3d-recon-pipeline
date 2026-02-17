"""Pipeline configuration parsing and normalization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from scripts.dashboard.state import PipelineConfig

DENOISE_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "balanced": {
        "denoise_algorithm": "dbscan_sor",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.02,
        "denoise_dbscan_min_samples": 10,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 8,
        "denoise_radius_radius_ratio": 0.015,
    },
    "detail_preserving": {
        "denoise_algorithm": "sor_only",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.02,
        "denoise_dbscan_min_samples": 10,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 16,
        "denoise_sor_std_ratio": 2.6,
        "denoise_radius_neighbors": 8,
        "denoise_radius_radius_ratio": 0.015,
    },
    "isolate_subject": {
        "denoise_algorithm": "dbscan_only",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.018,
        "denoise_dbscan_min_samples": 8,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 8,
        "denoise_radius_radius_ratio": 0.015,
    },
    "sparse_noise": {
        "denoise_algorithm": "radius_only",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.02,
        "denoise_dbscan_min_samples": 10,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 6,
        "denoise_radius_radius_ratio": 0.012,
    },
    "aggressive_cleanup": {
        "denoise_algorithm": "dbscan_radius",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.024,
        "denoise_dbscan_min_samples": 14,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 10,
        "denoise_radius_radius_ratio": 0.02,
    },
}

DENOISE_ALGORITHMS = {
    "dbscan_sor",
    "dbscan_only",
    "sor_only",
    "radius_only",
    "dbscan_radius",
}

MESH_METHODS = {"poisson", "diffcd"}


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

    default_mesh_method = parse_choice(env_map.get("MESH_METHOD"), MESH_METHODS, "poisson")

    raw_preset = str(raw.get("denoise_preset") or "").strip()
    if raw_preset == "custom":
        preset = "custom"
        denoise_defaults = DENOISE_PRESET_DEFAULTS["balanced"]
    else:
        preset = parse_choice(raw_preset, set(DENOISE_PRESET_DEFAULTS), "balanced")
        denoise_defaults = DENOISE_PRESET_DEFAULTS[preset]

    denoise_algorithm = parse_choice(
        raw.get("denoise_algorithm"),
        DENOISE_ALGORITHMS,
        str(denoise_defaults["denoise_algorithm"]),
    )

    max_frames = max(2, parse_int(raw.get("max_frames"), env_int("MAX_FRAMES", 50, env_map)))
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
        frame_interval=parse_int(raw.get("frame_interval"), env_int("FRAME_INTERVAL", 10, env_map)),
        max_frames=max_frames,
        pixel_limit=parse_int(raw.get("pixel_limit"), env_int("PIXEL_LIMIT", 255000, env_map)),
        pi3x_frame_target=pi3x_frame_target,
        confidence_threshold=parse_float(
            raw.get("confidence_threshold"),
            env_float("CONFIDENCE_THRESHOLD", 0.2, env_map),
        ),
        edge_rtol=parse_float(raw.get("edge_rtol"), env_float("EDGE_RTOL", 0.03, env_map)),
        sam2_model=str(raw.get("sam2_model") or env_map.get("SAM2_MODEL", "large")),
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
        diffcd_batch_size=parse_int(raw.get("diffcd_batch_size"), env_int("DIFFCD_BATCH_SIZE", 5000, env_map)),
        diffcd_n_batches=parse_int(raw.get("diffcd_n_batches"), env_int("DIFFCD_N_BATCHES", 30000, env_map)),
        diffcd_resolution=parse_int(raw.get("diffcd_resolution"), env_int("DIFFCD_RESOLUTION", 512, env_map)),
        meshwrap_poisson_depth=max(
            6,
            parse_int(raw.get("meshwrap_poisson_depth"), 6),
        ),
        meshwrap_poisson_scale=max(
            1.0,
            parse_float(raw.get("meshwrap_poisson_scale"), 1.18),
        ),
        meshwrap_density_trim_q=min(
            0.49,
            max(0.0, parse_float(raw.get("meshwrap_density_trim_q"), 0.06)),
        ),
        meshwrap_target_face_ratio=min(
            3.0,
            max(0.2, parse_float(raw.get("meshwrap_target_face_ratio"), 1.80)),
        ),
        meshwrap_iterations=max(
            1,
            parse_int(raw.get("meshwrap_iterations"), 2),
        ),
        meshwrap_crop_scale=max(
            1.0,
            parse_float(raw.get("meshwrap_crop_scale"), 2.0),
        ),
        meshwrap_sample_points=max(
            50_000,
            parse_int(raw.get("meshwrap_sample_points"), 180_000),
        ),
        meshwrap_normal_radius_ratio=max(
            0.001,
            parse_float(raw.get("meshwrap_normal_radius_ratio"), 0.035),
        ),
        mesh_repair_enabled=parse_bool(
            raw.get("mesh_repair_enabled"),
            env_bool("MESH_REPAIR_ENABLED", True, env_map),
        ),
        mesh_repair_max_diameter_ratio=max(
            0.005,
            min(
                1.50,
                parse_float(
                    raw.get("mesh_repair_max_diameter_ratio"),
                    env_float("MESH_REPAIR_MAX_DIAMETER_RATIO", 0.08, env_map),
                ),
            ),
        ),
        mesh_repair_y_band_ratio=max(
            0.005,
            min(
                0.50,
                parse_float(
                    raw.get("mesh_repair_y_band_ratio"),
                    env_float("MESH_REPAIR_Y_BAND_RATIO", 0.06, env_map),
                ),
            ),
        ),
        mesh_repair_smooth_iters=max(
            0,
            min(
                12,
                parse_int(
                    raw.get("mesh_repair_smooth_iters"),
                    env_int("MESH_REPAIR_SMOOTH_ITERS", 2, env_map),
                ),
            ),
        ),
        texture_size=parse_int(raw.get("texture_size"), env_int("TEXTURE_SIZE", 0, env_map)),
    )
