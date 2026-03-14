"""Pipeline configuration parsing and normalization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from scripts.config_defaults import (
    COLMAP_IMAGE_SIZE,
    COLMAP_MAX_FEATURES,
    COLMAP_MATCHER,
    EXTRACT_FRAME_INTERVAL,
    EXTRACT_MAX_FRAMES,
    GS2MESH_GS_ITERATIONS,
    GS2MESH_RUNTIME_PROFILE,
    GS2MESH_RUNTIME_PROFILES,
    GS2MESH_STEREO_MODEL,
    GS2MESH_TSDF_DEPTH_TRUNC,
    GS2MESH_TSDF_VOXEL_SIZE,
    GS2MESH_USE_MASKS,
    GROUND_PLANE_ENABLED,
    SAM2_DEFAULT_MODEL,
    TEXTURE_QUALITY_BOOST,
    TEXTURE_SIZE,
    TEXTURE_VIEW_ASSIGN_MODE,
    TEXTURE_VIEW_ASSIGN_MODES,
    _COLMAP_MATCHERS,
)
from scripts.dashboard.state import PipelineConfig


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

    max_frames = max(2, parse_int(raw.get("max_frames"), env_int("MAX_FRAMES", EXTRACT_MAX_FRAMES, env_map)))

    return PipelineConfig(
        video_path=video_path,
        output_dir=str(output_dir),
        object_name=object_name,
        frame_interval=parse_int(raw.get("frame_interval"), env_int("FRAME_INTERVAL", EXTRACT_FRAME_INTERVAL, env_map)),
        max_frames=max_frames,
        sam2_model=str(raw.get("sam2_model") or env_map.get("SAM2_MODEL", SAM2_DEFAULT_MODEL)),
        colmap_matcher=parse_choice(
            raw.get("colmap_matcher"),
            _COLMAP_MATCHERS,
            str(env_map.get("COLMAP_MATCHER", COLMAP_MATCHER)),
        ),
        colmap_max_features=max(
            1000,
            parse_int(raw.get("colmap_max_features"), env_int("COLMAP_MAX_FEATURES", COLMAP_MAX_FEATURES, env_map)),
        ),
        colmap_image_size=max(
            256,
            parse_int(raw.get("colmap_image_size"), env_int("COLMAP_IMAGE_SIZE", COLMAP_IMAGE_SIZE, env_map)),
        ),
        gs2mesh_gs_iterations=max(
            1000,
            parse_int(raw.get("gs2mesh_gs_iterations"), env_int("GS2MESH_GS_ITERATIONS", GS2MESH_GS_ITERATIONS, env_map)),
        ),
        gs2mesh_runtime_profile=parse_choice(
            raw.get("gs2mesh_runtime_profile") or env_map.get("GS2MESH_RUNTIME_PROFILE"),
            GS2MESH_RUNTIME_PROFILES,
            GS2MESH_RUNTIME_PROFILE,
        ),
        gs2mesh_stereo_model=str(raw.get("gs2mesh_stereo_model") or env_map.get("GS2MESH_STEREO_MODEL", GS2MESH_STEREO_MODEL)),
        gs2mesh_tsdf_voxel_size=max(
            0.001,
            parse_float(raw.get("gs2mesh_tsdf_voxel_size"), env_float("GS2MESH_TSDF_VOXEL_SIZE", GS2MESH_TSDF_VOXEL_SIZE, env_map)),
        ),
        gs2mesh_tsdf_depth_trunc=max(
            0.005,
            parse_float(raw.get("gs2mesh_tsdf_depth_trunc"), env_float("GS2MESH_TSDF_DEPTH_TRUNC", GS2MESH_TSDF_DEPTH_TRUNC, env_map)),
        ),
        gs2mesh_use_masks=parse_bool(
            raw.get("gs2mesh_use_masks"),
            env_bool("GS2MESH_USE_MASKS", GS2MESH_USE_MASKS, env_map),
        ),
        texture_size=parse_int(raw.get("texture_size"), env_int("TEXTURE_SIZE", TEXTURE_SIZE, env_map)),
        texture_view_assign_mode=parse_choice(
            raw.get("texture_view_assign_mode") or env_map.get("TEXTURE_VIEW_ASSIGN_MODE"),
            TEXTURE_VIEW_ASSIGN_MODES,
            TEXTURE_VIEW_ASSIGN_MODE,
        ),
        texture_quality_boost=parse_bool(
            raw.get("texture_quality_boost"),
            env_bool("TEXTURE_QUALITY_BOOST", TEXTURE_QUALITY_BOOST, env_map),
        ),
        ground_plane_enabled=parse_bool(
            raw.get("ground_plane_enabled"),
            env_bool("GROUND_PLANE_ENABLED", GROUND_PLANE_ENABLED, env_map),
        ),
        auto_accept=parse_bool(raw.get("auto_accept"), False),
    )
