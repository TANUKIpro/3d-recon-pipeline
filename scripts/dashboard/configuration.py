"""Pipeline configuration parsing and normalization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from scripts.config_defaults import (
    COLMAP_IMAGE_SIZE,
    COLMAP_MAX_FEATURES,
    COLMAP_MATCHER,
    COLMAP_DSP_SIFT,
    COLMAP_FIRST_OCTAVE,
    COLMAP_USE_GPU,
    EXTRACT_FRAME_INTERVAL,
    EXTRACT_MAX_FRAMES,
    GS2MESH_MASK_DEPTH_MODES,
    GS2MESH_PRESET,
    GS2MESH_PRESETS,
    GS2MESH_GS_ITERATIONS,
    GS2MESH_RUNTIME_PROFILE,
    GS2MESH_RUNTIME_PROFILES,
    GS2MESH_STEREO_MODEL,
    GS2MESH_TSDF_BLOCK_COUNT,
    GS2MESH_TSDF_CLEANING_THRESHOLD,
    GS2MESH_TSDF_CLOSING_KERNEL_SIZE,
    GS2MESH_TSDF_DEPTH_TRUNC,
    GS2MESH_TSDF_DILATE,
    GS2MESH_TSDF_ERODE_MASK,
    GS2MESH_TSDF_EROSION_KERNEL_SIZE,
    GS2MESH_TSDF_INVERT_MASK,
    GS2MESH_TSDF_MAX_DEPTH_BASELINES,
    GS2MESH_TSDF_MIN_DEPTH_BASELINES,
    GS2MESH_TSDF_SCALE,
    GS2MESH_TSDF_USE_OCCLUSION_MASK,
    GS2MESH_TSDF_VOXEL_SIZE,
    GS2MESH_USE_MASKS,
    GROUND_PLANE_ENABLED,
    POST_TEXTURE_CLEANUP_ENABLED,
    CLEANUP_LOWER_HALF_THRESHOLD,
    SAM2_DEFAULT_MODEL,
    TEXTURE_QUALITY_BOOST,
    TEXTURE_SIZE,
    TEXTURE_VIEW_ASSIGN_MODE,
    TEXTURE_VIEW_ASSIGN_MODES,
    _COLMAP_MATCHERS,
)
from scripts.dashboard.state import PipelineConfig
from scripts.gs2mesh_config import (
    GS2MESH_CONFIG_FIELDS,
    GS2MESH_INTERNAL_CONFIG_FIELDS,
    GS2MESH_PRESET_CONFIG_FIELDS,
    GS2MESH_PUBLIC_CONFIG_FIELDS,
    config_fields_from_preset,
    fields_match_preset,
    infer_preset_from_public_fields,
    normalize_preset,
)


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


_GS2MESH_ENV_FIELD_MAP: dict[str, str] = {
    "gs2mesh_gs_iterations": "GS2MESH_GS_ITERATIONS",
    "gs2mesh_runtime_profile": "GS2MESH_RUNTIME_PROFILE",
    "gs2mesh_stereo_model": "GS2MESH_STEREO_MODEL",
    "gs2mesh_tsdf_voxel_size": "GS2MESH_TSDF_VOXEL_SIZE",
    "gs2mesh_tsdf_depth_trunc": "GS2MESH_TSDF_DEPTH_TRUNC",
    "gs2mesh_use_masks": "GS2MESH_USE_MASKS",
    "gs2mesh_mask_depth_mode": "GS2MESH_MASK_DEPTH_MODE",
}
_GS2MESH_PRESET_CHOICES = set(GS2MESH_PRESETS) | {"custom"}


def _parse_gs2mesh_field_value(
    field_name: str,
    value: Any,
    fallback: Any,
) -> Any:
    if field_name == "gs2mesh_gs_iterations":
        return max(1000, parse_int(value, int(fallback)))
    if field_name == "gs2mesh_runtime_profile":
        return parse_choice(value, GS2MESH_RUNTIME_PROFILES, str(fallback))
    if field_name == "gs2mesh_stereo_model":
        candidate = str(value or fallback).strip()
        if candidate == "DLNR":
            return "DLNR_Middlebury"
        return candidate or str(fallback)
    if field_name == "gs2mesh_tsdf_voxel_size":
        return max(0.001, parse_float(value, float(fallback)))
    if field_name == "gs2mesh_tsdf_depth_trunc":
        return max(0.005, parse_float(value, float(fallback)))
    if field_name == "gs2mesh_use_masks":
        return parse_bool(value, bool(fallback))
    if field_name == "gs2mesh_mask_depth_mode":
        return parse_choice(value, GS2MESH_MASK_DEPTH_MODES, str(fallback))
    if field_name == "gs2mesh_tsdf_scale":
        return max(1e-6, parse_float(value, float(fallback)))
    if field_name in {
        "gs2mesh_tsdf_min_depth_baselines",
        "gs2mesh_tsdf_max_depth_baselines",
        "gs2mesh_tsdf_dilate",
        "gs2mesh_tsdf_erosion_kernel_size",
        "gs2mesh_tsdf_closing_kernel_size",
        "gs2mesh_tsdf_block_count",
    }:
        return max(1, parse_int(value, int(fallback)))
    if field_name == "gs2mesh_tsdf_cleaning_threshold":
        return max(0, parse_int(value, int(fallback)))
    if field_name in GS2MESH_INTERNAL_CONFIG_FIELDS:
        return parse_bool(value, bool(fallback))
    raise KeyError(f"Unknown gs2mesh field: {field_name}")


def _apply_gs2mesh_overrides(
    values: dict[str, Any],
    source: Mapping[str, Any],
    field_names: set[str] | tuple[str, ...],
) -> dict[str, Any]:
    for field_name in field_names:
        if field_name not in source:
            continue
        values[field_name] = _parse_gs2mesh_field_value(
            field_name,
            source.get(field_name),
            values[field_name],
        )
    values["gs2mesh_tsdf_max_depth_baselines"] = max(
        int(values["gs2mesh_tsdf_min_depth_baselines"]),
        int(values["gs2mesh_tsdf_max_depth_baselines"]),
    )
    return values


def _apply_gs2mesh_env_fallbacks(
    values: dict[str, Any],
    env_map: Mapping[str, str],
) -> dict[str, Any]:
    for field_name, env_name in _GS2MESH_ENV_FIELD_MAP.items():
        if env_name not in env_map:
            continue
        values[field_name] = _parse_gs2mesh_field_value(
            field_name,
            env_map.get(env_name),
            values[field_name],
        )
    if "GS2MESH_MASK_DEPTH_MODE" not in env_map:
        legacy_mode = env_map.get("GS2MESH_SILHOUETTE_DEPTH_MODE")
        if legacy_mode:
            values["gs2mesh_mask_depth_mode"] = _parse_gs2mesh_field_value(
                "gs2mesh_mask_depth_mode",
                legacy_mode,
                values["gs2mesh_mask_depth_mode"],
            )
        elif parse_bool(env_map.get("GS2MESH_SILHOUETTE_FILL"), False):
            values["gs2mesh_mask_depth_mode"] = "fill"
    return values


def _resolve_gs2mesh_config(
    raw: Mapping[str, Any],
    env_map: Mapping[str, str],
    *,
    explicit_keys: set[str] | None,
) -> dict[str, Any]:
    raw_preset = normalize_preset(raw.get("gs2mesh_preset"), fallback="")
    raw_preset_base = normalize_preset(raw.get("gs2mesh_preset_base"), fallback="")
    env_preset = normalize_preset(env_map.get("GS2MESH_PRESET"), fallback="")
    preserve_all_raw = explicit_keys is None
    stage4_keys_present = {key for key in GS2MESH_CONFIG_FIELDS if key in raw}

    if preserve_all_raw:
        selected_preset = raw_preset or env_preset or GS2MESH_PRESET
        base_preset = raw_preset_base if raw_preset_base in GS2MESH_PRESETS else ""
        if raw_preset in GS2MESH_PRESETS:
            base_preset = raw_preset
        elif raw_preset == "custom" and base_preset not in GS2MESH_PRESETS:
            inferred = infer_preset_from_public_fields(raw)
            base_preset = inferred if inferred in GS2MESH_PRESETS else GS2MESH_PRESET
        elif selected_preset not in GS2MESH_PRESETS:
            inferred = infer_preset_from_public_fields(raw)
            base_preset = inferred if inferred in GS2MESH_PRESETS else GS2MESH_PRESET
        else:
            base_preset = selected_preset
        values = config_fields_from_preset(base_preset)
        values = _apply_gs2mesh_overrides(values, raw, GS2MESH_CONFIG_FIELDS)
        if not stage4_keys_present and not raw_preset and not env_preset:
            values = _apply_gs2mesh_env_fallbacks(values, env_map)
        if raw_preset in GS2MESH_PRESETS:
            final_preset = (
                raw_preset
                if fields_match_preset(values, raw_preset)
                else "custom"
            )
        elif raw_preset == "custom":
            final_preset = "custom"
        elif env_preset in GS2MESH_PRESETS and not stage4_keys_present:
            final_preset = (
                env_preset
                if fields_match_preset(values, env_preset)
                else "custom"
            )
        else:
            final_preset = infer_preset_from_public_fields(values)
        values["gs2mesh_preset"] = final_preset
        values["gs2mesh_preset_base"] = (
            final_preset if final_preset in GS2MESH_PRESETS else base_preset
        )
        return values

    explicit = set(explicit_keys)
    selected_preset = raw_preset or env_preset
    if selected_preset not in _GS2MESH_PRESET_CHOICES:
        inferred = infer_preset_from_public_fields(raw)
        selected_preset = inferred if inferred in _GS2MESH_PRESET_CHOICES else GS2MESH_PRESET
    selected_base = raw_preset_base if raw_preset_base in GS2MESH_PRESETS else ""
    if not selected_base:
        selected_base = selected_preset if selected_preset in GS2MESH_PRESETS else GS2MESH_PRESET

    if "gs2mesh_preset" in explicit and selected_preset in GS2MESH_PRESETS:
        values = config_fields_from_preset(selected_preset)
    else:
        if selected_preset == "custom":
            base_preset = selected_base
        else:
            base_preset = selected_preset if selected_preset in GS2MESH_PRESETS else GS2MESH_PRESET
        values = config_fields_from_preset(base_preset)
        values = _apply_gs2mesh_overrides(values, raw, GS2MESH_CONFIG_FIELDS)

    explicit_stage4_fields = {
        key for key in explicit if key in GS2MESH_CONFIG_FIELDS
    }
    values = _apply_gs2mesh_overrides(values, raw, explicit_stage4_fields)
    explicit_preset_fields = {
        key for key in explicit if key in GS2MESH_PRESET_CONFIG_FIELDS
    }

    if "gs2mesh_preset" in explicit and selected_preset in GS2MESH_PRESETS:
        final_preset = (
            selected_preset
            if fields_match_preset(values, selected_preset)
            else "custom"
        )
    elif explicit_preset_fields:
        final_preset = "custom"
    elif selected_preset in GS2MESH_PRESETS and fields_match_preset(values, selected_preset):
        final_preset = selected_preset
    elif selected_preset == "custom":
        final_preset = "custom"
    else:
        final_preset = infer_preset_from_public_fields(values)

    values["gs2mesh_preset"] = final_preset
    values["gs2mesh_preset_base"] = (
        final_preset if final_preset in GS2MESH_PRESETS else selected_base
    )
    return values


def build_pipeline_config(
    raw: dict[str, Any],
    *,
    video_path: str,
    object_name: str,
    output_dir: Path,
    env: Mapping[str, str] | None = None,
    explicit_keys: set[str] | None = None,
) -> PipelineConfig:
    env_map = os.environ if env is None else env

    max_frames = max(2, parse_int(raw.get("max_frames"), env_int("MAX_FRAMES", EXTRACT_MAX_FRAMES, env_map)))
    gs2mesh_cfg = _resolve_gs2mesh_config(
        raw,
        env_map,
        explicit_keys=explicit_keys,
    )

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
        colmap_use_gpu=parse_bool(
            raw.get("colmap_use_gpu"),
            env_bool("COLMAP_USE_GPU", COLMAP_USE_GPU, env_map),
        ),
        colmap_dsp_sift=parse_bool(
            raw.get("colmap_dsp_sift"),
            env_bool("COLMAP_DSP_SIFT", COLMAP_DSP_SIFT, env_map),
        ),
        colmap_first_octave=parse_int(
            raw.get("colmap_first_octave"),
            env_int("COLMAP_FIRST_OCTAVE", COLMAP_FIRST_OCTAVE, env_map),
        ),
        gs2mesh_preset=str(gs2mesh_cfg["gs2mesh_preset"]),
        gs2mesh_preset_base=str(gs2mesh_cfg["gs2mesh_preset_base"]),
        gs2mesh_gs_iterations=int(gs2mesh_cfg["gs2mesh_gs_iterations"]),
        gs2mesh_runtime_profile=str(gs2mesh_cfg["gs2mesh_runtime_profile"]),
        gs2mesh_stereo_model=str(gs2mesh_cfg["gs2mesh_stereo_model"]),
        gs2mesh_tsdf_voxel_size=float(gs2mesh_cfg["gs2mesh_tsdf_voxel_size"]),
        gs2mesh_tsdf_depth_trunc=float(gs2mesh_cfg["gs2mesh_tsdf_depth_trunc"]),
        gs2mesh_use_masks=bool(gs2mesh_cfg["gs2mesh_use_masks"]),
        gs2mesh_mask_depth_mode=str(gs2mesh_cfg["gs2mesh_mask_depth_mode"]),
        gs2mesh_tsdf_scale=float(gs2mesh_cfg["gs2mesh_tsdf_scale"]),
        gs2mesh_tsdf_min_depth_baselines=int(gs2mesh_cfg["gs2mesh_tsdf_min_depth_baselines"]),
        gs2mesh_tsdf_max_depth_baselines=int(gs2mesh_cfg["gs2mesh_tsdf_max_depth_baselines"]),
        gs2mesh_tsdf_dilate=int(gs2mesh_cfg["gs2mesh_tsdf_dilate"]),
        gs2mesh_tsdf_cleaning_threshold=int(gs2mesh_cfg["gs2mesh_tsdf_cleaning_threshold"]),
        gs2mesh_tsdf_use_occlusion_mask=bool(gs2mesh_cfg["gs2mesh_tsdf_use_occlusion_mask"]),
        gs2mesh_tsdf_invert_mask=bool(gs2mesh_cfg["gs2mesh_tsdf_invert_mask"]),
        gs2mesh_tsdf_erode_mask=bool(gs2mesh_cfg["gs2mesh_tsdf_erode_mask"]),
        gs2mesh_tsdf_erosion_kernel_size=int(gs2mesh_cfg["gs2mesh_tsdf_erosion_kernel_size"]),
        gs2mesh_tsdf_closing_kernel_size=int(gs2mesh_cfg["gs2mesh_tsdf_closing_kernel_size"]),
        gs2mesh_tsdf_block_count=int(gs2mesh_cfg["gs2mesh_tsdf_block_count"]),
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
        post_texture_cleanup_enabled=parse_bool(
            raw.get("post_texture_cleanup_enabled"),
            env_bool("POST_TEXTURE_CLEANUP_ENABLED", POST_TEXTURE_CLEANUP_ENABLED, env_map),
        ),
        cleanup_lower_half_threshold=max(
            0.0,
            min(
                1.0,
                parse_float(
                    raw.get("cleanup_lower_half_threshold"),
                    env_float("CLEANUP_LOWER_HALF_THRESHOLD", CLEANUP_LOWER_HALF_THRESHOLD, env_map),
                ),
            ),
        ),
        ground_plane_enabled=parse_bool(
            raw.get("ground_plane_enabled"),
            env_bool("GROUND_PLANE_ENABLED", GROUND_PLANE_ENABLED, env_map),
        ),
        auto_accept=parse_bool(raw.get("auto_accept"), False),
    )
