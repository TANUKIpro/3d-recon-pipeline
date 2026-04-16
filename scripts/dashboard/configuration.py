"""Pipeline configuration parsing and normalization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from scripts.config_defaults import (
    COLMAP_IMAGE_SIZE,
    COLMAP_FILTER_ENABLED,
    COLMAP_FILTER_MASK_DILATE,
    COLMAP_FILTER_MIN_INSIDE_RATIO,
    COLMAP_FILTER_MIN_INSIDE_VIEWS,
    COLMAP_FILTER_MIN_POINTS,
    COLMAP_MAX_FEATURES,
    COLMAP_MATCHER,
    COLMAP_DSP_SIFT,
    COLMAP_FIRST_OCTAVE,
    COLMAP_USE_GPU,
    EXTRACT_FRAME_INTERVAL,
    EXTRACT_MAX_FRAMES,
    GROUND_PLANE_ENABLED,
    MILO_EXTRACTION_METHODS,
    MILO_MESH_CONFIGS,
    MILO_PRESET,
    MILO_PRESETS,
    MILO_RASTERIZERS,
    MILO_SCENE_TYPES,
    POST_TEXTURE_CLEANUP_ENABLED,
    CLEANUP_LOWER_HALF_THRESHOLD,
    UNTEXTURED_FILL_ENABLED,
    SAM2_DEFAULT_MODEL,
    TEXTURE_MODE,
    TEXTURE_MODES,
    TEXTURE_QUALITY_BOOST,
    TEXTURE_SIZE,
    TEXTURE_VIEW_ASSIGN_MODE,
    TEXTURE_VIEW_ASSIGN_MODES,
    _COLMAP_MATCHERS,
)
from scripts.dashboard.state import PipelineConfig
from scripts.milo_config import (
    MILO_CONFIG_FIELDS,
    MILO_INTERNAL_CONFIG_FIELDS,
    MILO_PUBLIC_CONFIG_FIELDS,
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


_MILO_ENV_FIELD_MAP: dict[str, str] = {
    "milo_iterations": "MILO_ITERATIONS",
    "milo_scene_type": "MILO_SCENE_TYPE",
    "milo_mesh_config": "MILO_MESH_CONFIG",
    "milo_extraction_method": "MILO_EXTRACTION_METHOD",
    "milo_use_masks": "MILO_USE_MASKS",
}
_MILO_PRESET_CHOICES = set(MILO_PRESETS) | {"custom"}


def _parse_milo_field_value(
    field_name: str,
    value: Any,
    fallback: Any,
) -> Any:
    if field_name == "milo_iterations":
        return max(1000, parse_int(value, int(fallback)))
    if field_name == "milo_scene_type":
        return parse_choice(value, MILO_SCENE_TYPES, str(fallback))
    if field_name == "milo_mesh_config":
        return parse_choice(value, MILO_MESH_CONFIGS, str(fallback))
    if field_name == "milo_extraction_method":
        return parse_choice(value, MILO_EXTRACTION_METHODS, str(fallback))
    if field_name == "milo_use_masks":
        return parse_bool(value, bool(fallback))
    if field_name == "milo_rasterizer":
        return parse_choice(value, MILO_RASTERIZERS, str(fallback))
    if field_name == "milo_dense_gaussians":
        return parse_bool(value, bool(fallback))
    if field_name == "milo_data_device":
        return parse_choice(value, {"cuda", "cpu"}, str(fallback))
    if field_name == "milo_mesh_res":
        return max(64, parse_int(value, int(fallback)))
    raise KeyError(f"Unknown milo field: {field_name}")


def _apply_milo_overrides(
    values: dict[str, Any],
    source: Mapping[str, Any],
    field_names: set[str] | tuple[str, ...],
) -> dict[str, Any]:
    for field_name in field_names:
        if field_name not in source:
            continue
        values[field_name] = _parse_milo_field_value(
            field_name,
            source.get(field_name),
            values[field_name],
        )
    return values


def _apply_milo_env_fallbacks(
    values: dict[str, Any],
    env_map: Mapping[str, str],
) -> dict[str, Any]:
    for field_name, env_name in _MILO_ENV_FIELD_MAP.items():
        if env_name not in env_map:
            continue
        values[field_name] = _parse_milo_field_value(
            field_name,
            env_map.get(env_name),
            values[field_name],
        )
    return values


def _is_legacy_fast_preset_payload(raw: Mapping[str, Any]) -> bool:
    if normalize_preset(raw.get("milo_preset"), fallback="") != "fast":
        return False
    if raw.get("milo_iterations") is not None and parse_int(raw.get("milo_iterations"), 7000) != 7000:
        return False
    if raw.get("milo_scene_type") is not None and parse_choice(raw.get("milo_scene_type"), MILO_SCENE_TYPES, "indoor") != "indoor":
        return False
    if raw.get("milo_mesh_config") is not None and parse_choice(raw.get("milo_mesh_config"), MILO_MESH_CONFIGS, "lowres") != "lowres":
        return False
    if raw.get("milo_extraction_method") is not None and parse_choice(raw.get("milo_extraction_method"), MILO_EXTRACTION_METHODS, "sdf") != "sdf":
        return False
    if raw.get("milo_use_masks") is not None and parse_bool(raw.get("milo_use_masks"), True) is not True:
        return False
    if raw.get("milo_rasterizer") is not None and parse_choice(raw.get("milo_rasterizer"), MILO_RASTERIZERS, "radegs") != "radegs":
        return False
    if raw.get("milo_dense_gaussians") is not None and parse_bool(raw.get("milo_dense_gaussians"), False) is not False:
        return False
    if raw.get("milo_data_device") is not None and parse_choice(raw.get("milo_data_device"), {"cuda", "cpu"}, "cuda") != "cuda":
        return False
    if raw.get("milo_mesh_res") is not None and parse_int(raw.get("milo_mesh_res"), 512) != 512:
        return False
    return True


def _resolve_milo_config(
    raw: Mapping[str, Any],
    env_map: Mapping[str, str],
    *,
    explicit_keys: set[str] | None,
) -> dict[str, Any]:
    if _is_legacy_fast_preset_payload(raw):
        values = config_fields_from_preset("fast")
        values["milo_preset"] = "fast"
        values["milo_preset_base"] = "fast"
        return values

    raw_preset = normalize_preset(raw.get("milo_preset"), fallback="")
    raw_preset_base = normalize_preset(raw.get("milo_preset_base"), fallback="")
    env_preset = normalize_preset(env_map.get("MILO_PRESET"), fallback="")
    preserve_all_raw = explicit_keys is None
    stage4_keys_present = {key for key in MILO_CONFIG_FIELDS if key in raw}

    if preserve_all_raw:
        selected_preset = raw_preset or env_preset or MILO_PRESET
        base_preset = raw_preset_base if raw_preset_base in MILO_PRESETS else ""
        if raw_preset in MILO_PRESETS:
            base_preset = raw_preset
        elif raw_preset == "custom" and base_preset not in MILO_PRESETS:
            inferred = infer_preset_from_public_fields(raw)
            base_preset = inferred if inferred in MILO_PRESETS else MILO_PRESET
        elif selected_preset not in MILO_PRESETS:
            inferred = infer_preset_from_public_fields(raw)
            base_preset = inferred if inferred in MILO_PRESETS else MILO_PRESET
        else:
            base_preset = selected_preset
        values = config_fields_from_preset(base_preset)
        values = _apply_milo_overrides(values, raw, MILO_CONFIG_FIELDS)
        if not stage4_keys_present and not raw_preset and not env_preset:
            values = _apply_milo_env_fallbacks(values, env_map)
        if raw_preset in MILO_PRESETS:
            final_preset = (
                raw_preset
                if fields_match_preset(values, raw_preset)
                else "custom"
            )
        elif raw_preset == "custom":
            final_preset = "custom"
        elif env_preset in MILO_PRESETS and not stage4_keys_present:
            final_preset = (
                env_preset
                if fields_match_preset(values, env_preset)
                else "custom"
            )
        else:
            final_preset = infer_preset_from_public_fields(values)
        values["milo_preset"] = final_preset
        values["milo_preset_base"] = (
            final_preset if final_preset in MILO_PRESETS else base_preset
        )
        return values

    explicit = set(explicit_keys)
    selected_preset = raw_preset or env_preset
    if selected_preset not in _MILO_PRESET_CHOICES:
        inferred = infer_preset_from_public_fields(raw)
        selected_preset = inferred if inferred in _MILO_PRESET_CHOICES else MILO_PRESET
    selected_base = raw_preset_base if raw_preset_base in MILO_PRESETS else ""
    if not selected_base:
        selected_base = selected_preset if selected_preset in MILO_PRESETS else MILO_PRESET

    if "milo_preset" in explicit and selected_preset in MILO_PRESETS:
        values = config_fields_from_preset(selected_preset)
    else:
        if selected_preset == "custom":
            base_preset = selected_base
        else:
            base_preset = selected_preset if selected_preset in MILO_PRESETS else MILO_PRESET
        values = config_fields_from_preset(base_preset)
        values = _apply_milo_overrides(values, raw, MILO_CONFIG_FIELDS)

    explicit_stage4_fields = {
        key for key in explicit if key in MILO_CONFIG_FIELDS
    }
    values = _apply_milo_overrides(values, raw, explicit_stage4_fields)

    if "milo_preset" in explicit and selected_preset in MILO_PRESETS:
        final_preset = (
            selected_preset
            if fields_match_preset(values, selected_preset)
            else "custom"
        )
    elif explicit_stage4_fields:
        final_preset = "custom"
    elif selected_preset in MILO_PRESETS and fields_match_preset(values, selected_preset):
        final_preset = selected_preset
    elif selected_preset == "custom":
        final_preset = "custom"
    else:
        final_preset = infer_preset_from_public_fields(values)

    values["milo_preset"] = final_preset
    values["milo_preset_base"] = (
        final_preset if final_preset in MILO_PRESETS else selected_base
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
    milo_cfg = _resolve_milo_config(
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
        colmap_filter_enabled=parse_bool(
            raw.get("colmap_filter_enabled"),
            env_bool("COLMAP_FILTER_ENABLED", COLMAP_FILTER_ENABLED, env_map),
        ),
        colmap_filter_min_inside_views=max(
            1,
            parse_int(
                raw.get("colmap_filter_min_inside_views"),
                env_int(
                    "COLMAP_FILTER_MIN_INSIDE_VIEWS",
                    COLMAP_FILTER_MIN_INSIDE_VIEWS,
                    env_map,
                ),
            ),
        ),
        colmap_filter_min_inside_ratio=max(
            0.0,
            min(
                1.0,
                parse_float(
                    raw.get("colmap_filter_min_inside_ratio"),
                    env_float(
                        "COLMAP_FILTER_MIN_INSIDE_RATIO",
                        COLMAP_FILTER_MIN_INSIDE_RATIO,
                        env_map,
                    ),
                ),
            ),
        ),
        colmap_filter_mask_dilate=max(
            0,
            parse_int(
                raw.get("colmap_filter_mask_dilate"),
                env_int(
                    "COLMAP_FILTER_MASK_DILATE",
                    COLMAP_FILTER_MASK_DILATE,
                    env_map,
                ),
            ),
        ),
        colmap_filter_min_points=max(
            1,
            parse_int(
                raw.get("colmap_filter_min_points"),
                env_int(
                    "COLMAP_FILTER_MIN_POINTS",
                    COLMAP_FILTER_MIN_POINTS,
                    env_map,
                ),
            ),
        ),
        milo_preset=str(milo_cfg["milo_preset"]),
        milo_preset_base=str(milo_cfg["milo_preset_base"]),
        milo_iterations=int(milo_cfg["milo_iterations"]),
        milo_scene_type=str(milo_cfg["milo_scene_type"]),
        milo_mesh_config=str(milo_cfg["milo_mesh_config"]),
        milo_extraction_method=str(milo_cfg["milo_extraction_method"]),
        milo_use_masks=bool(milo_cfg["milo_use_masks"]),
        milo_rasterizer=str(milo_cfg["milo_rasterizer"]),
        milo_dense_gaussians=bool(milo_cfg["milo_dense_gaussians"]),
        milo_data_device=str(milo_cfg["milo_data_device"]),
        milo_mesh_res=int(milo_cfg["milo_mesh_res"]),
        texture_mode=parse_choice(
            raw.get("texture_mode") or env_map.get("TEXTURE_MODE"),
            TEXTURE_MODES,
            TEXTURE_MODE,
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
        untextured_fill_enabled=parse_bool(
            raw.get("untextured_fill_enabled"),
            env_bool("UNTEXTURED_FILL_ENABLED", UNTEXTURED_FILL_ENABLED, env_map),
        ),
        ground_plane_enabled=parse_bool(
            raw.get("ground_plane_enabled"),
            env_bool("GROUND_PLANE_ENABLED", GROUND_PLANE_ENABLED, env_map),
        ),
        auto_accept=parse_bool(raw.get("auto_accept"), False),
    )
