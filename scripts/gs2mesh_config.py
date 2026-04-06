"""Shared gs2mesh preset metadata and runtime settings helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scripts.config_defaults import (
    GS2MESH_PRESET,
    GS2MESH_PRESET_CHOICES,
    GS2MESH_PRESET_CUSTOM,
    GS2MESH_PRESETS,
)

GS2MESH_PUBLIC_CONFIG_FIELDS: tuple[str, ...] = (
    "gs2mesh_gs_iterations",
    "gs2mesh_runtime_profile",
    "gs2mesh_stereo_model",
    "gs2mesh_tsdf_voxel_size",
    "gs2mesh_tsdf_depth_trunc",
    "gs2mesh_use_masks",
)

GS2MESH_INTERNAL_CONFIG_FIELDS: tuple[str, ...] = (
    "gs2mesh_tsdf_scale",
    "gs2mesh_tsdf_min_depth_baselines",
    "gs2mesh_tsdf_max_depth_baselines",
    "gs2mesh_tsdf_dilate",
    "gs2mesh_tsdf_cleaning_threshold",
    "gs2mesh_tsdf_use_occlusion_mask",
    "gs2mesh_tsdf_invert_mask",
    "gs2mesh_tsdf_erode_mask",
    "gs2mesh_tsdf_erosion_kernel_size",
    "gs2mesh_tsdf_closing_kernel_size",
    "gs2mesh_tsdf_block_count",
)

GS2MESH_CONFIG_FIELDS: tuple[str, ...] = (
    *GS2MESH_PUBLIC_CONFIG_FIELDS,
    *GS2MESH_INTERNAL_CONFIG_FIELDS,
)

_SETTING_TO_CONFIG_FIELD = {
    "gs_iterations": "gs2mesh_gs_iterations",
    "runtime_profile": "gs2mesh_runtime_profile",
    "stereo_model": "gs2mesh_stereo_model",
    "tsdf_voxel_size": "gs2mesh_tsdf_voxel_size",
    "tsdf_depth_trunc": "gs2mesh_tsdf_depth_trunc",
    "use_masks": "gs2mesh_use_masks",
    "tsdf_scale": "gs2mesh_tsdf_scale",
    "tsdf_min_depth_baselines": "gs2mesh_tsdf_min_depth_baselines",
    "tsdf_max_depth_baselines": "gs2mesh_tsdf_max_depth_baselines",
    "tsdf_dilate": "gs2mesh_tsdf_dilate",
    "tsdf_cleaning_threshold": "gs2mesh_tsdf_cleaning_threshold",
    "tsdf_use_occlusion_mask": "gs2mesh_tsdf_use_occlusion_mask",
    "tsdf_invert_mask": "gs2mesh_tsdf_invert_mask",
    "tsdf_erode_mask": "gs2mesh_tsdf_erode_mask",
    "tsdf_erosion_kernel_size": "gs2mesh_tsdf_erosion_kernel_size",
    "tsdf_closing_kernel_size": "gs2mesh_tsdf_closing_kernel_size",
    "block_count": "gs2mesh_tsdf_block_count",
}
_CONFIG_TO_SETTING_FIELD = {
    config_name: setting_name
    for setting_name, config_name in _SETTING_TO_CONFIG_FIELD.items()
}


@dataclass(frozen=True)
class Gs2meshSettings:
    gs_iterations: int
    runtime_profile: str
    stereo_model: str
    tsdf_voxel_size: float
    tsdf_depth_trunc: float
    use_masks: bool
    tsdf_scale: float
    tsdf_min_depth_baselines: int
    tsdf_max_depth_baselines: int
    tsdf_dilate: int
    tsdf_cleaning_threshold: int
    tsdf_use_occlusion_mask: bool
    tsdf_invert_mask: bool
    tsdf_erode_mask: bool
    tsdf_erosion_kernel_size: int
    tsdf_closing_kernel_size: int
    block_count: int

    @classmethod
    def from_preset(cls, preset: str = GS2MESH_PRESET) -> Gs2meshSettings:
        resolved = preset if preset in GS2MESH_PRESETS else GS2MESH_PRESET
        return cls(**GS2MESH_PRESETS[resolved])

    @classmethod
    def from_config_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        preset: str = GS2MESH_PRESET,
    ) -> Gs2meshSettings:
        data = dict(config_fields_from_preset(preset))
        for config_field, setting_field in _CONFIG_TO_SETTING_FIELD.items():
            if config_field in values:
                data[config_field] = values[config_field]
        kwargs = {
            setting_field: data[config_field]
            for config_field, setting_field in _CONFIG_TO_SETTING_FIELD.items()
        }
        return cls(**kwargs)

    def to_config_fields(self) -> dict[str, Any]:
        return {
            config_field: getattr(self, setting_field)
            for setting_field, config_field in _SETTING_TO_CONFIG_FIELD.items()
        }


def config_fields_from_preset(preset: str) -> dict[str, Any]:
    resolved = preset if preset in GS2MESH_PRESETS else GS2MESH_PRESET
    return {
        _SETTING_TO_CONFIG_FIELD[setting_name]: value
        for setting_name, value in GS2MESH_PRESETS[resolved].items()
    }


def normalize_preset(value: Any, fallback: str = GS2MESH_PRESET) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in GS2MESH_PRESET_CHOICES else fallback


def infer_preset_from_public_fields(values: Mapping[str, Any]) -> str:
    for preset in GS2MESH_PRESETS:
        preset_values = config_fields_from_preset(preset)
        if all(
            values.get(field_name) == preset_values[field_name]
            for field_name in GS2MESH_PUBLIC_CONFIG_FIELDS
            if field_name in values
        ) and all(
            field_name in values for field_name in GS2MESH_PUBLIC_CONFIG_FIELDS
        ):
            return preset
    return GS2MESH_PRESET_CUSTOM


def fields_match_preset(
    values: Mapping[str, Any],
    preset: str,
    *,
    public_only: bool = False,
) -> bool:
    if preset not in GS2MESH_PRESETS:
        return False
    preset_values = config_fields_from_preset(preset)
    field_names = (
        GS2MESH_PUBLIC_CONFIG_FIELDS
        if public_only
        else GS2MESH_CONFIG_FIELDS
    )
    return all(values.get(field_name) == preset_values[field_name] for field_name in field_names)

