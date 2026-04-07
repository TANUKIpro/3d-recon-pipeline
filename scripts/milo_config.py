"""Shared MILo preset metadata and runtime settings helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scripts.config_defaults import (
    MILO_PRESET,
    MILO_PRESET_CHOICES,
    MILO_PRESET_CUSTOM,
    MILO_PRESETS,
)

MILO_PUBLIC_CONFIG_FIELDS: tuple[str, ...] = (
    "milo_iterations",
    "milo_scene_type",
    "milo_mesh_config",
    "milo_extraction_method",
    "milo_use_masks",
)

MILO_INTERNAL_CONFIG_FIELDS: tuple[str, ...] = (
    "milo_rasterizer",
    "milo_dense_gaussians",
    "milo_data_device",
    "milo_mesh_res",
)

MILO_CONFIG_FIELDS: tuple[str, ...] = (
    *MILO_PUBLIC_CONFIG_FIELDS,
    *MILO_INTERNAL_CONFIG_FIELDS,
)

_SETTING_TO_CONFIG_FIELD = {
    "iterations": "milo_iterations",
    "scene_type": "milo_scene_type",
    "mesh_config": "milo_mesh_config",
    "extraction_method": "milo_extraction_method",
    "use_masks": "milo_use_masks",
    "rasterizer": "milo_rasterizer",
    "dense_gaussians": "milo_dense_gaussians",
    "data_device": "milo_data_device",
    "mesh_res": "milo_mesh_res",
}
_CONFIG_TO_SETTING_FIELD = {
    config_name: setting_name
    for setting_name, config_name in _SETTING_TO_CONFIG_FIELD.items()
}


@dataclass(frozen=True)
class MiloSettings:
    iterations: int
    scene_type: str
    mesh_config: str
    extraction_method: str
    rasterizer: str
    use_masks: bool
    dense_gaussians: bool
    data_device: str
    mesh_res: int

    @classmethod
    def from_preset(cls, preset: str = MILO_PRESET) -> MiloSettings:
        resolved = preset if preset in MILO_PRESETS else MILO_PRESET
        return cls(**MILO_PRESETS[resolved])

    @classmethod
    def from_config_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        preset: str = MILO_PRESET,
    ) -> MiloSettings:
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
    resolved = preset if preset in MILO_PRESETS else MILO_PRESET
    return {
        _SETTING_TO_CONFIG_FIELD[setting_name]: value
        for setting_name, value in MILO_PRESETS[resolved].items()
    }


def normalize_preset(value: Any, fallback: str = MILO_PRESET) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in MILO_PRESET_CHOICES else fallback


def infer_preset_from_public_fields(values: Mapping[str, Any]) -> str:
    for preset in MILO_PRESETS:
        preset_values = config_fields_from_preset(preset)
        if all(
            values.get(field_name) == preset_values[field_name]
            for field_name in MILO_PUBLIC_CONFIG_FIELDS
            if field_name in values
        ) and all(
            field_name in values for field_name in MILO_PUBLIC_CONFIG_FIELDS
        ):
            return preset
    return MILO_PRESET_CUSTOM


def fields_match_preset(
    values: Mapping[str, Any],
    preset: str,
    *,
    public_only: bool = False,
) -> bool:
    if preset not in MILO_PRESETS:
        return False
    preset_values = config_fields_from_preset(preset)
    field_names = (
        MILO_PUBLIC_CONFIG_FIELDS
        if public_only
        else MILO_CONFIG_FIELDS
    )
    return all(values.get(field_name) == preset_values[field_name] for field_name in field_names)
