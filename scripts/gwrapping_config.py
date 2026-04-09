"""Shared GaussianWrapping preset metadata and runtime settings helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scripts.config_defaults import (
    GWRAPPING_PRESET,
    GWRAPPING_PRESET_CHOICES,
    GWRAPPING_PRESET_CUSTOM,
    GWRAPPING_PRESETS,
)

GWRAPPING_PUBLIC_CONFIG_FIELDS: tuple[str, ...] = (
    "gwrapping_iterations",
    "gwrapping_rasterizer",
    "gwrapping_resolution",
    "gwrapping_use_masks",
    "gwrapping_extraction_method",
)

GWRAPPING_INTERNAL_CONFIG_FIELDS: tuple[str, ...] = (
    "gwrapping_use_depth",
)

GWRAPPING_CONFIG_FIELDS: tuple[str, ...] = (
    *GWRAPPING_PUBLIC_CONFIG_FIELDS,
    *GWRAPPING_INTERNAL_CONFIG_FIELDS,
)

_SETTING_TO_CONFIG_FIELD = {
    "iterations": "gwrapping_iterations",
    "rasterizer": "gwrapping_rasterizer",
    "resolution": "gwrapping_resolution",
    "use_masks": "gwrapping_use_masks",
    "extraction_method": "gwrapping_extraction_method",
    "use_depth": "gwrapping_use_depth",
}
_CONFIG_TO_SETTING_FIELD = {
    config_name: setting_name
    for setting_name, config_name in _SETTING_TO_CONFIG_FIELD.items()
}


@dataclass(frozen=True)
class GWrappingSettings:
    iterations: int
    rasterizer: str
    resolution: int
    use_masks: bool
    extraction_method: str
    use_depth: bool

    @classmethod
    def from_preset(cls, preset: str = GWRAPPING_PRESET) -> GWrappingSettings:
        resolved = preset if preset in GWRAPPING_PRESETS else GWRAPPING_PRESET
        return cls(**GWRAPPING_PRESETS[resolved])

    @classmethod
    def from_config_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        preset: str = GWRAPPING_PRESET,
    ) -> GWrappingSettings:
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
    resolved = preset if preset in GWRAPPING_PRESETS else GWRAPPING_PRESET
    return {
        _SETTING_TO_CONFIG_FIELD[setting_name]: value
        for setting_name, value in GWRAPPING_PRESETS[resolved].items()
    }


def normalize_preset(value: Any, fallback: str = GWRAPPING_PRESET) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in GWRAPPING_PRESET_CHOICES else fallback


def infer_preset_from_public_fields(values: Mapping[str, Any]) -> str:
    for preset in GWRAPPING_PRESETS:
        preset_values = config_fields_from_preset(preset)
        if all(
            values.get(field_name) == preset_values[field_name]
            for field_name in GWRAPPING_PUBLIC_CONFIG_FIELDS
            if field_name in values
        ) and all(
            field_name in values for field_name in GWRAPPING_PUBLIC_CONFIG_FIELDS
        ):
            return preset
    return GWRAPPING_PRESET_CUSTOM


def fields_match_preset(
    values: Mapping[str, Any],
    preset: str,
    *,
    public_only: bool = False,
) -> bool:
    if preset not in GWRAPPING_PRESETS:
        return False
    preset_values = config_fields_from_preset(preset)
    field_names = (
        GWRAPPING_PUBLIC_CONFIG_FIELDS
        if public_only
        else GWRAPPING_CONFIG_FIELDS
    )
    return all(values.get(field_name) == preset_values[field_name] for field_name in field_names)
