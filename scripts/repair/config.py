"""Parameter resolution for contact-hole repair."""

from __future__ import annotations

import os

from scripts.config_defaults import (
    REPAIR_ENABLED as _DEFAULT_ENABLED,
    REPAIR_MAX_DIAMETER_RATIO as _DEFAULT_MAX_DIAMETER_RATIO,
    REPAIR_SMOOTH_ITERS as _DEFAULT_SMOOTH_ITERS,
    REPAIR_Y_BAND_RATIO as _DEFAULT_Y_BAND_RATIO,
    _REPAIR_SMOOTH_LAMBDA as _DEFAULT_SMOOTH_LAMBDA,
)
from scripts.repair.types import ContactHoleRepairParams


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_params(overrides: dict | None = None) -> ContactHoleRepairParams:
    ov = overrides or {}

    def _ov_bool(key: str, env_key: str, default: bool) -> bool:
        if key in ov and ov[key] is not None:
            value = ov[key]
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}
        return _env_bool(env_key, default)

    def _ov_float(key: str, env_key: str, default: float) -> float:
        if key in ov and ov[key] is not None:
            try:
                return float(ov[key])
            except (TypeError, ValueError):
                pass
        return _env_float(env_key, default)

    def _ov_int(key: str, env_key: str, default: int) -> int:
        if key in ov and ov[key] is not None:
            try:
                return int(ov[key])
            except (TypeError, ValueError):
                pass
        return _env_int(env_key, default)

    return ContactHoleRepairParams(
        enabled=_ov_bool("enabled", "MESH_REPAIR_ENABLED", _DEFAULT_ENABLED),
        max_diameter_ratio=max(
            0.005,
            min(
                1.50,
                _ov_float(
                    "max_diameter_ratio",
                    "MESH_REPAIR_MAX_DIAMETER_RATIO",
                    _DEFAULT_MAX_DIAMETER_RATIO,
                ),
            ),
        ),
        y_band_ratio=max(
            0.005,
            min(
                0.50,
                _ov_float(
                    "y_band_ratio",
                    "MESH_REPAIR_Y_BAND_RATIO",
                    _DEFAULT_Y_BAND_RATIO,
                ),
            ),
        ),
        smooth_iters=max(
            0,
            min(
                12,
                _ov_int("smooth_iters", "MESH_REPAIR_SMOOTH_ITERS", _DEFAULT_SMOOTH_ITERS),
            ),
        ),
        smooth_lambda=max(0.0, min(0.45, _DEFAULT_SMOOTH_LAMBDA)),
    )
