"""LiTo backend configuration helpers.

Centralises the runtime-resolved values from scripts.config_defaults so the
lito modules don't import the giant defaults file every time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSelectionConfig:
    """Compound-score weights and quality gates for frame selection."""

    weight_mask_coverage: float
    weight_triangulation: float
    weight_sharpness: float
    manual_index: int | None
    gate_min_bbox_short_px: int
    gate_min_mask_coverage: float
    gate_max_mask_coverage: float
    gate_max_connected_components: int


def load_frame_selection_config() -> FrameSelectionConfig:
    """Build a FrameSelectionConfig from scripts.config_defaults."""
    from scripts.config_defaults import (
        LITO_FRAME_SELECTION_W,
        LITO_GATE_MAX_CONNECTED_COMPONENTS,
        LITO_GATE_MAX_MASK_COVERAGE,
        LITO_GATE_MIN_BBOX_SHORT_PX,
        LITO_GATE_MIN_MASK_COVERAGE,
        LITO_MANUAL_FRAME_INDEX,
    )

    w_mask, w_tri, w_sharp = LITO_FRAME_SELECTION_W
    return FrameSelectionConfig(
        weight_mask_coverage=float(w_mask),
        weight_triangulation=float(w_tri),
        weight_sharpness=float(w_sharp),
        manual_index=LITO_MANUAL_FRAME_INDEX,
        gate_min_bbox_short_px=int(LITO_GATE_MIN_BBOX_SHORT_PX),
        gate_min_mask_coverage=float(LITO_GATE_MIN_MASK_COVERAGE),
        gate_max_mask_coverage=float(LITO_GATE_MAX_MASK_COVERAGE),
        gate_max_connected_components=int(LITO_GATE_MAX_CONNECTED_COMPONENTS),
    )
