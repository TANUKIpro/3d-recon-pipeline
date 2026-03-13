"""Data types, callbacks, and progress helpers for contact-hole repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

ProgressCallback = Callable[[float, str | None], None]
CancelCallback = Callable[[], None]


@dataclass(frozen=True)
class ContactHoleRepairParams:
    enabled: bool
    max_diameter_ratio: float
    y_band_ratio: float
    smooth_iters: int
    smooth_lambda: float


@dataclass
class LoopEval:
    loop_index: int
    vertex_count: int
    centroid_y: float
    normal_y: float
    diameter: float
    y_limit: float
    diameter_limit: float
    candidate: bool
    reason: str


@dataclass
class RepairStats:
    total_boundary_edges: int = 0
    total_boundary_loops: int = 0
    closed_loops: int = 0
    candidate_loops: int = 0
    repaired_loops: int = 0
    added_faces: int = 0
    skipped_open: int = 0
    skipped_small: int = 0
    skipped_not_contact_band: int = 0
    skipped_too_large: int = 0
    skipped_normal_direction: int = 0
    skipped_triangulation: int = 0
    skipped_non_manifold: int = 0
    skipped_degenerate: int = 0


@dataclass(frozen=True)
class RepairLoopCandidate:
    loop_id: int
    vertex_indices: tuple[int, ...]
    points: np.ndarray
    centroid: np.ndarray
    normal_y: float
    diameter: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_id": int(self.loop_id),
            "vertex_count": int(len(self.vertex_indices)),
            "diameter": float(self.diameter),
            "centroid": [float(x) for x in self.centroid.tolist()],
            "normal_y": float(self.normal_y),
            "points": [[float(p[0]), float(p[1]), float(p[2])] for p in self.points.tolist()],
        }


@dataclass(frozen=True)
class RepairCandidateAnalysis:
    mesh_path: str
    vertex_count: int
    face_count: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_center: tuple[float, float, float]
    loops: tuple[RepairLoopCandidate, ...]
    stats: RepairStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_path": self.mesh_path,
            "vertex_count": int(self.vertex_count),
            "face_count": int(self.face_count),
            "bbox_min": [float(v) for v in self.bbox_min],
            "bbox_max": [float(v) for v in self.bbox_max],
            "bbox_center": [float(v) for v in self.bbox_center],
            "loop_count": len(self.loops),
            "loops": [c.to_dict() for c in self.loops],
            "stats": {
                "total_boundary_edges": int(self.stats.total_boundary_edges),
                "total_boundary_loops": int(self.stats.total_boundary_loops),
                "closed_loops": int(self.stats.closed_loops),
                "candidate_loops": int(self.stats.candidate_loops),
                "skipped_open": int(self.stats.skipped_open),
                "skipped_small": int(self.stats.skipped_small),
                "skipped_triangulation": int(self.stats.skipped_triangulation),
                "skipped_degenerate": int(self.stats.skipped_degenerate),
            },
        }


@dataclass(frozen=True)
class GroundPlaneProbe:
    shift: float
    section_loop_count: int
    selected_loop_area: float
    matched_boundary_area_before: float
    matched_boundary_area_after: float
    cap_added: int
    valid: bool


@dataclass(frozen=True)
class GroundPlaneSearchResult:
    selected_shift: float | None
    scanned_count: int
    first_valid_shift: float | None
    lowest_valid_shift: float | None
    selected_loop_area: float


@dataclass(frozen=True)
class GroundPlaneValidation:
    plane_loop_count: int
    plane_boundary_edges: int
    matched_loop_present: bool
    matched_loop_area: float
    valid: bool


@dataclass(frozen=True)
class GroundPlaneSectionLoop:
    point_indices: tuple[int, ...]
    points: np.ndarray
    area: float


@dataclass(frozen=True)
class GroundPlaneBoundaryLoop:
    vertex_indices: tuple[int, ...]
    points: np.ndarray
    area: float


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


def _check_cancel(cancel_cb: CancelCallback | None) -> None:
    if cancel_cb is not None:
        cancel_cb()
