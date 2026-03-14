"""Candidate collection and analysis for contact-hole repair."""

from __future__ import annotations

import numpy as np

from scripts.repair.boundary import (
    _candidate_from_path,
    _extract_boundary_edges,
    _extract_boundary_paths,
)
from scripts.repair.mesh_io import _load_mesh_arrays
from scripts.repair.types import (
    CancelCallback,
    ProgressCallback,
    RepairCandidateAnalysis,
    RepairLoopCandidate,
    RepairStats,
    _check_cancel,
    _emit_progress,
)


def _collect_repair_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
    base_progress: float = 20.0,
    span: float = 70.0,
    detail_prefix: str = "Contact Hole Repair: analyzing boundary loop",
) -> tuple[list[RepairLoopCandidate], RepairStats]:
    stats = RepairStats()

    boundary_edges = _extract_boundary_edges(faces)
    stats.total_boundary_edges = int(boundary_edges.shape[0])
    if boundary_edges.size == 0:
        return [], stats

    raw_paths = _extract_boundary_paths(boundary_edges)
    stats.total_boundary_loops = len(raw_paths)
    if not raw_paths:
        return [], stats

    mesh_center = vertices.mean(axis=0)
    candidates: list[RepairLoopCandidate] = []

    for i, path in enumerate(raw_paths):
        _check_cancel(cancel_cb)
        ratio = (i + 1) / max(len(raw_paths), 1)
        _emit_progress(
            progress_cb,
            base_progress + span * ratio,
            f"{detail_prefix} {i + 1}/{len(raw_paths)}",
        )

        candidate = _candidate_from_path(i, path, vertices, mesh_center, stats)
        if candidate is None:
            continue

        stats.candidate_loops += 1
        candidates.append(candidate)

    return candidates, stats


def analyze_contact_hole_candidates(
    mesh_ply: str,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
) -> RepairCandidateAnalysis:
    """Analyze boundary-loop candidates for interactive Stage 7 selection."""
    _emit_progress(progress_cb, 4.0, "Contact Hole Repair: loading mesh")
    _check_cancel(cancel_cb)
    _mesh, vertices, faces = _load_mesh_arrays(mesh_ply)

    _emit_progress(progress_cb, 12.0, "Contact Hole Repair: extracting selectable loops")
    candidates, stats = _collect_repair_candidates(
        vertices,
        faces,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        base_progress=15.0,
        span=80.0,
    )

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) * 0.5

    _emit_progress(progress_cb, 100.0, "Contact Hole Repair: candidate analysis complete")
    return RepairCandidateAnalysis(
        mesh_path=str(mesh_ply),
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
        bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
        bbox_center=(float(bbox_center[0]), float(bbox_center[1]), float(bbox_center[2])),
        loops=tuple(candidates),
        stats=stats,
    )
