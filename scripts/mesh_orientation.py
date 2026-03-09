"""Utilities for triangle winding orientation checks and correction."""

from __future__ import annotations

import numpy as np


def face_outward_ratio(vertices: np.ndarray, faces: np.ndarray) -> float | None:
    """Return ratio of faces whose normal points away from mesh centroid.

    This is a lightweight heuristic for closed-ish meshes. Returns ``None`` when
    orientation cannot be evaluated (empty or fully degenerate faces).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.size == 0 or tris.size == 0:
        return None
    if tris.ndim != 2 or tris.shape[1] != 3:
        return None

    tri_pts = verts[tris]
    normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = normal_norm > 1e-12
    if not np.any(valid):
        return None

    tri_centers = tri_pts.mean(axis=1)
    mesh_center = verts.mean(axis=0)
    dots = np.einsum("ij,ij->i", normals[valid], tri_centers[valid] - mesh_center)
    return float(np.mean(dots > 0.0))


def orient_faces_outward(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_outward_ratio: float = 0.5,
) -> tuple[np.ndarray, bool, float | None, float | None]:
    """Flip face winding when outward ratio is lower than threshold."""
    tris = np.asarray(faces)
    ratio_before = face_outward_ratio(vertices, tris)
    if ratio_before is None or ratio_before >= float(min_outward_ratio):
        return tris, False, ratio_before, ratio_before

    flipped = tris[:, [0, 2, 1]].copy()
    ratio_after = face_outward_ratio(vertices, flipped)
    return flipped, True, ratio_before, ratio_after

