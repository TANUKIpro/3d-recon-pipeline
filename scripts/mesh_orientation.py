"""Utilities for triangle winding orientation checks and correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraOrientationResult:
    """Result of camera-aware mesh winding orientation."""

    faces: np.ndarray
    flipped: bool
    outward_ratio_before: float | None
    outward_ratio_after: float | None
    camera_ratio_before: float | None
    camera_ratio_after: float | None
    source: str


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


def _camera_centers_from_poses(poses: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        centers = arr[:, :3, 3]
    elif arr.ndim == 2 and arr.shape[1] == 3:
        centers = arr
    else:
        return None

    finite = np.all(np.isfinite(centers), axis=1)
    centers = centers[finite]
    if centers.size == 0:
        return None
    return centers


def camera_facing_ratio(
    vertices: np.ndarray,
    faces: np.ndarray,
    poses: np.ndarray,
    *,
    chunk_size: int = 10000,
) -> tuple[float | None, float | None]:
    """Return face-normal agreement ratios against the nearest camera.

    The first ratio is for the supplied winding. The second ratio is for the
    globally flipped winding. Texture baking samples only surfaces seen by the
    reconstruction cameras, so this is a stronger signal than pure shape
    heuristics for open or non-convex object meshes.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    camera_centers = _camera_centers_from_poses(poses)
    if camera_centers is None:
        return None, None
    if verts.size == 0 or tris.size == 0:
        return None, None
    if tris.ndim != 2 or tris.shape[1] != 3:
        return None, None

    tri_pts = verts[tris]
    normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = normal_norm > 1e-12
    if not np.any(valid):
        return None, None

    normals = normals[valid] / normal_norm[valid, None]
    tri_centers = tri_pts[valid].mean(axis=1)

    dots: list[np.ndarray] = []
    step = max(1, int(chunk_size))
    for start in range(0, len(tri_centers), step):
        centers = tri_centers[start:start + step]
        delta = camera_centers[None, :, :] - centers[:, None, :]
        dist2 = np.einsum("ijk,ijk->ij", delta, delta)
        nearest = camera_centers[np.argmin(dist2, axis=1)]
        to_camera = nearest - centers
        to_camera_norm = np.linalg.norm(to_camera, axis=1)
        usable = to_camera_norm > 1e-12
        if not np.any(usable):
            continue
        dirs = to_camera[usable] / to_camera_norm[usable, None]
        chunk_normals = normals[start:start + len(centers)][usable]
        dots.append(np.einsum("ij,ij->i", chunk_normals, dirs))

    if not dots:
        return None, None

    all_dots = np.concatenate(dots)
    if all_dots.size == 0:
        return None, None
    return float(np.mean(all_dots > 0.0)), float(np.mean(all_dots < 0.0))


def orient_mesh_for_cameras(
    vertices: np.ndarray,
    faces: np.ndarray,
    poses: np.ndarray,
    *,
    min_camera_ratio: float = 0.6,
    min_camera_margin: float = 0.15,
) -> CameraOrientationResult:
    """Orient mesh winding using camera-facing normals before shape heuristics.

    Returns a dataclass so callers can log both the camera decision and the
    fallback shape heuristic without changing the legacy orientation helpers.
    """
    tris = np.asarray(faces)
    outward_before = face_outward_ratio(vertices, tris)
    camera_before, camera_after = camera_facing_ratio(vertices, tris, poses)

    if camera_before is not None and camera_after is not None:
        margin = float(min_camera_margin)
        min_ratio = float(min_camera_ratio)
        if camera_before >= min_ratio and camera_before >= camera_after + margin:
            return CameraOrientationResult(
                faces=tris,
                flipped=False,
                outward_ratio_before=outward_before,
                outward_ratio_after=outward_before,
                camera_ratio_before=camera_before,
                camera_ratio_after=camera_after,
                source="camera",
            )
        if camera_after >= min_ratio and camera_after >= camera_before + margin:
            flipped_faces = tris[:, [0, 2, 1]].copy()
            outward_after = face_outward_ratio(vertices, flipped_faces)
            return CameraOrientationResult(
                faces=flipped_faces,
                flipped=True,
                outward_ratio_before=outward_before,
                outward_ratio_after=outward_after,
                camera_ratio_before=camera_before,
                camera_ratio_after=camera_after,
                source="camera",
            )

    fixed_faces, flipped, _, outward_after = orient_mesh_outward(vertices, tris)
    if flipped and camera_before is not None:
        camera_fixed = camera_after
    else:
        camera_fixed = camera_before
    return CameraOrientationResult(
        faces=fixed_faces,
        flipped=flipped,
        outward_ratio_before=outward_before,
        outward_ratio_after=outward_after,
        camera_ratio_before=camera_before,
        camera_ratio_after=camera_fixed,
        source="boundary_fallback",
    )


def _boundary_face_vote(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    boundary_pct: float = 0.05,
    min_faces_per_axis: int = 2,
) -> bool | None:
    """Vote on face orientation using boundary-face normals along 6 axis directions.

    For each axis direction (±X, ±Y, ±Z), faces near the bounding-box boundary
    are selected and their average normal component along that axis is checked.
    On a correctly-oriented mesh, faces at +X boundary should have positive
    normal X component, faces at -X boundary should have negative X component, etc.

    Returns True if faces appear outward-oriented, False if inward, None if
    insufficient data to decide.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)

    tri_pts = verts[tris]
    normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms.ravel() > 1e-12
    if not np.any(valid):
        return None

    normals[valid] /= norms[valid]
    tri_centers = tri_pts.mean(axis=1)

    outward_votes = 0
    inward_votes = 0

    for axis in range(3):
        coords = tri_centers[:, axis]
        cmin, cmax = coords.min(), coords.max()
        span = cmax - cmin
        if span < 1e-12:
            continue
        threshold = span * boundary_pct

        # +axis boundary: faces near max, expect positive normal component
        hi_mask = valid & (coords >= cmax - threshold)
        if hi_mask.sum() >= min_faces_per_axis:
            avg_n = normals[hi_mask, axis].mean()
            if avg_n > 0:
                outward_votes += 1
            else:
                inward_votes += 1

        # -axis boundary: faces near min, expect negative normal component
        lo_mask = valid & (coords <= cmin + threshold)
        if lo_mask.sum() >= min_faces_per_axis:
            avg_n = normals[lo_mask, axis].mean()
            if avg_n < 0:
                outward_votes += 1
            else:
                inward_votes += 1

    if outward_votes + inward_votes == 0:
        return None
    return outward_votes >= inward_votes


def orient_mesh_outward(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, bool, float | None, float | None]:
    """Orient faces outward using boundary-face normal verification.

    Unlike ``orient_faces_outward`` which uses a centroid-based heuristic that
    fails on non-convex meshes, this function checks normals at the bounding-box
    boundaries where faces are reliably on the convex hull.

    Returns (faces, flipped, ratio_before, ratio_after) — same signature as
    ``orient_faces_outward`` for drop-in compatibility.
    """
    tris = np.asarray(faces)

    ratio_before = face_outward_ratio(vertices, tris)
    is_outward = _boundary_face_vote(vertices, tris)

    if is_outward is None:
        # Can't determine — leave as-is
        return tris, False, ratio_before, ratio_before

    if is_outward:
        return tris, False, ratio_before, ratio_before

    # Flip all faces
    flipped = tris[:, [0, 2, 1]].copy()
    ratio_after = face_outward_ratio(vertices, flipped)
    return flipped, True, ratio_before, ratio_after
