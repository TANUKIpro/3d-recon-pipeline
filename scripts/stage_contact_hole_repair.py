"""Stage 7: mesh repair for wrapped meshes.

This stage can run in two modes:
1) Automatic contact-hole repair (legacy behavior)
2) Selected-loop repair driven by dashboard interaction
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import open3d as o3d

from config_defaults import (
    REPAIR_ENABLED as _DEFAULT_ENABLED,
    REPAIR_MAX_DIAMETER_RATIO as _DEFAULT_MAX_DIAMETER_RATIO,
    REPAIR_SMOOTH_ITERS as _DEFAULT_SMOOTH_ITERS,
    REPAIR_Y_BAND_RATIO as _DEFAULT_Y_BAND_RATIO,
    _REPAIR_GROUND_SECTION_MIN_AREA_RATIO as _GROUND_SECTION_MIN_AREA_RATIO,
    _REPAIR_MIN_DOWNWARD_NORMAL_Y as _MIN_DOWNWARD_NORMAL_Y,
    _REPAIR_MIN_LOOP_VERTICES as _MIN_LOOP_VERTICES,
    _REPAIR_SMOOTH_LAMBDA as _DEFAULT_SMOOTH_LAMBDA,
)

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


def _write_mesh_safe(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    try:
        o3d.io.write_triangle_mesh(
            str(path),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
        )
    except TypeError:
        o3d.io.write_triangle_mesh(str(path), mesh)


def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()


def _load_mesh_arrays(mesh_ply: str) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh is empty: {mesh_ply}")
    _clean_mesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh is empty after cleanup: {mesh_ply}")
    return mesh, vertices, faces


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _extract_boundary_edges(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int32)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def _build_boundary_adjacency(boundary_edges: np.ndarray) -> dict[int, list[int]]:
    adjacency: dict[int, set[int]] = {}
    for a, b in boundary_edges:
        ai = int(a)
        bi = int(b)
        adjacency.setdefault(ai, set()).add(bi)
        adjacency.setdefault(bi, set()).add(ai)
    return {k: sorted(v) for k, v in adjacency.items()}


def _extract_boundary_paths(boundary_edges: np.ndarray) -> list[list[int]]:
    adjacency = _build_boundary_adjacency(boundary_edges)
    visited: set[tuple[int, int]] = set()
    paths: list[list[int]] = []
    max_steps = max(32, int(boundary_edges.shape[0] * 4))

    for a, b in boundary_edges.tolist():
        edge = _edge_key(int(a), int(b))
        if edge in visited:
            continue
        u = int(a)
        v = int(b)
        visited.add(edge)
        path = [u, v]
        prev = u
        cur = v

        for _ in range(max_steps):
            neighbors = adjacency.get(cur, [])
            if not neighbors:
                break

            unvisited = [n for n in neighbors if _edge_key(cur, n) not in visited]
            if unvisited:
                next_candidates = [n for n in unvisited if n != prev]
                nxt = next_candidates[0] if next_candidates else unvisited[0]
            else:
                break

            edge_next = _edge_key(cur, nxt)
            if edge_next in visited:
                break
            visited.add(edge_next)

            path.append(nxt)
            prev, cur = cur, nxt
            if cur == u:
                break

        compact: list[int] = []
        for idx in path:
            if not compact or compact[-1] != idx:
                compact.append(idx)
        if len(compact) >= 3:
            paths.append(compact)

    return paths


def _fit_loop_normal(loop_points: np.ndarray, centroid: np.ndarray, mesh_center: np.ndarray) -> np.ndarray:
    centered = loop_points - centroid
    cov = centered.T @ centered / max(loop_points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)

    outward = centroid - mesh_center
    if float(np.linalg.norm(outward)) > 1e-10 and float(np.dot(normal, outward)) < 0.0:
        normal = -normal
    return normal


def _polygon_area_2d(points_2d: np.ndarray) -> float:
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    bc = c - b
    return float(ab[0] * bc[1] - ab[1] * bc[0])


def _point_in_tri_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-10) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= eps:
        return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def _triangulate_polygon_ear_clip(loop_uv: np.ndarray) -> list[tuple[int, int, int]]:
    n = int(loop_uv.shape[0])
    if n < 3:
        return []

    area = _polygon_area_2d(loop_uv)
    if abs(area) < 1e-12:
        return []
    orientation = 1.0 if area > 0.0 else -1.0

    idx = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    max_iter = n * n
    steps = 0

    while len(idx) > 3 and steps < max_iter:
        ear_found = False
        m = len(idx)
        for i in range(m):
            i_prev = idx[(i - 1) % m]
            i_curr = idx[i]
            i_next = idx[(i + 1) % m]

            a = loop_uv[i_prev]
            b = loop_uv[i_curr]
            c = loop_uv[i_next]

            if _cross2(a, b, c) * orientation <= 1e-12:
                continue

            any_inside = False
            for j in idx:
                if j in {i_prev, i_curr, i_next}:
                    continue
                if _point_in_tri_2d(loop_uv[j], a, b, c):
                    any_inside = True
                    break
            if any_inside:
                continue

            if orientation > 0:
                triangles.append((i_prev, i_curr, i_next))
            else:
                triangles.append((i_prev, i_next, i_curr))
            del idx[i]
            ear_found = True
            break

        if not ear_found:
            return []
        steps += 1

    if len(idx) == 3:
        if orientation > 0:
            triangles.append((idx[0], idx[1], idx[2]))
        else:
            triangles.append((idx[0], idx[2], idx[1]))

    return triangles


def _loop_projection_uv(loop_points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(normal, axis))) > 0.92:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    tangent_u = np.cross(normal, axis)
    tangent_u = tangent_u / max(float(np.linalg.norm(tangent_u)), 1e-12)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v = tangent_v / max(float(np.linalg.norm(tangent_v)), 1e-12)

    centered = loop_points - loop_points.mean(axis=0)
    u = centered @ tangent_u
    v = centered @ tangent_v
    return np.column_stack((u, v))


def _has_non_manifold_edges(faces: np.ndarray) -> bool:
    if faces.size == 0:
        return False
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64)
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.any(counts > 2))


def _build_vertex_adjacency(num_vertices: int, faces: np.ndarray) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(num_vertices)]
    for tri in faces:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    return adj


def _apply_local_smoothing(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed_vertices: set[int],
    iterations: int,
    lamb: float,
) -> np.ndarray:
    if iterations <= 0 or not seed_vertices:
        return vertices

    adjacency = _build_vertex_adjacency(int(vertices.shape[0]), faces)
    smooth_set: set[int] = set(seed_vertices)
    expand: set[int] = set()
    for idx in smooth_set:
        expand.update(adjacency[idx])
    smooth_set |= expand

    verts = vertices.copy()
    for _ in range(iterations):
        updated = verts.copy()
        for idx in smooth_set:
            neighbors = adjacency[idx]
            if not neighbors:
                continue
            mean_pos = verts[np.fromiter(neighbors, dtype=np.int64)].mean(axis=0)
            updated[idx] = (1.0 - lamb) * verts[idx] + lamb * mean_pos
        verts = updated
    return verts


def _evaluate_loop(
    loop_index: int,
    loop_vertices: list[int],
    vertices: np.ndarray,
    mesh_center: np.ndarray,
    mesh_diag: float,
    mesh_min_y: float,
    params: ContactHoleRepairParams,
) -> LoopEval:
    pts = vertices[np.asarray(loop_vertices, dtype=np.int64)]
    centroid = pts.mean(axis=0)
    normal = _fit_loop_normal(pts, centroid, mesh_center)
    diameter = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    y_limit = mesh_min_y + params.y_band_ratio * mesh_diag
    diameter_limit = params.max_diameter_ratio * mesh_diag

    y_ok = float(centroid[1]) <= float(y_limit)
    diameter_ok = diameter <= diameter_limit
    normal_ok = float(normal[1]) <= -_MIN_DOWNWARD_NORMAL_Y

    candidate = y_ok and diameter_ok and normal_ok
    if candidate:
        reason = "candidate"
    elif not y_ok:
        reason = "outside_contact_band"
    elif not diameter_ok:
        reason = "diameter_too_large"
    else:
        reason = "not_downward_facing"

    return LoopEval(
        loop_index=loop_index,
        vertex_count=len(loop_vertices),
        centroid_y=float(centroid[1]),
        normal_y=float(normal[1]),
        diameter=diameter,
        y_limit=float(y_limit),
        diameter_limit=float(diameter_limit),
        candidate=candidate,
        reason=reason,
    )


def _candidate_from_path(
    loop_id: int,
    path: list[int],
    vertices: np.ndarray,
    mesh_center: np.ndarray,
    stats: RepairStats,
) -> RepairLoopCandidate | None:
    if len(path) < _MIN_LOOP_VERTICES:
        stats.skipped_small += 1
        return None
    if path[0] != path[-1]:
        stats.skipped_open += 1
        return None

    loop_vertices = path[:-1]
    if len(loop_vertices) < 3:
        stats.skipped_small += 1
        return None
    if len(set(loop_vertices)) < 3:
        stats.skipped_small += 1
        return None
    if len(set(loop_vertices)) != len(loop_vertices):
        # Complex self-touching boundary; skip to avoid unstable triangulation.
        stats.skipped_triangulation += 1
        return None

    stats.closed_loops += 1

    loop_points = vertices[np.asarray(loop_vertices, dtype=np.int64)]
    centroid = loop_points.mean(axis=0)
    normal = _fit_loop_normal(loop_points, centroid, mesh_center)
    loop_uv = _loop_projection_uv(loop_points, normal)
    local_tris = _triangulate_polygon_ear_clip(loop_uv)
    if not local_tris:
        stats.skipped_triangulation += 1
        return None

    new_faces = np.asarray(
        [[loop_vertices[a], loop_vertices[b], loop_vertices[c]] for a, b, c in local_tris],
        dtype=np.int64,
    )
    tri_pts = vertices[new_faces]
    tri_cross = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    tri_area2 = np.linalg.norm(tri_cross, axis=1)
    if np.any(tri_area2 <= 1e-11):
        stats.skipped_degenerate += 1
        return None

    diameter = float(np.linalg.norm(loop_points.max(axis=0) - loop_points.min(axis=0)))
    return RepairLoopCandidate(
        loop_id=int(loop_id),
        vertex_indices=tuple(int(v) for v in loop_vertices),
        points=loop_points,
        centroid=centroid,
        normal_y=float(normal[1]),
        diameter=diameter,
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


def _normalize_plane(
    plane_normal: np.ndarray,
    plane_d: float,
) -> tuple[np.ndarray, float]:
    plane_normal = np.asarray(plane_normal, dtype=np.float64)
    norm = max(float(np.linalg.norm(plane_normal)), 1e-12)
    return plane_normal / norm, float(plane_d) / norm


def _plane_distance_threshold(vertices: np.ndarray, tolerance_ratio: float) -> float:
    bbox_diag = max(
        float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))),
        1e-6,
    )
    return tolerance_ratio * bbox_diag


def _projected_bbox_area_on_plane(vertices: np.ndarray, plane_normal: np.ndarray) -> float:
    plane_normal, _ = _normalize_plane(plane_normal, 0.0)
    uv = _loop_projection_uv(vertices, plane_normal)
    span = uv.max(axis=0) - uv.min(axis=0)
    return max(float(span[0] * span[1]), 1e-12)


def _ground_section_min_area(vertices: np.ndarray, plane_normal: np.ndarray) -> float:
    return _projected_bbox_area_on_plane(vertices, plane_normal) * _GROUND_SECTION_MIN_AREA_RATIO


def _orient_ground_plane_toward_mesh(
    vertices: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
) -> tuple[np.ndarray, float, bool, int, int]:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    signed_dist = vertices @ plane_normal + plane_d
    above_count = int((signed_dist > 0).sum())
    below_count = int(vertices.shape[0] - above_count)
    mean_signed = float(signed_dist.mean())
    flipped = False
    if mean_signed < 0.0 or (abs(mean_signed) <= 1e-10 and above_count < below_count):
        plane_normal = -plane_normal
        plane_d = -plane_d
        flipped = True
        signed_dist = -signed_dist
        above_count = int((signed_dist > 0).sum())
        below_count = int(vertices.shape[0] - above_count)
    return plane_normal, plane_d, flipped, above_count, below_count


def _collect_plane_boundary_loops(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    original_vertex_count: int | None = None,
    tolerance_ratio: float = 0.01,
    snap_ratio: float = 0.0001,
) -> tuple[list[list[int]], np.ndarray]:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    boundary_edges = _extract_boundary_edges(faces)
    if boundary_edges.size == 0:
        return [], boundary_edges

    raw_paths = _extract_boundary_paths(boundary_edges)
    if not raw_paths:
        return [], boundary_edges

    dist_threshold = _plane_distance_threshold(vertices, tolerance_ratio)
    snap_threshold = max(_plane_distance_threshold(vertices, snap_ratio), 1e-6)
    plane_loops: list[list[int]] = []
    for path in raw_paths:
        if len(path) < 4 or path[0] != path[-1]:
            continue
        loop_vertices = path[:-1]
        if len(loop_vertices) < 3 or len(set(loop_vertices)) < 3:
            continue

        loop_indices = np.asarray(loop_vertices, dtype=np.int64)
        loop_dists = np.abs(vertices[loop_indices] @ plane_normal + plane_d)
        max_dist = float(loop_dists.max())
        if max_dist > dist_threshold:
            continue

        if original_vertex_count is not None:
            has_clip_vertex = any(int(idx) >= int(original_vertex_count) for idx in loop_vertices)
            if not has_clip_vertex and max_dist > snap_threshold:
                continue

        plane_loops.append(path)

    return plane_loops, boundary_edges


def _collect_plane_boundary_loop_infos(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    original_vertex_count: int | None = None,
    tolerance_ratio: float = 0.01,
) -> list[GroundPlaneBoundaryLoop]:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    plane_loops, _boundary_edges = _collect_plane_boundary_loops(
        vertices,
        faces,
        plane_normal,
        plane_d,
        original_vertex_count=original_vertex_count,
        tolerance_ratio=tolerance_ratio,
    )
    loop_infos: list[GroundPlaneBoundaryLoop] = []
    for path in plane_loops:
        loop_points = vertices[np.asarray(path[:-1], dtype=np.int64)]
        area = abs(_polygon_area_2d(_loop_projection_uv(loop_points, plane_normal)))
        if area <= 1e-12:
            continue
        loop_infos.append(
            GroundPlaneBoundaryLoop(
                vertex_indices=tuple(int(idx) for idx in path[:-1]),
                points=loop_points,
                area=float(area),
            )
        )
    return loop_infos


def _select_matching_plane_boundary_loop(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    original_vertex_count: int | None = None,
    tolerance_ratio: float = 0.01,
    target_loop: GroundPlaneSectionLoop | None = None,
) -> GroundPlaneBoundaryLoop | None:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    loop_infos = _collect_plane_boundary_loop_infos(
        vertices,
        faces,
        plane_normal,
        plane_d,
        original_vertex_count=original_vertex_count,
        tolerance_ratio=tolerance_ratio,
    )
    if not loop_infos:
        return None
    if target_loop is None:
        return max(loop_infos, key=lambda loop: loop.area)

    target_centroid = target_loop.points.mean(axis=0)
    bbox_diag = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-6)

    def _score(loop: GroundPlaneBoundaryLoop) -> float:
        area_error = abs(loop.area - target_loop.area) / max(target_loop.area, 1e-9)
        centroid_error = float(np.linalg.norm(loop.points.mean(axis=0) - target_centroid)) / bbox_diag
        return area_error + 0.25 * centroid_error

    best_loop = min(loop_infos, key=_score)
    best_area_error = abs(best_loop.area - target_loop.area) / max(target_loop.area, 1e-9)
    best_centroid_error = float(np.linalg.norm(best_loop.points.mean(axis=0) - target_centroid)) / bbox_diag
    if best_area_error > 0.15 or best_centroid_error > 0.08:
        return None
    return best_loop


def _count_plane_boundary_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    original_vertex_count: int | None = None,
    tolerance_ratio: float = 0.01,
    snap_ratio: float = 0.0001,
) -> int:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    boundary_edges = _extract_boundary_edges(faces)
    if boundary_edges.size == 0:
        return 0

    dist_threshold = _plane_distance_threshold(vertices, tolerance_ratio)
    snap_threshold = max(_plane_distance_threshold(vertices, snap_ratio), 1e-6)
    edge_indices = np.asarray(boundary_edges, dtype=np.int64)
    edge_vertices = vertices[edge_indices]
    edge_dists = np.abs(edge_vertices @ plane_normal + plane_d)
    edge_mask = np.all(edge_dists <= dist_threshold, axis=1)
    if original_vertex_count is not None:
        has_clip_vertex = np.any(edge_indices >= int(original_vertex_count), axis=1)
        snapped_to_plane = np.all(edge_dists <= snap_threshold, axis=1)
        edge_mask &= has_clip_vertex | snapped_to_plane
    return int(np.count_nonzero(edge_mask))


def _project_vertices_to_plane(
    vertices: np.ndarray,
    vertex_ids: set[int],
    plane_normal: np.ndarray,
    plane_d: float,
) -> np.ndarray:
    if not vertex_ids:
        return vertices

    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    updated = vertices.copy()
    indices = np.asarray(sorted(int(idx) for idx in vertex_ids), dtype=np.int64)
    signed = updated[indices] @ plane_normal + plane_d
    updated[indices] = updated[indices] - signed[:, None] * plane_normal[None, :]
    return updated


def _extract_plane_intersection_segments(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    signed_dist = vertices @ plane_normal + plane_d
    eps = max(_plane_distance_threshold(vertices, 1e-7), 1e-9)

    section_points: list[np.ndarray] = []
    edge_cache: dict[tuple[str, int, int], int] = {}

    def _point_from_vertex(idx: int) -> int:
        key = ("v", int(idx), int(idx))
        if key in edge_cache:
            return edge_cache[key]
        section_points.append(vertices[int(idx)].astype(np.float64))
        point_idx = len(section_points) - 1
        edge_cache[key] = point_idx
        return point_idx

    def _point_from_edge(a: int, b: int) -> int:
        ai = int(a)
        bi = int(b)
        key = ("e", min(ai, bi), max(ai, bi))
        cached = edge_cache.get(key)
        if cached is not None:
            return cached
        da = float(signed_dist[ai])
        db = float(signed_dist[bi])
        denom = da - db
        if abs(denom) <= 1e-15:
            t = 0.5
        else:
            t = da / denom
        t = max(0.0, min(1.0, t))
        point = (1.0 - t) * vertices[ai] + t * vertices[bi]
        section_points.append(point.astype(np.float64))
        point_idx = len(section_points) - 1
        edge_cache[key] = point_idx
        return point_idx

    segments: list[tuple[int, int]] = []
    for tri in np.asarray(faces, dtype=np.int64):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        tri_points: list[int] = []
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            da = float(signed_dist[a])
            db = float(signed_dist[b])
            if abs(da) <= eps and abs(db) <= eps:
                continue
            if abs(da) <= eps:
                tri_points.append(_point_from_vertex(a))
            if abs(db) <= eps:
                tri_points.append(_point_from_vertex(b))
            if da * db < -(eps * eps):
                tri_points.append(_point_from_edge(a, b))

        unique_points: list[int] = []
        seen: set[int] = set()
        for idx in tri_points:
            if idx not in seen:
                unique_points.append(idx)
                seen.add(idx)

        if len(unique_points) == 2:
            a_idx, b_idx = unique_points
            if a_idx != b_idx:
                segments.append((a_idx, b_idx))
        elif len(unique_points) > 2:
            best_pair: tuple[int, int] | None = None
            best_dist = -1.0
            for i in range(len(unique_points)):
                for j in range(i + 1, len(unique_points)):
                    pa = section_points[unique_points[i]]
                    pb = section_points[unique_points[j]]
                    dist = float(np.linalg.norm(pa - pb))
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (unique_points[i], unique_points[j])
            if best_pair is not None and best_pair[0] != best_pair[1]:
                segments.append(best_pair)

    if not section_points:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(section_points, dtype=np.float64), segments


def _extract_closed_section_loops(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
) -> list[GroundPlaneSectionLoop]:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    points, segments = _extract_plane_intersection_segments(vertices, faces, plane_normal, plane_d)
    if points.size == 0 or not segments:
        return []

    adjacency: dict[int, set[int]] = {}
    for a, b in segments:
        if a == b:
            continue
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    if not adjacency:
        return []

    visited: set[int] = set()
    loops: list[GroundPlaneSectionLoop] = []
    for start in sorted(adjacency):
        if start in visited:
            continue

        stack = [start]
        component: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            stack.extend(int(n) for n in adjacency.get(cur, ()))
        visited |= component

        if len(component) < 3 or any(len(adjacency.get(node, ())) != 2 for node in component):
            continue

        ordered = [min(component)]
        prev: int | None = None
        cur = ordered[0]
        while True:
            neighbors = sorted(int(n) for n in adjacency[cur])
            nxt = neighbors[0] if prev is None or neighbors[1] == prev else neighbors[1]
            if nxt == ordered[0]:
                break
            if nxt in ordered:
                ordered = []
                break
            ordered.append(nxt)
            prev, cur = cur, nxt
            if len(ordered) > len(component):
                ordered = []
                break

        if len(ordered) != len(component):
            continue
        if ordered[0] not in adjacency[ordered[-1]]:
            continue

        loop_points = points[np.asarray(ordered, dtype=np.int64)]
        area = abs(_polygon_area_2d(_loop_projection_uv(loop_points, plane_normal)))
        if area <= 1e-12:
            continue
        loops.append(
            GroundPlaneSectionLoop(
                point_indices=tuple(int(idx) for idx in ordered),
                points=loop_points,
                area=float(area),
            )
        )

    return loops


def _validate_ground_plane_output(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    tolerance_ratio: float = 0.01,
    target_loop: GroundPlaneSectionLoop | None = None,
) -> GroundPlaneValidation:
    plane_loops, _boundary_edges = _collect_plane_boundary_loops(
        vertices,
        faces,
        plane_normal,
        plane_d,
        tolerance_ratio=tolerance_ratio,
    )
    plane_boundary_edges = _count_plane_boundary_edges(
        vertices,
        faces,
        plane_normal,
        plane_d,
        tolerance_ratio=tolerance_ratio,
    )
    matched_loop = _select_matching_plane_boundary_loop(
        vertices,
        faces,
        plane_normal,
        plane_d,
        tolerance_ratio=tolerance_ratio,
        target_loop=target_loop,
    )
    matched_loop_present = matched_loop is not None if target_loop is not None else plane_boundary_edges > 0
    return GroundPlaneValidation(
        plane_loop_count=len(plane_loops),
        plane_boundary_edges=int(plane_boundary_edges),
        matched_loop_present=matched_loop_present,
        matched_loop_area=0.0 if matched_loop is None else float(matched_loop.area),
        valid=not matched_loop_present,
    )


def _clip_mesh_at_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    offset: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip mesh at a plane, keeping the 'above' side.

    Vertices with ``dot(v, normal) + d - offset > 0`` are considered above.
    Faces that straddle the plane are split so that only the above portion
    remains.
    """
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    dist = vertices @ plane_normal + plane_d
    dist = dist - offset  # cut slightly above ground

    above = dist > 0.0

    new_vertices_list: list[np.ndarray] = []
    new_vertex_count = int(vertices.shape[0])
    # Cache for interpolated edge vertices: (min_idx, max_idx) -> new_vertex_index
    edge_interp_cache: dict[tuple[int, int], int] = {}

    def _get_interp_vertex(idx_a: int, idx_b: int) -> int:
        nonlocal new_vertex_count
        key = (min(idx_a, idx_b), max(idx_a, idx_b))
        if key in edge_interp_cache:
            return edge_interp_cache[key]
        da = dist[idx_a]
        db = dist[idx_b]
        denom = da - db
        if abs(denom) < 1e-15:
            t = 0.5
        else:
            t = da / denom
        t = max(0.0, min(1.0, t))
        new_v = (1.0 - t) * vertices[idx_a] + t * vertices[idx_b]
        new_vertices_list.append(new_v)
        new_idx = new_vertex_count
        new_vertex_count += 1
        edge_interp_cache[key] = new_idx
        return new_idx

    kept_faces: list[list[int]] = []

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        a0, a1, a2 = above[i0], above[i1], above[i2]
        n_above = int(a0) + int(a1) + int(a2)

        if n_above == 3:
            # All above: keep as-is
            kept_faces.append([i0, i1, i2])
        elif n_above == 0:
            # All below: discard
            continue
        elif n_above == 2:
            # Two above, one below: produce two triangles
            if not a0:
                below_idx, above1, above2 = i0, i1, i2
            elif not a1:
                below_idx, above1, above2 = i1, i2, i0
            else:
                below_idx, above1, above2 = i2, i0, i1
            m1 = _get_interp_vertex(above1, below_idx)
            m2 = _get_interp_vertex(above2, below_idx)
            kept_faces.append([above1, above2, m1])
            kept_faces.append([above2, m2, m1])
        else:
            # One above, two below: produce one triangle
            if a0:
                above_idx, below1, below2 = i0, i1, i2
            elif a1:
                above_idx, below1, below2 = i1, i2, i0
            else:
                above_idx, below1, below2 = i2, i0, i1
            m1 = _get_interp_vertex(above_idx, below1)
            m2 = _get_interp_vertex(above_idx, below2)
            kept_faces.append([above_idx, m1, m2])

    if new_vertices_list:
        all_vertices = np.vstack((vertices, np.array(new_vertices_list, dtype=np.float64)))
    else:
        all_vertices = vertices.copy()

    if kept_faces:
        all_faces = np.asarray(kept_faces, dtype=np.int64)
    else:
        all_faces = np.zeros((0, 3), dtype=np.int64)

    return all_vertices, all_faces


def _cap_boundary_at_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    original_vertex_count: int | None = None,
    tolerance: float = 0.01,
    target_loop: GroundPlaneSectionLoop | None = None,
) -> tuple[np.ndarray, np.ndarray, set[int], float]:
    """Cap open boundary loops that lie near the ground plane.

    Uses the existing boundary extraction and ear-clip triangulation helpers
    to fill loops whose centroid is close to the clipping plane.
    """
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)

    selected_loop = _select_matching_plane_boundary_loop(
        vertices,
        faces,
        plane_normal,
        plane_d,
        original_vertex_count=original_vertex_count,
        tolerance_ratio=tolerance,
        target_loop=target_loop,
    )
    if selected_loop is None:
        return vertices, faces, set(), 0.0

    loop_vertices = list(selected_loop.vertex_indices)
    loop_indices = np.asarray(loop_vertices, dtype=np.int64)
    loop_points = vertices[loop_indices]
    loop_uv = _loop_projection_uv(loop_points, plane_normal)
    local_tris = _triangulate_polygon_ear_clip(loop_uv)
    if not local_tris:
        return vertices, faces, set(), float(selected_loop.area)

    new_faces = np.asarray(
        [[loop_vertices[a], loop_vertices[b], loop_vertices[c]] for a, b, c in local_tris],
        dtype=np.int64,
    )
    tri_pts = vertices[new_faces]
    tri_cross = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    tri_area2 = np.linalg.norm(tri_cross, axis=1)
    if np.any(tri_area2 <= 1e-11):
        return vertices, faces, set(), float(selected_loop.area)

    desired_normal = -plane_normal
    mean_dot = float(np.mean(tri_cross @ desired_normal))
    if mean_dot < 0.0:
        new_faces = new_faces[:, [0, 2, 1]]

    augmented_faces = np.vstack((faces, new_faces))
    return vertices, augmented_faces, {int(idx) for idx in loop_vertices}, float(selected_loop.area)


def _probe_ground_plane_shift(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    shift: float,
    *,
    tolerance_ratio: float = 0.01,
    min_section_area: float | None = None,
) -> GroundPlaneProbe:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    actual_plane_d = plane_d - shift
    section_loops = _extract_closed_section_loops(
        vertices,
        faces,
        plane_normal,
        actual_plane_d,
    )
    if not section_loops:
        return GroundPlaneProbe(
            shift=float(shift),
            section_loop_count=0,
            selected_loop_area=0.0,
            matched_boundary_area_before=0.0,
            matched_boundary_area_after=0.0,
            cap_added=0,
            valid=False,
        )

    selected_loop = max(section_loops, key=lambda loop: loop.area)
    required_area = max(0.0, 0.0 if min_section_area is None else float(min_section_area))
    if selected_loop.area < required_area:
        return GroundPlaneProbe(
            shift=float(shift),
            section_loop_count=len(section_loops),
            selected_loop_area=float(selected_loop.area),
            matched_boundary_area_before=0.0,
            matched_boundary_area_after=0.0,
            cap_added=0,
            valid=False,
        )

    original_vertex_count = int(vertices.shape[0])
    clipped_verts, clipped_faces = _clip_mesh_at_plane(
        vertices,
        faces,
        plane_normal,
        plane_d,
        offset=shift,
    )
    matched_boundary_before = _select_matching_plane_boundary_loop(
        clipped_verts,
        clipped_faces,
        plane_normal,
        actual_plane_d,
        original_vertex_count=original_vertex_count,
        tolerance_ratio=tolerance_ratio,
        target_loop=selected_loop,
    )

    capped_verts, capped_faces, _cap_vertex_ids, matched_boundary_area_before = _cap_boundary_at_plane(
        clipped_verts,
        clipped_faces,
        plane_normal,
        actual_plane_d,
        original_vertex_count=original_vertex_count,
        tolerance=tolerance_ratio,
        target_loop=selected_loop,
    )
    cap_added = int(capped_faces.shape[0]) - int(clipped_faces.shape[0])
    matched_boundary_after = _select_matching_plane_boundary_loop(
        capped_verts,
        capped_faces,
        plane_normal,
        actual_plane_d,
        original_vertex_count=original_vertex_count,
        tolerance_ratio=tolerance_ratio,
        target_loop=selected_loop,
    )
    valid = (
        matched_boundary_before is not None
        and matched_boundary_area_before >= required_area
        and cap_added > 0
        and matched_boundary_after is None
    )
    return GroundPlaneProbe(
        shift=float(shift),
        section_loop_count=len(section_loops),
        selected_loop_area=float(selected_loop.area),
        matched_boundary_area_before=(
            0.0 if matched_boundary_before is None else float(matched_boundary_before.area)
        ),
        matched_boundary_area_after=(
            0.0 if matched_boundary_after is None else float(matched_boundary_after.area)
        ),
        cap_added=int(cap_added),
        valid=bool(valid),
    )


def _find_ground_plane_shift(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    tolerance_ratio: float = 0.01,
) -> GroundPlaneSearchResult:
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    dists = vertices @ plane_normal + plane_d
    dist_min = float(dists.min())
    dist_max = float(dists.max())
    bbox_diag = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-6)
    interval_pad = max(1e-6, 1e-4 * bbox_diag)
    min_section_area = _ground_section_min_area(vertices, plane_normal)

    boundaries = [min(0.0, dist_min) - interval_pad]
    boundaries.extend(float(v) for v in np.unique(dists))
    boundaries.append(float(dist_max + interval_pad))

    scanned_count = 0
    first_valid_shift: float | None = None
    selected_shift: float | None = None
    selected_loop_area = 0.0
    probe_cache: dict[float, GroundPlaneProbe] = {}

    def _probe(shift: float) -> GroundPlaneProbe:
        nonlocal scanned_count
        key = float(shift)
        cached = probe_cache.get(key)
        if cached is not None:
            return cached
        probe = _probe_ground_plane_shift(
            vertices,
            faces,
            plane_normal,
            plane_d,
            key,
            tolerance_ratio=tolerance_ratio,
            min_section_area=min_section_area,
        )
        probe_cache[key] = probe
        scanned_count += 1
        return probe

    for idx in range(len(boundaries) - 1):
        low = float(boundaries[idx])
        high = float(boundaries[idx + 1])
        if high - low <= 1e-12:
            continue
        sample_shift = 0.5 * (low + high)
        sample_probe = _probe(sample_shift)
        if not sample_probe.valid:
            continue

        margin = min(interval_pad, 0.25 * (high - low))
        if margin <= 0.0 or low + margin >= high:
            lower_shift = sample_shift
            upper_shift = sample_shift
        else:
            lower_shift = low + margin
            upper_shift = high - margin
            if upper_shift <= lower_shift:
                lower_shift = sample_shift
                upper_shift = sample_shift

        stability_probe = _probe(upper_shift)
        if not stability_probe.valid:
            continue

        first_valid_shift = sample_shift
        lower_probe = _probe(lower_shift)
        if lower_probe.valid:
            selected_shift = float(lower_shift)
            selected_loop_area = float(lower_probe.selected_loop_area)
        else:
            selected_shift = float(sample_shift)
            selected_loop_area = float(sample_probe.selected_loop_area)
        break

    return GroundPlaneSearchResult(
        selected_shift=selected_shift,
        scanned_count=scanned_count,
        first_valid_shift=first_valid_shift,
        lowest_valid_shift=selected_shift,
        selected_loop_area=float(selected_loop_area),
    )


def _find_optimal_clip_offset(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    bottom_threshold_ratio: float = 0.20,
    epsilon: float = 0.002,
) -> float:
    """Find the minimum plane offset that eliminates all bottom-hole boundary edges.

    Moves the ground plane along its normal (toward the object) until the
    cross-section no longer intersects any boundary edges of existing holes.
    The resulting clip produces only closed contour loops that can be cleanly
    capped to form the model's bottom face.

    Returns the optimal offset value (>= 0) to pass to ``_clip_mesh_at_plane``.
    """
    plane_normal = plane_normal / max(float(np.linalg.norm(plane_normal)), 1e-12)

    # Signed distance of each vertex from the plane (positive = above/object side)
    dists = vertices @ plane_normal + plane_d
    dist_min = float(dists.min())

    boundary_edges = _extract_boundary_edges(faces)
    if boundary_edges.size == 0:
        # Mesh is already watertight; clip at model bottom + epsilon
        return max(dist_min + epsilon, epsilon)

    raw_paths = _extract_boundary_paths(boundary_edges)
    if not raw_paths:
        return max(dist_min + epsilon, epsilon)

    # Bounding-box height along the plane normal direction
    bbox_height = max(float(dists.max()) - dist_min, 1e-6)

    # Identify "bottom" loops: closed loops whose centroid is near the
    # bottom of the mesh (closest to the ground plane).  We measure from
    # `dist_min` rather than from 0 so that models sitting entirely above
    # (or below) the plane are handled correctly.
    bottom_max_dists: list[float] = []
    for path in raw_paths:
        if len(path) < 4 or path[0] != path[-1]:
            continue
        loop_vertices = path[:-1]
        if len(loop_vertices) < 3 or len(set(loop_vertices)) < 3:
            continue

        loop_indices = np.asarray(loop_vertices, dtype=np.int64)
        loop_dists = dists[loop_indices]
        centroid_dist = float(loop_dists.mean())

        # A "bottom" loop has its centroid within bottom_threshold_ratio of
        # the bbox height, measured from the mesh's lowest point along the
        # plane normal.
        dist_from_bottom = centroid_dist - dist_min
        if dist_from_bottom > bottom_threshold_ratio * bbox_height:
            continue

        # Record the maximum signed distance among this loop's vertices —
        # the plane must be pushed at least past this point
        bottom_max_dists.append(float(loop_dists.max()))

    if not bottom_max_dists:
        # No bottom loops found; clip at model bottom + epsilon
        return max(dist_min + epsilon, epsilon)

    # The plane must be above ALL bottom-loop boundary vertices
    critical_offset = max(bottom_max_dists) + epsilon

    # Clamp so we don't cut away more than 30% of the mesh height.
    # The cut depth into the mesh is (offset - dist_min).
    max_cut_depth = 0.30 * bbox_height
    max_allowed = dist_min + max_cut_depth
    if critical_offset > max_allowed:
        print(
            f"  warning: optimal clip offset {critical_offset:.4f} exceeds "
            f"30% cut depth ({max_cut_depth:.4f}), clamping to {max_allowed:.4f}"
        )
        critical_offset = max_allowed

    # Ensure offset is non-negative
    return max(critical_offset, epsilon)


def _repair_contact_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    params: ContactHoleRepairParams,
    *,
    progress_cb: ProgressCallback | None,
    cancel_cb: CancelCallback | None,
    selected_loop_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, RepairStats]:
    stats = RepairStats()

    boundary_edges = _extract_boundary_edges(faces)
    stats.total_boundary_edges = int(boundary_edges.shape[0])
    if boundary_edges.size == 0:
        return vertices, faces, stats

    raw_paths = _extract_boundary_paths(boundary_edges)
    stats.total_boundary_loops = len(raw_paths)

    mesh_center = vertices.mean(axis=0)
    mesh_diag = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-6)
    mesh_min_y = float(vertices[:, 1].min())

    current_faces = faces.copy()
    smooth_seed_vertices: set[int] = set()

    if not raw_paths:
        return vertices, current_faces, stats

    base_progress = 22.0
    span = 60.0

    for i, path in enumerate(raw_paths):
        _check_cancel(cancel_cb)
        ratio = (i + 1) / max(len(raw_paths), 1)
        _emit_progress(
            progress_cb,
            base_progress + span * ratio,
            f"Contact Hole Repair: evaluating boundary loop {i + 1}/{len(raw_paths)}",
        )

        if len(path) < _MIN_LOOP_VERTICES:
            stats.skipped_small += 1
            continue
        if path[0] != path[-1]:
            stats.skipped_open += 1
            continue

        loop_vertices = path[:-1]
        if len(loop_vertices) < 3:
            stats.skipped_small += 1
            continue
        if len(set(loop_vertices)) < 3:
            stats.skipped_small += 1
            continue
        if len(set(loop_vertices)) != len(loop_vertices):
            # Complex self-touching boundary; skip to avoid unstable triangulation.
            stats.skipped_triangulation += 1
            continue

        stats.closed_loops += 1

        if selected_loop_ids is not None:
            if i not in selected_loop_ids:
                continue
            stats.candidate_loops += 1
        else:
            ev = _evaluate_loop(
                i,
                loop_vertices,
                vertices,
                mesh_center,
                mesh_diag,
                mesh_min_y,
                params,
            )

            if not ev.candidate:
                if ev.reason == "outside_contact_band":
                    stats.skipped_not_contact_band += 1
                elif ev.reason == "diameter_too_large":
                    stats.skipped_too_large += 1
                else:
                    stats.skipped_normal_direction += 1
                continue

            stats.candidate_loops += 1

        loop_points = vertices[np.asarray(loop_vertices, dtype=np.int64)]
        normal = _fit_loop_normal(loop_points, loop_points.mean(axis=0), mesh_center)
        loop_uv = _loop_projection_uv(loop_points, normal)
        local_tris = _triangulate_polygon_ear_clip(loop_uv)
        if not local_tris:
            stats.skipped_triangulation += 1
            continue

        new_faces = np.asarray(
            [[loop_vertices[a], loop_vertices[b], loop_vertices[c]] for a, b, c in local_tris],
            dtype=np.int64,
        )

        tri_pts = vertices[new_faces]
        tri_cross = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
        tri_area2 = np.linalg.norm(tri_cross, axis=1)
        if np.any(tri_area2 <= 1e-11):
            stats.skipped_degenerate += 1
            continue

        candidate_faces = np.vstack((current_faces, new_faces))
        if _has_non_manifold_edges(candidate_faces):
            stats.skipped_non_manifold += 1
            continue

        current_faces = candidate_faces
        stats.repaired_loops += 1
        stats.added_faces += int(new_faces.shape[0])
        smooth_seed_vertices.update(int(v) for v in np.unique(new_faces))

    repaired_vertices = _apply_local_smoothing(
        vertices,
        current_faces,
        smooth_seed_vertices,
        iterations=params.smooth_iters,
        lamb=params.smooth_lambda,
    )
    return repaired_vertices, current_faces, stats


def _prepare_repaired_mesh(
    repaired_vertices: np.ndarray,
    repaired_faces: np.ndarray,
    *,
    remove_non_manifold: bool = True,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    repaired_mesh = o3d.geometry.TriangleMesh()
    repaired_mesh.vertices = o3d.utility.Vector3dVector(repaired_vertices)
    repaired_mesh.triangles = o3d.utility.Vector3iVector(repaired_faces.astype(np.int32))
    repaired_mesh.remove_degenerate_triangles()
    repaired_mesh.remove_duplicated_triangles()
    repaired_mesh.remove_duplicated_vertices()
    if remove_non_manifold:
        repaired_mesh.remove_non_manifold_edges()
    repaired_mesh.remove_unreferenced_vertices()
    repaired_mesh.compute_vertex_normals()

    final_vertices = np.asarray(repaired_mesh.vertices, dtype=np.float64).copy()
    final_faces = np.asarray(repaired_mesh.triangles, dtype=np.int64).copy()
    return repaired_mesh, final_vertices, final_faces


def _write_repaired_outputs(
    repaired_vertices: np.ndarray,
    repaired_faces: np.ndarray,
    repaired_path: Path,
    repaired_copy_path: Path,
    *,
    remove_non_manifold: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    repaired_mesh, final_vertices, final_faces = _prepare_repaired_mesh(
        repaired_vertices,
        repaired_faces,
        remove_non_manifold=remove_non_manifold,
    )

    _write_mesh_safe(repaired_path, repaired_mesh)
    _write_mesh_safe(repaired_copy_path, repaired_mesh)
    return final_vertices, final_faces


def _print_repair_stats(stats: RepairStats) -> None:
    print("Contact Hole Repair stats:")
    print(
        f"  boundary_edges={stats.total_boundary_edges:,}, "
        f"loops={stats.total_boundary_loops:,}, closed={stats.closed_loops:,}"
    )
    print(
        f"  candidates={stats.candidate_loops:,}, repaired={stats.repaired_loops:,}, "
        f"added_faces={stats.added_faces:,}"
    )
    print(
        "  skipped="
        f"open:{stats.skipped_open:,}, "
        f"small:{stats.skipped_small:,}, "
        f"not_contact_band:{stats.skipped_not_contact_band:,}, "
        f"too_large:{stats.skipped_too_large:,}, "
        f"normal:{stats.skipped_normal_direction:,}, "
        f"triangulation:{stats.skipped_triangulation:,}, "
        f"non_manifold:{stats.skipped_non_manifold:,}, "
        f"degenerate:{stats.skipped_degenerate:,}"
    )


def _finalize_ground_plane_outputs(
    capped_vertices: np.ndarray,
    capped_faces: np.ndarray,
    *,
    cap_vertex_ids: set[int],
    target_loop: GroundPlaneSectionLoop,
    plane_normal: np.ndarray,
    plane_d: float,
    smooth_iters: int,
    smooth_lambda: float,
    repaired_path: Path,
    repaired_copy_path: Path,
) -> tuple[np.ndarray, np.ndarray, str, GroundPlaneValidation]:
    projected_capped = _project_vertices_to_plane(
        capped_vertices,
        cap_vertex_ids,
        plane_normal,
        plane_d,
    )
    smoothed_vertices = _apply_local_smoothing(
        projected_capped,
        capped_faces,
        cap_vertex_ids,
        iterations=smooth_iters,
        lamb=smooth_lambda,
    )
    smoothed_projected = _project_vertices_to_plane(
        smoothed_vertices,
        cap_vertex_ids,
        plane_normal,
        plane_d,
    )

    attempts: list[tuple[str, np.ndarray, bool]] = [
        ("smoothed+strict-cleanup", smoothed_projected, True),
        ("smoothed+preserve-cap", smoothed_projected, False),
        ("flat-cap+preserve-cap", projected_capped, False),
    ]

    last_validation = GroundPlaneValidation(
        plane_loop_count=0,
        plane_boundary_edges=0,
        matched_loop_present=False,
        matched_loop_area=0.0,
        valid=False,
    )
    for label, candidate_vertices, remove_non_manifold in attempts:
        repaired_mesh, final_vertices, final_faces = _prepare_repaired_mesh(
            candidate_vertices,
            capped_faces,
            remove_non_manifold=remove_non_manifold,
        )
        validation = _validate_ground_plane_output(
            final_vertices,
            final_faces,
            plane_normal,
            plane_d,
            target_loop=target_loop,
        )
        last_validation = validation
        if validation.valid:
            _write_mesh_safe(repaired_path, repaired_mesh)
            _write_mesh_safe(repaired_copy_path, repaired_mesh)
            return final_vertices, final_faces, label, validation

    raise ValueError(
        "Ground-plane repaired mesh failed post-save validation "
        f"(plane_loops={last_validation.plane_loop_count}, "
        f"plane_boundary_edges={last_validation.plane_boundary_edges}, "
        f"matched_loop_present={last_validation.matched_loop_present}, "
        f"matched_loop_area={last_validation.matched_loop_area:.6f})"
    )


def run_contact_hole_repair(
    mesh_ply: str,
    output_dir: str,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
    enabled: bool | None = None,
    max_diameter_ratio: float | None = None,
    y_band_ratio: float | None = None,
    smooth_iters: int | None = None,
    selected_loop_ids: list[int] | None = None,
    require_selection: bool = False,
    ground_plane: dict | None = None,
) -> Path:
    """Repair mesh holes and save object_mesh_repaired.ply.

    When ``selected_loop_ids`` is provided, only those loops are filled.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    repair_dir = out / "contact_hole_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    repaired_path = out / "object_mesh_repaired.ply"
    repaired_copy_path = repair_dir / "object_mesh_repaired.ply"

    params = _resolve_params(
        {
            "enabled": enabled,
            "max_diameter_ratio": max_diameter_ratio,
            "y_band_ratio": y_band_ratio,
            "smooth_iters": smooth_iters,
        }
    )

    print("=== Mesh Repair (Stage 7) ===")
    print(
        "Params: "
        f"enabled={params.enabled}, "
        f"max_diameter_ratio={params.max_diameter_ratio:.4f}, "
        f"y_band_ratio={params.y_band_ratio:.4f}, "
        f"smooth_iters={params.smooth_iters}, "
        f"smooth_lambda={params.smooth_lambda:.3f}"
    )

    _emit_progress(progress_cb, 4.0, "Contact Hole Repair: loading mesh")
    _check_cancel(cancel_cb)

    mesh, vertices, faces = _load_mesh_arrays(mesh_ply)

    # ------------------------------------------------------------------
    # Ground-plane clipping path: always runs regardless of `enabled` flag
    # ------------------------------------------------------------------
    if ground_plane is not None:
        gp_normal = np.asarray(ground_plane["normal"], dtype=np.float64)
        gp_d = float(ground_plane["d"])
        gp_normal, gp_d, flipped, above_count, below_count = _orient_ground_plane_toward_mesh(
            vertices,
            gp_normal,
            gp_d,
        )
        if flipped:
            print(f"Ground-plane mode: flipped normal (above={above_count}, below={below_count})")

        print(
            f"Ground-plane mode: normal=[{gp_normal[0]:.4f}, {gp_normal[1]:.4f}, {gp_normal[2]:.4f}], "
            f"d={gp_d:.4f}"
        )
        print(f"  minimum stable section area: {_ground_section_min_area(vertices, gp_normal):.6f}")

        _emit_progress(progress_cb, 12.0, "Contact Hole Repair: scanning clip heights")
        _check_cancel(cancel_cb)
        search = _find_ground_plane_shift(vertices, faces, gp_normal, gp_d)
        print(f"  candidate offsets scanned: {search.scanned_count}")
        if search.first_valid_shift is not None:
            print(f"  first closed offset: {search.first_valid_shift:.4f}")
        if search.lowest_valid_shift is not None:
            print(f"  lowest valid closed offset: {search.lowest_valid_shift:.4f}")
        if search.selected_loop_area > 0.0:
            print(f"  selected section area: {search.selected_loop_area:.6f}")

        selected_shift = search.selected_shift
        if selected_shift is None:
            raise ValueError(
                "Ground-plane repair could not find a stable closed section loop "
                "of sufficient area while moving the plane toward the mesh"
            )

        actual_clip_d = gp_d - selected_shift
        section_loops = _extract_closed_section_loops(
            vertices,
            faces,
            gp_normal,
            actual_clip_d,
        )
        if not section_loops:
            raise ValueError(
                "Ground-plane repair selected a clip shift but no closed section loop "
                "was present at the chosen plane"
            )
        selected_section_loop = max(section_loops, key=lambda loop: loop.area)
        final_probe = _probe_ground_plane_shift(vertices, faces, gp_normal, gp_d, selected_shift)
        if not final_probe.valid:
            raise ValueError(
                "Ground-plane repair selected a clip shift but the matching section loop "
                "could not be capped consistently"
            )
        print(
            "  section-loop probe: "
            f"count={final_probe.section_loop_count}, "
            f"matched_area={final_probe.matched_boundary_area_before:.6f}"
            f"->{final_probe.matched_boundary_area_after:.6f}, "
            f"cap_added={final_probe.cap_added}"
        )

        _emit_progress(progress_cb, 15.0, "Contact Hole Repair: clipping at ground plane")
        _check_cancel(cancel_cb)
        clipped_verts, clipped_faces = _clip_mesh_at_plane(
            vertices, faces, gp_normal, gp_d, offset=selected_shift,
        )
        print(
            f"  clip: {vertices.shape[0]} -> {clipped_verts.shape[0]} vertices, "
            f"{faces.shape[0]} -> {clipped_faces.shape[0]} faces"
        )

        _emit_progress(progress_cb, 45.0, "Contact Hole Repair: capping ground boundary")
        _check_cancel(cancel_cb)
        capped_verts, capped_faces, cap_vertex_ids, matched_boundary_area = _cap_boundary_at_plane(
            clipped_verts,
            clipped_faces,
            gp_normal,
            actual_clip_d,
            original_vertex_count=vertices.shape[0],
            target_loop=selected_section_loop,
        )
        cap_added = int(capped_faces.shape[0]) - int(clipped_faces.shape[0])
        if cap_added <= 0:
            raise ValueError(
                "Ground-plane repair found a closed section loop but failed to cap the "
                "matching clipped boundary"
            )
        print(f"  cap: added {cap_added} faces")
        print(f"  matched boundary area: {matched_boundary_area:.6f}")

        _emit_progress(progress_cb, 70.0, "Contact Hole Repair: smoothing junction")
        _check_cancel(cancel_cb)

        _check_cancel(cancel_cb)
        _emit_progress(progress_cb, 90.0, "Contact Hole Repair: writing repaired mesh")

        repaired_vertices, repaired_faces, finalize_mode, post_validation = _finalize_ground_plane_outputs(
            capped_verts,
            capped_faces,
            cap_vertex_ids=cap_vertex_ids,
            target_loop=selected_section_loop,
            plane_normal=gp_normal,
            plane_d=actual_clip_d,
            smooth_iters=params.smooth_iters,
            smooth_lambda=params.smooth_lambda,
            repaired_path=repaired_path,
            repaired_copy_path=repaired_copy_path,
        )
        print(
            "  saved mesh validation: "
            f"plane_loops={post_validation.plane_loop_count}, "
            f"plane_boundary_edges={post_validation.plane_boundary_edges}, "
            f"matched_loop_present={post_validation.matched_loop_present}, "
            f"matched_loop_area={post_validation.matched_loop_area:.6f}"
        )
        print(f"  finalize mode: {finalize_mode}")

        print(f"Saved repaired mesh (ground-plane): {repaired_path}")
        _emit_progress(progress_cb, 100.0, "Contact Hole Repair: complete")
        return repaired_path

    if not params.enabled:
        print("Mesh repair disabled. Saving input mesh as repaired output.")
        mesh.compute_vertex_normals()
        _write_mesh_safe(repaired_path, mesh)
        _write_mesh_safe(repaired_copy_path, mesh)
        _emit_progress(progress_cb, 100.0, "Contact Hole Repair: skipped (disabled)")
        return repaired_path

    # ------------------------------------------------------------------
    # Legacy Y-band + normal heuristic path
    # ------------------------------------------------------------------
    if require_selection and selected_loop_ids is None:
        raise ValueError("selected_loop_ids is required for mesh repair")

    selected_set: set[int] | None = None
    if selected_loop_ids is not None:
        parsed: set[int] = set()
        for raw_id in selected_loop_ids:
            try:
                parsed.add(int(raw_id))
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid loop id: {raw_id!r}") from e
        if not parsed:
            if require_selection:
                print("Selected loop mode: 0 loops (skip)")
                _check_cancel(cancel_cb)
                _emit_progress(progress_cb, 90.0, "Contact Hole Repair: writing repaired mesh (skip)")
                mesh.compute_vertex_normals()
                _write_mesh_safe(repaired_path, mesh)
                _write_mesh_safe(repaired_copy_path, mesh)
                print(f"Saved repaired mesh: {repaired_path}")
                _emit_progress(progress_cb, 100.0, "Contact Hole Repair: skipped (no selection)")
                return repaired_path
            raise ValueError("selected_loop_ids must not be empty")

        analysis_candidates, _analysis_stats = _collect_repair_candidates(vertices, faces)
        selectable = {c.loop_id for c in analysis_candidates}
        missing = sorted(parsed - selectable)
        if missing:
            raise ValueError(f"Unknown or non-selectable loop ids: {missing}")
        selected_set = parsed
        print(f"Selected loop mode: {len(selected_set)} loops")

    _emit_progress(progress_cb, 12.0, "Contact Hole Repair: extracting boundary loops")
    repaired_vertices, repaired_faces, stats = _repair_contact_holes(
        vertices,
        faces,
        params,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        selected_loop_ids=selected_set,
    )

    if selected_set is not None and stats.repaired_loops == 0:
        raise ValueError("No selected loops were repaired")

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 90.0, "Contact Hole Repair: writing repaired mesh")

    _write_repaired_outputs(repaired_vertices, repaired_faces, repaired_path, repaired_copy_path)

    _print_repair_stats(stats)
    print(f"Saved repaired mesh: {repaired_path}")

    _emit_progress(progress_cb, 100.0, "Contact Hole Repair: complete")
    return repaired_path


def run_selected_contact_hole_repair(
    mesh_ply: str,
    output_dir: str,
    selected_loop_ids: list[int],
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
    enabled: bool | None = None,
    max_diameter_ratio: float | None = None,
    y_band_ratio: float | None = None,
    smooth_iters: int | None = None,
    ground_plane: dict | None = None,
) -> Path:
    """Explicit selected-loop entry point for dashboard interactive flow."""
    return run_contact_hole_repair(
        mesh_ply,
        output_dir,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        enabled=enabled,
        max_diameter_ratio=max_diameter_ratio,
        y_band_ratio=y_band_ratio,
        smooth_iters=smooth_iters,
        selected_loop_ids=selected_loop_ids,
        require_selection=True,
        ground_plane=ground_plane,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Repair contact-region mesh holes")
    parser.add_argument("mesh_ply", help="Input mesh path")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--selection-json",
        default="",
        help="Optional JSON file with {'selected_loop_ids': [..]} for selected-loop mode",
    )
    args = parser.parse_args()

    selected_ids: list[int] | None = None
    if args.selection_json:
        data = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
        raw = data.get("selected_loop_ids") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            raise ValueError("selection JSON must contain 'selected_loop_ids' list")
        selected_ids = [int(v) for v in raw]

    run_contact_hole_repair(
        args.mesh_ply,
        args.output_dir,
        selected_loop_ids=selected_ids,
        require_selection=bool(args.selection_json),
    )
