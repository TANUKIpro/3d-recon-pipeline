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

ProgressCallback = Callable[[float, str | None], None]
CancelCallback = Callable[[], None]

_DEFAULT_ENABLED = True
_DEFAULT_MAX_DIAMETER_RATIO = 0.08
_DEFAULT_Y_BAND_RATIO = 0.06
_DEFAULT_SMOOTH_ITERS = 2
_DEFAULT_SMOOTH_LAMBDA = 0.18

# Keep this conservative to reduce accidental top-side hole filling.
_MIN_DOWNWARD_NORMAL_Y = 0.25
_MIN_LOOP_VERTICES = 4


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


def _write_repaired_outputs(
    repaired_vertices: np.ndarray,
    repaired_faces: np.ndarray,
    repaired_path: Path,
    repaired_copy_path: Path,
) -> None:
    repaired_mesh = o3d.geometry.TriangleMesh()
    repaired_mesh.vertices = o3d.utility.Vector3dVector(repaired_vertices)
    repaired_mesh.triangles = o3d.utility.Vector3iVector(repaired_faces.astype(np.int32))
    repaired_mesh.remove_degenerate_triangles()
    repaired_mesh.remove_duplicated_triangles()
    repaired_mesh.remove_duplicated_vertices()
    repaired_mesh.remove_non_manifold_edges()
    repaired_mesh.remove_unreferenced_vertices()
    repaired_mesh.compute_vertex_normals()

    _write_mesh_safe(repaired_path, repaired_mesh)
    _write_mesh_safe(repaired_copy_path, repaired_mesh)


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

    if not params.enabled:
        print("Mesh repair disabled. Saving input mesh as repaired output.")
        mesh.compute_vertex_normals()
        _write_mesh_safe(repaired_path, mesh)
        _write_mesh_safe(repaired_copy_path, mesh)
        _emit_progress(progress_cb, 100.0, "Contact Hole Repair: skipped (disabled)")
        return repaired_path

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
