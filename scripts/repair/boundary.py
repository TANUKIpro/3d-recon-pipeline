"""Boundary detection, loop analysis, and smoothing for contact-hole repair."""

from __future__ import annotations

import numpy as np

from scripts.config_defaults import (
    _REPAIR_MIN_DOWNWARD_NORMAL_Y as _MIN_DOWNWARD_NORMAL_Y,
    _REPAIR_MIN_LOOP_VERTICES as _MIN_LOOP_VERTICES,
)
from scripts.repair.triangulate import (
    _loop_projection_uv,
    _triangulate_polygon_ear_clip,
)
from scripts.repair.types import (
    ContactHoleRepairParams,
    LoopEval,
    RepairLoopCandidate,
    RepairStats,
)


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
