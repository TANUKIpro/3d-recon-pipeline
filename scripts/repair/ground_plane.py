"""Ground plane detection, clipping, and capping for contact-hole repair."""

from __future__ import annotations

import numpy as np

from scripts.config_defaults import (
    _REPAIR_GROUND_SECTION_MIN_AREA_RATIO as _GROUND_SECTION_MIN_AREA_RATIO,
    _REPAIR_GROUND_SECTION_START_QUANTILE as _GROUND_SECTION_START_QUANTILE,
)
from scripts.repair.boundary import (
    _extract_boundary_edges,
    _extract_boundary_paths,
    _fit_loop_normal,
)
from scripts.repair.triangulate import (
    _loop_projection_uv,
    _polygon_area_2d,
    _triangulate_polygon_ear_clip,
)
from scripts.repair.types import (
    GroundPlaneBoundaryLoop,
    GroundPlaneProbe,
    GroundPlaneSearchResult,
    GroundPlaneSectionLoop,
    GroundPlaneValidation,
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
    faces_i64 = np.asarray(faces, dtype=np.int64)
    # Vectorized pre-filter: only iterate faces that straddle the plane
    d0 = signed_dist[faces_i64[:, 0]]
    d1 = signed_dist[faces_i64[:, 1]]
    d2 = signed_dist[faces_i64[:, 2]]
    all_above = (d0 > eps) & (d1 > eps) & (d2 > eps)
    all_below = (d0 < -eps) & (d1 < -eps) & (d2 < -eps)
    straddle_mask = ~all_above & ~all_below
    straddle_faces = faces_i64[straddle_mask]
    for tri in straddle_faces:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip mesh at a plane, keeping the 'above' side.

    Vertices with ``dot(v, normal) + d - offset > 0`` are considered above.
    Faces that straddle the plane are split so that only the above portion
    remains.

    Returns:
        Tuple of (vertices, faces, source_face_indices) where
        *source_face_indices* maps each output face to its index in the
        input ``faces`` array.
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
    source_face_indices: list[int] = []

    # Vectorized face classification: bulk-keep all-above faces
    a0 = above[faces[:, 0]]
    a1 = above[faces[:, 1]]
    a2 = above[faces[:, 2]]
    n_above_arr = a0.astype(np.int8) + a1.astype(np.int8) + a2.astype(np.int8)
    all_above_mask = n_above_arr == 3
    all_above_indices = np.where(all_above_mask)[0]
    all_above_faces = faces[all_above_mask]
    if all_above_faces.shape[0] > 0:
        kept_faces.extend(all_above_faces.tolist())
        source_face_indices.extend(all_above_indices.tolist())

    # Only Python-loop over straddling faces (n_above == 1 or 2)
    straddle_mask = (n_above_arr > 0) & (n_above_arr < 3)
    straddle_indices = np.where(straddle_mask)[0]
    for src_idx, tri in zip(straddle_indices, faces[straddle_mask]):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        a0_t, a1_t, a2_t = above[i0], above[i1], above[i2]
        n_above = int(a0_t) + int(a1_t) + int(a2_t)

        if n_above == 2:
            # Two above, one below: produce two triangles
            if not a0_t:
                below_idx, above1, above2 = i0, i1, i2
            elif not a1_t:
                below_idx, above1, above2 = i1, i2, i0
            else:
                below_idx, above1, above2 = i2, i0, i1
            m1 = _get_interp_vertex(above1, below_idx)
            m2 = _get_interp_vertex(above2, below_idx)
            kept_faces.append([above1, above2, m1])
            source_face_indices.append(int(src_idx))
            kept_faces.append([above2, m2, m1])
            source_face_indices.append(int(src_idx))
        else:
            # One above, two below: produce one triangle
            if a0_t:
                above_idx, below1, below2 = i0, i1, i2
            elif a1_t:
                above_idx, below1, below2 = i1, i2, i0
            else:
                above_idx, below1, below2 = i2, i0, i1
            m1 = _get_interp_vertex(above_idx, below1)
            m2 = _get_interp_vertex(above_idx, below2)
            kept_faces.append([above_idx, m1, m2])
            source_face_indices.append(int(src_idx))

    if new_vertices_list:
        all_vertices = np.vstack((vertices, np.array(new_vertices_list, dtype=np.float64)))
    else:
        all_vertices = vertices.copy()

    if kept_faces:
        all_faces = np.asarray(kept_faces, dtype=np.int64)
    else:
        all_faces = np.zeros((0, 3), dtype=np.int64)

    source_arr = np.asarray(source_face_indices, dtype=np.int64)
    return all_vertices, all_faces, source_arr


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


def _generate_bottom_skirt_cap(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    *,
    bottom_height_ratio: float = 0.25,
    min_loop_vertices: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate skirt walls and floor cap for bottom boundary loops.

    Finds boundary loops near the bottom of the mesh (above the ground plane
    but not on it), projects their vertices onto the ground plane, and creates
    skirt triangles connecting the original boundary to the projected loop,
    plus a triangulated floor cap on the ground plane.

    Returns:
        (augmented_vertices, new_faces, stats) where *new_faces* contains only
        the newly generated skirt+cap faces (not the input faces).
    """
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    stats: dict = {
        "loops_found": 0,
        "loops_capped": 0,
        "skirt_faces": 0,
        "cap_faces": 0,
        "new_vertices": 0,
    }

    boundary_edges = _extract_boundary_edges(faces)
    if boundary_edges.size == 0:
        return vertices, np.zeros((0, 3), dtype=np.int64), stats

    raw_paths = _extract_boundary_paths(boundary_edges)
    if not raw_paths:
        return vertices, np.zeros((0, 3), dtype=np.int64), stats

    # Mesh metrics
    signed_dist = vertices @ plane_normal + plane_d
    dist_min = float(signed_dist.min())
    dist_max = float(signed_dist.max())
    bbox_height = max(dist_max - dist_min, 1e-6)
    bbox_diag = max(
        float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))),
        1e-6,
    )
    on_plane_threshold = 0.005 * bbox_diag  # 0.5% of bbox diagonal
    mesh_center = vertices.mean(axis=0)

    new_vertices_list: list[np.ndarray] = []
    new_faces_list: list[np.ndarray] = []
    next_vertex = int(vertices.shape[0])

    for path in raw_paths:
        if len(path) < min_loop_vertices + 1:
            continue
        if path[0] != path[-1]:
            continue
        loop_verts = path[:-1]
        if len(set(loop_verts)) < min_loop_vertices:
            continue

        loop_indices = np.asarray(loop_verts, dtype=np.int64)
        loop_points = vertices[loop_indices]
        loop_dists = signed_dist[loop_indices]

        # Centroid must be within bottom portion of model height
        avg_dist = float(loop_dists.mean())
        dist_from_bottom = avg_dist - dist_min
        if dist_from_bottom > bottom_height_ratio * bbox_height:
            continue

        # Must be above (or on) ground plane
        if avg_dist < 0:
            continue

        # Loop normal must point downward (toward -plane_normal)
        centroid = loop_points.mean(axis=0)
        loop_normal = _fit_loop_normal(loop_points, centroid, mesh_center)
        if float(np.dot(loop_normal, plane_normal)) > -0.1:
            continue

        # Must NOT already be on the ground plane (those are handled by _cap_boundary_at_plane)
        max_dist = float(np.abs(loop_dists).max())
        if max_dist <= on_plane_threshold:
            continue

        stats["loops_found"] += 1
        n = len(loop_verts)

        # Project loop vertices onto the ground plane
        proj_points = loop_points - loop_dists[:, None] * plane_normal[None, :]

        # Allocate new vertex indices for projected points
        proj_indices = list(range(next_vertex, next_vertex + n))
        new_vertices_list.extend(proj_points)
        next_vertex += n

        # --- Skirt triangles ---
        skirt_tris: list[list[int]] = []
        for i in range(n):
            i_next = (i + 1) % n
            v0 = loop_verts[i]
            v1 = loop_verts[i_next]
            p0 = proj_indices[i]
            p1 = proj_indices[i_next]
            skirt_tris.append([v0, v1, p0])
            skirt_tris.append([v1, p1, p0])

        skirt_faces = np.asarray(skirt_tris, dtype=np.int64)

        # Orient skirt normals outward (away from mesh center)
        all_verts_so_far = np.vstack(
            (vertices, np.asarray(new_vertices_list, dtype=np.float64))
        )
        sample_tri = skirt_faces[0]
        sv0 = all_verts_so_far[sample_tri[0]]
        sv1 = all_verts_so_far[sample_tri[1]]
        sv2 = all_verts_so_far[sample_tri[2]]
        tri_normal = np.cross(sv1 - sv0, sv2 - sv0)
        tri_center = (sv0 + sv1 + sv2) / 3.0
        outward_dir = tri_center - mesh_center
        if float(np.dot(tri_normal, outward_dir)) < 0:
            skirt_faces = skirt_faces[:, [0, 2, 1]]

        new_faces_list.append(skirt_faces)
        stats["skirt_faces"] += int(skirt_faces.shape[0])

        # --- Floor cap ---
        uv = _loop_projection_uv(proj_points, plane_normal)
        local_tris = _triangulate_polygon_ear_clip(uv)
        cap_faces: np.ndarray | None = None
        if local_tris:
            cap_faces = np.asarray(
                [
                    [proj_indices[a], proj_indices[b], proj_indices[c]]
                    for a, b, c in local_tris
                ],
                dtype=np.int64,
            )
        else:
            # Fallback: centroid fan triangulation.
            # Projected polygon may self-intersect or have near-degenerate
            # edges after collapsing heights, causing ear-clip to fail.
            centroid_pt = proj_points.mean(axis=0)
            centroid_idx = next_vertex
            new_vertices_list.append(centroid_pt)
            next_vertex += 1
            fan_tris: list[list[int]] = []
            for i in range(n):
                i_next = (i + 1) % n
                fan_tris.append([centroid_idx, proj_indices[i], proj_indices[i_next]])
            cap_faces = np.asarray(fan_tris, dtype=np.int64)

        if cap_faces is not None and cap_faces.shape[0] > 0:
            # Rebuild vertex lookup to include any centroid vertex just added
            all_verts_cap = np.vstack(
                (vertices, np.asarray(new_vertices_list, dtype=np.float64))
            )
            # Remove degenerate triangles (area ≈ 0)
            tri_pts = all_verts_cap[cap_faces]
            tri_cross = np.cross(
                tri_pts[:, 1] - tri_pts[:, 0],
                tri_pts[:, 2] - tri_pts[:, 0],
            )
            tri_area2 = np.linalg.norm(tri_cross, axis=1)
            valid_mask = tri_area2 > 1e-10
            cap_faces = cap_faces[valid_mask]
            tri_cross = tri_cross[valid_mask]

        if cap_faces is not None and cap_faces.shape[0] > 0:
            # Orient each cap triangle so its normal aligns with -plane_normal
            dots = tri_cross @ (-plane_normal)
            flip_mask = dots < 0
            cap_faces[flip_mask] = cap_faces[flip_mask][:, [0, 2, 1]]

            new_faces_list.append(cap_faces)
            stats["cap_faces"] += int(cap_faces.shape[0])

        stats["loops_capped"] += 1

    if not new_faces_list:
        return vertices, np.zeros((0, 3), dtype=np.int64), stats

    augmented_vertices = np.vstack(
        (vertices, np.asarray(new_vertices_list, dtype=np.float64))
    )
    all_new_faces = np.vstack(new_faces_list)
    stats["new_vertices"] = len(new_vertices_list)
    return augmented_vertices, all_new_faces, stats


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
    clipped_verts, clipped_faces, _ = _clip_mesh_at_plane(
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
    """Find the lowest shift where a closed section loop of sufficient area exists.

    Uses a two-pass coarse grid scan:

    1. **Pass 1** — scan all probes to find the maximum section loop area
       across all shifts (``max_area``).
    2. **Pass 2** — scan bottom-up and select the first (lowest) shift whose
       largest closed section loop area >= ``max(min_section_area, max_area * 0.10)``.

    The relative threshold (10 % of the observed maximum) prevents tiny
    spurious loops near the mesh extremity from being selected as the bottom,
    while still accepting genuinely narrow footprints (vases, tapered bases).
    """
    plane_normal, plane_d = _normalize_plane(plane_normal, plane_d)
    dists = vertices @ plane_normal + plane_d
    dist_min = float(dists.min())
    bbox_diag = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-6)
    interval_pad = max(1e-6, 1e-4 * bbox_diag)
    min_section_area = _ground_section_min_area(vertices, plane_normal)
    start_shift = float(np.quantile(dists, _GROUND_SECTION_START_QUANTILE))

    _N_PROBES = 64
    _RELATIVE_AREA_RATIO = 0.10  # 10 % of max observed section area

    # Coarse evenly-spaced grid
    low_end = dist_min + interval_pad
    high_end = start_shift
    if low_end >= high_end:
        return GroundPlaneSearchResult(
            selected_shift=None,
            scanned_count=0,
            first_valid_shift=None,
            lowest_valid_shift=None,
            selected_loop_area=0.0,
        )

    shifts = np.linspace(low_end, high_end, _N_PROBES)

    # --- Pass 1: collect per-shift best loop area to determine max_area ---
    probe_results: list[tuple[float, float]] = []  # (shift, best_loop_area)
    for shift_val in shifts:
        shift_f = float(shift_val)
        actual_d = plane_d - shift_f
        loops = _extract_closed_section_loops(vertices, faces, plane_normal, actual_d)
        if loops:
            best_area = float(max(loops, key=lambda l: l.area).area)
        else:
            best_area = 0.0
        probe_results.append((shift_f, best_area))

    max_area = max((a for _, a in probe_results), default=0.0)
    effective_min = max(min_section_area, max_area * _RELATIVE_AREA_RATIO)

    # --- Pass 2: bottom-up selection using the effective threshold ---
    first_valid_shift: float | None = None
    for shift_f, best_area in probe_results:
        if best_area > 0.0 and first_valid_shift is None:
            first_valid_shift = shift_f
        if best_area >= effective_min:
            return GroundPlaneSearchResult(
                selected_shift=shift_f,
                scanned_count=len(probe_results),
                first_valid_shift=first_valid_shift,
                lowest_valid_shift=shift_f,
                selected_loop_area=best_area,
            )

    return GroundPlaneSearchResult(
        selected_shift=None,
        scanned_count=len(probe_results),
        first_valid_shift=first_valid_shift,
        lowest_valid_shift=None,
        selected_loop_area=0.0,
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
