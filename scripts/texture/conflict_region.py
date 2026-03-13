"""Conflict detection and region-graph label optimization for texture baking."""

from collections import defaultdict

import numpy as np

_TEXTURE_CONFLICT_RATIO = 1.35
_TEXTURE_CONFLICT_VIEW_ANGLE_DEG = 20.0
_TEXTURE_CONFLICT_FACE_MIN_TEXELS = 4
_TEXTURE_CONFLICT_FACE_MIN_FRAC = 0.2
_TEXTURE_CONFLICT_FACE_MIN_COVERAGE = 0.7
_TEXTURE_CONFLICT_SMOOTH_DOT = 0.95
_TEXTURE_CONFLICT_SMOOTH_GAIN = 1.05
_TEXTURE_CONFLICT_SMOOTH_MIN_NEIGHBORS = 2
_TEXTURE_REGION_NORMAL_DOT = 0.92
_TEXTURE_REGION_MIN_FACES = 3
_TEXTURE_REGION_TOP_LABELS = 4
_TEXTURE_REGION_MAX_ITERS = 4
_TEXTURE_REGION_SMOOTHNESS = 0.55
_TEXTURE_REGION_MIN_LABEL_COVERAGE = 0.15
_TEXTURE_REGION_MISSING_COST = 1.5
_TEXTURE_REGION_LOW_COVERAGE_PENALTY = 0.35


def _compute_conflict_texels(
    pos3d: np.ndarray,
    poses: np.ndarray,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    conflict_ratio: float = _TEXTURE_CONFLICT_RATIO,
    min_view_angle_deg: float = _TEXTURE_CONFLICT_VIEW_ANGLE_DEG,
) -> np.ndarray:
    """Mark texels where two competing views disagree strongly."""
    n_texels = best_scores.shape[0]
    if n_texels == 0 or best_scores.shape[1] <= 1 or len(poses) == 0:
        return np.zeros(n_texels, dtype=bool)

    top0 = best_scores[:, 0].astype(np.float64, copy=False)
    top1 = best_scores[:, 1].astype(np.float64, copy=False)
    view0 = best_views[:, 0]
    view1 = best_views[:, 1]

    close_scores = (top0 > 0) & (top1 > 0) & (top0 < conflict_ratio * top1)
    different_views = view0 >= 0
    different_views &= view1 >= 0
    different_views &= view0 != view1
    candidates = close_scores & different_views

    conflict = np.zeros(n_texels, dtype=bool)
    idx = np.where(candidates)[0]
    if idx.size == 0:
        return conflict

    cam_pos = poses[:, :3, 3]
    dir0 = cam_pos[view0[idx]] - pos3d[idx]
    dir1 = cam_pos[view1[idx]] - pos3d[idx]
    dir0 /= np.maximum(np.linalg.norm(dir0, axis=1, keepdims=True), 1e-10)
    dir1 /= np.maximum(np.linalg.norm(dir1, axis=1, keepdims=True), 1e-10)
    dots = np.sum(dir0 * dir1, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    conflict[idx] = angles_deg >= min_view_angle_deg
    return conflict


def _build_face_adjacency(faces: np.ndarray) -> list[np.ndarray]:
    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    edge_to_face: dict[tuple[int, int], int] = {}

    for fi, face in enumerate(faces):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            prev = edge_to_face.get(edge)
            if prev is None:
                edge_to_face[edge] = fi
                continue
            adjacency[fi].add(prev)
            adjacency[prev].add(fi)

    return [
        np.fromiter(sorted(neighbors), dtype=np.int32)
        if neighbors else np.empty(0, dtype=np.int32)
        for neighbors in adjacency
    ]


def _compute_face_locked_views(
    fids: np.ndarray,
    n_faces: int,
    n_views: int,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    conflict_texels: np.ndarray,
    face_normals: np.ndarray,
    faces: np.ndarray,
    min_conflict_texels: int = _TEXTURE_CONFLICT_FACE_MIN_TEXELS,
    min_conflict_frac: float = _TEXTURE_CONFLICT_FACE_MIN_FRAC,
    min_view_coverage: float = _TEXTURE_CONFLICT_FACE_MIN_COVERAGE,
    smooth_dot: float = _TEXTURE_CONFLICT_SMOOTH_DOT,
    smooth_gain: float = _TEXTURE_CONFLICT_SMOOTH_GAIN,
    smooth_min_neighbors: int = _TEXTURE_CONFLICT_SMOOTH_MIN_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a single dominant view for conflict-heavy faces."""
    face_locked_view = np.full(n_faces, -1, dtype=np.int32)
    if n_faces == 0 or n_views <= 0 or best_scores.size == 0 or not np.any(conflict_texels):
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    face_texel_count = np.bincount(fids, minlength=n_faces).astype(np.int32)
    face_conflict_count = np.bincount(fids[conflict_texels], minlength=n_faces).astype(np.int32)
    face_conflict_frac = np.divide(
        face_conflict_count,
        np.maximum(face_texel_count, 1),
        dtype=np.float32,
    )
    candidate_faces = np.where(
        (face_conflict_count >= min_conflict_texels)
        & (face_conflict_frac >= min_conflict_frac)
    )[0]
    if candidate_faces.size == 0:
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    candidate_mask = np.zeros(n_faces, dtype=bool)
    candidate_mask[candidate_faces] = True
    candidate_rows = np.full(n_faces, -1, dtype=np.int32)
    candidate_rows[candidate_faces] = np.arange(candidate_faces.size, dtype=np.int32)

    active_faces_parts: list[np.ndarray] = []
    active_views_parts: list[np.ndarray] = []
    active_scores_parts: list[np.ndarray] = []
    for k in range(best_scores.shape[1]):
        views_k = best_views[:, k]
        active = (views_k >= 0) & conflict_texels & candidate_mask[fids]
        if not np.any(active):
            continue
        active_faces_parts.append(fids[active].astype(np.int32, copy=False))
        active_views_parts.append(views_k[active].astype(np.int32, copy=False))
        active_scores_parts.append(best_scores[active, k].astype(np.float32, copy=False))

    if not active_faces_parts:
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    active_faces = np.concatenate(active_faces_parts)
    active_views = np.concatenate(active_views_parts)
    active_scores = np.concatenate(active_scores_parts)
    pair_keys = active_faces.astype(np.int64) * np.int64(n_views) + active_views.astype(np.int64)
    order = np.argsort(pair_keys, kind="stable")
    pair_keys = pair_keys[order]
    active_faces = active_faces[order]
    active_views = active_views[order]
    active_scores = active_scores[order]

    run_starts = np.flatnonzero(np.r_[True, pair_keys[1:] != pair_keys[:-1]])
    run_ends = np.r_[run_starts[1:], len(pair_keys)]
    pair_faces = active_faces[run_starts]
    pair_views = active_views[run_starts]
    pair_support = np.add.reduceat(active_scores, run_starts).astype(np.float32, copy=False)
    pair_coverage = (run_ends - run_starts).astype(np.int32, copy=False)

    dominant_support = np.zeros(candidate_faces.size, dtype=np.float32)
    dominant_coverage = np.zeros(candidate_faces.size, dtype=np.int32)
    dominant_cols = np.full(candidate_faces.size, -1, dtype=np.int32)
    face_view_stats: dict[tuple[int, int], tuple[float, int]] = {}
    for face_i, view_i, support_i, coverage_i in zip(
        pair_faces.tolist(),
        pair_views.tolist(),
        pair_support.tolist(),
        pair_coverage.tolist(),
        strict=False,
    ):
        row = candidate_rows[int(face_i)]
        if row < 0:
            continue
        face_view_stats[(int(face_i), int(view_i))] = (float(support_i), int(coverage_i))
        if support_i > dominant_support[row]:
            dominant_support[row] = float(support_i)
            dominant_coverage[row] = int(coverage_i)
            dominant_cols[row] = int(view_i)

    coverage_ratio = np.divide(
        dominant_coverage,
        np.maximum(face_conflict_count[candidate_faces], 1),
        dtype=np.float32,
    )
    eligible = (dominant_support > 0.0) & (coverage_ratio >= min_view_coverage)
    if not np.any(eligible):
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    face_locked_view[candidate_faces[eligible]] = dominant_cols[eligible].astype(np.int32)
    face_support = np.zeros(n_faces, dtype=np.float32)
    face_support[candidate_faces[eligible]] = dominant_support[eligible]

    adjacency = _build_face_adjacency(faces)
    locked_faces = np.where(face_locked_view >= 0)[0]
    if locked_faces.size == 0:
        return face_locked_view, face_support

    smoothed = face_locked_view.copy()
    for fi in locked_faces:
        row = candidate_rows[fi]
        if row < 0:
            continue
        neighbors = adjacency[fi]
        if neighbors.size == 0:
            continue
        same_region = neighbors[smoothed[neighbors] >= 0]
        if same_region.size == 0:
            continue
        normal_dot = np.sum(face_normals[same_region] * face_normals[fi], axis=1)
        similar = same_region[normal_dot >= smooth_dot]
        if similar.size < smooth_min_neighbors:
            continue
        labels = smoothed[similar]
        weights = face_support[similar]
        totals: dict[int, float] = {}
        for label, weight in zip(labels.tolist(), weights.tolist(), strict=False):
            label_i = int(label)
            local_support, _local_coverage = face_view_stats.get((int(fi), label_i), (0.0, 0))
            if local_support <= 0.0:
                continue
            totals[label_i] = totals.get(label_i, 0.0) + float(weight)
        if not totals:
            continue
        best_label, best_total = max(totals.items(), key=lambda item: item[1])
        if best_label != int(smoothed[fi]) and best_total > float(face_support[fi]) * smooth_gain:
            smoothed[fi] = int(best_label)

    changed = smoothed != face_locked_view
    if np.any(changed):
        face_locked_view = smoothed
        for fi in np.where(changed)[0]:
            local_support, _local_coverage = face_view_stats.get((int(fi), int(face_locked_view[fi])), (0.0, 0))
            face_support[fi] = float(local_support)

    return face_locked_view, face_support


def _aggregate_face_view_stats(
    fids: np.ndarray,
    n_faces: int,
    n_views: int,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    face_mask: np.ndarray,
) -> tuple[dict[int, list[tuple[int, float, int]]], np.ndarray]:
    """Aggregate per-face view support from texel-level top-K scores."""
    face_texel_count = np.bincount(fids, minlength=n_faces).astype(np.int32)
    if n_faces == 0 or n_views <= 0 or best_scores.size == 0 or not np.any(face_mask):
        return {}, face_texel_count

    active_faces_parts: list[np.ndarray] = []
    active_views_parts: list[np.ndarray] = []
    active_scores_parts: list[np.ndarray] = []
    for k in range(best_scores.shape[1]):
        views_k = best_views[:, k]
        active = face_mask[fids] & (views_k >= 0) & (best_scores[:, k] > 0)
        if not np.any(active):
            continue
        active_faces_parts.append(fids[active].astype(np.int32, copy=False))
        active_views_parts.append(views_k[active].astype(np.int32, copy=False))
        active_scores_parts.append(best_scores[active, k].astype(np.float32, copy=False))

    if not active_faces_parts:
        return {}, face_texel_count

    active_faces = np.concatenate(active_faces_parts)
    active_views = np.concatenate(active_views_parts)
    active_scores = np.concatenate(active_scores_parts)
    pair_keys = active_faces.astype(np.int64) * np.int64(n_views) + active_views.astype(np.int64)
    order = np.argsort(pair_keys, kind="stable")
    pair_keys = pair_keys[order]
    active_faces = active_faces[order]
    active_views = active_views[order]
    active_scores = active_scores[order]

    run_starts = np.flatnonzero(np.r_[True, pair_keys[1:] != pair_keys[:-1]])
    run_ends = np.r_[run_starts[1:], len(pair_keys)]
    pair_faces = active_faces[run_starts]
    pair_views = active_views[run_starts]
    pair_support = np.add.reduceat(active_scores, run_starts).astype(np.float32, copy=False)
    pair_coverage = (run_ends - run_starts).astype(np.int32, copy=False)

    face_view_stats: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for face_i, view_i, support_i, coverage_i in zip(
        pair_faces.tolist(),
        pair_views.tolist(),
        pair_support.tolist(),
        pair_coverage.tolist(),
        strict=False,
    ):
        face_view_stats[int(face_i)].append((int(view_i), float(support_i), int(coverage_i)))

    for face_i, stats in face_view_stats.items():
        face_view_stats[face_i] = sorted(stats, key=lambda item: item[1], reverse=True)
    return dict(face_view_stats), face_texel_count


def _collect_region_components(
    seed_faces: np.ndarray,
    adjacency: list[np.ndarray],
    face_normals: np.ndarray,
    face_label_sets: dict[int, set[int]],
    min_normal_dot: float = _TEXTURE_REGION_NORMAL_DOT,
) -> list[np.ndarray]:
    if seed_faces.size == 0:
        return []

    visited = np.zeros(len(face_normals), dtype=bool)
    components: list[np.ndarray] = []

    for start in seed_faces.tolist():
        start_i = int(start)
        if visited[start_i]:
            continue
        stack = [start_i]
        visited[start_i] = True
        faces_in_component: list[int] = []
        while stack:
            fi = stack.pop()
            faces_in_component.append(fi)
            for nb in adjacency[fi].tolist():
                nb_i = int(nb)
                if visited[nb_i]:
                    continue
                if float(np.dot(face_normals[fi], face_normals[nb_i])) < min_normal_dot:
                    continue
                labels_fi = face_label_sets.get(fi)
                labels_nb = face_label_sets.get(nb_i)
                if not labels_fi or not labels_nb or labels_fi.isdisjoint(labels_nb):
                    continue
                visited[nb_i] = True
                stack.append(nb_i)
        components.append(np.array(faces_in_component, dtype=np.int32))
    return components


def _compute_region_gc_locked_views(
    fids: np.ndarray,
    n_faces: int,
    n_views: int,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    conflict_texels: np.ndarray,
    face_normals: np.ndarray,
    faces: np.ndarray,
    base_locked_view: np.ndarray,
    min_conflict_texels: int = _TEXTURE_CONFLICT_FACE_MIN_TEXELS,
    min_conflict_frac: float = _TEXTURE_CONFLICT_FACE_MIN_FRAC,
    min_region_faces: int = _TEXTURE_REGION_MIN_FACES,
    max_region_labels: int = _TEXTURE_REGION_TOP_LABELS,
    max_iters: int = _TEXTURE_REGION_MAX_ITERS,
    smoothness: float = _TEXTURE_REGION_SMOOTHNESS,
    min_label_coverage: float = _TEXTURE_REGION_MIN_LABEL_COVERAGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth ambiguous curved regions with face-graph label optimization."""
    face_locked_view = base_locked_view.copy()
    face_support = np.zeros(n_faces, dtype=np.float32)
    region_id_per_face = np.full(n_faces, -1, dtype=np.int32)
    if n_faces == 0 or n_views <= 0 or best_scores.size == 0 or not np.any(conflict_texels):
        return face_locked_view, face_support, region_id_per_face

    face_texel_count = np.bincount(fids, minlength=n_faces).astype(np.int32)
    face_conflict_count = np.bincount(fids[conflict_texels], minlength=n_faces).astype(np.int32)
    face_conflict_frac = np.divide(
        face_conflict_count,
        np.maximum(face_texel_count, 1),
        dtype=np.float32,
    )
    candidate_faces = np.where(
        (face_conflict_count >= min_conflict_texels)
        & (face_conflict_frac >= min_conflict_frac)
    )[0]
    if candidate_faces.size == 0:
        return face_locked_view, face_support, region_id_per_face

    textured_face_mask = face_texel_count > 0
    face_view_stats, face_texel_count = _aggregate_face_view_stats(
        fids=fids,
        n_faces=n_faces,
        n_views=n_views,
        best_scores=best_scores,
        best_views=best_views,
        face_mask=textured_face_mask,
    )
    if not face_view_stats:
        return face_locked_view, face_support, region_id_per_face

    adjacency = _build_face_adjacency(faces)
    face_label_sets = {
        int(face_i): {int(view_i) for view_i, _support_i, _coverage_i in stats[:max_region_labels]}
        for face_i, stats in face_view_stats.items()
    }
    components = _collect_region_components(
        seed_faces=candidate_faces,
        adjacency=adjacency,
        face_normals=face_normals,
        face_label_sets=face_label_sets,
    )
    region_counter = 0

    for component in components:
        if component.size < min_region_faces:
            continue

        component_views: dict[int, float] = defaultdict(float)
        face_costs: dict[int, dict[int, float]] = {}
        face_coverages: dict[int, dict[int, float]] = {}
        init_labels: dict[int, int] = {}
        eligible_faces: list[int] = []

        for face_i in component.tolist():
            stats = face_view_stats.get(int(face_i), [])
            if not stats:
                continue
            max_support_local = max(float(item[1]) for item in stats)
            if max_support_local <= 0.0:
                continue
            eligible_faces.append(int(face_i))
            local_costs: dict[int, float] = {}
            local_coverages: dict[int, float] = {}
            for view_i, support_i, coverage_i in stats[:max_region_labels]:
                component_views[int(view_i)] += float(support_i)
                coverage_ratio = float(coverage_i) / max(float(face_texel_count[int(face_i)]), 1.0)
                cost = 1.0 - (float(support_i) / max_support_local)
                if coverage_ratio < min_label_coverage:
                    ratio = coverage_ratio / max(min_label_coverage, 1e-6)
                    cost += _TEXTURE_REGION_LOW_COVERAGE_PENALTY * (1.0 - max(0.0, min(1.0, ratio)))
                local_costs[int(view_i)] = float(cost)
                local_coverages[int(view_i)] = coverage_ratio
            face_costs[int(face_i)] = local_costs
            face_coverages[int(face_i)] = local_coverages
            base_label = int(base_locked_view[int(face_i)])
            if base_label >= 0 and base_label in local_costs:
                init_labels[int(face_i)] = base_label

        if len(eligible_faces) < min_region_faces or len(component_views) < 2:
            continue

        allowed_labels = [
            int(view_i)
            for view_i, _support_i in sorted(component_views.items(), key=lambda item: item[1], reverse=True)[:max_region_labels]
        ]
        if len(allowed_labels) < 2:
            continue

        face_indices = np.array(sorted(eligible_faces), dtype=np.int32)
        face_set = set(face_indices.tolist())
        face_update_order: list[tuple[float, int]] = []
        current_labels: dict[int, int] = {}
        for face_i in face_indices.tolist():
            costs = face_costs[int(face_i)]
            ranked_local = sorted(
                (
                    costs.get(label, _TEXTURE_REGION_MISSING_COST),
                    -face_coverages[int(face_i)].get(label, 0.0),
                    int(label),
                )
                for label in allowed_labels
            )
            if len(ranked_local) >= 2:
                margin = float(ranked_local[1][0] - ranked_local[0][0])
            else:
                margin = _TEXTURE_REGION_MISSING_COST
            face_update_order.append((margin, int(face_i)))
            best_label = min(
                allowed_labels,
                key=lambda label: (
                    costs.get(label, _TEXTURE_REGION_MISSING_COST),
                    -face_coverages[int(face_i)].get(label, 0.0),
                    label,
                ),
            )
            current_labels[int(face_i)] = init_labels.get(int(face_i), int(best_label))
        ordered_faces = [face_i for _margin, face_i in sorted(face_update_order, key=lambda item: (item[0], item[1]))]

        edge_weights: dict[tuple[int, int], float] = {}
        for face_i in face_indices.tolist():
            for nb in adjacency[int(face_i)].tolist():
                nb_i = int(nb)
                if nb_i not in face_set or nb_i <= int(face_i):
                    continue
                normal_dot = float(np.clip(np.dot(face_normals[int(face_i)], face_normals[nb_i]), -1.0, 1.0))
                if normal_dot < _TEXTURE_REGION_NORMAL_DOT:
                    continue
                smooth_weight = smoothness * (
                    0.25 + 0.75 * (normal_dot - _TEXTURE_REGION_NORMAL_DOT) / max(1e-6, 1.0 - _TEXTURE_REGION_NORMAL_DOT)
                )
                edge_weights[(int(face_i), nb_i)] = max(0.0, float(smooth_weight))

        if not edge_weights:
            continue

        for _iter_idx in range(max_iters):
            changed = False
            for face_i in ordered_faces:
                costs = face_costs[int(face_i)]
                best_energy: tuple[float, float, int] | None = None
                best_label = current_labels[int(face_i)]
                for label in allowed_labels:
                    local_cost = costs.get(label, _TEXTURE_REGION_MISSING_COST)
                    smooth_cost = 0.0
                    for nb in adjacency[int(face_i)].tolist():
                        nb_i = int(nb)
                        if nb_i not in face_set:
                            continue
                        edge_key = (min(int(face_i), nb_i), max(int(face_i), nb_i))
                        weight = edge_weights.get(edge_key, 0.0)
                        if weight <= 0.0:
                            continue
                        if label != current_labels[nb_i]:
                            smooth_cost += weight
                    energy = (local_cost + smooth_cost, local_cost, int(label))
                    if best_energy is None or energy < best_energy:
                        best_energy = energy
                        best_label = int(label)
                if best_label != current_labels[int(face_i)]:
                    current_labels[int(face_i)] = best_label
                    changed = True
            if not changed:
                break

        assigned = 0
        for face_i in face_indices.tolist():
            chosen_label = current_labels[int(face_i)]
            coverage_ratio = face_coverages[int(face_i)].get(chosen_label, 0.0)
            if coverage_ratio < min_label_coverage:
                continue
            support_candidates = {view_i: support_i for view_i, support_i, _cov_i in face_view_stats.get(int(face_i), [])}
            support_i = float(support_candidates.get(chosen_label, 0.0))
            if support_i <= 0.0:
                continue
            face_locked_view[int(face_i)] = int(chosen_label)
            face_support[int(face_i)] = support_i
            region_id_per_face[int(face_i)] = region_counter
            assigned += 1

        if assigned >= min_region_faces:
            region_counter += 1
        else:
            for face_i in face_indices.tolist():
                face_locked_view[int(face_i)] = int(base_locked_view[int(face_i)])
                face_support[int(face_i)] = 0.0
            region_id_per_face[face_indices] = -1

    return face_locked_view, face_support, region_id_per_face
