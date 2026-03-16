"""Repair pipeline orchestration for contact-hole repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.config_defaults import (
    _REPAIR_GROUND_SECTION_START_QUANTILE as _GROUND_SECTION_START_QUANTILE,
    _REPAIR_MIN_LOOP_VERTICES as _MIN_LOOP_VERTICES,
)
from scripts.repair.boundary import (
    _apply_local_smoothing,
    _evaluate_loop,
    _extract_boundary_edges,
    _extract_boundary_paths,
    _fit_loop_normal,
    _has_non_manifold_edges,
)
from scripts.repair.candidates import _collect_repair_candidates
from scripts.repair.config import _resolve_params
from scripts.repair.ground_plane import (
    _cap_boundary_at_plane,
    _clip_mesh_at_plane,
    _extract_closed_section_loops,
    _find_ground_plane_shift,
    _ground_section_min_area,
    _orient_ground_plane_toward_mesh,
    _project_vertices_to_plane,
    _validate_ground_plane_output,
)
from scripts.repair.mesh_io import (
    _load_mesh_arrays,
    _prepare_repaired_mesh,
    _write_mesh_safe,
    _write_repaired_outputs,
)
from scripts.repair.triangulate import (
    _loop_projection_uv,
    _triangulate_polygon_ear_clip,
)
from scripts.repair.types import (
    CancelCallback,
    GroundPlaneValidation,
    ProgressCallback,
    RepairStats,
    _check_cancel,
    _emit_progress,
)


def _repair_contact_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    params: 'ContactHoleRepairParams',
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
    target_loop: 'GroundPlaneSectionLoop',
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
            cap_vertex_ids=cap_vertex_ids,
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
            print(f"  seed stable offset: {search.first_valid_shift:.4f}")
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
        print(
            f"  section loops at clip plane: {len(section_loops)}, "
            f"selected area: {selected_section_loop.area:.6f}"
        )

        # Save section loops across the full scan range for dashboard debug
        _section_loop_path = repair_dir / "section_loop.json"
        _sl_dists = vertices @ gp_normal + gp_d
        _sl_dist_min = float(_sl_dists.min())
        _sl_start = float(np.quantile(_sl_dists, _GROUND_SECTION_START_QUANTILE))
        _sl_bbox_diag = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-6)
        _sl_pad = max(1e-6, 1e-4 * _sl_bbox_diag)
        _sl_sample_shifts = np.linspace(_sl_dist_min + _sl_pad, _sl_start, 32)
        _section_loop_data: dict = {
            "selected_shift": float(selected_shift),
            "plane_normal": gp_normal.tolist(),
            "plane_d": float(gp_d),
            "samples": [],
        }
        for _sh in _sl_sample_shifts:
            _sh_f = float(_sh)
            _sh_d = gp_d - _sh_f
            _sh_loops = _extract_closed_section_loops(vertices, faces, gp_normal, _sh_d)
            _sample: dict = {"shift": _sh_f, "loops": []}
            for _sl in sorted(_sh_loops, key=lambda l: l.area, reverse=True):
                pts = _sl.points.tolist()
                if len(pts) > 1 and pts[0] != pts[-1]:
                    pts.append(pts[0])
                _sample["loops"].append({"area": float(_sl.area), "points": pts})
            _section_loop_data["samples"].append(_sample)
        _section_loop_path.write_text(json.dumps(_section_loop_data))
        print(f"  section loop debug: {_section_loop_path} ({len(_sl_sample_shifts)} samples)")

        _emit_progress(progress_cb, 15.0, "Contact Hole Repair: clipping at ground plane")
        _check_cancel(cancel_cb)
        clipped_verts, clipped_faces, _ = _clip_mesh_at_plane(
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
