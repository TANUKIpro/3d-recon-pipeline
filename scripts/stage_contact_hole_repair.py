"""Stage 7: mesh repair for wrapped meshes.

This module is a backward-compatible shim. The implementation has been
split into the ``scripts.repair`` subpackage. All public and private
names are re-exported here so that existing imports continue to work.
"""

# Re-export everything from the repair subpackage
from scripts.repair.types import (  # noqa: F401
    CancelCallback,
    ContactHoleRepairParams,
    GroundPlaneBoundaryLoop,
    GroundPlaneProbe,
    GroundPlaneSearchResult,
    GroundPlaneSectionLoop,
    GroundPlaneValidation,
    LoopEval,
    ProgressCallback,
    RepairCandidateAnalysis,
    RepairLoopCandidate,
    RepairStats,
    _check_cancel,
    _emit_progress,
)
from scripts.repair.config import (  # noqa: F401
    _env_bool,
    _env_float,
    _env_int,
    _resolve_params,
)
from scripts.repair.mesh_io import (  # noqa: F401
    _clean_mesh,
    _load_mesh_arrays,
    _prepare_repaired_mesh,
    _write_mesh_safe,
    _write_repaired_outputs,
)
from scripts.repair.triangulate import (  # noqa: F401
    _cross2,
    _loop_projection_uv,
    _point_in_tri_2d,
    _polygon_area_2d,
    _triangulate_polygon_ear_clip,
)
from scripts.repair.boundary import (  # noqa: F401
    _apply_local_smoothing,
    _build_boundary_adjacency,
    _build_vertex_adjacency,
    _candidate_from_path,
    _edge_key,
    _evaluate_loop,
    _extract_boundary_edges,
    _extract_boundary_paths,
    _fit_loop_normal,
    _has_non_manifold_edges,
)
from scripts.repair.candidates import (  # noqa: F401
    _collect_repair_candidates,
    analyze_contact_hole_candidates,
)
from scripts.repair.ground_plane import (  # noqa: F401
    _cap_boundary_at_plane,
    _clip_mesh_at_plane,
    _collect_plane_boundary_loop_infos,
    _collect_plane_boundary_loops,
    _count_plane_boundary_edges,
    _extract_closed_section_loops,
    _extract_plane_intersection_segments,
    _find_ground_plane_shift,
    _find_optimal_clip_offset,
    _ground_section_min_area,
    _normalize_plane,
    _orient_ground_plane_toward_mesh,
    _plane_distance_threshold,
    _probe_ground_plane_shift,
    _project_vertices_to_plane,
    _projected_bbox_area_on_plane,
    _select_matching_plane_boundary_loop,
    _validate_ground_plane_output,
)
from scripts.repair.pipeline import (  # noqa: F401
    _finalize_ground_plane_outputs,
    _print_repair_stats,
    _repair_contact_holes,
    run_contact_hole_repair,
    run_selected_contact_hole_repair,
)

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

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
