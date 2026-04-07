"""Stage 7: Texture baking from camera-projected views.

This module is a backward-compatible shim. The implementation has been
split into the ``scripts.texture`` subpackage. All public and private
names are re-exported here so that existing imports continue to work.
"""

# Re-export everything from the texture subpackage
from scripts.texture.progress import (  # noqa: F401
    ProgressCallback,
    _emit_progress,
    _get_available_memory_mb,
)
from scripts.texture.io_utils import (  # noqa: F401
    _FrameCache,
    _list_indexed_files,
    _load_frame,
    _load_mask,
    _load_point_cloud,
    _load_poses,
    _resolve_indexed_file,
)
from scripts.texture.intrinsics import (  # noqa: F401
    _estimate_intrinsics,
    _make_K,
    _project_points,
    _project_simple,
)
from scripts.texture.config import (  # noqa: F401
    _resolve_texture_device,
    _resolve_texture_quality_boost,
    _resolve_texture_size,
    _resolve_texture_view_assign_mode,
    _resolve_uv_face_budget,
)
from scripts.texture.image_utils import (  # noqa: F401
    _bilinear_sample,
    _masked_detail_advantage,
    _masked_gaussian,
    _masked_gradient_difference,
    _masked_gray_difference,
    _masked_rgb_difference,
    _rgb_to_gray,
    _translate_patch,
)
from scripts.texture.view_scoring import (  # noqa: F401
    _apply_view_hardening,
    _evaluate_view_packet,
    _evaluate_view_samples,
    _rasterize_view_depth,
    _score_view_packet,
    _update_topk_scores,
    ViewEvalPacket,
    ViewPacketShapeError,
)
from scripts.texture.conflict_region import (  # noqa: F401
    _TEXTURE_CONFLICT_FACE_MIN_COVERAGE,
    _TEXTURE_CONFLICT_FACE_MIN_FRAC,
    _TEXTURE_CONFLICT_FACE_MIN_TEXELS,
    _TEXTURE_CONFLICT_RATIO,
    _TEXTURE_CONFLICT_SMOOTH_DOT,
    _TEXTURE_CONFLICT_SMOOTH_GAIN,
    _TEXTURE_CONFLICT_SMOOTH_MIN_NEIGHBORS,
    _TEXTURE_CONFLICT_VIEW_ANGLE_DEG,
    _TEXTURE_REGION_LOW_COVERAGE_PENALTY,
    _TEXTURE_REGION_MAX_ITERS,
    _TEXTURE_REGION_MIN_FACES,
    _TEXTURE_REGION_MIN_LABEL_COVERAGE,
    _TEXTURE_REGION_MISSING_COST,
    _TEXTURE_REGION_NORMAL_DOT,
    _TEXTURE_REGION_SMOOTHNESS,
    _TEXTURE_REGION_TOP_LABELS,
    _aggregate_face_view_stats,
    _build_face_adjacency,
    _collect_region_components,
    _compute_conflict_texels,
    _compute_face_locked_views,
    _compute_region_gc_locked_views,
)
from scripts.texture.seam_blend import (  # noqa: F401
    _TEXTURE_SEAM_HQ_COLOR_EPS,
    _TEXTURE_SEAM_HQ_DETAIL_GAIN,
    _TEXTURE_SEAM_HQ_DETAIL_WEIGHT,
    _TEXTURE_SEAM_HQ_DILATE,
    _TEXTURE_SEAM_HQ_ECC_EPS,
    _TEXTURE_SEAM_HQ_ECC_ITERS,
    _TEXTURE_SEAM_HQ_GRADIENT_WEIGHT,
    _TEXTURE_SEAM_HQ_LOW_BLEND,
    _TEXTURE_SEAM_HQ_MAX_COMPANIONS,
    _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS,
    _TEXTURE_SEAM_HQ_MIN_OVERLAP_RATIO,
    _TEXTURE_SEAM_HQ_PATCH_PAD,
    _TEXTURE_SEAM_HQ_RESPONSE_FLOOR,
    _TEXTURE_SEAM_HQ_SHIFT_LIMIT,
    _TEXTURE_SEAM_HQ_SIGMA,
    _TEXTURE_SEAM_LEVEL_BLEND,
    _TEXTURE_SEAM_LEVEL_DILATE,
    _TEXTURE_SEAM_LEVEL_SIGMA,
    _TEXTURE_SEAM_LOW_SIGMA,
    _TEXTURE_SEAM_SMOOTH_SIGMA,
    _apply_narrow_seam_leveling,
    _apply_quality_boost_detail_refinement,
    _compute_label_boundary,
    _estimate_detail_band_shift,
    _merge_detail_component,
    _normalize_candidate_patch_colors,
    _refine_detail_band_shift,
    _sample_companion_patch,
    _sample_companion_texture,
    _select_best_detail_candidate,
    _select_detail_companion_view_candidates,
    _select_detail_companion_views,
)
from scripts.texture.cap_region import (  # noqa: F401
    _fill_cap_region,
    _identify_cap_texels,
    _seed_cap_border,
)
from scripts.texture.bake import bake_texture  # noqa: F401

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Texture baking")
    parser.add_argument("mesh_ply", help="Mesh PLY file")
    parser.add_argument("--poses", required=True, help="camera_poses.json")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    bake_texture(args.mesh_ply, args.poses, args.frames_dir, args.masks_dir, args.output_dir)
