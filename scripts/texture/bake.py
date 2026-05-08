"""Main texture baking orchestrator."""

import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

from scripts.config_defaults import (
    TEXTURE_ANGLE_EXP,
    TEXTURE_BLEND_HARD_RATIO,
    TEXTURE_BLEND_TOPK,
    TEXTURE_COLOR_HARDENING,
    TEXTURE_COLOR_HARDENING_THRESHOLD,
    TEXTURE_DIST_POW,
    TEXTURE_MIN_COS,
    TEXTURE_OVERSAMPLE,
    TEXTURE_SHARPEN,
    _TEXTURE_SEAM_PAD_ITERS,
)
from scripts.mesh_orientation import orient_mesh_for_cameras
from scripts.output_layout import (
    colmap_sparse_dir,
    intrinsics_path as _intrinsics_path,
    texture_phase_dir,
    texture_png_path,
    textured_mesh_mtl_path,
    textured_mesh_obj_path,
)
from scripts.texture.cap_region import _fill_cap_region, _identify_cap_texels, _seed_cap_border
from scripts.texture.config import (
    _resolve_texture_device,
    _resolve_texture_quality_boost,
    _resolve_texture_size,
    _resolve_texture_view_assign_mode,
    _resolve_uv_face_budget,
)
from scripts.texture.conflict_region import (
    _compute_conflict_texels,
    _compute_face_locked_views,
    _compute_region_gc_locked_views,
)
from scripts.texture.image_utils import _bilinear_sample, _seam_pad_gpu
from scripts.texture.intrinsics import _estimate_intrinsics
from scripts.texture.io_utils import _FrameCache, _load_frame, _load_point_cloud, _load_poses
from scripts.texture.progress import ProgressCallback, _emit_progress, _get_available_memory_mb
from scripts.texture.seam_blend import (
    _TEXTURE_SEAM_HQ_DILATE,
    _TEXTURE_SEAM_HQ_SIGMA,
    _TEXTURE_SEAM_LEVEL_BLEND,
    _TEXTURE_SEAM_LEVEL_DILATE,
    _TEXTURE_SEAM_LEVEL_SIGMA,
    _apply_narrow_seam_leveling,
    _apply_quality_boost_detail_refinement,
)
from scripts.texture.view_scoring import (
    _apply_view_hardening,
    _evaluate_view_samples,
    _rasterize_view_depth,
    _update_topk_scores,
)


def _resolve_texture_color_hardening() -> bool:
    raw = os.environ.get("TEXTURE_COLOR_HARDENING")
    if raw is None:
        return bool(TEXTURE_COLOR_HARDENING)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_texture_color_hardening_threshold() -> float:
    raw = os.environ.get("TEXTURE_COLOR_HARDENING_THRESHOLD")
    if raw is None:
        return float(TEXTURE_COLOR_HARDENING_THRESHOLD)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return float(TEXTURE_COLOR_HARDENING_THRESHOLD)


def _array_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _compute_color_instability(
    candidate_colors: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted RGB spread and valid candidate count per texel.

    The spread is the weighted RMS Euclidean distance from each texel's
    top-K candidate colors to their weighted mean. It is near zero when
    candidate views agree and rises when top-K views see incompatible colors,
    which is typical for transparent/specular surfaces that expose background
    content or view-dependent reflections.
    """
    valid = candidate_valid & (candidate_weights > 0)
    counts = valid.sum(axis=1).astype(np.int32)
    spread = np.zeros(candidate_colors.shape[0], dtype=np.float32)
    usable = counts >= 2
    if not np.any(usable):
        return spread, counts

    weights = np.where(valid, candidate_weights, 0.0).astype(np.float64)
    weight_sum = weights.sum(axis=1)
    usable &= weight_sum > 0
    if not np.any(usable):
        return spread, counts

    colors = candidate_colors.astype(np.float64, copy=False)
    norm_w = np.zeros_like(weights, dtype=np.float64)
    norm_w[usable] = weights[usable] / weight_sum[usable, None]
    mean = np.sum(colors * norm_w[:, :, None], axis=1)
    diff = colors - mean[:, None, :]
    sq_dist = np.sum(diff * diff, axis=2)
    spread[usable] = np.sqrt(np.sum(norm_w[usable] * sq_dist[usable], axis=1)).astype(
        np.float32
    )
    return spread, counts


def _apply_color_instability_hardening(
    best_scores: np.ndarray,
    best_views: np.ndarray,
    color_instability: np.ndarray,
    candidate_counts: np.ndarray,
    threshold: float,
    eligible_mask: np.ndarray | None = None,
) -> int:
    """Single-view texels whose top-K colors disagree too much."""
    if best_scores.shape[1] <= 1 or threshold <= 0:
        return 0
    unstable = (
        (best_views[:, 0] >= 0)
        & (best_scores[:, 0] > 0)
        & (candidate_counts >= 2)
        & (color_instability > threshold)
    )
    if eligible_mask is not None:
        unstable &= eligible_mask
    n_hard = int(np.count_nonzero(unstable))
    if n_hard > 0:
        best_scores[unstable, 1:] = -1.0
        best_views[unstable, 1:] = -1
    return n_hard


def _cache_depth_buffer(
    depth_cache: OrderedDict[int, np.ndarray],
    max_depth_cache: int,
    vidx: int,
    depth_buffer: np.ndarray,
) -> None:
    if max_depth_cache <= 0 or vidx in depth_cache:
        return
    depth_cache[vidx] = depth_buffer
    while len(depth_cache) > max_depth_cache:
        depth_cache.popitem(last=False)


def _collect_topk_candidate_colors(
    *,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    pos3d: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
    pose_frame_indices: list[int],
    frames_dir: str,
    mask_dir: str,
    frame_cache: _FrameCache,
    depth_cache: OrderedDict[int, np.ndarray],
    max_depth_cache: int,
    new_vertices: np.ndarray,
    new_faces: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
    texture_device: str,
    dist_coeffs: np.ndarray | list | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_texels, topk = best_views.shape
    candidate_colors = np.zeros((n_texels, topk, 3), dtype=np.float32)
    candidate_valid = np.zeros((n_texels, topk), dtype=bool)

    for k in range(topk):
        col_views = best_views[:, k]
        col_scores = best_scores[:, k]
        for vidx_val in np.unique(col_views):
            vidx = int(vidx_val)
            if vidx < 0:
                continue
            tidx = np.where((col_views == vidx) & (col_scores > 0))[0]
            if tidx.size == 0:
                continue

            src_idx = int(pose_frame_indices[vidx])
            c2w = poses[vidx]
            try:
                frame = frame_cache.load_frame(frames_dir, src_idx)
                mask_bool = frame_cache.load_mask(mask_dir, src_idx)
            except FileNotFoundError:
                continue

            if vidx in depth_cache:
                depth_buffer = depth_cache[vidx]
            else:
                depth_buffer = _rasterize_view_depth(
                    new_vertices,
                    new_faces,
                    c2w,
                    K,
                    img_w,
                    img_h,
                    device=texture_device,
                    dist_coeffs=dist_coeffs,
                )
                _cache_depth_buffer(depth_cache, max_depth_cache, vidx, depth_buffer)

            valid, _score, px_proj, py_proj = _evaluate_view_samples(
                pos3d=pos3d[tidx],
                normals=normals[tidx],
                c2w=c2w,
                K=K,
                img_w=img_w,
                img_h=img_h,
                mask_bool=mask_bool,
                depth_buffer=depth_buffer,
                min_cos=min_cos,
                angle_exp=angle_exp,
                dist_pow=dist_pow,
                device=texture_device,
                dist_coeffs=dist_coeffs,
            )
            if not np.any(valid):
                continue

            global_tidx = tidx[valid]
            candidate_colors[global_tidx, k] = _bilinear_sample(
                frame,
                px_proj[valid],
                py_proj[valid],
            ).astype(np.float32)
            candidate_valid[global_tidx, k] = True

    return candidate_colors, candidate_valid

def _maybe_load_intrinsics(
    path: Path,
    img_w: int,
    img_h: int,
) -> dict | None:
    """Return a pre-computed intrinsics dict if it exists and matches the image size.

    The COLMAP stage writes this file during SfM and the texture stage writes
    the same schema on the slow path; either producer is fine to reuse on
    subsequent runs.  A resolution mismatch is a signal that the upstream
    frames were re-extracted at a different size, so we fall back to
    re-estimation.
    """
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    required_fields = ("fx", "fy", "cx", "cy", "image_width", "image_height", "K")
    if not all(field in data for field in required_fields):
        return None

    if int(data["image_width"]) != img_w or int(data["image_height"]) != img_h:
        print(
            f"  Cached intrinsics resolution mismatch "
            f"({data['image_width']}x{data['image_height']} vs {img_w}x{img_h}); "
            "re-estimating"
        )
        return None

    return data


def _maybe_load_colmap_intrinsics(
    output_path: Path,
    img_w: int,
    img_h: int,
) -> dict | None:
    """Recover COLMAP intrinsics directly from cameras.bin as a defensive fallback.

    Stage 2 writes ``intrinsics.json`` at the end of SfM, but the file can
    go missing on resume/restart (e.g., a stale checkpoint cleaning logic
    wipes it). COLMAP's bundle-adjusted pinhole values are far more
    accurate than the texture stage's grid-search + Nelder-Mead path,
    especially on cylindrical or self-similar surfaces where the
    estimator can fall into a sub-pixel-skewed local minimum and destroy
    fine texture detail. Read straight from cameras.bin so the bake is
    independent of intrinsics.json's presence.
    """
    cameras_bin = colmap_sparse_dir(output_path) / "0" / "cameras.bin"
    if not cameras_bin.exists():
        return None
    try:
        from scripts.stage_colmap_sfm import _read_colmap_intrinsics

        data = _read_colmap_intrinsics(cameras_bin.parent)
    except Exception as exc:  # pragma: no cover — best-effort fallback
        print(f"  COLMAP intrinsics fallback failed: {exc}")
        return None
    if data is None:
        return None
    if int(data.get("image_width", 0)) != img_w or int(data.get("image_height", 0)) != img_h:
        return None
    return data


def _resolve_intrinsics_point_cloud(
    output_path: Path,
    mesh_vertices: np.ndarray,
    mesh_vertex_colors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the best available colored point cloud for intrinsics fitting."""
    for candidate in ("object_denoised.ply", "object.ply"):
        path = output_path / candidate
        if path.exists():
            return _load_point_cloud(str(path))

    if mesh_vertex_colors is not None:
        return mesh_vertices, mesh_vertex_colors

    raise FileNotFoundError(
        "Missing colored point cloud for intrinsics estimation. "
        "Expected object_denoised.ply, object.ply, or vertex colors in mesh_ply."
    )


def _simplify_mesh_for_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        vertices.astype(np.float64, copy=False)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        faces.astype(np.int32, copy=False)
    )

    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_faces)
    )
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_unreferenced_vertices()

    simp_vertices = np.asarray(simplified.vertices, dtype=np.float64)
    simp_faces = np.asarray(simplified.triangles, dtype=np.int32)
    if len(simp_faces) == 0 or len(simp_vertices) == 0:
        raise RuntimeError(
            "UV proxy mesh simplification produced an empty mesh."
        )

    return simp_vertices, simp_faces


def _maybe_simplify_mesh_for_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_budget: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if face_budget <= 0 or len(faces) <= face_budget:
        return vertices, faces, False

    simplified_vertices, simplified_faces = _simplify_mesh_for_uv(
        vertices,
        faces,
        face_budget,
    )
    if len(simplified_faces) >= len(faces):
        return vertices, faces, False

    return simplified_vertices, simplified_faces, True


def bake_texture(
    mesh_ply: str,
    poses_path: str,
    frames_dir: str,
    mask_dir: str,
    output_dir: str,
    tex_size: int | None = None,
    view_assign_mode: str | None = None,
    quality_boost: bool | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Full texture baking pipeline: intrinsics → UV atlas → bake → export.

    Args:
        mesh_ply: Path to mesh PLY file.
        poses_path: Path to camera_poses.json.
        frames_dir: Path to JPEG frames directory.
        mask_dir: Path to mask PNGs directory.
        output_dir: Output directory for OBJ/MTL/PNG.
        tex_size: Texture resolution.
            If unset, reads TEXTURE_SIZE (default 0).
            TEXTURE_SIZE<=0 enables auto mode:
            sqrt(image_width * image_height), rounded to nearest int.

    Environment toggles for quality:
        TEXTURE_BLEND_TOPK (int): Number of views to blend per texel (default 3, 1=no blend).
        TEXTURE_DEVICE (str): 'cuda' (default), 'auto', or 'cpu' request hint.
            Current scoring path runs on CPU for deterministic z-buffering.
        TEXTURE_OVERSAMPLE (int): Internal supersampling multiplier (1=off, 2=default).
        TEXTURE_MIN_COS (float): Minimum normal·view cosine to accept a sample (default 0.2).
        TEXTURE_ANGLE_EXP (float): Exponent on cosine score (default 4.0).
        TEXTURE_DIST_POW (float): Distance falloff power for score (default 1.0).
        TEXTURE_SHARPEN (float): Unsharp mask amount (0=off, 0.1–0.4 typical).
        TEXTURE_BLEND_HARD_RATIO (float): When top-1/top-2 score ratio exceeds this,
            use single dominant view (default 2.0, 0=disabled).
        TEXTURE_QUALITY_BOOST (bool): Enables boundary-focused detail refinement
            for Region Optimized mode (default False).

    Returns:
        Path to the output OBJ file.
    """
    import xatlas

    oversample = max(1, int(os.environ.get("TEXTURE_OVERSAMPLE", str(TEXTURE_OVERSAMPLE))))
    min_cos = float(os.environ.get("TEXTURE_MIN_COS", str(TEXTURE_MIN_COS)))
    angle_exp = float(os.environ.get("TEXTURE_ANGLE_EXP", str(TEXTURE_ANGLE_EXP)))
    dist_pow = float(os.environ.get("TEXTURE_DIST_POW", str(TEXTURE_DIST_POW)))
    sharpen_amt = float(os.environ.get("TEXTURE_SHARPEN", str(TEXTURE_SHARPEN)))
    blend_topk = max(1, int(os.environ.get("TEXTURE_BLEND_TOPK", str(TEXTURE_BLEND_TOPK))))
    hard_ratio = float(os.environ.get("TEXTURE_BLEND_HARD_RATIO", str(TEXTURE_BLEND_HARD_RATIO)))
    color_hardening = _resolve_texture_color_hardening()
    color_hardening_threshold = _resolve_texture_color_hardening_threshold()
    texture_device = _resolve_texture_device()
    texture_view_assign_mode = _resolve_texture_view_assign_mode(view_assign_mode)
    texture_quality_boost = _resolve_texture_quality_boost(quality_boost)
    if texture_view_assign_mode == "region_gc" and texture_quality_boost:
        oversample = max(oversample, 3)
        blend_topk = max(blend_topk, 4)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_cb, 4.0, "Loading mesh")

    # --- Load mesh ---
    print("Loading mesh...")
    ply = PlyData.read(mesh_ply)
    v = ply["vertex"]
    vertices = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    f = ply["face"]
    faces = np.vstack(f["vertex_indices"]).astype(np.int32)
    print(f"  {len(vertices)} verts, {len(faces)} faces")

    # Try to read vertex colors (used to detect cap faces from repair stage)
    vert_colors = None
    try:
        vert_colors = np.column_stack(
            [v["red"], v["green"], v["blue"]]
        ).astype(np.float64) / 255.0
    except (ValueError, KeyError):
        pass

    # --- Load camera data ---
    poses, pose_frame_indices = _load_poses(poses_path)

    if len(poses) == 0:
        raise RuntimeError("No poses found in camera_poses.json")

    # Determine image size from first pose-linked frame.
    first_frame = _load_frame(frames_dir, int(pose_frame_indices[0]))
    img_h, img_w = first_frame.shape[:2]

    # The frame cache is constructed (with optional undistortion) once
    # intrinsics are resolved, since the distortion coefficients live in
    # the same JSON. Until then, callers should not load frames.
    frame_cache: _FrameCache | None = None

    tex_size, tex_is_auto = _resolve_texture_size(tex_size, img_w, img_h)
    tex_res = tex_size * oversample
    print(
        f"  Image size: {img_w}x{img_h}, {len(poses)} poses "
        f"(frame range: {min(pose_frame_indices)}..{max(pose_frame_indices)})"
    )
    if tex_is_auto:
        print(
            "  Texture size: auto -> %dx%d (from input %dx%d, pixel-equivalent square)"
            % (tex_size, tex_size, img_w, img_h)
        )
    else:
        print(f"  Texture size: manual -> {tex_size}x{tex_size}")

    # --- Load or estimate intrinsics ---
    # COLMAP's SfM stage already solves for fx/fy/cx/cy during bundle
    # adjustment and writes them to <output_dir>/intrinsics.json. Reuse that
    # whenever the schema matches our image resolution instead of re-solving
    # via the grid-search + Nelder-Mead path in _estimate_intrinsics (which
    # dominates the stage's wall-clock).
    intrinsics_file = _intrinsics_path(output_path)
    intrinsics = _maybe_load_intrinsics(intrinsics_file, img_w, img_h)
    if intrinsics is None:
        # Defensive fallback: pull directly from COLMAP cameras.bin if the
        # JSON copy went missing (e.g., a checkpoint reset wiped it).
        intrinsics = _maybe_load_colmap_intrinsics(output_path, img_w, img_h)
        if intrinsics is not None:
            print(
                f"  intrinsics.json missing; recovered from COLMAP "
                f"cameras.bin (source={intrinsics.get('source', 'colmap')})"
            )
            intrinsics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(intrinsics_file, "w") as f_out:
                json.dump(intrinsics, f_out, indent=2)
    elif (
        "dist_coeffs" not in intrinsics
        and str(intrinsics.get("source", "")).startswith("colmap:")
    ):
        # Older intrinsics.json files predate the distortion fields; refresh
        # from cameras.bin so undistortion can run on resume.
        refreshed = _maybe_load_colmap_intrinsics(output_path, img_w, img_h)
        if refreshed is not None:
            intrinsics = refreshed
            print(
                "  intrinsics.json upgraded with COLMAP distortion params "
                f"(model={intrinsics.get('distortion_model', '?')})"
            )
            with open(intrinsics_file, "w") as f_out:
                json.dump(intrinsics, f_out, indent=2)

    if intrinsics is not None:
        source = intrinsics.get("source", "cached")
        print(f"Loaded camera intrinsics from {intrinsics_file} (source={source})")
        _emit_progress(progress_cb, 52.0, "Camera intrinsics loaded from COLMAP")
    else:
        # Prefer the original colored point cloud when present, but fall back to
        # the stage-4 mesh if that's the only artifact available on resume.
        pc_points, pc_colors = _resolve_intrinsics_point_cloud(
            output_path,
            vertices,
            vert_colors,
        )

        print("Estimating camera intrinsics...")
        _emit_progress(progress_cb, 15.0, "Estimating camera intrinsics")
        _intrinsics_progress_last = {"value": 15.0}

        def _intrinsics_progress(local_progress: float, detail: str | None = None) -> None:
            stage_progress = 15.0 + (max(0.0, min(100.0, local_progress)) * 0.35)
            if stage_progress - _intrinsics_progress_last["value"] >= 0.5:
                _emit_progress(progress_cb, stage_progress, detail)
                _intrinsics_progress_last["value"] = stage_progress

        # Estimation uses uncached frame loads (cache is built below once we
        # know whether to apply COLMAP undistortion).
        intrinsics = _estimate_intrinsics(
            pc_points, pc_colors, poses, pose_frame_indices, frames_dir, mask_dir, img_w, img_h,
            progress_cb=_intrinsics_progress,
            cache=None,
        )
        _emit_progress(progress_cb, 52.0, "Camera intrinsics estimated")

        intrinsics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(intrinsics_file, "w") as f_out:
            json.dump(intrinsics, f_out, indent=2)

    K = np.array(intrinsics["K"], dtype=np.float64)
    dist_coeffs = intrinsics.get("dist_coeffs")
    print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    frame_cache = _FrameCache(img_w, img_h)
    if dist_coeffs is not None:
        dist_arr = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        if np.any(np.abs(dist_arr) > 1e-12):
            dist_label = intrinsics.get("distortion_model", "?")
            print(
                "  Projection model: %s (dist=[%s]) — sampling original frames"
                % (dist_label, ", ".join(f"{c:.5f}" for c in dist_arr))
            )

    # --- UV Atlas ---
    uv_face_budget, uv_budget_source = _resolve_uv_face_budget(len(faces))
    if uv_face_budget == 0:
        print(f"  UV face budget: unlimited (source={uv_budget_source})")
    else:
        print(f"  UV face budget: {uv_face_budget:,} faces (source={uv_budget_source})")
    uv_vertices, uv_faces, uv_simplified = _maybe_simplify_mesh_for_uv(
        vertices,
        faces,
        uv_face_budget,
    )
    if uv_simplified:
        print(
            "  UV proxy simplification: "
            f"{len(faces)} -> {len(uv_faces)} faces "
            f"(budget {uv_face_budget:,})"
        )

    print("Generating UV atlas...")
    _emit_progress(progress_cb, 58.0, "Generating UV atlas")

    # Compute per-vertex normals for better chart segmentation
    _face_n = np.cross(
        uv_vertices[uv_faces[:, 1]] - uv_vertices[uv_faces[:, 0]],
        uv_vertices[uv_faces[:, 2]] - uv_vertices[uv_faces[:, 0]],
    )
    _vert_normals = np.zeros_like(uv_vertices)
    for _k in range(3):
        np.add.at(_vert_normals, uv_faces[:, _k], _face_n)
    _vn_len = np.linalg.norm(_vert_normals, axis=1, keepdims=True)
    _vert_normals /= np.maximum(_vn_len, 1e-10)

    # Try parallel spatial-partition UV generation first
    from scripts.texture.parallel_uv import parallel_xatlas_generate

    import time as _time

    _uv_t0 = _time.monotonic()
    _parallel_result = parallel_xatlas_generate(
        uv_vertices, uv_faces, _vert_normals, progress_cb=progress_cb,
    )
    if _parallel_result is not None:
        vmapping, new_faces, uvs = _parallel_result
    else:
        # Fallback: single-threaded xatlas
        print(
            f"  Single-threaded xatlas: {len(uv_faces):,} faces, "
            f"{len(uv_vertices):,} vertices..."
        )
        _atlas = xatlas.Atlas()
        _atlas.add_mesh(
            uv_vertices.astype(np.float32),
            uv_faces.astype(np.uint32),
            normals=_vert_normals.astype(np.float32),
        )
        _chart_options = xatlas.ChartOptions()
        _pack_options = xatlas.PackOptions()
        _pack_options.blockAlign = True       # 4x4 block align — fewer packing candidates
        _pack_options.padding = 1
        print("  Running xatlas chart segmentation + packing...")
        _atlas.generate(chart_options=_chart_options, pack_options=_pack_options)
        vmapping, new_faces, uvs = _atlas[0]
    _uv_dt = _time.monotonic() - _uv_t0
    print(f"  UV atlas generated in {_uv_dt:.1f}s")
    new_vertices = uv_vertices[vmapping]
    orientation = orient_mesh_for_cameras(
        new_vertices, new_faces, poses
    )
    new_faces = orientation.faces
    print(f"  {len(new_vertices)} verts, {len(new_faces)} faces")
    if orientation.camera_ratio_before is not None and orientation.camera_ratio_after is not None:
        print(
            "  Orientation check: source=%s, camera_facing supplied/flipped %.3f/%.3f"
            % (
                orientation.source,
                orientation.camera_ratio_before,
                orientation.camera_ratio_after,
            )
        )
    if orientation.flipped:
        print(
            "  Orientation fix: flipped UV faces "
            "via %s (outward_ratio %s -> %s)"
            % (
                orientation.source,
                (
                    f"{orientation.outward_ratio_before:.3f}"
                    if orientation.outward_ratio_before is not None
                    else "n/a"
                ),
                (
                    f"{orientation.outward_ratio_after:.3f}"
                    if orientation.outward_ratio_after is not None
                    else "n/a"
                ),
            )
        )

    # Face normals
    v0 = new_vertices[new_faces[:, 0]]
    v1 = new_vertices[new_faces[:, 1]]
    v2 = new_vertices[new_faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(norms, 1e-10)

    # --- Build texel mapping ---
    print(
        "Building texel mapping "
        f"({tex_size}x{tex_size}, internal {tex_res}x{tex_res}, oversample x{oversample})"
    )
    print(
        "  View scoring: per-texel blend (topK=%d), min_cos=%.2f, angle_exp=%.2f, dist_pow=%.2f, sharpen=%.2f, mode=%s, quality_boost=%s, color_hardening=%s"
        % (
            blend_topk,
            min_cos,
            angle_exp,
            dist_pow,
            sharpen_amt,
            texture_view_assign_mode,
            texture_quality_boost,
            color_hardening,
        )
    )
    if texture_device == "cuda":
        print("  Projection backend: CUDA (GPU rasterization + scoring)")
    else:
        print("  Projection backend: CPU")
    _emit_progress(progress_cb, 62.0, "Building texel mapping")
    uv_scaled = uvs * tex_res

    # Try GPU UV-space rasterization (Phase 2)
    _used_gpu_uv = False
    if texture_device == "cuda":
        try:
            from scripts.texture.gpu_raster import gpu_rasterize_uv_space
            face_id_buf, bary_buf = gpu_rasterize_uv_space(uvs, new_faces, tex_res)
            _used_gpu_uv = True
            _emit_progress(progress_cb, 72.0, "Building texel mapping (GPU complete)")
        except Exception:
            _used_gpu_uv = False

    if not _used_gpu_uv:
        face_id_buf = np.full((tex_res, tex_res), -1, dtype=np.int32)

        # Pre-compute all triangle UV coords for fillConvexPoly (Optimization E)
        tri_coords = np.stack([
            uv_scaled[new_faces[:, 0]],
            uv_scaled[new_faces[:, 1]],
            uv_scaled[new_faces[:, 2]],
        ], axis=1).astype(np.int32).reshape(-1, 3, 1, 2)

        emit_every_face = max(1, len(new_faces) // 40)
        for fi in range(len(new_faces)):
            cv2.fillConvexPoly(face_id_buf, tri_coords[fi], int(fi))
            if fi % emit_every_face == 0 or fi == len(new_faces) - 1:
                ratio = (fi + 1) / max(len(new_faces), 1)
                _emit_progress(
                    progress_cb,
                    62.0 + ratio * 10.0,
                    f"Building texel mapping ({fi + 1}/{len(new_faces)} faces)",
                )

    valid_texels = face_id_buf >= 0
    ys, xs = np.where(valid_texels)
    fids = face_id_buf[ys, xs]

    if _used_gpu_uv:
        barys = bary_buf[ys, xs]
    else:
        # Barycentric coords (CPU path)
        px, py = xs + 0.5, ys + 0.5
        fi0, fi1, fi2 = new_faces[fids, 0], new_faces[fids, 1], new_faces[fids, 2]
        uv0, uv1, uv2 = uv_scaled[fi0], uv_scaled[fi1], uv_scaled[fi2]
        denom = (uv1[:, 1] - uv2[:, 1]) * (uv0[:, 0] - uv2[:, 0]) + (uv2[:, 0] - uv1[:, 0]) * (uv0[:, 1] - uv2[:, 1])
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        w0 = ((uv1[:, 1] - uv2[:, 1]) * (px - uv2[:, 0]) + (uv2[:, 0] - uv1[:, 0]) * (py - uv2[:, 1])) / denom
        w1 = ((uv2[:, 1] - uv0[:, 1]) * (px - uv2[:, 0]) + (uv0[:, 0] - uv2[:, 0]) * (py - uv2[:, 1])) / denom
        w2 = 1.0 - w0 - w1
        barys = np.column_stack([w0, w1, w2]).astype(np.float32)

    # 3D positions of texels
    tv0 = new_vertices[new_faces[fids, 0]]
    tv1 = new_vertices[new_faces[fids, 1]]
    tv2 = new_vertices[new_faces[fids, 2]]
    pos3d = barys[:, 0:1] * tv0 + barys[:, 1:2] * tv1 + barys[:, 2:3] * tv2
    normals = face_normals[fids]

    n_valid = len(pos3d)
    print(f"  {n_valid} valid texels")
    if n_valid == 0:
        raise RuntimeError("No valid texels were generated from UV atlas.")

    diagnostics: dict[str, object] = {
        "image": {"width": int(img_w), "height": int(img_h), "poses": int(len(poses))},
        "texture": {
            "size": int(tex_size),
            "internal_size": int(tex_res),
            "oversample": int(oversample),
            "valid_texels": int(n_valid),
        },
        "settings": {
            "blend_topk": int(blend_topk),
            "hard_ratio": float(hard_ratio),
            "color_hardening": bool(color_hardening),
            "color_hardening_threshold": float(color_hardening_threshold),
            "min_cos": float(min_cos),
            "angle_exp": float(angle_exp),
            "dist_pow": float(dist_pow),
            "sharpen": float(sharpen_amt),
            "view_assign_mode": texture_view_assign_mode,
            "quality_boost": bool(texture_quality_boost),
            "device": texture_device,
        },
        "intrinsics": {
            "source": intrinsics.get("source", "cached"),
            "distortion_model": intrinsics.get("distortion_model"),
            "has_distortion": bool(
                dist_coeffs is not None
                and np.any(np.abs(np.asarray(dist_coeffs, dtype=np.float64).ravel()) > 1e-12)
            ),
        },
    }

    # --- Per-texel top-K blend setup ---
    best_scores = np.full((n_valid, blend_topk), -1.0, dtype=np.float32)
    best_views = np.full((n_valid, blend_topk), -1, dtype=np.int32)

    # --- Pass 1: score all views per texel ---
    _emit_progress(progress_cb, 72.0, "Scoring camera views per texel")
    emit_every_view = max(1, len(poses) // 40)

    # Depth buffer cache for reuse across passes (Optimization C)
    depth_buf_bytes = img_w * img_h * 4  # float32
    avail_mb = _get_available_memory_mb()
    max_depth_cache = min(
        len(poses),
        max(0, int(avail_mb * 0.3 * 1024 * 1024 / max(depth_buf_bytes, 1))),
    )
    depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    for vidx in range(len(poses)):
        src_idx = int(pose_frame_indices[vidx])
        c2w = poses[vidx]
        try:
            mask_bool = frame_cache.load_mask(mask_dir, src_idx)
        except FileNotFoundError:
            continue

        depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h, device=texture_device, dist_coeffs=dist_coeffs)
        _cache_depth_buffer(depth_cache, max_depth_cache, vidx, depth_buffer)
        valid, score, _px_proj, _py_proj = _evaluate_view_samples(
            pos3d=pos3d,
            normals=normals,
            c2w=c2w,
            K=K,
            img_w=img_w,
            img_h=img_h,
            mask_bool=mask_bool,
            depth_buffer=depth_buffer,
            min_cos=min_cos,
            angle_exp=angle_exp,
            dist_pow=dist_pow,
            device=texture_device,
            dist_coeffs=dist_coeffs,
        )
        _update_topk_scores(best_scores, best_views, valid, score, vidx)

        if vidx % 5 == 0 or vidx == len(poses) - 1:
            covered = int(np.count_nonzero(best_views[:, 0] >= 0))
            print(
                f"  Candidate scoring {vidx + 1}/{len(poses)}: "
                f"texels_valid={int(valid.sum())}, covered={covered}/{n_valid}"
            )
        if vidx % emit_every_view == 0 or vidx == len(poses) - 1:
            ratio = (vidx + 1) / max(len(poses), 1)
            _emit_progress(
                progress_cb,
                72.0 + ratio * 12.0,
                f"Scoring views ({vidx + 1}/{len(poses)} views)",
            )

    covered = int(np.count_nonzero(best_views[:, 0] >= 0))
    print(f"  Per-texel scoring done: {covered}/{n_valid} texels have ≥1 valid view")

    ratio_valid = (best_scores[:, 0] > 0) & (best_scores[:, 1] > 0) if blend_topk > 1 else np.zeros(n_valid, dtype=bool)
    score_ratio = np.zeros(n_valid, dtype=np.float32)
    if np.any(ratio_valid):
        score_ratio[ratio_valid] = (
            best_scores[ratio_valid, 0]
            / np.maximum(best_scores[ratio_valid, 1], 1e-12)
        )
    diagnostics["view_scoring"] = {
        "covered_texels": covered,
        "coverage_ratio": float(covered / max(n_valid, 1)),
        "score_ratio_top1_top2": _array_summary(score_ratio[ratio_valid]),
    }

    n_hard_score = _apply_view_hardening(best_scores, best_views, hard_ratio)
    if n_hard_score > 0:
        print(f"  View hardening: {n_hard_score}/{n_valid} texels use single dominant view (ratio>{hard_ratio:.1f})")

    conflict_texels = _compute_conflict_texels(
        pos3d=pos3d,
        poses=poses,
        best_scores=best_scores,
        best_views=best_views,
    )
    n_conflict = int(conflict_texels.sum())
    n_region_faces = 0
    n_regions = 0
    if n_conflict > 0:
        face_locked_view, _face_lock_support = _compute_face_locked_views(
            fids=fids,
            n_faces=len(new_faces),
            n_views=len(poses),
            best_scores=best_scores,
            best_views=best_views,
            conflict_texels=conflict_texels,
            face_normals=face_normals,
            faces=new_faces,
        )
        region_id_per_face = np.full(len(new_faces), -1, dtype=np.int32)
        if texture_view_assign_mode == "region_gc":
            face_locked_view, _face_lock_support, region_id_per_face = _compute_region_gc_locked_views(
                fids=fids,
                n_faces=len(new_faces),
                n_views=len(poses),
                best_scores=best_scores,
                best_views=best_views,
                conflict_texels=conflict_texels,
                face_normals=face_normals,
                faces=new_faces,
                base_locked_view=face_locked_view,
            )
            n_region_faces = int(np.count_nonzero(region_id_per_face >= 0))
            n_regions = int(region_id_per_face.max()) + 1 if np.any(region_id_per_face >= 0) else 0
            if n_region_faces > 0:
                print(
                    "  Region labeling: %d faces optimized across %d curved regions"
                    % (n_region_faces, n_regions)
                )
        locked_faces = face_locked_view >= 0
        n_locked_faces = int(locked_faces.sum())
        locked_view_per_texel = face_locked_view[fids]
        region_id_per_texel = region_id_per_face[fids]
        n_locked_texels = int(np.count_nonzero(locked_view_per_texel >= 0))
        if n_locked_faces > 0:
            print(
                "  Conflict locking: %d/%d texels, %d faces locked to a single view"
                % (n_locked_texels, n_valid, n_locked_faces)
            )
        else:
            print(
                "  Conflict detection: %d texels flagged, but no faces met lock coverage"
                % n_conflict
            )
    else:
        face_locked_view = np.full(len(new_faces), -1, dtype=np.int32)
        locked_view_per_texel = np.full(n_valid, -1, dtype=np.int32)
        region_id_per_face = np.full(len(new_faces), -1, dtype=np.int32)
        region_id_per_texel = np.full(n_valid, -1, dtype=np.int32)
        print("  Conflict locking: no ambiguous texels detected")

    locked_faces = face_locked_view >= 0
    n_locked_faces = int(np.count_nonzero(locked_faces))
    n_locked_texels = int(np.count_nonzero(locked_view_per_texel >= 0))
    diagnostics["view_selection"] = {
        "score_hardened_texels": int(n_hard_score),
        "conflict_texels": int(n_conflict),
        "locked_faces": int(n_locked_faces),
        "locked_texels": int(n_locked_texels),
        "region_faces": int(n_region_faces),
        "regions": int(n_regions),
    }

    n_color_hard = 0
    if blend_topk > 1:
        _emit_progress(progress_cb, 83.0, "Checking top-K color consistency")
        candidate_colors, candidate_valid = _collect_topk_candidate_colors(
            best_scores=best_scores,
            best_views=best_views,
            pos3d=pos3d,
            normals=normals,
            poses=poses,
            pose_frame_indices=pose_frame_indices,
            frames_dir=frames_dir,
            mask_dir=mask_dir,
            frame_cache=frame_cache,
            depth_cache=depth_cache,
            max_depth_cache=max_depth_cache,
            new_vertices=new_vertices,
            new_faces=new_faces,
            K=K,
            img_w=img_w,
            img_h=img_h,
            min_cos=min_cos,
            angle_exp=angle_exp,
            dist_pow=dist_pow,
            texture_device=texture_device,
            dist_coeffs=dist_coeffs,
        )
        color_instability, candidate_counts = _compute_color_instability(
            candidate_colors,
            candidate_valid,
            np.maximum(best_scores, 0.0),
        )
        unstable_mask = candidate_counts >= 2
        diagnostics["color_consistency"] = {
            "candidate_texels": int(np.count_nonzero(unstable_mask)),
            "instability": _array_summary(color_instability[unstable_mask]),
            "threshold": float(color_hardening_threshold),
            "enabled": bool(color_hardening),
        }
        if color_hardening:
            n_color_hard = _apply_color_instability_hardening(
                best_scores,
                best_views,
                color_instability,
                candidate_counts,
                color_hardening_threshold,
                eligible_mask=locked_view_per_texel < 0,
            )
            if n_color_hard > 0:
                print(
                    "  Color hardening: %d/%d texels use top-1 view "
                    "(top-K color spread > %.3f)"
                    % (n_color_hard, n_valid, color_hardening_threshold)
                )
        diagnostics["color_consistency"]["hardened_texels"] = int(n_color_hard)
        del candidate_colors, candidate_valid
    else:
        diagnostics["color_consistency"] = {
            "candidate_texels": 0,
            "instability": _array_summary(np.array([], dtype=np.float32)),
            "threshold": float(color_hardening_threshold),
            "enabled": bool(color_hardening),
            "hardened_texels": 0,
        }
    diagnostics["view_selection"]["color_hardened_texels"] = int(n_color_hard)

    texture = np.zeros((tex_res, tex_res, 3), dtype=np.float64)
    weight_sum = np.zeros(n_valid, dtype=np.float64)
    has_color = np.zeros(n_valid, dtype=bool)

    # --- Pass 2: weighted multi-view blending ---
    _emit_progress(progress_cb, 84.0, "Blending textures from top-K views")

    # Group texels by view across all K slots
    view_texels: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    locked_mask = locked_view_per_texel >= 0
    if np.any(locked_mask):
        locked_indices = np.where(locked_mask)[0]
        locked_views = locked_view_per_texel[locked_mask]
        order = np.argsort(locked_views, kind="stable")
        locked_indices = locked_indices[order]
        locked_views = locked_views[order]
        start_idx = np.flatnonzero(np.r_[True, locked_views[1:] != locked_views[:-1]])
        end_idx = np.r_[start_idx[1:], len(locked_views)]
        for start, end in zip(start_idx.tolist(), end_idx.tolist(), strict=False):
            vidx_val = int(locked_views[start])
            tidx = locked_indices[start:end]
            view_texels[vidx_val].append(
                (tidx, np.ones(tidx.size, dtype=np.float64))
            )

    free_texels = ~locked_mask
    for k in range(blend_topk):
        col_views = best_views[:, k]
        col_scores = best_scores[:, k]
        for vidx_val in np.unique(col_views):
            if vidx_val < 0:
                continue
            mask_k = free_texels & (col_views == vidx_val)
            tidx = np.where(mask_k)[0]
            if tidx.size == 0:
                continue
            view_texels[int(vidx_val)].append(
                (tidx, col_scores[tidx].astype(np.float64))
            )

    sorted_views = sorted(view_texels.keys())
    for order_idx, vidx in enumerate(sorted_views):
        all_tidx = np.concatenate([t for t, _ in view_texels[vidx]])
        all_weights = np.concatenate([w for _, w in view_texels[vidx]])

        src_idx = int(pose_frame_indices[vidx])
        c2w = poses[vidx]
        try:
            frame = frame_cache.load_frame(frames_dir, src_idx)
            mask_bool = frame_cache.load_mask(mask_dir, src_idx)
        except FileNotFoundError:
            continue

        if vidx in depth_cache:
            depth_buffer = depth_cache[vidx]
        else:
            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h, device=texture_device, dist_coeffs=dist_coeffs)
        valid, _score, px_proj, py_proj = _evaluate_view_samples(
            pos3d=pos3d,
            normals=normals,
            c2w=c2w,
            K=K,
            img_w=img_w,
            img_h=img_h,
            mask_bool=mask_bool,
            depth_buffer=depth_buffer,
            min_cos=min_cos,
            angle_exp=angle_exp,
            dist_pow=dist_pow,
            device=texture_device,
            dist_coeffs=dist_coeffs,
        )

        ok = valid[all_tidx]
        fill_tidx = all_tidx[ok]
        fill_weights = all_weights[ok]

        if fill_tidx.size > 0:
            colors = _bilinear_sample(
                frame,
                px_proj[fill_tidx],
                py_proj[fill_tidx],
            )
            texture[ys[fill_tidx], xs[fill_tidx]] += fill_weights[:, None] * colors
            weight_sum[fill_tidx] += fill_weights

        if order_idx % 5 == 0 or order_idx == len(sorted_views) - 1:
            print(
                f"  Blend view {vidx + 1}/{len(poses)}: "
                f"texels={int(fill_tidx.size)}"
            )
        ratio = (order_idx + 1) / max(len(sorted_views), 1)
        _emit_progress(
            progress_cb,
            84.0 + ratio * 6.0,
            f"Blending views ({order_idx + 1}/{len(sorted_views)})",
        )

    # Normalize blended colors
    colored = np.where(weight_sum > 0)[0]
    if colored.size > 0:
        texture[ys[colored], xs[colored]] /= weight_sum[colored, None]
    has_color[colored] = True
    diagnostics["coverage"] = {
        "after_blend_texels": int(has_color.sum()),
        "after_blend_ratio": float(has_color.sum() / max(n_valid, 1)),
    }

    # --- Pass 3: simple fallback for uncovered texels ---
    missing = np.where(~has_color)[0]
    fallback_filled = 0
    if missing.size > 0 and missing.size > n_valid * 0.001:
        _emit_progress(progress_cb, 90.0, "Fallback for uncovered texels")
        fallback_min_cos = min(min_cos * 0.25, 0.0)
        fb_best_score = np.full(len(missing), -1.0, dtype=np.float64)
        fb_best_view = np.full(len(missing), -1, dtype=np.int32)

        for vidx in range(len(poses)):
            src_idx = int(pose_frame_indices[vidx])
            c2w = poses[vidx]
            try:
                mask_bool = frame_cache.load_mask(mask_dir, src_idx)
            except FileNotFoundError:
                continue

            if vidx in depth_cache:
                depth_buffer = depth_cache[vidx]
            else:
                depth_buffer = _rasterize_view_depth(
                    new_vertices, new_faces, c2w, K, img_w, img_h, device=texture_device,
                    dist_coeffs=dist_coeffs,
                )
            valid_fb, score_fb, _, _ = _evaluate_view_samples(
                pos3d=pos3d[missing],
                normals=normals[missing],
                c2w=c2w,
                K=K,
                img_w=img_w,
                img_h=img_h,
                mask_bool=mask_bool,
                depth_buffer=depth_buffer,
                min_cos=fallback_min_cos,
                angle_exp=angle_exp,
                dist_pow=dist_pow,
                device=texture_device,
                dist_coeffs=dist_coeffs,
            )
            better = valid_fb & (score_fb > fb_best_score)
            better_idx = np.where(better)[0]
            if better_idx.size > 0:
                fb_best_score[better_idx] = score_fb[better_idx]
                fb_best_view[better_idx] = vidx

        fb_assigned = np.where(fb_best_view >= 0)[0]
        fb_filled = 0
        if fb_assigned.size > 0:
            for vidx_val in np.unique(fb_best_view[fb_assigned]):
                vidx_val = int(vidx_val)
                fb_local = np.where(fb_best_view == vidx_val)[0]
                fb_global = missing[fb_local]

                src_idx = int(pose_frame_indices[vidx_val])
                c2w = poses[vidx_val]
                try:
                    frame = frame_cache.load_frame(frames_dir, src_idx)
                    mask_bool = frame_cache.load_mask(mask_dir, src_idx)
                except FileNotFoundError:
                    continue

                if vidx_val in depth_cache:
                    depth_buffer = depth_cache[vidx_val]
                else:
                    depth_buffer = _rasterize_view_depth(
                        new_vertices, new_faces, c2w, K, img_w, img_h, device=texture_device,
                        dist_coeffs=dist_coeffs,
                    )
                valid_p, _, px_p, py_p = _evaluate_view_samples(
                    pos3d=pos3d[fb_global],
                    normals=normals[fb_global],
                    c2w=c2w,
                    K=K,
                    img_w=img_w,
                    img_h=img_h,
                    mask_bool=mask_bool,
                    depth_buffer=depth_buffer,
                    min_cos=fallback_min_cos,
                    angle_exp=angle_exp,
                    dist_pow=dist_pow,
                    device=texture_device,
                    dist_coeffs=dist_coeffs,
                )
                if np.any(valid_p):
                    fill_global = fb_global[valid_p]
                    colors = _bilinear_sample(
                        frame, px_p[valid_p], py_p[valid_p]
                    ).astype(np.float32)
                    texture[ys[fill_global], xs[fill_global]] = colors
                    has_color[fill_global] = True
                    fb_filled += int(fill_global.size)

        fallback_filled = fb_filled
        remaining = int((~has_color).sum())
        print(
            f"  Fallback pass: min_cos={fallback_min_cos:.3f}, "
            f"filled={fb_filled}/{len(missing)}, remaining={remaining}"
        )
        _emit_progress(progress_cb, 92.0, "Fallback complete")
    elif missing.size > 0:
        print(f"  Skipping fallback: only {missing.size} uncovered texels (<0.1%)")
    diagnostics["coverage"].update({
        "missing_before_fallback": int(missing.size),
        "fallback_filled_texels": int(fallback_filled),
        "after_fallback_texels": int(has_color.sum()),
        "after_fallback_ratio": float(has_color.sum() / max(n_valid, 1)),
    })

    # --- Pass 3.5: Cap region infill ---
    if vert_colors is not None:
        cap_missing_idx, cap_face_mask = _identify_cap_texels(
            vert_colors, vmapping, new_faces, fids, has_color
        )
        if cap_missing_idx is not None:
            cap_mask = np.zeros((tex_res, tex_res), dtype=bool)
            cap_mask[ys[cap_missing_idx], xs[cap_missing_idx]] = True
            valid_mask = np.zeros((tex_res, tex_res), dtype=bool)
            valid_mask[ys[has_color], xs[has_color]] = True

            # Seed cap border from mesh-adjacent body texels (bridge UV gap)
            seeded = _seed_cap_border(
                texture, valid_mask, vmapping, new_faces,
                cap_face_mask, uvs, tex_res,
            )

            filled = _fill_cap_region(texture, cap_mask, valid_mask)

            for i in cap_missing_idx:
                if valid_mask[ys[i], xs[i]]:
                    has_color[i] = True
            remaining = int((~has_color).sum())
            print(
                f"  Cap infill: seeded={seeded}, filled={filled}/{len(cap_missing_idx)}, "
                f"remaining={remaining}"
            )
        else:
            print("  Cap region: all cap texels already covered (or no cap faces)")
    diagnostics["coverage"].update({
        "after_cap_texels": int(has_color.sum()),
        "after_cap_ratio": float(has_color.sum() / max(n_valid, 1)),
    })

    texture = texture.astype(np.float32)

    if texture_view_assign_mode == "region_gc" and np.any(region_id_per_texel >= 0):
        valid_color_mask = np.zeros((tex_res, tex_res), dtype=bool)
        valid_color_mask[ys, xs] = has_color
        label_buffer = np.full((tex_res, tex_res), -1, dtype=np.int32)
        region_mask = (region_id_per_texel >= 0) & (locked_view_per_texel >= 0)
        label_buffer[ys[region_mask], xs[region_mask]] = locked_view_per_texel[region_mask]
        _emit_progress(progress_cb, 92.0, "Leveling region seams")
        texture, seam_texels = _apply_narrow_seam_leveling(
            texture=texture,
            valid_mask=valid_color_mask,
            label_buffer=label_buffer,
            blend=_TEXTURE_SEAM_LEVEL_BLEND if not texture_quality_boost else 0.30,
            dilate_iters=_TEXTURE_SEAM_LEVEL_DILATE if not texture_quality_boost else _TEXTURE_SEAM_HQ_DILATE,
            sigma=_TEXTURE_SEAM_LEVEL_SIGMA if not texture_quality_boost else _TEXTURE_SEAM_HQ_SIGMA,
            quality_boost=False,
        )
        if seam_texels > 0:
            if texture_quality_boost:
                texture, _refined_seam_texels, companion_cov = _apply_quality_boost_detail_refinement(
                    texture=texture,
                    valid_mask=valid_color_mask,
                    label_buffer=label_buffer,
                    ys=ys,
                    xs=xs,
                    best_scores=best_scores,
                    best_views=best_views,
                    locked_view_per_texel=locked_view_per_texel,
                    pos3d=pos3d,
                    normals=normals,
                    poses=poses,
                    pose_frame_indices=pose_frame_indices,
                    frames_dir=frames_dir,
                    mask_dir=mask_dir,
                    frame_cache=frame_cache,
                    depth_cache=depth_cache,
                    new_vertices=new_vertices,
                    new_faces=new_faces,
                    K=K,
                    img_w=img_w,
                    img_h=img_h,
                    min_cos=min_cos,
                    angle_exp=angle_exp,
                    dist_pow=dist_pow,
                    device=texture_device,
                    dist_coeffs=dist_coeffs,
                )
                print(
                    "  Region seam refinement: softened %d boundary texels, refined detail samples=%d"
                    % (seam_texels, companion_cov)
                )
            else:
                print(f"  Narrow seam leveling: softened {seam_texels} boundary texels")

    # Release caches to free memory before seam padding
    depth_cache.clear()
    del frame_cache
    if texture_device == "cuda":
        try:
            from scripts.texture.gpu_raster import clear_cache
            clear_cache()
        except Exception:
            pass

    cov = int(has_color.sum())
    print(f"Texture coverage before seam padding: {cov}/{n_valid} ({100*cov/max(n_valid,1):.1f}%)")
    diagnostics["coverage"].update({
        "before_seam_padding_texels": int(cov),
        "before_seam_padding_ratio": float(cov / max(n_valid, 1)),
    })

    # --- Seam padding ---
    print("Padding seams...")
    _emit_progress(progress_cb, 92.0, "Padding UV seams")
    valid_pad = np.zeros((tex_res, tex_res), dtype=bool)
    valid_pad[ys, xs] = has_color
    result_tex = texture.copy()
    seam_pad_iters = int(_TEXTURE_SEAM_PAD_ITERS)

    gpu_seam_pad_done = False
    if texture_device == "cuda":
        try:
            result_tex, valid_pad = _seam_pad_gpu(result_tex, valid_pad, seam_pad_iters)
            gpu_seam_pad_done = True
            _emit_progress(progress_cb, 97.0, f"Padding seams (GPU, {seam_pad_iters} iters)")
        except Exception as exc:
            print(f"  GPU seam padding unavailable ({exc}); falling back to CPU")

    if not gpu_seam_pad_done:
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        for iter_idx in range(seam_pad_iters):
            empty = ~valid_pad
            if not np.any(empty):
                break
            nc = cv2.filter2D(
                valid_pad.astype(np.float32), -1, kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            fill = empty & (nc > 0)
            if not np.any(fill):
                break
            inv_nc = 1.0 / nc[fill]
            for c in range(3):
                ns = cv2.filter2D(
                    result_tex[:, :, c], -1, kernel,
                    borderType=cv2.BORDER_CONSTANT,
                )
                result_tex[:, :, c][fill] = ns[fill] * inv_nc
            valid_pad[fill] = True
            _emit_progress(
                progress_cb,
                92.0 + ((iter_idx + 1) / seam_pad_iters) * 5.0,
                f"Padding seams ({iter_idx + 1}/{seam_pad_iters})",
            )

    # Downsample from supersample grid and optionally sharpen
    final_tex = result_tex
    if oversample > 1:
        final_tex = cv2.resize(
            result_tex, (tex_size, tex_size), interpolation=cv2.INTER_AREA
        )
    if sharpen_amt > 0:
        blur = cv2.GaussianBlur(final_tex, (0, 0), sigmaX=1.0)
        final_tex = np.clip(final_tex + sharpen_amt * (final_tex - blur), 0.0, 1.0)

    # --- Export OBJ + MTL + PNG ---
    basename = "textured_mesh"
    texture_phase_dir(output_path).mkdir(parents=True, exist_ok=True)
    obj_path = textured_mesh_obj_path(output_path)
    mtl_path = textured_mesh_mtl_path(output_path)
    tex_path = texture_png_path(output_path)

    # Texture PNG (flip V for OBJ convention)
    tex_u8 = np.clip(final_tex * 255, 0, 255).astype(np.uint8)
    tex_bgr = cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR)[::-1]
    cv2.imwrite(str(tex_path), tex_bgr)
    print(f"Saved: {tex_path}")
    _emit_progress(progress_cb, 98.0, "Exporting textured mesh")

    # MTL
    with open(mtl_path, "w") as f_out:
        f_out.write(
            f"newmtl material_0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n"
            f"Ks 0.0 0.0 0.0\nd 1.0\nillum 1\nmap_Kd texture.png\n"
        )
    print(f"Saved: {mtl_path}")

    # OBJ (buffered write — Optimization F)
    with open(obj_path, "w") as f_out:
        lines = [f"mtllib {basename}.mtl\nusemtl material_0\n\n"]
        for vert in new_vertices:
            lines.append(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in new_faces:
            a, b, c = face + 1
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        f_out.write("".join(lines))
    print(f"Saved: {obj_path}")
    diagnostics_path = texture_phase_dir(output_path) / "diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f_out:
        json.dump(diagnostics, f_out, indent=2)
    print(f"Saved: {diagnostics_path}")
    _emit_progress(progress_cb, 100.0, "Texture stage complete")

    return obj_path
