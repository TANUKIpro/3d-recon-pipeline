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
    TEXTURE_DIST_POW,
    TEXTURE_MIN_COS,
    TEXTURE_OVERSAMPLE,
    TEXTURE_SHARPEN,
)
from scripts.mesh_orientation import orient_mesh_outward
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
from scripts.texture.image_utils import _bilinear_sample
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

    # Create frame/mask cache with memory-aware capacity (Optimization A)
    frame_cache = _FrameCache(img_w, img_h)

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

    # --- Estimate intrinsics ---
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

    intrinsics = _estimate_intrinsics(
        pc_points, pc_colors, poses, pose_frame_indices, frames_dir, mask_dir, img_w, img_h,
        progress_cb=_intrinsics_progress,
        cache=frame_cache,
    )
    K = np.array(intrinsics["K"], dtype=np.float64)
    print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    _emit_progress(progress_cb, 52.0, "Camera intrinsics estimated")

    # Save intrinsics
    with open(output_path / "intrinsics.json", "w") as f_out:
        json.dump(intrinsics, f_out, indent=2)

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
    new_faces, flipped_winding, ratio_before, ratio_after = orient_mesh_outward(
        new_vertices, new_faces
    )
    print(f"  {len(new_vertices)} verts, {len(new_faces)} faces")
    if flipped_winding:
        print(
            "  Orientation fix: flipped UV faces "
            f"(outward_ratio {ratio_before:.3f} -> {ratio_after:.3f})"
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
        "  View scoring: per-texel blend (topK=%d), min_cos=%.2f, angle_exp=%.2f, dist_pow=%.2f, sharpen=%.2f, mode=%s, quality_boost=%s"
        % (blend_topk, min_cos, angle_exp, dist_pow, sharpen_amt, texture_view_assign_mode, texture_quality_boost)
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

        depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h, device=texture_device)
        if max_depth_cache > 0:
            depth_cache[vidx] = depth_buffer
            while len(depth_cache) > max_depth_cache:
                depth_cache.popitem(last=False)
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

    n_hard = _apply_view_hardening(best_scores, best_views, hard_ratio)
    if n_hard > 0:
        print(f"  View hardening: {n_hard}/{n_valid} texels use single dominant view (ratio>{hard_ratio:.1f})")

    conflict_texels = _compute_conflict_texels(
        pos3d=pos3d,
        poses=poses,
        best_scores=best_scores,
        best_views=best_views,
    )
    n_conflict = int(conflict_texels.sum())
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
            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h, device=texture_device)
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

    # --- Pass 3: simple fallback for uncovered texels ---
    missing = np.where(~has_color)[0]
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
                )
                if np.any(valid_p):
                    fill_global = fb_global[valid_p]
                    colors = _bilinear_sample(
                        frame, px_p[valid_p], py_p[valid_p]
                    ).astype(np.float32)
                    texture[ys[fill_global], xs[fill_global]] = colors
                    has_color[fill_global] = True
                    fb_filled += int(fill_global.size)

        remaining = int((~has_color).sum())
        print(
            f"  Fallback pass: min_cos={fallback_min_cos:.3f}, "
            f"filled={fb_filled}/{len(missing)}, remaining={remaining}"
        )
        _emit_progress(progress_cb, 92.0, "Fallback complete")
    elif missing.size > 0:
        print(f"  Skipping fallback: only {missing.size} uncovered texels (<0.1%)")

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

    # --- Seam padding ---
    print("Padding seams...")
    _emit_progress(progress_cb, 92.0, "Padding UV seams")
    valid_pad = np.zeros((tex_res, tex_res), dtype=bool)
    valid_pad[ys, xs] = has_color
    result_tex = texture.copy()
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    for iter_idx in range(8):
        empty = ~valid_pad
        if not np.any(empty):
            break
        for c in range(3):
            ns = cv2.filter2D(result_tex[:, :, c], -1, kernel, borderType=cv2.BORDER_CONSTANT)
            nc = cv2.filter2D(valid_pad.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
            fill = empty & (nc > 0)
            if np.any(fill):
                result_tex[:, :, c][fill] = ns[fill] / nc[fill]
                valid_pad[fill] = True
        _emit_progress(
            progress_cb,
            92.0 + ((iter_idx + 1) / 8.0) * 5.0,
            f"Padding seams ({iter_idx + 1}/8)",
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
    obj_path = output_path / f"{basename}.obj"
    mtl_path = output_path / f"{basename}.mtl"
    tex_path = output_path / "texture.png"

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
    _emit_progress(progress_cb, 100.0, "Texture stage complete")

    return obj_path
