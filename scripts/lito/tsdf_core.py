"""Pure TSDF fusion core, shared by gs2mesh and lito Stage 4 backends.

This module exposes a side-effect-free TSDF fusion function that takes
already-loaded multi-view RGB+depth payloads and returns an Open3D mesh.
It performs no file I/O, reads no environment variables, and owns no GPU
lifetime — the caller is responsible for any CUDA cache release.

Refactored from scripts/gpu_tsdf.py (Phase 1.5 of .claude/plans/lito_integration.md).
The gs2mesh adapter (gpu_tsdf.gpu_tsdf_reconstruct) wraps this core with
gs2mesh-workspace I/O and visual-hull bookkeeping; the lito adapter
(scripts/lito/gaussian_to_mesh.py, Phase 3) renders Gaussians to depth
before calling here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class TsdfFusionParams:
    """All knobs for fuse_tsdf, kept separate from per-view payloads."""

    voxel_size: float
    sdf_trunc: float
    depth_min: float
    depth_max: float
    device: str = "CUDA:0"
    block_count: int = 100_000
    cleaning_threshold: int = 100_000
    tsdf_scale: float = 1.0
    smooth_iters: int = 0
    smooth_lambda: float = 0.5
    smooth_mu: float = -0.53
    smooth_min_iou: float = 0.985
    target_faces: int = 0
    decimate_min_iou: float = 0.985
    iou_views: int = 8


@dataclass(frozen=True)
class TsdfView:
    """One view's payload for fusion. No file I/O implied.

    All inputs must already be in world-consistent coordinates: T_cw maps
    world → camera in the same scale as voxel_size and sdf_trunc.
    """

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    T_cw: np.ndarray
    confidence: Optional[np.ndarray] = None
    confidence_min: float = 0.0


def _apply_confidence_mask(view: TsdfView) -> np.ndarray:
    """Return a depth array with low-confidence pixels zeroed out.

    Hard-mask interpretation of `confidence`: pixels with
    `confidence < confidence_min` are set to 0 (i.e. dropped from
    integration). For gs2mesh callers (`confidence=None`), the original
    depth is returned unchanged.
    """
    if view.confidence is None or view.confidence_min <= 0.0:
        return view.depth
    mask = view.confidence >= view.confidence_min
    return np.where(mask, view.depth, np.float32(0.0)).astype(np.float32)


def _has_any_valid_depth(views: list[TsdfView], params: TsdfFusionParams) -> bool:
    """Return True if at least one view has a pixel inside [depth_min, depth_max]."""
    for v in views:
        if v.depth is None:
            continue
        d = _apply_confidence_mask(v)
        valid = (d > params.depth_min) & (d < params.depth_max)
        if int(np.count_nonzero(valid)) > 0:
            return True
    return False


def fuse_tsdf(
    views: list[TsdfView],
    params: TsdfFusionParams,
    *,
    progress_cb: Optional[Callable[[float, str], None]] = None,
):
    """GPU TSDF fusion of pre-loaded multi-view RGB+depth.

    Returns:
        Open3D legacy TriangleMesh, or None if every view was skipped
        (no valid depth pixels in [depth_min, depth_max]).

    Notes:
        * Pure function: no file I/O, no env vars, no global state.
        * CUDA cache release is the CALLER's responsibility.
        * confidence < confidence_min zeros the corresponding depth pixels
          before integration (hard-mask). gs2mesh callers omit confidence
          and get binary-mask behaviour identical to the pre-refactor pipeline.
        * Phase 1.5 Step 2: VBG + integrate loop + extract + cleaning are
          implemented here. Smoothing, decimation, and silhouette IoU
          rollback remain caller-side until Step 3.
    """
    if not views:
        return None
    if not _has_any_valid_depth(views, params):
        return None

    import open3d as o3d
    import open3d.core as o3c

    device = o3c.Device(params.device)
    trunc_multiplier = params.sdf_trunc / params.voxel_size

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=float(params.voxel_size),
        block_resolution=16,
        block_count=int(params.block_count),
        device=device,
    )

    integrated_count = 0
    skipped_count = 0
    n_views = len(views)
    for step_i, view in enumerate(views):
        if progress_cb:
            progress_cb(
                step_i / max(n_views, 1),
                f"GPU TSDF fusion ({step_i + 1}/{n_views})",
            )

        depth = _apply_confidence_mask(view)
        valid = (depth > params.depth_min) & (depth < params.depth_max)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            skipped_count += 1
            continue

        intrinsic_t = o3c.Tensor(view.K.astype(np.float64))
        extrinsic_t = o3c.Tensor(view.T_cw.astype(np.float64))

        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth, device=device))
        color_f = view.rgb.astype(np.float32) / 255.0
        color_t = o3d.t.geometry.Image(o3c.Tensor(color_f, device=device))

        try:
            frustum_coords = vbg.compute_unique_block_coordinates(
                depth_t,
                intrinsic_t,
                extrinsic_t,
                depth_scale=params.tsdf_scale,
                depth_max=params.depth_max,
            )
            vbg.integrate(
                frustum_coords,
                depth_t,
                color_t,
                intrinsic_t,
                intrinsic_t,
                extrinsic_t,
                depth_scale=params.tsdf_scale,
                depth_max=params.depth_max,
                trunc_voxel_multiplier=trunc_multiplier,
            )
            integrated_count += 1
        except RuntimeError as e:
            if "No block is touched" in str(e):
                skipped_count += 1
                continue
            raise

    if integrated_count == 0:
        return None

    if progress_cb:
        progress_cb(0.92, "Extracting triangle mesh from TSDF volume")

    mesh = vbg.extract_triangle_mesh()
    mesh = mesh.to_legacy()
    del vbg

    mesh.scale(params.tsdf_scale, (0.0, 0.0, 0.0))
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    if progress_cb:
        progress_cb(0.96, "Cleaning mesh (removing small clusters)")

    threshold = params.cleaning_threshold / max(params.tsdf_scale, 1e-12)
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(triangle_clusters) > 0:
        keep = cluster_n_triangles[triangle_clusters] >= threshold
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()

    if progress_cb:
        progress_cb(0.97, "Post-processing mesh")

    n_tris_clean = len(mesh.triangles)
    if params.smooth_iters > 0 and n_tris_clean > 0:
        smoothed, did = _taubin_smooth(
            mesh,
            iters=params.smooth_iters,
            lambda_val=params.smooth_lambda,
            mu_val=params.smooth_mu,
        )
        if did:
            iou_ok, _mean_iou = _silhouette_iou_check(
                original=mesh,
                simplified=smoothed,
                views=views,
                n_views=params.iou_views,
                min_iou=params.smooth_min_iou,
            )
            if iou_ok:
                mesh = smoothed

    if params.target_faces > 0 and len(mesh.triangles) > params.target_faces:
        simplified, did = _decimate(mesh, target_faces=params.target_faces)
        if did:
            iou_ok, _mean_iou = _silhouette_iou_check(
                original=mesh,
                simplified=simplified,
                views=views,
                n_views=params.iou_views,
                min_iou=params.decimate_min_iou,
            )
            if iou_ok:
                mesh = simplified

    if progress_cb:
        progress_cb(1.0, "GPU TSDF fusion complete")
    return mesh


def _taubin_smooth(mesh, *, iters: int, lambda_val: float, mu_val: float):
    """Taubin λ/μ smoothing — preserves global shape, removes vertex noise.

    Returns (smoothed_mesh, did_smooth). Per-vertex colours are preserved
    by Open3D so downstream cap-marker detection keeps working.
    """
    if iters <= 0 or len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        return mesh, False
    smoothed = mesh.filter_smooth_taubin(
        number_of_iterations=int(iters),
        lambda_filter=float(lambda_val),
        mu=float(mu_val),
    )
    smoothed.remove_degenerate_triangles()
    smoothed.remove_duplicated_triangles()
    smoothed.remove_unreferenced_vertices()
    if len(smoothed.triangles) == 0 or len(smoothed.vertices) == 0:
        return mesh, False
    smoothed.compute_vertex_normals()
    return smoothed, True


def _decimate(mesh, *, target_faces: int):
    """Quadric edge-collapse decimation. Returns (mesh, did_simplify)."""
    n_tri = len(mesh.triangles)
    if target_faces <= 0 or n_tri <= target_faces:
        return mesh, False
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_faces)
    )
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_unreferenced_vertices()
    if (
        len(simplified.triangles) == 0
        or len(simplified.vertices) == 0
        or len(simplified.triangles) >= n_tri
    ):
        return mesh, False
    simplified.compute_vertex_normals()
    return simplified, True


def _silhouette_iou_check(
    *,
    original,
    simplified,
    views: list[TsdfView],
    n_views: int,
    min_iou: float,
    render_size: tuple[int, int] = (512, 512),
) -> tuple[bool, float]:
    """Validate that ``simplified`` preserves the silhouette of ``original``.

    Rasterises both meshes from ``n_views`` evenly-sampled views and returns
    (passed, mean_iou) where passed is True iff mean_iou >= min_iou. Any
    rasterisation failure is treated as a pass with IoU 1.0.
    """
    if n_views <= 0 or not views:
        return True, 1.0

    import open3d as o3d
    import open3d.core as o3c

    n_cam = len(views)
    step = max(1, n_cam // max(1, n_views))
    sample_idx = list(range(0, n_cam, step))[:n_views]
    if not sample_idx:
        return True, 1.0

    render_w, render_h = render_size

    def _build_scene(mesh) -> "o3d.t.geometry.RaycastingScene":
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        return scene

    try:
        scene_o = _build_scene(original)
        scene_s = _build_scene(simplified)
    except Exception as exc:  # pragma: no cover — Open3D edge cases
        print(f"TSDF: silhouette IoU check skipped (scene build failed: {exc})")
        return True, 1.0

    ious: list[float] = []
    for cam_idx in sample_idx:
        v = views[cam_idx]
        try:
            full_h, full_w = v.rgb.shape[:2]
            sx = render_w / max(1.0, float(full_w))
            sy = render_h / max(1.0, float(full_h))
            K = v.K.astype(np.float64)
            intr = np.array(
                [
                    [K[0, 0] * sx, 0.0, K[0, 2] * sx],
                    [0.0, K[1, 1] * sy, K[1, 2] * sy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            rays = scene_o.create_rays_pinhole(
                intrinsic_matrix=o3c.Tensor(intr),
                extrinsic_matrix=o3c.Tensor(v.T_cw.astype(np.float64)),
                width_px=render_w,
                height_px=render_h,
            )
            sil_o = np.isfinite(scene_o.cast_rays(rays)["t_hit"].numpy())
            sil_s = np.isfinite(scene_s.cast_rays(rays)["t_hit"].numpy())
        except Exception as exc:  # pragma: no cover
            print(f"TSDF: silhouette IoU view {cam_idx} skipped: {exc}")
            continue

        union = int(np.logical_or(sil_o, sil_s).sum())
        if union == 0:
            continue
        inter = int(np.logical_and(sil_o, sil_s).sum())
        ious.append(float(inter) / float(union))

    if not ious:
        return True, 1.0
    mean_iou = float(sum(ious) / len(ious))
    return (mean_iou >= min_iou), mean_iou
