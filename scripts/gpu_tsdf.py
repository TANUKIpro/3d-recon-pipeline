"""GPU-accelerated TSDF fusion using Open3D VoxelBlockGrid (CUDA).

Replaces gs2mesh's CPU ``ScalableTSDFVolume`` with the tensor-based
``VoxelBlockGrid`` running on CUDA for significant speedup.
Reads gs2mesh rendered views + stereo depth, fuses on GPU, and
extracts a cleaned triangle mesh.
"""

from __future__ import annotations

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Callable


def gpu_tsdf_reconstruct(
    output_dir_root: str,
    stereo_model: str,
    *,
    tsdf_voxel: int = 2,
    tsdf_sdf_trunc: float = 0.04,
    tsdf_scale: float = 1.0,
    tsdf_min_depth_baselines: int = 4,
    tsdf_max_depth_baselines: int = 20,
    tsdf_dilate: int = 1,
    tsdf_cleaning_threshold: int = 100_000,
    tsdf_use_mask: bool = True,
    tsdf_erode_mask: bool = True,
    tsdf_use_occlusion_mask: bool = True,
    tsdf_invert_mask: bool = False,
    tsdf_erosion_kernel_size: int = 10,
    tsdf_closing_kernel_size: int = 10,
    block_count: int = 100_000,
    tsdf_device: str = "CUDA:0",
    mesh_target_faces: int = 0,
    mesh_min_iou: float = 0.985,
    mesh_iou_views: int = 8,
    progress_cb: Callable[[float, str], None] | None = None,
) -> str:
    """Run GPU TSDF fusion and return path to cleaned mesh PLY.

    Parameters match gs2mesh TSDF parameters; defaults are for the
    *custom* dataset profile.
    """
    # Heavy imports deferred so the module can be imported without GPU.
    import open3d as o3d
    import open3d.core as o3c
    from PIL import Image

    from scripts.vram_utils import log_vram

    root = Path(output_dir_root)
    device_is_cuda = tsdf_device.lower().startswith("cuda")
    try:
        return _gpu_tsdf_impl(
            root=root,
            stereo_model=stereo_model,
            tsdf_voxel=tsdf_voxel,
            tsdf_sdf_trunc=tsdf_sdf_trunc,
            tsdf_scale=tsdf_scale,
            tsdf_min_depth_baselines=tsdf_min_depth_baselines,
            tsdf_max_depth_baselines=tsdf_max_depth_baselines,
            tsdf_dilate=tsdf_dilate,
            tsdf_cleaning_threshold=tsdf_cleaning_threshold,
            tsdf_use_mask=tsdf_use_mask,
            tsdf_erode_mask=tsdf_erode_mask,
            tsdf_use_occlusion_mask=tsdf_use_occlusion_mask,
            tsdf_invert_mask=tsdf_invert_mask,
            tsdf_erosion_kernel_size=tsdf_erosion_kernel_size,
            tsdf_closing_kernel_size=tsdf_closing_kernel_size,
            block_count=block_count,
            tsdf_device=tsdf_device,
            mesh_target_faces=mesh_target_faces,
            mesh_min_iou=mesh_min_iou,
            mesh_iou_views=mesh_iou_views,
            progress_cb=progress_cb,
        )
    finally:
        # VoxelBlockGrid on CUDA allocates ~8GB directly from Open3D's
        # own memory pool (not PyTorch's). torch.cuda.empty_cache()
        # cannot reclaim it — we must call open3d.core.cuda.release_cache()
        # explicitly. Only do any of this when the run actually used CUDA.
        import gc

        if device_is_cuda:
            log_vram("before GPU TSDF release")
            try:
                o3c.cuda.release_cache()
            except Exception as exc:  # pragma: no cover — API varies by version
                print(f"open3d release_cache failed (continuing): {exc}")
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            log_vram("after GPU TSDF release")
        else:
            gc.collect()


def _gpu_tsdf_impl(
    *,
    root: Path,
    stereo_model: str,
    tsdf_voxel: int,
    tsdf_sdf_trunc: float,
    tsdf_scale: float,
    tsdf_min_depth_baselines: int,
    tsdf_max_depth_baselines: int,
    tsdf_dilate: int,
    tsdf_cleaning_threshold: int,
    tsdf_use_mask: bool,
    tsdf_erode_mask: bool,
    tsdf_use_occlusion_mask: bool,
    tsdf_invert_mask: bool,
    tsdf_erosion_kernel_size: int,
    tsdf_closing_kernel_size: int,
    block_count: int,
    tsdf_device: str,
    mesh_target_faces: int,
    mesh_min_iou: float,
    mesh_iou_views: int,
    progress_cb: Callable[[float, str], None] | None,
) -> str:
    """Core implementation — called from `gpu_tsdf_reconstruct` inside its
    try/finally so GPU memory is released even on early return/error.
    """
    import open3d as o3d
    import open3d.core as o3c
    from PIL import Image

    # ------------------------------------------------------------------
    # Load camera data written by gs2mesh Renderer
    # ------------------------------------------------------------------
    with open(root / "camera_data.json") as f:
        cameras = json.load(f)
    left_cameras = [c["left"] for c in cameras]
    baseline = float(left_cameras[0]["baseline"])

    voxel_length = tsdf_voxel / 512.0
    trunc_multiplier = tsdf_sdf_trunc / voxel_length
    depth_max = baseline * tsdf_max_depth_baselines / tsdf_scale

    # ------------------------------------------------------------------
    # Create VoxelBlockGrid — tsdf_device selects CUDA or CPU backend.
    # On low-VRAM tiers (see scripts/vram_tier.py) this runs on CPU so
    # the exact same block_count / voxel_size / depth parameters stay
    # intact; only throughput drops.
    # ------------------------------------------------------------------
    device = o3c.Device(tsdf_device)
    print(f"GPU TSDF: using device={tsdf_device}")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=float(voxel_length),
        block_resolution=16,
        block_count=block_count,
        device=device,
    )

    # Select frames (dilate = take every N-th frame)
    n_cameras = len(left_cameras)
    valid_indices = [i for i in range(n_cameras) if i % tsdf_dilate == 0]
    n_valid = len(valid_indices)

    print(
        f"GPU TSDF: {n_valid} frames, voxel_length={voxel_length:.6f}, "
        f"sdf_trunc={tsdf_sdf_trunc}, baseline={baseline:.4f}, "
        f"depth_max={depth_max:.2f}"
    )

    # ------------------------------------------------------------------
    # Per-frame integration
    # ------------------------------------------------------------------
    integrated_count = 0
    skipped_count = 0
    min_depth_thresh = tsdf_min_depth_baselines * baseline

    for step_i, cam_idx in enumerate(valid_indices):
        cam = left_cameras[cam_idx]
        cam_dir = root / f"{cam_idx:03d}"

        if progress_cb:
            pct = step_i / max(n_valid, 1)
            progress_cb(
                pct * 0.80,
                f"GPU TSDF fusion ({step_i + 1}/{n_valid})",
            )

        # --- Load RGB & depth ---
        image = np.array(Image.open(cam_dir / "left.png")).astype(np.uint8)
        depth = np.load(
            cam_dir / f"out_{stereo_model}" / "depth.npy",
        ).astype(np.float32)

        # --- Object mask (file may not exist when masking was skipped) ---
        if tsdf_use_mask:
            mask_path = cam_dir / "left_mask.npy"
            if mask_path.exists():
                object_mask = np.load(str(mask_path)).astype(bool)
                if tsdf_invert_mask:
                    object_mask = ~object_mask
                if tsdf_erode_mask:
                    closing_kernel = np.ones(
                        (tsdf_closing_kernel_size, tsdf_closing_kernel_size),
                        np.uint8,
                    )
                    erosion_kernel = np.ones(
                        (tsdf_erosion_kernel_size, tsdf_erosion_kernel_size),
                        np.uint8,
                    )
                    closing = cv2.morphologyEx(
                        object_mask.astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        closing_kernel,
                    )
                    erosion = cv2.erode(closing, erosion_kernel, iterations=1)
                    object_mask = erosion > 0.5
                depth = depth * object_mask

        # --- Occlusion mask ---
        if tsdf_use_occlusion_mask:
            occ_path = cam_dir / f"out_{stereo_model}" / "occlusion_mask.npy"
            if occ_path.exists():
                occlusion_mask = np.load(str(occ_path)).astype(bool)
                depth = depth * occlusion_mask

        # --- Depth thresholding (baselines) ---
        depth = np.where(
            depth < min_depth_thresh, 0.0, depth,
        )

        # Fast check: skip frame if no valid depth pixels remain after
        # masking + thresholding.  This avoids Open3D's "No block is
        # touched" error which aborts the entire integration.
        valid_depth_count = int(np.count_nonzero(
            (depth > min_depth_thresh) & (depth < depth_max)
        ))
        if valid_depth_count == 0:
            skipped_count += 1
            continue

        # --- Camera matrices ---
        extrinsic = np.array(cam["extrinsic"], dtype=np.float64)
        extrinsic[:3, 3] /= tsdf_scale
        extrinsic_inv = np.linalg.inv(extrinsic)  # world-to-camera

        intrinsic_np = np.array(
            [
                [cam["fx"], 0.0, cam["cx"]],
                [0.0, cam["fy"], cam["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Camera matrices stay on CPU (Open3D requirement)
        intrinsic_t = o3c.Tensor(intrinsic_np)
        extrinsic_t = o3c.Tensor(extrinsic_inv)

        # Depth & colour → CUDA tensors wrapped as Images.
        # VoxelBlockGrid requires matching dtype pairs: (float32, float32)
        # or (uint16, uint8). Since depth is float32, colour must be too.
        depth_t = o3d.t.geometry.Image(
            o3c.Tensor(depth, device=device),
        )
        color_f = image.astype(np.float32) / 255.0
        color_t = o3d.t.geometry.Image(
            o3c.Tensor(color_f, device=device),
        )

        # Integrate — guard against rare cases where
        # compute_unique_block_coordinates finds no blocks despite the
        # NumPy pre-check above (floating-point edge cases).
        try:
            frustum_coords = vbg.compute_unique_block_coordinates(
                depth_t,
                intrinsic_t,
                extrinsic_t,
                depth_scale=tsdf_scale,
                depth_max=depth_max,
            )
            vbg.integrate(
                frustum_coords,
                depth_t,
                color_t,
                intrinsic_t,
                intrinsic_t,
                extrinsic_t,
                depth_scale=tsdf_scale,
                depth_max=depth_max,
                trunc_voxel_multiplier=trunc_multiplier,
            )
            integrated_count += 1
        except RuntimeError as e:
            if "No block is touched" in str(e):
                skipped_count += 1
                print(
                    f"GPU TSDF: skipping frame {cam_idx} "
                    f"(no voxel blocks touched, {valid_depth_count} "
                    f"valid depth pixels)"
                )
                continue
            raise

    if skipped_count > 0:
        print(
            f"GPU TSDF: integrated {integrated_count}/{n_valid} frames "
            f"(skipped {skipped_count} with no valid depth)"
        )
    if integrated_count == 0:
        raise RuntimeError(
            f"GPU TSDF: all {n_valid} frames skipped — no valid depth "
            f"in range [{min_depth_thresh:.3f}, {depth_max:.2f}]. "
            f"Check stereo depth output and mask coverage."
        )

    # ------------------------------------------------------------------
    # Extract mesh
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.85, "Extracting triangle mesh from TSDF volume")

    mesh = vbg.extract_triangle_mesh()
    mesh = mesh.to_legacy()

    # vbg is no longer needed after to_legacy() — drop the big GPU tensor
    # now so the Open3D pool can release ~8GB before cleaning+saving.
    del vbg

    # Undo the coordinate scaling applied during integration
    mesh.scale(tsdf_scale, (0, 0, 0))
    mesh.orient_triangles()  # Edge-based winding consistency (guards against rare MC artifacts)
    mesh.compute_vertex_normals()

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"GPU TSDF: raw mesh — {n_verts} vertices, {n_tris} triangles")

    # ------------------------------------------------------------------
    # Clean mesh — remove small connected components
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.92, "Cleaning mesh (removing small clusters)")

    thres = tsdf_cleaning_threshold / tsdf_scale
    triangle_clusters, cluster_n_triangles, _ = (
        mesh.cluster_connected_triangles()
    )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < thres
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    n_tris_clean = len(mesh.triangles)
    print(
        f"GPU TSDF: cleaned mesh — {len(mesh.vertices)} vertices, "
        f"{n_tris_clean} triangles (removed {n_tris - n_tris_clean})"
    )

    # ------------------------------------------------------------------
    # Decimate (post-MC, color-preserving QEM) with silhouette safety net
    # ------------------------------------------------------------------
    if mesh_target_faces > 0 and n_tris_clean > mesh_target_faces:
        if progress_cb:
            progress_cb(0.95, "Decimating mesh (quadric edge collapse)")
        simplified, did_simplify = _decimate_tsdf_mesh(
            mesh, mesh_target_faces
        )
        if did_simplify:
            n_tris_simp = len(simplified.triangles)
            n_verts_simp = len(simplified.vertices)
            print(
                f"GPU TSDF: decimated mesh — {n_verts_simp} vertices, "
                f"{n_tris_simp} triangles "
                f"(target={mesh_target_faces}, "
                f"reduction={1.0 - n_tris_simp / max(1, n_tris_clean):.1%})"
            )
            iou_ok, mean_iou = _silhouette_iou_check(
                original=mesh,
                simplified=simplified,
                left_cameras=left_cameras,
                tsdf_scale=tsdf_scale,
                min_iou=mesh_min_iou,
                n_views=mesh_iou_views,
            )
            if iou_ok:
                print(
                    f"GPU TSDF: silhouette IoU={mean_iou:.4f} "
                    f">= {mesh_min_iou:.4f} — keeping decimated mesh"
                )
                mesh = simplified
            else:
                print(
                    f"GPU TSDF: silhouette IoU={mean_iou:.4f} "
                    f"< {mesh_min_iou:.4f} — rolling back to pre-decimation mesh"
                )
        else:
            print("GPU TSDF: decimation produced no reduction, skipping")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    cleaned_path = root / "gpu_tsdf_cleaned_mesh.ply"
    o3d.io.write_triangle_mesh(str(cleaned_path), mesh)
    print(f"GPU TSDF: saved cleaned mesh → {cleaned_path}")

    if progress_cb:
        progress_cb(1.0, "GPU TSDF fusion complete")

    return str(cleaned_path)


def _decimate_tsdf_mesh(mesh, target_faces: int):
    """Quadric-decimate a TSDF mesh to ``target_faces`` triangles.

    Returns ``(mesh, did_simplify)``. The original mesh is returned
    unchanged when simplification would not reduce face count, when the
    output is empty, or when ``target_faces <= 0``.

    Open3D's ``simplify_quadric_decimation`` preserves per-vertex colours
    (RGB attribute carried through from VoxelBlockGrid), so the cap-face
    detection in ``scripts/texture/cap_region.py`` keeps working
    downstream. The cleanup trio mirrors
    ``scripts/texture/bake.py::_simplify_mesh_for_uv``.
    """
    import open3d as o3d  # noqa: F401  (kept for clarity / matching impl style)

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
    left_cameras: list[dict],
    tsdf_scale: float,
    min_iou: float,
    n_views: int,
) -> tuple[bool, float]:
    """Validate that ``simplified`` preserves the silhouette of ``original``.

    Rasterises both meshes from ``n_views`` evenly-sampled poses (drawn
    from ``left_cameras``, the same data used during TSDF integration)
    and computes the mean per-view foreground IoU. Returns
    ``(passed, mean_iou)`` where ``passed`` is true iff
    ``mean_iou >= min_iou``. Any rasterisation failure is treated as a
    pass with IoU 1.0 (we should not roll back on a tooling glitch).
    """

    if n_views <= 0 or not left_cameras:
        return True, 1.0

    import open3d as o3d
    import open3d.core as o3c

    n_cam = len(left_cameras)
    step = max(1, n_cam // max(1, n_views))
    sample_idx = list(range(0, n_cam, step))[:n_views]
    if not sample_idx:
        return True, 1.0

    # Modest fixed render resolution — silhouette comparison doesn't need
    # full image_h × image_w; this keeps the gate at ~100 ms on CUDA.
    render_w = 512
    render_h = 512

    def _build_scene(mesh) -> o3d.t.geometry.RaycastingScene:
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        return scene

    try:
        scene_o = _build_scene(original)
        scene_s = _build_scene(simplified)
    except Exception as exc:  # pragma: no cover — Open3D edge cases
        print(f"GPU TSDF: silhouette IoU check skipped (scene build failed: {exc})")
        return True, 1.0

    ious: list[float] = []
    for cam_idx in sample_idx:
        cam = left_cameras[cam_idx]
        try:
            extrinsic = np.array(cam["extrinsic"], dtype=np.float64)
            extrinsic[:3, 3] /= tsdf_scale
            w2c = np.linalg.inv(extrinsic)

            # Rebuild a compact intrinsic for the down-rendered resolution.
            full_w = float(cam.get("width", render_w))
            full_h = float(cam.get("height", render_h))
            sx = render_w / max(1.0, full_w)
            sy = render_h / max(1.0, full_h)
            intr = np.array(
                [
                    [cam["fx"] * sx, 0.0, cam["cx"] * sx],
                    [0.0, cam["fy"] * sy, cam["cy"] * sy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            rays = scene_o.create_rays_pinhole(
                intrinsic_matrix=o3c.Tensor(intr),
                extrinsic_matrix=o3c.Tensor(w2c),
                width_px=render_w,
                height_px=render_h,
            )
            sil_o = np.isfinite(scene_o.cast_rays(rays)["t_hit"].numpy())
            sil_s = np.isfinite(scene_s.cast_rays(rays)["t_hit"].numpy())
        except Exception as exc:  # pragma: no cover
            print(f"GPU TSDF: silhouette IoU view {cam_idx} skipped: {exc}")
            continue

        union = np.logical_or(sil_o, sil_s).sum()
        if union == 0:
            continue
        inter = np.logical_and(sil_o, sil_s).sum()
        ious.append(float(inter) / float(union))

    if not ious:
        return True, 1.0

    mean_iou = float(sum(ious) / len(ious))
    return (mean_iou >= min_iou), mean_iou
