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

    root = Path(output_dir_root)

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
    # Create GPU VoxelBlockGrid
    # ------------------------------------------------------------------
    device = o3c.Device("CUDA:0")
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
            depth < tsdf_min_depth_baselines * baseline, 0.0, depth,
        )

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

        # Integrate
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

    # ------------------------------------------------------------------
    # Extract mesh
    # ------------------------------------------------------------------
    if progress_cb:
        progress_cb(0.85, "Extracting triangle mesh from TSDF volume")

    mesh = vbg.extract_triangle_mesh()
    mesh = mesh.to_legacy()

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
    # Save
    # ------------------------------------------------------------------
    cleaned_path = root / "gpu_tsdf_cleaned_mesh.ply"
    o3d.io.write_triangle_mesh(str(cleaned_path), mesh)
    print(f"GPU TSDF: saved cleaned mesh → {cleaned_path}")

    if progress_cb:
        progress_cb(1.0, "GPU TSDF fusion complete")

    return str(cleaned_path)
