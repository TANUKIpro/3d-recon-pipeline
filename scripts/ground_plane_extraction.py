"""Ground plane extraction via COLMAP sparse points or mesh projection.

Primary method: classifies COLMAP sparse 3D points against SAM2 ground
masks using COLMAP's own tracked 2D observations (sub-pixel accuracy,
no intrinsics dependency).  Falls back to the legacy mesh-vertex
projection approach when COLMAP data is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _emit_progress(progress_cb, progress: float, detail: str) -> None:
    if progress_cb is not None:
        progress_cb(float(progress), str(detail))


def _check_cancel(cancel_cb) -> None:
    if cancel_cb is not None:
        cancel_cb()


def _find_recon_dir(colmap_sparse_dir: str | Path) -> Path | None:
    """Find the first valid COLMAP reconstruction subdirectory, or None."""
    sparse = Path(colmap_sparse_dir)
    if not sparse.is_dir():
        return None
    for d in sorted(sparse.iterdir()):
        if d.is_dir() and (d / "points3D.bin").exists():
            return d
    return None


def _fit_and_orient_ground_plane(
    ground_points: np.ndarray,
    all_points: np.ndarray,
    ground_mask: np.ndarray,
) -> dict | None:
    """RANSAC plane fit + orient normal toward non-ground centroid.

    *ground_points*: (M, 3) classified ground coordinates.
    *all_points*: (N, 3) all point coordinates (for orientation).
    *ground_mask*: (N,) bool mask identifying ground points in *all_points*.

    Returns the ground plane result dict or None on failure.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_points)

    # Adaptive distance_threshold based on scene scale
    bbox_diag = max(
        float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))),
        1e-6,
    )
    distance_threshold = bbox_diag * 0.005

    try:
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=2000,
        )
    except Exception:
        return None

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-10:
        return None
    normal = normal / norm_len
    plane_d = float(d) / norm_len

    # Orient normal toward the non-ground centroid
    non_ground = ~ground_mask
    if np.any(non_ground):
        object_centroid = all_points[non_ground].mean(axis=0)
    else:
        object_centroid = all_points.mean(axis=0)

    ground_center = ground_points.mean(axis=0)
    to_object = object_centroid - ground_center
    if np.dot(normal, to_object) < 0:
        normal = -normal
        plane_d = -plane_d

    inlier_ratio = len(inlier_indices) / len(ground_points) if len(ground_points) > 0 else 0.0

    return {
        "normal": normal.tolist(),
        "d": float(plane_d),
        "plane_normal": normal.tolist(),
        "plane_d": float(plane_d),
        "center": ground_center.tolist(),
        "point_count": int(len(ground_points)),
        "inlier_ratio": round(float(inlier_ratio), 4),
    }


def _save_result(result: dict, output_dir: str) -> None:
    output_path = Path(output_dir) / "ground_plane.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -------------------------------------------------------------------------
#  Primary: COLMAP sparse points
# -------------------------------------------------------------------------

def _extract_via_colmap_sparse(
    colmap_sparse_dir: str,
    ground_mask_dir: str,
    output_dir: str,
    *,
    min_ground_points: int = 50,
    vote_threshold: float = 0.3,
    min_inside_views: int = 1,
    progress_cb=None,
    cancel_cb=None,
) -> dict | None:
    """Classify COLMAP sparse points against ground masks and fit a plane."""
    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 5.0, "Loading COLMAP sparse model")

    recon_dir = _find_recon_dir(colmap_sparse_dir)
    if recon_dir is None:
        return None

    from scripts.colmap_sparse_filter import classify_points_by_mask, read_colmap_model

    try:
        model = read_colmap_model(recon_dir)
    except Exception:
        return None

    if not model.points3D:
        return None

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 20.0, "Classifying ground points via multi-view voting")

    ground_ids = classify_points_by_mask(
        model,
        ground_mask_dir,
        min_inside_views=min_inside_views,
        min_inside_ratio=vote_threshold,
        mask_dilate=0,
    )

    if len(ground_ids) < min_ground_points:
        _emit_progress(
            progress_cb, 50.0,
            f"COLMAP sparse: only {len(ground_ids)} ground points "
            f"(need {min_ground_points})",
        )
        return None

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 60.0, "Fitting ground plane (RANSAC)")

    # Build coordinate arrays
    all_ids = list(model.points3D.keys())
    all_points = np.array(
        [model.points3D[pid].xyz for pid in all_ids], dtype=np.float64,
    )
    ground_id_set = ground_ids
    ground_mask_arr = np.array(
        [pid in ground_id_set for pid in all_ids], dtype=bool,
    )
    ground_points = all_points[ground_mask_arr]

    result = _fit_and_orient_ground_plane(ground_points, all_points, ground_mask_arr)
    if result is None:
        return None

    result["source"] = "colmap_sparse"
    _save_result(result, output_dir)
    _emit_progress(progress_cb, 100.0, "Ground plane extracted from COLMAP sparse points")
    return result


# -------------------------------------------------------------------------
#  Fallback: mesh vertex projection (legacy)
# -------------------------------------------------------------------------

def _load_intrinsics(intrinsics_path: str | Path) -> tuple[np.ndarray, int, int] | None:
    """Load camera intrinsics from JSON. Returns (K, width, height) or None."""
    from scripts.texture.intrinsics import _make_K

    path = Path(intrinsics_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("K"), list):
        try:
            K = np.asarray(data["K"], dtype=np.float64)
            img_w = int(data.get("image_width") or data.get("width") or 0)
            img_h = int(data.get("image_height") or data.get("height") or 0)
            if K.shape == (3, 3) and img_w > 0 and img_h > 0:
                return K, img_w, img_h
        except Exception:
            pass
    try:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        img_w = int(data.get("image_width") or data.get("width") or 0)
        img_h = int(data.get("image_height") or data.get("height") or 0)
        if img_w > 0 and img_h > 0:
            return _make_K(fx, fy, cx, cy), img_w, img_h
    except Exception:
        return None
    return None


def _extract_via_mesh(
    mesh_ply_path: str,
    ground_mask_dir: str,
    output_dir: str,
    poses_path: str,
    intrinsics_path: str,
    *,
    object_mask_dir: str | None = None,
    min_ground_points: int = 100,
    vote_threshold: float = 0.3,
    max_views: int = 16,
    progress_cb=None,
    cancel_cb=None,
) -> dict | None:
    """Legacy mesh-vertex-projection approach (fallback)."""
    from scripts.texture.intrinsics import _project_simple
    from scripts.texture.io_utils import _load_mask, _load_poses

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 5.0, "Loading mesh and camera data (fallback)")

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_ply_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return None

    try:
        poses, frame_indices = _load_poses(poses_path)
    except Exception:
        return None
    if len(poses) == 0:
        return None

    intrinsics = _load_intrinsics(intrinsics_path)
    if intrinsics is None:
        return None
    K, img_w, img_h = intrinsics

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 15.0, "Projecting vertices into camera views")

    n_views = min(len(poses), max_views)
    view_indices = np.linspace(0, len(poses) - 1, n_views, dtype=int)

    ground_votes = np.zeros(len(vertices), dtype=np.float64)
    visible_count = np.zeros(len(vertices), dtype=np.float64)

    for vi, pose_idx in enumerate(view_indices):
        _check_cancel(cancel_cb)
        src_idx = int(frame_indices[int(pose_idx)])

        try:
            ground_mask = _load_mask(ground_mask_dir, src_idx)
        except Exception:
            continue

        uv, depths = _project_simple(vertices, poses[int(pose_idx)], K)
        u = uv[:, 0]
        v = uv[:, 1]

        valid = (
            (depths > 0.01)
            & (u >= 0.0)
            & (u < float(img_w))
            & (v >= 0.0)
            & (v < float(img_h))
        )
        if not np.any(valid):
            continue

        ui = np.clip(np.round(u[valid]).astype(np.int32), 0, ground_mask.shape[1] - 1)
        vi_px = np.clip(np.round(v[valid]).astype(np.int32), 0, ground_mask.shape[0] - 1)

        visible_count[valid] += 1.0

        in_ground = ground_mask[vi_px, ui]
        ground_votes[valid] += in_ground.astype(np.float64)

        progress = 15.0 + 55.0 * ((vi + 1) / n_views)
        _emit_progress(progress_cb, progress, f"Processing view {vi + 1}/{n_views}")

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 75.0, "Classifying ground vertices")

    with np.errstate(divide="ignore", invalid="ignore"):
        ground_ratio = np.where(visible_count > 0, ground_votes / visible_count, 0.0)

    ground_mask_verts = ground_ratio > vote_threshold
    ground_indices = np.where(ground_mask_verts)[0]

    if len(ground_indices) < min_ground_points:
        _emit_progress(progress_cb, 100.0, "Insufficient ground points for plane fit")
        return None

    ground_points = vertices[ground_indices]

    _emit_progress(progress_cb, 80.0, "Fitting ground plane (RANSAC)")

    result = _fit_and_orient_ground_plane(ground_points, vertices, ground_mask_verts)
    if result is None:
        return None

    result["source"] = "mesh_projection"
    _save_result(result, output_dir)
    _emit_progress(progress_cb, 100.0, "Ground plane extraction complete (mesh fallback)")
    return result


# -------------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------------

def extract_ground_plane(
    ground_mask_dir: str,
    output_dir: str,
    *,
    colmap_sparse_dir: str | None = None,
    mesh_ply_path: str | None = None,
    poses_path: str | None = None,
    intrinsics_path: str | None = None,
    object_mask_dir: str | None = None,
    min_ground_points: int = 50,
    vote_threshold: float = 0.3,
    progress_cb=None,
    cancel_cb=None,
) -> dict | None:
    """Extract ground plane — COLMAP sparse primary, mesh fallback.

    Returns the ground plane dict (also saved to ground_plane.json) or None.
    """
    # Primary: COLMAP sparse points (no intrinsics dependency, sub-pixel accuracy)
    if colmap_sparse_dir and _find_recon_dir(colmap_sparse_dir) is not None:
        _emit_progress(progress_cb, 2.0, "Trying COLMAP sparse point classification")
        result = _extract_via_colmap_sparse(
            colmap_sparse_dir,
            ground_mask_dir,
            output_dir,
            min_ground_points=min_ground_points,
            vote_threshold=vote_threshold,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        if result is not None:
            return result
        print("Ground plane: COLMAP sparse extraction insufficient, trying mesh fallback")

    # Fallback: mesh vertex projection
    if mesh_ply_path and poses_path and intrinsics_path:
        return _extract_via_mesh(
            mesh_ply_path,
            ground_mask_dir,
            output_dir,
            poses_path,
            intrinsics_path,
            object_mask_dir=object_mask_dir,
            min_ground_points=min_ground_points,
            vote_threshold=vote_threshold,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    return None


def extract_ground_plane_from_mesh(
    mesh_ply_path: str,
    ground_mask_dir: str,
    output_dir: str,
    poses_path: str,
    intrinsics_path: str,
    *,
    object_mask_dir: str | None = None,
    colmap_sparse_dir: str | None = None,
    min_ground_points: int = 50,
    vote_threshold: float = 0.3,
    max_views: int = 16,
    progress_cb=None,
    cancel_cb=None,
) -> dict | None:
    """Backward-compatible entry point.

    Delegates to :func:`extract_ground_plane` (COLMAP primary + mesh
    fallback).
    """
    return extract_ground_plane(
        ground_mask_dir,
        output_dir,
        colmap_sparse_dir=colmap_sparse_dir,
        mesh_ply_path=mesh_ply_path,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        object_mask_dir=object_mask_dir,
        min_ground_points=min_ground_points,
        vote_threshold=vote_threshold,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
