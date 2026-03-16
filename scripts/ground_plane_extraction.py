"""Ground plane extraction from mesh + SAM2 ground masks.

Projects mesh vertices into camera views, checks overlap with SAM2 ground
masks, and fits a RANSAC plane to ground-classified vertices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.texture.intrinsics import _make_K, _project_simple
from scripts.texture.io_utils import _load_mask, _load_poses


def _load_intrinsics(intrinsics_path: str | Path) -> tuple[np.ndarray, int, int] | None:
    """Load camera intrinsics from JSON. Returns (K, width, height) or None."""
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


def _emit_progress(progress_cb, progress: float, detail: str) -> None:
    if progress_cb is not None:
        progress_cb(float(progress), str(detail))


def _check_cancel(cancel_cb) -> None:
    if cancel_cb is not None:
        cancel_cb()


def extract_ground_plane_from_mesh(
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
    """Extract ground plane by projecting mesh vertices into SAM2 ground masks.

    Returns the ground plane dict (also saved to ground_plane.json) or None
    if no reliable plane could be fit.
    """
    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 5.0, "Loading mesh and camera data")

    import open3d as o3d

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_ply_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return None

    # Load poses and intrinsics
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

    # Sample views
    n_views = min(len(poses), max_views)
    view_indices = np.linspace(0, len(poses) - 1, n_views, dtype=int)

    # Per-vertex vote accumulators
    ground_votes = np.zeros(len(vertices), dtype=np.float64)
    visible_count = np.zeros(len(vertices), dtype=np.float64)

    for vi, pose_idx in enumerate(view_indices):
        _check_cancel(cancel_cb)
        src_idx = int(frame_indices[int(pose_idx)])

        # Load ground mask
        try:
            ground_mask = _load_mask(ground_mask_dir, src_idx)
        except Exception:
            continue

        # Load object mask if available (for filtering)
        obj_mask = None
        if object_mask_dir:
            try:
                obj_mask = _load_mask(object_mask_dir, src_idx)
            except Exception:
                pass

        # Project all vertices
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

        # Check ground mask
        in_ground = ground_mask[vi_px, ui]
        ground_votes[valid] += in_ground.astype(np.float64)

        progress = 15.0 + 55.0 * ((vi + 1) / n_views)
        _emit_progress(progress_cb, progress, f"Processing view {vi + 1}/{n_views}")

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 75.0, "Classifying ground vertices")

    # Compute ground ratio per vertex
    with np.errstate(divide="ignore", invalid="ignore"):
        ground_ratio = np.where(visible_count > 0, ground_votes / visible_count, 0.0)

    ground_mask_verts = ground_ratio > vote_threshold
    ground_indices = np.where(ground_mask_verts)[0]

    if len(ground_indices) < min_ground_points:
        _emit_progress(progress_cb, 100.0, "Insufficient ground points for plane fit")
        return None

    ground_points = vertices[ground_indices]

    _emit_progress(progress_cb, 80.0, "Fitting ground plane (RANSAC)")

    # RANSAC plane fit
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_points)
    try:
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000,
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

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 90.0, "Orienting ground plane normal")

    # Orient normal toward object centroid (non-ground vertices)
    non_ground_mask = ~ground_mask_verts
    if np.any(non_ground_mask):
        object_centroid = vertices[non_ground_mask].mean(axis=0)
    else:
        object_centroid = vertices.mean(axis=0)

    ground_center = ground_points.mean(axis=0)
    to_object = object_centroid - ground_center
    if np.dot(normal, to_object) < 0:
        normal = -normal
        plane_d = -plane_d

    inlier_ratio = len(inlier_indices) / len(ground_points) if len(ground_points) > 0 else 0.0

    result = {
        "normal": normal.tolist(),
        "d": float(plane_d),
        "plane_normal": normal.tolist(),
        "plane_d": float(plane_d),
        "center": ground_center.tolist(),
        "point_count": int(len(ground_indices)),
        "inlier_ratio": round(float(inlier_ratio), 4),
    }

    # Save to file
    output_path = Path(output_dir) / "ground_plane.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _emit_progress(progress_cb, 100.0, "Ground plane extraction complete")
    return result
