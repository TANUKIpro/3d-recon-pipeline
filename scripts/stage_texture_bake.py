"""Stage 7: Texture baking — point-cloud KNN colour with frame-projection fallback.

Primary colouring uses KNN inverse-distance-weighted interpolation from the
denoised point cloud (Pi3X RGB).  Texels that remain uncovered in sparse regions
fall back to the legacy intrinsics-estimation + camera-projected view pipeline.

Adapted from im2pc/host/extract_intrinsics.py and im2pc/host/texture_mesh.py.
"""

import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from plyfile import PlyData

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for GPU acceleration
    torch = None

ProgressCallback = Callable[[float, str | None], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


# ---------------------------------------------------------------------------
# Intrinsics estimation
# ---------------------------------------------------------------------------

def _load_point_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load PLY, return (points (N,3), colors (N,3) in [0,1])."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    colors = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float64) / 255.0
    return points, colors


def _load_poses(path: str) -> tuple[np.ndarray, list[int]]:
    with open(path) as f:
        data = json.load(f)
    poses = np.array(data["poses"], dtype=np.float64)
    frame_indices = data.get("frame_indices")
    if frame_indices is None:
        frame_indices = list(range(len(poses)))
    else:
        frame_indices = [int(i) for i in frame_indices]
    if len(frame_indices) != len(poses):
        print(
            f"Warning: poses={len(poses)} but frame_indices={len(frame_indices)}. "
            "Using positional indices."
        )
        frame_indices = list(range(len(poses)))
    return poses, frame_indices


def _resolve_indexed_file(base_dir: str, idx: int, suffix: str) -> Path:
    """Resolve by numbered filename first, then by sorted positional index."""
    path = Path(base_dir) / f"{idx:05d}{suffix}"
    if path.is_file():
        return path

    files = _list_indexed_files(base_dir, suffix)
    if 0 <= idx < len(files):
        return Path(files[idx])
    raise FileNotFoundError(f"Indexed file not found: {base_dir} idx={idx} suffix={suffix}")


@lru_cache(maxsize=16)
def _list_indexed_files(base_dir: str, suffix: str) -> tuple[str, ...]:
    return tuple(str(p) for p in sorted(Path(base_dir).glob(f"*{suffix}")))


def _load_frame(frames_dir: str, idx: int) -> np.ndarray:
    path = _resolve_indexed_file(frames_dir, idx, ".jpg")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def _load_mask(masks_dir: str, idx: int) -> np.ndarray:
    path = _resolve_indexed_file(masks_dir, idx, ".png")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask > 127


def _make_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _project_points(pts, c2w, K, img_w, img_h):
    """Project 3D→2D. Returns (uv (M,2), valid (N,), depths (N,))."""
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    depths = cam[:, 2]
    valid = depths > 0.01
    pts_v = cam[valid]
    sz = np.maximum(pts_v[:, 2], 1e-10)
    u = K[0, 0] * pts_v[:, 0] / sz + K[0, 2]
    v = K[1, 1] * pts_v[:, 1] / sz + K[1, 2]
    in_bounds = (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)
    uv = np.column_stack([u[in_bounds], v[in_bounds]])
    valid_idx = np.where(valid)[0]
    full_valid = np.zeros(len(pts), dtype=bool)
    full_valid[valid_idx[in_bounds]] = True
    return uv, full_valid, depths


def _estimate_intrinsics(
    points, colors, poses, pose_frame_indices, frames_dir, masks_dir, img_w, img_h,
    num_eval_frames=10, subsample_points=50000,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    """Estimate camera intrinsics via grid search + Nelder-Mead."""
    from scipy.optimize import minimize

    # Subsample
    if len(points) > subsample_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), subsample_points, replace=False)
        pts_sub, col_sub = points[idx], colors[idx]
    else:
        pts_sub, col_sub = points, colors

    n_frames = len(poses)
    eval_pose_indices = np.linspace(0, n_frames - 1, num_eval_frames, dtype=int).tolist()
    print(f"  Evaluating {len(eval_pose_indices)} frames, {len(pts_sub)} points")

    cx_init, cy_init = img_w / 2.0, img_h / 2.0

    def color_score(K):
        total_err, total_cnt = 0.0, 0
        for pose_idx in eval_pose_indices:
            src_idx = int(pose_frame_indices[pose_idx])
            try:
                frame = _load_frame(frames_dir, src_idx)
                mask = _load_mask(masks_dir, src_idx)
            except FileNotFoundError:
                continue
            uv, valid, _ = _project_points(pts_sub, poses[pose_idx], K, img_w, img_h)
            if uv.shape[0] == 0:
                continue
            ui, vi = uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)
            m = mask[vi, ui]
            if m.sum() == 0:
                continue
            diff = frame[vi[m], ui[m]] - col_sub[valid][m]
            total_err += np.mean(diff**2) * m.sum()
            total_cnt += m.sum()
        return -(total_err / total_cnt) if total_cnt > 0 else -1.0

    # Grid search
    best_score, best_fov = -np.inf, 60
    print("  Grid search over FOV...")
    fov_values = list(range(35, 85, 2))
    for idx, fov in enumerate(fov_values):
        fx = img_w / (2.0 * np.tan(np.radians(fov) / 2.0))
        score = color_score(_make_K(fx, fx, cx_init, cy_init))
        if score > best_score:
            best_score, best_fov = score, fov
        if idx % 2 == 0 or idx == len(fov_values) - 1:
            ratio = (idx + 1) / len(fov_values)
            _emit_progress(
                progress_cb,
                ratio * 70.0,
                f"Estimating intrinsics (grid search {idx + 1}/{len(fov_values)})",
            )

    print(f"  Best FOV: {best_fov}° (score={best_score:.6f})")

    # Refine
    fx_init = img_w / (2.0 * np.tan(np.radians(best_fov) / 2.0))

    def objective(params):
        fx, fy, cx, cy = params
        if fx < 100 or fy < 100 or fx > 5000 or fy > 5000:
            return 1.0
        if cx < 0 or cx > img_w or cy < 0 or cy > img_h:
            return 1.0
        return -color_score(_make_K(fx, fy, cx, cy))

    result = minimize(objective, [fx_init, fx_init, cx_init, cy_init],
                      method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 0.5, "fatol": 1e-7, "adaptive": True})
    _emit_progress(progress_cb, 100.0, "Intrinsics optimization complete")

    fx, fy, cx, cy = result.x
    print(f"  Optimized: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    return {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "image_width": img_w, "image_height": img_h,
        "K": _make_K(fx, fy, cx, cy).tolist(),
    }


# ---------------------------------------------------------------------------
# Texture baking
# ---------------------------------------------------------------------------

def _bilinear_sample(img, x, y):
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    fx = (x - x0.astype(np.float64))[:, None]
    fy = (y - y0.astype(np.float64))[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x1]
            + (1 - fx) * fy * img[y1, x0] + fx * fy * img[y1, x1])


def _project_simple(pts, c2w, K):
    """Simple 3D→2D projection. Returns (uv (N,2), depths (N,))."""
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    d = cam[:, 2].copy()
    sz = np.maximum(d, 1e-10)
    u = K[0, 0] * cam[:, 0] / sz + K[0, 2]
    v = K[1, 1] * cam[:, 1] / sz + K[1, 2]
    return np.column_stack([u, v]), d


# ---------------------------------------------------------------------------
# Point-cloud KNN colour interpolation
# ---------------------------------------------------------------------------


def _estimate_pc_normals(pc_points: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
    """Estimate point cloud normals via Open3D.

    Args:
        pc_points: (N, 3) point cloud positions.
        k_neighbors: number of neighbours for normal estimation.

    Returns:
        (N, 3) float64 normals, oriented towards the centroid.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )
    center = pc_points.mean(axis=0)
    pcd.orient_normals_towards_camera_location(center.tolist())
    normals = np.asarray(pcd.normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-10)
    return normals


def _kdtree_color_interpolation(
    pos3d: np.ndarray,
    pc_points: np.ndarray,
    pc_colors: np.ndarray,
    k: int = 8,
    max_distance: float = 0.0,
    idw_power: float = 2.0,
    batch_size: int = 50_000,
    progress_cb: ProgressCallback | None = None,
    texel_normals: np.ndarray | None = None,
    pc_normals: np.ndarray | None = None,
    normal_threshold_deg: float = 0.0,
    adaptive_k: bool = False,
    weighting: str = "idw",
    gaussian_sigma_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Colour texels via KNN interpolation from a point cloud.

    Args:
        pos3d: (M, 3) texel 3D positions.
        pc_points: (N, 3) point cloud XYZ.
        pc_colors: (N, 3) point cloud RGB in [0, 1].
        k: number of nearest neighbours.
        max_distance: discard neighbours farther than this.
            If 0, auto-set to ``median(k-th NN distance over 10 K samples) * 3``.
        idw_power: exponent for inverse-distance weighting.
        batch_size: query batch size to limit peak memory.
        progress_cb: optional ``(progress%, detail)`` callback.
        texel_normals: (M, 3) per-texel normals (face normals).
        pc_normals: (N, 3) per-point normals from the point cloud.
        normal_threshold_deg: discard neighbours whose normal deviates
            more than this many degrees from the texel normal (0 = disabled).
        adaptive_k: dynamically reduce k in dense regions for sharper results.
        weighting: ``"idw"`` (inverse-distance) or ``"gaussian"``.
        gaussian_sigma_factor: sigma = nearest-neighbour distance * factor.

    Returns:
        (colors (M, 3) float32, covered (M,) bool)
    """
    from scipy.spatial import cKDTree

    M = len(pos3d)
    use_normal_filter = (
        normal_threshold_deg > 0.0
        and texel_normals is not None
        and pc_normals is not None
    )
    normal_cos_thresh = math.cos(math.radians(normal_threshold_deg)) if use_normal_filter else -1.0
    query_k = min(k * 3, len(pc_points)) if use_normal_filter else min(k, len(pc_points))
    k = min(k, len(pc_points))
    tree = cKDTree(pc_points)

    # Auto max_distance from sampled k-th NN distances.
    if max_distance <= 0.0:
        n_sample = min(10_000, len(pc_points))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(pc_points), size=n_sample, replace=False)
        sample_dists, _ = tree.query(pc_points[sample_idx], k=k)
        kth_dists = sample_dists[:, -1] if sample_dists.ndim == 2 else sample_dists
        max_distance = float(np.median(kth_dists) * 3.0)
        print(f"  KNN auto max_distance = {max_distance:.6f}")

    # For adaptive_k: compute median nearest-neighbour distance over point cloud.
    median_nn_dist = None
    if adaptive_k:
        n_sample = min(10_000, len(pc_points))
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(pc_points), size=n_sample, replace=False)
        sample_dists_1, _ = tree.query(pc_points[sample_idx], k=2)
        median_nn_dist = float(np.median(sample_dists_1[:, 1]))

    colors = np.zeros((M, 3), dtype=np.float32)
    covered = np.zeros(M, dtype=bool)
    eps = 1e-12

    n_batches = math.ceil(M / batch_size)
    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, M)
        chunk_size = end - start
        dists, idxs = tree.query(pos3d[start:end], k=query_k)

        if dists.ndim == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        # --- Normal filtering ---
        if use_normal_filter:
            t_norms = texel_normals[start:end]  # (chunk, 3)
            nn_norms = pc_normals[idxs]  # (chunk, query_k, 3)
            cos_vals = np.einsum("ij,ikj->ik", t_norms, nn_norms)  # (chunk, query_k)
            normal_ok = cos_vals >= normal_cos_thresh
        else:
            normal_ok = np.ones((chunk_size, dists.shape[1]), dtype=bool)

        # --- Distance filtering ---
        within = dists <= max_distance
        valid_mask = within & normal_ok

        # Limit each row to at most k valid neighbours.
        for ri in range(chunk_size):
            valid_cols = np.where(valid_mask[ri])[0]
            if len(valid_cols) > k:
                valid_mask[ri, valid_cols[k:]] = False

        any_valid = np.any(valid_mask, axis=1)

        # Fallback: if normal filter removed ALL neighbours, keep nearest 1
        # (column 0 is always nearest since cKDTree returns sorted results).
        if use_normal_filter:
            no_valid = ~any_valid & within.any(axis=1)
            if np.any(no_valid):
                valid_mask[no_valid, 0] = True
                any_valid = np.any(valid_mask, axis=1)

        if np.any(any_valid):
            # --- Adaptive k: reduce effective neighbours in dense regions ---
            if adaptive_k and median_nn_dist is not None and median_nn_dist > 0:
                nn1_dist = dists[:, 0]  # nearest neighbour distance
                density_ratio = nn1_dist / max(median_nn_dist, eps)
                # In dense regions (ratio < 1), use fewer neighbours.
                eff_k = np.clip(
                    np.round(k * density_ratio).astype(int), 1, k
                )
                for ri in np.where(any_valid)[0]:
                    valid_cols = np.where(valid_mask[ri])[0]
                    if len(valid_cols) > eff_k[ri]:
                        valid_mask[ri, valid_cols[eff_k[ri]:]] = False

            # --- Weight computation ---
            if weighting == "gaussian":
                sigma = np.maximum(dists[:, 0:1], eps) * gaussian_sigma_factor
                w = np.where(
                    valid_mask,
                    np.exp(-0.5 * (dists / np.maximum(sigma, eps)) ** 2),
                    0.0,
                )
            else:
                w = np.where(
                    valid_mask,
                    1.0 / np.maximum(dists, eps) ** idw_power,
                    0.0,
                )

            w_sum = w.sum(axis=1, keepdims=True)
            w_sum = np.maximum(w_sum, eps)
            w_norm = w / w_sum  # (chunk, query_k)

            nn_colors = pc_colors[idxs]  # (chunk, query_k, 3)
            interp = np.einsum("ij,ijk->ik", w_norm, nn_colors)  # (chunk, 3)

            chunk_covered = any_valid
            colors[start:end][chunk_covered] = interp[chunk_covered].astype(np.float32)
            covered[start:end] |= chunk_covered

        if progress_cb is not None and (bi % max(1, n_batches // 20) == 0 or bi == n_batches - 1):
            ratio = (bi + 1) / n_batches
            _emit_progress(
                progress_cb,
                60.0 + ratio * 22.0,
                f"KNN colour interpolation ({bi + 1}/{n_batches} batches)",
            )

    return colors, covered


def _resolve_texture_device(requested: str | None = None) -> str:
    mode = (requested or os.environ.get("TEXTURE_DEVICE", "cuda")).strip().lower()
    if mode == "gpu":
        mode = "cuda"
    if mode not in {"auto", "cpu", "cuda"}:
        print(f"Warning: invalid TEXTURE_DEVICE='{mode}', falling back to cuda")
        mode = "cuda"

    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        print("Warning: TEXTURE_DEVICE=cuda requested but CUDA is unavailable; using CPU")
        return "cpu"
    # auto
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_texture_size(
    requested_size: int | None,
    img_w: int,
    img_h: int,
) -> tuple[int, bool]:
    """Resolve final texture size.

    Returns:
        (size_px, is_auto)
    """
    if requested_size is None:
        requested_size = os.environ.get("TEXTURE_SIZE", "0")

    try:
        resolved = int(requested_size)
    except (TypeError, ValueError):
        resolved = 0

    if resolved > 0:
        return resolved, False

    if img_w > 0 and img_h > 0:
        auto_size = max(1, int(round(math.sqrt(float(img_w) * float(img_h)))))
        return auto_size, True

    # Last-resort safety fallback when image metadata is unavailable.
    return 2048, True


def _build_face_charts(faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Group faces into UV charts by shared triangle edges."""
    n_faces = int(len(faces))
    if n_faces == 0:
        return np.zeros(0, dtype=np.int32), 0

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edges = ((a, b), (b, c), (c, a))
        for v0, v1 in edges:
            key = (v0, v1) if v0 < v1 else (v1, v0)
            edge_to_faces[key].append(fi)

    neighbors: list[list[int]] = [[] for _ in range(n_faces)]
    for attached in edge_to_faces.values():
        if len(attached) <= 1:
            continue
        root = attached[0]
        for other in attached[1:]:
            neighbors[root].append(other)
            neighbors[other].append(root)

    chart_ids = np.full(n_faces, -1, dtype=np.int32)
    chart_count = 0
    stack: list[int] = []
    for start in range(n_faces):
        if chart_ids[start] >= 0:
            continue
        chart_ids[start] = chart_count
        stack.append(start)
        while stack:
            cur = stack.pop()
            for nxt in neighbors[cur]:
                if chart_ids[nxt] >= 0:
                    continue
                chart_ids[nxt] = chart_count
                stack.append(nxt)
        chart_count += 1

    return chart_ids, chart_count


def _group_texels_by_chart(
    texel_chart_ids: np.ndarray,
    n_charts: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    counts = np.bincount(texel_chart_ids, minlength=n_charts).astype(np.int32, copy=False)
    if n_charts == 0:
        return [], counts
    if texel_chart_ids.size == 0:
        return [np.empty(0, dtype=np.int32) for _ in range(n_charts)], counts
    order = np.argsort(texel_chart_ids, kind="stable")
    splits = np.cumsum(counts[:-1], dtype=np.int64).tolist()
    grouped = np.split(order, splits)
    return [g.astype(np.int32, copy=False) for g in grouped], counts


def _rasterize_view_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Rasterize mesh depth into camera image space for a simple z-test."""
    uv_all, depth_all = _project_simple(vertices, c2w, K)
    depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)
    inside_eps = 1e-5

    for face in faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        z0, z1, z2 = float(depth_all[i0]), float(depth_all[i1]), float(depth_all[i2])
        if z0 <= 0.01 and z1 <= 0.01 and z2 <= 0.01:
            continue

        x0, y0 = uv_all[i0]
        x1, y1 = uv_all[i1]
        x2, y2 = uv_all[i2]

        min_x = int(max(0, math.floor(min(x0, x1, x2))))
        max_x = int(min(img_w - 1, math.ceil(max(x0, x1, x2))))
        min_y = int(max(0, math.floor(min(y0, y1, y2))))
        max_y = int(min(img_h - 1, math.ceil(max(y0, y1, y2))))
        if min_x > max_x or min_y > max_y:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
        w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -inside_eps) & (w1 >= -inside_eps) & (w2 >= -inside_eps)
        if not np.any(inside):
            continue

        depth = w0 * z0 + w1 * z1 + w2 * z2
        valid = inside & (depth > 0.01)
        if not np.any(valid):
            continue

        tile = depth_buffer[min_y:max_y + 1, min_x:max_x + 1]
        candidate = np.where(valid, depth, np.inf).astype(np.float32, copy=False)
        np.minimum(tile, candidate, out=tile)

    return depth_buffer


def _evaluate_view_samples(
    pos3d: np.ndarray,
    normals: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    mask_bool: np.ndarray,
    depth_buffer: np.ndarray,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate which texels can be sampled from a view and return per-texel scores."""
    cam_pos = c2w[:3, 3]
    view_dirs = cam_pos[None, :] - pos3d
    dists = np.linalg.norm(view_dirs, axis=1)
    view_dirs_n = view_dirs / np.maximum(dists[:, None], 1e-10)
    cos_angle = np.sum(normals * view_dirs_n, axis=1)

    uv2d, depths = _project_simple(pos3d, c2w, K)
    px_proj = uv2d[:, 0]
    py_proj = uv2d[:, 1]
    in_bounds = (
        (depths > 0.01)
        & (px_proj >= 0)
        & (px_proj < img_w - 1)
        & (py_proj >= 0)
        & (py_proj < img_h - 1)
    )

    pxi = np.clip(px_proj.astype(np.int32), 0, img_w - 1)
    pyi = np.clip(py_proj.astype(np.int32), 0, img_h - 1)
    mask_ok = mask_bool[pyi, pxi]

    depth_ref = depth_buffer[pyi, pxi].astype(np.float64)
    visibility_eps = np.maximum(1e-4, 0.01 * depth_ref)
    visible = depths <= (depth_ref + visibility_eps)

    facing = cos_angle > min_cos
    valid = in_bounds & mask_ok & visible & facing

    score = np.zeros(len(pos3d), dtype=np.float64)
    if np.any(valid):
        ang = np.power(np.maximum(cos_angle[valid], 0.0), angle_exp)
        dist_term = np.power(np.maximum(dists[valid], 1e-6), dist_pow)
        score[valid] = ang / np.maximum(dist_term, 1e-10)

    return valid, score, px_proj, py_proj


def _rank_chart_view_candidates(
    chart_candidates: list[list[tuple[float, float, int]]],
    n_views: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Rank candidate views per chart by coverage, then by score."""
    n_charts = len(chart_candidates)
    primary_views = np.full(n_charts, -1, dtype=np.int32)
    chart_view_orders: list[list[int]] = []

    for cid in range(n_charts):
        ranked = sorted(
            chart_candidates[cid],
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        ordered = [int(v) for _, _, v in ranked]
        if ordered:
            primary_views[cid] = ordered[0]
        if len(ordered) < n_views:
            seen = set(ordered)
            ordered.extend(vidx for vidx in range(n_views) if vidx not in seen)
        chart_view_orders.append(ordered)

    return primary_views, chart_view_orders


def _secondary_min_cos_levels(min_cos: float) -> list[float]:
    """Build progressively relaxed cosine thresholds for fallback sampling."""
    raw_levels = [
        min_cos * 0.5,
        min_cos * 0.25,
        min_cos - 0.1,
        0.0,
        -0.1,
        -0.25,
        -0.5,
    ]
    levels: list[float] = []
    for level in raw_levels:
        v = float(max(-1.0, min(1.0, level)))
        if v >= min_cos - 1e-8:
            continue
        if any(abs(v - existing) <= 1e-8 for existing in levels):
            continue
        levels.append(v)
    return levels


def bake_texture(
    mesh_ply: str,
    poses_path: str,
    frames_dir: str,
    mask_dir: str,
    output_dir: str,
    tex_size: int | None = None,
    pc_knn_k: int = 8,
    pc_max_distance: float = 0.0,
    pc_idw_power: float = 2.0,
    pc_normal_aware: bool = False,
    pc_normal_threshold_deg: float = 60.0,
    pc_adaptive_k: bool = False,
    pc_weighting: str = "idw",
    pc_gaussian_sigma: float = 1.0,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Full texture baking pipeline: point-cloud KNN colour + frame-projection fallback.

    Primary colouring uses KNN inverse-distance-weighted interpolation from the
    denoised point cloud.  Texels that remain uncovered (sparse regions) fall
    back to the legacy camera-projected view approach.

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
        pc_knn_k: Number of nearest neighbours for KNN colour interpolation.
        pc_max_distance: Max distance for KNN; 0 = auto-compute from point cloud.
        pc_idw_power: Exponent for inverse-distance weighting.
        pc_normal_aware: Enable normal-aware filtering to reduce cross-edge bleeding.
        pc_normal_threshold_deg: Max angle (degrees) between texel and neighbour normals.
        pc_adaptive_k: Dynamically reduce k in dense regions for sharper results.
        pc_weighting: ``"idw"`` or ``"gaussian"`` weight function.
        pc_gaussian_sigma: Sigma = nearest-neighbour distance * this factor.

    Environment toggles for quality:
        TEXTURE_DEVICE (str): 'cuda' (default), 'auto', or 'cpu' request hint.
        TEXTURE_OVERSAMPLE (int): Internal supersampling multiplier (1=off, 2=default).
        TEXTURE_MIN_COS (float): Minimum normal·view cosine to accept a sample (default 0.2).
        TEXTURE_ANGLE_EXP (float): Exponent on cosine score (default 2.0).
        TEXTURE_DIST_POW (float): Distance falloff power for score (default 1.0).
        TEXTURE_SHARPEN (float): Unsharp mask amount (0=off, 0.1–0.4 typical).
        TEXTURE_PC_KNN_K (int): Override pc_knn_k (default 8).
        TEXTURE_PC_MAX_DISTANCE (float): Override pc_max_distance (default 0.0).
        TEXTURE_PC_IDW_POWER (float): Override pc_idw_power (default 2.0).
        TEXTURE_PC_NORMAL_AWARE (bool): Override pc_normal_aware (default false).
        TEXTURE_PC_NORMAL_THRESHOLD_DEG (float): Override pc_normal_threshold_deg (default 60.0).
        TEXTURE_PC_ADAPTIVE_K (bool): Override pc_adaptive_k (default false).
        TEXTURE_PC_WEIGHTING (str): Override pc_weighting ('idw' or 'gaussian').
        TEXTURE_PC_GAUSSIAN_SIGMA (float): Override pc_gaussian_sigma (default 1.0).

    Returns:
        Path to the output OBJ file.
    """
    import xatlas

    oversample = max(1, int(os.environ.get("TEXTURE_OVERSAMPLE", "2")))
    min_cos = float(os.environ.get("TEXTURE_MIN_COS", "0.2"))
    angle_exp = float(os.environ.get("TEXTURE_ANGLE_EXP", "2.0"))
    dist_pow = float(os.environ.get("TEXTURE_DIST_POW", "1.0"))
    sharpen_amt = float(os.environ.get("TEXTURE_SHARPEN", "0.15"))
    texture_device = _resolve_texture_device()
    pc_knn_k = max(1, int(os.environ.get("TEXTURE_PC_KNN_K", str(pc_knn_k))))
    pc_max_distance = max(0.0, float(os.environ.get("TEXTURE_PC_MAX_DISTANCE", str(pc_max_distance))))
    pc_idw_power = max(0.1, float(os.environ.get("TEXTURE_PC_IDW_POWER", str(pc_idw_power))))
    _env_na = os.environ.get("TEXTURE_PC_NORMAL_AWARE", "")
    if _env_na:
        pc_normal_aware = _env_na.strip().lower() in {"1", "true", "yes"}
    pc_normal_threshold_deg = max(
        0.0,
        float(os.environ.get("TEXTURE_PC_NORMAL_THRESHOLD_DEG", str(pc_normal_threshold_deg))),
    )
    _env_ak = os.environ.get("TEXTURE_PC_ADAPTIVE_K", "")
    if _env_ak:
        pc_adaptive_k = _env_ak.strip().lower() in {"1", "true", "yes"}
    _env_wt = os.environ.get("TEXTURE_PC_WEIGHTING", "").strip().lower()
    if _env_wt in {"idw", "gaussian"}:
        pc_weighting = _env_wt
    pc_gaussian_sigma = max(
        0.01,
        float(os.environ.get("TEXTURE_PC_GAUSSIAN_SIGMA", str(pc_gaussian_sigma))),
    )

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

    # --- Load camera data ---
    poses, pose_frame_indices = _load_poses(poses_path)

    if len(poses) == 0:
        raise RuntimeError("No poses found in camera_poses.json")

    # Determine image size from first pose-linked frame.
    first_frame = _load_frame(frames_dir, int(pose_frame_indices[0]))
    img_h, img_w = first_frame.shape[:2]
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

    # --- Load point cloud (used for KNN colour and intrinsics estimation) ---
    denoised_ply = output_path / "object_denoised.ply"
    if denoised_ply.exists():
        pc_points, pc_colors = _load_point_cloud(str(denoised_ply))
    else:
        object_ply = output_path / "object.ply"
        pc_points, pc_colors = _load_point_cloud(str(object_ply))
    print(f"  Point cloud: {len(pc_points)} points")
    _emit_progress(progress_cb, 15.0, "Point cloud loaded")

    # --- UV Atlas ---
    print("Generating UV atlas...")
    _emit_progress(progress_cb, 58.0, "Generating UV atlas")
    vmapping, new_faces, uvs = xatlas.parametrize(
        vertices.astype(np.float32), faces.astype(np.uint32)
    )
    new_vertices = vertices[vmapping]
    print(f"  {len(new_vertices)} verts, {len(new_faces)} faces")

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
        "  View scoring: chart-best, min_cos=%.2f, angle_exp=%.2f, dist_pow=%.2f, sharpen=%.2f"
        % (min_cos, angle_exp, dist_pow, sharpen_amt)
    )
    if texture_device == "cuda":
        print("  Projection backend request: CUDA (chart scoring currently uses CPU z-buffer)")
    else:
        print("  Projection backend request: CPU")
    _emit_progress(progress_cb, 62.0, "Building texel mapping")
    face_id_buf = np.full((tex_res, tex_res), -1, dtype=np.int32)
    uv_scaled = uvs * tex_res

    emit_every_face = max(1, len(new_faces) // 40)
    for fi in range(len(new_faces)):
        i0, i1, i2 = new_faces[fi]
        pts = np.array([
            [uv_scaled[i0, 0], uv_scaled[i0, 1]],
            [uv_scaled[i1, 0], uv_scaled[i1, 1]],
            [uv_scaled[i2, 0], uv_scaled[i2, 1]],
        ], dtype=np.int32).reshape(3, 1, 2)
        cv2.fillConvexPoly(face_id_buf, pts, int(fi))
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

    # Barycentric coords
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

    # --- Point-cloud KNN colour interpolation (primary) ---
    print(
        f"KNN colour interpolation: k={pc_knn_k}, max_dist={pc_max_distance}, "
        f"idw_power={pc_idw_power}, weighting={pc_weighting}, "
        f"normal_aware={pc_normal_aware}, adaptive_k={pc_adaptive_k}"
    )

    pc_normals_arr = None
    if pc_normal_aware:
        print("  Estimating point cloud normals for normal-aware filtering...")
        _emit_progress(progress_cb, 59.0, "Estimating point cloud normals")
        pc_normals_arr = _estimate_pc_normals(pc_points)
        print(f"  Normal threshold: {pc_normal_threshold_deg:.1f} deg")

    _emit_progress(progress_cb, 60.0, "KNN colour interpolation")
    pc_colors_interp, pc_covered = _kdtree_color_interpolation(
        pos3d, pc_points, pc_colors,
        k=pc_knn_k,
        max_distance=pc_max_distance,
        idw_power=pc_idw_power,
        progress_cb=progress_cb,
        texel_normals=normals if pc_normal_aware else None,
        pc_normals=pc_normals_arr,
        normal_threshold_deg=pc_normal_threshold_deg if pc_normal_aware else 0.0,
        adaptive_k=pc_adaptive_k,
        weighting=pc_weighting,
        gaussian_sigma_factor=pc_gaussian_sigma,
    )

    texture = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    has_color = np.zeros(n_valid, dtype=bool)
    texture[ys[pc_covered], xs[pc_covered]] = pc_colors_interp[pc_covered]
    has_color[pc_covered] = True

    pc_cov = int(pc_covered.sum())
    print(
        f"  KNN coverage: {pc_cov}/{n_valid} "
        f"({100 * pc_cov / max(n_valid, 1):.1f}%)"
    )

    # --- Frame-projection fallback for uncovered texels ---
    if not np.all(has_color):
        print("Falling back to frame projection for uncovered texels...")

        # Estimate intrinsics (needed for projection)
        print("Estimating camera intrinsics...")
        _emit_progress(progress_cb, 82.0, "Estimating camera intrinsics")
        _intrinsics_progress_last = {"value": 82.0}

        def _intrinsics_progress(local_progress: float, detail: str | None = None) -> None:
            stage_progress = 82.0 + (max(0.0, min(100.0, local_progress)) * 0.02)
            if stage_progress - _intrinsics_progress_last["value"] >= 0.5:
                _emit_progress(progress_cb, stage_progress, detail)
                _intrinsics_progress_last["value"] = stage_progress

        intrinsics = _estimate_intrinsics(
            pc_points, pc_colors, poses, pose_frame_indices, frames_dir, mask_dir, img_w, img_h,
            progress_cb=_intrinsics_progress,
        )
        K = np.array(intrinsics["K"], dtype=np.float64)
        print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

        # Save intrinsics
        with open(output_path / "intrinsics.json", "w") as f_out:
            json.dump(intrinsics, f_out, indent=2)

        # UV chart grouping
        face_chart_ids, n_charts = _build_face_charts(new_faces)
        texel_chart_ids = face_chart_ids[fids]
        chart_texel_indices, chart_texel_counts = _group_texels_by_chart(texel_chart_ids, n_charts)
        print(f"  UV charts: {n_charts}")

        # Pass 1: score view candidates per chart
        _emit_progress(progress_cb, 84.0, "Scoring camera views per UV chart")
        emit_every_view = max(1, len(poses) // 40)
        chart_candidates: list[list[tuple[float, float, int]]] = [[] for _ in range(n_charts)]

        for vidx in range(len(poses)):
            src_idx = int(pose_frame_indices[vidx])
            c2w = poses[vidx]
            try:
                mask_bool = _load_mask(mask_dir, src_idx)
            except FileNotFoundError:
                continue

            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
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
            )
            valid_idx = np.where(valid)[0]
            if valid_idx.size > 0:
                chart_hits = texel_chart_ids[valid_idx]
                counts = np.bincount(chart_hits, minlength=n_charts)
                score_sums = np.bincount(chart_hits, weights=score[valid_idx], minlength=n_charts)
                active = np.where(counts > 0)[0]
                for cid in active:
                    total_chart = max(int(chart_texel_counts[cid]), 1)
                    coverage = float(counts[cid]) / float(total_chart)
                    mean_score = float(score_sums[cid] / counts[cid])
                    chart_candidates[int(cid)].append((coverage, mean_score, vidx))
                n_chart_hits = int(len(active))
            else:
                n_chart_hits = 0

            if vidx % 5 == 0 or vidx == len(poses) - 1:
                print(
                    f"  Candidate scoring {vidx + 1}/{len(poses)}: "
                    f"texels={int(valid_idx.size)}, charts={n_chart_hits}"
                )
            if vidx % emit_every_view == 0 or vidx == len(poses) - 1:
                ratio = (vidx + 1) / max(len(poses), 1)
                _emit_progress(
                    progress_cb,
                    84.0 + ratio * 4.0,
                    f"Scoring views ({vidx + 1}/{len(poses)} views)",
                )

        primary_chart_views, chart_view_orders = _rank_chart_view_candidates(chart_candidates, len(poses))
        n_primary = int(np.count_nonzero(primary_chart_views >= 0))
        print(f"Primary views assigned: {n_primary}/{n_charts} charts")

        # Pass 2: primary chart assignment (skip KNN-covered texels)
        _emit_progress(progress_cb, 88.0, "Projecting primary chart textures")
        charts_by_view: dict[int, list[int]] = defaultdict(list)
        for cid, vidx in enumerate(primary_chart_views):
            if int(chart_texel_counts[cid]) <= 0:
                continue
            if vidx >= 0:
                charts_by_view[int(vidx)].append(cid)

        primary_items = sorted(charts_by_view.items())
        for order_idx, (vidx, chart_ids_for_view) in enumerate(primary_items):
            src_idx = int(pose_frame_indices[vidx])
            c2w = poses[vidx]
            try:
                frame = _load_frame(frames_dir, src_idx)
                mask_bool = _load_mask(mask_dir, src_idx)
            except FileNotFoundError:
                continue

            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
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
            )

            filled_view = 0
            for cid in chart_ids_for_view:
                texel_idx = chart_texel_indices[cid]
                if texel_idx.size == 0:
                    continue
                pending = texel_idx[~has_color[texel_idx]]
                if pending.size == 0:
                    continue
                ok = valid[pending]
                if not np.any(ok):
                    continue
                fill_idx = pending[ok]
                colors = _bilinear_sample(frame, px_proj[fill_idx], py_proj[fill_idx]).astype(np.float32)
                texture[ys[fill_idx], xs[fill_idx]] = colors
                has_color[fill_idx] = True
                filled_view += int(fill_idx.size)

            print(
                f"  Primary projection view {vidx + 1}/{len(poses)}: "
                f"charts={len(chart_ids_for_view)}, filled={filled_view}"
            )
            ratio = (order_idx + 1) / max(len(primary_items), 1)
            _emit_progress(
                progress_cb,
                88.0 + ratio * 2.0,
                f"Applying primary views ({order_idx + 1}/{len(primary_items)})",
            )

        # Pass 3: secondary fallback with relaxed thresholds
        secondary_levels = _secondary_min_cos_levels(min_cos)
        chart_cursors = np.zeros(n_charts, dtype=np.int32)
        for cid in range(n_charts):
            primary_vidx = int(primary_chart_views[cid])
            if primary_vidx < 0:
                chart_cursors[cid] = 0
                continue
            order = chart_view_orders[cid]
            try:
                chart_cursors[cid] = int(order.index(primary_vidx)) + 1
            except ValueError:
                chart_cursors[cid] = 0

        if secondary_levels:
            _emit_progress(progress_cb, 90.0, "Secondary view search for uncovered texels")

        for level_idx, relaxed_min_cos in enumerate(secondary_levels):
            if not np.any(~has_color):
                break
            level_filled = 0

            for _round in range(2):
                missing_texels = np.where(~has_color)[0]
                if missing_texels.size == 0:
                    break
                missing_charts = np.unique(texel_chart_ids[missing_texels])
                charts_for_view: dict[int, list[int]] = defaultdict(list)

                for cid in missing_charts.tolist():
                    cursor = int(chart_cursors[cid])
                    order = chart_view_orders[cid]
                    if cursor >= len(order):
                        continue
                    vidx = int(order[cursor])
                    chart_cursors[cid] = cursor + 1
                    charts_for_view[vidx].append(cid)

                if not charts_for_view:
                    break

                for vidx, chart_ids_for_view in sorted(charts_for_view.items()):
                    src_idx = int(pose_frame_indices[vidx])
                    c2w = poses[vidx]
                    try:
                        frame = _load_frame(frames_dir, src_idx)
                        mask_bool = _load_mask(mask_dir, src_idx)
                    except FileNotFoundError:
                        continue

                    depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
                    valid, _score, px_proj, py_proj = _evaluate_view_samples(
                        pos3d=pos3d,
                        normals=normals,
                        c2w=c2w,
                        K=K,
                        img_w=img_w,
                        img_h=img_h,
                        mask_bool=mask_bool,
                        depth_buffer=depth_buffer,
                        min_cos=relaxed_min_cos,
                        angle_exp=angle_exp,
                        dist_pow=dist_pow,
                    )

                    for cid in chart_ids_for_view:
                        texel_idx = chart_texel_indices[cid]
                        if texel_idx.size == 0:
                            continue
                        pending = texel_idx[~has_color[texel_idx]]
                        if pending.size == 0:
                            continue
                        ok = valid[pending]
                        if not np.any(ok):
                            continue
                        fill_idx = pending[ok]
                        colors = _bilinear_sample(frame, px_proj[fill_idx], py_proj[fill_idx]).astype(
                            np.float32
                        )
                        texture[ys[fill_idx], xs[fill_idx]] = colors
                        has_color[fill_idx] = True
                        level_filled += int(fill_idx.size)

            remaining = int((~has_color).sum())
            print(
                f"  Secondary pass {level_idx + 1}/{len(secondary_levels)}: "
                f"min_cos={relaxed_min_cos:.3f}, filled={level_filled}, remaining={remaining}"
            )
            _emit_progress(
                progress_cb,
                90.0 + ((level_idx + 1) / max(len(secondary_levels), 1)) * 2.0,
                f"Secondary view search ({level_idx + 1}/{len(secondary_levels)})",
            )
    else:
        print("Full coverage from point cloud -- skipping frame projection")

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

    # OBJ
    with open(obj_path, "w") as f_out:
        f_out.write(f"mtllib {basename}.mtl\nusemtl material_0\n\n")
        for vert in new_vertices:
            f_out.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
        for uv in uvs:
            f_out.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in new_faces:
            a, b, c = face + 1
            f_out.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    print(f"Saved: {obj_path}")
    _emit_progress(progress_cb, 100.0, "Texture stage complete")

    return obj_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Texture baking")
    parser.add_argument("mesh_ply", help="Mesh PLY file")
    parser.add_argument("--poses", required=True, help="camera_poses.json")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", default="/data/output")
    parser.add_argument("--pc-knn-k", type=int, default=8, help="KNN neighbours (default 8)")
    parser.add_argument("--pc-max-distance", type=float, default=0.0, help="KNN max distance, 0=auto")
    parser.add_argument("--pc-idw-power", type=float, default=2.0, help="IDW exponent (default 2.0)")
    args = parser.parse_args()

    bake_texture(
        args.mesh_ply, args.poses, args.frames_dir, args.masks_dir, args.output_dir,
        pc_knn_k=args.pc_knn_k,
        pc_max_distance=args.pc_max_distance,
        pc_idw_power=args.pc_idw_power,
    )
