"""Camera intrinsics estimation for texture baking."""

import concurrent.futures
import os

import numpy as np

from scripts.texture.progress import ProgressCallback, _emit_progress
from scripts.texture.io_utils import _FrameCache, _load_frame, _load_mask


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


def _project_simple(pts, c2w, K):
    """Simple 3D→2D projection. Returns (uv (N,2), depths (N,))."""
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    d = cam[:, 2].copy()
    sz = np.maximum(d, 1e-10)
    u = K[0, 0] * cam[:, 0] / sz + K[0, 2]
    v = K[1, 1] * cam[:, 1] / sz + K[1, 2]
    return np.column_stack([u, v]), d


def _estimate_intrinsics(
    points, colors, poses, pose_frame_indices, frames_dir, masks_dir, img_w, img_h,
    num_eval_frames=10, subsample_points=50000,
    progress_cb: ProgressCallback | None = None,
    cache: '_FrameCache | None' = None,
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

    # Thread pool for parallel frame evaluation (Optimization D)
    n_workers = min(len(eval_pose_indices), max(1, os.cpu_count() or 4))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)

    def color_score(K):
        def _eval_frame(pose_idx):
            src_idx = int(pose_frame_indices[pose_idx])
            try:
                if cache is not None:
                    frame = cache.load_frame(frames_dir, src_idx)
                    mask = cache.load_mask(masks_dir, src_idx)
                else:
                    frame = _load_frame(frames_dir, src_idx)
                    mask = _load_mask(masks_dir, src_idx)
            except FileNotFoundError:
                return 0.0, 0
            uv, valid, _ = _project_points(pts_sub, poses[pose_idx], K, img_w, img_h)
            if uv.shape[0] == 0:
                return 0.0, 0
            ui, vi = uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)
            m = mask[vi, ui]
            if m.sum() == 0:
                return 0.0, 0
            diff = frame[vi[m], ui[m]] - col_sub[valid][m]
            return float(np.mean(diff**2) * m.sum()), int(m.sum())

        results = list(pool.map(_eval_frame, eval_pose_indices))
        total_err = sum(r[0] for r in results)
        total_cnt = sum(r[1] for r in results)
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
    pool.shutdown(wait=False)
    _emit_progress(progress_cb, 100.0, "Intrinsics optimization complete")

    fx, fy, cx, cy = result.x
    print(f"  Optimized: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    return {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "image_width": img_w, "image_height": img_h,
        "K": _make_K(fx, fy, cx, cy).tolist(),
    }
