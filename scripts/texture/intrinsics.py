"""Camera intrinsics estimation for texture baking."""

import concurrent.futures
import os

import numpy as np

from scripts.texture.progress import ProgressCallback, _emit_progress
from scripts.texture.io_utils import _FrameCache, _load_frame, _load_mask


def _make_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _normalize_dist_coeffs(dist_coeffs) -> np.ndarray | None:
    """Return a length-5 OpenCV layout [k1, k2, p1, p2, k3] or None.

    None / zero coefficients short-circuit to None so callers can skip the
    distortion path entirely when SIMPLE_RADIAL k1 is exactly 0 (synthetic
    data) or when intrinsics.json is from a pre-distortion pipeline.
    """
    if dist_coeffs is None:
        return None
    arr = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    if arr.size < 5:
        padded = np.zeros(5, dtype=np.float64)
        padded[: arr.size] = arr
        arr = padded
    if not np.any(np.abs(arr[:5]) > 1e-12):
        return None
    return arr[:5]


def _apply_brown_distortion(
    x: np.ndarray, y: np.ndarray, dist_coeffs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Brown-Conrady distortion to normalized image coordinates.

    Matches OpenCV / COLMAP convention with [k1, k2, p1, p2, k3]. SIMPLE_RADIAL
    fills (k1, 0, 0, 0, 0); RADIAL fills (k1, k2, 0, 0, 0); OPENCV fills
    (k1, k2, p1, p2, 0); FULL_OPENCV fills all five. Bundle adjustment in
    Stage 2 solved world geometry under this exact model, so projecting
    through the same model — and sampling the original frame directly —
    avoids the bilinear remap that ``cv2.undistort`` would otherwise do.
    The remap is what creates the cylindrical-surface speckle (per-texel
    sub-pixel error in the high-radius regions of every view).
    """
    k1, k2, p1, p2, k3 = (
        float(dist_coeffs[0]),
        float(dist_coeffs[1]),
        float(dist_coeffs[2]),
        float(dist_coeffs[3]),
        float(dist_coeffs[4]),
    )
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def _project_points(pts, c2w, K, img_w, img_h, dist_coeffs=None):
    """Project 3D→2D. Returns (uv (M,2), valid (N,), depths (N,)).

    When ``dist_coeffs`` is provided the projection applies the full COLMAP
    camera model (pinhole + Brown-Conrady) so callers can sample the original
    distorted frame instead of pre-undistorting it.
    """
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    depths = cam[:, 2]
    valid = depths > 0.01
    pts_v = cam[valid]
    sz = np.maximum(pts_v[:, 2], 1e-10)
    x_n = pts_v[:, 0] / sz
    y_n = pts_v[:, 1] / sz
    coeffs = _normalize_dist_coeffs(dist_coeffs)
    if coeffs is not None:
        x_n, y_n = _apply_brown_distortion(x_n, y_n, coeffs)
    u = K[0, 0] * x_n + K[0, 2]
    v = K[1, 1] * y_n + K[1, 2]
    in_bounds = (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)
    uv = np.column_stack([u[in_bounds], v[in_bounds]])
    valid_idx = np.where(valid)[0]
    full_valid = np.zeros(len(pts), dtype=bool)
    full_valid[valid_idx[in_bounds]] = True
    return uv, full_valid, depths


def _project_simple(pts, c2w, K, dist_coeffs=None):
    """Simple 3D→2D projection. Returns (uv (N,2), depths (N,)).

    Pass ``dist_coeffs`` (OpenCV layout) to apply the full COLMAP camera
    model. Without it the projection is pinhole and callers must keep their
    frames undistorted to remain coherent.
    """
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    d = cam[:, 2].copy()
    sz = np.maximum(d, 1e-10)
    x_n = cam[:, 0] / sz
    y_n = cam[:, 1] / sz
    coeffs = _normalize_dist_coeffs(dist_coeffs)
    if coeffs is not None:
        x_n, y_n = _apply_brown_distortion(x_n, y_n, coeffs)
    u = K[0, 0] * x_n + K[0, 2]
    v = K[1, 1] * y_n + K[1, 2]
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
    # Range covers wide-angle (~80°) GoPro-style shots down to telephoto /
    # close-up cellphone (~25°). Cup_Noodle's true FOV (~33.5°) sat just
    # below the previous 35° lower bound, forcing Nelder-Mead to start at
    # a boundary and land in a sub-pixel-skewed local minimum that
    # destroyed fine texture detail.
    best_score, best_fov = -np.inf, 60
    print("  Grid search over FOV...")
    fov_values = list(range(25, 91, 2))
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
        "source": "estimated",
    }
