"""Stage 7: Texture baking from camera-projected views.

Estimates camera intrinsics, generates UV atlas, selects best camera views
per UV chart, and exports OBJ + MTL + PNG.

Adapted from im2pc/host/extract_intrinsics.py and im2pc/host/texture_mesh.py.
"""

import concurrent.futures
import json
import math
import os
import threading
from collections import OrderedDict, defaultdict
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


def _get_available_memory_mb() -> float:
    """Read available memory from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except (OSError, ValueError, IndexError):
        pass
    return 4096.0  # conservative fallback


class _FrameCache:
    """LRU cache for decoded frames and masks with memory-aware capacity."""

    def __init__(self, img_w: int, img_h: int) -> None:
        avail_mb = _get_available_memory_mb()
        safety_mb = 1024.0
        budget_mb = max(0.0, avail_mb - safety_mb)

        frame_bytes = img_w * img_h * 3 * 8  # float64 RGB
        mask_bytes = img_w * img_h  # bool

        frame_budget_mb = budget_mb * 0.7
        mask_budget_mb = budget_mb * 0.3

        self._max_frames = max(1, int(frame_budget_mb * 1024 * 1024 / max(frame_bytes, 1)))
        self._max_masks = max(1, int(mask_budget_mb * 1024 * 1024 / max(mask_bytes, 1)))

        self._frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._masks: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._frame_lock = threading.Lock()
        self._mask_lock = threading.Lock()

    def load_frame(self, frames_dir: str, idx: int) -> np.ndarray:
        key = (frames_dir, idx)
        with self._frame_lock:
            if key in self._frames:
                self._frames.move_to_end(key)
                return self._frames[key]
        arr = _load_frame(frames_dir, idx)
        with self._frame_lock:
            if key not in self._frames:
                self._frames[key] = arr
                while len(self._frames) > self._max_frames:
                    self._frames.popitem(last=False)
            else:
                self._frames.move_to_end(key)
                arr = self._frames[key]
        return arr

    def load_mask(self, masks_dir: str, idx: int) -> np.ndarray:
        key = (masks_dir, idx)
        with self._mask_lock:
            if key in self._masks:
                self._masks.move_to_end(key)
                return self._masks[key]
        arr = _load_mask(masks_dir, idx)
        with self._mask_lock:
            if key not in self._masks:
                self._masks[key] = arr
                while len(self._masks) > self._max_masks:
                    self._masks.popitem(last=False)
            else:
                self._masks.move_to_end(key)
                arr = self._masks[key]
        return arr


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
    """Rasterize mesh depth into camera image space for a simple z-test.

    Uses vectorized preprocessing to batch-filter faces before the per-face
    rasterization loop (Optimization B).
    """
    uv_all, depth_all = _project_simple(vertices, c2w, K)
    depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)
    inside_eps = 1e-5

    n_faces = len(faces)
    if n_faces == 0:
        return depth_buffer

    # --- Vectorized pre-processing ---
    i0s = faces[:, 0]
    i1s = faces[:, 1]
    i2s = faces[:, 2]

    z0s = depth_all[i0s]
    z1s = depth_all[i1s]
    z2s = depth_all[i2s]

    # Back-face culling: skip faces where all three depths <= 0.01
    visible = ~((z0s <= 0.01) & (z1s <= 0.01) & (z2s <= 0.01))

    x0s = uv_all[i0s, 0]
    y0s = uv_all[i0s, 1]
    x1s = uv_all[i1s, 0]
    y1s = uv_all[i1s, 1]
    x2s = uv_all[i2s, 0]
    y2s = uv_all[i2s, 1]

    # Bounding boxes (vectorized)
    bb_min_x = np.floor(np.minimum(np.minimum(x0s, x1s), x2s)).astype(np.int32)
    bb_max_x = np.ceil(np.maximum(np.maximum(x0s, x1s), x2s)).astype(np.int32)
    bb_min_y = np.floor(np.minimum(np.minimum(y0s, y1s), y2s)).astype(np.int32)
    bb_max_y = np.ceil(np.maximum(np.maximum(y0s, y1s), y2s)).astype(np.int32)

    # Check overlap with image BEFORE clipping (matches original asymmetric clip logic)
    bb_valid = (bb_max_x >= 0) & (bb_min_x <= img_w - 1) & (bb_max_y >= 0) & (bb_min_y <= img_h - 1)

    np.clip(bb_min_x, 0, img_w - 1, out=bb_min_x)
    np.clip(bb_max_x, 0, img_w - 1, out=bb_max_x)
    np.clip(bb_min_y, 0, img_h - 1, out=bb_min_y)
    np.clip(bb_max_y, 0, img_h - 1, out=bb_max_y)

    # Barycentric denominator (vectorized)
    denoms = (y1s - y2s) * (x0s - x2s) + (x2s - x1s) * (y0s - y2s)
    non_degenerate = np.abs(denoms) >= 1e-12

    # Combined filter
    keep = visible & bb_valid & non_degenerate
    keep_idx = np.where(keep)[0]

    # Build contiguous arrays for the inner loop
    _x0 = x0s[keep_idx]
    _y0 = y0s[keep_idx]
    _x1 = x1s[keep_idx]
    _y1 = y1s[keep_idx]
    _x2 = x2s[keep_idx]
    _y2 = y2s[keep_idx]
    _z0 = z0s[keep_idx]
    _z1 = z1s[keep_idx]
    _z2 = z2s[keep_idx]
    _denom = denoms[keep_idx]
    _bb_min_x = bb_min_x[keep_idx]
    _bb_max_x = bb_max_x[keep_idx]
    _bb_min_y = bb_min_y[keep_idx]
    _bb_max_y = bb_max_y[keep_idx]

    for i in range(len(keep_idx)):
        mn_x = int(_bb_min_x[i])
        mx_x = int(_bb_max_x[i])
        mn_y = int(_bb_min_y[i])
        mx_y = int(_bb_max_y[i])
        d = _denom[i]

        xs = np.arange(mn_x, mx_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(mn_y, mx_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        gx2 = grid_x - _x2[i]
        gy2 = grid_y - _y2[i]
        w0 = ((_y1[i] - _y2[i]) * gx2 + (_x2[i] - _x1[i]) * gy2) / d
        w1 = ((_y2[i] - _y0[i]) * gx2 + (_x0[i] - _x2[i]) * gy2) / d
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -inside_eps) & (w1 >= -inside_eps) & (w2 >= -inside_eps)
        if not np.any(inside):
            continue

        depth = w0 * _z0[i] + w1 * _z1[i] + w2 * _z2[i]
        valid = inside & (depth > 0.01)
        if not np.any(valid):
            continue

        tile = depth_buffer[mn_y:mx_y + 1, mn_x:mx_x + 1]
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
        TEXTURE_DEVICE (str): 'cuda' (default), 'auto', or 'cpu' request hint.
            Current chart-selection path runs on CPU for deterministic z-buffering.
        TEXTURE_OVERSAMPLE (int): Internal supersampling multiplier (1=off, 2=default).
        TEXTURE_MIN_COS (float): Minimum normal·view cosine to accept a sample (default 0.2).
        TEXTURE_ANGLE_EXP (float): Exponent on cosine score (default 2.0).
        TEXTURE_DIST_POW (float): Distance falloff power for score (default 1.0).
        TEXTURE_SHARPEN (float): Unsharp mask amount (0=off, 0.1–0.4 typical).

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
    # Load denoised PLY for intrinsics estimation (original colored point cloud)
    denoised_ply = output_path / "object_denoised.ply"
    if denoised_ply.exists():
        pc_points, pc_colors = _load_point_cloud(str(denoised_ply))
    else:
        # Fallback: use object.ply
        object_ply = output_path / "object.ply"
        pc_points, pc_colors = _load_point_cloud(str(object_ply))

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

    # --- UV chart grouping ---
    face_chart_ids, n_charts = _build_face_charts(new_faces)
    texel_chart_ids = face_chart_ids[fids]
    chart_texel_indices, chart_texel_counts = _group_texels_by_chart(texel_chart_ids, n_charts)
    print(f"  UV charts: {n_charts}")

    # --- Pass 1: score view candidates per chart ---
    _emit_progress(progress_cb, 72.0, "Scoring camera views per UV chart")
    emit_every_view = max(1, len(poses) // 40)
    chart_candidates: list[list[tuple[float, float, int]]] = [[] for _ in range(n_charts)]

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

        depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
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
                72.0 + ratio * 12.0,
                f"Scoring views ({vidx + 1}/{len(poses)} views)",
            )

    primary_chart_views, chart_view_orders = _rank_chart_view_candidates(chart_candidates, len(poses))
    n_primary = int(np.count_nonzero(primary_chart_views >= 0))
    print(f"Primary views assigned: {n_primary}/{n_charts} charts")

    texture = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    has_color = np.zeros(n_valid, dtype=bool)

    # --- Pass 2: primary chart assignment ---
    _emit_progress(progress_cb, 84.0, "Projecting primary chart textures")
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
            frame = frame_cache.load_frame(frames_dir, src_idx)
            mask_bool = frame_cache.load_mask(mask_dir, src_idx)
        except FileNotFoundError:
            continue

        if vidx in depth_cache:
            depth_buffer = depth_cache[vidx]
        else:
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
            ok = valid[texel_idx]
            if not np.any(ok):
                continue
            fill_idx = texel_idx[ok]
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
            84.0 + ratio * 6.0,
            f"Applying primary views ({order_idx + 1}/{len(primary_items)})",
        )

    # --- Pass 3: secondary fallback with relaxed thresholds ---
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

        # Try two next-best rounds per relaxation level.
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
                    frame = frame_cache.load_frame(frames_dir, src_idx)
                    mask_bool = frame_cache.load_mask(mask_dir, src_idx)
                except FileNotFoundError:
                    continue

                if vidx in depth_cache:
                    depth_buffer = depth_cache[vidx]
                else:
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

    # Release caches to free memory before seam padding
    depth_cache.clear()
    del frame_cache

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Texture baking")
    parser.add_argument("mesh_ply", help="Mesh PLY file")
    parser.add_argument("--poses", required=True, help="camera_poses.json")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    bake_texture(args.mesh_ply, args.poses, args.frames_dir, args.masks_dir, args.output_dir)
