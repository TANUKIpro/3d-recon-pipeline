"""Stage 7: Texture baking from camera-projected views.

Estimates camera intrinsics, generates UV atlas, selects best camera views
per UV chart, and exports OBJ + MTL + PNG.
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

from config_defaults import (
    TEXTURE_ANGLE_EXP,
    TEXTURE_BLEND_HARD_RATIO,
    TEXTURE_BLEND_TOPK,
    TEXTURE_DIST_POW,
    TEXTURE_MIN_COS,
    TEXTURE_OVERSAMPLE,
    TEXTURE_SHARPEN,
    TEXTURE_VIEW_ASSIGN_MODE,
    _TEXTURE_CACHE_SAFETY_MB,
    _TEXTURE_FRAME_BUDGET_RATIO,
    _TEXTURE_MASK_BUDGET_RATIO,
    _TEXTURE_MEM_FALLBACK_MB,
)
from mesh_orientation import orient_faces_outward

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for GPU acceleration
    torch = None

ProgressCallback = Callable[[float, str | None], None]

_TEXTURE_CONFLICT_RATIO = 1.35
_TEXTURE_CONFLICT_VIEW_ANGLE_DEG = 20.0
_TEXTURE_CONFLICT_FACE_MIN_TEXELS = 4
_TEXTURE_CONFLICT_FACE_MIN_FRAC = 0.2
_TEXTURE_CONFLICT_FACE_MIN_COVERAGE = 0.7
_TEXTURE_CONFLICT_SMOOTH_DOT = 0.95
_TEXTURE_CONFLICT_SMOOTH_GAIN = 1.05
_TEXTURE_CONFLICT_SMOOTH_MIN_NEIGHBORS = 2
_TEXTURE_REGION_NORMAL_DOT = 0.92
_TEXTURE_REGION_MIN_FACES = 3
_TEXTURE_REGION_TOP_LABELS = 4
_TEXTURE_REGION_MAX_ITERS = 4
_TEXTURE_REGION_SMOOTHNESS = 0.55
_TEXTURE_REGION_MIN_LABEL_COVERAGE = 0.15
_TEXTURE_REGION_MISSING_COST = 1.5
_TEXTURE_REGION_LOW_COVERAGE_PENALTY = 0.35
_TEXTURE_SEAM_LEVEL_BLEND = 0.30
_TEXTURE_SEAM_LEVEL_DILATE = 2
_TEXTURE_SEAM_LEVEL_SIGMA = 0.8
_TEXTURE_SEAM_LOW_SIGMA = 0.75
_TEXTURE_SEAM_SMOOTH_SIGMA = 1.65
_TEXTURE_SEAM_HQ_SIGMA = 1.25
_TEXTURE_SEAM_HQ_DILATE = 3
_TEXTURE_SEAM_HQ_DETAIL_GAIN = 1.0
_TEXTURE_SEAM_HQ_LOW_BLEND = 0.35
_TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS = 9
_TEXTURE_SEAM_HQ_SHIFT_LIMIT = 1.5
_TEXTURE_SEAM_HQ_RESPONSE_FLOOR = 0.50
_TEXTURE_SEAM_HQ_MAX_COMPANIONS = 3
_TEXTURE_SEAM_HQ_PATCH_PAD = 3
_TEXTURE_SEAM_HQ_MIN_OVERLAP_RATIO = 0.35
_TEXTURE_SEAM_HQ_ECC_ITERS = 60
_TEXTURE_SEAM_HQ_ECC_EPS = 1e-5
_TEXTURE_SEAM_HQ_COLOR_EPS = 1e-4
_TEXTURE_SEAM_HQ_GRADIENT_WEIGHT = 0.35
_TEXTURE_SEAM_HQ_DETAIL_WEIGHT = 0.18


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
    return _TEXTURE_MEM_FALLBACK_MB  # conservative fallback


class _FrameCache:
    """LRU cache for decoded frames and masks with memory-aware capacity."""

    def __init__(self, img_w: int, img_h: int) -> None:
        avail_mb = _get_available_memory_mb()
        budget_mb = max(0.0, avail_mb - _TEXTURE_CACHE_SAFETY_MB)

        frame_bytes = img_w * img_h * 3 * 8  # float64 RGB
        mask_bytes = img_w * img_h  # bool

        frame_budget_mb = budget_mb * _TEXTURE_FRAME_BUDGET_RATIO
        mask_budget_mb = budget_mb * _TEXTURE_MASK_BUDGET_RATIO

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


def _resolve_texture_view_assign_mode(requested: str | None = None) -> str:
    mode = (requested or os.environ.get("TEXTURE_VIEW_ASSIGN_MODE", TEXTURE_VIEW_ASSIGN_MODE)).strip().lower()
    if mode == "region":
        mode = "region_gc"
    if mode not in {"legacy", "region_gc"}:
        print(f"Warning: invalid TEXTURE_VIEW_ASSIGN_MODE='{mode}', falling back to {TEXTURE_VIEW_ASSIGN_MODE}")
        return TEXTURE_VIEW_ASSIGN_MODE
    return mode


def _compute_label_boundary(label_buffer: np.ndarray) -> np.ndarray:
    valid_labels = label_buffer >= 0
    boundary = np.zeros(label_buffer.shape, dtype=bool)
    diff_x = valid_labels[:, 1:] & valid_labels[:, :-1] & (label_buffer[:, 1:] != label_buffer[:, :-1])
    diff_y = valid_labels[1:, :] & valid_labels[:-1, :] & (label_buffer[1:, :] != label_buffer[:-1, :])
    boundary[:, 1:] |= diff_x
    boundary[:, :-1] |= diff_x
    boundary[1:, :] |= diff_y
    boundary[:-1, :] |= diff_y
    return boundary


def _masked_gaussian(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    if image.ndim == 3:
        num = cv2.GaussianBlur(image * mask_f[:, :, None], (0, 0), sigmaX=sigma)
        den = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma)
        return num / np.maximum(den[:, :, None], 1e-6)
    num = cv2.GaussianBlur(image * mask_f, (0, 0), sigmaX=sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma)
    return num / np.maximum(den, 1e-6)


def _estimate_detail_band_shift(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
    max_shift: float = _TEXTURE_SEAM_HQ_SHIFT_LIMIT,
) -> tuple[float, float, float]:
    if reference_patch.size == 0 or candidate_patch.size == 0 or int(valid_mask.sum()) < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
        return 0.0, 0.0, 0.0

    ref_gray = (
        0.299 * reference_patch[:, :, 0]
        + 0.587 * reference_patch[:, :, 1]
        + 0.114 * reference_patch[:, :, 2]
    ).astype(np.float32, copy=False)
    cand_gray = (
        0.299 * candidate_patch[:, :, 0]
        + 0.587 * candidate_patch[:, :, 1]
        + 0.114 * candidate_patch[:, :, 2]
    ).astype(np.float32, copy=False)
    mask_f = valid_mask.astype(np.float32, copy=False)

    ref_hp = ref_gray - _masked_gaussian(ref_gray, valid_mask, sigma=1.0)
    cand_hp = cand_gray - _masked_gaussian(cand_gray, valid_mask, sigma=1.0)
    ref_hp *= mask_f
    cand_hp *= mask_f
    if float(np.std(ref_hp)) < 1e-4 or float(np.std(cand_hp)) < 1e-4:
        return 0.0, 0.0, 0.0

    try:
        shift, response = cv2.phaseCorrelate(ref_hp, cand_hp)
    except cv2.error:
        return 0.0, 0.0, 0.0
    dx, dy = float(shift[0]), float(shift[1])
    if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(response):
        return 0.0, 0.0, 0.0
    return (
        float(np.clip(dx, -max_shift, max_shift)),
        float(np.clip(dy, -max_shift, max_shift)),
        float(max(0.0, min(1.0, response))),
    )


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.float32, copy=False)


def _translate_patch(patch: np.ndarray, dx: float, dy: float, is_mask: bool = False) -> np.ndarray:
    if patch.size == 0 or (abs(dx) < 1e-6 and abs(dy) < 1e-6):
        return patch.copy()
    h, w = patch.shape[:2]
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(
        patch,
        warp,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _masked_gray_difference(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return float("inf")
    ref_gray = (
        0.299 * reference_patch[:, :, 0]
        + 0.587 * reference_patch[:, :, 1]
        + 0.114 * reference_patch[:, :, 2]
    )
    cand_gray = (
        0.299 * candidate_patch[:, :, 0]
        + 0.587 * candidate_patch[:, :, 1]
        + 0.114 * candidate_patch[:, :, 2]
    )
    diff = np.abs(ref_gray[overlap] - cand_gray[overlap])
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff))


def _masked_rgb_difference(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return float("inf")
    diff = np.abs(reference_patch[overlap] - candidate_patch[overlap])
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff))


def _masked_gradient_difference(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return float("inf")
    ref_gray = _rgb_to_gray(reference_patch)
    cand_gray = _rgb_to_gray(candidate_patch)
    ref_dx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    ref_dy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    cand_dx = cv2.Sobel(cand_gray, cv2.CV_32F, 1, 0, ksize=3)
    cand_dy = cv2.Sobel(cand_gray, cv2.CV_32F, 0, 1, ksize=3)
    ref_mag = np.sqrt(ref_dx * ref_dx + ref_dy * ref_dy)
    cand_mag = np.sqrt(cand_dx * cand_dx + cand_dy * cand_dy)
    diff = np.abs(ref_mag[overlap] - cand_mag[overlap])
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff))


def _masked_detail_advantage(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return 0.0
    ref_gray = _rgb_to_gray(reference_patch)
    cand_gray = _rgb_to_gray(candidate_patch)
    ref_detail = ref_gray - _masked_gaussian(ref_gray, valid_mask, sigma=1.0)
    cand_detail = cand_gray - _masked_gaussian(cand_gray, valid_mask, sigma=1.0)
    ref_mag = np.abs(ref_detail[overlap])
    cand_mag = np.abs(cand_detail[overlap])
    if ref_mag.size == 0 or cand_mag.size == 0:
        return 0.0
    gain = np.maximum(cand_mag - ref_mag, 0.0)
    scale = max(float(np.mean(ref_mag)), 1e-4)
    return float(np.clip(np.mean(gain) / scale, 0.0, 1.5))


def _normalize_candidate_patch_colors(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = _TEXTURE_SEAM_HQ_COLOR_EPS,
) -> np.ndarray:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return candidate_patch.copy()

    normalized = candidate_patch.astype(np.float32, copy=True)
    ref_low = _masked_gaussian(reference_patch.astype(np.float32, copy=False), valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    cand_low = _masked_gaussian(candidate_patch.astype(np.float32, copy=False), valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    cand_detail = normalized - cand_low
    ref_vals = ref_low[overlap].astype(np.float32, copy=False)
    cand_vals = cand_low[overlap].astype(np.float32, copy=False)
    ref_mean = ref_vals.mean(axis=0)
    cand_mean = cand_vals.mean(axis=0)
    ref_std = ref_vals.std(axis=0)
    cand_std = cand_vals.std(axis=0)
    scale = np.where(cand_std > eps, ref_std / np.maximum(cand_std, eps), 1.0)
    normalized_low = (cand_low - cand_mean[None, None, :]) * scale[None, None, :] + ref_mean[None, None, :]
    normalized = normalized_low + cand_detail
    return np.clip(normalized, 0.0, 1.0)


def _refine_detail_band_shift(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
    init_dx: float = 0.0,
    init_dy: float = 0.0,
    max_shift: float = _TEXTURE_SEAM_HQ_SHIFT_LIMIT,
) -> tuple[float, float, float]:
    if reference_patch.size == 0 or candidate_patch.size == 0 or int(valid_mask.sum()) < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
        return 0.0, 0.0, 0.0

    ref_gray = _rgb_to_gray(reference_patch)
    cand_gray = _rgb_to_gray(candidate_patch)
    ref_hp = ref_gray - _masked_gaussian(ref_gray, valid_mask, sigma=1.0)
    cand_hp = cand_gray - _masked_gaussian(cand_gray, valid_mask, sigma=1.0)
    overlap = valid_mask.astype(bool, copy=False)
    ref_std = float(np.std(ref_hp[overlap])) if np.any(overlap) else 0.0
    cand_std = float(np.std(cand_hp[overlap])) if np.any(overlap) else 0.0
    if ref_std < 1e-4 or cand_std < 1e-4:
        return 0.0, 0.0, 0.0

    ref_hp = ref_hp.astype(np.float32, copy=False)
    cand_hp = cand_hp.astype(np.float32, copy=False)
    ref_hp[overlap] = (ref_hp[overlap] - float(ref_hp[overlap].mean())) / max(ref_std, 1e-4)
    cand_hp[overlap] = (cand_hp[overlap] - float(cand_hp[overlap].mean())) / max(cand_std, 1e-4)
    mask_u8 = (valid_mask.astype(np.uint8) * 255)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        _TEXTURE_SEAM_HQ_ECC_ITERS,
        _TEXTURE_SEAM_HQ_ECC_EPS,
    )

    seeds = [
        (float(init_dx), float(init_dy)),
        (-float(init_dx), -float(init_dy)),
        (0.0, 0.0),
    ]
    best_state: tuple[float, float, float, float, float, float] | None = None
    seen: set[tuple[int, int]] = set()
    for seed_dx, seed_dy in seeds:
        key = (int(round(seed_dx * 1000.0)), int(round(seed_dy * 1000.0)))
        if key in seen:
            continue
        seen.add(key)
        seed_shifted = _translate_patch(candidate_patch, seed_dx, seed_dy)
        seed_rgb_diff = _masked_rgb_difference(reference_patch, seed_shifted, valid_mask)
        seed_gray_diff = _masked_gray_difference(reference_patch, seed_shifted, valid_mask)
        seed_response = 0.5 if abs(seed_dx) > 1e-6 or abs(seed_dy) > 1e-6 else 0.0
        seed_state = (
            seed_rgb_diff,
            seed_gray_diff,
            -float(seed_response),
            float(seed_dx),
            float(seed_dy),
            float(seed_response),
        )
        if best_state is None or seed_state < best_state:
            best_state = seed_state
        warp = np.array([[1.0, 0.0, seed_dx], [0.0, 1.0, seed_dy]], dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(
                ref_hp,
                cand_hp,
                warp,
                cv2.MOTION_TRANSLATION,
                criteria,
                inputMask=mask_u8,
                gaussFiltSize=3,
            )
        except cv2.error:
            continue
        dx = float(np.clip(warp[0, 2], -max_shift, max_shift))
        dy = float(np.clip(warp[1, 2], -max_shift, max_shift))
        shifted = _translate_patch(candidate_patch, dx, dy)
        rgb_diff = _masked_rgb_difference(reference_patch, shifted, valid_mask)
        gray_diff = _masked_gray_difference(reference_patch, shifted, valid_mask)
        state = (rgb_diff, gray_diff, -float(cc), dx, dy, float(np.clip(cc, 0.0, 1.0)))
        if best_state is None or state < best_state:
            best_state = state

    if best_state is None:
        return 0.0, 0.0, 0.0
    rgb_diff, gray_diff, neg_cc, dx, dy, response = best_state
    del rgb_diff, gray_diff
    return dx, dy, float(np.clip(max(response, -neg_cc), 0.0, 1.0))


def _select_detail_companion_view_candidates(
    primary_views: np.ndarray,
    best_views: np.ndarray,
    best_scores: np.ndarray,
    max_candidates: int = _TEXTURE_SEAM_HQ_MAX_COMPANIONS,
) -> list[np.ndarray]:
    selections: list[np.ndarray] = []
    if max_candidates <= 0 or best_views.size == 0:
        return selections

    for _cand_idx in range(max_candidates):
        companion = np.full(primary_views.shape, -1, dtype=np.int32)
        for k in range(best_views.shape[1]):
            cand_views = best_views[:, k]
            cand_scores = best_scores[:, k]
            take = (
                (companion < 0)
                & (primary_views >= 0)
                & (cand_views >= 0)
                & (cand_scores > 0)
                & (cand_views != primary_views)
            )
            for prev in selections:
                take &= cand_views != prev
            companion[take] = cand_views[take].astype(np.int32, copy=False)
        if not np.any(companion >= 0):
            break
        selections.append(companion)
    return selections


def _select_detail_companion_views(
    primary_views: np.ndarray,
    best_views: np.ndarray,
    best_scores: np.ndarray,
) -> np.ndarray:
    candidates = _select_detail_companion_view_candidates(primary_views, best_views, best_scores, max_candidates=1)
    if candidates:
        return candidates[0]
    return np.full(primary_views.shape, -1, dtype=np.int32)


def _select_best_detail_candidate(
    reference_patch: np.ndarray,
    component_mask: np.ndarray,
    candidate_sources: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if reference_patch.size == 0 or not np.any(component_mask) or not candidate_sources:
        return None

    component_pixels = int(component_mask.sum())
    best_choice: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None

    for candidate_patch, candidate_valid_mask in candidate_sources:
        overlap_seed = component_mask & candidate_valid_mask
        if int(overlap_seed.sum()) < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
            continue

        normalized = _normalize_candidate_patch_colors(reference_patch, candidate_patch, overlap_seed)
        phase_dx, phase_dy, phase_response = _estimate_detail_band_shift(reference_patch, normalized, overlap_seed)
        ecc_dx, ecc_dy, ecc_response = _refine_detail_band_shift(
            reference_patch,
            normalized,
            overlap_seed,
            init_dx=phase_dx,
            init_dy=phase_dy,
        )

        proposals = [
            (0.0, 0.0, max(0.0, phase_response * 0.25)),
            (phase_dx, phase_dy, phase_response),
            (-phase_dx, -phase_dy, phase_response * 0.85),
            (ecc_dx, ecc_dy, max(phase_response, ecc_response)),
            (-ecc_dx, -ecc_dy, max(phase_response, ecc_response) * 0.85),
        ]
        seen_shifts: set[tuple[int, int]] = set()
        for shift_dx, shift_dy, response in proposals:
            shift_key = (int(round(shift_dx * 1000.0)), int(round(shift_dy * 1000.0)))
            if shift_key in seen_shifts:
                continue
            seen_shifts.add(shift_key)

            shifted_patch = _translate_patch(normalized, shift_dx, shift_dy)
            shifted_mask = _translate_patch(candidate_valid_mask.astype(np.uint8), shift_dx, shift_dy, is_mask=True) > 0
            merge_mask = component_mask & shifted_mask
            overlap_pixels = int(merge_mask.sum())
            if overlap_pixels < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
                continue

            overlap_ratio = overlap_pixels / max(component_pixels, 1)
            if overlap_ratio < _TEXTURE_SEAM_HQ_MIN_OVERLAP_RATIO:
                continue

            gray_diff = _masked_gray_difference(reference_patch, shifted_patch, merge_mask)
            gradient_diff = _masked_gradient_difference(reference_patch, shifted_patch, merge_mask)
            detail_advantage = _masked_detail_advantage(reference_patch, shifted_patch, merge_mask)
            score = (
                1.10 * overlap_ratio
                + 0.35 * max(0.0, min(1.0, response))
                + (_TEXTURE_SEAM_HQ_DETAIL_WEIGHT * detail_advantage)
                - gray_diff
                - (_TEXTURE_SEAM_HQ_GRADIENT_WEIGHT * gradient_diff)
            )
            choice = (score, shifted_patch, shifted_mask, merge_mask, max(overlap_ratio, response))
            if best_choice is None or choice[0] > best_choice[0]:
                best_choice = choice

    if best_choice is None:
        return None
    _score, shifted_patch, shifted_mask, merge_mask, confidence = best_choice
    return shifted_patch, shifted_mask, merge_mask, float(confidence)


def _merge_detail_component(
    base_low_patch: np.ndarray,
    base_detail_patch: np.ndarray,
    shifted_patch: np.ndarray,
    shifted_mask: np.ndarray,
    merge_mask: np.ndarray,
    confidence: float,
) -> np.ndarray:
    cand_low = _masked_gaussian(shifted_patch, shifted_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    cand_detail = shifted_patch - cand_low
    ref_mag = np.linalg.norm(base_detail_patch, axis=2)
    cand_mag = np.linalg.norm(cand_detail, axis=2)
    stronger = cand_mag > (ref_mag * 1.03)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    low_mix = min(0.60, _TEXTURE_SEAM_HQ_LOW_BLEND * max(confidence, 0.55))
    detail_gain = min(1.00, max(0.70, _TEXTURE_SEAM_HQ_DETAIL_GAIN * max(confidence, 0.70)))
    chosen_detail = np.where(stronger[:, :, None], cand_detail, base_detail_patch)
    merged = np.clip(
        (1.0 - low_mix) * base_low_patch
        + low_mix * cand_low
        + (1.0 - detail_gain) * base_detail_patch
        + detail_gain * chosen_detail,
        0.0,
        1.0,
    )
    merged = np.where(
        stronger[:, :, None],
        np.maximum(merged, shifted_patch),
        merged,
    )
    color_advantage = np.max(np.abs(shifted_patch - base_low_patch), axis=2) > 0.18
    if np.any(color_advantage & merge_mask):
        color_mix = min(0.85, max(0.60, confidence))
        boosted = np.clip(
            (1.0 - color_mix) * merged + color_mix * shifted_patch,
            0.0,
            1.0,
        )
        merged = np.where(color_advantage[:, :, None], boosted, merged)
    return merged


def _resolve_texture_quality_boost(requested: bool | str | int | None = None) -> bool:
    if requested is None:
        requested = os.environ.get("TEXTURE_QUALITY_BOOST", "0")
    if isinstance(requested, bool):
        return requested
    if isinstance(requested, (int, np.integer)):
        return bool(requested)
    text = str(requested).strip().lower()
    return text in {"1", "true", "yes", "on", "hq", "high"}


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

    pxi = np.clip(np.rint(px_proj).astype(np.int32), 0, img_w - 1)
    pyi = np.clip(np.rint(py_proj).astype(np.int32), 0, img_h - 1)
    mask_ok = mask_bool[pyi, pxi]

    depth_ref = depth_buffer[pyi, pxi].astype(np.float64)
    cos_safe = np.maximum(np.abs(cos_angle), 0.1)
    visibility_eps = np.maximum(1e-4, 0.003 * depth_ref / cos_safe)
    visible = depths <= (depth_ref + visibility_eps)

    facing = cos_angle > min_cos
    valid = in_bounds & mask_ok & visible & facing

    score = np.zeros(len(pos3d), dtype=np.float64)
    if np.any(valid):
        ang = np.power(np.maximum(cos_angle[valid], 0.0), angle_exp)
        dist_term = np.power(np.maximum(dists[valid], 1e-6), dist_pow)
        score[valid] = ang / np.maximum(dist_term, 1e-10)

    return valid, score, px_proj, py_proj


def _update_topk_scores(
    best_scores: np.ndarray,
    best_views: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    vidx: int,
) -> None:
    """Update per-texel top-K view scores in-place.

    For texels where *valid* is True and *score* exceeds the current worst
    entry in the top-K list, the worst slot is replaced and the array is
    re-sorted to maintain descending order.

    Args:
        best_scores: (n_texels, K) float32 — scores kept in descending order.
        best_views: (n_texels, K) int32 — corresponding view indices.
        valid: (n_texels,) bool — which texels have a usable projection.
        score: (n_texels,) float64 — per-texel score for view *vidx*.
        vidx: View index to record.
    """
    K = best_scores.shape[1]
    candidates = valid & (score > best_scores[:, K - 1])
    idx = np.where(candidates)[0]
    if idx.size == 0:
        return

    # Replace worst slot (last column)
    best_scores[idx, K - 1] = score[idx].astype(np.float32)
    best_views[idx, K - 1] = vidx

    # Insert-sort: bubble the new element up from slot K-1
    for j in range(K - 2, -1, -1):
        swap = best_scores[idx, j] < best_scores[idx, j + 1]
        swap_idx = idx[swap]
        if swap_idx.size == 0:
            break
        tmp_s = best_scores[swap_idx, j].copy()
        best_scores[swap_idx, j] = best_scores[swap_idx, j + 1]
        best_scores[swap_idx, j + 1] = tmp_s
        tmp_v = best_views[swap_idx, j].copy()
        best_views[swap_idx, j] = best_views[swap_idx, j + 1]
        best_views[swap_idx, j + 1] = tmp_v
        idx = swap_idx


def _apply_view_hardening(
    best_scores: np.ndarray,
    best_views: np.ndarray,
    hard_ratio: float,
) -> int:
    """Zero out non-dominant views when top-1 clearly wins."""
    if best_scores.shape[1] <= 1 or hard_ratio <= 0:
        return 0
    s0 = best_scores[:, 0]
    s1 = best_scores[:, 1]
    dominant = (s0 > 0) & (s1 > 0) & (s0 > hard_ratio * s1)
    n_hard = int(dominant.sum())
    if n_hard > 0:
        best_scores[dominant, 1:] = -1.0
        best_views[dominant, 1:] = -1
    return n_hard


def _compute_conflict_texels(
    pos3d: np.ndarray,
    poses: np.ndarray,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    conflict_ratio: float = _TEXTURE_CONFLICT_RATIO,
    min_view_angle_deg: float = _TEXTURE_CONFLICT_VIEW_ANGLE_DEG,
) -> np.ndarray:
    """Mark texels where two competing views disagree strongly."""
    n_texels = best_scores.shape[0]
    if n_texels == 0 or best_scores.shape[1] <= 1 or len(poses) == 0:
        return np.zeros(n_texels, dtype=bool)

    top0 = best_scores[:, 0].astype(np.float64, copy=False)
    top1 = best_scores[:, 1].astype(np.float64, copy=False)
    view0 = best_views[:, 0]
    view1 = best_views[:, 1]

    close_scores = (top0 > 0) & (top1 > 0) & (top0 < conflict_ratio * top1)
    different_views = view0 >= 0
    different_views &= view1 >= 0
    different_views &= view0 != view1
    candidates = close_scores & different_views

    conflict = np.zeros(n_texels, dtype=bool)
    idx = np.where(candidates)[0]
    if idx.size == 0:
        return conflict

    cam_pos = poses[:, :3, 3]
    dir0 = cam_pos[view0[idx]] - pos3d[idx]
    dir1 = cam_pos[view1[idx]] - pos3d[idx]
    dir0 /= np.maximum(np.linalg.norm(dir0, axis=1, keepdims=True), 1e-10)
    dir1 /= np.maximum(np.linalg.norm(dir1, axis=1, keepdims=True), 1e-10)
    dots = np.sum(dir0 * dir1, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    conflict[idx] = angles_deg >= min_view_angle_deg
    return conflict


def _build_face_adjacency(faces: np.ndarray) -> list[np.ndarray]:
    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    edge_to_face: dict[tuple[int, int], int] = {}

    for fi, face in enumerate(faces):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            prev = edge_to_face.get(edge)
            if prev is None:
                edge_to_face[edge] = fi
                continue
            adjacency[fi].add(prev)
            adjacency[prev].add(fi)

    return [
        np.fromiter(sorted(neighbors), dtype=np.int32)
        if neighbors else np.empty(0, dtype=np.int32)
        for neighbors in adjacency
    ]


def _compute_face_locked_views(
    fids: np.ndarray,
    n_faces: int,
    n_views: int,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    conflict_texels: np.ndarray,
    face_normals: np.ndarray,
    faces: np.ndarray,
    min_conflict_texels: int = _TEXTURE_CONFLICT_FACE_MIN_TEXELS,
    min_conflict_frac: float = _TEXTURE_CONFLICT_FACE_MIN_FRAC,
    min_view_coverage: float = _TEXTURE_CONFLICT_FACE_MIN_COVERAGE,
    smooth_dot: float = _TEXTURE_CONFLICT_SMOOTH_DOT,
    smooth_gain: float = _TEXTURE_CONFLICT_SMOOTH_GAIN,
    smooth_min_neighbors: int = _TEXTURE_CONFLICT_SMOOTH_MIN_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a single dominant view for conflict-heavy faces."""
    face_locked_view = np.full(n_faces, -1, dtype=np.int32)
    if n_faces == 0 or n_views <= 0 or best_scores.size == 0 or not np.any(conflict_texels):
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    face_texel_count = np.bincount(fids, minlength=n_faces).astype(np.int32)
    face_conflict_count = np.bincount(fids[conflict_texels], minlength=n_faces).astype(np.int32)
    face_conflict_frac = np.divide(
        face_conflict_count,
        np.maximum(face_texel_count, 1),
        dtype=np.float32,
    )
    candidate_faces = np.where(
        (face_conflict_count >= min_conflict_texels)
        & (face_conflict_frac >= min_conflict_frac)
    )[0]
    if candidate_faces.size == 0:
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    candidate_mask = np.zeros(n_faces, dtype=bool)
    candidate_mask[candidate_faces] = True
    candidate_rows = np.full(n_faces, -1, dtype=np.int32)
    candidate_rows[candidate_faces] = np.arange(candidate_faces.size, dtype=np.int32)

    active_faces_parts: list[np.ndarray] = []
    active_views_parts: list[np.ndarray] = []
    active_scores_parts: list[np.ndarray] = []
    for k in range(best_scores.shape[1]):
        views_k = best_views[:, k]
        active = (views_k >= 0) & conflict_texels & candidate_mask[fids]
        if not np.any(active):
            continue
        active_faces_parts.append(fids[active].astype(np.int32, copy=False))
        active_views_parts.append(views_k[active].astype(np.int32, copy=False))
        active_scores_parts.append(best_scores[active, k].astype(np.float32, copy=False))

    if not active_faces_parts:
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    active_faces = np.concatenate(active_faces_parts)
    active_views = np.concatenate(active_views_parts)
    active_scores = np.concatenate(active_scores_parts)
    pair_keys = active_faces.astype(np.int64) * np.int64(n_views) + active_views.astype(np.int64)
    order = np.argsort(pair_keys, kind="stable")
    pair_keys = pair_keys[order]
    active_faces = active_faces[order]
    active_views = active_views[order]
    active_scores = active_scores[order]

    run_starts = np.flatnonzero(np.r_[True, pair_keys[1:] != pair_keys[:-1]])
    run_ends = np.r_[run_starts[1:], len(pair_keys)]
    pair_faces = active_faces[run_starts]
    pair_views = active_views[run_starts]
    pair_support = np.add.reduceat(active_scores, run_starts).astype(np.float32, copy=False)
    pair_coverage = (run_ends - run_starts).astype(np.int32, copy=False)

    dominant_support = np.zeros(candidate_faces.size, dtype=np.float32)
    dominant_coverage = np.zeros(candidate_faces.size, dtype=np.int32)
    dominant_cols = np.full(candidate_faces.size, -1, dtype=np.int32)
    face_view_stats: dict[tuple[int, int], tuple[float, int]] = {}
    for face_i, view_i, support_i, coverage_i in zip(
        pair_faces.tolist(),
        pair_views.tolist(),
        pair_support.tolist(),
        pair_coverage.tolist(),
        strict=False,
    ):
        row = candidate_rows[int(face_i)]
        if row < 0:
            continue
        face_view_stats[(int(face_i), int(view_i))] = (float(support_i), int(coverage_i))
        if support_i > dominant_support[row]:
            dominant_support[row] = float(support_i)
            dominant_coverage[row] = int(coverage_i)
            dominant_cols[row] = int(view_i)

    coverage_ratio = np.divide(
        dominant_coverage,
        np.maximum(face_conflict_count[candidate_faces], 1),
        dtype=np.float32,
    )
    eligible = (dominant_support > 0.0) & (coverage_ratio >= min_view_coverage)
    if not np.any(eligible):
        return face_locked_view, np.zeros(n_faces, dtype=np.float32)

    face_locked_view[candidate_faces[eligible]] = dominant_cols[eligible].astype(np.int32)
    face_support = np.zeros(n_faces, dtype=np.float32)
    face_support[candidate_faces[eligible]] = dominant_support[eligible]

    adjacency = _build_face_adjacency(faces)
    locked_faces = np.where(face_locked_view >= 0)[0]
    if locked_faces.size == 0:
        return face_locked_view, face_support

    smoothed = face_locked_view.copy()
    for fi in locked_faces:
        row = candidate_rows[fi]
        if row < 0:
            continue
        neighbors = adjacency[fi]
        if neighbors.size == 0:
            continue
        same_region = neighbors[smoothed[neighbors] >= 0]
        if same_region.size == 0:
            continue
        normal_dot = np.sum(face_normals[same_region] * face_normals[fi], axis=1)
        similar = same_region[normal_dot >= smooth_dot]
        if similar.size < smooth_min_neighbors:
            continue
        labels = smoothed[similar]
        weights = face_support[similar]
        totals: dict[int, float] = {}
        for label, weight in zip(labels.tolist(), weights.tolist(), strict=False):
            label_i = int(label)
            local_support, _local_coverage = face_view_stats.get((int(fi), label_i), (0.0, 0))
            if local_support <= 0.0:
                continue
            totals[label_i] = totals.get(label_i, 0.0) + float(weight)
        if not totals:
            continue
        best_label, best_total = max(totals.items(), key=lambda item: item[1])
        if best_label != int(smoothed[fi]) and best_total > float(face_support[fi]) * smooth_gain:
            smoothed[fi] = int(best_label)

    changed = smoothed != face_locked_view
    if np.any(changed):
        face_locked_view = smoothed
        for fi in np.where(changed)[0]:
            local_support, _local_coverage = face_view_stats.get((int(fi), int(face_locked_view[fi])), (0.0, 0))
            face_support[fi] = float(local_support)

    return face_locked_view, face_support



def _aggregate_face_view_stats(
    fids: np.ndarray,
    n_faces: int,
    n_views: int,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    face_mask: np.ndarray,
) -> tuple[dict[int, list[tuple[int, float, int]]], np.ndarray]:
    """Aggregate per-face view support from texel-level top-K scores."""
    face_texel_count = np.bincount(fids, minlength=n_faces).astype(np.int32)
    if n_faces == 0 or n_views <= 0 or best_scores.size == 0 or not np.any(face_mask):
        return {}, face_texel_count

    active_faces_parts: list[np.ndarray] = []
    active_views_parts: list[np.ndarray] = []
    active_scores_parts: list[np.ndarray] = []
    for k in range(best_scores.shape[1]):
        views_k = best_views[:, k]
        active = face_mask[fids] & (views_k >= 0) & (best_scores[:, k] > 0)
        if not np.any(active):
            continue
        active_faces_parts.append(fids[active].astype(np.int32, copy=False))
        active_views_parts.append(views_k[active].astype(np.int32, copy=False))
        active_scores_parts.append(best_scores[active, k].astype(np.float32, copy=False))

    if not active_faces_parts:
        return {}, face_texel_count

    active_faces = np.concatenate(active_faces_parts)
    active_views = np.concatenate(active_views_parts)
    active_scores = np.concatenate(active_scores_parts)
    pair_keys = active_faces.astype(np.int64) * np.int64(n_views) + active_views.astype(np.int64)
    order = np.argsort(pair_keys, kind="stable")
    pair_keys = pair_keys[order]
    active_faces = active_faces[order]
    active_views = active_views[order]
    active_scores = active_scores[order]

    run_starts = np.flatnonzero(np.r_[True, pair_keys[1:] != pair_keys[:-1]])
    run_ends = np.r_[run_starts[1:], len(pair_keys)]
    pair_faces = active_faces[run_starts]
    pair_views = active_views[run_starts]
    pair_support = np.add.reduceat(active_scores, run_starts).astype(np.float32, copy=False)
    pair_coverage = (run_ends - run_starts).astype(np.int32, copy=False)

    face_view_stats: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for face_i, view_i, support_i, coverage_i in zip(
        pair_faces.tolist(),
        pair_views.tolist(),
        pair_support.tolist(),
        pair_coverage.tolist(),
        strict=False,
    ):
        face_view_stats[int(face_i)].append((int(view_i), float(support_i), int(coverage_i)))

    for face_i, stats in face_view_stats.items():
        face_view_stats[face_i] = sorted(stats, key=lambda item: item[1], reverse=True)
    return dict(face_view_stats), face_texel_count


def _collect_region_components(
    seed_faces: np.ndarray,
    adjacency: list[np.ndarray],
    face_normals: np.ndarray,
    face_label_sets: dict[int, set[int]],
    min_normal_dot: float = _TEXTURE_REGION_NORMAL_DOT,
) -> list[np.ndarray]:
    if seed_faces.size == 0:
        return []

    visited = np.zeros(len(face_normals), dtype=bool)
    components: list[np.ndarray] = []

    for start in seed_faces.tolist():
        start_i = int(start)
        if visited[start_i]:
            continue
        stack = [start_i]
        visited[start_i] = True
        faces_in_component: list[int] = []
        while stack:
            fi = stack.pop()
            faces_in_component.append(fi)
            for nb in adjacency[fi].tolist():
                nb_i = int(nb)
                if visited[nb_i]:
                    continue
                if float(np.dot(face_normals[fi], face_normals[nb_i])) < min_normal_dot:
                    continue
                labels_fi = face_label_sets.get(fi)
                labels_nb = face_label_sets.get(nb_i)
                if not labels_fi or not labels_nb or labels_fi.isdisjoint(labels_nb):
                    continue
                visited[nb_i] = True
                stack.append(nb_i)
        components.append(np.array(faces_in_component, dtype=np.int32))
    return components


def _compute_region_gc_locked_views(
    fids: np.ndarray,
    n_faces: int,
    n_views: int,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    conflict_texels: np.ndarray,
    face_normals: np.ndarray,
    faces: np.ndarray,
    base_locked_view: np.ndarray,
    min_conflict_texels: int = _TEXTURE_CONFLICT_FACE_MIN_TEXELS,
    min_conflict_frac: float = _TEXTURE_CONFLICT_FACE_MIN_FRAC,
    min_region_faces: int = _TEXTURE_REGION_MIN_FACES,
    max_region_labels: int = _TEXTURE_REGION_TOP_LABELS,
    max_iters: int = _TEXTURE_REGION_MAX_ITERS,
    smoothness: float = _TEXTURE_REGION_SMOOTHNESS,
    min_label_coverage: float = _TEXTURE_REGION_MIN_LABEL_COVERAGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth ambiguous curved regions with face-graph label optimization."""
    face_locked_view = base_locked_view.copy()
    face_support = np.zeros(n_faces, dtype=np.float32)
    region_id_per_face = np.full(n_faces, -1, dtype=np.int32)
    if n_faces == 0 or n_views <= 0 or best_scores.size == 0 or not np.any(conflict_texels):
        return face_locked_view, face_support, region_id_per_face

    face_texel_count = np.bincount(fids, minlength=n_faces).astype(np.int32)
    face_conflict_count = np.bincount(fids[conflict_texels], minlength=n_faces).astype(np.int32)
    face_conflict_frac = np.divide(
        face_conflict_count,
        np.maximum(face_texel_count, 1),
        dtype=np.float32,
    )
    candidate_faces = np.where(
        (face_conflict_count >= min_conflict_texels)
        & (face_conflict_frac >= min_conflict_frac)
    )[0]
    if candidate_faces.size == 0:
        return face_locked_view, face_support, region_id_per_face

    textured_face_mask = face_texel_count > 0
    face_view_stats, face_texel_count = _aggregate_face_view_stats(
        fids=fids,
        n_faces=n_faces,
        n_views=n_views,
        best_scores=best_scores,
        best_views=best_views,
        face_mask=textured_face_mask,
    )
    if not face_view_stats:
        return face_locked_view, face_support, region_id_per_face

    adjacency = _build_face_adjacency(faces)
    face_label_sets = {
        int(face_i): {int(view_i) for view_i, _support_i, _coverage_i in stats[:max_region_labels]}
        for face_i, stats in face_view_stats.items()
    }
    components = _collect_region_components(
        seed_faces=candidate_faces,
        adjacency=adjacency,
        face_normals=face_normals,
        face_label_sets=face_label_sets,
    )
    region_counter = 0

    for component in components:
        if component.size < min_region_faces:
            continue

        component_views: dict[int, float] = defaultdict(float)
        face_costs: dict[int, dict[int, float]] = {}
        face_coverages: dict[int, dict[int, float]] = {}
        init_labels: dict[int, int] = {}
        eligible_faces: list[int] = []

        for face_i in component.tolist():
            stats = face_view_stats.get(int(face_i), [])
            if not stats:
                continue
            max_support_local = max(float(item[1]) for item in stats)
            if max_support_local <= 0.0:
                continue
            eligible_faces.append(int(face_i))
            local_costs: dict[int, float] = {}
            local_coverages: dict[int, float] = {}
            for view_i, support_i, coverage_i in stats[:max_region_labels]:
                component_views[int(view_i)] += float(support_i)
                coverage_ratio = float(coverage_i) / max(float(face_texel_count[int(face_i)]), 1.0)
                cost = 1.0 - (float(support_i) / max_support_local)
                if coverage_ratio < min_label_coverage:
                    ratio = coverage_ratio / max(min_label_coverage, 1e-6)
                    cost += _TEXTURE_REGION_LOW_COVERAGE_PENALTY * (1.0 - max(0.0, min(1.0, ratio)))
                local_costs[int(view_i)] = float(cost)
                local_coverages[int(view_i)] = coverage_ratio
            face_costs[int(face_i)] = local_costs
            face_coverages[int(face_i)] = local_coverages
            base_label = int(base_locked_view[int(face_i)])
            if base_label >= 0 and base_label in local_costs:
                init_labels[int(face_i)] = base_label

        if len(eligible_faces) < min_region_faces or len(component_views) < 2:
            continue

        allowed_labels = [
            int(view_i)
            for view_i, _support_i in sorted(component_views.items(), key=lambda item: item[1], reverse=True)[:max_region_labels]
        ]
        if len(allowed_labels) < 2:
            continue

        face_indices = np.array(sorted(eligible_faces), dtype=np.int32)
        face_set = set(face_indices.tolist())
        face_update_order: list[tuple[float, int]] = []
        current_labels: dict[int, int] = {}
        for face_i in face_indices.tolist():
            costs = face_costs[int(face_i)]
            ranked_local = sorted(
                (
                    costs.get(label, _TEXTURE_REGION_MISSING_COST),
                    -face_coverages[int(face_i)].get(label, 0.0),
                    int(label),
                )
                for label in allowed_labels
            )
            if len(ranked_local) >= 2:
                margin = float(ranked_local[1][0] - ranked_local[0][0])
            else:
                margin = _TEXTURE_REGION_MISSING_COST
            face_update_order.append((margin, int(face_i)))
            best_label = min(
                allowed_labels,
                key=lambda label: (
                    costs.get(label, _TEXTURE_REGION_MISSING_COST),
                    -face_coverages[int(face_i)].get(label, 0.0),
                    label,
                ),
            )
            current_labels[int(face_i)] = init_labels.get(int(face_i), int(best_label))
        ordered_faces = [face_i for _margin, face_i in sorted(face_update_order, key=lambda item: (item[0], item[1]))]

        edge_weights: dict[tuple[int, int], float] = {}
        for face_i in face_indices.tolist():
            for nb in adjacency[int(face_i)].tolist():
                nb_i = int(nb)
                if nb_i not in face_set or nb_i <= int(face_i):
                    continue
                normal_dot = float(np.clip(np.dot(face_normals[int(face_i)], face_normals[nb_i]), -1.0, 1.0))
                if normal_dot < _TEXTURE_REGION_NORMAL_DOT:
                    continue
                smooth_weight = smoothness * (
                    0.25 + 0.75 * (normal_dot - _TEXTURE_REGION_NORMAL_DOT) / max(1e-6, 1.0 - _TEXTURE_REGION_NORMAL_DOT)
                )
                edge_weights[(int(face_i), nb_i)] = max(0.0, float(smooth_weight))

        if not edge_weights:
            continue

        for _iter_idx in range(max_iters):
            changed = False
            for face_i in ordered_faces:
                costs = face_costs[int(face_i)]
                best_energy: tuple[float, float, int] | None = None
                best_label = current_labels[int(face_i)]
                for label in allowed_labels:
                    local_cost = costs.get(label, _TEXTURE_REGION_MISSING_COST)
                    smooth_cost = 0.0
                    for nb in adjacency[int(face_i)].tolist():
                        nb_i = int(nb)
                        if nb_i not in face_set:
                            continue
                        edge_key = (min(int(face_i), nb_i), max(int(face_i), nb_i))
                        weight = edge_weights.get(edge_key, 0.0)
                        if weight <= 0.0:
                            continue
                        if label != current_labels[nb_i]:
                            smooth_cost += weight
                    energy = (local_cost + smooth_cost, local_cost, int(label))
                    if best_energy is None or energy < best_energy:
                        best_energy = energy
                        best_label = int(label)
                if best_label != current_labels[int(face_i)]:
                    current_labels[int(face_i)] = best_label
                    changed = True
            if not changed:
                break

        assigned = 0
        for face_i in face_indices.tolist():
            chosen_label = current_labels[int(face_i)]
            coverage_ratio = face_coverages[int(face_i)].get(chosen_label, 0.0)
            if coverage_ratio < min_label_coverage:
                continue
            support_candidates = {view_i: support_i for view_i, support_i, _cov_i in face_view_stats.get(int(face_i), [])}
            support_i = float(support_candidates.get(chosen_label, 0.0))
            if support_i <= 0.0:
                continue
            face_locked_view[int(face_i)] = int(chosen_label)
            face_support[int(face_i)] = support_i
            region_id_per_face[int(face_i)] = region_counter
            assigned += 1

        if assigned >= min_region_faces:
            region_counter += 1
        else:
            for face_i in face_indices.tolist():
                face_locked_view[int(face_i)] = int(base_locked_view[int(face_i)])
                face_support[int(face_i)] = 0.0
            region_id_per_face[face_indices] = -1

    return face_locked_view, face_support, region_id_per_face


def _apply_narrow_seam_leveling(
    texture: np.ndarray,
    valid_mask: np.ndarray,
    label_buffer: np.ndarray,
    blend: float = _TEXTURE_SEAM_LEVEL_BLEND,
    dilate_iters: int = _TEXTURE_SEAM_LEVEL_DILATE,
    sigma: float = _TEXTURE_SEAM_LEVEL_SIGMA,
    detail_texture: np.ndarray | None = None,
    detail_valid_mask: np.ndarray | None = None,
    detail_textures: list[np.ndarray] | None = None,
    detail_valid_masks: list[np.ndarray] | None = None,
    quality_boost: bool = False,
) -> tuple[np.ndarray, int]:
    """Soften region-label seams while preserving local texture detail."""
    valid_labels = label_buffer >= 0
    if texture.size == 0 or not np.any(valid_mask & valid_labels):
        return texture, 0

    boundary = _compute_label_boundary(label_buffer)
    if not np.any(boundary):
        return texture, 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    band = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=max(1, int(dilate_iters))).astype(bool)
    editable = band & valid_mask & valid_labels
    if not np.any(editable):
        return texture, int(boundary.sum())

    result = texture.copy()
    local_low = _masked_gaussian(result, valid_mask, sigma=max(0.4, _TEXTURE_SEAM_LOW_SIGMA))
    smooth_low = _masked_gaussian(result, valid_mask, sigma=max(sigma, _TEXTURE_SEAM_SMOOTH_SIGMA))
    detail = result - local_low
    boundary_distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
    boundary_weight = np.clip(
        1.0 - (boundary_distance / max(float(dilate_iters) + 0.5, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    local_blend = (0.7 * blend * boundary_weight[editable]).astype(np.float32, copy=False)
    result[editable] = np.clip(
        (1.0 - local_blend[:, None]) * local_low[editable]
        + local_blend[:, None] * smooth_low[editable]
        + detail[editable],
        0.0,
        1.0,
    )

    if (
        quality_boost
    ):
        detail_sources: list[tuple[np.ndarray, np.ndarray]] = []
        if (
            detail_texture is not None
            and detail_valid_mask is not None
            and detail_texture.shape == texture.shape
            and detail_valid_mask.shape == valid_mask.shape
        ):
            detail_sources.append((detail_texture, detail_valid_mask))
        if detail_textures is not None and detail_valid_masks is not None:
            for tex_src, mask_src in zip(detail_textures, detail_valid_masks, strict=False):
                if tex_src.shape != texture.shape or mask_src.shape != valid_mask.shape:
                    continue
                detail_sources.append((tex_src, mask_src))

        if not detail_sources:
            return result, int(boundary.sum())

        detail_avail = np.zeros(valid_mask.shape, dtype=bool)
        for _detail_texture, detail_mask_src in detail_sources:
            detail_avail |= detail_mask_src
        detail_mask = editable & detail_avail
        if np.any(detail_mask):
            num_components, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                detail_mask.astype(np.uint8),
                connectivity=8,
            )
            refined = result.copy()
            base_low = _masked_gaussian(refined, valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
            base_detail = refined - base_low
            for comp_id in range(1, num_components):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
                    continue
                x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                y = int(stats[comp_id, cv2.CC_STAT_TOP])
                w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                pad = _TEXTURE_SEAM_HQ_PATCH_PAD
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(texture.shape[1], x + w + pad)
                y1 = min(texture.shape[0], y + h + pad)
                local_component = labels[y0:y1, x0:x1] == comp_id
                local_ref = refined[y0:y1, x0:x1]
                candidate_patches: list[tuple[np.ndarray, np.ndarray]] = []
                for candidate_texture, candidate_mask in detail_sources:
                    local_cand_mask = candidate_mask[y0:y1, x0:x1]
                    if not np.any(local_component & local_cand_mask):
                        continue
                    candidate_patches.append(
                        (
                            candidate_texture[y0:y1, x0:x1],
                            local_cand_mask,
                        )
                    )
                selected = _select_best_detail_candidate(local_ref, local_component, candidate_patches)
                if selected is None:
                    continue
                shifted_cand, shifted_mask, merge_mask, confidence = selected
                base_low_patch = base_low[y0:y1, x0:x1]
                base_detail_patch = base_detail[y0:y1, x0:x1]
                merged = _merge_detail_component(
                    base_low_patch=base_low_patch,
                    base_detail_patch=base_detail_patch,
                    shifted_patch=shifted_cand,
                    shifted_mask=shifted_mask,
                    merge_mask=merge_mask,
                    confidence=confidence,
                )
                patch = refined[y0:y1, x0:x1]
                patch[merge_mask] = merged[merge_mask]
                refined[y0:y1, x0:x1] = patch
            result = refined
    return result, int(boundary.sum())


def _sample_companion_texture(
    tex_shape: tuple[int, int, int],
    tidx: np.ndarray,
    companion_views: np.ndarray,
    tex_xs: np.ndarray,
    tex_ys: np.ndarray,
    pos3d: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
    pose_frame_indices: list[int],
    frames_dir: str,
    mask_dir: str,
    frame_cache: _FrameCache,
    depth_cache: OrderedDict[int, np.ndarray],
    new_vertices: np.ndarray,
    new_faces: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
) -> tuple[np.ndarray, np.ndarray]:
    companion_texture = np.zeros(tex_shape, dtype=np.float32)
    companion_valid = np.zeros(tex_shape[:2], dtype=bool)
    if tidx.size == 0:
        return companion_texture, companion_valid

    local_views = companion_views[tidx]
    usable = local_views >= 0
    if not np.any(usable):
        return companion_texture, companion_valid
    active_tidx = tidx[usable]
    active_views = local_views[usable]

    for vidx in np.unique(active_views):
        if vidx < 0:
            continue
        view_tidx = active_tidx[active_views == vidx]
        if view_tidx.size == 0:
            continue
        src_idx = int(pose_frame_indices[int(vidx)])
        c2w = poses[int(vidx)]
        try:
            frame = frame_cache.load_frame(frames_dir, src_idx)
            mask_bool = frame_cache.load_mask(mask_dir, src_idx)
        except FileNotFoundError:
            continue

        if int(vidx) in depth_cache:
            depth_buffer = depth_cache[int(vidx)]
        else:
            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
            depth_cache[int(vidx)] = depth_buffer
        valid, _score, px_proj, py_proj = _evaluate_view_samples(
            pos3d=pos3d[view_tidx],
            normals=normals[view_tidx],
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
        if not np.any(valid):
            continue
        fill_tidx = view_tidx[valid]
        colors = _bilinear_sample(frame, px_proj[valid], py_proj[valid]).astype(np.float32)
        companion_texture[tex_ys[fill_tidx], tex_xs[fill_tidx]] = colors
        companion_valid[tex_ys[fill_tidx], tex_xs[fill_tidx]] = True
    return companion_texture, companion_valid


def _sample_companion_patch(
    patch_shape: tuple[int, int, int],
    global_tidx: np.ndarray,
    companion_views: np.ndarray,
    patch_x0: int,
    patch_y0: int,
    tex_xs: np.ndarray,
    tex_ys: np.ndarray,
    pos3d: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
    pose_frame_indices: list[int],
    frames_dir: str,
    mask_dir: str,
    frame_cache: _FrameCache,
    depth_cache: OrderedDict[int, np.ndarray],
    new_vertices: np.ndarray,
    new_faces: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
) -> tuple[np.ndarray, np.ndarray]:
    companion_patch = np.zeros(patch_shape, dtype=np.float32)
    companion_valid = np.zeros(patch_shape[:2], dtype=bool)
    if global_tidx.size == 0 or companion_views.size == 0:
        return companion_patch, companion_valid

    usable = companion_views >= 0
    if not np.any(usable):
        return companion_patch, companion_valid

    active_tidx = global_tidx[usable]
    active_views = companion_views[usable]
    for vidx in np.unique(active_views):
        if vidx < 0:
            continue
        view_tidx = active_tidx[active_views == vidx]
        if view_tidx.size == 0:
            continue
        src_idx = int(pose_frame_indices[int(vidx)])
        c2w = poses[int(vidx)]
        try:
            frame = frame_cache.load_frame(frames_dir, src_idx)
            mask_bool = frame_cache.load_mask(mask_dir, src_idx)
        except FileNotFoundError:
            continue

        if int(vidx) in depth_cache:
            depth_buffer = depth_cache[int(vidx)]
        else:
            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
            depth_cache[int(vidx)] = depth_buffer
        valid, _score, px_proj, py_proj = _evaluate_view_samples(
            pos3d=pos3d[view_tidx],
            normals=normals[view_tidx],
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
        if not np.any(valid):
            continue
        fill_tidx = view_tidx[valid]
        local_x = tex_xs[fill_tidx] - int(patch_x0)
        local_y = tex_ys[fill_tidx] - int(patch_y0)
        inside = (
            (local_x >= 0)
            & (local_x < patch_shape[1])
            & (local_y >= 0)
            & (local_y < patch_shape[0])
        )
        if not np.any(inside):
            continue
        fill_tidx = fill_tidx[inside]
        local_x = local_x[inside]
        local_y = local_y[inside]
        colors = _bilinear_sample(frame, px_proj[valid][inside], py_proj[valid][inside]).astype(np.float32)
        companion_patch[local_y, local_x] = colors
        companion_valid[local_y, local_x] = True
    return companion_patch, companion_valid


def _apply_quality_boost_detail_refinement(
    texture: np.ndarray,
    valid_mask: np.ndarray,
    label_buffer: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    locked_view_per_texel: np.ndarray,
    pos3d: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
    pose_frame_indices: list[int],
    frames_dir: str,
    mask_dir: str,
    frame_cache: _FrameCache,
    depth_cache: OrderedDict[int, np.ndarray],
    new_vertices: np.ndarray,
    new_faces: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
    dilate_iters: int = _TEXTURE_SEAM_HQ_DILATE,
) -> tuple[np.ndarray, int, int]:
    valid_labels = label_buffer >= 0
    if texture.size == 0 or not np.any(valid_mask & valid_labels):
        return texture, 0, 0

    boundary = _compute_label_boundary(label_buffer)
    if not np.any(boundary):
        return texture, 0, 0

    band = cv2.dilate(
        boundary.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=max(1, int(dilate_iters)),
    ).astype(bool)
    detail_mask = band & valid_mask & valid_labels
    if not np.any(detail_mask):
        return texture, int(boundary.sum()), 0

    num_components, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        detail_mask.astype(np.uint8),
        connectivity=8,
    )
    refined = texture.copy()
    base_low = _masked_gaussian(refined, valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    base_detail = refined - base_low
    refined_components = 0
    sampled_detail_pixels = 0

    for comp_id in range(1, num_components):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
            continue
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        pad = _TEXTURE_SEAM_HQ_PATCH_PAD
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(texture.shape[1], x + w + pad)
        y1 = min(texture.shape[0], y + h + pad)
        local_component = labels[y0:y1, x0:x1] == comp_id

        in_bbox = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
        if not np.any(in_bbox):
            continue
        bbox_tidx = np.where(in_bbox)[0]
        local_y = ys[bbox_tidx] - y0
        local_x = xs[bbox_tidx] - x0
        component_hits = local_component[local_y, local_x]
        component_tidx = bbox_tidx[component_hits]
        if component_tidx.size < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
            continue

        candidate_view_maps = _select_detail_companion_view_candidates(
            primary_views=locked_view_per_texel[component_tidx],
            best_views=best_views[component_tidx],
            best_scores=best_scores[component_tidx],
        )
        if not candidate_view_maps:
            continue

        local_ref = refined[y0:y1, x0:x1]
        candidate_sources: list[tuple[np.ndarray, np.ndarray]] = []
        for companion_views in candidate_view_maps:
            cand_patch, cand_valid = _sample_companion_patch(
                patch_shape=(y1 - y0, x1 - x0, texture.shape[2]),
                global_tidx=component_tidx,
                companion_views=companion_views,
                patch_x0=x0,
                patch_y0=y0,
                tex_xs=xs,
                tex_ys=ys,
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
            )
            if not np.any(local_component & cand_valid):
                continue
            candidate_sources.append((cand_patch, cand_valid))

        if not candidate_sources:
            continue

        selected = _select_best_detail_candidate(local_ref, local_component, candidate_sources)
        if selected is None:
            continue
        shifted_patch, shifted_mask, merge_mask, confidence = selected
        base_low_patch = base_low[y0:y1, x0:x1]
        base_detail_patch = base_detail[y0:y1, x0:x1]
        merged = _merge_detail_component(
            base_low_patch=base_low_patch,
            base_detail_patch=base_detail_patch,
            shifted_patch=shifted_patch,
            shifted_mask=shifted_mask,
            merge_mask=merge_mask,
            confidence=confidence,
        )
        patch = refined[y0:y1, x0:x1]
        patch[merge_mask] = merged[merge_mask]
        refined[y0:y1, x0:x1] = patch
        refined_components += 1
        sampled_detail_pixels += int(merge_mask.sum())

    return refined, int(boundary.sum()), sampled_detail_pixels


def _identify_cap_texels(
    vert_colors: np.ndarray,
    vmapping: np.ndarray,
    new_faces: np.ndarray,
    fids: np.ndarray,
    has_color: np.ndarray,
) -> np.ndarray | None:
    """Return indices into texel arrays for cap texels missing color, or None."""
    orange = np.array([1.0, 0.55, 0.0])
    orig_colors = vert_colors[vmapping]
    is_orange = np.linalg.norm(orig_colors - orange, axis=1) < 0.05
    cap_face = (
        is_orange[new_faces[:, 0]]
        & is_orange[new_faces[:, 1]]
        & is_orange[new_faces[:, 2]]
    )
    cap_missing = cap_face[fids] & ~has_color
    idx = np.where(cap_missing)[0]
    return idx if idx.size > 0 else None


def _fill_cap_region(
    texture: np.ndarray,
    cap_mask: np.ndarray,
    valid_mask: np.ndarray,
    max_iters: int = 300,
) -> int:
    """Fill cap region by iterative neighbor-averaging. Returns texels filled."""
    cy, cx = np.where(cap_mask)
    margin = 10
    y0 = max(0, int(cy.min()) - margin)
    y1 = min(texture.shape[0], int(cy.max()) + margin + 1)
    x0 = max(0, int(cx.min()) - margin)
    x1 = min(texture.shape[1], int(cx.max()) + margin + 1)

    tex_crop = texture[y0:y1, x0:x1].copy()
    cap_crop = cap_mask[y0:y1, x0:x1]
    val_crop = valid_mask[y0:y1, x0:x1].copy()

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    filled = 0

    for _ in range(max_iters):
        empty = cap_crop & ~val_crop
        if not np.any(empty):
            break
        nc = cv2.filter2D(
            val_crop.astype(np.float32), -1, kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        newly = empty & (nc > 0)
        if not np.any(newly):
            break
        for c in range(3):
            ns = cv2.filter2D(
                tex_crop[:, :, c], -1, kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            tex_crop[:, :, c][newly] = ns[newly] / nc[newly]
        val_crop[newly] = True
        filled += int(newly.sum())

    texture[y0:y1, x0:x1] = tex_crop
    valid_mask[y0:y1, x0:x1] = val_crop
    return filled


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
    new_faces, flipped_winding, ratio_before, ratio_after = orient_faces_outward(
        new_vertices,
        new_faces,
        min_outward_ratio=0.5,
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
        print("  Projection backend request: CUDA (scoring currently uses CPU z-buffer)")
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

        ok = valid[all_tidx]
        fill_tidx = all_tidx[ok]
        fill_weights = all_weights[ok]

        if fill_tidx.size > 0:
            colors = _bilinear_sample(frame, px_proj[fill_tidx], py_proj[fill_tidx])
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
                    new_vertices, new_faces, c2w, K, img_w, img_h
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
                        new_vertices, new_faces, c2w, K, img_w, img_h
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
        cap_missing_idx = _identify_cap_texels(
            vert_colors, vmapping, new_faces, fids, has_color
        )
        if cap_missing_idx is not None:
            cap_mask = np.zeros((tex_res, tex_res), dtype=bool)
            cap_mask[ys[cap_missing_idx], xs[cap_missing_idx]] = True
            valid_mask = np.zeros((tex_res, tex_res), dtype=bool)
            valid_mask[ys[has_color], xs[has_color]] = True

            filled = _fill_cap_region(texture, cap_mask, valid_mask)

            for i in cap_missing_idx:
                if valid_mask[ys[i], xs[i]]:
                    has_color[i] = True
            remaining = int((~has_color).sum())
            print(
                f"  Cap infill: filled={filled}/{len(cap_missing_idx)}, "
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
