"""Data loading and caching utilities for texture baking."""

import json
import re
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

from scripts.config_defaults import (
    _TEXTURE_CACHE_SAFETY_MB,
    _TEXTURE_FRAME_BUDGET_RATIO,
    _TEXTURE_MASK_BUDGET_RATIO,
)
from scripts.texture.progress import _get_available_memory_mb


def _load_point_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load PLY, return (points (N,3), colors (N,3) in [0,1])."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    colors = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float64) / 255.0
    return points, colors


def _frame_name_to_index(frame_name: str | None, default: int) -> int:
    if not frame_name:
        return default
    stem = Path(frame_name).stem
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return default
    return int(match.group(1))


def _load_poses(path: str) -> tuple[np.ndarray, list[int]]:
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        poses = np.array(data["poses"], dtype=np.float64)
        frame_indices = data.get("frame_indices")
    elif isinstance(data, list):
        poses_list: list[object] = []
        frame_indices = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict) or "transform_matrix" not in item:
                raise TypeError(
                    "camera_poses.json list entries must include transform_matrix"
                )
            poses_list.append(item["transform_matrix"])
            frame_indices.append(
                int(
                    item.get(
                        "frame_index",
                        _frame_name_to_index(item.get("frame_name"), idx),
                    )
                )
            )
        poses = np.array(poses_list, dtype=np.float64)
    else:
        raise TypeError(
            "camera_poses.json must be either a dict with poses or a list of pose entries"
        )

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
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.0))


def _load_mask(masks_dir: str, idx: int) -> np.ndarray:
    path = _resolve_indexed_file(masks_dir, idx, ".png")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask > 127


def _has_distortion(dist_coeffs: np.ndarray | list | None) -> bool:
    """Return True iff at least one distortion coefficient is non-trivial."""
    if dist_coeffs is None:
        return False
    arr = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    if arr.size == 0:
        return False
    return bool(np.any(np.abs(arr) > 1e-9))


class _FrameCache:
    """LRU cache for decoded frames and masks with memory-aware capacity.

    When ``K`` and ``dist_coeffs`` are provided and the distortion is
    non-trivial, frames and masks are undistorted on load using a precomputed
    remap. The bundle-adjusted geometry is consistent with the COLMAP camera
    model (e.g., SIMPLE_RADIAL); after remap, the rest of the texture pipeline
    can keep using a pure-pinhole projection without color leakage from
    sub-pixel mis-projection. Without this, k1≈0.03 produces >2 px error at
    moderate radial offsets, which destroys fine detail on cylindrical
    objects.
    """

    def __init__(
        self,
        img_w: int,
        img_h: int,
        K: np.ndarray | None = None,
        dist_coeffs: np.ndarray | list | None = None,
    ) -> None:
        avail_mb = _get_available_memory_mb()
        budget_mb = max(0.0, avail_mb - _TEXTURE_CACHE_SAFETY_MB)

        frame_bytes = img_w * img_h * 3 * 4  # float32 RGB
        mask_bytes = img_w * img_h  # bool

        frame_budget_mb = budget_mb * _TEXTURE_FRAME_BUDGET_RATIO
        mask_budget_mb = budget_mb * _TEXTURE_MASK_BUDGET_RATIO

        self._max_frames = max(1, int(frame_budget_mb * 1024 * 1024 / max(frame_bytes, 1)))
        self._max_masks = max(1, int(mask_budget_mb * 1024 * 1024 / max(mask_bytes, 1)))

        self._frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._masks: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._frame_lock = threading.Lock()
        self._mask_lock = threading.Lock()

        self._undistort_map_x: np.ndarray | None = None
        self._undistort_map_y: np.ndarray | None = None
        if K is not None and _has_distortion(dist_coeffs):
            K_arr = np.asarray(K, dtype=np.float64)
            d_arr = np.asarray(dist_coeffs, dtype=np.float64).ravel().astype(np.float32)
            map_x, map_y = cv2.initUndistortRectifyMap(
                K_arr.astype(np.float32),
                d_arr,
                R=None,
                newCameraMatrix=K_arr.astype(np.float32),
                size=(img_w, img_h),
                m1type=cv2.CV_32FC1,
            )
            self._undistort_map_x = map_x
            self._undistort_map_y = map_y

    @property
    def undistort_enabled(self) -> bool:
        return self._undistort_map_x is not None

    def _maybe_undistort_frame(self, arr: np.ndarray) -> np.ndarray:
        if self._undistort_map_x is None:
            return arr
        return cv2.remap(
            arr,
            self._undistort_map_x,
            self._undistort_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    def _maybe_undistort_mask(self, arr: np.ndarray) -> np.ndarray:
        if self._undistort_map_x is None:
            return arr
        # remap with NEAREST keeps the boolean character of the SAM2 mask;
        # bilinear would invent half-edge texels along the silhouette and
        # let background pixels pass the mask-validity test on the bake
        # path.
        m = cv2.remap(
            arr.astype(np.uint8),
            self._undistort_map_x,
            self._undistort_map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return m.astype(bool)

    def load_frame(self, frames_dir: str, idx: int) -> np.ndarray:
        key = (frames_dir, idx)
        with self._frame_lock:
            if key in self._frames:
                self._frames.move_to_end(key)
                return self._frames[key]
        arr = self._maybe_undistort_frame(_load_frame(frames_dir, idx))
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
        arr = self._maybe_undistort_mask(_load_mask(masks_dir, idx))
        with self._mask_lock:
            if key not in self._masks:
                self._masks[key] = arr
                while len(self._masks) > self._max_masks:
                    self._masks.popitem(last=False)
            else:
                self._masks.move_to_end(key)
                arr = self._masks[key]
        return arr
