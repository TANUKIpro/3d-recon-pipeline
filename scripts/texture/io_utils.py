"""Data loading and caching utilities for texture baking."""

import json
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
