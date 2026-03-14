"""Texture configuration resolution."""

import math
import os

import numpy as np

from scripts.config_defaults import TEXTURE_VIEW_ASSIGN_MODE

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for GPU acceleration
    torch = None


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


def _resolve_texture_quality_boost(requested: bool | str | int | None = None) -> bool:
    if requested is None:
        requested = os.environ.get("TEXTURE_QUALITY_BOOST", "0")
    if isinstance(requested, bool):
        return requested
    if isinstance(requested, (int, np.integer)):
        return bool(requested)
    text = str(requested).strip().lower()
    return text in {"1", "true", "yes", "on", "hq", "high"}
