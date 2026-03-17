"""Texture configuration resolution."""

import math
import os

import numpy as np

from scripts.texture.progress import _get_available_memory_mb

from scripts.config_defaults import (
    TEXTURE_VIEW_ASSIGN_MODE,
    _TEXTURE_UV_BYTES_PER_FACE,
    _TEXTURE_UV_MAX_FACES,
    _TEXTURE_UV_MIN_FACES,
    _TEXTURE_UV_RAM_RESERVE_MB,
)

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
            try:
                import nvdiffrast  # noqa: F401
                print("  nvdiffrast available: GPU rasterization enabled")
            except ImportError:
                print("  nvdiffrast not available: GPU rasterization disabled (Phase 1/2 CPU fallback)")
            return "cuda"
        print("Warning: TEXTURE_DEVICE=cuda requested but CUDA is unavailable; using CPU")
        return "cpu"
    # auto
    if torch is not None and torch.cuda.is_available():
        try:
            import nvdiffrast  # noqa: F401
            print("  nvdiffrast available: GPU rasterization enabled")
        except ImportError:
            print("  nvdiffrast not available: GPU rasterization disabled (Phase 1/2 CPU fallback)")
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


def _resolve_uv_face_budget(n_input_faces: int) -> tuple[int, str]:
    """Resolve the xatlas UV face budget.

    The budget is driven by the input mesh size, not by available RAM.
    RAM acts only as a hard constraint to prevent OOM.

    Args:
        n_input_faces: Number of faces in the input mesh.

    Returns:
        (budget, source) where source is ``"env"`` or ``"auto"``.
        A budget of ``0`` means unlimited (skip simplification).
    """
    raw = os.environ.get("TEXTURE_UV_MAX_FACES", "").strip()
    if raw:
        try:
            val = int(raw)
        except ValueError:
            val = -1  # fall through to auto
        if val == 0:
            return 0, "env"
        if val > 0:
            return val, "env"
        # negative or unparseable → fall through to auto

    # RAM hard limit: prevent xatlas OOM
    available_mb = _get_available_memory_mb()
    usable_mb = max(0.0, available_mb - _TEXTURE_UV_RAM_RESERVE_MB)
    usable_bytes = usable_mb * 1024.0 * 1024.0
    ram_limit = int(usable_bytes / _TEXTURE_UV_BYTES_PER_FACE)

    # budget = min(input faces, time cap, RAM limit), floor at _TEXTURE_UV_MIN_FACES
    budget = max(_TEXTURE_UV_MIN_FACES, min(n_input_faces, _TEXTURE_UV_MAX_FACES, ram_limit))
    return budget, "auto"
