"""VRAM management utilities for sequential GPU pipeline stages."""

import gc
import os
import subprocess
import time

from scripts.config_defaults import (
    _VRAM_PI3X_ESTIMATED_MODEL_MB as _PI3X_ESTIMATED_MODEL_MB,
    _VRAM_PI3X_FRAME_PIXELS_PER_MB as _PI3X_FRAME_PIXELS_PER_MB,
    _VRAM_PI3X_RUNTIME_OVERHEAD_MB as _PI3X_RUNTIME_OVERHEAD_MB,
    _VRAM_PI3X_TARGET_UTILIZATION as _PI3X_TARGET_VRAM_UTILIZATION,
    _VRAM_GATE_MIN_FREE_MB,
)


def _parse_nvidia_int(value: str) -> int | None:
    """Parse nvidia-smi numeric fields that may contain N/A."""
    raw = value.strip()
    if not raw or raw.upper() == "N/A":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def query_nvidia_smi(
    fields: list[str],
    timeout_sec: int = 5,
) -> list[dict[str, str]] | None:
    """Query nvidia-smi and return rows as dicts keyed by field name."""
    if not fields:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    rows: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(fields):
            continue
        rows.append({field: parts[i] for i, field in enumerate(fields)})
    return rows


def get_gpu_inventory() -> list[dict[str, str | int | None]]:
    """Return GPU inventory (one dict per GPU) from nvidia-smi."""
    fields = [
        "index",
        "name",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "utilization.memory",
    ]
    rows = query_nvidia_smi(fields)
    if rows is None:
        return []

    gpus: list[dict[str, str | int | None]] = []
    for row in rows:
        gpus.append(
            {
                "index": _parse_nvidia_int(row.get("index", "")),
                "name": row.get("name", ""),
                "memory_total_mb": _parse_nvidia_int(row.get("memory.total", "")),
                "memory_used_mb": _parse_nvidia_int(row.get("memory.used", "")),
                "memory_free_mb": _parse_nvidia_int(row.get("memory.free", "")),
                "utilization_gpu_pct": _parse_nvidia_int(row.get("utilization.gpu", "")),
                "utilization_memory_pct": _parse_nvidia_int(
                    row.get("utilization.memory", "")
                ),
            }
        )
    return gpus


def pick_gpu_with_most_free_vram(
    inventory: list[dict[str, str | int | None]] | None = None,
) -> dict[str, str | int | None] | None:
    """Pick the GPU with the largest free VRAM from *inventory*."""
    gpus = inventory if inventory is not None else get_gpu_inventory()
    if not gpus:
        return None

    def _free(gpu: dict[str, str | int | None]) -> int:
        value = gpu.get("memory_free_mb")
        return int(value) if isinstance(value, int) else -1

    return max(gpus, key=_free)


def cleanup_pytorch_vram(*models):
    """Release PyTorch models and free GPU memory.

    Args:
        *models: Model objects to delete. Pass the model variables directly.
    """
    import torch

    for model in models:
        del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    log_vram("after cleanup")


def get_free_vram_mb() -> int | None:
    """Return free GPU VRAM in MB via nvidia-smi, or None if unavailable."""
    inventory = get_gpu_inventory()
    if inventory:
        first = inventory[0].get("memory_free_mb")
        if isinstance(first, int):
            return first
    return None


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _env_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _env_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def estimate_pi3x_frame_plan(
    requested_frames: int,
    pixel_limit: int,
    target_utilization: float | None = None,
) -> dict[str, int | float | bool | str | None]:
    """Estimate Pi3X frame count that fits a VRAM utilization target.

    Returns a JSON-serializable dict suitable for both runtime logging and UI.
    """
    req_frames = max(2, int(requested_frames))
    px_limit = max(1, int(pixel_limit))

    util = (
        _clamp_float(float(target_utilization), 0.5, 0.99)
        if target_utilization is not None
        else _clamp_float(
            _env_float("PI3X_VRAM_TARGET_UTILIZATION", _PI3X_TARGET_VRAM_UTILIZATION),
            0.5,
            0.99,
        )
    )
    model_mb = max(0, _env_int("PI3X_ESTIMATED_MODEL_MB", _PI3X_ESTIMATED_MODEL_MB))
    runtime_overhead_mb = max(0, _env_int("PI3X_RUNTIME_OVERHEAD_MB", _PI3X_RUNTIME_OVERHEAD_MB))
    pixels_per_mb = max(1, _env_int("PI3X_FRAME_PIXELS_PER_MB", _PI3X_FRAME_PIXELS_PER_MB))

    inventory = get_gpu_inventory()
    if not inventory:
        return {
            "requested_frames": req_frames,
            "pixel_limit": px_limit,
            "auto_target_frames": req_frames,
            "auto_reduced": False,
            "target_vram_utilization": util,
            "gpu_name": None,
            "gpu_total_mb": None,
            "gpu_used_mb": None,
            "gpu_free_mb": None,
            "target_used_mb": None,
            "frame_pixel_budget": None,
            "estimated_input_mb": None,
            "predicted_used_mb": None,
            "predicted_used_pct": None,
            "reason": "nvidia-smi unavailable",
        }

    gpu = inventory[0]
    total_mb = int(gpu.get("memory_total_mb") or 0)
    used_mb = int(gpu.get("memory_used_mb") or 0)
    free_mb = int(gpu.get("memory_free_mb") or max(total_mb - used_mb, 0))

    if total_mb <= 0:
        return {
            "requested_frames": req_frames,
            "pixel_limit": px_limit,
            "auto_target_frames": req_frames,
            "auto_reduced": False,
            "target_vram_utilization": util,
            "gpu_name": str(gpu.get("name") or ""),
            "gpu_total_mb": total_mb,
            "gpu_used_mb": used_mb,
            "gpu_free_mb": free_mb,
            "target_used_mb": None,
            "frame_pixel_budget": None,
            "estimated_input_mb": None,
            "predicted_used_mb": None,
            "predicted_used_pct": None,
            "reason": "invalid gpu memory.total from nvidia-smi",
        }

    target_used_mb = int(total_mb * util)
    available_for_pi3x_mb = max(target_used_mb - used_mb, 0)
    input_budget_mb = max(available_for_pi3x_mb - model_mb - runtime_overhead_mb, 0)
    frame_pixel_budget = max(2 * px_limit, input_budget_mb * pixels_per_mb)

    auto_frames = max(2, frame_pixel_budget // px_limit)
    auto_frames = min(req_frames, auto_frames)
    estimated_input_mb = int((auto_frames * px_limit) / pixels_per_mb)
    predicted_used_mb = min(
        total_mb,
        used_mb + model_mb + runtime_overhead_mb + estimated_input_mb,
    )
    predicted_used_pct = (100.0 * predicted_used_mb / total_mb) if total_mb > 0 else None

    return {
        "requested_frames": req_frames,
        "pixel_limit": px_limit,
        "auto_target_frames": int(auto_frames),
        "auto_reduced": bool(auto_frames < req_frames),
        "target_vram_utilization": util,
        "gpu_name": str(gpu.get("name") or ""),
        "gpu_total_mb": total_mb,
        "gpu_used_mb": used_mb,
        "gpu_free_mb": free_mb,
        "target_used_mb": target_used_mb,
        "frame_pixel_budget": int(frame_pixel_budget),
        "estimated_input_mb": estimated_input_mb,
        "predicted_used_mb": int(predicted_used_mb),
        "predicted_used_pct": float(predicted_used_pct) if predicted_used_pct is not None else None,
        "reason": "ok",
    }


def ensure_vram_available(
    min_free_mb: int = _VRAM_GATE_MIN_FREE_MB,
    stage_name: str = "",
    max_retries: int = 3,
) -> None:
    """Verify VRAM availability, retrying gc+empty_cache if insufficient.

    Logs a WARNING if free VRAM stays below *min_free_mb* after retries,
    but never blocks the pipeline.
    """
    import torch

    label = f" [{stage_name}]" if stage_name else ""

    for attempt in range(max_retries):
        free = get_free_vram_mb()
        if free is None:
            print(f"VRAM gate{label}: nvidia-smi unavailable, skipping check")
            return
        if free >= min_free_mb:
            print(f"VRAM gate{label}: {free}MB free >= {min_free_mb}MB required ✓")
            return

        print(
            f"VRAM gate{label}: {free}MB free < {min_free_mb}MB required "
            f"(attempt {attempt + 1}/{max_retries}), forcing cleanup..."
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(1)

    # Final check — warn but do not block
    free = get_free_vram_mb()
    if free is not None and free < min_free_mb:
        print(
            f"WARNING: VRAM gate{label}: only {free}MB free after "
            f"{max_retries} retries (wanted {min_free_mb}MB). Continuing anyway."
        )


def prepare_for_jax():
    """Set environment variables for JAX subprocess execution.

    Must be called before spawning a JAX subprocess (DiffCD).
    Prevents JAX from pre-allocating all GPU memory.
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"


def offload_module(module, target: str = "cpu") -> None:
    """Move a torch.nn.Module (or iterable of modules) to *target* device.

    Accepts a single module or an iterable (ModuleList, list, etc.).
    After moving, calls ``torch.cuda.empty_cache()`` to release freed VRAM.
    """
    import torch

    modules = [module] if hasattr(module, "parameters") else list(module)
    for m in modules:
        m.to(target)

    if target == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_vram(stage_name: str = ""):
    """Log current GPU memory usage via nvidia-smi."""
    inventory = get_gpu_inventory()
    if not inventory:
        return

    first = inventory[0]
    used = first.get("memory_used_mb")
    total = first.get("memory_total_mb")
    free = first.get("memory_free_mb")
    if not all(isinstance(v, int) for v in [used, total, free]):
        return
    label = f" [{stage_name}]" if stage_name else ""
    print(f"VRAM{label}: {used}MB / {total}MB (free: {free}MB)", flush=True)
