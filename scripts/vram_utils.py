"""VRAM management utilities for sequential GPU pipeline stages."""

import gc
import os
import subprocess
import time


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
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def ensure_vram_available(
    min_free_mb: int = 12000,
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
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            used, total, free = result.stdout.strip().split(", ")
            label = f" [{stage_name}]" if stage_name else ""
            print(f"VRAM{label}: {used}MB / {total}MB (free: {free}MB)", flush=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
