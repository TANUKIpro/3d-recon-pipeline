"""VRAM management utilities for sequential GPU pipeline stages."""

import gc
import os
import subprocess


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


def prepare_for_jax():
    """Set environment variables for JAX subprocess execution.

    Must be called before spawning a JAX subprocess (DiffCD).
    Prevents JAX from pre-allocating all GPU memory.
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"


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
