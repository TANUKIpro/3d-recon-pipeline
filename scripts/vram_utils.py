"""VRAM management utilities for sequential GPU pipeline stages."""

import gc
import os
import subprocess
import time

from scripts.config_defaults import _VRAM_GATE_MIN_FREE_MB


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
