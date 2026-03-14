"""Progress reporting and memory utilities for texture baking."""

from typing import Callable

from scripts.config_defaults import _TEXTURE_MEM_FALLBACK_MB

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
    return _TEXTURE_MEM_FALLBACK_MB  # conservative fallback
