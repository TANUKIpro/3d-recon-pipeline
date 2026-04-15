"""Auto-detected VRAM tier used to pick TSDF device at runtime.

The GS2Mesh GPU TSDF stage (``scripts/gpu_tsdf.py``) allocates roughly 8 GB
directly on the GPU from Open3D's own memory pool at the default
``block_count=100_000``. On cards with less VRAM this is impossible to
satisfy without reducing reconstruction parameters. To keep the
parameters identical on low-VRAM hardware, we auto-detect a tier at
process start and relocate the VoxelBlockGrid to CPU when needed.

Tiers are hidden from the user-facing preset dropdown. A tier only
overrides ``tsdf_device`` on the final ``Gs2meshSettings``; every other
field comes from whichever public preset the user picked.
"""

from __future__ import annotations

import functools
import os
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.gs2mesh_config import Gs2meshSettings


TIER_ORDER: tuple[str, ...] = ("t18", "t16", "t12", "t8", "t_other")

# Lower bound (inclusive) in MB for each tier. Order matters — walked
# top-down so the first matching threshold wins. ``t_other`` is the
# implicit fallback when no threshold matches.
_THRESHOLDS_MB: tuple[tuple[str, int], ...] = (
    ("t18", 17000),
    ("t16", 14000),
    ("t12", 10000),
    ("t8", 7000),
)

TIER_HEARTS: dict[str, str] = {
    "t18": "\u2764\u2764\u2764",
    "t16": "\u2764\u2764",
    "t12": "\u2764",
    "t8": "\u2764",
    "t_other": "\u2764",
}

# Per-tier overrides layered on top of the user's chosen preset. Only
# ``tsdf_device`` differs today; the shape is kept open so future
# low-VRAM tweaks can be added without touching call sites.
VRAM_TIER_OVERRIDES: dict[str, dict[str, object]] = {
    "t18": {},
    "t16": {},
    "t12": {},
    "t8": {"tsdf_device": "CPU:0"},
    "t_other": {"tsdf_device": "CPU:0"},
}

_ENV_VAR = "VRAM_TIER_OVERRIDE"


@functools.lru_cache(maxsize=1)
def detect_tier() -> str:
    """Return the VRAM tier for the current process.

    Honors ``VRAM_TIER_OVERRIDE`` env var (useful for testing and for
    CLI users who want to force CPU TSDF on a larger GPU). Falls back to
    reading total VRAM via nvidia-smi; returns ``t_other`` when no GPU
    is present.
    """

    forced = os.environ.get(_ENV_VAR, "").strip().lower()
    if forced in TIER_ORDER:
        return forced

    from scripts.vram_utils import get_total_vram_mb

    total = get_total_vram_mb()
    if total is None:
        return "t_other"
    for tier, threshold in _THRESHOLDS_MB:
        if total >= threshold:
            return tier
    return "t_other"


def _total_vram_mb_safe() -> int | None:
    try:
        from scripts.vram_utils import get_total_vram_mb

        return get_total_vram_mb()
    except Exception:
        return None


def apply_tier_override(settings: "Gs2meshSettings") -> "Gs2meshSettings":
    """Layer the current tier's overrides on top of the given settings.

    A no-op for upper tiers (empty override dict).
    """

    overrides = VRAM_TIER_OVERRIDES[detect_tier()]
    if not overrides:
        return settings
    return replace(settings, **overrides)


def uses_cpu_tsdf(tier_or_device: "str | Gs2meshSettings | None" = None) -> bool:
    """Return True when the effective TSDF device is CPU.

    Accepts a tier name, a settings object, a raw device string, or
    ``None`` (in which case the detected tier is used).
    """

    if tier_or_device is None:
        tier = detect_tier()
        overrides = VRAM_TIER_OVERRIDES[tier]
        device = str(overrides.get("tsdf_device", "CUDA:0"))
    elif isinstance(tier_or_device, str):
        if tier_or_device in VRAM_TIER_OVERRIDES:
            device = str(
                VRAM_TIER_OVERRIDES[tier_or_device].get("tsdf_device", "CUDA:0")
            )
        else:
            device = tier_or_device
    else:
        device = getattr(tier_or_device, "tsdf_device", "CUDA:0")
    return device.lower().startswith("cpu")


def tier_info() -> dict[str, object]:
    """Return a JSON-serialisable snapshot for the dashboard header."""

    tier = detect_tier()
    overrides = VRAM_TIER_OVERRIDES[tier]
    device = str(overrides.get("tsdf_device", "CUDA:0"))
    return {
        "tier": tier,
        "hearts": TIER_HEARTS[tier],
        "tsdf_device": device,
        "total_vram_mb": _total_vram_mb_safe(),
    }
