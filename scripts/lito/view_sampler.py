"""Sample evenly-distributed camera poses around the LiTo canonical object.

LiTo outputs 3D Gaussians in an object-centric canonical frame. To extract
a mesh we need depth maps from many synthesised views; this module
generates those poses on a sphere using Fibonacci lattice sampling.

The input view (the photo we fed to LiTo) is added as the first pose so
opacity-weighted TSDF integration can give it the highest confidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CanonicalView:
    """One synthesised view in the LiTo canonical frame."""

    index: int
    position: tuple[float, float, float]
    label: str
    is_input_view: bool
    H_c2w: np.ndarray  # (4, 4) camera-to-world
    K: np.ndarray  # (3, 3) intrinsic
    image_size: tuple[int, int]  # (height, width)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a camera-to-world matrix for an eye looking at target.

    Camera convention: +z forward (away from view), +y up — matches LiTo's
    `render_3dgs_gsplat` H_c2w argument.
    """
    forward = target - eye
    fn = np.linalg.norm(forward)
    if fn < 1e-12:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / fn
    right = np.cross(up, forward)
    rn = np.linalg.norm(right)
    if rn < 1e-12:
        # eye looks along +y; pick a different temporary up
        right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        rn = np.linalg.norm(right)
    right = right / max(rn, 1e-12)
    new_up = np.cross(forward, right)

    H = np.eye(4, dtype=np.float64)
    H[:3, 0] = right
    H[:3, 1] = new_up
    H[:3, 2] = forward
    H[:3, 3] = eye
    return H


def _build_intrinsic(image_size: tuple[int, int], fov_y_deg: float) -> np.ndarray:
    h, w = image_size
    fy = (h / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
    fx = fy  # square pixels at this fov
    cx = w / 2.0
    cy = h / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fibonacci_sphere(n: int, radius: float) -> list[tuple[float, float, float]]:
    """N evenly-spaced points on a sphere via the Fibonacci lattice."""
    if n <= 0:
        return []
    pts: list[tuple[float, float, float]] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i + 1.0) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts.append((x * radius, y * radius, z * radius))
    return pts


def sample_canonical_views(
    n: int,
    *,
    radius: float = 2.0,
    fov_y_deg: float = 40.0,
    image_size: tuple[int, int] = (512, 512),
    input_view_position: tuple[float, float, float] = (0.0, 0.0, 2.0),
) -> list[CanonicalView]:
    """Generate `n+1` views around origin (input view first, then `n` Fibonacci)."""
    target = np.zeros(3, dtype=np.float64)
    up_default = np.array([0.0, 1.0, 0.0])
    K = _build_intrinsic(image_size, fov_y_deg)
    views: list[CanonicalView] = []

    eye_in = np.array(input_view_position, dtype=np.float64)
    H_in = _look_at(eye_in, target, up_default)
    views.append(
        CanonicalView(
            index=0,
            position=tuple(input_view_position),
            label="input",
            is_input_view=True,
            H_c2w=H_in,
            K=K,
            image_size=image_size,
        )
    )

    for i, pos in enumerate(fibonacci_sphere(n, radius)):
        eye = np.array(pos, dtype=np.float64)
        H = _look_at(eye, target, up_default)
        views.append(
            CanonicalView(
                index=i + 1,
                position=pos,
                label=f"fib_{i:04d}",
                is_input_view=False,
                H_c2w=H,
                K=K,
                image_size=image_size,
            )
        )
    return views


def save_views_json(views: Sequence[CanonicalView], path: str | Path) -> None:
    """Serialise a view list to JSON for the bridge subprocess."""
    payload = {
        "views": [
            {
                "index": v.index,
                "position": list(v.position),
                "label": v.label,
                "is_input_view": v.is_input_view,
                "H_c2w": v.H_c2w.tolist(),
                "K": v.K.tolist(),
                "image_size": list(v.image_size),
            }
            for v in views
        ]
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_views_json(path: str | Path) -> list[CanonicalView]:
    """Inverse of save_views_json."""
    payload = json.loads(Path(path).read_text())
    out: list[CanonicalView] = []
    for v in payload["views"]:
        out.append(
            CanonicalView(
                index=int(v["index"]),
                position=tuple(v["position"]),
                label=str(v["label"]),
                is_input_view=bool(v["is_input_view"]),
                H_c2w=np.asarray(v["H_c2w"], dtype=np.float64),
                K=np.asarray(v["K"], dtype=np.float64),
                image_size=tuple(v["image_size"]),
            )
        )
    return out
