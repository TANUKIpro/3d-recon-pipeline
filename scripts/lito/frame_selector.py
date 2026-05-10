"""Pick the single best frame to feed to LiTo image-to-3D inference.

LiTo consumes one RGBA image and predicts an entire 3D Gaussian field, so
frame choice dominates output quality. The compound score combines:

* SAM2 mask coverage     — how much of the frame the object occupies
* COLMAP triangulation   — how many sparse 3D points are anchored to this view
* Laplacian sharpness    — focus quality (motion blur penalty)

Quality gates run before scoring; gate failures raise ValueError so the
caller can decide whether to fall back to gs2mesh.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


_FRAME_NAME_RE = re.compile(r"^(\d+)$")


@dataclass(frozen=True)
class FrameMetric:
    """Per-frame measurements feeding the compound score."""

    frame_index: int
    frame_path: str
    mask_path: str
    mask_coverage: float
    sharpness: float
    triangulation_count: int = 0
    bbox: Optional[tuple[int, int, int, int]] = None  # (xmin, ymin, xmax, ymax)
    mask_components: int = 0


@dataclass(frozen=True)
class GateOutcome:
    """Quality-gate result for one frame."""

    frame_index: int
    passed: bool
    reasons: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class FrameSelection:
    """Final output of select_best_frame."""

    frame_index: int
    frame_path: str
    mask_path: str
    score: float
    breakdown: dict
    metric: FrameMetric


def _list_frame_pairs(frames_dir: str, mask_dir: str) -> list[tuple[int, Path, Path]]:
    """Return (index, frame_path, mask_path) triples for every paired frame."""
    frames = Path(frames_dir)
    masks = Path(mask_dir)
    pairs: list[tuple[int, Path, Path]] = []
    for frame_path in sorted(frames.iterdir()):
        if frame_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = _FRAME_NAME_RE.match(frame_path.stem)
        if not m:
            continue
        idx = int(m.group(1))
        mask_path = masks / f"{frame_path.stem}.png"
        if not mask_path.exists():
            continue
        pairs.append((idx, frame_path, mask_path))
    return pairs


def _laplacian_variance(rgb: np.ndarray) -> float:
    """Sharpness proxy: Laplacian variance on the green channel.

    Avoids the cv2 dependency by implementing the discrete Laplacian as a
    pure-numpy convolution. Higher values = sharper.
    """
    if rgb.ndim != 3:
        return 0.0
    g = rgb[..., 1].astype(np.float32)
    lap = (
        4.0 * g[1:-1, 1:-1]
        - g[:-2, 1:-1]
        - g[2:, 1:-1]
        - g[1:-1, :-2]
        - g[1:-1, 2:]
    )
    return float(lap.var())


def _mask_components(mask: np.ndarray) -> int:
    """Count connected components of a binary mask via flood fill (numpy only)."""
    if mask.ndim != 2:
        return 0
    visited = np.zeros_like(mask, dtype=bool)
    fg = mask > 0
    count = 0
    h, w = mask.shape
    stack: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not fg[y, x] or visited[y, x]:
                continue
            count += 1
            stack.append((y, x))
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or not fg[cy, cx]:
                    continue
                visited[cy, cx] = True
                stack.extend(((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)))
    return count


def _mask_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def measure_frame(
    frame_index: int,
    frame_path: Path,
    mask_path: Path,
) -> FrameMetric:
    """Load a frame + mask and compute per-frame metrics."""
    from PIL import Image

    rgb = np.array(Image.open(str(frame_path)).convert("RGB"))
    mask = np.array(Image.open(str(mask_path)).convert("L"))
    if mask.shape != rgb.shape[:2]:
        # Resize mask to frame using nearest-neighbour through PIL.
        mask = np.array(
            Image.fromarray(mask).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST
            )
        )

    coverage = float((mask > 0).mean())
    sharpness = _laplacian_variance(rgb)
    bbox = _mask_bbox(mask)
    components = _mask_components(mask)
    return FrameMetric(
        frame_index=frame_index,
        frame_path=str(frame_path),
        mask_path=str(mask_path),
        mask_coverage=coverage,
        sharpness=sharpness,
        triangulation_count=0,
        bbox=bbox,
        mask_components=components,
    )


def evaluate_gate(metric: FrameMetric, cfg) -> GateOutcome:
    """Run quality gates against one frame's metrics."""
    reasons: list[str] = []
    if not (cfg.gate_min_mask_coverage <= metric.mask_coverage <= cfg.gate_max_mask_coverage):
        reasons.append(
            f"mask_coverage={metric.mask_coverage:.3f} outside "
            f"[{cfg.gate_min_mask_coverage:.2f}, {cfg.gate_max_mask_coverage:.2f}]"
        )
    if metric.mask_components > cfg.gate_max_connected_components:
        reasons.append(
            f"connected_components={metric.mask_components} > "
            f"{cfg.gate_max_connected_components}"
        )
    elif metric.mask_components == 0:
        reasons.append("connected_components=0 (no foreground)")
    if metric.bbox is not None:
        xmin, ymin, xmax, ymax = metric.bbox
        short = min(xmax - xmin, ymax - ymin)
        if short < cfg.gate_min_bbox_short_px:
            reasons.append(
                f"bbox_short={short}px < {cfg.gate_min_bbox_short_px}px"
            )
    else:
        reasons.append("bbox missing (empty mask)")
    return GateOutcome(
        frame_index=metric.frame_index,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _normalise_weights(cfg) -> tuple[float, float, float]:
    """Re-normalise weights to sum to 1; if all zero, fall back to mask-only."""
    w = (cfg.weight_mask_coverage, cfg.weight_triangulation, cfg.weight_sharpness)
    total = sum(max(0.0, x) for x in w)
    if total <= 0.0:
        return (1.0, 0.0, 0.0)
    return tuple(max(0.0, x) / total for x in w)  # type: ignore[return-value]


def _score_frame(
    metric: FrameMetric,
    *,
    sharpness_norm: float,
    triangulation_norm: float,
    weights: tuple[float, float, float],
) -> tuple[float, dict]:
    w_mask, w_tri, w_sharp = weights
    sharp_term = (
        metric.sharpness / sharpness_norm if sharpness_norm > 0 else 0.0
    )
    tri_term = (
        metric.triangulation_count / triangulation_norm
        if triangulation_norm > 0
        else 0.0
    )
    score = (
        w_mask * metric.mask_coverage
        + w_tri * tri_term
        + w_sharp * sharp_term
    )
    breakdown = {
        "mask_coverage": metric.mask_coverage,
        "sharpness_norm": sharp_term,
        "triangulation_norm": tri_term,
        "weight_mask_coverage": w_mask,
        "weight_triangulation": w_tri,
        "weight_sharpness": w_sharp,
    }
    return float(score), breakdown


def select_best_frame(
    frames_dir: str,
    mask_dir: str,
    cfg=None,
) -> FrameSelection:
    """Pick the single best frame for LiTo image-to-3D inference.

    Pipeline:
      1. Enumerate (frame, mask) pairs and measure per-frame metrics.
      2. Run quality gates; gather all gate failures.
      3. If manual_index is set, force-select that frame (gates still run
         but failures are reported as warnings, not raised).
      4. Otherwise pick the gate-passing frame with the highest compound
         score. Raise ValueError if no frame passes.
    """
    if cfg is None:
        from scripts.lito.config import load_frame_selection_config

        cfg = load_frame_selection_config()

    pairs = _list_frame_pairs(frames_dir, mask_dir)
    if not pairs:
        raise ValueError(
            f"lito frame selection: no paired frames found "
            f"(frames_dir={frames_dir}, mask_dir={mask_dir})"
        )

    metrics = [measure_frame(idx, fp, mp) for idx, fp, mp in pairs]
    gates = [evaluate_gate(m, cfg) for m in metrics]
    sharp_max = max((m.sharpness for m in metrics), default=0.0)
    tri_max = max((m.triangulation_count for m in metrics), default=0)
    weights = _normalise_weights(cfg)

    if cfg.manual_index is not None:
        target = next(
            (m for m in metrics if m.frame_index == cfg.manual_index),
            None,
        )
        if target is None:
            raise ValueError(
                f"lito frame selection: manual_index={cfg.manual_index} "
                f"not found among {len(metrics)} frames"
            )
        gate = next(g for g in gates if g.frame_index == target.frame_index)
        if not gate.passed:
            print(
                f"[lito] WARNING: manual frame {target.frame_index} fails gates: "
                f"{', '.join(gate.reasons)}"
            )
        score, breakdown = _score_frame(
            target,
            sharpness_norm=sharp_max,
            triangulation_norm=tri_max,
            weights=weights,
        )
        return FrameSelection(
            frame_index=target.frame_index,
            frame_path=target.frame_path,
            mask_path=target.mask_path,
            score=score,
            breakdown=breakdown,
            metric=target,
        )

    eligible = [
        (m, _score_frame(
            m,
            sharpness_norm=sharp_max,
            triangulation_norm=tri_max,
            weights=weights,
        ))
        for m, g in zip(metrics, gates)
        if g.passed
    ]
    if not eligible:
        gate_summary = "; ".join(
            f"#{g.frame_index}:{','.join(g.reasons)}"
            for g in gates if not g.passed
        )
        raise ValueError(
            f"lito frame selection: no frame passes quality gates. "
            f"({len(metrics)} candidates, all failed). Gate failures: {gate_summary}"
        )

    eligible.sort(key=lambda item: item[1][0], reverse=True)
    chosen, (score, breakdown) = eligible[0]
    return FrameSelection(
        frame_index=chosen.frame_index,
        frame_path=chosen.frame_path,
        mask_path=chosen.mask_path,
        score=score,
        breakdown=breakdown,
        metric=chosen,
    )
