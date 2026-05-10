"""Compose a LiTo-ready 518×518 RGBA letterbox from a frame + SAM2 mask.

LiTo's `preprocess_image()` (apple/ml-lito/demos/lito/fastapi_lito_demo.py)
expects a square RGBA where the alpha channel is the foreground mask. The
pipeline is:

  1. SAM2 mask → alpha (0/255)
  2. tight bbox of α≥1 region with `fill_ratio` padding (≈ 0.8 of side)
  3. centred letterbox (transparent background) onto a 518×518 canvas
  4. Lanczos resize to exactly 518×518

The output PNG is saved to `lito_workspace/selected_frame.png` and handed
to the LiTo subprocess bridge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_rgb_and_mask(frame_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB and binary mask, resize mask to frame shape if needed."""
    from PIL import Image

    rgb = np.array(Image.open(frame_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != rgb.shape[:2]:
        mask = np.array(
            Image.fromarray(mask).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST
            )
        )
    return rgb, mask


def _bbox_with_padding(
    mask: np.ndarray, fill_ratio: float = 0.8
) -> tuple[int, int, int, int]:
    """Square crop bbox around mask with `fill_ratio` of side as content."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("mask_compositor: empty mask, cannot compute bbox")
    xmin, ymin = int(xs.min()), int(ys.min())
    xmax, ymax = int(xs.max()) + 1, int(ys.max()) + 1
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    side = max(xmax - xmin, ymax - ymin) / max(fill_ratio, 1e-6)
    half = side / 2.0
    h, w = mask.shape
    x0 = int(round(cx - half))
    x1 = int(round(cx + half))
    y0 = int(round(cy - half))
    y1 = int(round(cy + half))
    # Clamp to frame and re-square — we lose some padding at edges but the
    # mask region stays inside.
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    return x0, y0, x1, y1


def compose_lito_input(
    frame_path: str,
    mask_path: str,
    out_path: str,
    *,
    resolution: int = 518,
    fill_ratio: float = 0.8,
) -> dict:
    """Build a `resolution × resolution` RGBA letterbox at out_path.

    Returns metadata including original frame size, bbox, padding offsets,
    and the affine transform from output→original-frame pixel coordinates
    (needed later to project LiTo outputs back into COLMAP world frame).
    """
    from PIL import Image

    rgb, mask = _load_rgb_and_mask(frame_path, mask_path)
    src_h, src_w = rgb.shape[:2]
    x0, y0, x1, y1 = _bbox_with_padding(mask, fill_ratio=fill_ratio)
    crop_w = x1 - x0
    crop_h = y1 - y0

    # Build RGBA crop. Background pixels (mask == 0) inside the crop go
    # transparent; LiTo's preprocess uses alpha, not the RGB they cover.
    rgba_crop = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
    rgba_crop[..., :3] = rgb[y0:y1, x0:x1]
    rgba_crop[..., 3] = mask[y0:y1, x0:x1]

    # Square letterbox if crop is non-square (clamped to image edge can leave it rectangular).
    side = max(crop_w, crop_h)
    pad_x = (side - crop_w) // 2
    pad_y = (side - crop_h) // 2
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    canvas[pad_y:pad_y + crop_h, pad_x:pad_x + crop_w] = rgba_crop

    # Resize to target resolution. PIL Lanczos handles RGBA cleanly.
    out_img = Image.fromarray(canvas).resize(
        (resolution, resolution), Image.LANCZOS
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)

    scale = resolution / max(side, 1)
    return {
        "src_width": int(src_w),
        "src_height": int(src_h),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "letterbox_pad_x": int(pad_x),
        "letterbox_pad_y": int(pad_y),
        "letterbox_side": int(side),
        "resolution": int(resolution),
        "scale_to_canvas": float(scale),
        "fill_ratio": float(fill_ratio),
    }
