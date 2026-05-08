"""Frame verification overlay routes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from scripts.dashboard.dependencies import get_state
from scripts.output_layout import frames_dir, masks_dir, masks_ground_dir

router = APIRouter()


@router.get("/api/verification/frame/{idx}")
async def verification_frame(idx: int):
    """Composite frame + green mask overlay at 40% opacity for verification."""
    out = get_state().active_output_dir()
    frame_path = frames_dir(out) / f"{idx:05d}.jpg"
    mask_path = masks_dir(out) / f"{idx:05d}.png"

    if not frame_path.is_file():
        return JSONResponse({"error": f"Frame {idx} not found"}, status_code=404)

    def _composite(fp: str, mp: str) -> bytes:
        import cv2
        import numpy as np
        frame = cv2.imread(fp)
        if frame is None:
            raise FileNotFoundError(f"Cannot read frame: {fp}")
        if Path(mp).is_file():
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize mask to match frame if needed
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                # Green overlay at 40% opacity where mask > 0
                overlay = frame.copy()
                overlay[mask > 0] = (
                    overlay[mask > 0] * 0.6 + np.array([0, 180, 0], dtype=np.float64) * 0.4
                ).astype(np.uint8)
                frame = overlay
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    try:
        jpeg_bytes = await asyncio.to_thread(_composite, str(frame_path), str(mask_path))
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/verification/ground-frame/{idx}")
async def verification_ground_frame(idx: int):
    """Composite frame + orange ground mask overlay for verification."""
    out = get_state().active_output_dir()
    frame_path = frames_dir(out) / f"{idx:05d}.jpg"
    mask_path = masks_ground_dir(out) / f"{idx:05d}.png"

    if not frame_path.is_file():
        return JSONResponse({"error": f"Frame {idx} not found"}, status_code=404)

    def _composite(fp: str, mp: str) -> bytes:
        import cv2
        import numpy as np
        frame = cv2.imread(fp)
        if frame is None:
            raise FileNotFoundError(f"Cannot read frame: {fp}")
        if Path(mp).is_file():
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                overlay = frame.copy()
                # Orange overlay (BGR: 0, 165, 255) at 35% opacity
                overlay[mask > 0] = (
                    overlay[mask > 0] * 0.65 + np.array([0, 165, 255], dtype=np.float64) * 0.35
                ).astype(np.uint8)
                frame = overlay
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    try:
        jpeg_bytes = await asyncio.to_thread(_composite, str(frame_path), str(mask_path))
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
