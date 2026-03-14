"""SAM2 interactive segmentation routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from scripts.dashboard.dependencies import get_state

router = APIRouter()


@router.post("/api/sam2/click")
async def sam2_click(body: dict):
    sam2_service = get_state().sam2_service
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    norm_x = float(body["x"])
    norm_y = float(body["y"])
    label = int(body.get("label", 1))  # 1=positive, 0=negative
    png_bytes = await asyncio.to_thread(sam2_service.add_click, norm_x, norm_y, label)
    return Response(content=png_bytes, media_type="image/png")


@router.post("/api/sam2/undo")
async def sam2_undo():
    sam2_service = get_state().sam2_service
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    png_bytes = await asyncio.to_thread(sam2_service.undo_click)
    return Response(content=png_bytes, media_type="image/png")


@router.post("/api/sam2/clear")
async def sam2_clear():
    sam2_service = get_state().sam2_service
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    png_bytes = await asyncio.to_thread(sam2_service.clear_clicks)
    return Response(content=png_bytes, media_type="image/png")


@router.post("/api/sam2/confirm")
async def sam2_confirm():
    state = get_state()
    if not state.sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    state.session.sam2_confirm_event.set()
    return JSONResponse({"status": "confirming"})


@router.post("/api/sam2/mode")
async def sam2_mode(body: dict):
    sam2_service = get_state().sam2_service
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    mode = str(body.get("mode", "")).strip()
    if mode not in ("object", "ground"):
        return JSONResponse({"error": "mode must be 'object' or 'ground'"}, status_code=400)
    sam2_service.set_mode(mode)
    return JSONResponse({"status": "ok", "mode": mode})


@router.get("/api/sam2/mode")
async def sam2_get_mode():
    return JSONResponse({"mode": get_state().sam2_service.segmentation_mode})


@router.post("/api/sam2/skip-ground")
async def sam2_skip_ground():
    get_state().session.sam2_ground_skip_event.set()
    return JSONResponse({"status": "skipping_ground"})


@router.post("/api/sam2/approve")
async def sam2_approve():
    session = get_state().session
    session.sam2_approved = True
    session.sam2_approve_event.set()
    return JSONResponse({"status": "approved"})


@router.post("/api/sam2/redo")
async def sam2_redo():
    session = get_state().session
    session.sam2_approved = False
    session.sam2_approve_event.set()
    return JSONResponse({"status": "redo"})


@router.get("/api/sam2/frame/{idx}")
async def sam2_frame(idx: int):
    sam2_service = get_state().sam2_service
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    try:
        jpeg_bytes = await asyncio.to_thread(sam2_service.get_frame_jpeg, idx)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except IndexError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@router.get("/api/sam2/mask/{idx}")
async def sam2_mask(idx: int):
    sam2_service = get_state().sam2_service
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    png_bytes = await asyncio.to_thread(sam2_service.get_mask_png, idx)
    if png_bytes is None:
        return JSONResponse({"error": "Mask not found"}, status_code=404)
    return Response(content=png_bytes, media_type="image/png")
