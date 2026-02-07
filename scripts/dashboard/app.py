"""FastAPI web dashboard for im2pc-pipeline.

Entry point: uvicorn scripts.dashboard.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from scripts.dashboard.log_capture import LogBroadcaster
from scripts.dashboard.pipeline_runner import broadcast, run_pipeline
from scripts.dashboard.sam2_service import SAM2Service
from scripts.dashboard.state import PipelineConfig, PipelineSession

# ── Globals ───────────────────────────────────────────────────────

app = FastAPI(title="im2pc-pipeline Dashboard")
session = PipelineSession()
sam2_service = SAM2Service()
log_broadcaster: LogBroadcaster | None = None

INPUT_DIR = os.environ.get("INPUT_DIR", "/data/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output")

STATIC_DIR = Path(__file__).parent / "static"


# ── Lifecycle ─────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    global log_broadcaster
    loop = asyncio.get_event_loop()
    log_broadcaster = LogBroadcaster(loop)
    log_broadcaster.install()
    asyncio.create_task(log_broadcaster.drain(session.ws_clients))


@app.on_event("shutdown")
async def _shutdown() -> None:
    if log_broadcaster:
        log_broadcaster.uninstall()
    sam2_service.release()


# ── Static files ──────────────────────────────────────────────────

# Serve index.html at root
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# Serve other static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── WebSocket ─────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session.ws_clients.append(ws)
    try:
        # Send current status on connect
        await ws.send_text(json.dumps({
            "type": "status",
            **session.to_status_dict(),
        }))
        # Keep alive — read messages (client pings / future commands)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in session.ws_clients:
            session.ws_clients.remove(ws)


# ── Pipeline API ──────────────────────────────────────────────────

@app.get("/api/pipeline/status")
async def pipeline_status():
    return JSONResponse(session.to_status_dict())


@app.get("/api/pipeline/videos")
async def pipeline_videos():
    """List video files in /data/input/."""
    input_dir = Path(INPUT_DIR)
    videos: list[dict[str, Any]] = []
    if input_dir.is_dir():
        for f in sorted(input_dir.iterdir()):
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                videos.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                })
    return JSONResponse({"videos": videos})


@app.get("/api/pipeline/video-info")
async def pipeline_video_info(path: str):
    """Return metadata (fps, frames, resolution, duration) for a video file."""
    target = Path(path).resolve()
    input_dir = Path(INPUT_DIR).resolve()
    try:
        target.relative_to(input_dir)
    except ValueError:
        return JSONResponse({"error": "Access denied"}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    def _probe(p: str) -> dict:
        import cv2
        from stage_extract_frames import _detect_rotation, _normalize_fps
        cap = cv2.VideoCapture(p)
        try:
            raw_fps = cap.get(cv2.CAP_PROP_FPS) or 0
            fps = _normalize_fps(raw_fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = total_frames / raw_fps if raw_fps > 0 else 0
            frame_interval = max(1, int(round(fps / 2)))
            max_frames = (
                max(1, (total_frames + frame_interval - 1) // frame_interval)
                if total_frames > 0
                else 0
            )
            rotation = _detect_rotation(cap, p)
            # Swap width/height for 90°/270° rotation
            if rotation in (90, 270):
                width, height = height, width
            return {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": round(duration, 2),
                "rotation": rotation,
                "suggested_frame_interval": frame_interval,
                "suggested_max_frames": max_frames,
            }
        finally:
            cap.release()

    try:
        info = await asyncio.to_thread(_probe, str(target))
        return JSONResponse(info)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/pipeline/start")
async def pipeline_start(body: dict | None = None):
    if session.running:
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)

    session.reset()

    # Build config from body + env defaults
    raw = body or {}
    cfg = PipelineConfig(
        video_path=raw.get("video_path", ""),
        output_dir=raw.get("output_dir", OUTPUT_DIR),
        frame_interval=int(raw.get("frame_interval", os.environ.get("FRAME_INTERVAL", 10))),
        max_frames=int(raw.get("max_frames", os.environ.get("MAX_FRAMES", 50))),
        pixel_limit=int(raw.get("pixel_limit", os.environ.get("PIXEL_LIMIT", 255000))),
        confidence_threshold=float(raw.get("confidence_threshold", os.environ.get("CONFIDENCE_THRESHOLD", 0.1))),
        edge_rtol=float(raw.get("edge_rtol", os.environ.get("EDGE_RTOL", 0.03))),
        sam2_model=raw.get("sam2_model", os.environ.get("SAM2_MODEL", "large")),
        diffcd_batch_size=int(raw.get("diffcd_batch_size", os.environ.get("DIFFCD_BATCH_SIZE", 3000))),
        diffcd_n_batches=int(raw.get("diffcd_n_batches", os.environ.get("DIFFCD_N_BATCHES", 25000))),
        diffcd_resolution=int(raw.get("diffcd_resolution", os.environ.get("DIFFCD_RESOLUTION", 384))),
        texture_size=int(raw.get("texture_size", os.environ.get("TEXTURE_SIZE", 2048))),
    )

    if not cfg.video_path:
        return JSONResponse({"error": "video_path is required"}, status_code=400)

    session.config = cfg

    # Set env vars that stages read directly
    os.environ["DIFFCD_BATCH_SIZE"] = str(cfg.diffcd_batch_size)
    os.environ["DIFFCD_N_BATCHES"] = str(cfg.diffcd_n_batches)
    os.environ["DIFFCD_RESOLUTION"] = str(cfg.diffcd_resolution)

    # Launch pipeline as background task
    session._task = asyncio.create_task(run_pipeline(session, sam2_service))

    return JSONResponse({"status": "started"})


@app.post("/api/pipeline/cancel")
async def pipeline_cancel():
    if not session.running:
        return JSONResponse({"error": "No pipeline running"}, status_code=409)
    session.cancelled = True
    # If waiting for SAM2 or Pi3X confirmation/approval, unblock it
    session.sam2_confirm_event.set()
    session.sam2_approve_event.set()
    session.pi3x_approve_event.set()
    return JSONResponse({"status": "cancelling"})


# ── Pi3X API ──────────────────────────────────────────────────────

@app.post("/api/pi3x/approve")
async def pi3x_approve():
    session.pi3x_approve_event.set()
    return JSONResponse({"status": "approved"})


# ── SAM2 API ──────────────────────────────────────────────────────

@app.post("/api/sam2/click")
async def sam2_click(body: dict):
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    norm_x = float(body["x"])
    norm_y = float(body["y"])
    label = int(body.get("label", 1))  # 1=positive, 0=negative
    png_bytes = await asyncio.to_thread(sam2_service.add_click, norm_x, norm_y, label)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/sam2/undo")
async def sam2_undo():
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    png_bytes = await asyncio.to_thread(sam2_service.undo_click)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/sam2/clear")
async def sam2_clear():
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    png_bytes = await asyncio.to_thread(sam2_service.clear_clicks)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/sam2/confirm")
async def sam2_confirm():
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    session.sam2_confirm_event.set()
    return JSONResponse({"status": "confirming"})


@app.post("/api/sam2/approve")
async def sam2_approve():
    session.sam2_approved = True
    session.sam2_approve_event.set()
    return JSONResponse({"status": "approved"})


@app.post("/api/sam2/redo")
async def sam2_redo():
    session.sam2_approved = False
    session.sam2_approve_event.set()
    return JSONResponse({"status": "redo"})


@app.get("/api/sam2/frame/{idx}")
async def sam2_frame(idx: int):
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    try:
        jpeg_bytes = await asyncio.to_thread(sam2_service.get_frame_jpeg, idx)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except IndexError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.get("/api/verification/frame/{idx}")
async def verification_frame(idx: int):
    """Composite frame + green mask overlay at 40% opacity for verification."""
    out = Path(session.config.output_dir or OUTPUT_DIR)
    frame_path = out / "frames" / f"{idx:05d}.jpg"
    mask_path = out / "masks" / f"{idx:05d}.png"

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


@app.get("/api/sam2/mask/{idx}")
async def sam2_mask(idx: int):
    if not sam2_service.initialized:
        return JSONResponse({"error": "SAM2 not ready"}, status_code=409)
    png_bytes = await asyncio.to_thread(sam2_service.get_mask_png, idx)
    if png_bytes is None:
        return JSONResponse({"error": "Mask not found"}, status_code=404)
    return Response(content=png_bytes, media_type="image/png")


# ── Preview API ───────────────────────────────────────────────────

@app.get("/api/preview/outputs")
async def preview_outputs():
    """List output files available for preview."""
    out = Path(session.config.output_dir or OUTPUT_DIR)
    files: list[dict] = []
    if out.is_dir():
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.suffix.lower() in (
                ".ply", ".obj", ".mtl", ".png", ".jpg", ".json",
            ):
                rel = str(f.relative_to(out))
                files.append({
                    "path": rel,
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                    "ext": f.suffix.lower(),
                })
    return JSONResponse({"files": files})


@app.get("/api/preview/file/{path:path}")
async def preview_file(path: str):
    """Serve an output file by relative path."""
    out = Path(session.config.output_dir or OUTPUT_DIR)
    target = (out / path).resolve()
    # Security: ensure target is within output directory
    if not str(target).startswith(str(out.resolve())):
        return JSONResponse({"error": "Access denied"}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    media_types = {
        ".ply": "application/octet-stream",
        ".obj": "text/plain",
        ".mtl": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".json": "application/json",
    }
    mt = media_types.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(str(target), media_type=mt)


# ── VRAM endpoint ─────────────────────────────────────────────────

@app.get("/api/vram")
async def vram_info():
    """Return current VRAM usage."""
    try:
        from vram_utils import get_free_vram_mb
        free = get_free_vram_mb()
        return JSONResponse({"free_mb": free})
    except Exception:
        return JSONResponse({"free_mb": None})
