"""FastAPI web dashboard for im2pc-pipeline.

Entry point: uvicorn scripts.dashboard.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from scripts.dashboard.log_capture import LogBroadcaster
from scripts.dashboard.pipeline_runner import run_pipeline
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
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PREVIEW_FILE_EXTENSIONS = {".ply", ".obj", ".mtl", ".png", ".jpg", ".json"}
OBJECTS_SUBDIR = "objects"
OBJECT_META_FILE = "object_meta.json"
OBJECT_RESET_DIRS = ("frames", "masks", "diffcd")
OBJECT_RESET_FILES = (
    "object_full.ply",
    "pi3x_cache.npz",
    "camera_poses.json",
    "object.ply",
    "object_denoised.ply",
    "object_mesh.ply",
    "textured_mesh.obj",
    "textured_mesh.mtl",
    "texture.png",
    "intrinsics.json",
    "object_points.npy",
)
PRIMARY_ARTIFACT_PATHS = (
    "object_full.ply",
    "camera_poses.json",
    "object.ply",
    "object_denoised.ply",
    "object_mesh.ply",
    "textured_mesh.obj",
    "texture.png",
    "intrinsics.json",
)


def _utc_iso(ts: float | None = None) -> str:
    dt = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts is not None
        else datetime.now(timezone.utc)
    )
    return dt.isoformat(timespec="seconds")


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _sanitize_object_name(name: str) -> str:
    candidate = str(name or "").strip().replace("/", "-").replace("\\", "-")
    candidate = re.sub(r"\s+", "-", candidate)
    candidate = re.sub(r"[^\w.-]", "-", candidate, flags=re.UNICODE)
    candidate = re.sub(r"-{2,}", "-", candidate).strip("-.")
    if not candidate:
        raise ValueError("object_name is required")
    return candidate[:80]


def _validate_object_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not candidate:
        raise ValueError("object name is required")
    if candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
        raise ValueError("invalid object name")
    if candidate != _sanitize_object_name(candidate):
        raise ValueError("invalid object name")
    return candidate


def _suggest_object_name(video_path: str) -> str:
    stem = Path(video_path).stem.strip() if video_path else "object"
    if not stem:
        stem = "object"
    try:
        return _sanitize_object_name(stem)
    except ValueError:
        return f"object-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _resolve_output_root(root: str | None) -> Path:
    out = Path(root or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _objects_root(base_output: Path) -> Path:
    root = base_output / OBJECTS_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _object_dir(object_name: str, base_output: Path) -> Path:
    return _objects_root(base_output) / object_name


def _active_output_dir() -> Path:
    cfg_out = (session.config.output_dir or "").strip()
    if cfg_out:
        return Path(cfg_out)
    return _resolve_output_root(OUTPUT_DIR)


def _list_preview_files(out: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not out.is_dir():
        return files
    for f in sorted(out.rglob("*")):
        if (
            not f.is_file()
            or f.name == OBJECT_META_FILE
            or f.suffix.lower() not in PREVIEW_FILE_EXTENSIONS
        ):
            continue
        size_bytes = f.stat().st_size
        files.append(
            {
                "path": str(f.relative_to(out)),
                "name": f.name,
                "size_mb": round(size_bytes / 1024 / 1024, 2),
                "size_bytes": size_bytes,
                "ext": f.suffix.lower(),
            }
        )
    return files


def _count_indexed_files(dir_path: Path, suffix: str) -> int:
    if not dir_path.is_dir():
        return 0
    return sum(1 for _ in dir_path.glob(f"*{suffix}"))


def _stage_completion_flags(out: Path) -> tuple[dict[str, bool], int, int]:
    frame_count = _count_indexed_files(out / "frames", ".jpg")
    mask_count = _count_indexed_files(out / "masks", ".png")
    stages = {
        "1": frame_count > 0,
        "2": (out / "object_full.ply").is_file() and (out / "camera_poses.json").is_file(),
        "3": (out / "object.ply").is_file() and mask_count > 0,
        "4": (out / "object_denoised.ply").is_file(),
        "5": (out / "object_mesh.ply").is_file(),
        "6": (out / "textured_mesh.obj").is_file(),
    }
    return stages, frame_count, mask_count


def _latest_update_ts(out: Path, fallback: str | None) -> str | None:
    latest: float | None = None
    if out.exists():
        latest = out.stat().st_mtime
    for f in out.rglob("*"):
        if f.is_file():
            ts = f.stat().st_mtime
            latest = ts if latest is None else max(latest, ts)
    if latest is not None:
        return _utc_iso(latest)
    return fallback


def _prepare_object_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for rel in OBJECT_RESET_DIRS:
        target = out / rel
        if target.is_dir():
            shutil.rmtree(target)
    for rel in OBJECT_RESET_FILES:
        target = out / rel
        if target.is_file():
            target.unlink()


def _write_object_meta(
    object_name: str,
    object_dir: Path,
    video_path: str,
) -> None:
    meta_path = object_dir / OBJECT_META_FILE
    existing = _safe_json_load(meta_path)
    now = _utc_iso()
    payload = {
        "object_name": object_name,
        "video_path": video_path,
        "output_dir": str(object_dir),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _summarize_object(
    object_name: str,
    object_dir: Path,
    include_files: bool = False,
) -> dict[str, Any]:
    meta = _safe_json_load(object_dir / OBJECT_META_FILE)
    files = _list_preview_files(object_dir)
    file_map = {f["path"]: f for f in files}
    primary_files = [file_map[p] for p in PRIMARY_ARTIFACT_PATHS if p in file_map]
    stages, frame_count, mask_count = _stage_completion_flags(object_dir)
    updated_at = _latest_update_ts(object_dir, meta.get("updated_at"))
    total_bytes = sum(f["size_bytes"] for f in files)

    item: dict[str, Any] = {
        "name": object_name,
        "video_path": meta.get("video_path"),
        "video_name": Path(meta["video_path"]).name if meta.get("video_path") else None,
        "output_dir": str(object_dir),
        "created_at": meta.get("created_at"),
        "updated_at": updated_at,
        "stages": stages,
        "complete_stages": sum(1 for ok in stages.values() if ok),
        "frame_count": frame_count,
        "mask_count": mask_count,
        "file_count": len(files),
        "size_mb": round(total_bytes / 1024 / 1024, 2),
        "artifacts": primary_files,
    }
    if include_files:
        item["files"] = files
    return item


def _list_objects(base_output: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    root = _objects_root(base_output)
    for d in root.iterdir():
        if not d.is_dir():
            continue
        objects.append(_summarize_object(d.name, d, include_files=False))
    objects.sort(key=lambda o: o.get("updated_at") or "", reverse=True)
    return objects


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
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                    "suggested_object_name": _suggest_object_name(str(f)),
                })
    return JSONResponse({"videos": videos})


@app.get("/api/pipeline/objects")
async def pipeline_objects():
    base_output = _resolve_output_root(OUTPUT_DIR)
    objects = _list_objects(base_output)
    return JSONResponse(
        {
            "objects": objects,
            "active_object": session.config.object_name or None,
        }
    )


@app.get("/api/pipeline/object-info")
async def pipeline_object_info(name: str):
    try:
        object_name = _validate_object_name(name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    base_output = _resolve_output_root(OUTPUT_DIR)
    out = _object_dir(object_name, base_output)
    if not out.is_dir():
        return JSONResponse({"error": "Object not found"}, status_code=404)
    return JSONResponse({"object": _summarize_object(object_name, out, include_files=True)})


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

    raw = body or {}
    video_path = str(raw.get("video_path", "")).strip()
    requested_object = str(raw.get("object_name", "")).strip()
    if requested_object:
        try:
            object_name = _sanitize_object_name(requested_object)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    else:
        object_name = _suggest_object_name(video_path)

    if not video_path:
        return JSONResponse({"error": "video_path is required"}, status_code=400)

    output_root = _resolve_output_root(str(raw.get("output_dir", OUTPUT_DIR)))
    object_output_dir = _object_dir(object_name, output_root)
    _prepare_object_output_dir(object_output_dir)
    _write_object_meta(object_name, object_output_dir, video_path)

    # Build config from body + env defaults
    cfg = PipelineConfig(
        video_path=video_path,
        output_dir=str(object_output_dir),
        object_name=object_name,
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
    session.config = cfg

    # Set env vars that stages read directly
    os.environ["DIFFCD_BATCH_SIZE"] = str(cfg.diffcd_batch_size)
    os.environ["DIFFCD_N_BATCHES"] = str(cfg.diffcd_n_batches)
    os.environ["DIFFCD_RESOLUTION"] = str(cfg.diffcd_resolution)

    # Launch pipeline as background task
    session._task = asyncio.create_task(run_pipeline(session, sam2_service))

    return JSONResponse(
        {
            "status": "started",
            "object_name": cfg.object_name,
            "output_dir": cfg.output_dir,
        }
    )


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
    out = _active_output_dir()
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
    out = _active_output_dir()
    files = _list_preview_files(out)
    for f in files:
        f.pop("size_bytes", None)
    return JSONResponse({"files": files})


@app.get("/api/preview/file/{path:path}")
async def preview_file(path: str):
    """Serve an output file by relative path."""
    out = _active_output_dir()
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
