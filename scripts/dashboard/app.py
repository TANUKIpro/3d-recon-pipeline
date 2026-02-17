"""FastAPI web dashboard for clip2mesh.

Entry point: uvicorn scripts.dashboard.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from scripts.dashboard.configuration import (
    build_pipeline_config,
    env_int,
    parse_bool,
    parse_choice,
    parse_float,
    parse_int,
)
from scripts.dashboard.log_capture import LogBroadcaster
from scripts.dashboard.object_store import (
    OBJECT_META_FILE,
    PRIMARY_ARTIFACT_PATHS,
    STAGE_RESET_PATHS,
    _infer_resume_stage,
    _list_objects,
    _list_preview_files,
    _object_dir,
    _prepare_object_output_dir,
    _reset_outputs_from_stage,
    _resolve_output_root,
    _safe_json_load,
    _sanitize_object_name,
    _suggest_object_name,
    _summarize_object,
    _validate_object_name,
    _validate_resume_prerequisites,
    _write_object_meta,
)
from scripts.dashboard.pipeline_runner import broadcast, run_pipeline
from scripts.dashboard.sam2_service import SAM2Service
from scripts.dashboard.state import (
    PipelineSession,
    PipelineStage,
)
from vram_utils import estimate_pi3x_frame_plan

# ── Globals ───────────────────────────────────────────────────────

app = FastAPI(title="clip2mesh Dashboard")
session = PipelineSession()
sam2_service = SAM2Service()
log_broadcaster: LogBroadcaster | None = None

INPUT_DIR = os.environ.get("INPUT_DIR", "/data/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output")

STATIC_DIR = Path(__file__).parent / "static"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MESH_POSTPROCESS_METHODS = {"laplacian", "taubin"}


def _active_output_dir() -> Path:
    cfg_out = (session.config.output_dir or "").strip()
    if cfg_out:
        return Path(cfg_out)
    return _resolve_output_root(OUTPUT_DIR)


def _load_object_into_session(object_name: str, object_dir: Path) -> dict[str, Any]:
    meta = _safe_json_load(object_dir / OBJECT_META_FILE)
    raw_cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    cfg = build_pipeline_config(
        raw_cfg,
        video_path=str(meta.get("video_path", "")),
        object_name=object_name,
        output_dir=object_dir,
    )

    session.reset()
    session.config = cfg
    session.hydrate_from_output_dir(object_dir)
    session.resume_from_stage = PipelineStage(_infer_resume_stage(object_dir))
    return _summarize_object(object_name, object_dir, include_files=True)


# ── Lifecycle ─────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    global log_broadcaster
    loop = asyncio.get_event_loop()

    def _resolve_current_stage() -> int | None:
        try:
            stage = int(session.current_stage)
        except Exception:
            return None
        if 1 <= stage <= int(PipelineStage.TEXTURE_BAKE):
            return stage
        return None

    log_broadcaster = LogBroadcaster(loop, stage_resolver=_resolve_current_stage)
    log_broadcaster.install()
    asyncio.create_task(log_broadcaster.drain(session.ws_clients))
    try:
        base_output = _resolve_output_root(OUTPUT_DIR)
        objects = _list_objects(base_output)
        if objects:
            latest = objects[0]["name"]
            _load_object_into_session(latest, _object_dir(latest, base_output))
    except Exception:
        # Best-effort auto-load only.
        pass


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


@app.post("/api/pipeline/load-object")
async def pipeline_load_object(body: dict | None = None):
    if session.running:
        return JSONResponse({"error": "Cannot switch object while pipeline is running"}, status_code=409)

    raw = body or {}
    try:
        object_name = _validate_object_name(str(raw.get("name", "")).strip())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    base_output = _resolve_output_root(str(raw.get("output_dir", OUTPUT_DIR)))
    out = _object_dir(object_name, base_output)
    if not out.is_dir():
        return JSONResponse({"error": "Object not found"}, status_code=404)

    obj = _load_object_into_session(object_name, out)
    await broadcast(session, {"type": "status", **session.to_status_dict()})
    return JSONResponse(
        {
            "status": "loaded",
            "object": obj,
            "pipeline_status": session.to_status_dict(),
        }
    )


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


@app.get("/api/pipeline/pi3x-plan")
async def pipeline_pi3x_plan(requested_frames: int, pixel_limit: int):
    """Estimate Pi3X frame auto-tuning result before pipeline start."""
    req = max(2, int(requested_frames))
    px = max(1, int(pixel_limit))
    plan = await asyncio.to_thread(estimate_pi3x_frame_plan, req, px)
    return JSONResponse(plan)


@app.post("/api/pipeline/start")
async def pipeline_start(body: dict | None = None):
    if session.running:
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)

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

    output_root = _resolve_output_root(str(raw.get("output_dir", OUTPUT_DIR)))
    object_output_dir = _object_dir(object_name, output_root)
    existing_meta = _safe_json_load(object_output_dir / OBJECT_META_FILE)
    if not video_path:
        video_path = str(existing_meta.get("video_path", "")).strip()

    inferred_stage = (
        _infer_resume_stage(object_output_dir)
        if object_output_dir.is_dir()
        else int(PipelineStage.EXTRACT_FRAMES)
    )
    start_stage = parse_int(raw.get("resume_from_stage"), inferred_stage)
    if start_stage < int(PipelineStage.EXTRACT_FRAMES) or start_stage > int(PipelineStage.TEXTURE_BAKE):
        return JSONResponse(
            {"error": f"resume_from_stage must be between 1 and {int(PipelineStage.TEXTURE_BAKE)}"},
            status_code=400,
        )

    if start_stage == int(PipelineStage.EXTRACT_FRAMES) and not video_path:
        return JSONResponse({"error": "video_path is required for stage 1 restart"}, status_code=400)

    if start_stage > int(PipelineStage.EXTRACT_FRAMES):
        if not object_output_dir.is_dir():
            return JSONResponse({"error": "Object output does not exist for resume"}, status_code=400)
        missing = _validate_resume_prerequisites(object_output_dir, start_stage)
        if missing:
            return JSONResponse(
                {
                    "error": "Cannot resume from selected stage due to missing artifacts",
                    "missing": missing,
                },
                status_code=400,
            )

    if start_stage == int(PipelineStage.EXTRACT_FRAMES):
        _prepare_object_output_dir(object_output_dir)
    else:
        _reset_outputs_from_stage(object_output_dir, start_stage)

    cfg_source = {}
    if isinstance(existing_meta.get("config"), dict):
        cfg_source.update(existing_meta["config"])
    cfg_source.update(raw)
    cfg = build_pipeline_config(
        cfg_source,
        video_path=video_path,
        object_name=object_name,
        output_dir=object_output_dir,
    )

    session.reset()
    session.config = cfg
    session.resume_from_stage = PipelineStage(start_stage)
    session.hydrate_from_output_dir(object_output_dir)
    _write_object_meta(
        object_name,
        object_output_dir,
        cfg.video_path,
        config=cfg.to_dict(),
    )

    # Set env vars that stages read directly
    os.environ["DIFFCD_BATCH_SIZE"] = str(cfg.diffcd_batch_size)
    os.environ["DIFFCD_N_BATCHES"] = str(cfg.diffcd_n_batches)
    os.environ["DIFFCD_RESOLUTION"] = str(cfg.diffcd_resolution)
    os.environ["MESH_REPAIR_ENABLED"] = "1" if cfg.mesh_repair_enabled else "0"
    os.environ["MESH_REPAIR_MAX_DIAMETER_RATIO"] = str(cfg.mesh_repair_max_diameter_ratio)
    os.environ["MESH_REPAIR_Y_BAND_RATIO"] = str(cfg.mesh_repair_y_band_ratio)
    os.environ["MESH_REPAIR_SMOOTH_ITERS"] = str(cfg.mesh_repair_smooth_iters)

    # Launch pipeline as background task
    session._task = asyncio.create_task(run_pipeline(session, sam2_service))

    return JSONResponse(
        {
            "status": "started",
            "object_name": cfg.object_name,
            "output_dir": cfg.output_dir,
            "resume_from_stage": int(session.resume_from_stage),
        }
    )


@app.post("/api/pipeline/cancel")
async def pipeline_cancel():
    if not session.running:
        return JSONResponse({"error": "No pipeline running"}, status_code=409)

    session.request_cancel(force=True)
    terminated = await asyncio.to_thread(session.terminate_active_processes, 1.0)
    stage_num = int(session.current_stage)
    checkpoint_id = session.current_checkpoint_id
    if int(PipelineStage.EXTRACT_FRAMES) <= stage_num <= int(PipelineStage.TEXTURE_BAKE):
        stage = PipelineStage(stage_num)
        session.stage_progress(
            stage,
            detail="Cancellation requested. Stopping immediately.",
            checkpoint_id=checkpoint_id,
        )
        await broadcast(
            session,
            {
                "type": "stage_progress",
                "stage": stage_num,
                "progress": round(session.stages[stage_num].progress, 1),
                "detail": session.stages[stage_num].detail,
                "checkpoint_id": session.stages[stage_num].checkpoint_id,
                "overall_progress": session.overall_progress(),
            },
        )

    # If waiting for SAM2 or Pi3X confirmation/approval, unblock it
    session.sam2_confirm_event.set()
    session.sam2_approve_event.set()
    session.next_stage_confirm_event.set()
    session.mesh_repair_confirm_event.set()
    return JSONResponse(
        {
            "status": "cancelling",
            "mode": "force",
            "stage": stage_num,
            "checkpoint_id": checkpoint_id,
            "terminated_processes": terminated,
            "cleanup_started": True,
        }
    )


# ── Global stage-transition approval API ───────────────────────────

@app.post("/api/pipeline/confirm-next")
async def pipeline_confirm_next():
    if not session.running:
        return JSONResponse({"error": "No pipeline running"}, status_code=409)
    if not session.next_stage_confirmation_required:
        return JSONResponse({"status": "no_waiting_confirmation"})
    session.next_stage_confirm_event.set()
    return JSONResponse({"status": "confirmed"})


# ── Mesh repair interactive API ───────────────────────────────────

@app.get("/api/mesh-repair/candidates")
async def mesh_repair_candidates():
    if not session.running:
        return JSONResponse({"error": "Pipeline is not running"}, status_code=409)
    if not session.mesh_repair_ready:
        return JSONResponse({"error": "Mesh repair selection is not ready"}, status_code=409)

    source_rel: str | None = None
    source_path = session.mesh_repair_source_mesh_path
    if source_path:
        try:
            out = _active_output_dir().resolve()
            source_obj = Path(source_path).resolve()
            source_rel = str(source_obj.relative_to(out))
        except Exception:
            source_rel = None

    return JSONResponse(
        {
            "status": "ok",
            "source_mesh_path": source_path,
            "source_mesh_relpath": source_rel,
            "loop_count": len(session.mesh_repair_candidates),
            "loops": session.mesh_repair_candidates,
            "analysis": session.mesh_repair_analysis,
            "color_scheme": {
                "candidate": "#f4d03f",
                "selected": "#e74c3c",
                "confirmed": "#2ecc71",
            },
        }
    )


@app.post("/api/mesh-repair/confirm")
async def mesh_repair_confirm(body: dict | None = None):
    if not session.running:
        return JSONResponse({"error": "Pipeline is not running"}, status_code=409)
    if not session.mesh_repair_ready:
        return JSONResponse({"error": "Mesh repair selection is not ready"}, status_code=409)

    raw = body or {}
    selected_raw = raw.get("selected_loop_ids")
    if not isinstance(selected_raw, list):
        return JSONResponse({"error": "selected_loop_ids must be a list"}, status_code=400)

    selected: list[int] = []
    seen: set[int] = set()
    for value in selected_raw:
        try:
            loop_id = int(value)
        except (TypeError, ValueError):
            return JSONResponse({"error": f"Invalid loop id: {value!r}"}, status_code=400)
        if loop_id in seen:
            continue
        selected.append(loop_id)
        seen.add(loop_id)

    available_ids = {
        int(item.get("loop_id"))
        for item in session.mesh_repair_candidates
        if isinstance(item, dict) and item.get("loop_id") is not None
    }
    missing = sorted(loop_id for loop_id in selected if loop_id not in available_ids)
    if missing:
        return JSONResponse({"error": f"Unknown loop ids: {missing}"}, status_code=400)

    session.mesh_repair_selected_loop_ids = selected
    session.mesh_repair_confirm_event.set()
    return JSONResponse(
        {
            "status": "confirmed",
            "mode": "skip" if len(selected) == 0 else "repair",
            "selected_loop_ids": selected,
            "selected_count": len(selected),
        }
    )


# ── Mesh post-process API ────────────────────────────────────────

@app.post("/api/mesh/postprocess")
async def mesh_postprocess(body: dict | None = None):
    if session.running:
        waiting_stage5 = (
            session.next_stage_confirmation_required
            and session.next_stage_confirmation_from == int(PipelineStage.DIFFCD_MESH)
        )
        if not waiting_stage5:
            return JSONResponse(
                {
                    "error": (
                        "Mesh post-process is available only when the pipeline is idle "
                        "or paused after Stage 5."
                    )
                },
                status_code=409,
            )

    raw = body or {}
    method = parse_choice(raw.get("method"), MESH_POSTPROCESS_METHODS, "laplacian")
    iterations = max(0, min(100, parse_int(raw.get("iterations"), 6)))
    lamb = max(0.01, min(1.5, parse_float(raw.get("lamb"), 0.5)))
    taubin_nu = max(-1.5, min(-0.01, parse_float(raw.get("taubin_nu"), -0.53)))
    downsample_enabled = parse_bool(
        raw.get("downsample_enabled"),
        parse_bool(os.environ.get("CLASSICAL_DOWNSAMPLE_ENABLED"), True),
    )
    downsample_target_faces = max(
        1000,
        parse_int(
            raw.get("downsample_target_faces"),
            env_int("CLASSICAL_DOWNSAMPLE_TARGET_FACES", 120000),
        ),
    )
    downsample_trigger_faces = max(
        parse_int(
            raw.get("downsample_trigger_faces"),
            env_int("CLASSICAL_DOWNSAMPLE_TRIGGER_FACES", 170000),
        ),
        downsample_target_faces,
    )
    source = str(raw.get("source") or "raw").strip().lower()
    if source not in {"raw", "current"}:
        source = "raw"
    invalidate_texture = parse_bool(raw.get("invalidate_texture"), True)

    out = _active_output_dir()
    mesh_path = out / "object_mesh.ply"
    if not mesh_path.is_file():
        return JSONResponse({"error": "object_mesh.ply not found"}, status_code=404)

    source_path = out / "object_mesh_raw.ply" if source == "raw" else mesh_path
    if not source_path.is_file():
        source_path = mesh_path
        source = "current"

    def _apply() -> tuple[int, int, bool]:
        from stage_classical_mesh import generate_preview_mesh
        from stage_diffcd_mesh import mesh_vertex_face_count, smooth_mesh_file
        import open3d as o3d

        smooth_mesh_file(
            source_path,
            mesh_path,
            method=method,
            iterations=iterations,
            lamb=lamb,
            taubin_nu=taubin_nu,
        )
        downsample_applied = False
        if downsample_enabled:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            face_count = int(len(mesh.triangles))
            if face_count > downsample_trigger_faces:
                target_faces = min(max(4, downsample_target_faces), max(4, face_count - 1))
                simplified = mesh.simplify_quadric_decimation(target_faces)
                if len(simplified.vertices) > 0 and len(simplified.triangles) > 0:
                    simplified.remove_degenerate_triangles()
                    simplified.remove_duplicated_triangles()
                    simplified.remove_duplicated_vertices()
                    simplified.remove_non_manifold_edges()
                    simplified.remove_unreferenced_vertices()
                    simplified.compute_vertex_normals()
                    try:
                        o3d.io.write_triangle_mesh(
                            str(mesh_path),
                            simplified,
                            write_ascii=False,
                            compressed=False,
                            write_vertex_normals=True,
                        )
                    except TypeError:
                        o3d.io.write_triangle_mesh(str(mesh_path), simplified)
                    downsample_applied = True

        preview_mesh_path = out / "object_mesh_preview.ply"
        try:
            generate_preview_mesh(mesh_path, preview_mesh_path)
        except Exception as preview_err:
            print(f"Preview mesh generation failed after post-process (non-fatal): {preview_err}")

        vertices, faces = mesh_vertex_face_count(mesh_path)
        return vertices, faces, downsample_applied

    try:
        vertices, faces, downsample_applied = await asyncio.to_thread(_apply)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    texture_invalidated = False
    if invalidate_texture:
        invalidate_from = int(PipelineStage.MESH_WRAP)
        downstream_files: tuple[str, ...] = tuple(
            rel
            for stage_id in range(invalidate_from, int(PipelineStage.TEXTURE_BAKE) + 1)
            for rel in STAGE_RESET_PATHS.get(stage_id, {}).get("files", ())
        )
        if any((out / rel).is_file() for rel in downstream_files):
            _reset_outputs_from_stage(out, invalidate_from)
            texture_invalidated = True

    session.mesh_ply = str(mesh_path)
    if not session.running:
        session.hydrate_from_output_dir(out)
        await broadcast(session, {"type": "status", **session.to_status_dict()})

    return JSONResponse(
        {
            "status": "ok",
            "method": method,
            "iterations": iterations,
            "lamb": lamb,
            "taubin_nu": taubin_nu,
            "source": source,
            "mesh_path": str(mesh_path.relative_to(out)),
            "vertices": vertices,
            "faces": faces,
            "downsample_enabled": downsample_enabled,
            "downsample_target_faces": downsample_target_faces,
            "downsample_trigger_faces": downsample_trigger_faces,
            "downsample_applied": downsample_applied,
            "texture_invalidated": texture_invalidated,
        }
    )


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


@app.get("/api/preview/crop-obb")
async def preview_crop_obb():
    """Return OBB (center, extent, rotation) for the object mesh."""
    out = _active_output_dir()
    mesh_path = out / "object_mesh.ply"
    if not mesh_path.is_file():
        return JSONResponse({"error": "object_mesh.ply not found"}, status_code=404)
    try:
        import open3d as o3d
        import numpy as np
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        obb = mesh.get_oriented_bounding_box()
        return JSONResponse({
            "center": np.asarray(obb.center).tolist(),
            "extent": np.asarray(obb.extent).tolist(),
            "rotation": np.asarray(obb.R).tolist(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
