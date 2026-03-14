"""Pipeline management routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from scripts.dashboard.configuration import (
    build_pipeline_config,
    parse_int,
)
from scripts.dashboard.dependencies import VIDEO_EXTENSIONS, get_state
from scripts.dashboard.object_store import (
    OBJECT_META_FILE,
    infer_resume_stage,
    list_objects,
    object_dir,
    prepare_object_output_dir,
    reset_outputs_from_stage,
    resolve_output_root,
    safe_json_load,
    sanitize_object_name,
    suggest_object_name,
    summarize_object,
    validate_object_name,
    validate_resume_prerequisites,
    write_object_meta,
)
from scripts.dashboard.pipeline_runner import broadcast, run_pipeline
from scripts.dashboard.state import PipelineStage

router = APIRouter()


@router.get("/api/pipeline/status")
async def pipeline_status():
    return JSONResponse(get_state().session.to_status_dict())


@router.get("/api/pipeline/videos")
async def pipeline_videos():
    """List video files in /data/input/."""
    input_dir = Path(get_state().input_dir)
    videos: list[dict[str, Any]] = []
    if input_dir.is_dir():
        for f in sorted(input_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                    "suggested_object_name": suggest_object_name(str(f)),
                })
    return JSONResponse({"videos": videos})


@router.get("/api/pipeline/objects")
async def pipeline_objects():
    state = get_state()
    base_output = resolve_output_root(state.output_dir)
    objects = list_objects(base_output)
    return JSONResponse(
        {
            "objects": objects,
            "active_object": state.session.config.object_name or None,
        }
    )


@router.get("/api/pipeline/object-info")
async def pipeline_object_info(name: str):
    try:
        obj_name = validate_object_name(name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    base_output = resolve_output_root(get_state().output_dir)
    out = object_dir(obj_name, base_output)
    if not out.is_dir():
        return JSONResponse({"error": "Object not found"}, status_code=404)
    return JSONResponse({"object": summarize_object(obj_name, out, include_files=True)})


@router.post("/api/pipeline/load-object")
async def pipeline_load_object(body: dict | None = None):
    state = get_state()
    session = state.session
    if session.running:
        return JSONResponse({"error": "Cannot switch object while pipeline is running"}, status_code=409)

    raw = body or {}
    try:
        obj_name = validate_object_name(str(raw.get("name", "")).strip())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    base_output = resolve_output_root(str(raw.get("output_dir", state.output_dir)))
    out = object_dir(obj_name, base_output)
    if not out.is_dir():
        return JSONResponse({"error": "Object not found"}, status_code=404)

    obj = state.load_object_into_session(obj_name, out)
    await broadcast(session, {"type": "status", **session.to_status_dict()})
    return JSONResponse(
        {
            "status": "loaded",
            "object": obj,
            "pipeline_status": session.to_status_dict(),
        }
    )


@router.get("/api/pipeline/video-info")
async def pipeline_video_info(path: str):
    """Return metadata (fps, frames, resolution, duration) for a video file."""
    target = Path(path).resolve()
    input_dir = Path(get_state().input_dir).resolve()
    try:
        target.relative_to(input_dir)
    except ValueError:
        return JSONResponse({"error": "Access denied"}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    def _probe(p: str) -> dict:
        import cv2
        from scripts.stage_extract_frames import _detect_rotation, _normalize_fps
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
            # Swap width/height for 90/270 rotation
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


@router.post("/api/pipeline/start")
async def pipeline_start(body: dict | None = None):
    state = get_state()
    session = state.session
    if session.running:
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)

    raw = body or {}
    video_path = str(raw.get("video_path", "")).strip()
    requested_object = str(raw.get("object_name", "")).strip()
    if requested_object:
        try:
            obj_name = sanitize_object_name(requested_object)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    else:
        obj_name = suggest_object_name(video_path)

    output_root = resolve_output_root(str(raw.get("output_dir", state.output_dir)))
    object_output_dir = object_dir(obj_name, output_root)
    existing_meta = safe_json_load(object_output_dir / OBJECT_META_FILE)
    if not video_path:
        video_path = str(existing_meta.get("video_path", "")).strip()

    inferred_stage = (
        infer_resume_stage(object_output_dir)
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
        missing = validate_resume_prerequisites(object_output_dir, start_stage)
        if missing:
            return JSONResponse(
                {
                    "error": "Cannot resume from selected stage due to missing artifacts",
                    "missing": missing,
                },
                status_code=400,
            )

    if start_stage == int(PipelineStage.EXTRACT_FRAMES):
        prepare_object_output_dir(object_output_dir)
    else:
        reset_outputs_from_stage(object_output_dir, start_stage)

    cfg_source = {}
    if isinstance(existing_meta.get("config"), dict):
        cfg_source.update(existing_meta["config"])
    cfg_source.update(raw)
    cfg = build_pipeline_config(
        cfg_source,
        video_path=video_path,
        object_name=obj_name,
        output_dir=object_output_dir,
    )

    session.reset()
    session.config = cfg
    session.resume_from_stage = PipelineStage(start_stage)
    session.hydrate_from_output_dir(object_output_dir)
    write_object_meta(
        obj_name,
        object_output_dir,
        cfg.video_path,
        config=cfg.to_dict(),
    )

    # Launch pipeline as background task
    session._task = asyncio.create_task(run_pipeline(session, state.sam2_service))

    return JSONResponse(
        {
            "status": "started",
            "object_name": cfg.object_name,
            "output_dir": cfg.output_dir,
            "resume_from_stage": int(session.resume_from_stage),
        }
    )


@router.post("/api/pipeline/cancel")
async def pipeline_cancel():
    session = get_state().session
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

    # If waiting for SAM2 confirmation/approval, unblock it
    session.sam2_confirm_event.set()
    session.sam2_approve_event.set()
    session.sam2_ground_skip_event.set()
    session.next_stage_confirm_event.set()
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


@router.post("/api/pipeline/confirm-next")
async def pipeline_confirm_next():
    session = get_state().session
    if not session.running:
        return JSONResponse({"error": "No pipeline running"}, status_code=409)
    if not session.next_stage_confirmation_required:
        return JSONResponse({"status": "no_waiting_confirmation"})
    session.next_stage_confirm_event.set()
    return JSONResponse({"status": "confirmed"})
