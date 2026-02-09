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
from scripts.dashboard.pipeline_runner import broadcast, run_pipeline
from scripts.dashboard.sam2_service import SAM2Service
from scripts.dashboard.state import (
    PipelineConfig,
    PipelineSession,
    PipelineStage,
    detect_stage_outputs,
)
from vram_utils import estimate_pi3x_frame_plan

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
STAGE_RESET_PATHS: dict[int, dict[str, tuple[str, ...]]] = {
    1: {"dirs": ("frames",), "files": ()},
    2: {"dirs": (), "files": ("object_full.ply", "pi3x_cache.npz", "camera_poses.json")},
    3: {"dirs": ("masks",), "files": ("object.ply",)},
    4: {"dirs": (), "files": ("object_denoised.ply",)},
    5: {
        "dirs": ("diffcd", "classical_mesh"),
        "files": (
            "object_mesh.ply",
            "object_mesh_raw.ply",
            "object_mesh_postprocessed.ply",
            "object_mesh_input.ply",
            "object_points.npy",
            "object_points_with_normals.ply",
        ),
    },
    6: {"dirs": (), "files": ("textured_mesh.obj", "textured_mesh.mtl", "texture.png", "intrinsics.json")},
}

RESUME_PREREQUISITES: dict[int, dict[str, tuple[str, ...]]] = {
    2: {"dirs": ("frames",), "files": ()},
    3: {"dirs": ("frames",), "files": ("object_full.ply", "camera_poses.json", "pi3x_cache.npz")},
    4: {"dirs": (), "files": ("object.ply",)},
    5: {"dirs": (), "files": ("object_denoised.ply",)},
    6: {"dirs": ("frames", "masks"), "files": ("object_mesh.ply", "camera_poses.json")},
}

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
DENOISE_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "balanced": {
        "denoise_algorithm": "dbscan_sor",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.02,
        "denoise_dbscan_min_samples": 10,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 8,
        "denoise_radius_radius_ratio": 0.015,
    },
    "detail_preserving": {
        "denoise_algorithm": "sor_only",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.02,
        "denoise_dbscan_min_samples": 10,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 16,
        "denoise_sor_std_ratio": 2.6,
        "denoise_radius_neighbors": 8,
        "denoise_radius_radius_ratio": 0.015,
    },
    "isolate_subject": {
        "denoise_algorithm": "dbscan_only",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.018,
        "denoise_dbscan_min_samples": 8,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 8,
        "denoise_radius_radius_ratio": 0.015,
    },
    "sparse_noise": {
        "denoise_algorithm": "radius_only",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.02,
        "denoise_dbscan_min_samples": 10,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 6,
        "denoise_radius_radius_ratio": 0.012,
    },
    "aggressive_cleanup": {
        "denoise_algorithm": "dbscan_radius",
        "denoise_dbscan_eps": 0.0,
        "denoise_dbscan_eps_ratio": 0.024,
        "denoise_dbscan_min_samples": 14,
        "denoise_dbscan_max_points": 500000,
        "denoise_sor_neighbors": 20,
        "denoise_sor_std_ratio": 2.0,
        "denoise_radius_neighbors": 10,
        "denoise_radius_radius_ratio": 0.02,
    },
}
DENOISE_ALGORITHMS = {
    "dbscan_sor",
    "dbscan_only",
    "sor_only",
    "radius_only",
    "dbscan_radius",
}
MESH_METHODS = {"poisson", "diffcd"}
MESH_POSTPROCESS_METHODS = {"laplacian", "taubin"}
_LEGACY_DIFFCD_DEFAULTS = (3000, 2500, 384)


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


def _parse_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _parse_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_choice(value: Any, choices: set[str], fallback: str) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in choices else fallback


def _parse_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "on", "y"}:
        return True
    if candidate in {"0", "false", "no", "off", "n"}:
        return False
    return fallback


def _env_int(name: str, fallback: int) -> int:
    return _parse_int(os.environ.get(name, fallback), fallback)


def _env_float(name: str, fallback: float) -> float:
    return _parse_float(os.environ.get(name, fallback), fallback)


def _upgrade_legacy_diffcd_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(raw)
    batch = _parse_int(upgraded.get("diffcd_batch_size"), -1)
    n_batches = _parse_int(upgraded.get("diffcd_n_batches"), -1)
    resolution = _parse_int(upgraded.get("diffcd_resolution"), -1)
    if (batch, n_batches, resolution) != _LEGACY_DIFFCD_DEFAULTS:
        return upgraded

    upgraded["diffcd_batch_size"] = _env_int("DIFFCD_BATCH_SIZE", 5000)
    upgraded["diffcd_n_batches"] = _env_int("DIFFCD_N_BATCHES", 30000)
    upgraded["diffcd_resolution"] = _env_int("DIFFCD_RESOLUTION", 512)
    return upgraded


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


def _stage_completion_flags(out: Path) -> tuple[dict[str, bool], int, int]:
    stages, frame_count, mask_count = detect_stage_outputs(out)
    return {str(k): v for k, v in stages.items()}, frame_count, mask_count


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
    _reset_outputs_from_stage(out, int(PipelineStage.EXTRACT_FRAMES))


def _reset_outputs_from_stage(out: Path, start_stage: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stage in range(max(1, start_stage), 7):
        plan = STAGE_RESET_PATHS.get(stage, {})
        for rel in plan.get("dirs", ()):
            target = out / rel
            if target.is_dir():
                shutil.rmtree(target)
        for rel in plan.get("files", ()):
            target = out / rel
            if target.is_file():
                target.unlink()


def _infer_resume_stage(out: Path) -> int:
    stage_complete, _, _ = detect_stage_outputs(out)
    for stage in range(1, 7):
        if not stage_complete.get(stage, False):
            return stage
    return int(PipelineStage.TEXTURE_BAKE)


def _validate_resume_prerequisites(out: Path, start_stage: int) -> list[str]:
    issues: list[str] = []
    req = RESUME_PREREQUISITES.get(start_stage)
    if not req:
        return issues

    for rel in req.get("dirs", ()):
        path = out / rel
        if not path.is_dir():
            issues.append(f"missing directory: {rel}/")
            continue
        suffix = ".jpg" if rel == "frames" else ".png" if rel == "masks" else None
        if suffix is not None and not any(path.glob(f"*{suffix}")):
            issues.append(f"empty directory: {rel}/ ({suffix})")
    for rel in req.get("files", ()):
        if not (out / rel).is_file():
            issues.append(f"missing file: {rel}")
    return issues


def _build_pipeline_config(
    raw: dict[str, Any],
    *,
    video_path: str,
    object_name: str,
    output_dir: Path,
    upgrade_legacy_diffcd_defaults: bool = False,
) -> PipelineConfig:
    source = (
        _upgrade_legacy_diffcd_defaults(raw)
        if upgrade_legacy_diffcd_defaults
        else raw
    )
    default_mesh_method = _parse_choice(os.environ.get("MESH_METHOD"), MESH_METHODS, "poisson")
    raw_preset = str(source.get("denoise_preset") or "").strip()
    if raw_preset == "custom":
        preset = "custom"
        denoise_defaults = DENOISE_PRESET_DEFAULTS["balanced"]
    else:
        preset = _parse_choice(
            raw_preset,
            set(DENOISE_PRESET_DEFAULTS),
            "balanced",
        )
        denoise_defaults = DENOISE_PRESET_DEFAULTS[preset]
    denoise_algorithm = _parse_choice(
        source.get("denoise_algorithm"),
        DENOISE_ALGORITHMS,
        str(denoise_defaults["denoise_algorithm"]),
    )
    max_frames = max(2, _parse_int(source.get("max_frames"), _env_int("MAX_FRAMES", 50)))
    max_pi3x_target = max_frames
    pi3x_frame_target = max(
        2,
        min(
            _parse_int(source.get("pi3x_frame_target"), max_frames),
            max_pi3x_target,
        ),
    )

    return PipelineConfig(
        video_path=video_path,
        output_dir=str(output_dir),
        object_name=object_name,
        frame_interval=_parse_int(source.get("frame_interval"), _env_int("FRAME_INTERVAL", 10)),
        max_frames=max_frames,
        pixel_limit=_parse_int(source.get("pixel_limit"), _env_int("PIXEL_LIMIT", 255000)),
        pi3x_frame_target=pi3x_frame_target,
        confidence_threshold=_parse_float(source.get("confidence_threshold"), _env_float("CONFIDENCE_THRESHOLD", 0.2)),
        edge_rtol=_parse_float(source.get("edge_rtol"), _env_float("EDGE_RTOL", 0.03)),
        sam2_model=str(source.get("sam2_model") or os.environ.get("SAM2_MODEL", "large")),
        denoise_preset=preset,
        denoise_algorithm=denoise_algorithm,
        denoise_dbscan_eps=max(0.0, _parse_float(source.get("denoise_dbscan_eps"), float(denoise_defaults["denoise_dbscan_eps"]))),
        denoise_dbscan_eps_ratio=max(0.0001, _parse_float(source.get("denoise_dbscan_eps_ratio"), float(denoise_defaults["denoise_dbscan_eps_ratio"]))),
        denoise_dbscan_min_samples=max(1, _parse_int(source.get("denoise_dbscan_min_samples"), int(denoise_defaults["denoise_dbscan_min_samples"]))),
        denoise_dbscan_max_points=max(1000, _parse_int(source.get("denoise_dbscan_max_points"), int(denoise_defaults["denoise_dbscan_max_points"]))),
        denoise_sor_neighbors=max(2, _parse_int(source.get("denoise_sor_neighbors"), int(denoise_defaults["denoise_sor_neighbors"]))),
        denoise_sor_std_ratio=max(0.1, _parse_float(source.get("denoise_sor_std_ratio"), float(denoise_defaults["denoise_sor_std_ratio"]))),
        denoise_radius_neighbors=max(1, _parse_int(source.get("denoise_radius_neighbors"), int(denoise_defaults["denoise_radius_neighbors"]))),
        denoise_radius_radius_ratio=max(0.0001, _parse_float(source.get("denoise_radius_radius_ratio"), float(denoise_defaults["denoise_radius_radius_ratio"]))),
        mesh_method=_parse_choice(source.get("mesh_method"), MESH_METHODS, default_mesh_method),
        diffcd_batch_size=_parse_int(source.get("diffcd_batch_size"), _env_int("DIFFCD_BATCH_SIZE", 5000)),
        diffcd_n_batches=_parse_int(source.get("diffcd_n_batches"), _env_int("DIFFCD_N_BATCHES", 30000)),
        diffcd_resolution=_parse_int(source.get("diffcd_resolution"), _env_int("DIFFCD_RESOLUTION", 512)),
        texture_size=_parse_int(source.get("texture_size"), _env_int("TEXTURE_SIZE", 2048)),
    )


def _write_object_meta(
    object_name: str,
    object_dir: Path,
    video_path: str,
    *,
    config: dict[str, Any] | None = None,
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
        "config": config if isinstance(config, dict) else existing.get("config", {}),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_object_into_session(object_name: str, object_dir: Path) -> dict[str, Any]:
    meta = _safe_json_load(object_dir / OBJECT_META_FILE)
    raw_cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    cfg = _build_pipeline_config(
        raw_cfg,
        video_path=str(meta.get("video_path", "")),
        object_name=object_name,
        output_dir=object_dir,
        upgrade_legacy_diffcd_defaults=True,
    )

    session.reset()
    session.config = cfg
    session.hydrate_from_output_dir(object_dir)
    session.resume_from_stage = PipelineStage(_infer_resume_stage(object_dir))
    return _summarize_object(object_name, object_dir, include_files=True)


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
        "resume_from_stage": _infer_resume_stage(object_dir),
    }
    if include_files:
        item["files"] = files
        if isinstance(meta.get("config"), dict):
            cfg = _build_pipeline_config(
                meta["config"],
                video_path=str(meta.get("video_path", "")),
                object_name=object_name,
                output_dir=object_dir,
                upgrade_legacy_diffcd_defaults=True,
            )
            item["config"] = cfg.to_dict()
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

    def _resolve_current_stage() -> int | None:
        try:
            stage = int(session.current_stage)
        except Exception:
            return None
        if 1 <= stage <= 6:
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
    start_stage = _parse_int(raw.get("resume_from_stage"), inferred_stage)
    if start_stage < int(PipelineStage.EXTRACT_FRAMES) or start_stage > int(PipelineStage.TEXTURE_BAKE):
        return JSONResponse({"error": "resume_from_stage must be between 1 and 6"}, status_code=400)

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
    explicit_diffcd = any(
        key in raw
        for key in ("diffcd_batch_size", "diffcd_n_batches", "diffcd_resolution")
    )
    cfg = _build_pipeline_config(
        cfg_source,
        video_path=video_path,
        object_name=object_name,
        output_dir=object_output_dir,
        upgrade_legacy_diffcd_defaults=not explicit_diffcd,
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

    session.cancelled = True
    stage_num = int(session.current_stage)
    if int(PipelineStage.EXTRACT_FRAMES) <= stage_num <= int(PipelineStage.TEXTURE_BAKE):
        stage = PipelineStage(stage_num)
        session.stage_progress(
            stage,
            detail="Cancellation requested. Waiting for a safe stop point.",
        )
        await broadcast(
            session,
            {
                "type": "stage_progress",
                "stage": stage_num,
                "progress": round(session.stages[stage_num].progress, 1),
                "detail": session.stages[stage_num].detail,
                "overall_progress": session.overall_progress(),
            },
        )

    # If waiting for SAM2 or Pi3X confirmation/approval, unblock it
    session.sam2_confirm_event.set()
    session.sam2_approve_event.set()
    session.next_stage_confirm_event.set()
    return JSONResponse({"status": "cancelling", "stage": stage_num})


# ── Pi3X API ──────────────────────────────────────────────────────

@app.post("/api/pi3x/approve")
async def pi3x_approve():
    # Backward-compatible alias for global stage transition approval.
    session.next_stage_confirm_event.set()
    return JSONResponse({"status": "approved"})


# ── Global stage-transition approval API ───────────────────────────

@app.post("/api/pipeline/confirm-next")
async def pipeline_confirm_next():
    if not session.running:
        return JSONResponse({"error": "No pipeline running"}, status_code=409)
    if not session.next_stage_confirmation_required:
        return JSONResponse({"status": "no_waiting_confirmation"})
    session.next_stage_confirm_event.set()
    return JSONResponse({"status": "confirmed"})


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
    method = _parse_choice(raw.get("method"), MESH_POSTPROCESS_METHODS, "laplacian")
    iterations = max(0, min(100, _parse_int(raw.get("iterations"), 6)))
    lamb = max(0.01, min(1.5, _parse_float(raw.get("lamb"), 0.5)))
    taubin_nu = max(-1.5, min(-0.01, _parse_float(raw.get("taubin_nu"), -0.53)))
    source = str(raw.get("source") or "raw").strip().lower()
    if source not in {"raw", "current"}:
        source = "raw"
    invalidate_texture = _parse_bool(raw.get("invalidate_texture"), True)

    out = _active_output_dir()
    mesh_path = out / "object_mesh.ply"
    if not mesh_path.is_file():
        return JSONResponse({"error": "object_mesh.ply not found"}, status_code=404)

    source_path = out / "object_mesh_raw.ply" if source == "raw" else mesh_path
    if not source_path.is_file():
        source_path = mesh_path
        source = "current"

    def _apply() -> tuple[int, int, bool]:
        from stage_diffcd_mesh import mesh_vertex_face_count, smooth_mesh_file
        import open3d as o3d

        downsample_enabled = _parse_bool(
            os.environ.get("CLASSICAL_DOWNSAMPLE_ENABLED"),
            True,
        )
        downsample_target_faces = max(
            _env_int("CLASSICAL_DOWNSAMPLE_TARGET_FACES", 220000),
            1000,
        )
        downsample_trigger_faces = max(
            _env_int("CLASSICAL_DOWNSAMPLE_TRIGGER_FACES", 280000),
            downsample_target_faces,
        )

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
        texture_outputs = STAGE_RESET_PATHS[int(PipelineStage.TEXTURE_BAKE)]["files"]
        if any((out / rel).is_file() for rel in texture_outputs):
            _reset_outputs_from_stage(out, int(PipelineStage.TEXTURE_BAKE))
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
