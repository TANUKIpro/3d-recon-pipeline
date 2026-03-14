"""FastAPI web dashboard for clip2mesh.

Entry point: uvicorn scripts.dashboard.app:app --host 0.0.0.0 --port 7860

This module retains globals, lifecycle, static/index/WebSocket, and includes
route sub-modules.  All endpoint handler names are re-exported here so that
existing ``from scripts.dashboard.app import X`` imports keep working.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scripts.dashboard.dependencies import (
    AppState,  # noqa: F401
    MESH_POSTPROCESS_METHODS,  # noqa: F401
    VIDEO_EXTENSIONS,  # noqa: F401
    get_state,  # noqa: F401
    init_state,
)
from scripts.dashboard.log_capture import LogBroadcaster
from scripts.dashboard.object_store import (
    OBJECT_META_FILE,
    PRIMARY_ARTIFACT_PATHS,  # noqa: F401
    STAGE_RESET_PATHS,  # noqa: F401
    infer_resume_stage,
    list_objects,
    list_preview_files,  # noqa: F401
    object_dir,
    prepare_object_output_dir,  # noqa: F401
    reset_outputs_from_stage,  # noqa: F401
    resolve_output_root,
    safe_json_load,
    sanitize_object_name,  # noqa: F401
    suggest_object_name,  # noqa: F401
    summarize_object,
    validate_object_name,
    validate_resume_prerequisites,  # noqa: F401
    write_object_meta,  # noqa: F401
)
from scripts.dashboard.pipeline_runner import broadcast, run_pipeline  # noqa: F401
from scripts.dashboard.state import (
    PipelineSession,
    PipelineStage,
)

# ── Globals ───────────────────────────────────────────────────────

app = FastAPI(title="clip2mesh Dashboard")
STATIC_DIR = Path(__file__).parent / "static"

_app_state = init_state()
session = _app_state.session            # backward compat
sam2_service = _app_state.sam2_service   # backward compat
INPUT_DIR = _app_state.input_dir        # backward compat
OUTPUT_DIR = _app_state.output_dir      # backward compat
log_broadcaster: LogBroadcaster | None = None


def _active_output_dir() -> Path:
    return _app_state.active_output_dir()


def _load_object_into_session(object_name: str, object_dir: Path) -> dict[str, Any]:
    return _app_state.load_object_into_session(object_name, object_dir)


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
        base_output = resolve_output_root(_app_state.output_dir)
        objects = list_objects(base_output)
        if objects:
            latest = objects[0]["name"]
            _load_object_into_session(latest, object_dir(latest, base_output))
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


# ── Include route sub-modules ─────────────────────────────────────

from scripts.dashboard.routes.pipeline import router as _pipeline_router  # noqa: E402
from scripts.dashboard.routes.sam2 import router as _sam2_router  # noqa: E402
from scripts.dashboard.routes.verification import router as _verification_router  # noqa: E402
from scripts.dashboard.routes.preview import router as _preview_router  # noqa: E402
from scripts.dashboard.routes.mesh import router as _mesh_router  # noqa: E402
from scripts.dashboard.routes.health import router as _health_router  # noqa: E402

app.include_router(_pipeline_router)
app.include_router(_sam2_router)
app.include_router(_verification_router)
app.include_router(_preview_router)
app.include_router(_mesh_router)
app.include_router(_health_router)


# ── Test backward-compatibility re-exports ────────────────────────
# All handler function names remain importable from scripts.dashboard.app.

from scripts.dashboard.routes.pipeline import (  # noqa: E402, F401
    pipeline_cancel,
    pipeline_confirm_next,
    pipeline_load_object,
    pipeline_object_info,
    pipeline_objects,
    pipeline_start,
    pipeline_status,
    pipeline_video_info,
    pipeline_videos,
)
from scripts.dashboard.routes.sam2 import (  # noqa: E402, F401
    sam2_approve,
    sam2_clear,
    sam2_click,
    sam2_confirm,
    sam2_frame,
    sam2_get_mode,
    sam2_mask,
    sam2_mode,
    sam2_redo,
    sam2_skip_ground,
    sam2_undo,
)
from scripts.dashboard.routes.verification import (  # noqa: E402, F401
    verification_frame,
    verification_ground_frame,
)
from scripts.dashboard.routes.preview import (  # noqa: E402, F401
    preview_crop_obb,
    preview_file,
    preview_object_file,
    preview_outputs,
)
from scripts.dashboard.routes.health import (  # noqa: E402, F401
    vram_info,
)
