"""Async pipeline orchestrator for the dashboard.

Runs each stage via asyncio.to_thread, broadcasting progress over WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# Add scripts/ directory to path so stage modules can be imported by bare name
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from scripts.dashboard.state import (
    STAGE_LABELS,
    PipelineStage,
    StageStatus,
)

if TYPE_CHECKING:
    from scripts.dashboard.sam2_service import SAM2Service
    from scripts.dashboard.state import PipelineSession


async def broadcast(session: PipelineSession, msg: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    payload = json.dumps(msg)
    stale: list[int] = []
    for i, ws in enumerate(session.ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(i)
    for i in reversed(stale):
        session.ws_clients.pop(i)


async def run_pipeline(session: PipelineSession, sam2_service: SAM2Service) -> None:
    """Execute the full 6-stage pipeline asynchronously."""
    cfg = session.config
    output_dir = cfg.output_dir
    session.running = True
    session.pipeline_start_time = time.time()

    try:
        # ── Stage 1: Frame Extraction ─────────────────────────────
        await _run_stage(
            session,
            PipelineStage.EXTRACT_FRAMES,
            _stage_extract_frames,
            cfg.video_path,
            output_dir,
            cfg.frame_interval,
            cfg.max_frames,
        )
        session.frames_dir = str(Path(output_dir) / "frames")

        _check_cancelled(session)

        # ── Stage 2: SAM2 Interactive Segmentation ────────────────
        session.stage_start(PipelineStage.SAM2_SEGMENT)
        await broadcast(session, {
            "type": "stage_start",
            "stage": int(PipelineStage.SAM2_SEGMENT),
            "label": STAGE_LABELS[PipelineStage.SAM2_SEGMENT],
        })

        # Initialize SAM2 model in a thread
        meta = await asyncio.to_thread(
            sam2_service.initialize,
            session.frames_dir,
            output_dir,
            cfg.sam2_model,
        )

        session.sam2_frame_count = meta["frame_count"]
        session.sam2_width = meta["width"]
        session.sam2_height = meta["height"]

        # Signal frontend: SAM2 is ready for interaction
        session.stage_interactive(PipelineStage.SAM2_SEGMENT)
        await broadcast(session, {
            "type": "sam2_ready",
            "frame_count": meta["frame_count"],
            "width": meta["width"],
            "height": meta["height"],
        })

        # Wait for user to confirm segmentation via REST API
        session.sam2_confirm_event.clear()
        await session.sam2_confirm_event.wait()

        _check_cancelled(session)

        # Propagate masks
        await broadcast(session, {"type": "sam2_propagating"})

        loop = asyncio.get_event_loop()

        def _propagate_cb(frame_idx: int, total: int) -> None:
            asyncio.run_coroutine_threadsafe(
                broadcast(session, {
                    "type": "sam2_propagate_progress",
                    "frame": frame_idx + 1,
                    "total": total,
                }),
                loop,
            )

        mask_dir = await asyncio.to_thread(
            sam2_service.propagate_and_save, _propagate_cb
        )
        session.mask_dir = mask_dir

        # Release SAM2 model
        await asyncio.to_thread(sam2_service.release)

        session.stage_complete(PipelineStage.SAM2_SEGMENT)
        await broadcast(session, {
            "type": "stage_complete",
            "stage": int(PipelineStage.SAM2_SEGMENT),
            "elapsed": session.stages[int(PipelineStage.SAM2_SEGMENT)].elapsed,
        })

        _check_cancelled(session)

        # VRAM gate
        await asyncio.to_thread(_vram_gate)

        # ── Stage 3: Pi3X 3D Reconstruction ───────────────────────
        await _run_stage(
            session,
            PipelineStage.PI3X_RECONSTRUCT,
            _stage_pi3x,
            session.frames_dir,
            session.mask_dir,
            output_dir,
            cfg.pixel_limit,
            cfg.max_frames,
            cfg.confidence_threshold,
            cfg.edge_rtol,
        )
        session.ply_path = str(Path(output_dir) / "object.ply")
        session.poses_path = str(Path(output_dir) / "camera_poses.json")

        _check_cancelled(session)

        # ── Stage 4: Denoising ────────────────────────────────────
        await _run_stage(
            session,
            PipelineStage.DENOISE,
            _stage_denoise,
            session.ply_path,
            output_dir,
        )
        session.denoised_ply = str(Path(output_dir) / "object_denoised.ply")

        _check_cancelled(session)

        # ── Stage 5: DiffCD Mesh ──────────────────────────────────
        await _run_stage(
            session,
            PipelineStage.DIFFCD_MESH,
            _stage_diffcd,
            session.denoised_ply,
            output_dir,
        )
        session.mesh_ply = str(Path(output_dir) / "object_mesh.ply")

        _check_cancelled(session)

        # ── Stage 6: Texture Bake ─────────────────────────────────
        await _run_stage(
            session,
            PipelineStage.TEXTURE_BAKE,
            _stage_texture_bake,
            session.mesh_ply,
            session.poses_path,
            session.frames_dir,
            session.mask_dir,
            output_dir,
            cfg.texture_size,
        )
        session.obj_path = str(Path(output_dir) / "textured_mesh.obj")

        # ── Complete ──────────────────────────────────────────────
        session.current_stage = PipelineStage.COMPLETE
        elapsed = time.time() - session.pipeline_start_time
        await broadcast(session, {
            "type": "pipeline_complete",
            "elapsed": round(elapsed, 1),
        })

    except _CancelledError:
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": int(session.current_stage),
            "error": "Pipeline cancelled by user",
        })
    except Exception as e:
        stage = int(session.current_stage)
        session.stage_failed(PipelineStage(session.current_stage), str(e))
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": stage,
            "error": str(e),
        })
    finally:
        session.running = False


# ── Helpers ───────────────────────────────────────────────────────

class _CancelledError(Exception):
    pass


def _check_cancelled(session: PipelineSession) -> None:
    if session.cancelled:
        raise _CancelledError()


async def _run_stage(
    session: PipelineSession,
    stage: PipelineStage,
    fn,
    *args,
) -> None:
    """Run a blocking stage function in a thread with lifecycle broadcasts."""
    session.stage_start(stage)
    await broadcast(session, {
        "type": "stage_start",
        "stage": int(stage),
        "label": STAGE_LABELS[stage],
    })
    try:
        await asyncio.to_thread(fn, *args)
    except Exception as e:
        session.stage_failed(stage, str(e))
        await broadcast(session, {
            "type": "stage_complete",
            "stage": int(stage),
            "elapsed": session.stages[int(stage)].elapsed,
            "error": str(e),
        })
        raise
    session.stage_complete(stage)
    await broadcast(session, {
        "type": "stage_complete",
        "stage": int(stage),
        "elapsed": session.stages[int(stage)].elapsed,
    })


# ── Stage wrappers (call existing functions) ──────────────────────

def _stage_extract_frames(video_path: str, output_dir: str, frame_interval: int, max_frames: int) -> None:
    from stage_extract_frames import extract_frames
    extract_frames(video_path, output_dir, frame_interval=frame_interval, max_frames=max_frames)


def _stage_pi3x(
    frames_dir: str, mask_dir: str, output_dir: str,
    pixel_limit: int, max_frames: int,
    conf_threshold: float, edge_rtol: float,
) -> None:
    from stage_pi3x_reconstruct import run_pi3x
    from vram_utils import cleanup_pytorch_vram
    run_pi3x(
        frames_dir, mask_dir, output_dir,
        pixel_limit=pixel_limit,
        max_frames=max_frames,
        conf_threshold=conf_threshold,
        edge_rtol=edge_rtol,
    )
    cleanup_pytorch_vram()


def _stage_denoise(ply_path: str, output_dir: str) -> None:
    from stage_denoise import denoise
    denoise(ply_path, output_dir)


def _stage_diffcd(denoised_ply: str, output_dir: str) -> None:
    from stage_diffcd_mesh import run_diffcd
    run_diffcd(denoised_ply, output_dir)


def _stage_texture_bake(
    mesh_ply: str, poses_path: str, frames_dir: str,
    mask_dir: str, output_dir: str, texture_size: int,
) -> None:
    from stage_texture_bake import bake_texture
    bake_texture(mesh_ply, poses_path, frames_dir, mask_dir, output_dir, tex_size=texture_size)


def _vram_gate() -> None:
    from vram_utils import cleanup_pytorch_vram, ensure_vram_available
    cleanup_pytorch_vram()
    ensure_vram_available(min_free_mb=12000, stage_name="before Pi3X")
