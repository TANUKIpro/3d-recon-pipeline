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


async def _broadcast_stage_progress(
    session: PipelineSession,
    stage: PipelineStage,
    progress: float | None = None,
    detail: str | None = None,
) -> None:
    session.stage_progress(stage, progress=progress, detail=detail)
    info = session.stages[int(stage)]
    await broadcast(session, {
        "type": "stage_progress",
        "stage": int(stage),
        "progress": round(info.progress, 1),
        "detail": info.detail,
        "overall_progress": session.overall_progress(),
    })


async def run_pipeline(session: PipelineSession, sam2_service: SAM2Service) -> None:
    """Execute the full 6-stage pipeline asynchronously."""
    cfg = session.config
    output_dir = cfg.output_dir
    start_stage = int(session.resume_from_stage)
    if start_stage < int(PipelineStage.EXTRACT_FRAMES) or start_stage > int(PipelineStage.TEXTURE_BAKE):
        start_stage = int(PipelineStage.EXTRACT_FRAMES)
    cancelled = False
    session.running = True
    session.cancelled = False
    session.pipeline_start_time = time.time()

    try:
        if start_stage <= int(PipelineStage.EXTRACT_FRAMES):
            # ── Stage 1: Frame Extraction ─────────────────────────
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
            frame_count = len(list(Path(session.frames_dir).glob("*.jpg")))
            session.frame_count = frame_count
            await broadcast(session, {
                "type": "extract_frames_result",
                "frame_count": frame_count,
            })
            _check_cancelled(session)

        if start_stage <= int(PipelineStage.PI3X_RECONSTRUCT):
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")
            # VRAM gate: ensure sufficient free VRAM before Pi3X
            await asyncio.to_thread(_vram_gate)

            # ── Stage 2: Pi3X 3D Reconstruction ───────────────────
            await _run_stage(
                session,
                PipelineStage.PI3X_RECONSTRUCT,
                _stage_pi3x_inference,
                session.frames_dir,
                output_dir,
                cfg.pixel_limit,
                cfg.max_frames,
                cfg.confidence_threshold,
                cfg.edge_rtol,
            )
            session.ply_full_path = str(Path(output_dir) / "object_full.ply")
            session.poses_path = str(Path(output_dir) / "camera_poses.json")
            session.pi3x_cache_path = str(Path(output_dir) / "pi3x_cache.npz")

            # Pi3X preview approval — pause for user to review 3D preview
            await _broadcast_stage_progress(
                session,
                PipelineStage.PI3X_RECONSTRUCT,
                detail="Waiting for preview approval",
            )
            await broadcast(session, {"type": "pi3x_preview_ready"})
            session.pi3x_approve_event.clear()
            await session.pi3x_approve_event.wait()
            _check_cancelled(session)

        if start_stage <= int(PipelineStage.SAM2_SEGMENT):
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")
            _require_file(session.pi3x_cache_path, "Pi3X cache file")

            # ── Stage 3: SAM2 Interactive Segmentation ────────────
            session.stage_start(PipelineStage.SAM2_SEGMENT)
            await broadcast(session, {
                "type": "stage_start",
                "stage": int(PipelineStage.SAM2_SEGMENT),
                "label": STAGE_LABELS[PipelineStage.SAM2_SEGMENT],
            })
            await _broadcast_stage_progress(
                session,
                PipelineStage.SAM2_SEGMENT,
                progress=5.0,
                detail="Initializing SAM2 model",
            )

            # SAM2 interact → propagate → verify loop (supports redo)
            while True:
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
                await _broadcast_stage_progress(
                    session,
                    PipelineStage.SAM2_SEGMENT,
                    progress=20.0,
                    detail="Waiting for interactive clicks",
                )
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
                await _broadcast_stage_progress(
                    session,
                    PipelineStage.SAM2_SEGMENT,
                    progress=30.0,
                    detail="Propagating masks",
                )
                await broadcast(session, {"type": "sam2_propagating"})

                loop = asyncio.get_event_loop()

                def _propagate_cb(frame_idx: int, total: int) -> None:
                    total = max(total, 1)
                    ratio = (frame_idx + 1) / total
                    progress = 30.0 + ratio * 40.0
                    detail = f"Propagating masks ({frame_idx + 1}/{total})"

                    def _push() -> None:
                        session.stage_progress(
                            PipelineStage.SAM2_SEGMENT,
                            progress=progress,
                            detail=detail,
                        )
                        overall = session.overall_progress()
                        asyncio.create_task(
                            broadcast(session, {
                                "type": "sam2_propagate_progress",
                                "frame": frame_idx + 1,
                                "total": total,
                                "progress": round(progress, 1),
                                "overall_progress": overall,
                            })
                        )
                        asyncio.create_task(
                            broadcast(session, {
                                "type": "stage_progress",
                                "stage": int(PipelineStage.SAM2_SEGMENT),
                                "progress": round(progress, 1),
                                "detail": detail,
                                "overall_progress": overall,
                            })
                        )

                    loop.call_soon_threadsafe(_push)

                mask_dir = await asyncio.to_thread(
                    sam2_service.propagate_and_save, _propagate_cb
                )
                session.mask_dir = mask_dir
                session.mask_count = len(list(Path(mask_dir).glob("*.png")))

                # Release SAM2 model
                await asyncio.to_thread(sam2_service.release)

                # Clear event BEFORE notifying frontend to avoid race condition
                session.sam2_approve_event.clear()
                session.sam2_approved = False

                # Signal frontend: verification strip ready
                await _broadcast_stage_progress(
                    session,
                    PipelineStage.SAM2_SEGMENT,
                    progress=72.0,
                    detail="Waiting for mask verification",
                )
                await broadcast(session, {
                    "type": "sam2_verification_ready",
                    "frame_count": meta["frame_count"],
                })

                # Wait for user to approve or redo
                await session.sam2_approve_event.wait()
                _check_cancelled(session)

                if session.sam2_approved:
                    break
                # User chose redo — loop back to re-init SAM2
                await _broadcast_stage_progress(
                    session,
                    PipelineStage.SAM2_SEGMENT,
                    progress=15.0,
                    detail="Redoing SAM2 interaction",
                )

            # Apply SAM2 masks to Pi3X cache
            loop = asyncio.get_event_loop()

            def _mask_progress_cb(progress: float, detail: str | None = None) -> None:
                mapped_progress = 72.0 + (max(0.0, min(100.0, progress)) * 0.28)

                def _push() -> None:
                    session.stage_progress(
                        PipelineStage.SAM2_SEGMENT,
                        progress=mapped_progress,
                        detail=detail,
                    )
                    asyncio.create_task(
                        broadcast(session, {
                            "type": "stage_progress",
                            "stage": int(PipelineStage.SAM2_SEGMENT),
                            "progress": round(mapped_progress, 1),
                            "detail": detail,
                            "overall_progress": session.overall_progress(),
                        })
                    )

                loop.call_soon_threadsafe(_push)

            await asyncio.to_thread(
                _stage_apply_masks,
                session.pi3x_cache_path,
                session.mask_dir,
                output_dir,
                progress_cb=_mask_progress_cb,
            )
            session.ply_path = str(Path(output_dir) / "object.ply")

            session.stage_complete(PipelineStage.SAM2_SEGMENT)
            await broadcast(session, {
                "type": "stage_complete",
                "stage": int(PipelineStage.SAM2_SEGMENT),
                "elapsed": session.stages[int(PipelineStage.SAM2_SEGMENT)].elapsed,
                "overall_progress": session.overall_progress(),
            })
            _check_cancelled(session)

        if start_stage <= int(PipelineStage.DENOISE):
            _require_file(session.ply_path, "Masked point cloud")
            # ── Stage 4: Denoising ────────────────────────────────
            await _run_stage(
                session,
                PipelineStage.DENOISE,
                _stage_denoise,
                session.ply_path,
                output_dir,
            )
            session.denoised_ply = str(Path(output_dir) / "object_denoised.ply")
            _check_cancelled(session)

        if start_stage <= int(PipelineStage.DIFFCD_MESH):
            _require_file(session.denoised_ply, "Denoised point cloud")
            # ── Stage 5: DiffCD Mesh ──────────────────────────────
            await _run_stage(
                session,
                PipelineStage.DIFFCD_MESH,
                _stage_diffcd,
                session.denoised_ply,
                output_dir,
            )
            session.mesh_ply = str(Path(output_dir) / "object_mesh.ply")
            _check_cancelled(session)

        if start_stage <= int(PipelineStage.TEXTURE_BAKE):
            _require_file(session.mesh_ply, "Mesh point cloud")
            _require_file(session.poses_path, "Camera poses file")
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")
            _require_dir(session.mask_dir, "SAM2 masks directory", must_have_suffix=".png")

            # ── Stage 6: Texture Bake ─────────────────────────────
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
            "overall_progress": 100.0,
        })

    except _CancelledError:
        cancelled = True
        stage = _safe_current_stage(session.current_stage)
        if stage is not None:
            info = session.stages[int(stage)]
            if info.status in {StageStatus.RUNNING, StageStatus.INTERACTIVE}:
                session.stage_failed(stage, "Pipeline cancelled by user")
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": int(stage) if stage is not None else int(PipelineStage.IDLE),
            "error": "Pipeline cancelled by user",
        })
    except asyncio.CancelledError:
        cancelled = True
        stage = _safe_current_stage(session.current_stage)
        if stage is not None:
            info = session.stages[int(stage)]
            if info.status in {StageStatus.RUNNING, StageStatus.INTERACTIVE}:
                session.stage_failed(stage, "Pipeline cancelled by user")
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": int(stage) if stage is not None else int(PipelineStage.IDLE),
            "error": "Pipeline cancelled by user",
        })
    except Exception as e:
        stage = _safe_current_stage(session.current_stage)
        if stage is not None:
            session.stage_failed(stage, str(e))
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": int(stage) if stage is not None else int(PipelineStage.IDLE),
            "error": str(e),
        })
    finally:
        try:
            await asyncio.to_thread(sam2_service.release)
        except Exception:
            pass
        session.running = False
        session._task = None
        if cancelled:
            session.hydrate_from_output_dir(output_dir)
            session.cancelled = False
            await broadcast(session, {
                "type": "status",
                **session.to_status_dict(),
            })


# ── Helpers ───────────────────────────────────────────────────────

class _CancelledError(Exception):
    pass


def _safe_current_stage(stage: PipelineStage | int) -> PipelineStage | None:
    try:
        parsed = PipelineStage(int(stage))
    except Exception:
        return None
    if PipelineStage.EXTRACT_FRAMES <= parsed <= PipelineStage.TEXTURE_BAKE:
        return parsed
    return None


def _require_file(path: str | None, label: str) -> None:
    if not path:
        raise FileNotFoundError(f"{label} is missing")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")


def _require_dir(path: str | None, label: str, must_have_suffix: str | None = None) -> None:
    if not path:
        raise FileNotFoundError(f"{label} is missing")
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{label} not found: {p}")
    if must_have_suffix is not None and not any(p.glob(f"*{must_have_suffix}")):
        raise FileNotFoundError(f"{label} has no '{must_have_suffix}' files: {p}")


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
    await _broadcast_stage_progress(session, stage, progress=0.0, detail="Starting")
    loop = asyncio.get_running_loop()

    def _progress_cb(progress: float, detail: str | None = None) -> None:
        def _push() -> None:
            session.stage_progress(stage, progress=progress, detail=detail)
            asyncio.create_task(
                broadcast(session, {
                    "type": "stage_progress",
                    "stage": int(stage),
                    "progress": round(session.stages[int(stage)].progress, 1),
                    "detail": session.stages[int(stage)].detail,
                    "overall_progress": session.overall_progress(),
                })
            )

        loop.call_soon_threadsafe(_push)

    try:
        await asyncio.to_thread(fn, *args, progress_cb=_progress_cb)
    except Exception as e:
        session.stage_failed(stage, str(e))
        await broadcast(session, {
            "type": "stage_complete",
            "stage": int(stage),
            "elapsed": session.stages[int(stage)].elapsed,
            "error": str(e),
            "overall_progress": session.overall_progress(),
        })
        raise
    session.stage_complete(stage)
    await broadcast(session, {
        "type": "stage_complete",
        "stage": int(stage),
        "elapsed": session.stages[int(stage)].elapsed,
        "overall_progress": session.overall_progress(),
    })


# ── Stage wrappers (call existing functions) ──────────────────────

def _stage_extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int,
    max_frames: int,
    progress_cb=None,
) -> None:
    from stage_extract_frames import extract_frames
    extract_frames(
        video_path,
        output_dir,
        frame_interval=frame_interval,
        max_frames=max_frames,
        progress_cb=progress_cb,
    )


def _stage_pi3x_inference(
    frames_dir: str, output_dir: str,
    pixel_limit: int, max_frames: int,
    conf_threshold: float, edge_rtol: float,
    progress_cb=None,
) -> None:
    from stage_pi3x_reconstruct import run_pi3x_inference
    from vram_utils import cleanup_pytorch_vram
    run_pi3x_inference(
        frames_dir, output_dir,
        pixel_limit=pixel_limit,
        max_frames=max_frames,
        conf_threshold=conf_threshold,
        edge_rtol=edge_rtol,
        progress_cb=progress_cb,
    )
    cleanup_pytorch_vram()


def _stage_apply_masks(
    cache_path: str,
    mask_dir: str,
    output_dir: str,
    progress_cb=None,
) -> None:
    from stage_pi3x_reconstruct import apply_sam2_masks
    apply_sam2_masks(cache_path, mask_dir, output_dir, progress_cb=progress_cb)


def _stage_denoise(ply_path: str, output_dir: str, progress_cb=None) -> None:
    from stage_denoise import denoise
    denoise(ply_path, output_dir, progress_cb=progress_cb)


def _stage_diffcd(denoised_ply: str, output_dir: str, progress_cb=None) -> None:
    from stage_diffcd_mesh import run_diffcd
    run_diffcd(denoised_ply, output_dir, progress_cb=progress_cb)


def _stage_texture_bake(
    mesh_ply: str, poses_path: str, frames_dir: str,
    mask_dir: str, output_dir: str, texture_size: int,
    progress_cb=None,
) -> None:
    from stage_texture_bake import bake_texture
    bake_texture(
        mesh_ply,
        poses_path,
        frames_dir,
        mask_dir,
        output_dir,
        tex_size=texture_size,
        progress_cb=progress_cb,
    )


def _vram_gate() -> None:
    from vram_utils import cleanup_pytorch_vram, ensure_vram_available
    cleanup_pytorch_vram()
    ensure_vram_available(min_free_mb=12000, stage_name="before Pi3X")
