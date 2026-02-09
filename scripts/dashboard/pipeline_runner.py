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

from scripts.dashboard.log_capture import stage_log_scope
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


def _mesh_method_key(value: str | None) -> str:
    method = str(value or "").strip().lower()
    if method == "diffcd":
        return "diffcd"
    return "poisson"


def _mesh_method_label(method: str) -> str:
    if method == "diffcd":
        return "Learning Mesh (DiffCD)"
    return "Classical Mesh (Pre -> Main -> Post -> Downsample)"


def _preview_rel_path(output_dir: str, path: str | Path) -> str:
    out = Path(output_dir).resolve()
    target = Path(path).resolve()
    try:
        rel = target.relative_to(out)
        return str(rel).replace("\\", "/")
    except Exception:
        return Path(path).name


async def _broadcast_classical_preview_update(
    session: PipelineSession,
    output_dir: str,
    *,
    step: str,
    file_path: str | Path,
    detail: str,
) -> None:
    await broadcast(session, {
        "type": "stage_preview_update",
        "stage": int(PipelineStage.DIFFCD_MESH),
        "mesh_method": "poisson",
        "step": step,
        "file": _preview_rel_path(output_dir, file_path),
        "detail": detail,
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
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.EXTRACT_FRAMES,
                PipelineStage.PI3X_RECONSTRUCT,
                "Extract Frames complete. Continue to Pi3X 3D Reconstruction?",
            )

        if start_stage <= int(PipelineStage.PI3X_RECONSTRUCT):
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")
            # VRAM gate: ensure sufficient free VRAM before Pi3X
            with stage_log_scope(int(PipelineStage.PI3X_RECONSTRUCT)):
                await asyncio.to_thread(_vram_gate)

            # ── Stage 2: Pi3X 3D Reconstruction ───────────────────
            await _run_stage(
                session,
                PipelineStage.PI3X_RECONSTRUCT,
                _stage_pi3x_inference,
                session.frames_dir,
                output_dir,
                cfg.pixel_limit,
                cfg.pi3x_frame_target,
                cfg.confidence_threshold,
                cfg.edge_rtol,
            )
            session.ply_full_path = str(Path(output_dir) / "object_full.ply")
            session.poses_path = str(Path(output_dir) / "camera_poses.json")
            session.pi3x_cache_path = str(Path(output_dir) / "pi3x_cache.npz")

            # Pause for user to review the 3D preview before moving to Stage 3.
            await broadcast(session, {"type": "pi3x_preview_ready"})
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.PI3X_RECONSTRUCT,
                PipelineStage.SAM2_SEGMENT,
                "Pi3X 3D Reconstruction complete. Continue to SAM2 Segmentation?",
            )

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

            with stage_log_scope(int(PipelineStage.SAM2_SEGMENT)):
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
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.SAM2_SEGMENT,
                PipelineStage.DENOISE,
                "SAM2 Segmentation complete. Continue to Point Cloud Denoise?",
            )

        if start_stage <= int(PipelineStage.DENOISE):
            _require_file(session.ply_path, "Masked point cloud")
            # ── Stage 4: Denoising ────────────────────────────────
            await _run_stage(
                session,
                PipelineStage.DENOISE,
                _stage_denoise,
                session.ply_path,
                output_dir,
                cfg.denoise_preset,
                cfg.denoise_algorithm,
                cfg.denoise_dbscan_eps,
                cfg.denoise_dbscan_eps_ratio,
                cfg.denoise_dbscan_min_samples,
                cfg.denoise_dbscan_max_points,
                cfg.denoise_sor_neighbors,
                cfg.denoise_sor_std_ratio,
                cfg.denoise_radius_neighbors,
                cfg.denoise_radius_radius_ratio,
            )
            session.denoised_ply = str(Path(output_dir) / "object_denoised.ply")
            _check_cancelled(session)
            mesh_method = _mesh_method_key(cfg.mesh_method)
            mesh_label = _mesh_method_label(mesh_method)
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.DENOISE,
                PipelineStage.DIFFCD_MESH,
                f"Point Cloud Denoise complete. Continue to {mesh_label}?",
            )

        if start_stage <= int(PipelineStage.DIFFCD_MESH):
            _require_file(session.denoised_ply, "Denoised point cloud")
            # ── Stage 5: Mesh Reconstruction ──────────────────────
            mesh_method = _mesh_method_key(cfg.mesh_method)
            mesh_label = _mesh_method_label(mesh_method)
            if mesh_method == "diffcd":
                await _run_stage(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    _stage_diffcd,
                    session.denoised_ply,
                    output_dir,
                    label=mesh_label,
                )
            else:
                preprocess_ply = Path(output_dir) / "object_mesh_input.ply"
                raw_mesh_ply = Path(output_dir) / "object_mesh_raw.ply"
                post_mesh_ply = Path(output_dir) / "object_mesh_postprocessed.ply"
                final_mesh_ply = Path(output_dir) / "object_mesh.ply"

                # Single stage lifecycle for the entire Classical Mesh flow
                session.stage_start(PipelineStage.DIFFCD_MESH)
                await broadcast(session, {
                    "type": "stage_start",
                    "stage": int(PipelineStage.DIFFCD_MESH),
                    "label": mesh_label,
                })

                # Sub-phase 1: Preprocess (0-24%)
                await _run_sub_stage(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    _stage_classical_preprocess,
                    session.denoised_ply,
                    output_dir,
                    label="Classical/Preprocess",
                    progress_start=0.0,
                    progress_end=24.0,
                )
                _check_cancelled(session)
                _require_file(str(preprocess_ply), "Classical preprocessed point cloud")
                await _broadcast_classical_preview_update(
                    session,
                    output_dir,
                    step="preprocess",
                    file_path=preprocess_ply,
                    detail="Classical preprocess output ready",
                )
                await _wait_for_next_stage_confirmation(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    PipelineStage.DIFFCD_MESH,
                    "Classical preprocess complete. Continue to Main Poisson?",
                )

                # Sub-phase 2: Main Poisson (24-72%)
                await _run_sub_stage(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    _stage_classical_main,
                    str(preprocess_ply),
                    output_dir,
                    label="Classical/Main",
                    progress_start=24.0,
                    progress_end=72.0,
                )
                _check_cancelled(session)
                _require_file(str(raw_mesh_ply), "Classical raw mesh")
                await _broadcast_classical_preview_update(
                    session,
                    output_dir,
                    step="main",
                    file_path=raw_mesh_ply,
                    detail="Classical main Poisson output ready",
                )
                await _wait_for_next_stage_confirmation(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    PipelineStage.DIFFCD_MESH,
                    "Classical main Poisson complete. Continue to Postprocess?",
                )

                # Sub-phase 3: Postprocess (72-92%)
                await _run_sub_stage(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    _stage_classical_postprocess,
                    str(raw_mesh_ply),
                    output_dir,
                    label="Classical/Postprocess",
                    progress_start=72.0,
                    progress_end=92.0,
                )
                _check_cancelled(session)
                _require_file(str(post_mesh_ply), "Classical postprocessed mesh")
                await _broadcast_classical_preview_update(
                    session,
                    output_dir,
                    step="postprocess",
                    file_path=post_mesh_ply,
                    detail="Classical postprocess output ready",
                )
                await _wait_for_next_stage_confirmation(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    PipelineStage.DIFFCD_MESH,
                    "Classical postprocess complete. Continue to Mesh Downsample?",
                )

                # Sub-phase 4: Downsample (92-100%)
                await _run_sub_stage(
                    session,
                    PipelineStage.DIFFCD_MESH,
                    _stage_classical_downsample,
                    str(post_mesh_ply),
                    output_dir,
                    label="Classical/Downsample",
                    progress_start=92.0,
                    progress_end=100.0,
                )
                _check_cancelled(session)
                _require_file(str(final_mesh_ply), "Classical final mesh")
                await _broadcast_classical_preview_update(
                    session,
                    output_dir,
                    step="downsample",
                    file_path=final_mesh_ply,
                    detail="Classical downsample output ready",
                )

                # Complete the stage once all sub-phases are done
                session.stage_complete(PipelineStage.DIFFCD_MESH)
                await broadcast(session, {
                    "type": "stage_complete",
                    "stage": int(PipelineStage.DIFFCD_MESH),
                    "elapsed": session.stages[int(PipelineStage.DIFFCD_MESH)].elapsed,
                    "overall_progress": session.overall_progress(),
                })

            session.mesh_ply = str(Path(output_dir) / "object_mesh.ply")
            _check_cancelled(session)
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.DIFFCD_MESH,
                PipelineStage.TEXTURE_BAKE,
                f"{mesh_label} complete. Continue to Texture Bake?",
            )

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
        session.clear_next_stage_confirmation()
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


async def _wait_for_next_stage_confirmation(
    session: PipelineSession,
    from_stage: PipelineStage,
    to_stage: PipelineStage,
    message: str,
) -> None:
    session.require_next_stage_confirmation(from_stage, to_stage, message)
    await _broadcast_stage_progress(
        session,
        from_stage,
        detail="Waiting for next-stage confirmation",
    )
    await broadcast(
        session,
        {
            "type": "next_stage_confirmation_required",
            "from_stage": int(from_stage),
            "to_stage": int(to_stage),
            "message": message,
            "overall_progress": session.overall_progress(),
        },
    )
    await session.next_stage_confirm_event.wait()
    session.clear_next_stage_confirmation()
    await broadcast(
        session,
        {
            "type": "next_stage_confirmation_cleared",
            "from_stage": int(from_stage),
            "to_stage": int(to_stage),
            "overall_progress": session.overall_progress(),
        },
    )
    _check_cancelled(session)


async def _run_stage(
    session: PipelineSession,
    stage: PipelineStage,
    fn,
    *args,
    label: str | None = None,
) -> None:
    """Run a blocking stage function in a thread with lifecycle broadcasts."""
    session.stage_start(stage)
    await broadcast(session, {
        "type": "stage_start",
        "stage": int(stage),
        "label": label or STAGE_LABELS[stage],
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

    def _run_with_stage_scope() -> None:
        with stage_log_scope(int(stage)):
            fn(*args, progress_cb=_progress_cb)

    try:
        await asyncio.to_thread(_run_with_stage_scope)
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


async def _run_sub_stage(
    session: PipelineSession,
    stage: PipelineStage,
    fn,
    *args,
    label: str | None = None,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
) -> None:
    """Run a sub-phase within an already-started stage.

    Unlike ``_run_stage``, this does NOT send ``stage_start`` / ``stage_complete``.
    It maps the sub-phase's internal progress (0-100%) into the caller-specified
    range (``progress_start`` .. ``progress_end``) of the overall stage, and
    sends ``stage_progress`` messages with the mapped value.

    On error it marks the stage as failed and broadcasts ``stage_complete`` with
    the error (so the frontend learns of the failure), then re-raises.
    """
    detail_prefix = f"{label}: " if label else ""
    await _broadcast_stage_progress(
        session, stage,
        progress=progress_start,
        detail=f"{detail_prefix}Starting",
    )
    loop = asyncio.get_running_loop()
    span = max(0.0, progress_end - progress_start)

    def _progress_cb(progress: float, detail: str | None = None) -> None:
        clamped = max(0.0, min(100.0, float(progress)))
        mapped = progress_start + (clamped / 100.0) * span

        def _push() -> None:
            session.stage_progress(stage, progress=mapped, detail=detail)
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

    def _run_with_stage_scope() -> None:
        with stage_log_scope(int(stage)):
            fn(*args, progress_cb=_progress_cb)

    try:
        await asyncio.to_thread(_run_with_stage_scope)
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

    # Update progress to the end of this sub-phase range (don't mark stage complete)
    await _broadcast_stage_progress(
        session, stage,
        progress=progress_end,
        detail=f"{detail_prefix}Done",
    )


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
    pixel_limit: int, pi3x_frame_target: int,
    conf_threshold: float, edge_rtol: float,
    progress_cb=None,
) -> None:
    from stage_pi3x_reconstruct import run_pi3x_inference
    from vram_utils import cleanup_pytorch_vram
    run_pi3x_inference(
        frames_dir, output_dir,
        pixel_limit=pixel_limit,
        max_frames=pi3x_frame_target,
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


def _stage_denoise(
    ply_path: str,
    output_dir: str,
    denoise_preset: str,
    denoise_algorithm: str,
    denoise_dbscan_eps: float,
    denoise_dbscan_eps_ratio: float,
    denoise_dbscan_min_samples: int,
    denoise_dbscan_max_points: int,
    denoise_sor_neighbors: int,
    denoise_sor_std_ratio: float,
    denoise_radius_neighbors: int,
    denoise_radius_radius_ratio: float,
    progress_cb=None,
) -> None:
    from stage_denoise import denoise
    denoise(
        ply_path,
        output_dir,
        preset=denoise_preset,
        algorithm=denoise_algorithm,
        dbscan_eps=denoise_dbscan_eps,
        dbscan_eps_ratio=denoise_dbscan_eps_ratio,
        dbscan_min_samples=denoise_dbscan_min_samples,
        dbscan_max_points=denoise_dbscan_max_points,
        sor_neighbors=denoise_sor_neighbors,
        sor_std_ratio=denoise_sor_std_ratio,
        radius_neighbors=denoise_radius_neighbors,
        radius_ratio=denoise_radius_radius_ratio,
        progress_cb=progress_cb,
    )


def _stage_diffcd(denoised_ply: str, output_dir: str, progress_cb=None) -> None:
    from stage_diffcd_mesh import run_diffcd
    run_diffcd(denoised_ply, output_dir, progress_cb=progress_cb)


def _stage_classical_mesh(denoised_ply: str, output_dir: str, progress_cb=None) -> None:
    from stage_classical_mesh import run_classical_mesh

    run_classical_mesh(denoised_ply, output_dir, progress_cb=progress_cb)


def _stage_classical_preprocess(denoised_ply: str, output_dir: str, progress_cb=None) -> None:
    from stage_classical_mesh import run_classical_preprocess

    run_classical_preprocess(denoised_ply, output_dir, progress_cb=progress_cb)


def _stage_classical_main(preprocess_ply: str, output_dir: str, progress_cb=None) -> None:
    from stage_classical_mesh import run_classical_main

    run_classical_main(preprocess_ply, output_dir, progress_cb=progress_cb)


def _stage_classical_postprocess(raw_mesh_ply: str, output_dir: str, progress_cb=None) -> None:
    from stage_classical_mesh import run_classical_postprocess

    run_classical_postprocess(raw_mesh_ply, output_dir, progress_cb=progress_cb)


def _stage_classical_downsample(post_mesh_ply: str, output_dir: str, progress_cb=None) -> None:
    from stage_classical_mesh import run_classical_downsample

    run_classical_downsample(post_mesh_ply, output_dir, progress_cb=progress_cb)


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
