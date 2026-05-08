"""Async pipeline orchestrator for the dashboard.

Runs each stage via asyncio.to_thread, broadcasting progress over WebSocket.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.dashboard.checkpoints import (
    cleanup_checkpoint_outputs,
    first_checkpoint_id,
    resolve_checkpoint_id,
)
from scripts.dashboard.log_capture import broadcast_to_clients, stage_log_scope
from scripts.dashboard.stage_wrappers import (
    _stage_colmap_sfm,
    _stage_extract_frames,
    _stage_extract_ground_plane,
    _stage_gs2mesh_reconstruct,
    _stage_post_texture_contact_cleanup_apply,
    _stage_post_texture_contact_cleanup_prepare,
    _stage_post_texture_contact_cleanup_skip,
    _stage_texture_bake,
    _vram_gate,
)
from scripts.dashboard.state import (
    STAGE_LABELS,
    PipelineStage,
    StageStatus,
)
from scripts.output_layout import (
    camera_poses_path,
    cleanup_final_dir,
    cleanup_proposal_dir,
    colmap_sparse_dir,
    frames_dir,
    ground_plane_path,
    intrinsics_path,
    object_mesh_path,
    textured_mesh_obj_path,
)

if TYPE_CHECKING:
    from scripts.dashboard.sam2_service import SAM2Service
    from scripts.dashboard.state import PipelineSession


async def broadcast(session: PipelineSession, msg: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    await broadcast_to_clients(session.ws_clients, msg)


async def _broadcast_stage_progress(
    session: PipelineSession,
    stage: PipelineStage,
    progress: float | None = None,
    detail: str | None = None,
    checkpoint_id: str | None = None,
) -> None:
    payload = _build_stage_progress_payload(
        session,
        stage,
        progress=progress,
        detail=detail,
        checkpoint_id=checkpoint_id,
    )
    await broadcast(session, payload)


def _build_stage_progress_payload(
    session: PipelineSession,
    stage: PipelineStage,
    progress: float | None = None,
    detail: str | None = None,
    checkpoint_id: str | None = None,
) -> dict:
    stage_num = int(stage)
    current = None
    if int(session.current_stage) == stage_num:
        current = session.current_checkpoint_id
    resolved_checkpoint = checkpoint_id or resolve_checkpoint_id(
        stage_num,
        detail,
        current_checkpoint_id=current,
    )
    session.stage_progress(
        stage,
        progress=progress,
        detail=detail,
        checkpoint_id=resolved_checkpoint,
    )
    info = session.stages[stage_num]
    return {
        "type": "stage_progress",
        "stage": stage_num,
        "progress": round(info.progress, 1),
        "detail": info.detail,
        "checkpoint_id": info.checkpoint_id,
        "overall_progress": session.overall_progress(),
    }


def _cleanup_cancelled_outputs(
    session: PipelineSession,
    stage: PipelineStage | None,
) -> dict | None:
    if stage is None:
        return None
    info = session.stages.get(int(stage))
    if info is not None and info.status == StageStatus.COMPLETE:
        return {
            "stage": int(stage),
            "checkpoint_id": session.current_checkpoint_id,
            "skipped": True,
            "reason": "stage_already_complete",
            "removed_dirs": [],
            "removed_files": [],
        }
    output_dir = session.config.output_dir
    if not output_dir:
        return None
    return cleanup_checkpoint_outputs(
        output_dir,
        int(stage),
        session.current_checkpoint_id,
    )


async def run_pipeline(session: PipelineSession, sam2_service: SAM2Service) -> None:
    """Execute the full 6-stage pipeline asynchronously."""
    cfg = session.config
    output_dir = cfg.output_dir
    start_stage = int(session.resume_from_stage)
    if start_stage < int(PipelineStage.EXTRACT_FRAMES) or start_stage > int(PipelineStage.POST_TEXTURE_CONTACT_CLEANUP):
        start_stage = int(PipelineStage.EXTRACT_FRAMES)
    cancelled = False
    session.running = True
    session.clear_cancel()
    session.current_checkpoint_id = None
    session.pipeline_start_time = time.time()

    # Clear residual GPU memory from previous pipeline runs before
    # starting GPU-intensive stages.
    try:
        from scripts.vram_utils import cleanup_pytorch_vram, log_vram_detailed
        await asyncio.to_thread(cleanup_pytorch_vram)
        await asyncio.to_thread(log_vram_detailed, "pipeline start")
    except Exception:
        pass

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
            session.frames_dir = str(frames_dir(output_dir))
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
                PipelineStage.COLMAP_SFM,
                "Extract Frames complete. Continue to COLMAP SfM?",
            )

        if start_stage <= int(PipelineStage.COLMAP_SFM):
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")

            # ── Stage 2: COLMAP SfM ──────────────────────────────
            await _run_stage(
                session,
                PipelineStage.COLMAP_SFM,
                _stage_colmap_sfm,
                session.frames_dir,
                output_dir,
                cfg.colmap_matcher,
                cfg.colmap_max_features,
                cfg.colmap_image_size,
                cfg.colmap_use_gpu,
                cfg.colmap_dsp_sift,
                cfg.colmap_first_octave,
            )
            session.poses_path = str(camera_poses_path(output_dir))
            session.colmap_sparse_path = str(colmap_sparse_dir(output_dir))

            # Pause for user to review COLMAP results
            await broadcast(session, {"type": "colmap_preview_ready"})
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.COLMAP_SFM,
                PipelineStage.SAM2_SEGMENT,
                "COLMAP SfM complete. Continue to SAM2 Segmentation?",
            )

        if start_stage <= int(PipelineStage.SAM2_SEGMENT):
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")

            # ── Stage 3: SAM2 Interactive Segmentation ────────────
            session.stage_start(PipelineStage.SAM2_SEGMENT)
            sam2_checkpoint = first_checkpoint_id(int(PipelineStage.SAM2_SEGMENT))
            if sam2_checkpoint:
                session.stage_progress(PipelineStage.SAM2_SEGMENT, checkpoint_id=sam2_checkpoint)
            await broadcast(session, {
                "type": "stage_start",
                "stage": int(PipelineStage.SAM2_SEGMENT),
                "label": STAGE_LABELS[PipelineStage.SAM2_SEGMENT],
                "checkpoint_id": sam2_checkpoint,
            })
            await _broadcast_stage_progress(
                session,
                PipelineStage.SAM2_SEGMENT,
                progress=5.0,
                detail="Initializing SAM2 model",
            )

            with stage_log_scope(int(PipelineStage.SAM2_SEGMENT)):
                # SAM2 interact → propagate → verify loop (supports redo)
                ground_mask_dir: str | None = None
                while True:
                    # Clear all SAM2 events at redo loop top
                    session.sam2_confirm_event.clear()
                    session.sam2_ground_skip_event.clear()
                    session.sam2_approve_event.clear()
                    session.sam2_approved = False

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

                    # Wait for user to confirm object segmentation via REST API
                    await session.sam2_confirm_event.wait()
                    _check_cancelled(session)

                    # Ground plane segmentation phase
                    if cfg.ground_plane_enabled:
                        # Switch service to ground mode
                        sam2_service.set_mode("ground")
                        session.sam2_confirm_event.clear()

                        await _broadcast_stage_progress(
                            session,
                            PipelineStage.SAM2_SEGMENT,
                            progress=22.0,
                            detail="Waiting for ground plane segmentation",
                        )
                        await broadcast(session, {
                            "type": "sam2_ground_phase",
                            "frame_count": meta["frame_count"],
                            "width": meta["width"],
                            "height": meta["height"],
                        })

                        # Wait for confirm OR skip
                        confirm_task = asyncio.create_task(session.sam2_confirm_event.wait())
                        skip_task = asyncio.create_task(session.sam2_ground_skip_event.wait())
                        done, pending = await asyncio.wait(
                            {confirm_task, skip_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()
                        _check_cancelled(session)

                        if session.sam2_ground_skip_event.is_set():
                            await broadcast(session, {"type": "sam2_ground_skipped"})

                        # Switch back to object mode
                        sam2_service.set_mode("object")

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
                        _check_cancelled(session)
                        total = max(total, 1)
                        ratio = (frame_idx + 1) / total
                        progress = 30.0 + ratio * 40.0
                        detail = f"Propagating masks ({frame_idx + 1}/{total})"

                        def _push() -> None:
                            payload = _build_stage_progress_payload(
                                session,
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
                                broadcast(session, payload)
                            )

                        loop.call_soon_threadsafe(_push)

                    propagate_result = await asyncio.to_thread(
                        sam2_service.propagate_and_save, _propagate_cb
                    )
                    mask_dir, ground_mask_dir = propagate_result
                    session.mask_dir = mask_dir
                    session.mask_count = len(list(Path(mask_dir).glob("*.png")))
                    session.ground_mask_dir = ground_mask_dir

                    # Drain any broadcast tasks scheduled by _propagate_cb so
                    # their closures drop their session references before we
                    # release the SAM2 model.
                    await asyncio.sleep(0)

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
                        "has_ground": ground_mask_dir is not None,
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
                PipelineStage.GS2MESH_RECONSTRUCT,
                "SAM2 Segmentation complete. Continue to gs2mesh Reconstruction?",
            )

        if start_stage <= int(PipelineStage.GS2MESH_RECONSTRUCT):
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")
            _require_file(session.poses_path, "Camera poses file")

            # VRAM gate: ensure sufficient free VRAM before gs2mesh
            with stage_log_scope(int(PipelineStage.GS2MESH_RECONSTRUCT)):
                await asyncio.to_thread(_vram_gate)

            # ── Stage 4: gs2mesh Reconstruction ──────────────────
            await _run_stage(
                session,
                PipelineStage.GS2MESH_RECONSTRUCT,
                _stage_gs2mesh_reconstruct,
                session.frames_dir,
                session.colmap_sparse_path or str(colmap_sparse_dir(output_dir)),
                session.mask_dir,
                output_dir,
                cfg.to_gs2mesh_settings(),
            )
            session.mesh_ply = str(object_mesh_path(output_dir))
            _check_cancelled(session)
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.GS2MESH_RECONSTRUCT,
                PipelineStage.TEXTURE_BAKE,
                "gs2mesh Reconstruction complete. Continue to Texture Bake?",
            )

        if start_stage <= int(PipelineStage.TEXTURE_BAKE):
            _require_file(session.mesh_ply, "Reconstructed mesh")
            _require_file(session.poses_path, "Camera poses file")
            _require_dir(session.frames_dir, "Extracted frames directory", must_have_suffix=".jpg")
            _require_dir(session.mask_dir, "SAM2 masks directory", must_have_suffix=".png")

            # ── Stage 5: Texture Bake ─────────────────────────────
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
                cfg.texture_view_assign_mode,
                cfg.texture_quality_boost,
            )
            session.obj_path = str(textured_mesh_obj_path(output_dir))
            _check_cancelled(session)
            await _wait_for_next_stage_confirmation(
                session,
                PipelineStage.TEXTURE_BAKE,
                PipelineStage.POST_TEXTURE_CONTACT_CLEANUP,
                "Texture Bake complete. Continue to Post Cleanup?",
            )

        if start_stage <= int(PipelineStage.POST_TEXTURE_CONTACT_CLEANUP):
            _require_file(session.obj_path, "Textured mesh")
            await _run_post_texture_cleanup_stage(session)
            session.cleaned_obj_path = str(
                cleanup_final_dir(output_dir) / "textured_mesh_cleaned.obj"
            )
            proposal_path = cleanup_proposal_dir(output_dir) / "proposal.json"
            session.cleanup_proposal_path = str(proposal_path) if proposal_path.is_file() else None

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
        cleanup_meta = None
        try:
            cleanup_meta = _cleanup_cancelled_outputs(session, stage)
        except Exception as cleanup_error:
            cleanup_meta = {
                "stage": int(stage) if stage is not None else None,
                "checkpoint_id": session.current_checkpoint_id,
                "error": str(cleanup_error),
            }
        if stage is not None:
            info = session.stages[int(stage)]
            if info.status in {StageStatus.RUNNING, StageStatus.INTERACTIVE}:
                session.stage_failed(stage, "Pipeline cancelled by user")
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": int(stage) if stage is not None else int(PipelineStage.IDLE),
            "error": "Pipeline cancelled by user",
            "reason_code": "cancelled_force" if session.cancel_force else "cancelled",
            "checkpoint_id": session.current_checkpoint_id,
            "cleanup": cleanup_meta,
            "overall_progress": session.overall_progress(),
        })
    except asyncio.CancelledError:
        cancelled = True
        stage = _safe_current_stage(session.current_stage)
        cleanup_meta = None
        try:
            cleanup_meta = _cleanup_cancelled_outputs(session, stage)
        except Exception as cleanup_error:
            cleanup_meta = {
                "stage": int(stage) if stage is not None else None,
                "checkpoint_id": session.current_checkpoint_id,
                "error": str(cleanup_error),
            }
        if stage is not None:
            info = session.stages[int(stage)]
            if info.status in {StageStatus.RUNNING, StageStatus.INTERACTIVE}:
                session.stage_failed(stage, "Pipeline cancelled by user")
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": int(stage) if stage is not None else int(PipelineStage.IDLE),
            "error": "Pipeline cancelled by user",
            "reason_code": "cancelled_force" if session.cancel_force else "cancelled",
            "checkpoint_id": session.current_checkpoint_id,
            "cleanup": cleanup_meta,
            "overall_progress": session.overall_progress(),
        })
    except Exception as e:
        stage = _safe_current_stage(session.current_stage)
        if stage is not None:
            session.stage_failed(stage, str(e))
        await broadcast(session, {
            "type": "pipeline_error",
            "stage": int(stage) if stage is not None else int(PipelineStage.IDLE),
            "error": str(e),
            "checkpoint_id": session.current_checkpoint_id,
        })
    finally:
        try:
            session.terminate_active_processes(grace_seconds=0.3)
        except Exception:
            pass
        try:
            await asyncio.to_thread(sam2_service.release)
        except Exception:
            pass
        try:
            from scripts.vram_utils import cleanup_pytorch_vram
            await asyncio.to_thread(cleanup_pytorch_vram)
        except Exception:
            pass
        session.clear_next_stage_confirmation()
        session.running = False
        session._task = None
        if cancelled:
            session.hydrate_from_output_dir(output_dir)
            session.clear_cancel()
            await broadcast(session, {
                "type": "status",
                **session.to_status_dict(),
            })


# ── Helpers ───────────────────────────────────────────────────────

class _CancelledError(Exception):
    def __init__(self, message: str = "Pipeline cancelled by user") -> None:
        super().__init__(message)


def _safe_current_stage(stage: PipelineStage | int) -> PipelineStage | None:
    try:
        parsed = PipelineStage(int(stage))
    except Exception:
        return None
    if PipelineStage.EXTRACT_FRAMES <= parsed <= PipelineStage.POST_TEXTURE_CONTACT_CLEANUP:
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
    if session.cancelled or session.cancel_requested or session.cancel_event.is_set():
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
    auto_accepted = session.config.auto_accept
    await broadcast(
        session,
        {
            "type": "next_stage_confirmation_required",
            "from_stage": int(from_stage),
            "to_stage": int(to_stage),
            "message": message,
            "auto_accepted": auto_accepted,
            "overall_progress": session.overall_progress(),
        },
    )
    if auto_accepted:
        session.next_stage_confirm_event.set()
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


def _make_progress_cb(session, stage, loop):
    """Create a thread-safe progress callback for broadcasting stage progress."""

    def _progress_cb(
        progress: float,
        detail: str | None = None,
        checkpoint_id: str | None = None,
    ) -> None:
        _check_cancelled(session)

        def _push() -> None:
            payload = _build_stage_progress_payload(
                session,
                stage,
                progress=progress,
                detail=detail,
                checkpoint_id=checkpoint_id,
            )
            asyncio.create_task(broadcast(session, payload))

        loop.call_soon_threadsafe(_push)

    return _progress_cb


def _make_cancel_cb(session):
    """Create a cancel callback that raises on cancellation."""

    def _cancel_cb() -> None:
        _check_cancelled(session)

    return _cancel_cb


async def _run_stage(
    session: PipelineSession,
    stage: PipelineStage,
    fn,
    *args,
    label: str | None = None,
) -> None:
    """Run a blocking stage function in a thread with lifecycle broadcasts."""
    session.stage_start(stage)
    checkpoint_id = first_checkpoint_id(int(stage))
    if checkpoint_id:
        session.stage_progress(stage, checkpoint_id=checkpoint_id)
    await broadcast(session, {
        "type": "stage_start",
        "stage": int(stage),
        "label": label or STAGE_LABELS[stage],
        "checkpoint_id": checkpoint_id,
    })
    await _broadcast_stage_progress(session, stage, progress=0.0, detail="Starting")
    loop = asyncio.get_running_loop()
    _progress_cb = _make_progress_cb(session, stage, loop)
    _cancel_cb = _make_cancel_cb(session)

    def _run_with_stage_scope() -> None:
        with stage_log_scope(int(stage)):
            fn(
                *args,
                progress_cb=_progress_cb,
                cancel_cb=_cancel_cb,
                register_process=session.register_active_process,
                unregister_process=session.unregister_active_process,
            )

    try:
        await asyncio.to_thread(_run_with_stage_scope)
    except _CancelledError:
        raise
    except Exception as e:
        session.stage_failed(stage, str(e))
        await broadcast(session, {
            "type": "stage_complete",
            "stage": int(stage),
            "elapsed": session.stages[int(stage)].elapsed,
            "error": str(e),
            "checkpoint_id": session.stages[int(stage)].checkpoint_id,
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

async def _run_post_texture_cleanup_stage(session: PipelineSession) -> None:
    stage = PipelineStage.POST_TEXTURE_CONTACT_CLEANUP
    output_dir = str(session.config.output_dir)
    session.stage_start(stage)
    checkpoint_id = first_checkpoint_id(int(stage))
    if checkpoint_id:
        session.stage_progress(stage, checkpoint_id=checkpoint_id)
    await broadcast(
        session,
        {
            "type": "stage_start",
            "stage": int(stage),
            "label": STAGE_LABELS[stage],
            "checkpoint_id": checkpoint_id,
        },
    )
    await _broadcast_stage_progress(
        session,
        stage,
        progress=0.0,
        detail="Starting",
        checkpoint_id=checkpoint_id,
    )

    loop = asyncio.get_running_loop()
    progress_cb = _make_progress_cb(session, stage, loop)
    cancel_cb = _make_cancel_cb(session)

    def _run_blocking(fn, *args):
        with stage_log_scope(int(stage)):
            return fn(
                *args,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                register_process=session.register_active_process,
                unregister_process=session.unregister_active_process,
            )

    try:
        # Extract ground plane from mesh + SAM2 ground masks if not yet done
        if session.ground_plane_path is None and session.ground_mask_dir is not None:
            intrinsics_file = intrinsics_path(output_dir)
            mesh_ply = session.mesh_ply or str(object_mesh_path(output_dir))
            if Path(mesh_ply).is_file() and intrinsics_file.is_file() and session.poses_path:
                await _broadcast_stage_progress(
                    session, stage, progress=5.0,
                    detail="Extracting ground plane from mesh",
                )
                gp_result = await asyncio.to_thread(
                    _run_blocking,
                    _stage_extract_ground_plane,
                    mesh_ply,
                    session.ground_mask_dir,
                    output_dir,
                    session.poses_path,
                    str(intrinsics_file),
                    session.mask_dir,
                )
                if gp_result is not None:
                    gp_path = ground_plane_path(output_dir)
                    session.ground_plane_path = str(gp_path) if gp_path.is_file() else None

        intrinsics_file = intrinsics_path(output_dir)
        proposal = await asyncio.to_thread(
            _run_blocking,
            _stage_post_texture_contact_cleanup_prepare,
            str(session.obj_path),
            output_dir,
            session.poses_path,
            str(intrinsics_file) if intrinsics_file.is_file() else None,
            session.mask_dir,
            session.ground_mask_dir,
            session.ground_plane_path,
            session.config.cleanup_lower_half_threshold,
        )

        proposal_path = cleanup_proposal_dir(output_dir) / "proposal.json"
        session.cleanup_proposal_path = str(proposal_path) if proposal_path.is_file() else None

        decision = "skip"
        if session.config.post_texture_cleanup_enabled and proposal.get("requires_review") is True:
            session.cleanup_review_event.clear()
            session.cleanup_decision = None
            auto_accepted = bool(session.config.auto_accept)
            if auto_accepted:
                session.cleanup_decision = str(proposal.get("recommended_decision") or "apply")
            session.stage_interactive(stage)
            await _broadcast_stage_progress(
                session,
                stage,
                progress=max(session.stages[int(stage)].progress, 74.0),
                detail="Waiting for cleanup review decision",
                checkpoint_id="s6.review",
            )
            await broadcast(
                session,
                {
                    "type": "post_texture_cleanup_review_ready",
                    "proposal": proposal,
                    "proposal_path": session.cleanup_proposal_path,
                    "auto_accepted": auto_accepted,
                    "overall_progress": session.overall_progress(),
                },
            )
            if auto_accepted:
                session.cleanup_review_event.set()
            await session.cleanup_review_event.wait()
            _check_cancelled(session)
            decision = str(session.cleanup_decision or "skip")
        elif session.config.post_texture_cleanup_enabled and proposal.get("recommended_decision") == "apply":
            decision = "apply"

        await _broadcast_stage_progress(
            session,
            stage,
            progress=max(session.stages[int(stage)].progress, 82.0),
            detail=f"Applying cleanup decision ({decision})",
            checkpoint_id="s6.apply",
        )
        if decision == "apply":
            cleaned_obj = await asyncio.to_thread(
                _run_blocking,
                _stage_post_texture_contact_cleanup_apply,
                output_dir,
                session.config.cleanup_lower_half_threshold,
            )
        else:
            cleaned_obj = await asyncio.to_thread(
                _run_blocking,
                _stage_post_texture_contact_cleanup_skip,
                output_dir,
            )
        session.cleaned_obj_path = str(cleaned_obj)
    except _CancelledError:
        raise
    except Exception as e:
        session.stage_failed(stage, str(e))
        await broadcast(
            session,
            {
                "type": "stage_complete",
                "stage": int(stage),
                "elapsed": session.stages[int(stage)].elapsed,
                "error": str(e),
                "checkpoint_id": session.stages[int(stage)].checkpoint_id,
                "overall_progress": session.overall_progress(),
            },
        )
        raise

    session.stage_complete(stage)
    await broadcast(
        session,
        {
            "type": "stage_complete",
            "stage": int(stage),
            "elapsed": session.stages[int(stage)].elapsed,
            "overall_progress": session.overall_progress(),
        },
    )
