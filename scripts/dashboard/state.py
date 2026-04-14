"""Pipeline state management for the web dashboard."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

from scripts.config_defaults import (
    COLMAP_IMAGE_SIZE,
    COLMAP_MAX_FEATURES,
    COLMAP_MATCHER,
    COLMAP_DSP_SIFT,
    COLMAP_FIRST_OCTAVE,
    COLMAP_USE_GPU,
    EXTRACT_FRAME_INTERVAL,
    EXTRACT_MAX_FRAMES,
    GS2MESH_PRESET,
    GS2MESH_GS_ITERATIONS,
    GS2MESH_RUNTIME_PROFILE,
    GS2MESH_STEREO_MODEL,
    GS2MESH_TSDF_BLOCK_COUNT,
    GS2MESH_TSDF_CLEANING_THRESHOLD,
    GS2MESH_TSDF_CLOSING_KERNEL_SIZE,
    GS2MESH_TSDF_DEPTH_TRUNC,
    GS2MESH_TSDF_DILATE,
    GS2MESH_TSDF_ERODE_MASK,
    GS2MESH_TSDF_EROSION_KERNEL_SIZE,
    GS2MESH_TSDF_INVERT_MASK,
    GS2MESH_TSDF_MAX_DEPTH_BASELINES,
    GS2MESH_TSDF_MIN_DEPTH_BASELINES,
    GS2MESH_TSDF_SCALE,
    GS2MESH_TSDF_USE_OCCLUSION_MASK,
    GS2MESH_TSDF_VOXEL_SIZE,
    GS2MESH_USE_MASKS,
    GROUND_PLANE_ENABLED,
    POST_TEXTURE_CLEANUP_ENABLED,
    CLEANUP_LOWER_HALF_THRESHOLD,
    SAM2_DEFAULT_MODEL,
    TEXTURE_QUALITY_BOOST,
    TEXTURE_SIZE,
    TEXTURE_VIEW_ASSIGN_MODE,
    OUTPUT_DIR_DEFAULT,
)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERACTIVE = "interactive"


class PipelineStage(IntEnum):
    IDLE = 0
    EXTRACT_FRAMES = 1
    COLMAP_SFM = 2
    SAM2_SEGMENT = 3
    GS2MESH_RECONSTRUCT = 4
    TEXTURE_BAKE = 5
    POST_TEXTURE_CONTACT_CLEANUP = 6
    COMPLETE = 7


STAGE_LABELS: dict[int, str] = {
    PipelineStage.EXTRACT_FRAMES: "Extract Frames",
    PipelineStage.COLMAP_SFM: "COLMAP SfM",
    PipelineStage.SAM2_SEGMENT: "SAM2 Segmentation",
    PipelineStage.GS2MESH_RECONSTRUCT: "gs2mesh Reconstruction",
    PipelineStage.TEXTURE_BAKE: "Texture Bake",
    PipelineStage.POST_TEXTURE_CONTACT_CLEANUP: "PostTextureContactCleanup",
}

STAGE_OUTPUT_FILES: dict[int, tuple[str, ...]] = {
    2: ("camera_poses.json",),
    3: (),  # masks dir checked separately
    4: ("object_mesh.ply",),
    5: ("textured_mesh.obj",),
    # Stage 6 writes into the {object_name}/ final-deliverable subfolder; the
    # relative path is resolved at runtime because it depends on output_dir's
    # basename. See ``final_deliverable_dir()`` / ``cleaned_obj_relative_path()``.
    6: (),
}


def final_deliverable_dir(output_dir: str | Path) -> Path:
    """Subfolder (named after the object) holding the cleaned textured mesh."""
    root = Path(output_dir)
    return root / root.name


def cleaned_obj_relative_path(output_dir: str | Path) -> str:
    """Return ``<object_name>/textured_mesh_cleaned.obj`` for preview/routing."""
    return f"{Path(output_dir).name}/textured_mesh_cleaned.obj"


def _count_indexed_files(dir_path: Path, suffix: str) -> int:
    if not dir_path.is_dir():
        return 0
    return sum(1 for _ in dir_path.glob(f"*{suffix}"))


def detect_stage_outputs(output_dir: str | Path) -> tuple[dict[int, bool], int, int]:
    out = Path(output_dir)
    frame_count = _count_indexed_files(out / "frames", ".jpg")
    mask_count = _count_indexed_files(out / "masks", ".png")
    textured_ready = (out / "textured_mesh.obj").is_file()
    cleaned_ready = (final_deliverable_dir(out) / "textured_mesh_cleaned.obj").is_file()
    colmap_sparse_dir = out / "colmap_sparse"
    stage_complete = {
        1: frame_count > 0,
        2: (out / "camera_poses.json").is_file() and colmap_sparse_dir.is_dir(),
        3: mask_count > 0,
        4: (out / "object_mesh.ply").is_file(),
        5: textured_ready,
        6: cleaned_ready,
    }
    return stage_complete, frame_count, mask_count


@dataclass
class PipelineConfig:
    """All pipeline parameters — defaults match docker-compose env vars."""

    video_path: str = ""
    output_dir: str = OUTPUT_DIR_DEFAULT
    object_name: str = ""
    frame_interval: int = EXTRACT_FRAME_INTERVAL
    max_frames: int = EXTRACT_MAX_FRAMES
    sam2_model: str = SAM2_DEFAULT_MODEL
    colmap_matcher: str = COLMAP_MATCHER
    colmap_max_features: int = COLMAP_MAX_FEATURES
    colmap_image_size: int = COLMAP_IMAGE_SIZE
    colmap_use_gpu: bool = COLMAP_USE_GPU
    colmap_dsp_sift: bool = COLMAP_DSP_SIFT
    colmap_first_octave: int = COLMAP_FIRST_OCTAVE
    gs2mesh_preset: str = GS2MESH_PRESET
    gs2mesh_preset_base: str = GS2MESH_PRESET
    gs2mesh_gs_iterations: int = GS2MESH_GS_ITERATIONS
    gs2mesh_runtime_profile: str = GS2MESH_RUNTIME_PROFILE
    gs2mesh_stereo_model: str = GS2MESH_STEREO_MODEL
    gs2mesh_tsdf_voxel_size: float = GS2MESH_TSDF_VOXEL_SIZE
    gs2mesh_tsdf_depth_trunc: float = GS2MESH_TSDF_DEPTH_TRUNC
    gs2mesh_use_masks: bool = GS2MESH_USE_MASKS
    gs2mesh_tsdf_scale: float = GS2MESH_TSDF_SCALE
    gs2mesh_tsdf_min_depth_baselines: int = GS2MESH_TSDF_MIN_DEPTH_BASELINES
    gs2mesh_tsdf_max_depth_baselines: int = GS2MESH_TSDF_MAX_DEPTH_BASELINES
    gs2mesh_tsdf_dilate: int = GS2MESH_TSDF_DILATE
    gs2mesh_tsdf_cleaning_threshold: int = GS2MESH_TSDF_CLEANING_THRESHOLD
    gs2mesh_tsdf_use_occlusion_mask: bool = GS2MESH_TSDF_USE_OCCLUSION_MASK
    gs2mesh_tsdf_invert_mask: bool = GS2MESH_TSDF_INVERT_MASK
    gs2mesh_tsdf_erode_mask: bool = GS2MESH_TSDF_ERODE_MASK
    gs2mesh_tsdf_erosion_kernel_size: int = GS2MESH_TSDF_EROSION_KERNEL_SIZE
    gs2mesh_tsdf_closing_kernel_size: int = GS2MESH_TSDF_CLOSING_KERNEL_SIZE
    gs2mesh_tsdf_block_count: int = GS2MESH_TSDF_BLOCK_COUNT
    texture_size: int = TEXTURE_SIZE
    texture_view_assign_mode: str = TEXTURE_VIEW_ASSIGN_MODE
    texture_quality_boost: bool = TEXTURE_QUALITY_BOOST
    post_texture_cleanup_enabled: bool = POST_TEXTURE_CLEANUP_ENABLED
    cleanup_lower_half_threshold: float = CLEANUP_LOWER_HALF_THRESHOLD
    ground_plane_enabled: bool = GROUND_PLANE_ENABLED
    auto_accept: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
        }

    def to_gs2mesh_settings(self):
        from scripts.gs2mesh_config import Gs2meshSettings

        return Gs2meshSettings.from_config_mapping(
            self.to_dict(),
            preset=self.gs2mesh_preset,
        )


@dataclass
class StageInfo:
    status: StageStatus = StageStatus.PENDING
    start_time: float | None = None
    elapsed: float | None = None
    error: str | None = None
    progress: float = 0.0
    detail: str | None = None
    checkpoint_id: str | None = None


@dataclass
class PipelineSession:
    """Mutable pipeline run state — one per dashboard lifetime."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    current_stage: PipelineStage = PipelineStage.IDLE
    stages: dict[int, StageInfo] = field(default_factory=dict)
    running: bool = False
    cancelled: bool = False
    cancel_requested: bool = False
    cancel_force: bool = False
    resume_from_stage: PipelineStage = PipelineStage.EXTRACT_FRAMES
    pipeline_start_time: float | None = None
    current_checkpoint_id: str | None = None

    # WebSocket clients
    ws_clients: list[Any] = field(default_factory=list)

    # SAM2 interactive handshake
    sam2_confirm_event: asyncio.Event = field(default_factory=asyncio.Event)
    sam2_approve_event: asyncio.Event = field(default_factory=asyncio.Event)
    sam2_approved: bool = False
    sam2_frame_count: int = 0
    sam2_width: int = 0
    sam2_height: int = 0
    sam2_ground_skip_event: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_review_event: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_decision: str | None = None

    # Global stage-to-stage approval
    next_stage_confirm_event: asyncio.Event = field(default_factory=asyncio.Event)
    next_stage_confirmation_required: bool = False
    next_stage_confirmation_from: int | None = None
    next_stage_confirmation_to: int | None = None
    next_stage_confirmation_message: str | None = None

    # Pipeline task handle
    _task: asyncio.Task | None = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _active_processes: set[Any] = field(default_factory=set, repr=False)
    _active_process_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Artifacts produced by each stage
    frames_dir: str | None = None
    mask_dir: str | None = None
    ground_mask_dir: str | None = None
    ground_plane_path: str | None = None
    colmap_sparse_path: str | None = None
    poses_path: str | None = None
    mesh_ply: str | None = None
    obj_path: str | None = None
    cleaned_obj_path: str | None = None
    cleanup_proposal_path: str | None = None
    frame_count: int = 0
    mask_count: int = 0

    def __post_init__(self) -> None:
        if not self.stages:
            for s in range(1, int(PipelineStage.POST_TEXTURE_CONTACT_CLEANUP) + 1):
                self.stages[s] = StageInfo()

    def reset(self) -> None:
        self.current_stage = PipelineStage.IDLE
        self.running = False
        self.cancelled = False
        self.cancel_requested = False
        self.cancel_force = False
        self.resume_from_stage = PipelineStage.EXTRACT_FRAMES
        self.pipeline_start_time = None
        self.current_checkpoint_id = None
        self.sam2_confirm_event = asyncio.Event()
        self.sam2_approve_event = asyncio.Event()
        self.sam2_approved = False
        self.sam2_ground_skip_event = asyncio.Event()
        self.cleanup_review_event = asyncio.Event()
        self.cleanup_decision = None
        self.next_stage_confirm_event = asyncio.Event()
        self.next_stage_confirmation_required = False
        self.next_stage_confirmation_from = None
        self.next_stage_confirmation_to = None
        self.next_stage_confirmation_message = None
        self.sam2_frame_count = 0
        self._task = None
        self.cancel_event = threading.Event()
        self._active_processes = set()
        self._active_process_lock = threading.Lock()
        self.frames_dir = None
        self.mask_dir = None
        self.ground_mask_dir = None
        self.ground_plane_path = None
        self.colmap_sparse_path = None
        self.poses_path = None
        self.mesh_ply = None
        self.obj_path = None
        self.cleaned_obj_path = None
        self.cleanup_proposal_path = None
        self.frame_count = 0
        self.mask_count = 0
        for s in self.stages.values():
            s.status = StageStatus.PENDING
            s.start_time = None
            s.elapsed = None
            s.error = None
            s.progress = 0.0
            s.detail = None
            s.checkpoint_id = None

    def hydrate_from_output_dir(self, output_dir: str | Path) -> dict[str, Any]:
        out = Path(output_dir)
        stage_complete, frame_count, mask_count = detect_stage_outputs(out)

        self.frame_count = frame_count
        self.mask_count = mask_count
        self.frames_dir = str(out / "frames") if frame_count > 0 else None
        self.mask_dir = str(out / "masks") if mask_count > 0 else None
        ground_mask_dir = out / "masks_ground"
        self.ground_mask_dir = str(ground_mask_dir) if ground_mask_dir.is_dir() and any(ground_mask_dir.glob("*.png")) else None
        ground_plane_path = out / "ground_plane.json"
        self.ground_plane_path = str(ground_plane_path) if ground_plane_path.is_file() else None
        colmap_sparse = out / "colmap_sparse"
        self.colmap_sparse_path = str(colmap_sparse) if colmap_sparse.is_dir() else None
        self.poses_path = str(out / "camera_poses.json") if (out / "camera_poses.json").is_file() else None
        base_mesh = out / "object_mesh.ply"
        self.mesh_ply = str(base_mesh) if base_mesh.is_file() else None
        self.obj_path = str(out / "textured_mesh.obj") if (out / "textured_mesh.obj").is_file() else None
        cleaned_obj_candidate = final_deliverable_dir(out) / "textured_mesh_cleaned.obj"
        self.cleaned_obj_path = (
            str(cleaned_obj_candidate) if cleaned_obj_candidate.is_file() else None
        )
        cleanup_proposal = out / "post_texture_contact_cleanup" / "proposal.json"
        self.cleanup_proposal_path = str(cleanup_proposal) if cleanup_proposal.is_file() else None

        for stage_id in range(1, int(PipelineStage.POST_TEXTURE_CONTACT_CLEANUP) + 1):
            info = self.stages[stage_id]
            info.start_time = None
            info.elapsed = None
            info.error = None
            info.detail = None
            info.checkpoint_id = None
            if stage_complete.get(stage_id, False):
                info.status = StageStatus.COMPLETE
                info.progress = 100.0
            else:
                info.status = StageStatus.PENDING
                info.progress = 0.0

        completed = [stage_id for stage_id, ok in stage_complete.items() if ok]
        if len(completed) == int(PipelineStage.POST_TEXTURE_CONTACT_CLEANUP):
            self.current_stage = PipelineStage.COMPLETE
        elif completed:
            self.current_stage = PipelineStage(max(completed))
        else:
            self.current_stage = PipelineStage.IDLE

        self.running = False
        self.cancelled = False
        self.cancel_requested = False
        self.cancel_force = False
        self.current_checkpoint_id = None
        self.pipeline_start_time = None
        self.cancel_event = threading.Event()
        self._active_processes = set()
        self._active_process_lock = threading.Lock()
        self.sam2_approved = False
        self.sam2_frame_count = 0
        self.sam2_width = 0
        self.sam2_height = 0
        self.sam2_ground_skip_event = asyncio.Event()
        self.cleanup_review_event = asyncio.Event()
        self.cleanup_decision = None
        self.next_stage_confirmation_required = False
        self.next_stage_confirmation_from = None
        self.next_stage_confirmation_to = None
        self.next_stage_confirmation_message = None
        self.next_stage_confirm_event = asyncio.Event()
        return {
            "stage_complete": stage_complete,
            "frame_count": frame_count,
            "mask_count": mask_count,
        }

    def stage_start(self, stage: PipelineStage) -> None:
        self.current_stage = stage
        info = self.stages[int(stage)]
        info.status = StageStatus.RUNNING
        info.start_time = time.time()
        info.progress = 0.0
        info.detail = None
        info.checkpoint_id = None

    def stage_complete(self, stage: PipelineStage) -> None:
        info = self.stages[int(stage)]
        info.status = StageStatus.COMPLETE
        info.progress = 100.0
        info.detail = None
        info.checkpoint_id = None
        if info.start_time is not None:
            info.elapsed = time.time() - info.start_time

    def stage_failed(self, stage: PipelineStage, error: str) -> None:
        info = self.stages[int(stage)]
        info.status = StageStatus.FAILED
        info.error = error
        info.detail = error
        if info.start_time is not None:
            info.elapsed = time.time() - info.start_time

    def stage_interactive(self, stage: PipelineStage) -> None:
        info = self.stages[int(stage)]
        info.status = StageStatus.INTERACTIVE

    def require_next_stage_confirmation(
        self,
        from_stage: PipelineStage,
        to_stage: PipelineStage,
        message: str,
    ) -> None:
        self.next_stage_confirmation_required = True
        self.next_stage_confirmation_from = int(from_stage)
        self.next_stage_confirmation_to = int(to_stage)
        self.next_stage_confirmation_message = str(message)
        self.next_stage_confirm_event.clear()

    def clear_next_stage_confirmation(self) -> None:
        self.next_stage_confirmation_required = False
        self.next_stage_confirmation_from = None
        self.next_stage_confirmation_to = None
        self.next_stage_confirmation_message = None

    def stage_progress(
        self,
        stage: PipelineStage,
        progress: float | None = None,
        detail: str | None = None,
        checkpoint_id: str | None = None,
    ) -> None:
        info = self.stages[int(stage)]
        if progress is not None:
            info.progress = max(0.0, min(100.0, float(progress)))
        if detail is not None:
            info.detail = detail
        if checkpoint_id is not None:
            info.checkpoint_id = checkpoint_id
            self.current_checkpoint_id = checkpoint_id

    def request_cancel(self, force: bool = False) -> None:
        self.cancelled = True
        self.cancel_requested = True
        self.cancel_force = bool(force)
        self.cancel_event.set()

    def clear_cancel(self) -> None:
        self.cancelled = False
        self.cancel_requested = False
        self.cancel_force = False
        self.cancel_event.clear()

    def register_active_process(self, process: Any) -> None:
        with self._active_process_lock:
            self._active_processes.add(process)

    def unregister_active_process(self, process: Any) -> None:
        with self._active_process_lock:
            self._active_processes.discard(process)

    def terminate_active_processes(self, grace_seconds: float = 1.0) -> int:
        with self._active_process_lock:
            snapshot = list(self._active_processes)
        terminated = 0
        for proc in snapshot:
            if proc is None:
                continue
            poll = getattr(proc, "poll", None)
            if callable(poll):
                try:
                    if poll() is not None:
                        continue
                except Exception:
                    pass
            if _terminate_process_tree(proc, grace_seconds=grace_seconds):
                terminated += 1
        return terminated

    def overall_progress(self) -> float:
        total = 0.0
        stage_count = int(PipelineStage.POST_TEXTURE_CONTACT_CLEANUP)
        for stage_id in range(1, stage_count + 1):
            total += self.stages[stage_id].progress
        return round(total / float(stage_count), 1)

    def to_status_dict(self) -> dict:
        return {
            "current_stage": int(self.current_stage),
            "running": self.running,
            "cancelled": self.cancelled,
            "cancel_requested": self.cancel_requested,
            "cancel_force": self.cancel_force,
            "resume_from_stage": int(self.resume_from_stage),
            "object_name": self.config.object_name,
            "video_path": self.config.video_path,
            "output_dir": self.config.output_dir,
            "current_checkpoint_id": self.current_checkpoint_id,
            "frame_count": self.frame_count,
            "mask_count": self.mask_count,
            "elapsed": (time.time() - self.pipeline_start_time)
            if self.pipeline_start_time
            else None,
            "next_stage_confirmation": {
                "required": self.next_stage_confirmation_required,
                "from_stage": self.next_stage_confirmation_from,
                "to_stage": self.next_stage_confirmation_to,
                "message": self.next_stage_confirmation_message,
            },
            "stages": {
                str(k): {
                    "status": v.status.value,
                    "label": STAGE_LABELS.get(k, ""),
                    "elapsed": v.elapsed,
                    "error": v.error,
                    "progress": round(v.progress, 1),
                    "detail": v.detail,
                    "checkpoint_id": v.checkpoint_id,
                }
                for k, v in self.stages.items()
            },
            "has_ground_plane": self.ground_plane_path is not None,
            "cleanup_proposal_path": self.cleanup_proposal_path,
            "auto_accept": self.config.auto_accept,
            "overall_progress": self.overall_progress(),
        }


def _terminate_process_tree(process: Any, grace_seconds: float = 1.0) -> bool:
    """Terminate a process and its process group when available."""
    try:
        pid = int(getattr(process, "pid", 0) or 0)
    except Exception:
        pid = 0

    wait = getattr(process, "wait", None)
    terminate = getattr(process, "terminate", None)
    kill = getattr(process, "kill", None)

    if pid > 0:
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            if callable(terminate):
                try:
                    terminate()
                except Exception:
                    pass
    elif callable(terminate):
        try:
            terminate()
        except Exception:
            pass

    if callable(wait):
        try:
            wait(timeout=max(0.1, float(grace_seconds)))
            return True
        except Exception:
            pass

    if pid > 0:
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            if callable(kill):
                try:
                    kill()
                except Exception:
                    pass
    elif callable(kill):
        try:
            kill()
        except Exception:
            pass

    if callable(wait):
        try:
            wait(timeout=max(0.1, float(grace_seconds)))
        except Exception:
            pass
    return True
