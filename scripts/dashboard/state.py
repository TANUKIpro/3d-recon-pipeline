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
    CLASSICAL_AUTO_SMOOTH,
    CLASSICAL_DEFAULT_PRESET,
    CLASSICAL_DENSITY_TRIM_Q,
    CLASSICAL_DOWNSAMPLE_ENABLED,
    CLASSICAL_DOWNSAMPLE_TARGET_FACES,
    CLASSICAL_POISSON_DEPTH,
    CLASSICAL_PREPROCESS_ENABLED,
    CLASSICAL_SMOOTH_ITERATIONS,
    DENOISE_DEFAULT_ALGORITHM,
    DENOISE_DEFAULT_PRESET,
    DIFFCD_BATCH_SIZE,
    DIFFCD_N_BATCHES,
    DIFFCD_RESOLUTION,
    EXTRACT_FRAME_INTERVAL,
    EXTRACT_MAX_FRAMES,
    MESH_DEFAULT_METHOD,
    MESHWRAP_ALPHA_RATIO,
    MESHWRAP_CROP_SCALE,
    MESHWRAP_DENSITY_TRIM_Q,
    MESHWRAP_ITERATIONS,
    MESHWRAP_NORMAL_RADIUS_RATIO,
    MESHWRAP_OFFSET_RATIO,
    MESHWRAP_POISSON_DEPTH,
    MESHWRAP_POISSON_SCALE,
    MESHWRAP_QUALITY_THRESHOLD,
    MESHWRAP_SAMPLE_POINTS,
    MESHWRAP_SMOOTH_ITERATIONS,
    MESHWRAP_TARGET_FACE_RATIO,
    PI3X_CONFIDENCE_THRESHOLD,
    PI3X_EDGE_RTOL,
    PI3X_FRAME_TARGET,
    PI3X_PIXEL_LIMIT,
    REPAIR_ENABLED,
    REPAIR_MAX_DIAMETER_RATIO,
    REPAIR_SMOOTH_ITERS,
    REPAIR_Y_BAND_RATIO,
    SAM2_DEFAULT_MODEL,
    TEXTURE_QUALITY_BOOST,
    TEXTURE_SIZE,
    TEXTURE_VIEW_ASSIGN_MODE,
    _MESHWRAP_METHOD,
    _OUTPUT_DIR_DEFAULT,
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
    PI3X_RECONSTRUCT = 2
    SAM2_SEGMENT = 3
    DENOISE = 4
    DIFFCD_MESH = 5
    MESH_WRAP = 6
    MESH_REPAIR = 7
    TEXTURE_BAKE = 8
    COMPLETE = 9


STAGE_LABELS: dict[int, str] = {
    PipelineStage.EXTRACT_FRAMES: "Extract Frames",
    PipelineStage.PI3X_RECONSTRUCT: "Pi3X 3D Reconstruction",
    PipelineStage.SAM2_SEGMENT: "SAM2 Segmentation",
    PipelineStage.DENOISE: "Point Cloud Denoise",
    PipelineStage.DIFFCD_MESH: "Mesh Reconstruction",
    PipelineStage.MESH_WRAP: "Mesh Wrap",
    PipelineStage.MESH_REPAIR: "Mesh Repair",
    PipelineStage.TEXTURE_BAKE: "Texture Bake",
}

STAGE_OUTPUT_FILES: dict[int, tuple[str, ...]] = {
    2: ("object_full.ply", "camera_poses.json", "pi3x_cache.npz"),
    3: ("object.ply",),
    4: ("object_denoised.ply",),
    5: ("object_mesh.ply",),
    6: ("object_mesh_wrapped.ply",),
    7: ("object_mesh_repaired.ply",),
    8: ("textured_mesh.obj",),
}


def _count_indexed_files(dir_path: Path, suffix: str) -> int:
    if not dir_path.is_dir():
        return 0
    return sum(1 for _ in dir_path.glob(f"*{suffix}"))


def detect_stage_outputs(output_dir: str | Path) -> tuple[dict[int, bool], int, int]:
    out = Path(output_dir)
    frame_count = _count_indexed_files(out / "frames", ".jpg")
    mask_count = _count_indexed_files(out / "masks", ".png")
    textured_ready = (out / "textured_mesh.obj").is_file()
    stage_complete = {
        1: frame_count > 0,
        2: all((out / rel).is_file() for rel in STAGE_OUTPUT_FILES[2]),
        3: (out / "object.ply").is_file() and mask_count > 0,
        4: (out / "object_denoised.ply").is_file(),
        5: (out / "object_mesh.ply").is_file(),
        6: (out / "object_mesh_wrapped.ply").is_file(),
        7: (out / "object_mesh_repaired.ply").is_file(),
        8: textured_ready,
    }
    return stage_complete, frame_count, mask_count


@dataclass
class PipelineConfig:
    """All pipeline parameters — defaults match docker-compose env vars."""

    video_path: str = ""
    output_dir: str = _OUTPUT_DIR_DEFAULT
    object_name: str = ""
    frame_interval: int = EXTRACT_FRAME_INTERVAL
    max_frames: int = EXTRACT_MAX_FRAMES
    pixel_limit: int = PI3X_PIXEL_LIMIT
    pi3x_frame_target: int = PI3X_FRAME_TARGET
    confidence_threshold: float = PI3X_CONFIDENCE_THRESHOLD
    edge_rtol: float = PI3X_EDGE_RTOL
    sam2_model: str = SAM2_DEFAULT_MODEL
    denoise_preset: str = DENOISE_DEFAULT_PRESET
    denoise_algorithm: str = DENOISE_DEFAULT_ALGORITHM
    denoise_dbscan_eps: float = 0.0
    denoise_dbscan_eps_ratio: float = 0.02
    denoise_dbscan_min_samples: int = 10
    denoise_dbscan_max_points: int = 500000
    denoise_sor_neighbors: int = 20
    denoise_sor_std_ratio: float = 2.0
    denoise_radius_neighbors: int = 8
    denoise_radius_radius_ratio: float = 0.015
    mesh_method: str = MESH_DEFAULT_METHOD
    diffcd_batch_size: int = DIFFCD_BATCH_SIZE
    diffcd_n_batches: int = DIFFCD_N_BATCHES
    diffcd_resolution: int = DIFFCD_RESOLUTION
    meshwrap_method: str = _MESHWRAP_METHOD
    meshwrap_poisson_depth: int = MESHWRAP_POISSON_DEPTH
    meshwrap_poisson_scale: float = MESHWRAP_POISSON_SCALE
    meshwrap_density_trim_q: float = MESHWRAP_DENSITY_TRIM_Q
    meshwrap_target_face_ratio: float = MESHWRAP_TARGET_FACE_RATIO
    meshwrap_iterations: int = MESHWRAP_ITERATIONS
    meshwrap_crop_scale: float = MESHWRAP_CROP_SCALE
    meshwrap_sample_points: int = MESHWRAP_SAMPLE_POINTS
    meshwrap_normal_radius_ratio: float = MESHWRAP_NORMAL_RADIUS_RATIO
    meshwrap_smooth_iterations: int = MESHWRAP_SMOOTH_ITERATIONS
    meshwrap_quality_threshold: float = MESHWRAP_QUALITY_THRESHOLD
    meshwrap_alpha_ratio: float = MESHWRAP_ALPHA_RATIO
    meshwrap_offset_ratio: float = MESHWRAP_OFFSET_RATIO
    classical_preset: str = CLASSICAL_DEFAULT_PRESET
    classical_preprocess_enabled: bool = CLASSICAL_PREPROCESS_ENABLED
    classical_poisson_depth: int = CLASSICAL_POISSON_DEPTH
    classical_density_trim_q: float = CLASSICAL_DENSITY_TRIM_Q
    classical_auto_smooth: bool = CLASSICAL_AUTO_SMOOTH
    classical_smooth_iterations: int = CLASSICAL_SMOOTH_ITERATIONS
    classical_downsample_enabled: bool = CLASSICAL_DOWNSAMPLE_ENABLED
    classical_downsample_target_faces: int = CLASSICAL_DOWNSAMPLE_TARGET_FACES
    mesh_repair_enabled: bool = REPAIR_ENABLED
    mesh_repair_max_diameter_ratio: float = REPAIR_MAX_DIAMETER_RATIO
    mesh_repair_y_band_ratio: float = REPAIR_Y_BAND_RATIO
    mesh_repair_smooth_iters: int = REPAIR_SMOOTH_ITERS
    texture_size: int = TEXTURE_SIZE
    texture_view_assign_mode: str = TEXTURE_VIEW_ASSIGN_MODE
    texture_quality_boost: bool = TEXTURE_QUALITY_BOOST
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

    # Global stage-to-stage approval
    next_stage_confirm_event: asyncio.Event = field(default_factory=asyncio.Event)
    next_stage_confirmation_required: bool = False
    next_stage_confirmation_from: int | None = None
    next_stage_confirmation_to: int | None = None
    next_stage_confirmation_message: str | None = None

    # Stage 7 interactive mesh-repair handshake
    mesh_repair_confirm_event: asyncio.Event = field(default_factory=asyncio.Event)
    mesh_repair_ready: bool = False
    mesh_repair_candidates: list[dict[str, Any]] = field(default_factory=list)
    mesh_repair_selected_loop_ids: list[int] = field(default_factory=list)
    mesh_repair_source_mesh_path: str | None = None
    mesh_repair_analysis: dict[str, Any] = field(default_factory=dict)

    # Pipeline task handle
    _task: asyncio.Task | None = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _active_processes: set[Any] = field(default_factory=set, repr=False)
    _active_process_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Artifacts produced by each stage
    frames_dir: str | None = None
    mask_dir: str | None = None
    ply_full_path: str | None = None    # full (unmasked) PLY from Pi3X
    pi3x_cache_path: str | None = None  # cache for mask filtering
    ply_path: str | None = None
    poses_path: str | None = None
    denoised_ply: str | None = None
    mesh_ply: str | None = None
    obj_path: str | None = None
    frame_count: int = 0
    mask_count: int = 0

    def __post_init__(self) -> None:
        if not self.stages:
            for s in range(1, int(PipelineStage.TEXTURE_BAKE) + 1):
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
        self.next_stage_confirm_event = asyncio.Event()
        self.next_stage_confirmation_required = False
        self.next_stage_confirmation_from = None
        self.next_stage_confirmation_to = None
        self.next_stage_confirmation_message = None
        self.mesh_repair_confirm_event = asyncio.Event()
        self.mesh_repair_ready = False
        self.mesh_repair_candidates = []
        self.mesh_repair_selected_loop_ids = []
        self.mesh_repair_source_mesh_path = None
        self.mesh_repair_analysis = {}
        self.sam2_frame_count = 0
        self._task = None
        self.cancel_event = threading.Event()
        self._active_processes = set()
        self._active_process_lock = threading.Lock()
        self.frames_dir = None
        self.mask_dir = None
        self.ply_full_path = None
        self.pi3x_cache_path = None
        self.ply_path = None
        self.poses_path = None
        self.denoised_ply = None
        self.mesh_ply = None
        self.obj_path = None
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
        self.ply_full_path = str(out / "object_full.ply") if (out / "object_full.ply").is_file() else None
        self.pi3x_cache_path = str(out / "pi3x_cache.npz") if (out / "pi3x_cache.npz").is_file() else None
        self.poses_path = str(out / "camera_poses.json") if (out / "camera_poses.json").is_file() else None
        self.ply_path = str(out / "object.ply") if (out / "object.ply").is_file() else None
        self.denoised_ply = str(out / "object_denoised.ply") if (out / "object_denoised.ply").is_file() else None
        repaired_mesh = out / "object_mesh_repaired.ply"
        wrapped_mesh = out / "object_mesh_wrapped.ply"
        base_mesh = out / "object_mesh.ply"
        if repaired_mesh.is_file():
            self.mesh_ply = str(repaired_mesh)
        elif wrapped_mesh.is_file():
            self.mesh_ply = str(wrapped_mesh)
        elif base_mesh.is_file():
            self.mesh_ply = str(base_mesh)
        else:
            self.mesh_ply = None
        self.obj_path = str(out / "textured_mesh.obj") if (out / "textured_mesh.obj").is_file() else None

        for stage_id in range(1, int(PipelineStage.TEXTURE_BAKE) + 1):
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
        if len(completed) == int(PipelineStage.TEXTURE_BAKE):
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
        self.next_stage_confirmation_required = False
        self.next_stage_confirmation_from = None
        self.next_stage_confirmation_to = None
        self.next_stage_confirmation_message = None
        self.next_stage_confirm_event = asyncio.Event()
        self.mesh_repair_confirm_event = asyncio.Event()
        self.mesh_repair_ready = False
        self.mesh_repair_candidates = []
        self.mesh_repair_selected_loop_ids = []
        self.mesh_repair_source_mesh_path = None
        self.mesh_repair_analysis = {}
        return {
            "stage_complete": stage_complete,
            "frame_count": frame_count,
            "mask_count": mask_count,
        }

    def set_mesh_repair_candidates(
        self,
        mesh_path: str,
        candidates: list[dict[str, Any]],
        analysis: dict[str, Any] | None = None,
    ) -> None:
        self.mesh_repair_ready = True
        self.mesh_repair_source_mesh_path = str(mesh_path)
        self.mesh_repair_candidates = list(candidates)
        self.mesh_repair_analysis = dict(analysis or {})
        self.mesh_repair_selected_loop_ids = []
        self.mesh_repair_confirm_event.clear()

    def clear_mesh_repair_candidates(self) -> None:
        self.mesh_repair_ready = False
        self.mesh_repair_source_mesh_path = None
        self.mesh_repair_candidates = []
        self.mesh_repair_selected_loop_ids = []
        self.mesh_repair_analysis = {}
        self.mesh_repair_confirm_event = asyncio.Event()

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
        stage_count = int(PipelineStage.TEXTURE_BAKE)
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
            "mesh_method": self.config.mesh_method,
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
            "mesh_repair": {
                "ready": self.mesh_repair_ready,
                "source_mesh_path": self.mesh_repair_source_mesh_path,
                "candidate_count": len(self.mesh_repair_candidates),
                "selected_loop_count": len(self.mesh_repair_selected_loop_ids),
                "analysis": self.mesh_repair_analysis,
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
