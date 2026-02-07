"""Pipeline state management for the web dashboard."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


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
    TEXTURE_BAKE = 6
    COMPLETE = 7


STAGE_LABELS: dict[int, str] = {
    PipelineStage.EXTRACT_FRAMES: "Extract Frames",
    PipelineStage.PI3X_RECONSTRUCT: "Pi3X 3D Reconstruction",
    PipelineStage.SAM2_SEGMENT: "SAM2 Segmentation",
    PipelineStage.DENOISE: "Point Cloud Denoise",
    PipelineStage.DIFFCD_MESH: "DiffCD Mesh",
    PipelineStage.TEXTURE_BAKE: "Texture Bake",
}


@dataclass
class PipelineConfig:
    """All pipeline parameters — defaults match docker-compose env vars."""

    video_path: str = ""
    output_dir: str = "/data/output"
    object_name: str = ""
    frame_interval: int = 10
    max_frames: int = 50
    pixel_limit: int = 255_000
    confidence_threshold: float = 0.1
    edge_rtol: float = 0.03
    sam2_model: str = "large"
    diffcd_batch_size: int = 3000
    diffcd_n_batches: int = 25000
    diffcd_resolution: int = 384
    texture_size: int = 2048

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class StageInfo:
    status: StageStatus = StageStatus.PENDING
    start_time: float | None = None
    elapsed: float | None = None
    error: str | None = None
    progress: float = 0.0
    detail: str | None = None


@dataclass
class PipelineSession:
    """Mutable pipeline run state — one per dashboard lifetime."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    current_stage: PipelineStage = PipelineStage.IDLE
    stages: dict[int, StageInfo] = field(default_factory=dict)
    running: bool = False
    cancelled: bool = False
    pipeline_start_time: float | None = None

    # WebSocket clients
    ws_clients: list[Any] = field(default_factory=list)

    # SAM2 interactive handshake
    sam2_confirm_event: asyncio.Event = field(default_factory=asyncio.Event)
    sam2_approve_event: asyncio.Event = field(default_factory=asyncio.Event)
    sam2_approved: bool = False
    sam2_frame_count: int = 0
    sam2_width: int = 0
    sam2_height: int = 0

    # Pi3X preview approval
    pi3x_approve_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Pipeline task handle
    _task: asyncio.Task | None = field(default=None, repr=False)

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

    def __post_init__(self) -> None:
        if not self.stages:
            for s in range(1, 7):
                self.stages[s] = StageInfo()

    def reset(self) -> None:
        self.current_stage = PipelineStage.IDLE
        self.running = False
        self.cancelled = False
        self.pipeline_start_time = None
        self.sam2_confirm_event = asyncio.Event()
        self.sam2_approve_event = asyncio.Event()
        self.sam2_approved = False
        self.pi3x_approve_event = asyncio.Event()
        self.sam2_frame_count = 0
        self._task = None
        self.frames_dir = None
        self.mask_dir = None
        self.ply_full_path = None
        self.pi3x_cache_path = None
        self.ply_path = None
        self.poses_path = None
        self.denoised_ply = None
        self.mesh_ply = None
        self.obj_path = None
        for s in self.stages.values():
            s.status = StageStatus.PENDING
            s.start_time = None
            s.elapsed = None
            s.error = None
            s.progress = 0.0
            s.detail = None

    def stage_start(self, stage: PipelineStage) -> None:
        self.current_stage = stage
        info = self.stages[int(stage)]
        info.status = StageStatus.RUNNING
        info.start_time = time.time()
        info.progress = 0.0
        info.detail = None

    def stage_complete(self, stage: PipelineStage) -> None:
        info = self.stages[int(stage)]
        info.status = StageStatus.COMPLETE
        info.progress = 100.0
        info.detail = None
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

    def stage_progress(
        self,
        stage: PipelineStage,
        progress: float | None = None,
        detail: str | None = None,
    ) -> None:
        info = self.stages[int(stage)]
        if progress is not None:
            info.progress = max(0.0, min(100.0, float(progress)))
        if detail is not None:
            info.detail = detail

    def overall_progress(self) -> float:
        total = 0.0
        for stage_id in range(1, 7):
            total += self.stages[stage_id].progress
        return round(total / 6.0, 1)

    def to_status_dict(self) -> dict:
        return {
            "current_stage": int(self.current_stage),
            "running": self.running,
            "cancelled": self.cancelled,
            "object_name": self.config.object_name,
            "video_path": self.config.video_path,
            "output_dir": self.config.output_dir,
            "elapsed": (time.time() - self.pipeline_start_time)
            if self.pipeline_start_time
            else None,
            "stages": {
                str(k): {
                    "status": v.status.value,
                    "label": STAGE_LABELS.get(k, ""),
                    "elapsed": v.elapsed,
                    "error": v.error,
                    "progress": round(v.progress, 1),
                    "detail": v.detail,
                }
                for k, v in self.stages.items()
            },
            "overall_progress": self.overall_progress(),
        }
