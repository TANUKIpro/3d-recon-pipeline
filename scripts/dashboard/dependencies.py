"""Shared state container and accessors for the dashboard.

Provides ``AppState`` (the single source of truth for mutable dashboard
state) and accessor functions so that route handlers can reach shared
state without importing ``app.py`` -- keeping ``app.py`` as a thin
composition root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scripts.dashboard.configuration import build_pipeline_config
from scripts.dashboard.git_utils import branch_slug as _branch_slug, detect_branch
from scripts.dashboard.object_store import (
    OBJECT_META_FILE,
    infer_resume_stage,
    resolve_output_root,
    safe_json_load,
    summarize_object,
)
from scripts.dashboard.sam2_service import SAM2Service
from scripts.dashboard.state import PipelineSession, PipelineStage

# ── Constants (moved from app.py) ─────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MESH_POSTPROCESS_METHODS = {"laplacian", "taubin"}


# ── Shared state container ────────────────────────────────────────


class AppState:
    """Holds the mutable singletons that route handlers need."""

    __slots__ = ("session", "sam2_service", "input_dir", "output_dir",
                 "current_branch", "branch_slug")

    def __init__(
        self,
        *,
        session: PipelineSession | None = None,
        sam2_service: SAM2Service | None = None,
        input_dir: str | None = None,
        output_dir: str | None = None,
        current_branch: str | None = None,
    ) -> None:
        self.session = session or PipelineSession()
        self.sam2_service = sam2_service or SAM2Service()
        self.input_dir = input_dir or os.environ.get("INPUT_DIR", "/data/input")
        self.output_dir = output_dir or os.environ.get("OUTPUT_DIR", "/data/output")
        branch = current_branch or detect_branch()
        self.current_branch = branch
        self.branch_slug = _branch_slug(branch)

    def active_output_dir(self) -> Path:
        """Return the effective output directory for the active object."""
        cfg_out = (self.session.config.output_dir or "").strip()
        if cfg_out:
            return Path(cfg_out)
        return resolve_output_root(self.output_dir)

    def load_object_into_session(
        self, object_name: str, object_dir: Path,
    ) -> dict[str, Any]:
        """Load an existing object into the pipeline session."""
        meta = safe_json_load(object_dir / OBJECT_META_FILE)
        raw_cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        cfg = build_pipeline_config(
            raw_cfg,
            video_path=str(meta.get("video_path", "")),
            object_name=object_name,
            output_dir=object_dir,
        )
        self.session.reset()
        self.session.config = cfg
        self.session.hydrate_from_output_dir(object_dir)
        self.session.resume_from_stage = PipelineStage(infer_resume_stage(object_dir))
        return summarize_object(object_name, object_dir, include_files=True)


# ── Singleton management ──────────────────────────────────────────

_state: AppState | None = None


def init_state(
    *,
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> AppState:
    """Create the singleton AppState.  Called once from ``app.py``."""
    global _state
    _state = AppState(input_dir=input_dir, output_dir=output_dir)
    return _state


def get_state() -> AppState:
    """Accessor for route handlers to reach shared state."""
    if _state is None:
        raise RuntimeError("AppState not initialised -- call init_state() first")
    return _state
