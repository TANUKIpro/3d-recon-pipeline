"""Object management utilities and data constants for the dashboard.

Pure-logic helpers for managing pipeline output objects (creation,
discovery, metadata, stage reset, resume validation).  Extracted from
``app.py`` to keep endpoint code focused on HTTP concerns.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.dashboard.configuration import build_pipeline_config
from scripts.dashboard.state import PipelineStage, detect_stage_outputs

# ── Data constants ────────────────────────────────────────────────

OBJECTS_SUBDIR = "objects"
OBJECT_META_FILE = "object_meta.json"
PREVIEW_FILE_EXTENSIONS = {".ply", ".obj", ".mtl", ".png", ".jpg", ".json"}

STAGE_RESET_PATHS: dict[int, dict[str, tuple[str, ...]]] = {
    1: {"dirs": ("frames",), "files": ()},
    2: {"dirs": ("colmap_sparse", "colmap_workspace"), "files": ("camera_poses.json", "colmap_sparse_points.ply")},
    3: {"dirs": ("masks", "masks_ground"), "files": ("ground_plane.json",)},
    4: {"dirs": ("gs2mesh_workspace",), "files": ("object_mesh.ply",)},
    5: {"dirs": (), "files": ("textured_mesh.obj", "textured_mesh.mtl", "texture.png", "intrinsics.json")},
}

RESUME_PREREQUISITES: dict[int, dict[str, tuple[str, ...]]] = {
    2: {"dirs": ("frames",), "files": ()},
    3: {"dirs": ("frames",), "files": ("camera_poses.json",)},
    4: {"dirs": ("frames",), "files": ("camera_poses.json",)},
    5: {"dirs": ("frames", "masks"), "files": ("camera_poses.json", "object_mesh.ply")},
}

PRIMARY_ARTIFACT_PATHS = (
    "camera_poses.json",
    "colmap_sparse_points.ply",
    "object_mesh.ply",
    "textured_mesh.obj",
    "texture.png",
    "intrinsics.json",
)

# ── Utility functions ─────────────────────────────────────────────


def _utc_iso(ts: float | None = None) -> str:
    dt = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts is not None
        else datetime.now(timezone.utc)
    )
    return dt.isoformat(timespec="seconds")


def safe_json_load(path: Path) -> dict[str, Any]:
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


def sanitize_object_name(name: str) -> str:
    candidate = str(name or "").strip().replace("/", "-").replace("\\", "-")
    candidate = re.sub(r"\s+", "-", candidate)
    candidate = re.sub(r"[^\w.-]", "-", candidate, flags=re.UNICODE)
    candidate = re.sub(r"-{2,}", "-", candidate).strip("-.")
    if not candidate:
        raise ValueError("object_name is required")
    return candidate[:80]


def validate_object_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not candidate:
        raise ValueError("object name is required")
    if candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
        raise ValueError("invalid object name")
    if candidate != sanitize_object_name(candidate):
        raise ValueError("invalid object name")
    return candidate


def suggest_object_name(video_path: str) -> str:
    stem = Path(video_path).stem.strip() if video_path else "object"
    if not stem:
        stem = "object"
    try:
        return sanitize_object_name(stem)
    except ValueError:
        return f"object-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def resolve_output_root(root: str | None) -> Path:
    out = Path(root) if root else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _objects_root(base_output: Path) -> Path:
    root = base_output / OBJECTS_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def object_dir(object_name: str, base_output: Path) -> Path:
    return _objects_root(base_output) / object_name


def list_preview_files(out: Path) -> list[dict[str, Any]]:
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


def prepare_object_output_dir(out: Path) -> None:
    reset_outputs_from_stage(out, int(PipelineStage.EXTRACT_FRAMES))


def reset_outputs_from_stage(out: Path, start_stage: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stage in range(max(1, start_stage), int(PipelineStage.TEXTURE_BAKE) + 1):
        plan = STAGE_RESET_PATHS.get(stage, {})
        for rel in plan.get("dirs", ()):
            target = out / rel
            if target.is_dir():
                shutil.rmtree(target)
        for rel in plan.get("files", ()):
            target = out / rel
            if target.is_file():
                target.unlink()


def infer_resume_stage(out: Path) -> int:
    stage_complete, _, _ = detect_stage_outputs(out)
    for stage in range(1, int(PipelineStage.TEXTURE_BAKE) + 1):
        if not stage_complete.get(stage, False):
            return stage
    return int(PipelineStage.TEXTURE_BAKE)


def validate_resume_prerequisites(out: Path, start_stage: int) -> list[str]:
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


def write_object_meta(
    object_name: str,
    object_dir: Path,
    video_path: str,
    *,
    config: dict[str, Any] | None = None,
) -> None:
    meta_path = object_dir / OBJECT_META_FILE
    existing = safe_json_load(meta_path)
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


def summarize_object(
    object_name: str,
    object_dir: Path,
    include_files: bool = False,
) -> dict[str, Any]:
    meta = safe_json_load(object_dir / OBJECT_META_FILE)
    files = list_preview_files(object_dir)
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
        "resume_from_stage": infer_resume_stage(object_dir),
    }
    if include_files:
        item["files"] = files
        if isinstance(meta.get("config"), dict):
            cfg = build_pipeline_config(
                meta["config"],
                video_path=str(meta.get("video_path", "")),
                object_name=object_name,
                output_dir=object_dir,
            )
            item["config"] = cfg.to_dict()
    return item


def list_objects(base_output: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    root = _objects_root(base_output)
    for d in root.iterdir():
        if not d.is_dir():
            continue
        objects.append(summarize_object(d.name, d, include_files=False))
    objects.sort(key=lambda o: o.get("updated_at") or "", reverse=True)
    return objects
