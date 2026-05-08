"""Checkpoint definitions and cleanup helpers for dashboard pipeline stages."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from scripts.output_layout import (
    CAMERA_POSES_FILENAME,
    CLEANUP_PROPOSAL_DIRNAME,
    COLMAP_SPARSE_DIRNAME,
    COLMAP_SPARSE_POINTS_FILENAME,
    COLMAP_WORKSPACE_DIRNAME,
    FRAMES_DIR,
    GROUND_PLANE_FILENAME,
    GS2MESH_WORKSPACE_DIRNAME,
    MASKS_DIRNAME,
    MASKS_GROUND_DIRNAME,
    MASKS_OBJECT_RAW_DIRNAME,
    OBJECT_MESH_FILENAME,
    P2_COLMAP,
    P3_MASKS,
    P4_MESH,
    P5_TEXTURE,
    P6_CLEANUP,
    TEXTURE_PNG_FILENAME,
    TEXTURED_MESH_MTL_FILENAME,
    TEXTURED_MESH_OBJ_FILENAME,
    cleanup_final_dir,
)

_EMPTY_TUPLE: tuple[str, ...] = ()


def _patterns(*items: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(item, re.IGNORECASE) for item in items)


def _checkpoint(
    checkpoint_id: str,
    *,
    label: str,
    patterns: tuple[re.Pattern[str], ...] = (),
    cleanup_dirs: tuple[str, ...] = _EMPTY_TUPLE,
    cleanup_files: tuple[str, ...] = _EMPTY_TUPLE,
) -> dict[str, Any]:
    return {
        "id": checkpoint_id,
        "label": label,
        "patterns": patterns,
        "cleanup_dirs": cleanup_dirs,
        "cleanup_files": cleanup_files,
    }


_CHECKPOINTS: dict[int, Any] = {
    1: (
        _checkpoint(
            "s1.inspect",
            label="Inspect video metadata",
            patterns=_patterns(r"inspect|metadata"),
        ),
        _checkpoint(
            "s1.extract",
            label="Extract frames",
            patterns=_patterns(r"extracting frames|extract frames"),
            cleanup_dirs=(FRAMES_DIR,),
        ),
        _checkpoint(
            "s1.finalize",
            label="Finalize frame set",
            patterns=_patterns(r"extracted\s+\d+\s+frames|finalize|complete"),
        ),
    ),
    2: (
        _checkpoint(
            "s2.features",
            label="Extract features (COLMAP)",
            patterns=_patterns(r"feature extraction|feature_extractor"),
        ),
        _checkpoint(
            "s2.match",
            label="Match features",
            patterns=_patterns(r"matcher|matching"),
        ),
        _checkpoint(
            "s2.reconstruct",
            label="Sparse reconstruction",
            patterns=_patterns(r"sparse reconstruction|mapper"),
        ),
        _checkpoint(
            "s2.export",
            label="Export camera poses",
            patterns=_patterns(r"exporting camera poses|colmap sfm complete"),
            cleanup_files=(
                f"{P2_COLMAP}/{CAMERA_POSES_FILENAME}",
                f"{P2_COLMAP}/{COLMAP_SPARSE_POINTS_FILENAME}",
            ),
            cleanup_dirs=(
                f"{P2_COLMAP}/{COLMAP_SPARSE_DIRNAME}",
                f"{P2_COLMAP}/{COLMAP_WORKSPACE_DIRNAME}",
            ),
        ),
    ),
    3: (
        _checkpoint(
            "s3.initialize",
            label="Initialize SAM2 model",
            patterns=_patterns(r"initializing sam2 model|sam2 ready"),
        ),
        _checkpoint(
            "s3.interact",
            label="Wait for interactive clicks",
            patterns=_patterns(r"waiting for interactive clicks|redoing sam2 interaction"),
        ),
        _checkpoint(
            "s3.propagate",
            label="Propagate masks",
            patterns=_patterns(r"propagating masks"),
            cleanup_dirs=(f"{P3_MASKS}/{MASKS_DIRNAME}",),
        ),
        _checkpoint(
            "s3.verify",
            label="Wait for mask verification",
            patterns=_patterns(r"waiting for mask verification|verification"),
        ),
    ),
    4: (
        _checkpoint(
            "s4.train_gs",
            label="Train 3D Gaussian Splatting",
            patterns=_patterns(r"training 3d gaussian|3dgs training|iteration"),
        ),
        _checkpoint(
            "s4.stereo",
            label="Stereo depth estimation",
            patterns=_patterns(r"stereo depth|dlnr|running gs2mesh"),
        ),
        _checkpoint(
            "s4.tsdf",
            label="TSDF fusion + mesh extraction",
            patterns=_patterns(r"tsdf|fusion|mesh extraction|collecting output|GPU TSDF|VoxelBlockGrid"),
        ),
        _checkpoint(
            "s4.save",
            label="Save output mesh",
            patterns=_patterns(r"gs2mesh reconstruction complete|gs2mesh complete"),
            cleanup_files=(f"{P4_MESH}/{OBJECT_MESH_FILENAME}",),
            cleanup_dirs=(f"{P4_MESH}/{GS2MESH_WORKSPACE_DIRNAME}",),
        ),
    ),
    5: (
        _checkpoint(
            "s5.load",
            label="Load mesh",
            patterns=_patterns(r"loading mesh"),
        ),
        _checkpoint(
            "s5.intrinsics",
            label="Estimate camera intrinsics",
            patterns=_patterns(r"estimating camera intrinsics|intrinsics optimization complete|camera intrinsics estimated"),
            # Do NOT clean up intrinsics.json: it may have been written by
            # Stage 2 (COLMAP) with the optimal pinhole values from bundle
            # adjustment. Wiping it here forces the bake stage onto its
            # grid-search fallback, which can land on a sub-pixel-skewed
            # local minimum on cylindrical / self-similar surfaces and
            # destroy fine texture detail (see Cup_Noodle regression).
            cleanup_files=(),
        ),
        _checkpoint(
            "s5.uv",
            label="Generate UV atlas / texel mapping",
            patterns=_patterns(r"generating uv atlas|building texel mapping"),
        ),
        _checkpoint(
            "s5.score",
            label="Score and project views",
            patterns=_patterns(
                r"scoring camera views|scoring views|projecting primary chart textures|applying primary views",
            ),
        ),
        _checkpoint(
            "s5.fill",
            label="Secondary fill and seam padding",
            patterns=_patterns(r"secondary view search|padding uv seams|padding seams"),
        ),
        _checkpoint(
            "s5.export",
            label="Export textured mesh",
            patterns=_patterns(r"exporting textured mesh|texture stage complete"),
            cleanup_files=(
                f"{P5_TEXTURE}/{TEXTURED_MESH_OBJ_FILENAME}",
                f"{P5_TEXTURE}/{TEXTURED_MESH_MTL_FILENAME}",
                f"{P5_TEXTURE}/{TEXTURE_PNG_FILENAME}",
            ),
        ),
    ),
    6: (
        _checkpoint(
            "s6.proposal",
            label="Generate cleanup proposal",
            patterns=_patterns(r"loading textured mesh|scoring cleanup proposal|cleanup proposal ready"),
        ),
        _checkpoint(
            "s6.review",
            label="Wait for cleanup review",
            patterns=_patterns(r"waiting for cleanup review decision|cleanup review ready"),
        ),
        _checkpoint(
            "s6.holefill",
            label="Bottom hole-fill (skirt + cap)",
            patterns=_patterns(r"bottom hole-fill"),
        ),
        _checkpoint(
            "s6.noise",
            label="Post-holefill noise removal",
            patterns=_patterns(r"post-holefill noise removal"),
        ),
        _checkpoint(
            "s6.watertight",
            label="General hole-fill (watertight)",
            patterns=_patterns(r"general hole-fill"),
        ),
        _checkpoint(
            "s6.finalclean",
            label="Final component cleanup",
            patterns=_patterns(r"final cleanup"),
        ),
        _checkpoint(
            "s6.apply",
            label="Write cleaned mesh",
            patterns=_patterns(r"writing cleaned obj/mtl|post-texture cleanup complete|post-texture cleanup skipped|preparing cleanup geometry"),
            cleanup_dirs=(f"{P6_CLEANUP}/{CLEANUP_PROPOSAL_DIRNAME}",),
            # The cleaned mesh artifacts live in the dynamic
            # ``p6_cleanup/{object_name}/`` final subfolder;
            # ``cleanup_checkpoint_outputs`` removes that directory explicitly
            # for stage 6.
            cleanup_files=(),
        ),
    ),
}

_STAGE_FALLBACK_RESET: dict[int, dict[str, tuple[str, ...]]] = {
    1: {"dirs": (FRAMES_DIR,), "files": ()},
    2: {
        "dirs": (
            f"{P2_COLMAP}/{COLMAP_SPARSE_DIRNAME}",
            f"{P2_COLMAP}/{COLMAP_WORKSPACE_DIRNAME}",
        ),
        "files": (
            f"{P2_COLMAP}/{CAMERA_POSES_FILENAME}",
            f"{P2_COLMAP}/{COLMAP_SPARSE_POINTS_FILENAME}",
        ),
    },
    3: {
        "dirs": (
            f"{P3_MASKS}/{MASKS_DIRNAME}",
            f"{P3_MASKS}/{MASKS_GROUND_DIRNAME}",
            f"{P3_MASKS}/{MASKS_OBJECT_RAW_DIRNAME}",
        ),
        "files": (f"{P3_MASKS}/{GROUND_PLANE_FILENAME}",),
    },
    4: {
        "dirs": (f"{P4_MESH}/{GS2MESH_WORKSPACE_DIRNAME}",),
        "files": (f"{P4_MESH}/{OBJECT_MESH_FILENAME}",),
    },
    # intrinsics.json is intentionally omitted here too — see s5.intrinsics
    # checkpoint comment. COLMAP's intrinsics is the authoritative source.
    5: {
        "dirs": (),
        "files": (
            f"{P5_TEXTURE}/{TEXTURED_MESH_OBJ_FILENAME}",
            f"{P5_TEXTURE}/{TEXTURED_MESH_MTL_FILENAME}",
            f"{P5_TEXTURE}/{TEXTURE_PNG_FILENAME}",
        ),
    },
    6: {
        "dirs": (f"{P6_CLEANUP}/{CLEANUP_PROPOSAL_DIRNAME}",),
        # The cleaned mesh outputs live under the dynamic
        # ``p6_cleanup/{object_name}/`` subfolder, which is wiped directly in
        # ``cleanup_checkpoint_outputs``.
        "files": (),
    },
}


def checkpoint_specs(stage: int, mesh_method: str = "") -> tuple[dict[str, Any], ...]:
    specs = _CHECKPOINTS.get(int(stage))
    if isinstance(specs, tuple):
        return specs
    return ()


def first_checkpoint_id(stage: int, mesh_method: str = "") -> str | None:
    specs = checkpoint_specs(stage, mesh_method)
    if not specs:
        return None
    return str(specs[0]["id"])


def checkpoint_index(stage: int, checkpoint_id: str | None, mesh_method: str = "") -> int | None:
    if not checkpoint_id:
        return None
    specs = checkpoint_specs(stage, mesh_method)
    for idx, spec in enumerate(specs):
        if spec["id"] == checkpoint_id:
            return idx
    return None


def resolve_checkpoint_id(
    stage: int,
    detail: str | None,
    mesh_method: str = "",
    *,
    current_checkpoint_id: str | None = None,
) -> str | None:
    specs = checkpoint_specs(stage, mesh_method)
    if not specs:
        return None
    text = str(detail or "").strip()
    if not text:
        return current_checkpoint_id
    if "starting" in text.lower():
        return str(specs[0]["id"])
    if re.search(r"waiting for next-stage confirmation", text, re.IGNORECASE):
        return str(specs[-1]["id"])
    for spec in specs:
        for pat in spec["patterns"]:
            if pat.search(text):
                return str(spec["id"])
    if re.search(r"\bcomplete\b", text, re.IGNORECASE):
        return str(specs[-1]["id"])
    return current_checkpoint_id


def checkpoint_cleanup_plan(
    stage: int,
    checkpoint_id: str | None,
    mesh_method: str = "",
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    specs = checkpoint_specs(stage, mesh_method)
    if checkpoint_id:
        for spec in specs:
            if spec["id"] == checkpoint_id:
                return tuple(spec["cleanup_dirs"]), tuple(spec["cleanup_files"]), False
    fallback = _STAGE_FALLBACK_RESET.get(int(stage), {"dirs": (), "files": ()})
    return tuple(fallback.get("dirs", ())), tuple(fallback.get("files", ())), True


def cleanup_checkpoint_outputs(
    output_dir: str | Path,
    stage: int,
    checkpoint_id: str | None,
    mesh_method: str = "",
) -> dict[str, Any]:
    out = Path(output_dir)
    dirs, files, used_fallback = checkpoint_cleanup_plan(stage, checkpoint_id, mesh_method)
    removed_dirs: list[str] = []
    removed_files: list[str] = []

    for rel in dirs:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            continue
        target = out / rel_path
        if target.is_dir():
            shutil.rmtree(target)
            removed_dirs.append(rel)

    for rel in files:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            continue
        target = out / rel_path
        if target.is_file():
            target.unlink()
            removed_files.append(rel)

    # Stage 6 writes its final deliverables into
    # ``{output_dir}/p6_cleanup/{object_name}/`` (a subfolder named after the
    # output directory itself, scoped under the phase 6 directory). Remove it
    # when cleaning the "apply" checkpoint or a fallback reset of stage 6.
    if int(stage) == 6 and (checkpoint_id in (None, "s6.apply") or used_fallback):
        final_dir = cleanup_final_dir(out)
        if final_dir.is_dir():
            shutil.rmtree(final_dir)
            removed_dirs.append(str(final_dir.relative_to(out)))

    return {
        "stage": int(stage),
        "checkpoint_id": checkpoint_id,
        "used_fallback": used_fallback,
        "removed_dirs": removed_dirs,
        "removed_files": removed_files,
    }
