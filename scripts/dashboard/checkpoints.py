"""Checkpoint definitions and cleanup helpers for dashboard pipeline stages."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

_EMPTY_TUPLE: tuple[str, ...] = ()


def mesh_method_key(value: str | None) -> str:
    return "diffcd" if str(value or "").strip().lower() == "diffcd" else "poisson"


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
            cleanup_dirs=("frames",),
        ),
        _checkpoint(
            "s1.finalize",
            label="Finalize frame set",
            patterns=_patterns(r"extracted\s+\d+\s+frames|finalize|complete"),
        ),
    ),
    2: (
        _checkpoint(
            "s2.load_model",
            label="Load Pi3X model",
            patterns=_patterns(
                r"scanning input frames|selected up to|loading pi3x model|pi3x model loaded",
            ),
        ),
        _checkpoint(
            "s2.infer",
            label="Run Pi3X inference",
            patterns=_patterns(
                r"running pi3x inference|pi3x chunk inference|oom fallback|pi3x inference complete",
            ),
        ),
        _checkpoint(
            "s2.filter",
            label="Apply confidence/edge filters",
            patterns=_patterns(r"aligning world axes|confidence|edge filters"),
        ),
        _checkpoint(
            "s2.save_outputs",
            label="Save point cloud and camera poses",
            patterns=_patterns(r"saving point cloud|saving camera poses"),
            cleanup_files=("object_full.ply", "camera_poses.json", "pi3x_cache.npz"),
        ),
        _checkpoint(
            "s2.save_cache",
            label="Save Pi3X cache",
            patterns=_patterns(r"saving pi3x cache|collecting pi3x outputs|pi3x stage complete"),
            cleanup_files=("pi3x_cache.npz",),
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
            cleanup_dirs=("masks",),
        ),
        _checkpoint(
            "s3.verify",
            label="Wait for mask verification",
            patterns=_patterns(r"waiting for mask verification|verification"),
        ),
        _checkpoint(
            "s3.apply",
            label="Apply SAM2 masks to point cloud",
            patterns=_patterns(
                r"loading pi3x cache|loading sam2 masks|applying sam2 mask filter|"
                r"applying sam2 masks|mask filtering complete",
            ),
            cleanup_files=("object.ply",),
        ),
    ),
    4: (
        _checkpoint(
            "s4.load",
            label="Load point cloud",
            patterns=_patterns(r"loading point cloud|loaded\s+\d+"),
        ),
        _checkpoint(
            "s4.denoise",
            label="Run denoise algorithm",
            patterns=_patterns(
                r"running dbscan|running statistical outlier removal|running radius outlier removal|"
                r"dbscan complete|sor complete|radius outlier complete",
            ),
        ),
        _checkpoint(
            "s4.save",
            label="Save denoised point cloud",
            patterns=_patterns(r"saving denoised point cloud|denoise complete"),
            cleanup_files=("object_denoised.ply",),
        ),
    ),
    5: {
        "poisson": (
            _checkpoint(
                "s5.poisson.preprocess",
                label="Preprocess point cloud",
                patterns=_patterns(r"classical\/preprocess"),
                cleanup_files=("object_mesh_input.ply", "classical_mesh/object_mesh_input.ply"),
            ),
            _checkpoint(
                "s5.poisson.reconstruct",
                label="Reconstruct mesh (Poisson)",
                patterns=_patterns(
                    r"classical\/main|poisson reconstruction|raw poisson mesh saved|point normals estimated",
                ),
                cleanup_files=(
                    "object_points_with_normals.ply",
                    "object_mesh_raw.ply",
                    "classical_mesh/object_mesh_poisson_raw.ply",
                ),
            ),
            _checkpoint(
                "s5.poisson.postprocess",
                label="Postprocess mesh",
                patterns=_patterns(r"classical\/postprocess"),
                cleanup_files=(
                    "object_mesh_postprocessed.ply",
                    "classical_mesh/object_mesh_postprocessed.ply",
                ),
            ),
            _checkpoint(
                "s5.poisson.downsample",
                label="Downsample mesh",
                patterns=_patterns(r"classical\/downsample|classical mesh stage complete"),
                cleanup_files=("object_mesh.ply", "classical_mesh/object_mesh_final.ply"),
            ),
        ),
        "diffcd": (
            _checkpoint(
                "s5.diffcd.prepare",
                label="Prepare point cloud for DiffCD",
                patterns=_patterns(r"preparing point cloud for diffcd|point cloud prepared"),
                cleanup_files=("object_points.npy",),
            ),
            _checkpoint(
                "s5.diffcd.fit",
                label="Run DiffCD fitting",
                patterns=_patterns(r"running diffcd fitting|diffcd attempt|diffcd fitting"),
                cleanup_dirs=("diffcd",),
            ),
            _checkpoint(
                "s5.diffcd.collect",
                label="Collect DiffCD outputs",
                patterns=_patterns(r"collecting diffcd outputs|diffcd complete"),
                cleanup_files=("object_mesh_raw.ply",),
            ),
            _checkpoint(
                "s5.diffcd.smooth",
                label="Apply mesh smoothing",
                patterns=_patterns(r"applying mesh smoothing|diffcd stage complete"),
                cleanup_files=("object_mesh.ply",),
            ),
        ),
    },
    6: (
        _checkpoint(
            "s6.load",
            label="Load mesh",
            patterns=_patterns(r"mesh wrap: loading mesh"),
        ),
        _checkpoint(
            "s6.wrap",
            label="Iterative mesh wrapping",
            patterns=_patterns(
                r"mesh wrap: sampling surface points|mesh wrap: estimating normals|"
                r"mesh wrap: running poisson reconstruction|mesh wrap: cleaning wrapped mesh|"
                r"mesh wrap: iteration",
            ),
        ),
        _checkpoint(
            "s6.adjust",
            label="Adjust face count",
            patterns=_patterns(r"mesh wrap: adjusting final face count"),
        ),
        _checkpoint(
            "s6.save",
            label="Save wrapped mesh",
            patterns=_patterns(r"mesh wrap: saving wrapped mesh|mesh wrap: complete"),
            cleanup_files=("object_mesh_wrapped.ply", "mesh_wrap/object_mesh_wrapped.ply"),
        ),
    ),
    7: (
        _checkpoint(
            "s7.load",
            label="Load wrapped mesh",
            patterns=_patterns(r"contact hole repair: loading mesh"),
        ),
        _checkpoint(
            "s7.find_loops",
            label="Find and evaluate boundary loops",
            patterns=_patterns(
                r"contact hole repair: extracting boundary loops|"
                r"contact hole repair: evaluating boundary loop",
            ),
        ),
        _checkpoint(
            "s7.save",
            label="Save repaired mesh",
            patterns=_patterns(
                r"contact hole repair: writing repaired mesh|"
                r"saved repaired mesh|contact hole repair: complete|"
                r"contact hole repair: skipped",
            ),
            cleanup_files=("object_mesh_repaired.ply", "contact_hole_repair/object_mesh_repaired.ply"),
        ),
    ),
    8: (
        _checkpoint(
            "s8.load",
            label="Load mesh",
            patterns=_patterns(r"loading mesh"),
        ),
        _checkpoint(
            "s8.intrinsics",
            label="Estimate camera intrinsics",
            patterns=_patterns(r"estimating camera intrinsics|intrinsics optimization complete|camera intrinsics estimated"),
            cleanup_files=("intrinsics.json",),
        ),
        _checkpoint(
            "s8.uv",
            label="Generate UV atlas / texel mapping",
            patterns=_patterns(r"generating uv atlas|building texel mapping"),
        ),
        _checkpoint(
            "s8.score",
            label="Score and project views",
            patterns=_patterns(
                r"scoring camera views|scoring views|projecting primary chart textures|applying primary views",
            ),
        ),
        _checkpoint(
            "s8.fill",
            label="Secondary fill and seam padding",
            patterns=_patterns(r"secondary view search|padding uv seams|padding seams"),
        ),
        _checkpoint(
            "s8.export",
            label="Export textured mesh",
            patterns=_patterns(r"exporting textured mesh|texture stage complete"),
            cleanup_files=("textured_mesh.obj", "textured_mesh.mtl", "texture.png"),
        ),
    ),
}

_STAGE_FALLBACK_RESET: dict[int, dict[str, tuple[str, ...]]] = {
    1: {"dirs": ("frames",), "files": ()},
    2: {"dirs": (), "files": ("object_full.ply", "pi3x_cache.npz", "camera_poses.json")},
    3: {"dirs": ("masks",), "files": ("object.ply",)},
    4: {"dirs": (), "files": ("object_denoised.ply",)},
    5: {
        "dirs": ("diffcd", "classical_mesh"),
        "files": (
            "object_mesh.ply",
            "object_mesh_raw.ply",
            "object_mesh_postprocessed.ply",
            "object_mesh_input.ply",
            "object_points.npy",
            "object_points_with_normals.ply",
        ),
    },
    6: {"dirs": ("mesh_wrap",), "files": ("object_mesh_wrapped.ply",)},
    7: {"dirs": ("contact_hole_repair",), "files": ("object_mesh_repaired.ply",)},
    8: {"dirs": (), "files": ("textured_mesh.obj", "textured_mesh.mtl", "texture.png", "intrinsics.json")},
}


def checkpoint_specs(stage: int, mesh_method: str) -> tuple[dict[str, Any], ...]:
    if stage == 5:
        branch = _CHECKPOINTS[5]
        return tuple(branch["diffcd"] if mesh_method_key(mesh_method) == "diffcd" else branch["poisson"])
    specs = _CHECKPOINTS.get(int(stage))
    if isinstance(specs, tuple):
        return specs
    return ()


def first_checkpoint_id(stage: int, mesh_method: str) -> str | None:
    specs = checkpoint_specs(stage, mesh_method)
    if not specs:
        return None
    return str(specs[0]["id"])


def checkpoint_index(stage: int, checkpoint_id: str | None, mesh_method: str) -> int | None:
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
    mesh_method: str,
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
    mesh_method: str,
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
    mesh_method: str,
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

    return {
        "stage": int(stage),
        "checkpoint_id": checkpoint_id,
        "used_fallback": used_fallback,
        "removed_dirs": removed_dirs,
        "removed_files": removed_files,
    }
