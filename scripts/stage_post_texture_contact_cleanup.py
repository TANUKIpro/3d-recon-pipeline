"""Post-texture cleanup stage for contact-region artifacts.

This stage operates on the textured OBJ produced by texture baking. It builds a
proposal for removing ground-contact protrusions, optionally waits for a review
decision in the dashboard, and then either:

- applies a local cut + cap and writes ``textured_mesh_cleaned.obj``, or
- copies the stage-5 OBJ/MTL forward unchanged when the cleanup is skipped.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.repair.ground_plane import (
    _cap_boundary_at_plane,
    _clip_mesh_at_plane,
    _extract_closed_section_loops,
    _find_ground_plane_shift,
    _orient_ground_plane_toward_mesh,
)
from scripts.repair.triangulate import _loop_projection_uv
from scripts.texture.intrinsics import _make_K, _project_simple
from scripts.texture.io_utils import _load_mask, _load_poses


_DEFAULT_CAP_TEXTURE_SIZE = 128
_MERGE_VERTEX_TOLERANCE = 1e-6
_PROPOSAL_DIR_NAME = "post_texture_contact_cleanup"
_PROPOSAL_FILE_NAME = "proposal.json"
_PROPOSAL_OVERLAY_NAME = "proposal_removed_region.ply"
_CLEANED_OBJ_NAME = "textured_mesh_cleaned.obj"
_CLEANED_MTL_NAME = "textured_mesh_cleaned.mtl"
_CAP_TEXTURE_NAME = "texture_cap.png"

_MASK_GROUND_REMOVAL_THRESHOLD = 0.5
_MASK_OBJECT_PRESERVATION_THRESHOLD = 0.2
_MASK_NEAR_PLANE_RATIO = 2.0


@dataclass(frozen=True)
class _ObjFace:
    vertex_indices: tuple[int, int, int]
    texcoord_indices: tuple[int, int, int]
    material: str


@dataclass(frozen=True)
class _ObjMesh:
    vertices: np.ndarray
    texcoords: np.ndarray
    faces: list[_ObjFace]
    mtllib: str
    source_mtl_path: Path | None
    base_material: str


def _emit_progress(progress_cb, progress: float, detail: str) -> None:
    if progress_cb is not None:
        progress_cb(float(progress), str(detail))


def _check_cancel(cancel_cb) -> None:
    if cancel_cb is not None:
        cancel_cb()


def _proposal_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / _PROPOSAL_DIR_NAME


def _proposal_path(output_dir: str | Path) -> Path:
    return _proposal_dir(output_dir) / _PROPOSAL_FILE_NAME


def _overlay_path(output_dir: str | Path) -> Path:
    return _proposal_dir(output_dir) / _PROPOSAL_OVERLAY_NAME


def _cleaned_obj_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / _CLEANED_OBJ_NAME


def _cleaned_mtl_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / _CLEANED_MTL_NAME


def _cap_texture_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / _CAP_TEXTURE_NAME


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_obj_mesh(obj_path: str | Path) -> _ObjMesh:
    path = Path(obj_path)
    text = path.read_text(encoding="utf-8")
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[_ObjFace] = []
    mtllib = "textured_mesh.mtl"
    current_material = "material_0"

    def _parse_face_vertex(token: str) -> tuple[int, int]:
        parts = token.split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"OBJ face entry must use v/vt form: {token}")
        return int(parts[0]), int(parts[1])

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("mtllib "):
            mtllib = line.split(None, 1)[1].strip()
            continue
        if line.startswith("usemtl "):
            current_material = line.split(None, 1)[1].strip() or current_material
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            continue
        if line.startswith("vt "):
            parts = line.split()
            if len(parts) < 3:
                continue
            texcoords.append([float(parts[1]), float(parts[2])])
            continue
        if line.startswith("f "):
            tokens = line.split()[1:]
            if len(tokens) < 3:
                continue
            parsed = [_parse_face_vertex(token) for token in tokens]
            for i in range(1, len(parsed) - 1):
                tri = (parsed[0], parsed[i], parsed[i + 1])
                faces.append(
                    _ObjFace(
                        vertex_indices=(tri[0][0], tri[1][0], tri[2][0]),
                        texcoord_indices=(tri[0][1], tri[1][1], tri[2][1]),
                        material=current_material,
                    )
                )

    if not vertices or not faces:
        raise ValueError(f"Textured OBJ is empty or invalid: {path}")

    source_mtl = path.with_name(mtllib)
    if not source_mtl.is_file():
        source_mtl = None

    base_material = faces[0].material if faces else "material_0"
    return _ObjMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        texcoords=np.asarray(texcoords, dtype=np.float64),
        faces=faces,
        mtllib=mtllib,
        source_mtl_path=source_mtl,
        base_material=base_material,
    )


def _merge_vertices(
    vertices: np.ndarray,
    faces: list[_ObjFace],
    *,
    tol: float = _MERGE_VERTEX_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    quantized = np.round(vertices / max(tol, 1e-12)).astype(np.int64)
    key_to_idx: dict[tuple[int, int, int], int] = {}
    merged_vertices: list[np.ndarray] = []
    old_to_merged = np.zeros((len(vertices),), dtype=np.int64)
    merged_to_originals: list[list[int]] = []

    for idx, key_arr in enumerate(quantized):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        merged_idx = key_to_idx.get(key)
        if merged_idx is None:
            merged_idx = len(merged_vertices)
            key_to_idx[key] = merged_idx
            merged_vertices.append(vertices[idx].copy())
            merged_to_originals.append([idx + 1])
        else:
            merged_to_originals[merged_idx].append(idx + 1)
        old_to_merged[idx] = merged_idx

    merged_faces: list[list[int]] = []
    original_face_map: list[int] = []
    for face_idx, face in enumerate(faces):
        tri = [int(old_to_merged[v_idx - 1]) for v_idx in face.vertex_indices]
        if len(set(tri)) < 3:
            continue
        merged_faces.append(tri)
        original_face_map.append(face_idx)

    if not merged_faces:
        raise ValueError("Failed to build merged geometry from textured OBJ")

    return (
        np.asarray(merged_vertices, dtype=np.float64),
        np.asarray(merged_faces, dtype=np.int64),
        np.asarray(original_face_map, dtype=np.int64),
        merged_to_originals,
    )


def _resolve_ground_plane(
    mesh_vertices: np.ndarray,
    ground_plane_path: str | Path | None,
) -> tuple[np.ndarray, float, str]:
    plane_source = "mesh_bottom_fallback"
    plane_normal = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    plane_d = -float(mesh_vertices[:, 1].min())

    if ground_plane_path:
        data = _safe_read_json(Path(ground_plane_path))
        raw_normal = data.get("plane_normal") or data.get("normal")
        raw_d = data.get("plane_d") if data.get("plane_d") is not None else data.get("d")
        if isinstance(raw_normal, list) and len(raw_normal) == 3 and raw_d is not None:
            try:
                plane_normal = np.asarray(raw_normal, dtype=np.float64)
                plane_d = float(raw_d)
                plane_source = "ground_plane_json"
            except Exception:
                plane_normal = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
                plane_d = -float(mesh_vertices[:, 1].min())
                plane_source = "mesh_bottom_fallback"

    plane_normal, plane_d, _flipped, _above, _below = _orient_ground_plane_toward_mesh(
        mesh_vertices,
        plane_normal,
        plane_d,
    )
    return plane_normal, float(plane_d), plane_source


def _load_intrinsics(intrinsics_path: str | Path | None) -> tuple[np.ndarray, int, int] | None:
    if not intrinsics_path:
        return None
    data = _safe_read_json(Path(intrinsics_path))
    if not data:
        return None
    if isinstance(data.get("K"), list):
        try:
            K = np.asarray(data["K"], dtype=np.float64)
            img_w = int(data.get("image_width") or data.get("width") or 0)
            img_h = int(data.get("image_height") or data.get("height") or 0)
            if K.shape == (3, 3) and img_w > 0 and img_h > 0:
                return K, img_w, img_h
        except Exception:
            pass
    try:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        img_w = int(data.get("image_width") or data.get("width") or 0)
        img_h = int(data.get("image_height") or data.get("height") or 0)
        if img_w > 0 and img_h > 0:
            return _make_K(fx, fy, cx, cy), img_w, img_h
    except Exception:
        return None
    return None


def _compute_mask_consistency_score(
    face_centroids: np.ndarray,
    *,
    poses_path: str | Path | None,
    intrinsics_path: str | Path | None,
    masks_dir: str | Path | None,
    ground_masks_dir: str | Path | None,
) -> dict[str, float | int | None]:
    if (
        face_centroids.size == 0
        or not poses_path
        or not intrinsics_path
        or not masks_dir
    ):
        return {
            "score": None,
            "samples": 0,
            "outside_ratio": None,
            "ground_ratio": None,
        }

    intrinsics = _load_intrinsics(intrinsics_path)
    if intrinsics is None:
        return {
            "score": None,
            "samples": 0,
            "outside_ratio": None,
            "ground_ratio": None,
        }
    K, img_w, img_h = intrinsics

    try:
        poses, frame_indices = _load_poses(str(poses_path))
    except Exception:
        return {
            "score": None,
            "samples": 0,
            "outside_ratio": None,
            "ground_ratio": None,
        }
    if len(poses) == 0:
        return {
            "score": None,
            "samples": 0,
            "outside_ratio": None,
            "ground_ratio": None,
        }

    pose_samples = np.linspace(0, len(poses) - 1, min(len(poses), 8), dtype=int)
    point_samples = np.linspace(0, len(face_centroids) - 1, min(len(face_centroids), 256), dtype=int)
    points = face_centroids[point_samples]

    sample_count = 0
    outside_count = 0
    ground_count = 0

    for pose_idx in pose_samples:
        src_idx = int(frame_indices[int(pose_idx)])
        try:
            obj_mask = _load_mask(str(masks_dir), src_idx)
        except Exception:
            continue
        ground_mask = None
        if ground_masks_dir:
            try:
                ground_mask = _load_mask(str(ground_masks_dir), src_idx)
            except Exception:
                ground_mask = None

        uv, depths = _project_simple(points, poses[int(pose_idx)], K)
        u = uv[:, 0]
        v = uv[:, 1]
        valid = (
            (depths > 0.01)
            & (u >= 0.0)
            & (u < float(img_w))
            & (v >= 0.0)
            & (v < float(img_h))
        )
        if not np.any(valid):
            continue
        ui = np.clip(np.round(u[valid]).astype(np.int32), 0, obj_mask.shape[1] - 1)
        vi = np.clip(np.round(v[valid]).astype(np.int32), 0, obj_mask.shape[0] - 1)
        inside_obj = obj_mask[vi, ui]
        sample_count += int(valid.sum())
        outside_count += int((~inside_obj).sum())
        if ground_mask is not None:
            gi = np.clip(ui, 0, ground_mask.shape[1] - 1)
            gv = np.clip(vi, 0, ground_mask.shape[0] - 1)
            ground_count += int(ground_mask[gv, gi].sum())

    if sample_count <= 0:
        return {
            "score": None,
            "samples": 0,
            "outside_ratio": None,
            "ground_ratio": None,
        }

    outside_ratio = outside_count / float(sample_count)
    ground_ratio = ground_count / float(sample_count)
    score = max(0.0, min(1.0, (outside_ratio + ground_ratio) * 0.5))
    return {
        "score": round(score, 4),
        "samples": int(sample_count),
        "outside_ratio": round(outside_ratio, 4),
        "ground_ratio": round(ground_ratio, 4),
    }


def _compute_per_face_ground_score(
    face_centroids: np.ndarray,
    *,
    poses_path: str | Path | None,
    intrinsics_path: str | Path | None,
    masks_dir: str | Path | None,
    ground_masks_dir: str | Path | None,
    max_views: int = 16,
) -> np.ndarray | None:
    """Compute per-face ground score in [0, 1].

    For each face centroid, projects to sampled camera views and computes:
    - outside_ratio: fraction of views where centroid falls outside object mask
    - ground_ratio: fraction of views where centroid falls inside ground mask
    - combined: max(outside_ratio, ground_ratio)

    Returns array of shape (N_faces,) or None if inputs are insufficient.
    """
    if (
        face_centroids.size == 0
        or not poses_path
        or not intrinsics_path
        or not masks_dir
    ):
        return None

    intrinsics = _load_intrinsics(intrinsics_path)
    if intrinsics is None:
        return None
    K, img_w, img_h = intrinsics

    try:
        poses, frame_indices = _load_poses(str(poses_path))
    except Exception:
        return None
    if len(poses) == 0:
        return None

    n_views = min(len(poses), max_views)
    pose_samples = np.linspace(0, len(poses) - 1, n_views, dtype=int)

    n_faces = len(face_centroids)
    outside_counts = np.zeros(n_faces, dtype=np.float64)
    ground_counts = np.zeros(n_faces, dtype=np.float64)
    visible_counts = np.zeros(n_faces, dtype=np.float64)

    for pose_idx in pose_samples:
        src_idx = int(frame_indices[int(pose_idx)])
        try:
            obj_mask = _load_mask(str(masks_dir), src_idx)
        except Exception:
            continue

        ground_mask = None
        if ground_masks_dir:
            try:
                ground_mask = _load_mask(str(ground_masks_dir), src_idx)
            except Exception:
                pass

        uv, depths = _project_simple(face_centroids, poses[int(pose_idx)], K)
        u = uv[:, 0]
        v = uv[:, 1]
        valid = (
            (depths > 0.01)
            & (u >= 0.0)
            & (u < float(img_w))
            & (v >= 0.0)
            & (v < float(img_h))
        )
        if not np.any(valid):
            continue

        ui = np.clip(np.round(u[valid]).astype(np.int32), 0, obj_mask.shape[1] - 1)
        vi = np.clip(np.round(v[valid]).astype(np.int32), 0, obj_mask.shape[0] - 1)

        visible_counts[valid] += 1.0
        inside_obj = obj_mask[vi, ui]
        outside_counts[valid] += (~inside_obj).astype(np.float64)

        if ground_mask is not None:
            gi = np.clip(ui, 0, ground_mask.shape[1] - 1)
            gv = np.clip(vi, 0, ground_mask.shape[0] - 1)
            ground_counts[valid] += ground_mask[gv, gi].astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        outside_ratio = np.where(visible_counts > 0, outside_counts / visible_counts, 0.0)
        ground_ratio = np.where(visible_counts > 0, ground_counts / visible_counts, 0.0)

    return np.maximum(outside_ratio, ground_ratio)


def _sample_cap_color(
    texture_path: Path,
    obj_mesh: _ObjMesh,
    removed_face_indices: np.ndarray,
) -> np.ndarray:
    image = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    if image is None:
        return np.asarray([176, 176, 176], dtype=np.uint8)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if obj_mesh.texcoords.size == 0 or removed_face_indices.size == 0:
        return np.asarray(np.mean(rgb.reshape(-1, 3), axis=0), dtype=np.uint8)

    colors: list[np.ndarray] = []
    for face_idx in removed_face_indices[:128]:
        face = obj_mesh.faces[int(face_idx)]
        for vt_idx in face.texcoord_indices:
            if vt_idx <= 0 or vt_idx > len(obj_mesh.texcoords):
                continue
            uv = obj_mesh.texcoords[vt_idx - 1]
            x = int(np.clip(round(float(uv[0]) * (w - 1)), 0, w - 1))
            y = int(np.clip(round((1.0 - float(uv[1])) * (h - 1)), 0, h - 1))
            colors.append(rgb[y, x].astype(np.float64))
    if not colors:
        return np.asarray(np.mean(rgb.reshape(-1, 3), axis=0), dtype=np.uint8)
    mean_color = np.mean(np.vstack(colors), axis=0)
    return np.clip(np.round(mean_color), 0, 255).astype(np.uint8)


def _write_cap_texture(path: Path, color_rgb: np.ndarray) -> None:
    img = np.tile(color_rgb.reshape(1, 1, 3), (_DEFAULT_CAP_TEXTURE_SIZE, _DEFAULT_CAP_TEXTURE_SIZE, 1))
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _write_overlay_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    if faces.size == 0 or vertices.size == 0:
        if path.exists():
            path.unlink()
        return

    unique = sorted({int(idx) for idx in faces.reshape(-1)})
    if not unique:
        if path.exists():
            path.unlink()
        return

    remap = {old: new for new, old in enumerate(unique)}
    subset_vertices = vertices[np.asarray(unique, dtype=np.int64)]
    subset_faces = np.asarray(
        [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces],
        dtype=np.int64,
    )
    colors = np.tile(np.asarray([[255, 80, 80]], dtype=np.int64), (len(subset_vertices), 1))

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(subset_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(subset_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vert, color in zip(subset_vertices, colors, strict=False):
            f.write(
                f"{vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for tri in subset_faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _analyze_cleanup(
    output_dir: str | Path,
    *,
    poses_path: str | Path | None = None,
    intrinsics_path: str | Path | None = None,
    masks_dir: str | Path | None = None,
    ground_masks_dir: str | Path | None = None,
    ground_plane_path: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    obj_mesh = _parse_obj_mesh(output_root / "textured_mesh.obj")
    merged_vertices, merged_faces, merged_face_to_obj_face, merged_to_originals = _merge_vertices(
        obj_mesh.vertices,
        obj_mesh.faces,
    )

    plane_normal, plane_d, plane_source = _resolve_ground_plane(merged_vertices, ground_plane_path)
    search = _find_ground_plane_shift(merged_vertices, merged_faces, plane_normal, plane_d)
    selected_shift = search.selected_shift
    if selected_shift is None:
        bbox_diag = max(
            float(np.linalg.norm(merged_vertices.max(axis=0) - merged_vertices.min(axis=0))),
            1e-6,
        )
        selected_shift = max(0.001, 0.0025 * bbox_diag)

    actual_plane_d = plane_d - float(selected_shift)
    section_loops = _extract_closed_section_loops(merged_vertices, merged_faces, plane_normal, actual_plane_d)
    selected_loop = max(section_loops, key=lambda loop: loop.area) if section_loops else None

    clipped_vertices, clipped_faces = _clip_mesh_at_plane(
        merged_vertices,
        merged_faces,
        plane_normal,
        plane_d,
        offset=float(selected_shift),
    )
    capped_vertices, capped_faces, cap_vertex_ids, matched_boundary_area = _cap_boundary_at_plane(
        clipped_vertices,
        clipped_faces,
        plane_normal,
        actual_plane_d,
        original_vertex_count=int(merged_vertices.shape[0]),
        target_loop=selected_loop,
    )

    orig_face_dist = merged_vertices[merged_faces] @ plane_normal + plane_d - float(selected_shift)
    keep_merged_mask = np.all(orig_face_dist > 0.0, axis=1)

    # Mask-based face filtering refinement
    mask_filtering_stats: dict[str, Any] = {"applied": False}
    all_centroids = merged_vertices[merged_faces].mean(axis=1)
    face_ground_scores = _compute_per_face_ground_score(
        all_centroids,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        masks_dir=masks_dir,
        ground_masks_dir=ground_masks_dir,
    )
    if face_ground_scores is not None and len(face_ground_scores) == len(keep_merged_mask):
        min_face_dist = orig_face_dist.min(axis=1)
        near_plane_threshold = _MASK_NEAR_PLANE_RATIO * float(selected_shift)
        near_plane_band = np.abs(min_face_dist) < near_plane_threshold

        mask_removed_count = 0
        mask_preserved_count = 0

        for i in range(len(keep_merged_mask)):
            if not near_plane_band[i]:
                continue
            score = face_ground_scores[i]
            if keep_merged_mask[i] and score > _MASK_GROUND_REMOVAL_THRESHOLD:
                keep_merged_mask[i] = False
                mask_removed_count += 1
            elif not keep_merged_mask[i] and score < _MASK_OBJECT_PRESERVATION_THRESHOLD:
                keep_merged_mask[i] = True
                mask_preserved_count += 1

        mask_filtering_stats = {
            "applied": True,
            "mask_removed_count": mask_removed_count,
            "mask_preserved_count": mask_preserved_count,
            "near_plane_faces": int(near_plane_band.sum()),
            "avg_ground_score_removed": (
                round(float(face_ground_scores[~keep_merged_mask].mean()), 4)
                if np.any(~keep_merged_mask) else None
            ),
        }

    removed_merged_faces = merged_faces[~keep_merged_mask]
    removed_obj_face_indices = np.unique(merged_face_to_obj_face[~keep_merged_mask])

    kept_face_keys = {
        tuple(sorted((int(face[0]), int(face[1]), int(face[2]))))
        for face in merged_faces[keep_merged_mask]
    }
    split_faces = np.asarray(
        [
            face
            for face in clipped_faces
            if tuple(sorted((int(face[0]), int(face[1]), int(face[2])))) not in kept_face_keys
        ],
        dtype=np.int64,
    )
    if split_faces.size == 0:
        split_faces = np.zeros((0, 3), dtype=np.int64)
    cap_faces = capped_faces[int(clipped_faces.shape[0]):].copy()
    if cap_faces.size == 0:
        cap_faces = np.zeros((0, 3), dtype=np.int64)
    cleanup_faces = np.vstack([arr for arr in (split_faces, cap_faces) if arr.size > 0]) if (split_faces.size > 0 or cap_faces.size > 0) else np.zeros((0, 3), dtype=np.int64)

    removed_centroids = np.zeros((0, 3), dtype=np.float64)
    if removed_merged_faces.size > 0:
        removed_centroids = merged_vertices[removed_merged_faces].mean(axis=1)

    mask_score = _compute_mask_consistency_score(
        removed_centroids,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        masks_dir=masks_dir,
        ground_masks_dir=ground_masks_dir,
    )

    texture_path = output_root / "texture.png"
    cap_color = _sample_cap_color(texture_path, obj_mesh, removed_obj_face_indices)

    has_candidate = (
        removed_merged_faces.size > 0
        and cleanup_faces.size > 0
        and selected_loop is not None
    )
    recommended_decision = "apply" if has_candidate else "skip"
    reason = "proposal_ready" if has_candidate else "no_cleanup_candidate"

    return {
        "obj_mesh": obj_mesh,
        "merged_vertices": merged_vertices,
        "merged_faces": merged_faces,
        "merged_to_originals": merged_to_originals,
        "plane_normal": plane_normal,
        "plane_d": float(plane_d),
        "plane_source": plane_source,
        "selected_shift": float(selected_shift),
        "actual_plane_d": float(actual_plane_d),
        "selected_loop": selected_loop,
        "capped_vertices": capped_vertices,
        "cleanup_faces": cleanup_faces,
        "split_faces": split_faces,
        "cap_faces": cap_faces,
        "removed_merged_faces": removed_merged_faces,
        "removed_obj_face_indices": removed_obj_face_indices,
        "keep_merged_mask": keep_merged_mask,
        "keep_original_face_mask": np.asarray(
            [
                np.all((obj_mesh.vertices[np.asarray(face.vertex_indices, dtype=np.int64) - 1] @ plane_normal + plane_d - float(selected_shift)) > 0.0)
                for face in obj_mesh.faces
            ],
            dtype=bool,
        ),
        "mask_score": mask_score,
        "mask_filtering": mask_filtering_stats,
        "cap_color": cap_color,
        "matched_boundary_area": float(matched_boundary_area),
        "has_candidate": bool(has_candidate),
        "recommended_decision": recommended_decision,
        "reason": reason,
    }


def _proposal_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    selected_loop = analysis.get("selected_loop")
    mask_score = analysis.get("mask_score", {})
    removed_faces = int(analysis["removed_merged_faces"].shape[0])
    mask_consistency_score = mask_score.get("score")
    mask_samples = int(mask_score.get("samples") or 0)
    outside_ratio = mask_score.get("outside_ratio")
    ground_ratio = mask_score.get("ground_ratio")
    stats = {
        "removed_faces": removed_faces,
        "removed_ratio": round(
            removed_faces / float(max(len(analysis["merged_faces"]), 1)),
            4,
        ),
        "cut_faces": int(analysis["split_faces"].shape[0]),
        "cap_faces": int(analysis["cap_faces"].shape[0]),
        "selected_shift": round(float(analysis["selected_shift"]), 6),
        "clip_plane_d": round(float(analysis["actual_plane_d"]), 6),
        "matched_boundary_area": round(float(analysis["matched_boundary_area"]), 6),
        "section_loop_area": round(float(selected_loop.area), 6) if selected_loop is not None else 0.0,
        "mask_consistency_score": mask_consistency_score,
        "mask_samples": mask_samples,
        "mask_outside_ratio": outside_ratio,
        "mask_ground_ratio": ground_ratio,
    }
    requires_review = bool(analysis["has_candidate"])
    return {
        "status": "proposal_ready" if analysis["has_candidate"] else "noop",
        "requires_review": requires_review,
        "review_required": requires_review,
        "recommended_decision": str(analysis["recommended_decision"]),
        "reason": str(analysis["reason"]),
        "plane": {
            "normal": analysis["plane_normal"].tolist(),
            "d": float(analysis["plane_d"]),
            "source": str(analysis["plane_source"]),
        },
        "ground_plane": {
            "normal": analysis["plane_normal"].tolist(),
            "d": float(analysis["plane_d"]),
            "source": str(analysis["plane_source"]),
            "selected_shift": round(float(analysis["selected_shift"]), 6),
            "selected_loop_area": round(float(selected_loop.area), 6) if selected_loop is not None else 0.0,
        },
        "artifacts": {
            "overlay_ply": f"{_PROPOSAL_DIR_NAME}/{_PROPOSAL_OVERLAY_NAME}",
            "removed_region_ply": f"{_PROPOSAL_DIR_NAME}/{_PROPOSAL_OVERLAY_NAME}",
            "output_obj": _CLEANED_OBJ_NAME,
            "output_mtl": _CLEANED_MTL_NAME,
            "texture_cap": _CAP_TEXTURE_NAME,
        },
        "stats": stats,
        "summary": {
            "removed_face_count": removed_faces,
            "cap_face_count": int(analysis["cap_faces"].shape[0]),
            "confidence": float(mask_consistency_score) if mask_consistency_score is not None else 0.0,
        },
        "mask_consistency": {
            "visible_samples": mask_samples,
            "object_outside_ratio": outside_ratio,
            "ground_overlap_ratio": ground_ratio,
        },
        "mask_filtering": analysis.get("mask_filtering", {"applied": False}),
    }


def prepare_cleanup_review(
    textured_obj_path: str | Path,
    output_dir: str | Path,
    *,
    poses_path: str | Path | None = None,
    intrinsics_path: str | Path | None = None,
    masks_dir: str | Path | None = None,
    ground_masks_dir: str | Path | None = None,
    ground_plane_path: str | Path | None = None,
    progress_cb=None,
    cancel_cb=None,
) -> dict[str, Any]:
    del textured_obj_path
    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 10.0, "Loading textured mesh")
    proposal_dir = _proposal_dir(output_dir)
    proposal_dir.mkdir(parents=True, exist_ok=True)

    analysis = _analyze_cleanup(
        output_dir,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        masks_dir=masks_dir,
        ground_masks_dir=ground_masks_dir,
        ground_plane_path=ground_plane_path,
    )
    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 55.0, "Scoring cleanup proposal")

    payload = _proposal_payload(analysis)
    if analysis["removed_merged_faces"].size > 0:
        _write_overlay_ply(
            _overlay_path(output_dir),
            analysis["merged_vertices"],
            analysis["removed_merged_faces"],
        )
    elif _overlay_path(output_dir).exists():
        _overlay_path(output_dir).unlink()

    _proposal_path(output_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _emit_progress(progress_cb, 100.0, "Cleanup proposal ready")
    return payload


def _copy_stage5_outputs(output_dir: str | Path) -> str:
    output_root = Path(output_dir)
    src_obj = output_root / "textured_mesh.obj"
    src_mtl = output_root / "textured_mesh.mtl"
    dst_obj = _cleaned_obj_path(output_root)
    dst_mtl = _cleaned_mtl_path(output_root)

    obj_text = src_obj.read_text(encoding="utf-8")
    if re.search(r"^mtllib\s+", obj_text, flags=re.MULTILINE):
        obj_text = re.sub(
            r"^mtllib\s+.+$",
            f"mtllib {_CLEANED_MTL_NAME}",
            obj_text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        obj_text = f"mtllib {_CLEANED_MTL_NAME}\n" + obj_text
    dst_obj.write_text(obj_text, encoding="utf-8")

    if src_mtl.is_file():
        shutil.copy2(src_mtl, dst_mtl)
    else:
        dst_mtl.write_text(
            "newmtl material_0\n"
            "Ka 1.0 1.0 1.0\n"
            "Kd 1.0 1.0 1.0\n"
            "Ks 0.0 0.0 0.0\n"
            "d 1.0\n"
            "illum 1\n"
            "map_Kd texture.png\n",
            encoding="utf-8",
        )
    return str(dst_obj)


def copy_stage5_as_cleaned(
    output_dir: str | Path,
    *,
    progress_cb=None,
    cancel_cb=None,
) -> str:
    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 20.0, "Copying textured mesh without cleanup")
    out = _copy_stage5_outputs(output_dir)
    _emit_progress(progress_cb, 100.0, "Post-texture cleanup skipped")
    return out


def apply_cleanup_proposal(
    output_dir: str | Path,
    *,
    progress_cb=None,
    cancel_cb=None,
) -> str:
    output_root = Path(output_dir)
    proposal = _safe_read_json(_proposal_path(output_root))
    if not proposal:
        raise FileNotFoundError("Cleanup proposal not found")

    analysis = _analyze_cleanup(
        output_root,
        poses_path=output_root / "camera_poses.json",
        intrinsics_path=output_root / "intrinsics.json",
        masks_dir=output_root / "masks",
        ground_masks_dir=(output_root / "masks_ground") if (output_root / "masks_ground").is_dir() else None,
        ground_plane_path=(output_root / "ground_plane.json") if (output_root / "ground_plane.json").is_file() else None,
    )
    if not analysis["has_candidate"]:
        return _copy_stage5_outputs(output_root)

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 12.0, "Preparing cleanup geometry")

    obj_mesh: _ObjMesh = analysis["obj_mesh"]
    keep_original_face_mask = analysis["keep_original_face_mask"]
    cleanup_faces = analysis["cleanup_faces"]
    capped_vertices = analysis["capped_vertices"]
    merged_to_originals = analysis["merged_to_originals"]
    merged_original_count = int(analysis["merged_vertices"].shape[0])
    plane_normal = analysis["plane_normal"]

    texture_cap = _cap_texture_path(output_root)
    _write_cap_texture(texture_cap, analysis["cap_color"])

    cleanup_vertex_ids = sorted({int(idx) for idx in cleanup_faces.reshape(-1)}) if cleanup_faces.size > 0 else []
    cleanup_positions = capped_vertices[np.asarray(cleanup_vertex_ids, dtype=np.int64)] if cleanup_vertex_ids else np.zeros((0, 3), dtype=np.float64)
    cleanup_uv_coords = np.zeros((len(cleanup_vertex_ids), 2), dtype=np.float64)
    if cleanup_positions.size > 0:
        projected = _loop_projection_uv(cleanup_positions, plane_normal)
        uv_min = projected.min(axis=0)
        uv_span = np.maximum(projected.max(axis=0) - uv_min, 1e-8)
        normalized = (projected - uv_min) / uv_span
        cleanup_uv_coords = 0.05 + normalized * 0.90

    next_vertex_index = int(obj_mesh.vertices.shape[0]) + 1
    merged_to_output_vertex: dict[int, int] = {}
    appended_vertices: list[np.ndarray] = []
    for merged_vid in cleanup_vertex_ids:
        if merged_vid < merged_original_count:
            merged_to_output_vertex[merged_vid] = int(merged_to_originals[merged_vid][0])
        else:
            merged_to_output_vertex[merged_vid] = next_vertex_index
            appended_vertices.append(capped_vertices[int(merged_vid)])
            next_vertex_index += 1

    next_vt_index = int(obj_mesh.texcoords.shape[0]) + 1
    merged_to_output_vt: dict[int, int] = {}
    appended_vts: list[np.ndarray] = []
    for merged_vid, uv in zip(cleanup_vertex_ids, cleanup_uv_coords, strict=False):
        merged_to_output_vt[int(merged_vid)] = next_vt_index
        appended_vts.append(uv)
        next_vt_index += 1

    _check_cancel(cancel_cb)
    _emit_progress(progress_cb, 60.0, "Writing cleaned OBJ/MTL")

    cleaned_mtl = _cleaned_mtl_path(output_root)
    base_mtl_text = ""
    if obj_mesh.source_mtl_path is not None and obj_mesh.source_mtl_path.is_file():
        base_mtl_text = obj_mesh.source_mtl_path.read_text(encoding="utf-8").rstrip() + "\n\n"
    elif proposal.get("artifacts", {}).get("output_mtl"):
        base_mtl_text = (
            "newmtl material_0\n"
            "Ka 1.0 1.0 1.0\n"
            "Kd 1.0 1.0 1.0\n"
            "Ks 0.0 0.0 0.0\n"
            "d 1.0\n"
            "illum 1\n"
            "map_Kd texture.png\n\n"
        )
    cleaned_mtl.write_text(
        base_mtl_text
        + "newmtl material_cap\n"
        + "Ka 1.0 1.0 1.0\n"
        + "Kd 1.0 1.0 1.0\n"
        + "Ks 0.0 0.0 0.0\n"
        + "d 1.0\n"
        + "illum 1\n"
        + f"map_Kd {_CAP_TEXTURE_NAME}\n",
        encoding="utf-8",
    )

    cleaned_obj = _cleaned_obj_path(output_root)
    lines: list[str] = [f"mtllib {_CLEANED_MTL_NAME}\n"]

    for vert in obj_mesh.vertices:
        lines.append(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
    for vert in appended_vertices:
        lines.append(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")

    for uv in obj_mesh.texcoords:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
    for uv in appended_vts:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

    current_material = None
    for keep, face in zip(keep_original_face_mask, obj_mesh.faces, strict=False):
        if not keep:
            continue
        if face.material != current_material:
            lines.append(f"usemtl {face.material}\n")
            current_material = face.material
        lines.append(
            f"f {face.vertex_indices[0]}/{face.texcoord_indices[0]} "
            f"{face.vertex_indices[1]}/{face.texcoord_indices[1]} "
            f"{face.vertex_indices[2]}/{face.texcoord_indices[2]}\n"
        )

    if cleanup_faces.size > 0:
        lines.append("usemtl material_cap\n")
        for face in cleanup_faces:
            a, b, c = (int(face[0]), int(face[1]), int(face[2]))
            va = merged_to_output_vertex[a]
            vb = merged_to_output_vertex[b]
            vc = merged_to_output_vertex[c]
            ta = merged_to_output_vt[a]
            tb = merged_to_output_vt[b]
            tc = merged_to_output_vt[c]
            lines.append(f"f {va}/{ta} {vb}/{tb} {vc}/{tc}\n")

    cleaned_obj.write_text("".join(lines), encoding="utf-8")
    _emit_progress(progress_cb, 100.0, "Post-texture cleanup complete")
    return str(cleaned_obj)


def generate_post_texture_cleanup_proposal(
    textured_obj_path: str | Path,
    output_dir: str | Path,
    *,
    poses_path: str | Path | None = None,
    intrinsics_path: str | Path | None = None,
    masks_dir: str | Path | None = None,
    ground_masks_dir: str | Path | None = None,
    ground_plane_path: str | Path | None = None,
    progress_cb=None,
    cancel_cb=None,
) -> dict[str, Any]:
    return prepare_cleanup_review(
        textured_obj_path,
        output_dir,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        masks_dir=masks_dir,
        ground_masks_dir=ground_masks_dir,
        ground_plane_path=ground_plane_path,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def finalize_post_texture_cleanup(
    textured_obj_path: str | Path,
    output_dir: str | Path,
    *,
    decision: str,
    progress_cb=None,
    cancel_cb=None,
) -> dict[str, Any]:
    del textured_obj_path
    normalized = str(decision or "").strip().lower()
    if normalized not in {"apply", "skip"}:
        raise ValueError("decision must be 'apply' or 'skip'")

    proposal_path = _proposal_path(output_dir)
    proposal = _safe_read_json(proposal_path)

    if normalized == "apply":
        obj_path = apply_cleanup_proposal(
            output_dir,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        applied = Path(obj_path).name == _CLEANED_OBJ_NAME and bool(proposal.get("requires_review", False) or proposal.get("review_required", False))
    else:
        obj_path = copy_stage5_as_cleaned(
            output_dir,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        applied = False

    if proposal:
        proposal["final_decision"] = normalized
        proposal["applied"] = bool(applied)
        proposal["outputs"] = {
            "obj": _CLEANED_OBJ_NAME,
            "mtl": _CLEANED_MTL_NAME,
            "texture_cap": _CAP_TEXTURE_NAME,
        }
        proposal_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "decision": normalized,
        "applied": bool(applied),
        "obj_path": str(obj_path),
    }
