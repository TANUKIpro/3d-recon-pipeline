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

from scripts.repair.boundary import (
    _apply_local_smoothing,
    _extract_boundary_edges,
    _extract_boundary_paths,
    _fit_loop_normal,
)
from scripts.repair.ground_plane import (
    _cap_boundary_at_plane,
    _clip_mesh_at_plane,
    _extract_closed_section_loops,
    _generate_bottom_skirt_cap,
    _orient_ground_plane_toward_mesh,
)
from scripts.repair.triangulate import _loop_projection_uv, _triangulate_polygon_ear_clip
from scripts.texture.conflict_region import _build_face_adjacency
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
_COMPONENT_MIN_FACE_RATIO = 0.02
_MASK_ABOVE_PLANE_REMOVAL_THRESHOLD = 0.7
_NEAR_PLANE_MIN_BBOX_RATIO = 0.005
_MASK_LOWER_HALF_REMOVAL_THRESHOLD = 0.0
_NEIGHBOR_REMOVAL_RATIO_THRESHOLD = 0.6
_CONVERGENCE_MAX_ITERATIONS = 3
_ISLAND_FACE_RATIO = 0.1
_ISLAND_GAP_RATIO = 0.02
_FLIPPED_NORMAL_NEAR_GROUND_BBOX_RATIO = 0.01


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
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute per-face mask scores in [0, 1].

    For each face centroid, projects to sampled camera views and computes:
    - outside_ratio: fraction of views where centroid falls outside object mask
    - ground_ratio: fraction of views where centroid falls inside ground mask

    Returns ``(outside_ratio, ground_ratio)`` arrays of shape ``(N_faces,)``
    or ``None`` if inputs are insufficient.
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

    return outside_ratio, ground_ratio


_CAP_ATLAS_STRIP_HEIGHT = 16


def _build_obj_vertex_uv_index(
    obj_mesh: _ObjMesh,
) -> dict[int, list[tuple[int, int]]]:
    """Map OBJ 1-based vertex ID → [(face_idx, corner_pos)] for UV lookup."""
    index: dict[int, list[tuple[int, int]]] = {}
    for fi, face in enumerate(obj_mesh.faces):
        for corner, vid in enumerate(face.vertex_indices):
            index.setdefault(vid, []).append((fi, corner))
    return index


def _sample_boundary_vertex_color(
    texture_path: Path,
    obj_mesh: _ObjMesh,
    merged_to_originals: list[list[int]],
    merged_vertex_ids: list[int],
    obj_vid_index: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    """Sample median texture color from a set of merged boundary vertices."""
    fallback = np.asarray([176, 176, 176], dtype=np.uint8)
    image = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    if image is None:
        return fallback
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    colors: list[np.ndarray] = []
    for mvid in merged_vertex_ids:
        if mvid < 0 or mvid >= len(merged_to_originals):
            continue
        originals = merged_to_originals[mvid]
        for orig_vid in originals:
            # merged_to_originals stores OBJ 1-based vertex indices already
            entries = obj_vid_index.get(orig_vid)
            if entries is None:
                continue
            for fi, corner in entries:
                vt_idx = obj_mesh.faces[fi].texcoord_indices[corner]
                if vt_idx <= 0 or vt_idx > len(obj_mesh.texcoords):
                    continue
                uv = obj_mesh.texcoords[vt_idx - 1]
                x = int(np.clip(round(float(uv[0]) * (w - 1)), 0, w - 1))
                y = int(np.clip(round((1.0 - float(uv[1])) * (h - 1)), 0, h - 1))
                colors.append(rgb[y, x].astype(np.float64))
    if not colors:
        return fallback
    median_color = np.median(np.vstack(colors), axis=0)
    return np.clip(np.round(median_color), 0, 255).astype(np.uint8)


def _write_cap_texture_atlas(
    path: Path,
    group_colors: dict[int, np.ndarray],
) -> tuple[int, dict[int, float]]:
    """Write a texture atlas with one horizontal color strip per group.

    Returns (atlas_height, group_v_centers) where group_v_centers maps each
    group ID to its V coordinate center in [0, 1].
    """
    n = max(len(group_colors), 1)
    atlas_h = n * _CAP_ATLAS_STRIP_HEIGHT
    atlas_w = _DEFAULT_CAP_TEXTURE_SIZE
    img = np.full((atlas_h, atlas_w, 3), 176, dtype=np.uint8)

    group_v_centers: dict[int, float] = {}
    sorted_groups = sorted(group_colors.keys())
    for strip_idx, gid in enumerate(sorted_groups):
        color = group_colors[gid]
        y_start = strip_idx * _CAP_ATLAS_STRIP_HEIGHT
        y_end = y_start + _CAP_ATLAS_STRIP_HEIGHT
        img[y_start:y_end, :] = color.reshape(1, 1, 3)
        # V center: OBJ textures have V=0 at bottom, image row 0 at top
        pixel_center = (y_start + y_end) / 2.0
        group_v_centers[gid] = 1.0 - pixel_center / atlas_h

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return atlas_h, group_v_centers


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


def _derive_original_face_mask(
    keep_merged_mask: np.ndarray,
    merged_face_to_obj_face: np.ndarray,
    num_obj_faces: int,
) -> np.ndarray:
    """Derive per-original-face keep mask from the merged-face keep mask.

    Faces not referenced by any merged face (e.g. degenerate faces skipped
    during vertex merging) default to kept (True).
    """
    keep_original = np.ones(num_obj_faces, dtype=bool)
    removed_indices = merged_face_to_obj_face[~keep_merged_mask]
    if removed_indices.size > 0:
        keep_original[removed_indices] = False
    return keep_original


def _filter_small_components(
    keep_merged_mask: np.ndarray,
    merged_faces: np.ndarray,
    min_face_ratio: float = _COMPONENT_MIN_FACE_RATIO,
    adjacency: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Remove small connected components from the kept face set (in-place).

    Uses shared-edge adjacency (via ``_build_face_adjacency``) and BFS to
    enumerate connected components among the currently-kept faces.  Components
    whose face count is below ``min_face_ratio * kept_count`` are marked as
    removed in *keep_merged_mask*.
    """
    if adjacency is None:
        adjacency = _build_face_adjacency(merged_faces)
    kept_indices = np.where(keep_merged_mask)[0]
    if len(kept_indices) == 0:
        return {"applied": False, "total_components": 0, "removed_components": 0,
                "removed_faces": 0, "largest_component_faces": 0}

    kept_set = set(kept_indices.tolist())
    visited: set[int] = set()
    components: list[list[int]] = []

    for start in kept_indices:
        start_int = int(start)
        if start_int in visited:
            continue
        queue = [start_int]
        visited.add(start_int)
        component: list[int] = []
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                nb = int(neighbor)
                if nb not in visited and nb in kept_set:
                    visited.add(nb)
                    queue.append(nb)
        components.append(component)

    min_faces = max(1, int(min_face_ratio * len(kept_indices)))
    removed_faces = 0
    removed_components = 0
    largest = 0
    for comp in components:
        largest = max(largest, len(comp))
        if len(comp) < min_faces:
            for idx in comp:
                keep_merged_mask[idx] = False
            removed_faces += len(comp)
            removed_components += 1

    return {
        "applied": True,
        "total_components": len(components),
        "removed_components": removed_components,
        "removed_faces": removed_faces,
        "largest_component_faces": largest,
    }


# ---------------------------------------------------------------------------
# Multi-pass cleanup functions
# ---------------------------------------------------------------------------


def _pass_geometric_clip(
    keep_merged_mask: np.ndarray,
    orig_face_dist: np.ndarray,
    adjacency: list[np.ndarray],
    merged_faces: np.ndarray,
    min_face_ratio: float = _COMPONENT_MIN_FACE_RATIO,
) -> dict[str, Any]:
    """Pass 1: remove faces below the ground plane and filter remnants."""
    below_plane = ~np.all(orig_face_dist > 0.0, axis=1)
    geometric_removed = int((keep_merged_mask & below_plane).sum())
    keep_merged_mask[below_plane] = False
    component_stats = _filter_small_components(
        keep_merged_mask, merged_faces, min_face_ratio, adjacency=adjacency,
    )
    return {"geometric_removed": geometric_removed, "component_stats": component_stats}


def _pass_mask_boundary_refinement(
    keep_merged_mask: np.ndarray,
    outside_ratio_arr: np.ndarray,
    ground_ratio_arr: np.ndarray,
    near_plane_band: np.ndarray,
    adjacency: list[np.ndarray],
    merged_faces: np.ndarray,
    min_face_ratio: float = _COMPONENT_MIN_FACE_RATIO,
) -> dict[str, Any]:
    """Pass 2: refine mask boundary — remove ground faces, restore object faces."""
    mask_removed_count = 0
    mask_preserved_count = 0
    for i in range(len(keep_merged_mask)):
        if not near_plane_band[i]:
            continue
        score = max(float(outside_ratio_arr[i]), float(ground_ratio_arr[i]))
        if keep_merged_mask[i] and score > _MASK_GROUND_REMOVAL_THRESHOLD:
            keep_merged_mask[i] = False
            mask_removed_count += 1
        elif not keep_merged_mask[i] and score < _MASK_OBJECT_PRESERVATION_THRESHOLD:
            keep_merged_mask[i] = True
            mask_preserved_count += 1
    component_stats = _filter_small_components(
        keep_merged_mask, merged_faces, min_face_ratio, adjacency=adjacency,
    )
    return {
        "applied": True,
        "mask_removed_count": mask_removed_count,
        "mask_preserved_count": mask_preserved_count,
        "near_plane_faces": int(near_plane_band.sum()),
        "component_stats": component_stats,
    }


def _pass_above_plane_noise(
    keep_merged_mask: np.ndarray,
    outside_ratio_arr: np.ndarray,
    near_plane_band: np.ndarray,
    min_face_dist: np.ndarray,
    adjacency: list[np.ndarray],
    merged_faces: np.ndarray,
    threshold: float = _MASK_ABOVE_PLANE_REMOVAL_THRESHOLD,
    min_face_ratio: float = _COMPONENT_MIN_FACE_RATIO,
) -> dict[str, Any]:
    """Pass 3: remove floating noise above the ground plane."""
    above_plane_candidates = (
        keep_merged_mask
        & ~near_plane_band
        & (min_face_dist > 0.0)
        & (outside_ratio_arr > threshold)
    )
    above_plane_removed = int(above_plane_candidates.sum())
    keep_merged_mask[above_plane_candidates] = False
    component_stats = _filter_small_components(
        keep_merged_mask, merged_faces, min_face_ratio, adjacency=adjacency,
    )
    return {
        "applied": True,
        "above_plane_removed": above_plane_removed,
        "threshold": threshold,
        "component_stats": component_stats,
    }


def _pass_lower_half_mask_noise(
    keep_merged_mask: np.ndarray,
    outside_ratio_arr: np.ndarray,
    centroid_heights: np.ndarray,
    half_height: float,
    adjacency: list[np.ndarray],
    merged_faces: np.ndarray,
    threshold: float = _MASK_LOWER_HALF_REMOVAL_THRESHOLD,
    min_face_ratio: float = _COMPONENT_MIN_FACE_RATIO,
) -> dict[str, Any]:
    """Pass 3b: remove noise in the lower half using full SAM2 mask trust."""
    candidates = (
        keep_merged_mask
        & (centroid_heights >= 0)
        & (centroid_heights <= half_height)
        & (outside_ratio_arr > threshold)
    )
    lower_half_removed = int(candidates.sum())
    keep_merged_mask[candidates] = False
    component_stats = _filter_small_components(
        keep_merged_mask, merged_faces, min_face_ratio, adjacency=adjacency,
    )
    return {
        "applied": True,
        "lower_half_removed": lower_half_removed,
        "threshold": threshold,
        "half_height": half_height,
        "component_stats": component_stats,
    }


def _pass_neighbor_erosion(
    keep_merged_mask: np.ndarray,
    adjacency: list[np.ndarray],
    merged_faces: np.ndarray,
    removal_ratio_threshold: float = _NEIGHBOR_REMOVAL_RATIO_THRESHOLD,
    max_iterations: int = _CONVERGENCE_MAX_ITERATIONS,
    min_face_ratio: float = _COMPONENT_MIN_FACE_RATIO,
) -> dict[str, Any]:
    """Pass 4: erode peninsulas where most neighbors are already removed.

    Each iteration uses a **snapshot** of the mask so that removal decisions
    are independent of face-index ordering, then applies all removals at once.
    """
    total_eroded = 0
    iterations = 0
    for iteration in range(max_iterations):
        # Snapshot: evaluate every kept face against the frozen state
        snapshot = keep_merged_mask.copy()
        kept_indices = np.where(snapshot)[0]
        to_erode: list[int] = []
        for fi in kept_indices:
            neighbors = adjacency[fi]
            if len(neighbors) == 0:
                continue
            removed_ratio = np.sum(~snapshot[neighbors]) / len(neighbors)
            if removed_ratio > removal_ratio_threshold:
                to_erode.append(fi)
        # Batch apply
        if to_erode:
            keep_merged_mask[to_erode] = False
        total_eroded += len(to_erode)
        iterations = iteration + 1
        if len(to_erode) == 0:
            break
    component_stats = _filter_small_components(
        keep_merged_mask, merged_faces, min_face_ratio, adjacency=adjacency,
    )
    return {
        "total_eroded": total_eroded,
        "iterations": iterations,
        "component_stats": component_stats,
    }


def _pass_floating_island_removal(
    keep_merged_mask: np.ndarray,
    merged_vertices: np.ndarray,
    merged_faces: np.ndarray,
    adjacency: list[np.ndarray],
    island_face_ratio: float = _ISLAND_FACE_RATIO,
    gap_ratio: float = _ISLAND_GAP_RATIO,
) -> dict[str, Any]:
    """Pass 5: remove floating islands far from the main body.

    Two removal criteria (OR):
    - **Spatial**: component vertices are all farther than
      ``gap_ratio * bbox_diagonal`` from the main body's bounding box.
    - **Size**: component has fewer than ``island_face_ratio`` of the
      largest component's face count.
    """
    kept_indices = np.where(keep_merged_mask)[0]
    if len(kept_indices) == 0:
        return {"removed_islands": 0, "removed_faces": 0}

    # BFS — enumerate connected components among kept faces
    kept_set = set(kept_indices.tolist())
    visited: set[int] = set()
    components: list[list[int]] = []

    for start in kept_indices:
        si = int(start)
        if si in visited:
            continue
        queue = [si]
        visited.add(si)
        comp: list[int] = []
        while queue:
            node = queue.pop()
            comp.append(node)
            for nb in adjacency[node]:
                nbi = int(nb)
                if nbi not in visited and nbi in kept_set:
                    visited.add(nbi)
                    queue.append(nbi)
        components.append(comp)

    if len(components) <= 1:
        return {"removed_islands": 0, "removed_faces": 0}

    # Identify main body (largest component)
    largest_comp = max(components, key=len)
    largest_id = id(largest_comp)
    min_faces = max(1, int(island_face_ratio * len(largest_comp)))

    # Main body bounding box
    main_vert_idx = np.unique(merged_faces[np.array(largest_comp, dtype=np.int64)].ravel())
    main_verts = merged_vertices[main_vert_idx]
    main_bbox_min = main_verts.min(axis=0)
    main_bbox_max = main_verts.max(axis=0)
    bbox_diag = float(np.linalg.norm(main_bbox_max - main_bbox_min))
    gap_threshold = gap_ratio * bbox_diag

    removed_islands = 0
    removed_faces = 0

    for comp in components:
        if id(comp) == largest_id:
            continue

        # Size criterion — small relative to main body
        is_small = len(comp) < min_faces

        # Spatial criterion — min vertex-to-bbox distance
        comp_vert_idx = np.unique(merged_faces[np.array(comp, dtype=np.int64)].ravel())
        comp_verts = merged_vertices[comp_vert_idx]
        clamped = np.clip(comp_verts, main_bbox_min, main_bbox_max)
        dists = np.linalg.norm(comp_verts - clamped, axis=1)
        min_dist = float(dists.min())
        is_distant = min_dist > gap_threshold

        if is_small or is_distant:
            for fi in comp:
                keep_merged_mask[fi] = False
            removed_islands += 1
            removed_faces += len(comp)

    return {"removed_islands": removed_islands, "removed_faces": removed_faces}


def _pass_flipped_ground_normals(
    keep_merged_mask: np.ndarray,
    merged_vertices: np.ndarray,
    merged_faces: np.ndarray,
    plane_normal: np.ndarray,
    orig_face_dist: np.ndarray,
    adjacency: list[np.ndarray],
    near_ground_bbox_ratio: float = _FLIPPED_NORMAL_NEAR_GROUND_BBOX_RATIO,
    min_face_ratio: float = _COMPONENT_MIN_FACE_RATIO,
) -> dict[str, Any]:
    """Pass 6: remove faces near the ground whose normals point away from it.

    Faces very close to the ground plane whose surface normals are opposite
    to the plane normal are thin reconstruction artifacts visible only from
    below.
    """
    bbox_diag = float(np.linalg.norm(
        merged_vertices.max(axis=0) - merged_vertices.min(axis=0)
    ))
    threshold = near_ground_bbox_ratio * bbox_diag
    min_dist = orig_face_dist.min(axis=1)
    near_ground = min_dist < threshold

    v0 = merged_vertices[merged_faces[:, 0]]
    v1 = merged_vertices[merged_faces[:, 1]]
    v2 = merged_vertices[merged_faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    dot_plane = face_normals @ plane_normal
    flipped = dot_plane < 0.0

    to_remove = keep_merged_mask & near_ground & flipped
    removed_count = int(to_remove.sum())
    keep_merged_mask[to_remove] = False

    component_stats = _filter_small_components(
        keep_merged_mask, merged_faces, min_face_ratio, adjacency=adjacency,
    )
    return {"flipped_removed": removed_count, "component_stats": component_stats}


def _pass_post_holefill_cleanup(
    keep_merged_mask: np.ndarray,
    merged_faces: np.ndarray,
    cleanup_faces: np.ndarray,
    capped_vertices: np.ndarray,
    cleanup_face_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    """Pass 8/10: remove disconnected components after hole-fill.

    After Pass 7 generates skirt + floor cap faces the combined mesh may still
    contain small disconnected fragments that survived earlier passes.  This
    pass builds a single adjacency graph over *all* remaining faces (kept
    merged + cleanup) and removes every connected component except the largest.
    """
    _empty_stats: dict[str, Any] = {
        "removed_components": 0,
        "removed_faces_merged": 0,
        "removed_faces_cleanup": 0,
        "removed_faces_total": 0,
    }
    kept_merged_indices = np.where(keep_merged_mask)[0]
    n_merged = len(kept_merged_indices)

    if n_merged == 0 and cleanup_faces.size == 0:
        return cleanup_faces, _empty_stats, cleanup_face_groups

    # Build combined face array indexing into capped_vertices
    parts = []
    if n_merged > 0:
        parts.append(merged_faces[kept_merged_indices])
    if cleanup_faces.size > 0:
        parts.append(cleanup_faces)
    combined_faces = np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.int64)
    total_combined = len(combined_faces)

    if total_combined <= 1:
        return cleanup_faces, _empty_stats, cleanup_face_groups

    adjacency = _build_face_adjacency(combined_faces)

    # BFS — enumerate connected components
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(total_combined):
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        comp: list[int] = []
        while queue:
            node = queue.pop()
            comp.append(node)
            for nb in adjacency[node]:
                nbi = int(nb)
                if nbi not in visited:
                    visited.add(nbi)
                    queue.append(nbi)
        components.append(comp)

    if len(components) <= 1:
        return cleanup_faces, _empty_stats, cleanup_face_groups

    # Keep only the largest component
    largest_comp = max(components, key=len)

    removed_merged = 0
    removed_cleanup = 0
    cleanup_keep_mask = np.ones(max(cleanup_faces.shape[0], 0), dtype=bool) if cleanup_faces.size > 0 else np.zeros(0, dtype=bool)

    for comp in components:
        if comp is largest_comp:
            continue
        for fi in comp:
            if fi < n_merged:
                keep_merged_mask[kept_merged_indices[fi]] = False
                removed_merged += 1
            else:
                ci = fi - n_merged
                cleanup_keep_mask[ci] = False
                removed_cleanup += 1

    if cleanup_faces.size > 0:
        cleanup_faces = cleanup_faces[cleanup_keep_mask]
        if cleanup_face_groups is not None:
            cleanup_face_groups = cleanup_face_groups[cleanup_keep_mask]
        if cleanup_faces.size == 0:
            cleanup_faces = np.zeros((0, 3), dtype=np.int64)
            cleanup_face_groups = np.zeros(0, dtype=np.int32) if cleanup_face_groups is not None else None

    return cleanup_faces, {
        "removed_components": len(components) - 1,
        "removed_faces_merged": removed_merged,
        "removed_faces_cleanup": removed_cleanup,
        "removed_faces_total": removed_merged + removed_cleanup,
    }, cleanup_face_groups


def _build_edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Build a mapping of (sorted) edge → face-use count."""
    if faces.size == 0:
        return {}
    edges = np.vstack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
    ).astype(np.int64)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return {(int(e[0]), int(e[1])): int(c) for e, c in zip(uniq, counts)}


def _cap_would_create_non_manifold(
    edge_counts: dict[tuple[int, int], int],
    new_faces: np.ndarray,
) -> bool:
    """Check whether *new_faces* would push any edge above count 2.

    Only edges incident to *new_faces* are inspected, so pre-existing
    non-manifold edges elsewhere in the mesh are ignored.
    """
    nf_edges = np.vstack(
        (new_faces[:, [0, 1]], new_faces[:, [1, 2]], new_faces[:, [2, 0]]),
    ).astype(np.int64)
    nf_edges = np.sort(nf_edges, axis=1)
    new_edge_adds: dict[tuple[int, int], int] = {}
    for e in nf_edges.tolist():
        key = (e[0], e[1])
        new_edge_adds[key] = new_edge_adds.get(key, 0) + 1
    for key, add in new_edge_adds.items():
        if edge_counts.get(key, 0) + add > 2:
            return True
    return False


def _update_edge_counts(
    edge_counts: dict[tuple[int, int], int],
    faces: np.ndarray,
) -> None:
    """Add *faces*' edges into a running edge-count dict (in place)."""
    fe = np.vstack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
    ).astype(np.int64)
    fe = np.sort(fe, axis=1)
    for e in fe.tolist():
        key = (e[0], e[1])
        edge_counts[key] = edge_counts.get(key, 0) + 1


def _split_self_touching_loop(loop_verts: list[int]) -> list[list[int]]:
    """Split a self-touching loop at repeated vertices into simple sub-loops.

    A self-touching loop visits some vertex more than once, forming a
    figure-8 or more complex topology.  This function peels off simple
    (non-self-touching) sub-loops at each repeated vertex until no
    duplicates remain.

    Returns a list of vertex lists, each representing a simple closed
    sub-loop (without the closing duplicate — caller appends ``[0]``).
    """
    result: list[list[int]] = []
    remaining = list(loop_verts)

    while True:
        seen: dict[int, int] = {}
        dup_start = -1
        dup_end = -1
        for i, v in enumerate(remaining):
            if v in seen:
                dup_start = seen[v]
                dup_end = i
                break
            seen[v] = i

        if dup_start == -1:
            # No duplicates — remaining is a clean simple loop.
            if len(remaining) >= 3:
                result.append(remaining)
            break

        # Extract the inner sub-loop between the two occurrences.
        sub = remaining[dup_start:dup_end]
        if len(sub) >= 3:
            result.append(sub)

        # Collapse the sub-loop out of remaining, keeping the junction
        # vertex once so the outer chain stays connected.
        remaining = remaining[: dup_start + 1] + remaining[dup_end + 1 :]

    return result


def _merge_open_boundary_paths(
    open_paths: list[list[int]],
    vertices: np.ndarray,
    close_gap_tolerance: float,
) -> list[list[int]]:
    """Chain open boundary paths at shared endpoints to form closed loops.

    Open paths arise when ``_extract_boundary_paths`` encounters junction
    vertices (>2 boundary edges) and the greedy walk splits a single loop
    into multiple open segments.  This function greedily concatenates
    segments whose endpoints coincide, then optionally closes near-closed
    chains whose remaining gap is within *close_gap_tolerance*.
    """
    if not open_paths:
        return []

    chains: list[list[int]] = [list(p) for p in open_paths]
    used = [False] * len(chains)
    closed: list[list[int]] = []

    for seed in range(len(chains)):
        if used[seed]:
            continue
        used[seed] = True
        chain = list(chains[seed])

        # Greedily extend chain by matching endpoints (exact vertex match).
        progress = True
        while progress:
            progress = False
            if chain[0] == chain[-1]:
                break
            for j in range(len(chains)):
                if used[j]:
                    continue
                p = chains[j]
                if chain[-1] == p[0]:
                    chain.extend(p[1:])
                    used[j] = True
                    progress = True
                    break
                if chain[-1] == p[-1]:
                    chain.extend(list(reversed(p))[1:])
                    used[j] = True
                    progress = True
                    break
                if chain[0] == p[-1]:
                    chain = p[:-1] + chain
                    used[j] = True
                    progress = True
                    break
                if chain[0] == p[0]:
                    chain = list(reversed(p))[:-1] + chain
                    used[j] = True
                    progress = True
                    break

        if chain[0] == chain[-1]:
            closed.append(chain)
        elif close_gap_tolerance > 0:
            d = float(np.linalg.norm(vertices[chain[0]] - vertices[chain[-1]]))
            if d <= close_gap_tolerance:
                chain.append(chain[0])
                closed.append(chain)

    return closed


def _pass_general_holefill(
    keep_merged_mask: np.ndarray,
    merged_faces: np.ndarray,
    cleanup_faces: np.ndarray,
    capped_vertices: np.ndarray,
    *,
    smooth_iterations: int = 3,
    smooth_lambda: float = 0.18,
    min_loop_vertices: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pass 9: fill all remaining boundary loops to make the mesh watertight.

    Unlike Pass 7 (bottom-only), this pass targets every remaining open
    boundary regardless of location or normal direction.  In addition to
    natively closed loops, open boundary paths are chained at shared
    endpoints and near-closed gaps to recover additional fillable loops.
    """
    stats: dict[str, Any] = {
        "loops_found": 0,
        "loops_capped": 0,
        "loops_skipped_small": 0,
        "self_touching_splits": 0,
        "loops_skipped_degenerate": 0,
        "loops_skipped_non_manifold": 0,
        "ear_clip_to_fan_fallbacks": 0,
        "open_paths_found": 0,
        "open_paths_merged": 0,
        "cap_faces_added": 0,
        "new_vertices": 0,
        "smoothing_seed_count": 0,
    }

    # --- build combined mesh ---
    kept_merged = merged_faces[keep_merged_mask]
    parts = [arr for arr in (kept_merged, cleanup_faces) if arr.size > 0]
    if not parts:
        return capped_vertices, cleanup_faces, stats
    combined_faces = np.vstack(parts)

    # --- boundary detection ---
    boundary_edges = _extract_boundary_edges(combined_faces)
    if boundary_edges.size == 0:
        return capped_vertices, cleanup_faces, stats

    raw_paths = _extract_boundary_paths(boundary_edges)

    # Separate closed loops from open paths
    closed_paths = [p for p in raw_paths if len(p) >= 3 and p[0] == p[-1]]
    open_paths = [p for p in raw_paths if len(p) >= 3 and p[0] != p[-1]]
    stats["open_paths_found"] = len(open_paths)

    # Try to merge open paths into closed loops
    if open_paths:
        bbox_diag = float(np.linalg.norm(
            capped_vertices.max(axis=0) - capped_vertices.min(axis=0),
        ))
        gap_tolerance = 0.02 * bbox_diag  # 2 % of bounding-box diagonal
        recovered = _merge_open_boundary_paths(open_paths, capped_vertices, gap_tolerance)
        stats["open_paths_merged"] = len(recovered)
        closed_paths.extend(recovered)

    # Split self-touching loops (vertex appears >1 time) into simple
    # sub-loops so each one can be triangulated independently.
    sanitised_paths: list[list[int]] = []
    n_split = 0
    for cp in closed_paths:
        verts = cp[:-1]  # strip closing duplicate
        if len(set(verts)) == len(verts):
            sanitised_paths.append(cp)
        else:
            subs = _split_self_touching_loop(verts)
            n_split += len(subs)
            for sub in subs:
                # Re-close into [v0, ..., vN, v0]
                sanitised_paths.append(sub + [sub[0]])
    stats["self_touching_splits"] = n_split
    closed_paths = sanitised_paths

    stats["loops_found"] = len(closed_paths)
    if not closed_paths:
        return capped_vertices, cleanup_faces, stats

    # mesh center for outward-normal orientation
    used_verts = np.unique(combined_faces.ravel())
    mesh_center = capped_vertices[used_verts].mean(axis=0)

    # Incremental edge-count dict — ignores pre-existing non-manifold edges
    # in combined_faces; only checks whether *new* cap faces would create new
    # non-manifold problems.
    edge_counts = _build_edge_counts(combined_faces)

    all_new_faces: list[np.ndarray] = []
    new_vertex_list: list[np.ndarray] = []
    smooth_seed_vertices: set[int] = set()
    boundary_loop_verts: list[list[int]] = []
    per_loop_face_counts: list[int] = []
    next_vertex = int(capped_vertices.shape[0])

    def _make_fan(loop_verts: list[int], centroid: np.ndarray) -> tuple[np.ndarray, bool]:
        """Create a centroid-fan triangulation (always non-manifold safe)."""
        nonlocal next_vertex
        centroid_idx = next_vertex
        new_vertex_list.append(centroid)
        next_vertex += 1
        n_lv = len(loop_verts)
        fan = [[centroid_idx, loop_verts[i], loop_verts[(i + 1) % n_lv]] for i in range(n_lv)]
        return np.asarray(fan, dtype=np.int64), True

    for path in closed_paths:
        loop_verts = path[:-1]  # strip duplicate closing vertex
        n = len(loop_verts)
        if n < min_loop_vertices:
            stats["loops_skipped_small"] += 1
            continue
        if len(set(loop_verts)) != n:
            continue  # safety: should not reach here after pre-split

        loop_indices = np.asarray(loop_verts, dtype=np.int64)
        loop_points = capped_vertices[loop_indices]
        centroid = loop_points.mean(axis=0)
        loop_normal = _fit_loop_normal(loop_points, centroid, mesh_center)

        # --- triangulate: ear-clip → local NM check → fan fallback ---
        uv = _loop_projection_uv(loop_points, loop_normal)
        local_tris = _triangulate_polygon_ear_clip(uv)
        used_fan = False

        if local_tris is not None and len(local_tris) > 0:
            new_faces = np.asarray(
                [[loop_verts[a], loop_verts[b], loop_verts[c]] for a, b, c in local_tris],
                dtype=np.int64,
            )
            # Ear-clip diagonals may coincide with existing interior edges;
            # if so, fall back to fan (which uses only boundary + centroid
            # edges and is inherently non-manifold safe).
            if _cap_would_create_non_manifold(edge_counts, new_faces):
                new_faces, used_fan = _make_fan(loop_verts, centroid)
                stats["ear_clip_to_fan_fallbacks"] += 1
        else:
            new_faces, used_fan = _make_fan(loop_verts, centroid)

        # resolve vertices for area / normal checks
        if new_vertex_list:
            all_verts = np.vstack((capped_vertices, np.asarray(new_vertex_list, dtype=np.float64)))
        else:
            all_verts = capped_vertices

        # filter degenerate triangles
        tri_pts = all_verts[new_faces]
        tri_cross = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
        tri_area = np.linalg.norm(tri_cross, axis=1) * 0.5
        valid_mask = tri_area > 1e-10
        new_faces = new_faces[valid_mask]
        if new_faces.shape[0] == 0:
            stats["loops_skipped_degenerate"] += 1
            if used_fan and new_vertex_list:
                new_vertex_list.pop()
                next_vertex -= 1
            continue

        # orient normals outward (toward mesh exterior)
        tri_pts = all_verts[new_faces]
        tri_normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
        tri_centers = tri_pts.mean(axis=1)
        outward_dir = tri_centers - mesh_center
        dots = np.sum(tri_normals * outward_dir, axis=1)
        flip_mask = dots < 0.0
        new_faces[flip_mask] = new_faces[flip_mask][:, ::-1]

        # Final non-manifold safety check (should rarely trigger for fans)
        if _cap_would_create_non_manifold(edge_counts, new_faces):
            stats["loops_skipped_non_manifold"] += 1
            if used_fan and new_vertex_list:
                new_vertex_list.pop()
                next_vertex -= 1
            continue

        # Accept — update running edge counts and collect faces
        _update_edge_counts(edge_counts, new_faces)
        all_new_faces.append(new_faces)
        smooth_seed_vertices.update(loop_verts)
        stats["loops_capped"] += 1
        stats["cap_faces_added"] += int(new_faces.shape[0])
        boundary_loop_verts.append(list(loop_verts))
        per_loop_face_counts.append(int(new_faces.shape[0]))

    # append new vertices
    if new_vertex_list:
        new_verts_arr = np.asarray(new_vertex_list, dtype=np.float64)
        if new_verts_arr.ndim == 1:
            new_verts_arr = new_verts_arr.reshape(1, 3)
        capped_vertices = np.vstack((capped_vertices, new_verts_arr))
        stats["new_vertices"] = len(new_vertex_list)

    # append new faces to cleanup_faces
    if all_new_faces:
        new_faces_arr = np.vstack(all_new_faces)
        cleanup_faces = (
            np.vstack((cleanup_faces, new_faces_arr))
            if cleanup_faces.size > 0
            else new_faces_arr
        )

    # local smoothing at junction edges
    stats["smoothing_seed_count"] = len(smooth_seed_vertices)
    if smooth_seed_vertices and smooth_iterations > 0:
        smooth_parts = [arr for arr in (merged_faces[keep_merged_mask], cleanup_faces) if arr.size > 0]
        if smooth_parts:
            smooth_combined = np.vstack(smooth_parts)
            capped_vertices = _apply_local_smoothing(
                capped_vertices,
                smooth_combined,
                smooth_seed_vertices,
                iterations=smooth_iterations,
                lamb=smooth_lambda,
            )

    stats["boundary_loops"] = boundary_loop_verts
    stats["per_loop_face_counts"] = per_loop_face_counts
    return capped_vertices, cleanup_faces, stats


def _aggregate_pass_stats(
    pass1: dict[str, Any],
    pass2: dict[str, Any],
    pass3: dict[str, Any],
    pass3b: dict[str, Any],
    pass4: dict[str, Any],
    pass5: dict[str, Any],
    outside_ratio_arr: np.ndarray | None,
    ground_ratio_arr: np.ndarray | None,
    keep_merged_mask: np.ndarray,
    pass6: dict[str, Any] | None = None,
    pass8: dict[str, Any] | None = None,
    pass9: dict[str, Any] | None = None,
    pass10: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate per-pass statistics into the structures expected by _proposal_payload."""
    all_comp = [p.get("component_stats", {}) for p in (pass1, pass2, pass3, pass3b, pass4)]
    if pass6 is not None:
        all_comp.append(pass6.get("component_stats", {}))
    total_component_removed = (
        sum(c.get("removed_faces", 0) for c in all_comp)
        + pass5.get("removed_faces", 0)
        + (pass8.get("removed_faces_total", 0) if pass8 else 0)
        + (pass10.get("removed_faces_total", 0) if pass10 else 0)
    )
    total_component_removed_components = (
        sum(c.get("removed_components", 0) for c in all_comp)
        + pass5.get("removed_islands", 0)
        + (pass8.get("removed_components", 0) if pass8 else 0)
        + (pass10.get("removed_components", 0) if pass10 else 0)
    )
    component_stats: dict[str, Any] = {
        "applied": any(c.get("applied", False) for c in all_comp) or pass5.get("removed_islands", 0) > 0,
        "total_components": max((c.get("total_components", 0) for c in all_comp), default=0),
        "removed_components": total_component_removed_components,
        "removed_faces": total_component_removed,
        "largest_component_faces": max(
            (c.get("largest_component_faces", 0) for c in all_comp), default=0,
        ),
    }

    mask_filtering_stats: dict[str, Any]
    if pass2.get("applied") or pass3.get("applied") or pass3b.get("applied"):
        combined_scores: np.ndarray | None = None
        if outside_ratio_arr is not None and ground_ratio_arr is not None:
            combined_scores = np.maximum(outside_ratio_arr, ground_ratio_arr)
        mask_filtering_stats = {
            "applied": True,
            "mask_removed_count": pass2.get("mask_removed_count", 0),
            "mask_preserved_count": pass2.get("mask_preserved_count", 0),
            "near_plane_faces": pass2.get("near_plane_faces", 0),
            "avg_ground_score_removed": (
                round(float(combined_scores[~keep_merged_mask].mean()), 4)
                if combined_scores is not None and np.any(~keep_merged_mask) else None
            ),
            "above_plane_filtering": {
                "removed_count": pass3.get("above_plane_removed", 0),
                "threshold": pass3.get("threshold", _MASK_ABOVE_PLANE_REMOVAL_THRESHOLD),
            },
            "lower_half_mask_noise": {
                "removed_count": pass3b.get("lower_half_removed", 0),
                "threshold": pass3b.get("threshold", _MASK_LOWER_HALF_REMOVAL_THRESHOLD),
                "half_height": pass3b.get("half_height", 0.0),
            },
            "component_filtering": component_stats,
            "neighbor_erosion": {
                "total_eroded": pass4.get("total_eroded", 0),
                "iterations": pass4.get("iterations", 0),
            },
            "floating_island_removal": {
                "removed_islands": pass5.get("removed_islands", 0),
                "removed_faces": pass5.get("removed_faces", 0),
            },
            "flipped_ground_normals": {
                "removed_count": pass6.get("flipped_removed", 0) if pass6 else 0,
            },
            "post_holefill_cleanup": {
                "removed_components": pass8.get("removed_components", 0) if pass8 else 0,
                "removed_faces": pass8.get("removed_faces_total", 0) if pass8 else 0,
            },
            "general_holefill": {
                "loops_capped": pass9.get("loops_capped", 0) if pass9 else 0,
                "cap_faces_added": pass9.get("cap_faces_added", 0) if pass9 else 0,
            },
            "final_cleanup": {
                "removed_components": pass10.get("removed_components", 0) if pass10 else 0,
                "removed_faces": pass10.get("removed_faces_total", 0) if pass10 else 0,
            },
        }
    else:
        mask_filtering_stats = {"applied": False}
        if pass5.get("removed_islands", 0) > 0 or (pass6 and pass6.get("flipped_removed", 0) > 0) or (pass8 and pass8.get("removed_faces_total", 0) > 0) or (pass9 and pass9.get("loops_capped", 0) > 0) or (pass10 and pass10.get("removed_faces_total", 0) > 0):
            mask_filtering_stats = {
                "applied": True,
                "floating_island_removal": {
                    "removed_islands": pass5["removed_islands"],
                    "removed_faces": pass5["removed_faces"],
                },
                "flipped_ground_normals": {
                    "removed_count": pass6.get("flipped_removed", 0) if pass6 else 0,
                },
                "post_holefill_cleanup": {
                    "removed_components": pass8.get("removed_components", 0) if pass8 else 0,
                    "removed_faces": pass8.get("removed_faces_total", 0) if pass8 else 0,
                },
                "general_holefill": {
                    "loops_capped": pass9.get("loops_capped", 0) if pass9 else 0,
                    "cap_faces_added": pass9.get("cap_faces_added", 0) if pass9 else 0,
                },
                "final_cleanup": {
                    "removed_components": pass10.get("removed_components", 0) if pass10 else 0,
                    "removed_faces": pass10.get("removed_faces_total", 0) if pass10 else 0,
                },
                "component_filtering": component_stats,
            }

    return mask_filtering_stats, component_stats


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
    selected_shift = 0.0
    actual_plane_d = plane_d
    section_loops = _extract_closed_section_loops(merged_vertices, merged_faces, plane_normal, actual_plane_d)
    selected_loop = max(section_loops, key=lambda loop: loop.area) if section_loops else None

    clipped_vertices, clipped_faces, clipped_source = _clip_mesh_at_plane(
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

    # Shared data computed once
    adjacency = _build_face_adjacency(merged_faces)
    orig_face_dist = merged_vertices[merged_faces] @ plane_normal + plane_d - float(selected_shift)
    keep_merged_mask = np.ones(len(merged_faces), dtype=bool)

    # Mask score pre-computation (expensive, done once)
    all_centroids = merged_vertices[merged_faces].mean(axis=1)
    face_score_result = _compute_per_face_ground_score(
        all_centroids,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        masks_dir=masks_dir,
        ground_masks_dir=ground_masks_dir,
    )
    outside_ratio_arr: np.ndarray | None = None
    ground_ratio_arr: np.ndarray | None = None
    near_plane_band: np.ndarray | None = None
    min_face_dist: np.ndarray | None = None
    if face_score_result is not None:
        _outside, _ground = face_score_result
        if len(_outside) == len(keep_merged_mask):
            outside_ratio_arr = _outside
            ground_ratio_arr = _ground
            min_face_dist = orig_face_dist.min(axis=1)
            bbox_diag = float(np.linalg.norm(
                merged_vertices.max(axis=0) - merged_vertices.min(axis=0)
            ))
            near_plane_threshold = max(
                _MASK_NEAR_PLANE_RATIO * float(selected_shift),
                _NEAR_PLANE_MIN_BBOX_RATIO * bbox_diag,
            )
            near_plane_band = np.abs(min_face_dist) < near_plane_threshold

    total_faces = len(merged_faces)

    # === Pass 1: Geometric clip (below ground plane) ===
    pass1 = _pass_geometric_clip(
        keep_merged_mask, orig_face_dist, adjacency, merged_faces,
    )
    p1_comp = pass1["component_stats"].get("removed_faces", 0)
    print(f"  Pass 1 (geometric clip): {pass1['geometric_removed']} below-plane"
          f" + {p1_comp} component-filter  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    # Compute half-height from Pass 1 kept vertices for lower/upper half split
    centroid_heights = all_centroids @ plane_normal + plane_d - float(selected_shift)
    kept_vert_indices = np.unique(merged_faces[keep_merged_mask].ravel())
    kept_vert_heights = merged_vertices[kept_vert_indices] @ plane_normal + plane_d - float(selected_shift)
    model_centroid_height = float(kept_vert_heights.mean())
    half_height = max(model_centroid_height / 2.0, 0.0)
    upper_half_mask = centroid_heights > half_height

    # === Pass 2: Mask boundary refinement (upper half only) ===
    pass2: dict[str, Any] = {"applied": False}
    if outside_ratio_arr is not None and near_plane_band is not None:
        pass2 = _pass_mask_boundary_refinement(
            keep_merged_mask, outside_ratio_arr, ground_ratio_arr,  # type: ignore[arg-type]
            near_plane_band & upper_half_mask, adjacency, merged_faces,
        )
        p2_comp = pass2["component_stats"].get("removed_faces", 0)
        print(f"  Pass 2 (mask boundary): -{pass2['mask_removed_count']} removed"
              f" +{pass2['mask_preserved_count']} restored"
              f" + {p2_comp} component-filter  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    # === Pass 3: Above-plane noise removal (upper half only) ===
    pass3: dict[str, Any] = {"applied": False}
    if outside_ratio_arr is not None and near_plane_band is not None and min_face_dist is not None:
        pass3 = _pass_above_plane_noise(
            keep_merged_mask, outside_ratio_arr, near_plane_band | ~upper_half_mask,
            min_face_dist, adjacency, merged_faces,
        )
        p3_comp = pass3["component_stats"].get("removed_faces", 0)
        print(f"  Pass 3 (above-plane noise): {pass3['above_plane_removed']} removed"
              f" + {p3_comp} component-filter  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    # === Pass 3b: Lower-half full-mask noise removal ===
    pass3b: dict[str, Any] = {"applied": False}
    if outside_ratio_arr is not None and half_height > 0:
        pass3b = _pass_lower_half_mask_noise(
            keep_merged_mask, outside_ratio_arr, centroid_heights,
            half_height, adjacency, merged_faces,
        )
        p3b_comp = pass3b["component_stats"].get("removed_faces", 0)
        print(f"  Pass 3b (lower-half mask): {pass3b['lower_half_removed']} removed"
              f" + {p3b_comp} component-filter  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    # === Pass 4: Neighbor erosion (peninsula removal) ===
    pass4 = _pass_neighbor_erosion(keep_merged_mask, adjacency, merged_faces)
    p4_comp = pass4["component_stats"].get("removed_faces", 0)
    print(f"  Pass 4 (erosion): {pass4['total_eroded']} eroded in {pass4['iterations']} iter"
          f" + {p4_comp} component-filter  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    # === Pass 5: Floating island removal ===
    pass5 = _pass_floating_island_removal(
        keep_merged_mask, merged_vertices, merged_faces, adjacency,
    )
    print(f"  Pass 5 (floating islands): {pass5['removed_islands']} islands"
          f" ({pass5['removed_faces']} faces)  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    # === Pass 6: Flipped normals at ground plane ===
    pass6 = _pass_flipped_ground_normals(
        keep_merged_mask, merged_vertices, merged_faces, plane_normal,
        orig_face_dist, adjacency,
    )
    p6_comp = pass6["component_stats"].get("removed_faces", 0)
    print(f"  Pass 6 (flipped normals): {pass6['flipped_removed']} removed"
          f" + {p6_comp} component-filter  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    removed_merged_faces = merged_faces[~keep_merged_mask]
    removed_obj_face_indices = np.unique(merged_face_to_obj_face[~keep_merged_mask])

    all_merged_face_keys = {
        tuple(sorted((int(face[0]), int(face[1]), int(face[2]))))
        for face in merged_faces
    }
    split_faces = np.asarray(
        [
            face
            for cf_idx, face in enumerate(clipped_faces)
            if tuple(sorted((int(face[0]), int(face[1]), int(face[2])))) not in all_merged_face_keys
            and keep_merged_mask[clipped_source[cf_idx]]
        ],
        dtype=np.int64,
    )
    if split_faces.size == 0:
        split_faces = np.zeros((0, 3), dtype=np.int64)
    # Remove split fragments whose normals point opposite to the ground plane
    if split_faces.shape[0] > 0:
        _sv0 = capped_vertices[split_faces[:, 0]]
        _sv1 = capped_vertices[split_faces[:, 1]]
        _sv2 = capped_vertices[split_faces[:, 2]]
        _split_dot = np.cross(_sv1 - _sv0, _sv2 - _sv0) @ plane_normal
        split_faces = split_faces[_split_dot >= 0.0]
        if split_faces.size == 0:
            split_faces = np.zeros((0, 3), dtype=np.int64)
    cap_faces = capped_faces[int(clipped_faces.shape[0]):].copy()
    if cap_faces.size == 0:
        cap_faces = np.zeros((0, 3), dtype=np.int64)
    cleanup_faces = np.vstack([arr for arr in (split_faces, cap_faces) if arr.size > 0]) if (split_faces.size > 0 or cap_faces.size > 0) else np.zeros((0, 3), dtype=np.int64)
    # Group 0 = split/cap faces from passes 1-6
    cleanup_face_groups = np.zeros(cleanup_faces.shape[0], dtype=np.int32)

    # === Pass 7: Bottom hole-fill (skirt + floor cap) ===
    kept_merged = merged_faces[keep_merged_mask]
    final_faces_for_skirt = (
        np.vstack((kept_merged, cleanup_faces))
        if cleanup_faces.size > 0
        else kept_merged
    )
    capped_vertices, skirt_cap_faces, skirt_stats = _generate_bottom_skirt_cap(
        capped_vertices,
        final_faces_for_skirt,
        plane_normal,
        actual_plane_d,
    )
    if skirt_cap_faces.size > 0:
        cleanup_faces = (
            np.vstack((cleanup_faces, skirt_cap_faces))
            if cleanup_faces.size > 0
            else skirt_cap_faces
        )
        # Group 1 = bottom skirt + floor cap
        cleanup_face_groups = np.concatenate((
            cleanup_face_groups,
            np.ones(skirt_cap_faces.shape[0], dtype=np.int32),
        ))
    print(
        f"  Pass 7 (bottom hole-fill): {skirt_stats['loops_found']} loops,"
        f" {skirt_stats['skirt_faces']} skirt + {skirt_stats['cap_faces']} cap faces,"
        f" {skirt_stats['new_vertices']} new vertices"
    )

    # === Pass 8: Post-hole-fill disconnected component removal ===
    cleanup_faces, pass8, cleanup_face_groups = _pass_post_holefill_cleanup(
        keep_merged_mask, merged_faces, cleanup_faces, capped_vertices,
        cleanup_face_groups,
    )
    print(f"  Pass 8 (post-holefill noise): {pass8['removed_components']} components"
          f" ({pass8['removed_faces_total']} faces)  →  {int(keep_merged_mask.sum())}/{total_faces} kept")

    # === Pass 9: General hole-fill (watertight) ===
    capped_vertices, cleanup_faces, pass9 = _pass_general_holefill(
        keep_merged_mask, merged_faces, cleanup_faces, capped_vertices,
    )
    # Groups 2+ = one per capped general hole-fill loop
    p9_loop_counts = pass9.get("per_loop_face_counts", [])
    p9_group_start = int(cleanup_face_groups.max()) + 1 if cleanup_face_groups.size > 0 else 2
    if p9_loop_counts:
        next_group = p9_group_start
        new_group_tags: list[np.ndarray] = []
        for lfc in p9_loop_counts:
            new_group_tags.append(np.full(lfc, next_group, dtype=np.int32))
            next_group += 1
        cleanup_face_groups = np.concatenate(
            [cleanup_face_groups] + new_group_tags,
        )
    print(
        f"  Pass 9 (general hole-fill): {pass9['loops_found']} loops"
        f" ({pass9['open_paths_merged']} recovered from {pass9['open_paths_found']} open paths),"
        f" {pass9['loops_capped']} capped,"
        f" {pass9['cap_faces_added']} cap faces,"
        f" {pass9['new_vertices']} new vertices"
    )

    # === Pass 10: Final disconnected component removal ===
    cleanup_faces, pass10, cleanup_face_groups = _pass_post_holefill_cleanup(
        keep_merged_mask, merged_faces, cleanup_faces, capped_vertices,
        cleanup_face_groups,
    )
    print(
        f"  Pass 10 (final cleanup): {pass10['removed_components']} components"
        f" ({pass10['removed_faces_total']} faces)"
        f"  ->  {int(keep_merged_mask.sum())}/{total_faces} kept"
    )

    # Aggregate statistics
    mask_filtering_stats, component_stats = _aggregate_pass_stats(
        pass1, pass2, pass3, pass3b, pass4, pass5,
        outside_ratio_arr, ground_ratio_arr, keep_merged_mask,
        pass6=pass6, pass8=pass8, pass9=pass9, pass10=pass10,
    )

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

    # Per-group color sampling
    group_colors: dict[int, np.ndarray] = {}
    # Group 0 = split/cap faces — sample from removed faces' texture (existing behavior)
    group_colors[0] = _sample_cap_color(texture_path, obj_mesh, removed_obj_face_indices)

    # Groups 1+ = boundary-based sampling
    unique_groups = sorted(set(int(g) for g in cleanup_face_groups)) if cleanup_face_groups is not None and cleanup_face_groups.size > 0 else []
    boundary_groups_to_sample: dict[int, list[int]] = {}  # group_id → merged vertex ids

    # Group 1 = bottom skirt/cap → boundary root vertices from Pass 7
    skirt_boundary_loops = skirt_stats.get("boundary_loops", [])
    if skirt_boundary_loops and 1 in unique_groups:
        all_skirt_boundary = []
        for loop in skirt_boundary_loops:
            all_skirt_boundary.extend(loop)
        boundary_groups_to_sample[1] = all_skirt_boundary

    # Groups from Pass 9 (IDs start at p9_group_start, matching tagging above)
    p9_boundary_loops = pass9.get("boundary_loops", [])
    if p9_boundary_loops:
        next_group_base = p9_group_start
        for loop_verts in p9_boundary_loops:
            gid = next_group_base
            next_group_base += 1
            if gid in unique_groups:
                boundary_groups_to_sample[gid] = loop_verts

    if boundary_groups_to_sample:
        obj_vid_index = _build_obj_vertex_uv_index(obj_mesh)
        for gid, mvids in boundary_groups_to_sample.items():
            group_colors[gid] = _sample_boundary_vertex_color(
                texture_path, obj_mesh, merged_to_originals, mvids, obj_vid_index,
            )

    # Ensure every group present in cleanup_face_groups has a color
    for gid in unique_groups:
        if gid not in group_colors:
            group_colors[gid] = group_colors[0]

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
        "keep_original_face_mask": _derive_original_face_mask(
            keep_merged_mask, merged_face_to_obj_face, len(obj_mesh.faces),
        ),
        "mask_score": mask_score,
        "mask_filtering": mask_filtering_stats,
        "component_filtering": component_stats,
        "cap_color": group_colors[0],
        "group_colors": group_colors,
        "cleanup_face_groups": cleanup_face_groups,
        "matched_boundary_area": float(matched_boundary_area),
        "skirt_stats": skirt_stats,
        "pass8_stats": pass8,
        "pass9_stats": pass9,
        "pass10_stats": pass10,
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
        "component_removed_faces": analysis.get("component_filtering", {}).get("removed_faces", 0),
        "above_plane_removed_faces": analysis.get("mask_filtering", {}).get("above_plane_filtering", {}).get("removed_count", 0),
        "lower_half_removed_faces": analysis.get("mask_filtering", {}).get("lower_half_mask_noise", {}).get("removed_count", 0),
        "bottom_hole_fill": analysis.get("skirt_stats", {}),
        "post_holefill_cleanup": analysis.get("pass8_stats", {}),
        "general_holefill": analysis.get("pass9_stats", {}),
        "final_cleanup": analysis.get("pass10_stats", {}),
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
            "skirt_face_count": analysis.get("skirt_stats", {}).get("skirt_faces", 0),
            "floor_cap_face_count": analysis.get("skirt_stats", {}).get("cap_faces", 0),
            "general_holefill_faces": analysis.get("pass9_stats", {}).get("cap_faces_added", 0),
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

    skirt = analysis.get("skirt_stats", {})
    if skirt.get("loops_capped", 0) > 0:
        _emit_progress(
            progress_cb, 5.0,
            f"Bottom hole-fill: {skirt['skirt_faces']} skirt"
            f" + {skirt['cap_faces']} cap faces",
        )
    else:
        _emit_progress(progress_cb, 5.0, "Bottom hole-fill: no bottom loops")

    p8 = analysis.get("pass8_stats") or {}
    _emit_progress(
        progress_cb, 7.0,
        f"Post-holefill noise removal: {p8.get('removed_components', 0)} components"
        f" ({p8.get('removed_faces_total', 0)} faces)",
    )

    p9 = analysis.get("pass9_stats") or {}
    _emit_progress(
        progress_cb, 9.0,
        f"General hole-fill: {p9.get('loops_capped', 0)}/{p9.get('loops_found', 0)} loops capped"
        f" ({p9.get('cap_faces_added', 0)} faces)",
    )

    p10 = analysis.get("pass10_stats") or {}
    _emit_progress(
        progress_cb, 11.0,
        f"Final cleanup: {p10.get('removed_components', 0)} components"
        f" ({p10.get('removed_faces_total', 0)} faces)",
    )

    _emit_progress(progress_cb, 15.0, "Preparing cleanup geometry")

    obj_mesh: _ObjMesh = analysis["obj_mesh"]
    keep_original_face_mask = analysis["keep_original_face_mask"]
    cleanup_faces = analysis["cleanup_faces"]
    capped_vertices = analysis["capped_vertices"]
    merged_to_originals = analysis["merged_to_originals"]
    merged_original_count = int(analysis["merged_vertices"].shape[0])
    plane_normal = analysis["plane_normal"]

    texture_cap = _cap_texture_path(output_root)
    group_colors: dict[int, np.ndarray] = analysis["group_colors"]
    cleanup_face_groups: np.ndarray | None = analysis["cleanup_face_groups"]
    _atlas_h, group_v_centers = _write_cap_texture_atlas(texture_cap, group_colors)

    cleanup_vertex_ids = sorted({int(idx) for idx in cleanup_faces.reshape(-1)}) if cleanup_faces.size > 0 else []

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

    # Per-face-corner UV allocation: each cleanup face gets 3 UV entries
    # mapped to (0.5, group_v_center) for its group.  This avoids the
    # vertex-sharing-between-groups issue.
    next_vt_index = int(obj_mesh.texcoords.shape[0]) + 1
    appended_vts: list[np.ndarray] = []
    cleanup_face_vt_indices: list[tuple[int, int, int]] = []
    for fi in range(cleanup_faces.shape[0]):
        gid = int(cleanup_face_groups[fi]) if cleanup_face_groups is not None and fi < len(cleanup_face_groups) else 0
        v_center = group_v_centers.get(gid, group_v_centers.get(0, 0.5))
        uv = np.array([0.5, v_center], dtype=np.float64)
        t0 = next_vt_index
        t1 = next_vt_index + 1
        t2 = next_vt_index + 2
        appended_vts.extend([uv, uv, uv])
        next_vt_index += 3
        cleanup_face_vt_indices.append((t0, t1, t2))

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
        for fi, face in enumerate(cleanup_faces):
            a, b, c = (int(face[0]), int(face[1]), int(face[2]))
            va = merged_to_output_vertex[a]
            vb = merged_to_output_vertex[b]
            vc = merged_to_output_vertex[c]
            ta, tb, tc = cleanup_face_vt_indices[fi]
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
