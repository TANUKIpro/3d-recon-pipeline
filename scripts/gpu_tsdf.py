"""GPU-accelerated TSDF fusion using Open3D VoxelBlockGrid (CUDA).

Replaces gs2mesh's CPU ``ScalableTSDFVolume`` with the tensor-based
``VoxelBlockGrid`` running on CUDA for significant speedup.
Reads gs2mesh rendered views + stereo depth, fuses on GPU, and
extracts a cleaned triangle mesh.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path
from typing import Callable

from scripts.config_defaults import (
    GS2MESH_MASK_DEPTH_MODE,
    GS2MESH_MASK_DEPTH_MODES,
    GS2MESH_SILHOUETTE_CONSENSUS,
    GS2MESH_SILHOUETTE_FILL,
    GS2MESH_SILHOUETTE_MASK_DILATE_PX,
    GS2MESH_SILHOUETTE_MIN_VIEWS,
    GS2MESH_SILHOUETTE_VOXELS,
)


@dataclass(frozen=True)
class _SilhouetteFillConfig:
    enabled: bool
    depth_mode: str
    voxel_resolution: int
    min_views: int
    consensus: float
    mask_dilate_px: int


@dataclass(frozen=True)
class _VisualHull:
    vertices: np.ndarray
    triangles: np.ndarray
    occupied_count: int
    total_voxels: int
    voxel_size: float


@dataclass
class _SilhouetteFillRuntime:
    scene: object
    hull: _VisualHull
    filled_pixels: int = 0
    filled_frames: int = 0


def gpu_tsdf_reconstruct(
    output_dir_root: str,
    stereo_model: str,
    *,
    tsdf_voxel: int = 2,
    tsdf_sdf_trunc: float = 0.04,
    tsdf_scale: float = 1.0,
    tsdf_min_depth_baselines: int = 4,
    tsdf_max_depth_baselines: int = 20,
    tsdf_dilate: int = 1,
    tsdf_cleaning_threshold: int = 100_000,
    tsdf_use_mask: bool = True,
    tsdf_erode_mask: bool = True,
    tsdf_use_occlusion_mask: bool = True,
    tsdf_invert_mask: bool = False,
    tsdf_erosion_kernel_size: int = 10,
    tsdf_closing_kernel_size: int = 10,
    block_count: int = 100_000,
    tsdf_device: str = "CUDA:0",
    mesh_target_faces: int = 0,
    mesh_min_iou: float = 0.985,
    mesh_iou_views: int = 8,
    mesh_smooth_iters: int = 0,
    mesh_smooth_lambda: float = 0.5,
    mesh_smooth_mu: float = -0.53,
    mesh_smooth_min_iou: float = 0.985,
    mask_depth_mode: str | None = None,
    silhouette_voxel_resolution: int | None = None,
    silhouette_extra_views_path: str | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> str:
    """Run GPU TSDF fusion and return path to cleaned mesh PLY.

    Parameters match gs2mesh TSDF parameters; defaults are for the
    *custom* dataset profile.
    """
    # Heavy imports deferred so the module can be imported without GPU.
    import open3d as o3d
    import open3d.core as o3c
    from PIL import Image

    from scripts.vram_utils import log_vram

    root = Path(output_dir_root)
    device_is_cuda = tsdf_device.lower().startswith("cuda")
    try:
        return _gpu_tsdf_impl(
            root=root,
            stereo_model=stereo_model,
            tsdf_voxel=tsdf_voxel,
            tsdf_sdf_trunc=tsdf_sdf_trunc,
            tsdf_scale=tsdf_scale,
            tsdf_min_depth_baselines=tsdf_min_depth_baselines,
            tsdf_max_depth_baselines=tsdf_max_depth_baselines,
            tsdf_dilate=tsdf_dilate,
            tsdf_cleaning_threshold=tsdf_cleaning_threshold,
            tsdf_use_mask=tsdf_use_mask,
            tsdf_erode_mask=tsdf_erode_mask,
            tsdf_use_occlusion_mask=tsdf_use_occlusion_mask,
            tsdf_invert_mask=tsdf_invert_mask,
            tsdf_erosion_kernel_size=tsdf_erosion_kernel_size,
            tsdf_closing_kernel_size=tsdf_closing_kernel_size,
            block_count=block_count,
            tsdf_device=tsdf_device,
            mesh_target_faces=mesh_target_faces,
            mesh_min_iou=mesh_min_iou,
            mesh_iou_views=mesh_iou_views,
            mesh_smooth_iters=mesh_smooth_iters,
            mesh_smooth_lambda=mesh_smooth_lambda,
            mesh_smooth_mu=mesh_smooth_mu,
            mesh_smooth_min_iou=mesh_smooth_min_iou,
            mask_depth_mode=mask_depth_mode,
            silhouette_voxel_resolution=silhouette_voxel_resolution,
            silhouette_extra_views_path=silhouette_extra_views_path,
            progress_cb=progress_cb,
        )
    finally:
        # VoxelBlockGrid on CUDA allocates ~8GB directly from Open3D's
        # own memory pool (not PyTorch's). torch.cuda.empty_cache()
        # cannot reclaim it — we must call open3d.core.cuda.release_cache()
        # explicitly. Only do any of this when the run actually used CUDA.
        import gc

        if device_is_cuda:
            log_vram("before GPU TSDF release")
            try:
                o3c.cuda.release_cache()
            except Exception as exc:  # pragma: no cover — API varies by version
                print(f"open3d release_cache failed (continuing): {exc}")
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            log_vram("after GPU TSDF release")
        else:
            gc.collect()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int, *, min_value: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float,
    max_value: float,
) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, min_value), max_value)


def _normalize_mask_depth_mode(value: str | None, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in GS2MESH_MASK_DEPTH_MODES:
        return raw
    if raw in {"replace", "override", "visual_hull", "visual-hull"}:
        return "replace"
    if raw in {"fill", "holes", "hole_fill", "hole-fill"}:
        return "fill"
    return fallback if fallback in GS2MESH_MASK_DEPTH_MODES else "crop"


def _resolve_mask_depth_mode(explicit_mode: str | None) -> str:
    if explicit_mode is not None:
        return _normalize_mask_depth_mode(explicit_mode, GS2MESH_MASK_DEPTH_MODE)
    raw = os.environ.get("GS2MESH_MASK_DEPTH_MODE")
    if raw is not None and raw.strip():
        return _normalize_mask_depth_mode(raw, GS2MESH_MASK_DEPTH_MODE)
    legacy_mode = os.environ.get("GS2MESH_SILHOUETTE_DEPTH_MODE")
    if legacy_mode is not None and legacy_mode.strip():
        return _normalize_mask_depth_mode(legacy_mode, GS2MESH_MASK_DEPTH_MODE)
    if _env_bool("GS2MESH_SILHOUETTE_FILL", GS2MESH_SILHOUETTE_FILL):
        return "fill"
    return GS2MESH_MASK_DEPTH_MODE


def _resolve_silhouette_fill_config(
    *,
    mask_depth_mode: str | None = None,
    default_voxel_resolution: int | None = None,
) -> _SilhouetteFillConfig:
    voxel_default = (
        int(default_voxel_resolution)
        if default_voxel_resolution is not None
        else GS2MESH_SILHOUETTE_VOXELS
    )
    resolved_mode = _resolve_mask_depth_mode(mask_depth_mode)
    return _SilhouetteFillConfig(
        enabled=resolved_mode in {"fill", "replace"},
        depth_mode=resolved_mode,
        voxel_resolution=_env_int(
            "GS2MESH_SILHOUETTE_VOXELS",
            voxel_default,
            min_value=16,
        ),
        min_views=_env_int(
            "GS2MESH_SILHOUETTE_MIN_VIEWS",
            GS2MESH_SILHOUETTE_MIN_VIEWS,
            min_value=1,
        ),
        consensus=_env_float(
            "GS2MESH_SILHOUETTE_CONSENSUS",
            GS2MESH_SILHOUETTE_CONSENSUS,
            min_value=0.5,
            max_value=1.0,
        ),
        mask_dilate_px=_env_int(
            "GS2MESH_SILHOUETTE_MASK_DILATE_PX",
            GS2MESH_SILHOUETTE_MASK_DILATE_PX,
            min_value=0,
        ),
    )


def _load_left_mask(cam_dir: Path) -> np.ndarray | None:
    mask_path = cam_dir / "left_mask.npy"
    if not mask_path.exists():
        return None
    return np.load(str(mask_path)).astype(bool)


def _resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask.astype(bool, copy=False)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def _dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool, copy=False)
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def _prepare_tsdf_object_mask(
    raw_mask: np.ndarray,
    *,
    invert: bool,
    erode: bool,
    erosion_kernel_size: int,
    closing_kernel_size: int,
) -> np.ndarray:
    object_mask = raw_mask.astype(bool, copy=False)
    if invert:
        object_mask = ~object_mask
    if erode:
        erosion_kernel_size = max(1, int(erosion_kernel_size))
        closing_kernel_size = max(1, int(closing_kernel_size))
        closing_kernel = np.ones(
            (closing_kernel_size, closing_kernel_size),
            np.uint8,
        )
        erosion_kernel = np.ones(
            (erosion_kernel_size, erosion_kernel_size),
            np.uint8,
        )
        closing = cv2.morphologyEx(
            object_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            closing_kernel,
        )
        erosion = cv2.erode(closing, erosion_kernel, iterations=1)
        object_mask = erosion > 0.5
    return object_mask


def _scaled_c2w(camera: dict, tsdf_scale: float) -> np.ndarray:
    c2w = np.array(camera["extrinsic"], dtype=np.float64)
    c2w[:3, 3] /= tsdf_scale
    return c2w


def _world_to_camera(
    points: np.ndarray,
    camera: dict,
    tsdf_scale: float,
) -> np.ndarray:
    w2c = np.linalg.inv(_scaled_c2w(camera, tsdf_scale))
    return points @ w2c[:3, :3].T + w2c[:3, 3]


def _project_points_to_mask(
    points: np.ndarray,
    camera: dict,
    mask_shape: tuple[int, int],
    *,
    tsdf_scale: float,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam_pts = _world_to_camera(points, camera, tsdf_scale)
    z = cam_pts[:, 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, 1.0)
    u = camera["fx"] * (cam_pts[:, 0] / safe_z) + camera["cx"]
    v = camera["fy"] * (cam_pts[:, 1] / safe_z) + camera["cy"]
    h, w = mask_shape
    valid = (
        (z > depth_min)
        & (z < depth_max)
        & (u >= 0.0)
        & (u < float(w))
        & (v >= 0.0)
        & (v < float(h))
    )
    ui = np.clip(np.rint(u).astype(np.int64), 0, max(w - 1, 0))
    vi = np.clip(np.rint(v).astype(np.int64), 0, max(h - 1, 0))
    return ui, vi, z, valid


def _backproject_pixels_to_world(
    pixels: np.ndarray,
    depth: float,
    camera: dict,
    tsdf_scale: float,
) -> np.ndarray:
    x = (pixels[:, 0] - float(camera["cx"])) / float(camera["fx"]) * depth
    y = (pixels[:, 1] - float(camera["cy"])) / float(camera["fy"]) * depth
    z = np.full_like(x, depth, dtype=np.float64)
    cam_pts = np.column_stack([x, y, z])
    c2w = _scaled_c2w(camera, tsdf_scale)
    return cam_pts @ c2w[:3, :3].T + c2w[:3, 3]


def _mask_bbox_pixels(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0 = float(xs.min())
    x1 = float(xs.max())
    y0 = float(ys.min())
    y1 = float(ys.max())
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [(x0 + x1) * 0.5, (y0 + y1) * 0.5],
        ],
        dtype=np.float64,
    )


def _load_visual_hull_masks(
    root: Path,
    n_cameras: int,
    *,
    mask_dilate_px: int,
) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for cam_idx in range(n_cameras):
        mask = _load_left_mask(root / f"{cam_idx:03d}")
        if mask is None or not mask.any():
            continue
        masks[cam_idx] = _dilate_mask(mask, mask_dilate_px)
    return masks


def _load_visual_hull_extra_views(
    extra_views_path: str | Path | None,
    *,
    mask_dilate_px: int,
) -> list[tuple[dict, np.ndarray]]:
    if extra_views_path is None:
        return []
    path = Path(extra_views_path)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"GPU TSDF: skipped SAM2 extra visual-hull views ({exc})")
        return []

    records = data.get("views") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []

    views: list[tuple[dict, np.ndarray]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        camera = record.get("camera")
        mask_raw = record.get("mask_path")
        if not isinstance(camera, dict) or not isinstance(mask_raw, str):
            continue
        mask_path = Path(mask_raw)
        if not mask_path.is_absolute():
            mask_path = path.parent / mask_path
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        mask = mask_img > 0
        if not mask.any():
            continue
        if not all(key in camera for key in ("fx", "fy", "cx", "cy", "extrinsic")):
            continue
        views.append((camera, _dilate_mask(mask, mask_dilate_px)))

    if views:
        print(f"GPU TSDF: loaded {len(views)} SAM2 extra visual-hull views")
    return views


def _estimate_visual_hull_bounds(
    left_cameras: list[dict],
    masks_by_idx: dict[int, np.ndarray],
    *,
    depth_min: float,
    depth_max: float,
    tsdf_scale: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    points: list[np.ndarray] = []
    for cam_idx, mask in masks_by_idx.items():
        bbox = _mask_bbox_pixels(mask)
        if bbox is None:
            continue
        cam = left_cameras[cam_idx]
        points.append(
            _backproject_pixels_to_world(bbox, depth_min, cam, tsdf_scale)
        )
        points.append(
            _backproject_pixels_to_world(bbox, depth_max, cam, tsdf_scale)
        )
    if not points:
        return None
    all_points = np.vstack(points)
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    extent = bbox_max - bbox_min
    margin = max(float(np.max(extent)) * 0.05, 1e-3)
    return bbox_min - margin, bbox_max + margin


def _carve_visual_hull_occupancy(
    left_cameras: list[dict],
    masks_by_idx: dict[int, np.ndarray],
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_resolution: int,
    min_views: int,
    consensus: float,
    depth_min: float,
    depth_max: float,
    tsdf_scale: float,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, float]:
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    voxel_size = float(np.max(extent)) / float(max(1, voxel_resolution))
    dims_arr = np.maximum(np.ceil(extent / voxel_size).astype(np.int64), 2)

    max_total_voxels = 12_000_000
    total = int(np.prod(dims_arr))
    if total > max_total_voxels:
        scale = (float(total) / float(max_total_voxels)) ** (1.0 / 3.0)
        voxel_size *= scale
        dims_arr = np.maximum(np.ceil(extent / voxel_size).astype(np.int64), 2)
        total = int(np.prod(dims_arr))

    dims = tuple(int(v) for v in dims_arr)
    occupancy = np.zeros(total, dtype=bool)
    camera_items = list(masks_by_idx.items())
    effective_min_views = max(1, min(int(min_views), len(camera_items)))
    chunk_size = 250_000
    total_chunks = max(1, (total + chunk_size - 1) // chunk_size)

    for chunk_idx, start in enumerate(range(0, total, chunk_size), start=1):
        end = min(start + chunk_size, total)
        flat_idx = np.arange(start, end, dtype=np.int64)
        ix, iy, iz = np.unravel_index(flat_idx, dims)
        points = np.column_stack(
            [
                bbox_min[0] + (ix.astype(np.float64) + 0.5) * voxel_size,
                bbox_min[1] + (iy.astype(np.float64) + 0.5) * voxel_size,
                bbox_min[2] + (iz.astype(np.float64) + 0.5) * voxel_size,
            ]
        )
        valid_counts = np.zeros(len(points), dtype=np.uint16)
        inside_counts = np.zeros(len(points), dtype=np.uint16)

        for cam_idx, mask in camera_items:
            ui, vi, _z, valid = _project_points_to_mask(
                points,
                left_cameras[cam_idx],
                mask.shape[:2],
                tsdf_scale=tsdf_scale,
                depth_min=depth_min,
                depth_max=depth_max,
            )
            if not np.any(valid):
                continue
            valid_counts[valid] += 1
            inside_counts[valid] += mask[vi[valid], ui[valid]]

        enough_views = valid_counts >= effective_min_views
        inside_ratio = np.zeros(len(points), dtype=np.float32)
        np.divide(
            inside_counts,
            np.maximum(valid_counts, 1),
            out=inside_ratio,
            where=valid_counts > 0,
        )
        occupancy[start:end] = enough_views & (inside_ratio >= consensus)
        if progress_cb:
            progress_cb(chunk_idx, total_chunks)

    return occupancy.reshape(dims), voxel_size


def _build_visual_hull(
    *,
    root: Path,
    left_cameras: list[dict],
    depth_min: float,
    depth_max: float,
    tsdf_scale: float,
    config: _SilhouetteFillConfig,
    extra_views_path: str | Path | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> _VisualHull | None:
    from skimage import measure

    if progress_cb:
        progress_cb(0.05, "Loading SAM2 visual hull masks")
    masks_by_idx = _load_visual_hull_masks(
        root,
        len(left_cameras),
        mask_dilate_px=config.mask_dilate_px,
    )
    if not masks_by_idx:
        print("GPU TSDF: silhouette fill skipped (no left_mask.npy files)")
        return None
    hull_cameras = list(left_cameras)
    if progress_cb:
        progress_cb(
            0.15,
            f"Loaded SAM2 visual hull masks ({len(masks_by_idx)} views)",
        )
    extra_views = _load_visual_hull_extra_views(
        extra_views_path,
        mask_dilate_px=config.mask_dilate_px,
    )
    for camera, mask in extra_views:
        masks_by_idx[len(hull_cameras)] = mask
        hull_cameras.append(camera)
    if progress_cb and extra_views:
        progress_cb(
            0.20,
            f"Loaded SAM2 primary extra visual hull views ({len(extra_views)} views)",
        )

    if progress_cb:
        progress_cb(0.25, "Estimating SAM2 visual hull bounds")
    bounds = _estimate_visual_hull_bounds(
        hull_cameras,
        masks_by_idx,
        depth_min=depth_min,
        depth_max=depth_max,
        tsdf_scale=tsdf_scale,
    )
    if bounds is None:
        print("GPU TSDF: silhouette fill skipped (empty masks)")
        return None

    bbox_min, bbox_max = bounds

    def _carve_progress(chunk_idx: int, total_chunks: int) -> None:
        if not progress_cb:
            return
        ratio = chunk_idx / max(total_chunks, 1)
        progress_cb(
            0.30 + min(ratio, 1.0) * 0.55,
            f"Carving SAM2 visual hull ({chunk_idx}/{total_chunks})",
        )

    if progress_cb:
        progress_cb(0.30, "Carving SAM2 visual hull")
    occupancy, voxel_size = _carve_visual_hull_occupancy(
        hull_cameras,
        masks_by_idx,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_resolution=config.voxel_resolution,
        min_views=config.min_views,
        consensus=config.consensus,
        depth_min=depth_min,
        depth_max=depth_max,
        tsdf_scale=tsdf_scale,
        progress_cb=_carve_progress,
    )
    occupied_count = int(np.count_nonzero(occupancy))
    total_voxels = int(occupancy.size)
    if occupied_count == 0:
        print(
            "GPU TSDF: silhouette fill skipped "
            f"(visual hull empty, total_voxels={total_voxels})"
        )
        return None

    if progress_cb:
        progress_cb(0.90, "Extracting SAM2 visual hull mesh")
    padded = np.pad(occupancy.astype(np.float32), 1, constant_values=0.0)
    vertices, triangles, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
    )
    vertices = vertices + bbox_min - (0.5 * voxel_size)
    return _VisualHull(
        vertices=vertices.astype(np.float32),
        triangles=triangles.astype(np.int32),
        occupied_count=occupied_count,
        total_voxels=total_voxels,
        voxel_size=voxel_size,
    )


def _build_visual_hull_scene(hull: _VisualHull):
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(hull.vertices.astype(np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(hull.triangles.astype(np.int32))
    legacy.remove_degenerate_triangles()
    legacy.remove_duplicated_triangles()
    legacy.remove_unreferenced_vertices()
    legacy.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _render_visual_hull_depth(
    scene,
    camera: dict,
    image_shape: tuple[int, int],
    *,
    tsdf_scale: float,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    import open3d.core as o3c

    h, w = image_shape
    c2w = _scaled_c2w(camera, tsdf_scale)
    w2c = np.linalg.inv(c2w)
    intrinsic = np.array(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3c.Tensor(intrinsic),
        extrinsic_matrix=o3c.Tensor(w2c),
        width_px=int(w),
        height_px=int(h),
    )
    t_hit = scene.cast_rays(rays)["t_hit"].numpy().reshape(h, w)
    finite = np.isfinite(t_hit)
    depth = np.full((h, w), np.inf, dtype=np.float32)
    if not np.any(finite):
        return depth

    rays_np = rays.numpy().reshape(h, w, 6)
    hit_t = t_hit[finite][:, None]
    hit_points = (
        rays_np[..., :3][finite]
        + rays_np[..., 3:][finite] * hit_t
    )
    cam_pts = hit_points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam_pts[:, 2].astype(np.float32)
    valid_z = (z > depth_min) & (z < depth_max)
    finite_indices = np.flatnonzero(finite)
    depth_flat = depth.reshape(-1)
    depth_flat[finite_indices[valid_z]] = z[valid_z]
    return depth


def _apply_silhouette_depth_fill(
    depth: np.ndarray,
    object_mask: np.ndarray | None,
    hull_depth: np.ndarray,
    *,
    depth_min: float,
    depth_max: float,
    mode: str = "fill",
) -> tuple[np.ndarray, int]:
    if object_mask is None:
        return depth, 0
    hull_valid = (
        np.isfinite(hull_depth)
        & (hull_depth > depth_min)
        & (hull_depth < depth_max)
    )
    if mode == "replace":
        filled = depth.copy()
        # In replace mode SAM2 is authoritative inside the object mask: do not
        # let stale transparent-object stereo depth leak through where the
        # visual hull has no support.
        filled[object_mask] = 0.0
        fill_pixels = object_mask & hull_valid
        count = int(np.count_nonzero(fill_pixels))
        if count > 0:
            filled[fill_pixels] = hull_depth[fill_pixels]
        return filled, count

    fill_target = (
        object_mask
        & (
            ~np.isfinite(depth)
            | (depth <= depth_min)
            | (depth >= depth_max)
        )
    )
    fill_pixels = fill_target & hull_valid
    count = int(np.count_nonzero(fill_pixels))
    if count == 0:
        return depth, 0
    filled = depth.copy()
    filled[fill_pixels] = hull_depth[fill_pixels]
    return filled, count


def _gpu_tsdf_impl(
    *,
    root: Path,
    stereo_model: str,
    tsdf_voxel: int,
    tsdf_sdf_trunc: float,
    tsdf_scale: float,
    tsdf_min_depth_baselines: int,
    tsdf_max_depth_baselines: int,
    tsdf_dilate: int,
    tsdf_cleaning_threshold: int,
    tsdf_use_mask: bool,
    tsdf_erode_mask: bool,
    tsdf_use_occlusion_mask: bool,
    tsdf_invert_mask: bool,
    tsdf_erosion_kernel_size: int,
    tsdf_closing_kernel_size: int,
    block_count: int,
    tsdf_device: str,
    mesh_target_faces: int,
    mesh_min_iou: float,
    mesh_iou_views: int,
    mesh_smooth_iters: int,
    mesh_smooth_lambda: float,
    mesh_smooth_mu: float,
    mesh_smooth_min_iou: float,
    mask_depth_mode: str | None,
    silhouette_voxel_resolution: int | None,
    silhouette_extra_views_path: str | None,
    progress_cb: Callable[[float, str], None] | None,
) -> str:
    """Core implementation — called from `gpu_tsdf_reconstruct` inside its
    try/finally so GPU memory is released even on early return/error.
    """
    import open3d as o3d
    import open3d.core as o3c
    from PIL import Image

    # ------------------------------------------------------------------
    # Load camera data written by gs2mesh Renderer
    # ------------------------------------------------------------------
    with open(root / "camera_data.json") as f:
        cameras = json.load(f)
    left_cameras = [c["left"] for c in cameras]
    baseline = float(left_cameras[0]["baseline"])

    voxel_length = tsdf_voxel / 512.0
    depth_max = baseline * tsdf_max_depth_baselines / tsdf_scale
    print(f"GPU TSDF: using device={tsdf_device}")

    # Select frames (dilate = take every N-th frame)
    n_cameras = len(left_cameras)
    valid_indices = [i for i in range(n_cameras) if i % tsdf_dilate == 0]
    n_valid = len(valid_indices)

    print(
        f"GPU TSDF: {n_valid} frames, voxel_length={voxel_length:.6f}, "
        f"sdf_trunc={tsdf_sdf_trunc}, baseline={baseline:.4f}, "
        f"depth_max={depth_max:.2f}"
    )
    min_depth_thresh = tsdf_min_depth_baselines * baseline

    silhouette_runtime: _SilhouetteFillRuntime | None = None
    silhouette_config = _resolve_silhouette_fill_config(
        mask_depth_mode=mask_depth_mode,
        default_voxel_resolution=silhouette_voxel_resolution,
    )
    if silhouette_config.enabled and tsdf_use_mask and not tsdf_invert_mask:
        if progress_cb:
            progress_cb(0.02, "Building SAM2 primary visual hull")

        def _visual_hull_progress(local_pct: float, detail: str) -> None:
            if progress_cb:
                progress_cb(
                    0.02 + max(0.0, min(1.0, local_pct)) * 0.13,
                    detail,
                )

        try:
            hull = _build_visual_hull(
                root=root,
                left_cameras=left_cameras,
                depth_min=min_depth_thresh,
                depth_max=depth_max,
                tsdf_scale=tsdf_scale,
                config=silhouette_config,
                extra_views_path=silhouette_extra_views_path,
                progress_cb=_visual_hull_progress,
            )
            if hull is not None:
                if progress_cb:
                    progress_cb(0.15, "Preparing SAM2 primary visual hull scene")
                scene = _build_visual_hull_scene(hull)
                silhouette_runtime = _SilhouetteFillRuntime(
                    scene=scene,
                    hull=hull,
                )
                print(
                    "GPU TSDF: SAM2 visual hull ready "
                    f"(depth_mode={silhouette_config.depth_mode}, "
                    f"occupied={hull.occupied_count}/{hull.total_voxels}, "
                    f"voxel_size={hull.voxel_size:.6f}, "
                    f"mesh={len(hull.vertices)}v/{len(hull.triangles)}f)"
                )
                if progress_cb:
                    progress_cb(0.16, "SAM2 primary visual hull ready")
            elif silhouette_config.depth_mode == "replace":
                raise RuntimeError(
                    "SAM2 visual hull replace mode could not build a hull. "
                    "Check that left_mask.npy files exist and overlap across views."
                )
        except Exception as exc:
            if silhouette_config.depth_mode == "replace":
                raise RuntimeError(
                    "SAM2 visual hull replace mode failed before TSDF fusion"
                ) from exc
            print(f"GPU TSDF: silhouette fill disabled after hull error: {exc}")
    elif silhouette_config.enabled and tsdf_invert_mask:
        print("GPU TSDF: silhouette fill skipped with inverted TSDF masks")

    # ------------------------------------------------------------------
    # Per-frame depth/RGB preparation → TsdfView list
    # gs2mesh-specific I/O (mask, occlusion, visual-hull fill, baseline
    # thresholding) lives here in the adapter; the actual VBG fusion is
    # delegated to scripts.lito.tsdf_core.fuse_tsdf (Phase 1.5 refactor).
    # ------------------------------------------------------------------
    from scripts.lito.tsdf_core import TsdfFusionParams, TsdfView, fuse_tsdf

    fusion_start = 0.16 if silhouette_runtime is not None else 0.0
    fusion_span = 0.64 if silhouette_runtime is not None else 0.80

    views: list[TsdfView] = []
    skipped_count = 0
    for step_i, cam_idx in enumerate(valid_indices):
        cam = left_cameras[cam_idx]
        cam_dir = root / f"{cam_idx:03d}"

        if progress_cb:
            pct = step_i / max(n_valid, 1)
            progress_cb(
                fusion_start + pct * fusion_span * 0.5,
                f"Loading view {step_i + 1}/{n_valid}",
            )

        # --- Load RGB & depth ---
        image = np.array(Image.open(cam_dir / "left.png")).astype(np.uint8)
        depth = np.load(
            cam_dir / f"out_{stereo_model}" / "depth.npy",
        ).astype(np.float32)

        # --- Object mask (file may not exist when masking was skipped) ---
        fill_object_mask: np.ndarray | None = None
        if tsdf_use_mask:
            raw_mask = _load_left_mask(cam_dir)
            if raw_mask is not None:
                raw_mask = _resize_mask_to_shape(raw_mask, depth.shape[:2])
                object_mask = _prepare_tsdf_object_mask(
                    raw_mask,
                    invert=tsdf_invert_mask,
                    erode=tsdf_erode_mask,
                    erosion_kernel_size=tsdf_erosion_kernel_size,
                    closing_kernel_size=tsdf_closing_kernel_size,
                )
                depth = depth * object_mask
                if not tsdf_invert_mask:
                    fill_object_mask = raw_mask

        # --- Occlusion mask ---
        if tsdf_use_occlusion_mask:
            occ_path = cam_dir / f"out_{stereo_model}" / "occlusion_mask.npy"
            if occ_path.exists():
                occlusion_mask = np.load(str(occ_path)).astype(bool)
                depth = depth * occlusion_mask

        # --- Depth thresholding (baselines) ---
        depth = np.where(depth < min_depth_thresh, 0.0, depth)

        if silhouette_runtime is not None and fill_object_mask is not None:
            hull_depth = _render_visual_hull_depth(
                silhouette_runtime.scene,
                cam,
                depth.shape[:2],
                tsdf_scale=tsdf_scale,
                depth_min=min_depth_thresh,
                depth_max=depth_max,
            )
            depth, filled_count = _apply_silhouette_depth_fill(
                depth,
                fill_object_mask,
                hull_depth,
                depth_min=min_depth_thresh,
                depth_max=depth_max,
                mode=silhouette_config.depth_mode,
            )
            if filled_count > 0:
                silhouette_runtime.filled_pixels += filled_count
                silhouette_runtime.filled_frames += 1

        # Fast check: skip frame if no valid depth pixels remain.
        valid_depth_count = int(
            np.count_nonzero((depth > min_depth_thresh) & (depth < depth_max))
        )
        if valid_depth_count == 0:
            skipped_count += 1
            continue

        extrinsic = np.array(cam["extrinsic"], dtype=np.float64)
        extrinsic[:3, 3] /= tsdf_scale
        T_cw = np.linalg.inv(extrinsic)
        K = np.array(
            [
                [cam["fx"], 0.0, cam["cx"]],
                [0.0, cam["fy"], cam["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        views.append(TsdfView(rgb=image, depth=depth, K=K, T_cw=T_cw))

    if silhouette_runtime is not None:
        print(
            "GPU TSDF: SAM2 silhouette depth stats "
            f"(mode={silhouette_config.depth_mode}, "
            f"pixels={silhouette_runtime.filled_pixels}, "
            f"filled_frames={silhouette_runtime.filled_frames}, "
            f"hull_voxels={silhouette_runtime.hull.occupied_count})"
        )
    if skipped_count > 0:
        print(
            f"GPU TSDF: prepared {len(views)}/{n_valid} views "
            f"(skipped {skipped_count} with no valid depth)"
        )
    if not views:
        raise RuntimeError(
            f"GPU TSDF: all {n_valid} frames skipped — no valid depth "
            f"in range [{min_depth_thresh:.3f}, {depth_max:.2f}]. "
            f"Check stereo depth output and mask coverage."
        )

    # ------------------------------------------------------------------
    # Delegate VBG creation, integration, extraction, and cleaning to the
    # pure core. Smoothing and decimation remain below until Step 3.
    # ------------------------------------------------------------------
    fusion_progress_cb = None
    if progress_cb is not None:
        fusion_lo = fusion_start + fusion_span * 0.5
        fusion_hi = 0.92

        def fusion_progress_cb(local_pct: float, detail: str) -> None:
            mapped = fusion_lo + max(0.0, min(1.0, local_pct)) * (fusion_hi - fusion_lo)
            progress_cb(mapped, detail)

    fusion_params = TsdfFusionParams(
        voxel_size=float(voxel_length),
        sdf_trunc=float(tsdf_sdf_trunc),
        depth_min=float(min_depth_thresh),
        depth_max=float(depth_max),
        device=tsdf_device,
        block_count=int(block_count),
        cleaning_threshold=int(tsdf_cleaning_threshold),
        tsdf_scale=float(tsdf_scale),
        smooth_iters=int(mesh_smooth_iters),
        smooth_lambda=float(mesh_smooth_lambda),
        smooth_mu=float(mesh_smooth_mu),
        smooth_min_iou=float(mesh_smooth_min_iou),
        target_faces=int(mesh_target_faces),
        decimate_min_iou=float(mesh_min_iou),
        iou_views=int(mesh_iou_views),
    )
    mesh = fuse_tsdf(views, fusion_params, progress_cb=fusion_progress_cb)
    if mesh is None:
        raise RuntimeError(
            f"GPU TSDF: integration produced no output from {len(views)} views"
        )

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(
        f"GPU TSDF: final mesh — {n_verts} vertices, {n_tris} triangles"
    )

    # ------------------------------------------------------------------
    # Save (smoothing + decimation + IoU rollback now happen inside
    # tsdf_core.fuse_tsdf, see Phase 1.5 Step 3)
    # ------------------------------------------------------------------
    cleaned_path = root / "gpu_tsdf_cleaned_mesh.ply"
    o3d.io.write_triangle_mesh(str(cleaned_path), mesh)
    print(f"GPU TSDF: saved cleaned mesh → {cleaned_path}")

    if progress_cb:
        progress_cb(1.0, "GPU TSDF fusion complete")

    return str(cleaned_path)


