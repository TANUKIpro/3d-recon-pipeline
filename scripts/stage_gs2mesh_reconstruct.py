"""Stage 4: gs2mesh Reconstruction.

Runs gs2mesh pipeline: 3DGS training -> stereo depth -> GPU TSDF fusion -> mesh.
Takes COLMAP output + optional SAM2 masks, produces object_mesh.ply.
"""

from __future__ import annotations

import importlib
import gc
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from scripts.config_defaults import (
    GS2MESH_MASK_DEPTH_MODE,
    GS2MESH_MASK_DEPTH_MODES,
    GS2MESH_RUNTIME_PROFILE,
    GS2MESH_RUNTIME_PROFILES,
    GS2MESH_SAM2_PRIMARY_MAX_EXTRA_VIEWS,
)
from scripts.gs2mesh_config import Gs2meshSettings
from scripts.output_layout import (
    camera_poses_path,
    frame_manifest_path,
    gs2mesh_workspace_dir,
    intrinsics_path,
    mesh_phase_dir,
    object_mesh_path,
    sam2_clicks_path,
)

_GS2MESH_BASE = Path("/opt/gs2mesh")
_GAUSSIAN_SPLATTING_ACCEL = Path("/opt/gaussian-splatting")
# The compat retry reuses the same gaussian-splatting source tree and only
# overrides the rasterizer extension from a separate site-packages overlay.
_GAUSSIAN_SPLATTING_COMPAT = _GAUSSIAN_SPLATTING_ACCEL
_GS2MESH_GS_LINK = _GS2MESH_BASE / "third_party" / "gaussian-splatting"
_GS_COMPAT_SITE_PACKAGES = Path("/opt/gs-compat-site")
_SAM2_PRIMARY_DENSE_FRAMES_DIRNAME = "sam2_primary_dense_frames"
_SAM2_PRIMARY_DENSE_MASKS_DIRNAME = "sam2_primary_dense_masks"
_SAM2_PRIMARY_DENSE_OBJECT_RAW_DIRNAME = "sam2_primary_dense_masks_object_raw"
_SAM2_PRIMARY_DENSE_GROUND_DIRNAME = "sam2_primary_dense_masks_ground"
_SAM2_PRIMARY_DENSE_VIEWS_FILENAME = "sam2_primary_dense_views.json"
_RETRYABLE_GS2MESH_FAILURES = frozenset(
    {
        "cuda_illegal_memory_access",
        "cuda_rasterizer_failure",
        "renderer_signature_mismatch",
        "missing_rasterizer_extension",
        "missing_output_artifacts",
    }
)


@dataclass(frozen=True)
class _Gs2meshRuntimeStack:
    name: str
    python_executable: str
    gaussian_splatting_root: Path
    env_overrides: dict[str, str]
    extra_pythonpaths: tuple[str, ...] = ()


class _SubprocessFailure(RuntimeError):
    """Structured subprocess failure with captured output for classification."""

    def __init__(
        self,
        message: str,
        *,
        args: list[str],
        exit_code: int,
        output_text: str,
    ) -> None:
        super().__init__(message)
        self.args_run = list(args)
        self.exit_code = exit_code
        self.output_text = output_text


def _is_gpu_tsdf_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        and ("cuda" in text or "open3d" in text)
    )


def _normalize_mask_depth_mode(value: object, fallback: str = GS2MESH_MASK_DEPTH_MODE) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in GS2MESH_MASK_DEPTH_MODES:
        return candidate
    if candidate in {"visual_hull", "visual-hull", "override"}:
        return "replace"
    if candidate in {"holes", "hole_fill", "hole-fill"}:
        return "fill"
    return fallback


def _resolve_mask_depth_mode_env(fallback: str = GS2MESH_MASK_DEPTH_MODE) -> str:
    raw = os.environ.get("GS2MESH_MASK_DEPTH_MODE", "").strip()
    if raw:
        return _normalize_mask_depth_mode(raw, fallback)
    legacy_raw = os.environ.get("GS2MESH_SILHOUETTE_DEPTH_MODE", "").strip()
    if legacy_raw:
        return _normalize_mask_depth_mode(legacy_raw, fallback)
    legacy_fill = os.environ.get("GS2MESH_SILHOUETTE_FILL", "").strip().lower()
    if legacy_fill in {"1", "true", "yes", "on"}:
        return "fill"
    return fallback


def _build_gpu_tsdf_attempts(
    settings: Gs2meshSettings,
) -> list[tuple[str, Gs2meshSettings]]:
    attempts: list[tuple[str, Gs2meshSettings]] = [("requested", settings)]

    reduced_vram = replace(
        settings,
        tsdf_voxel_size=max(settings.tsdf_voxel_size, 0.005),
        tsdf_depth_trunc=max(settings.tsdf_depth_trunc, 0.04),
        tsdf_max_depth_baselines=min(settings.tsdf_max_depth_baselines, 20),
        tsdf_dilate=max(settings.tsdf_dilate, 2),
        block_count=min(settings.block_count, 100_000),
    )
    if reduced_vram != settings:
        attempts.append(("reduced_vram", reduced_vram))

    safe_vram = replace(
        reduced_vram,
        tsdf_max_depth_baselines=min(reduced_vram.tsdf_max_depth_baselines, 16),
        tsdf_dilate=max(reduced_vram.tsdf_dilate, 3),
        block_count=min(reduced_vram.block_count, 75_000),
    )
    if safe_vram != reduced_vram:
        attempts.append(("safe_vram", safe_vram))

    return attempts


def _resolve_mesh_decimation_params(
    settings: Gs2meshSettings,
) -> tuple[int, float, int]:
    """Pick decimation target, IoU gate threshold, and view count.

    Target is derived from VRAM tier × inferred preset; env vars
    ``MESH_DECIMATION``, ``MESH_TARGET_FACES``, ``MESH_DECIMATION_MIN_IOU``,
    ``MESH_DECIMATION_IOU_VIEWS`` override.
    """

    from scripts.config_defaults import (
        MESH_DECIMATION_IOU_VIEWS,
        MESH_DECIMATION_MIN_IOU,
    )
    from scripts.gs2mesh_config import fields_match_preset
    from scripts.vram_tier import mesh_decimation_target

    config_fields = settings.to_config_fields()
    if fields_match_preset(config_fields, "high"):
        preset = "high"
    else:
        preset = "default"
    target = mesh_decimation_target(preset=preset)

    raw_iou = os.environ.get("MESH_DECIMATION_MIN_IOU", "").strip()
    try:
        min_iou = float(raw_iou) if raw_iou else MESH_DECIMATION_MIN_IOU
    except ValueError:
        min_iou = MESH_DECIMATION_MIN_IOU

    raw_views = os.environ.get("MESH_DECIMATION_IOU_VIEWS", "").strip()
    try:
        n_views = int(raw_views) if raw_views else MESH_DECIMATION_IOU_VIEWS
    except ValueError:
        n_views = MESH_DECIMATION_IOU_VIEWS

    return target, min_iou, max(0, n_views)


def _resolve_mesh_smoothing_params() -> tuple[int, float, float, float]:
    """Resolve Taubin smoothing iters / λ / μ / IoU gate from config + env.

    ``MESH_SMOOTH_ITERS=0`` (env) disables the smoothing pass; the rollback
    gate threshold can be loosened with ``MESH_SMOOTH_MIN_IOU`` for objects
    where the noise reduction is worth a slight silhouette change.
    """
    from scripts.config_defaults import (
        MESH_SMOOTH_ENABLED,
        MESH_SMOOTH_ITERS,
        MESH_SMOOTH_LAMBDA,
        MESH_SMOOTH_MIN_IOU,
        MESH_SMOOTH_MU,
    )

    if not MESH_SMOOTH_ENABLED:
        default_iters = 0
    else:
        default_iters = MESH_SMOOTH_ITERS

    raw_iters = os.environ.get("MESH_SMOOTH_ITERS", "").strip()
    try:
        iters = int(raw_iters) if raw_iters else default_iters
    except ValueError:
        iters = default_iters

    raw_lambda = os.environ.get("MESH_SMOOTH_LAMBDA", "").strip()
    try:
        lambda_val = float(raw_lambda) if raw_lambda else MESH_SMOOTH_LAMBDA
    except ValueError:
        lambda_val = MESH_SMOOTH_LAMBDA

    raw_mu = os.environ.get("MESH_SMOOTH_MU", "").strip()
    try:
        mu_val = float(raw_mu) if raw_mu else MESH_SMOOTH_MU
    except ValueError:
        mu_val = MESH_SMOOTH_MU

    raw_iou = os.environ.get("MESH_SMOOTH_MIN_IOU", "").strip()
    try:
        min_iou = float(raw_iou) if raw_iou else MESH_SMOOTH_MIN_IOU
    except ValueError:
        min_iou = MESH_SMOOTH_MIN_IOU

    return max(0, iters), lambda_val, mu_val, min_iou


def _run_gpu_tsdf_with_fallbacks(
    *,
    settings: Gs2meshSettings,
    gs2mesh_output: Path,
    stereo_model: str,
    silhouette_extra_views_path: Path | None,
    report,
) -> str:
    from scripts.gpu_tsdf import gpu_tsdf_reconstruct
    from scripts.vram_utils import cleanup_pytorch_vram

    mesh_target_faces, mesh_min_iou, mesh_iou_views = (
        _resolve_mesh_decimation_params(settings)
    )
    if mesh_target_faces > 0:
        print(
            f"GPU TSDF: mesh decimation target={mesh_target_faces:,} faces "
            f"(min_iou={mesh_min_iou:.3f}, n_views={mesh_iou_views})"
        )
    else:
        print("GPU TSDF: mesh decimation disabled")

    (
        mesh_smooth_iters,
        mesh_smooth_lambda,
        mesh_smooth_mu,
        mesh_smooth_min_iou,
    ) = _resolve_mesh_smoothing_params()
    if mesh_smooth_iters > 0:
        print(
            f"GPU TSDF: mesh smoothing iters={mesh_smooth_iters} "
            f"(λ={mesh_smooth_lambda:.2f}, μ={mesh_smooth_mu:.2f}, "
            f"min_iou={mesh_smooth_min_iou:.3f})"
        )
    else:
        print("GPU TSDF: mesh smoothing disabled")

    from scripts.config_defaults import (
        GS2MESH_SILHOUETTE_VOXELS,
        GS2MESH_SILHOUETTE_VOXELS_HIGH,
    )
    from scripts.gs2mesh_config import fields_match_preset

    silhouette_voxels = (
        GS2MESH_SILHOUETTE_VOXELS_HIGH
        if fields_match_preset(
            settings.to_config_fields(),
            "high",
            public_only=True,
        )
        else GS2MESH_SILHOUETTE_VOXELS
    )

    tsdf_device = settings.tsdf_device
    # CPU TSDF has no VRAM OOM to recover from — skip the
    # VRAM-reduction fallback ladder entirely so errors surface cleanly.
    if tsdf_device.lower().startswith("cpu"):
        attempts = [("requested", settings)]
    else:
        attempts = _build_gpu_tsdf_attempts(settings)
    last_exc: RuntimeError | None = None

    for idx, (label, attempt_settings) in enumerate(attempts, start=1):
        tsdf_voxel = max(1, round(attempt_settings.tsdf_voxel_size * 512))
        if idx > 1:
            print(
                "Retrying GPU TSDF with reduced VRAM settings: "
                f"attempt={label} voxel={tsdf_voxel}/512 "
                f"block_count={attempt_settings.block_count} "
                f"depth_baselines<={attempt_settings.tsdf_max_depth_baselines} "
                f"dilate={attempt_settings.tsdf_dilate}"
            )
            report(80.0, f"GPU TSDF fusion retry ({label})")
        try:
            return gpu_tsdf_reconstruct(
                output_dir_root=str(gs2mesh_output),
                stereo_model=stereo_model,
                tsdf_voxel=tsdf_voxel,
                tsdf_sdf_trunc=attempt_settings.tsdf_depth_trunc,
                tsdf_scale=attempt_settings.tsdf_scale,
                tsdf_min_depth_baselines=attempt_settings.tsdf_min_depth_baselines,
                tsdf_max_depth_baselines=attempt_settings.tsdf_max_depth_baselines,
                tsdf_dilate=attempt_settings.tsdf_dilate,
                tsdf_cleaning_threshold=attempt_settings.tsdf_cleaning_threshold,
                tsdf_use_mask=attempt_settings.use_masks,
                tsdf_erode_mask=attempt_settings.tsdf_erode_mask,
                tsdf_use_occlusion_mask=attempt_settings.tsdf_use_occlusion_mask,
                tsdf_invert_mask=attempt_settings.tsdf_invert_mask,
                tsdf_erosion_kernel_size=attempt_settings.tsdf_erosion_kernel_size,
                tsdf_closing_kernel_size=attempt_settings.tsdf_closing_kernel_size,
                block_count=attempt_settings.block_count,
                tsdf_device=tsdf_device,
                mesh_target_faces=mesh_target_faces,
                mesh_min_iou=mesh_min_iou,
                mesh_iou_views=mesh_iou_views,
                mesh_smooth_iters=mesh_smooth_iters,
                mesh_smooth_lambda=mesh_smooth_lambda,
                mesh_smooth_mu=mesh_smooth_mu,
                mesh_smooth_min_iou=mesh_smooth_min_iou,
                mask_depth_mode=attempt_settings.mask_depth_mode,
                silhouette_voxel_resolution=silhouette_voxels,
                silhouette_extra_views_path=(
                    None
                    if silhouette_extra_views_path is None
                    else str(silhouette_extra_views_path)
                ),
                progress_cb=lambda pct, msg: report(80.0 + pct * 15.0, msg),
            )
        except RuntimeError as exc:
            last_exc = exc
            if idx >= len(attempts) or not _is_gpu_tsdf_oom_error(exc):
                raise
            print(f"GPU TSDF OOM detected on attempt={label}: {exc}")
            cleanup_pytorch_vram()
            gc.collect()

    assert last_exc is not None
    raise RuntimeError(
        "GPU TSDF fusion failed after VRAM-reduction retries"
    ) from last_exc


def _recompute_sam2_masks_from_clicks(
    *,
    frames_dir: str,
    output_dir: str,
    progress_callback=None,
    **kwargs,
) -> tuple[Path, Path | None] | None:
    from scripts.stage_sam2_ui import recompute_masks_from_saved_clicks

    return recompute_masks_from_saved_clicks(
        frames_dir,
        output_dir,
        progress_callback=progress_callback,
        **kwargs,
    )


def _load_json_dict(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _load_frame_manifest(output_dir: str | Path) -> dict | None:
    return _load_json_dict(frame_manifest_path(output_dir))


def _manifest_saved_frames(manifest: dict) -> list[dict]:
    records = manifest.get("saved_frames", [])
    return [record for record in records if isinstance(record, dict)]


def _saved_source_lookup(manifest: dict) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for record in _manifest_saved_frames(manifest):
        try:
            saved_idx = int(record["saved_index"])
            source_idx = int(record["source_frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        lookup[saved_idx] = source_idx
    return lookup


def _saved_index_by_source(manifest: dict) -> dict[int, int]:
    return {
        source_idx: saved_idx
        for saved_idx, source_idx in _saved_source_lookup(manifest).items()
    }


def _dense_source_indices(manifest: dict) -> list[int]:
    sources = sorted(_saved_index_by_source(manifest))
    if not sources:
        return []
    return list(range(sources[0], sources[-1] + 1))


def _env_int_min(name: str, default: int, *, min_value: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


def _evenly_sample_ints(values: list[int], max_count: int) -> list[int]:
    if max_count <= 0:
        return []
    if len(values) <= max_count:
        return list(values)
    if max_count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    selected: list[int] = []
    seen: set[int] = set()
    for i in range(max_count):
        src_pos = round(i * last / (max_count - 1))
        value = values[int(src_pos)]
        if value not in seen:
            selected.append(value)
            seen.add(value)
    return selected


def _extract_dense_video_frames(
    *,
    manifest: dict,
    dense_frames_dir: Path,
    source_indices: list[int] | None = None,
) -> list[dict]:
    import cv2
    from scripts.stage_extract_frames import _rotation_to_cv2_code

    selected_sources = (
        _dense_source_indices(manifest)
        if source_indices is None
        else sorted({int(idx) for idx in source_indices})
    )
    if not selected_sources:
        return []

    video_path = Path(str(manifest.get("video_path") or ""))
    if not video_path.is_file():
        print(
            "SAM2 primary: dense skipped-frame masks skipped "
            f"(video not found: {video_path})"
        )
        return []

    if dense_frames_dir.exists():
        shutil.rmtree(dense_frames_dir)
    dense_frames_dir.mkdir(parents=True, exist_ok=True)

    saved_by_source = _saved_index_by_source(manifest)
    rotate_code = _rotation_to_cv2_code(int(manifest.get("rotation") or 0))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"SAM2 primary: could not open source video: {video_path}")
        return []
    try:
        records: list[dict] = []
        for current_source in selected_sources:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_source)
            ret, frame = cap.read()
            if not ret:
                continue
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            dense_idx = len(records)
            frame_name = f"{dense_idx:05d}.jpg"
            frame_path = dense_frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_idx = saved_by_source.get(current_source)
            records.append(
                {
                    "dense_index": dense_idx,
                    "dense_name": frame_name,
                    "source_frame_index": current_source,
                    "is_extracted": saved_idx is not None,
                    "saved_index": saved_idx,
                }
            )
    finally:
        cap.release()
    return records


def _matrix4(value: object) -> "np.ndarray | None":
    import numpy as np

    if not isinstance(value, list):
        return None
    matrix = np.asarray(value, dtype=np.float64)
    return matrix if matrix.shape == (4, 4) else None


def _load_pose_keyframes(
    output_dir: str | Path,
    manifest: dict,
) -> list[tuple[int, "np.ndarray"]]:
    import numpy as np

    pose_path = camera_poses_path(output_dir)
    if not pose_path.is_file():
        return []
    with pose_path.open("r", encoding="utf-8") as f:
        pose_data = json.load(f)

    saved_to_source = _saved_source_lookup(manifest)
    keyframes: list[tuple[int, np.ndarray]] = []
    if isinstance(pose_data, dict) and isinstance(pose_data.get("poses"), list):
        for saved_idx, matrix_like in enumerate(pose_data["poses"]):
            source_idx = saved_to_source.get(saved_idx)
            matrix = _matrix4(matrix_like)
            if source_idx is not None and matrix is not None:
                keyframes.append((source_idx, matrix))
    elif isinstance(pose_data, list):
        for default_idx, record in enumerate(pose_data):
            if not isinstance(record, dict):
                continue
            frame_idx = None
            frame_name = record.get("frame_name")
            if isinstance(frame_name, str):
                frame_idx = _extract_frame_index_from_text(frame_name)
            if frame_idx is None:
                frame_idx = default_idx
            source_idx = saved_to_source.get(frame_idx)
            matrix = _matrix4(record.get("transform_matrix"))
            if source_idx is not None and matrix is not None:
                keyframes.append((source_idx, matrix))

    by_source: dict[int, np.ndarray] = {}
    for source_idx, matrix in keyframes:
        by_source.setdefault(source_idx, matrix)
    return sorted(by_source.items(), key=lambda item: item[0])


def _interpolate_pose(
    keyframes: list[tuple[int, "np.ndarray"]],
    source_frame_index: int,
) -> "np.ndarray | None":
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    if not keyframes:
        return None
    source_frame_index = int(source_frame_index)
    for key_source, matrix in keyframes:
        if source_frame_index == key_source:
            return matrix.copy()
    for (lo_source, lo_pose), (hi_source, hi_pose) in zip(keyframes, keyframes[1:]):
        if lo_source < source_frame_index < hi_source:
            ratio = (
                float(source_frame_index - lo_source)
                / float(max(1, hi_source - lo_source))
            )
            rotations = Rotation.from_matrix(
                np.stack([lo_pose[:3, :3], hi_pose[:3, :3]])
            )
            interp_rot = Slerp([0.0, 1.0], rotations)([ratio]).as_matrix()[0]
            interp_t = (
                lo_pose[:3, 3] * (1.0 - ratio)
                + hi_pose[:3, 3] * ratio
            )
            result = np.eye(4, dtype=np.float64)
            result[:3, :3] = interp_rot
            result[:3, 3] = interp_t
            return result
    return None


def _load_dense_intrinsics(output_dir: str | Path) -> dict | None:
    data = _load_json_dict(intrinsics_path(output_dir))
    if data is None:
        return None
    required = ("fx", "fy", "cx", "cy")
    try:
        return {key: float(data[key]) for key in required} | {
            "image_width": float(data.get("image_width") or 0.0),
            "image_height": float(data.get("image_height") or 0.0),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _scale_intrinsics_for_shape(
    intrinsics: dict,
    image_shape: tuple[int, int],
) -> dict[str, float]:
    h, w = image_shape
    base_w = float(intrinsics.get("image_width") or w)
    base_h = float(intrinsics.get("image_height") or h)
    sx = float(w) / max(base_w, 1.0)
    sy = float(h) / max(base_h, 1.0)
    return {
        "fx": float(intrinsics["fx"]) * sx,
        "fy": float(intrinsics["fy"]) * sy,
        "cx": float(intrinsics["cx"]) * sx,
        "cy": float(intrinsics["cy"]) * sy,
    }


def _load_camera_baseline(gs2mesh_output: Path) -> float:
    camera_data_path = gs2mesh_output / "camera_data.json"
    try:
        with camera_data_path.open("r", encoding="utf-8") as f:
            camera_data = json.load(f)
        first_left = camera_data[0]["left"]
        return float(first_left.get("baseline") or 1.0)
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return 1.0


def _write_dense_extra_views(
    *,
    output_dir: str | Path,
    manifest: dict,
    dense_records: list[dict],
    dense_mask_dir: Path,
    gs2mesh_output: Path,
    extra_views_path: Path,
) -> Path | None:
    import cv2

    keyframes = _load_pose_keyframes(output_dir, manifest)
    if len(keyframes) < 2:
        print("SAM2 primary: dense skipped-frame views skipped (not enough poses)")
        return None
    intrinsics = _load_dense_intrinsics(output_dir)
    if intrinsics is None:
        print("SAM2 primary: dense skipped-frame views skipped (no intrinsics)")
        return None

    baseline = _load_camera_baseline(gs2mesh_output)
    views: list[dict] = []
    for record in dense_records:
        if bool(record.get("is_extracted")):
            continue
        try:
            dense_idx = int(record["dense_index"])
            source_idx = int(record["source_frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        pose = _interpolate_pose(keyframes, source_idx)
        if pose is None:
            continue
        mask_path = dense_mask_dir / f"{dense_idx:05d}.png"
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None or not bool((mask_img > 0).any()):
            continue
        scaled = _scale_intrinsics_for_shape(intrinsics, mask_img.shape[:2])
        views.append(
            {
                "source_frame_index": source_idx,
                "dense_frame_index": dense_idx,
                "mask_path": str(mask_path),
                "camera": {
                    **scaled,
                    "baseline": baseline,
                    "extrinsic": pose.tolist(),
                },
            }
        )

    if not views:
        print("SAM2 primary: dense skipped-frame views skipped (no usable masks)")
        return None

    extra_views_path.parent.mkdir(parents=True, exist_ok=True)
    extra_views_path.write_text(
        json.dumps({"version": 1, "views": views}, indent=2),
        encoding="utf-8",
    )
    print(f"SAM2 primary: added {len(views)} dense skipped-frame visual-hull views")
    return extra_views_path


def _prepare_sam2_primary_dense_extra_views(
    *,
    frames_dir: str,
    output_dir: str,
    gs2mesh_workdir: Path,
    gs2mesh_output: Path,
    report,
) -> Path | None:
    del frames_dir  # dense frames are decoded from the original video manifest.
    manifest = _load_frame_manifest(output_dir)
    if manifest is None:
        print("SAM2 primary: frame manifest not found; dense skipped frames disabled")
        return None

    saved_by_source = _saved_index_by_source(manifest)
    dense_sources = _dense_source_indices(manifest)
    skipped_sources = [idx for idx in dense_sources if idx not in saved_by_source]
    if not skipped_sources:
        print("SAM2 primary: no frameextract-skipped frames in manifest range")
        return None

    keyframes = _load_pose_keyframes(output_dir, manifest)
    if len(keyframes) < 2:
        print("SAM2 primary: dense skipped-frame views disabled (not enough poses)")
        return None
    pose_min = keyframes[0][0]
    pose_max = keyframes[-1][0]
    eligible_skipped_sources = [
        source_idx
        for source_idx in skipped_sources
        if pose_min < source_idx < pose_max
    ]
    if not eligible_skipped_sources:
        print("SAM2 primary: dense skipped-frame views disabled (outside pose range)")
        return None

    if _load_dense_intrinsics(output_dir) is None:
        print("SAM2 primary: dense skipped-frame views disabled (missing intrinsics)")
        return None

    max_extra_views = _env_int_min(
        "GS2MESH_SAM2_PRIMARY_MAX_EXTRA_VIEWS",
        GS2MESH_SAM2_PRIMARY_MAX_EXTRA_VIEWS,
        min_value=0,
    )
    selected_skipped_sources = _evenly_sample_ints(
        eligible_skipped_sources,
        max_extra_views,
    )
    if not selected_skipped_sources:
        print("SAM2 primary: dense skipped-frame views disabled (max extra views is 0)")
        return None
    if len(selected_skipped_sources) < len(eligible_skipped_sources):
        print(
            "SAM2 primary: capped dense skipped-frame visual-hull views "
            f"to {len(selected_skipped_sources)}/{len(eligible_skipped_sources)}"
        )
    else:
        print(
            "SAM2 primary: using "
            f"{len(selected_skipped_sources)} dense skipped-frame visual-hull views"
        )
    selected_sources = sorted(
        set(saved_by_source) | set(selected_skipped_sources)
    )

    dense_frames_dir = gs2mesh_workdir / _SAM2_PRIMARY_DENSE_FRAMES_DIRNAME
    dense_mask_dir = gs2mesh_workdir / _SAM2_PRIMARY_DENSE_MASKS_DIRNAME
    dense_object_raw_dir = gs2mesh_workdir / _SAM2_PRIMARY_DENSE_OBJECT_RAW_DIRNAME
    dense_ground_dir = gs2mesh_workdir / _SAM2_PRIMARY_DENSE_GROUND_DIRNAME
    for path in (dense_mask_dir, dense_object_raw_dir, dense_ground_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    report(77.0, "Extracting dense skipped frames for SAM2 primary")
    dense_records = _extract_dense_video_frames(
        manifest=manifest,
        dense_frames_dir=dense_frames_dir,
        source_indices=selected_sources,
    )
    if not dense_records:
        return None

    report(78.0, "Propagating dense SAM2 primary masks")

    def _progress(frame_idx: int, total: int) -> None:
        total = max(total, 1)
        ratio = (frame_idx + 1) / total
        report(
            78.0 + min(ratio, 1.0),
            f"Propagating dense SAM2 primary masks ({frame_idx + 1}/{total})",
        )

    result = _recompute_sam2_masks_from_clicks(
        frames_dir=str(dense_frames_dir),
        output_dir=output_dir,
        click_output_dir=output_dir,
        mask_dir=dense_mask_dir,
        object_mask_raw_dir=dense_object_raw_dir,
        ground_mask_dir=dense_ground_dir,
        save_clicks=False,
        progress_callback=_progress,
    )
    if result is None:
        return None

    extra_views_path = gs2mesh_workdir / _SAM2_PRIMARY_DENSE_VIEWS_FILENAME
    return _write_dense_extra_views(
        output_dir=output_dir,
        manifest=manifest,
        dense_records=dense_records,
        dense_mask_dir=dense_mask_dir,
        gs2mesh_output=gs2mesh_output,
        extra_views_path=extra_views_path,
    )


def _maybe_refresh_sam2_primary_masks(
    *,
    frames_dir: str,
    output_dir: str,
    mask_dir: str,
    gs2mesh_workdir: Path,
    gs2mesh_output: Path,
    settings: Gs2meshSettings,
    report,
) -> tuple[str, Path | None]:
    """Recompute saved SAM2 masks only for SAM2-primary geometry mode."""
    if settings.mask_depth_mode != "replace":
        return mask_dir, None

    click_path = sam2_clicks_path(output_dir)
    if not click_path.is_file():
        print(
            "SAM2 primary: saved click points not found; "
            f"using existing masks at {mask_dir}"
        )
        return mask_dir, None

    print(f"SAM2 primary: recomputing masks from saved clicks: {click_path}")
    report(78.0, "Recomputing SAM2 primary masks")

    def _progress(frame_idx: int, total: int) -> None:
        total = max(total, 1)
        ratio = (frame_idx + 1) / total
        report(
            78.0 + min(ratio, 1.0),
            f"Recomputing SAM2 primary masks ({frame_idx + 1}/{total})",
        )

    result = _recompute_sam2_masks_from_clicks(
        frames_dir=frames_dir,
        output_dir=output_dir,
        progress_callback=_progress,
    )
    refreshed_mask_dir = Path(mask_dir) if result is None else result[0]

    extra_views_path = _prepare_sam2_primary_dense_extra_views(
        frames_dir=frames_dir,
        output_dir=output_dir,
        gs2mesh_workdir=gs2mesh_workdir,
        gs2mesh_output=gs2mesh_output,
        report=report,
    )
    return str(refreshed_mask_dir), extra_views_path


def run_gs2mesh(
    frames_dir: str,
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: str,
    settings: Gs2meshSettings | None = None,
    gs_iterations: int = 5000,
    runtime_profile: str = GS2MESH_RUNTIME_PROFILE,
    stereo_model: str = "DLNR_Middlebury",
    tsdf_voxel_size: float = 0.005,
    tsdf_depth_trunc: float = 0.04,
    use_masks: bool = True,
    mask_depth_mode: str | None = None,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    """gs2mesh pipeline: 3DGS training -> stereo depth -> GPU TSDF -> mesh.

    Returns path to object_mesh.ply.
    """
    out = Path(output_dir)
    mesh_phase_dir(out).mkdir(parents=True, exist_ok=True)
    gs2mesh_workdir = gs2mesh_workspace_dir(out)
    debug_dir = gs2mesh_workdir / "debug"
    gs2mesh_workdir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = Gs2meshSettings.from_preset()
        settings = Gs2meshSettings(
            **{
                **settings.__dict__,
                "gs_iterations": gs_iterations,
                "runtime_profile": runtime_profile,
                "stereo_model": stereo_model,
                "tsdf_voxel_size": tsdf_voxel_size,
                "tsdf_depth_trunc": tsdf_depth_trunc,
                "use_masks": use_masks,
                "mask_depth_mode": (
                    _normalize_mask_depth_mode(mask_depth_mode)
                    if mask_depth_mode is not None
                    else _resolve_mask_depth_mode_env()
                ),
            }
        )

    # Apply the hardware-tier override (CPU TSDF on low-VRAM hosts).
    # The dashboard wrapper also calls this, so this is the single
    # safety net for CLI invocations that construct their own settings.
    from scripts.vram_tier import apply_tier_override

    settings = apply_tier_override(settings)
    gs_iterations = settings.gs_iterations
    runtime_profile = settings.runtime_profile
    stereo_model = settings.stereo_model
    tsdf_voxel_size = settings.tsdf_voxel_size
    tsdf_depth_trunc = settings.tsdf_depth_trunc
    use_masks = settings.use_masks

    # Map shortened stereo model names for backward compatibility
    if stereo_model == "DLNR":
        stereo_model = "DLNR_Middlebury"

    def _report(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if cancel_cb:
            cancel_cb()

    resolved_profile = _normalize_runtime_profile(runtime_profile)
    training_stack = _resolve_training_runtime_stack(resolved_profile)

    # Find the COLMAP reconstruction subdir (0/, 1/, etc.)
    recon_dir = _find_recon_dir(colmap_sparse_dir)

    # Step 0a: Undistort images. 3DGS requires PINHOLE cameras, but
    # COLMAP defaults to SIMPLE_RADIAL.
    _report(1.0, "Undistorting images for 3DGS")
    undistorted_dir = gs2mesh_workdir / "undistorted"
    _run_colmap_cmd(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(frames_dir),
            "--input_path",
            str(recon_dir),
            "--output_path",
            str(undistorted_dir),
            "--output_type",
            "COLMAP",
        ],
        "image_undistorter",
        undistorted_dir,
        register_process,
        unregister_process,
    )
    _ensure_sparse_0(undistorted_dir / "sparse")

    # Step 0b: Convert binary COLMAP model -> text format.
    # gs2mesh's Renderer reads images.txt / cameras.txt, not binary files.
    _report(2.0, "Converting COLMAP model to text format")
    undist_model = undistorted_dir / "sparse" / "0"
    _run_colmap_cmd(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(undist_model),
            "--output_path",
            str(undist_model),
            "--output_type",
            "TXT",
        ],
        "model_converter (binary->text)",
        None,
        register_process,
        unregister_process,
    )

    # Step 0c: Set up gs2mesh data directory with symlinks to undistorted output.
    # gs2mesh expects: {base}/data/custom/{scene}/images/ + sparse/0/
    scene_name = out.name
    gs2mesh_data = _GS2MESH_BASE / "data" / "custom" / scene_name
    _setup_gs2mesh_dirs(gs2mesh_data, undistorted_dir)
    _ensure_gs2mesh_renderer_compat()
    _ensure_gaussian_renderer_compat(training_stack.gaussian_splatting_root)

    # Step 1: Train 3D Gaussian Splatting model.
    # We train separately for better progress tracking and error handling.
    _report(5.0, "Training 3D Gaussian Splatting model")
    splatting_string = f"custom_nw_iterations{gs_iterations}"
    gs_model_dir = (
        _GS2MESH_BASE / "splatting_output" / splatting_string / scene_name
    )
    persistent_gs_model_dir = (
        gs2mesh_workdir / "splatting_output" / splatting_string / scene_name
    )
    _ensure_gs_model_output_link(gs_model_dir, persistent_gs_model_dir)
    final_point_cloud = (
        persistent_gs_model_dir
        / "point_cloud"
        / f"iteration_{gs_iterations}"
        / "point_cloud.ply"
    )
    gs_train_args, resolved_profile, optimizer_type, fallback_reason = (
        _build_gs_train_args(
            gs2mesh_data,
            gs_model_dir,
            gs_iterations,
            resolved_profile,
            python_executable=training_stack.python_executable,
            gaussian_splatting_root=training_stack.gaussian_splatting_root,
        )
    )
    print(f"3DGS runtime profile: {resolved_profile}")
    print(f"3DGS runtime stack: {training_stack.name}")
    print(f"3DGS optimizer: {optimizer_type}")
    if fallback_reason:
        print(f"3DGS optimizer fallback: {fallback_reason}")

    if final_point_cloud.is_file():
        print(f"Reusing existing 3DGS checkpoint: {final_point_cloud}")
        _report(50.0, "Reusing existing 3D Gaussian Splatting checkpoint")
    else:
        from scripts.vram_utils import log_vram_detailed

        log_vram_detailed("before 3DGS training subprocess")
        _run_subprocess(
            gs_train_args,
            prefix="3DGS",
            progress_fn=lambda line: _parse_gs_progress(
                line, gs_iterations, _report,
            ),
            register_process=register_process,
            unregister_process=unregister_process,
            error_msg="3DGS training failed",
            env=_build_runtime_env(
                training_stack.gaussian_splatting_root,
                training_stack.env_overrides,
                extra_pythonpaths=training_stack.extra_pythonpaths,
            ),
        )

    if cancel_cb:
        cancel_cb()

    from scripts.vram_utils import cleanup_pytorch_vram, log_vram

    log_vram("after 3DGS training subprocess exit")
    cleanup_pytorch_vram()

    # Step 2: Run gs2mesh (rendering + stereo depth only; TSDF on GPU below).
    _report(55.0, "Running gs2mesh stereo depth estimation")
    tsdf_voxel = max(1, round(tsdf_voxel_size * 512))
    gs2mesh_output = _GS2MESH_BASE / "output" / "clip2mesh" / scene_name

    _run_gs2mesh_stereo_with_retries(
        scene_name=scene_name,
        stereo_model=stereo_model,
        gs_iterations=gs_iterations,
        tsdf_voxel=tsdf_voxel,
        tsdf_depth_trunc=tsdf_depth_trunc,
        runtime_profile=resolved_profile,
        gs2mesh_output=gs2mesh_output,
        debug_dir=debug_dir,
        report=_report,
        register_process=register_process,
        unregister_process=unregister_process,
    )

    if cancel_cb:
        cancel_cb()

    log_vram("after stereo depth subprocess exit")
    cleanup_pytorch_vram()

    silhouette_extra_views_path: Path | None = None
    if use_masks and mask_dir:
        mask_dir, silhouette_extra_views_path = _maybe_refresh_sam2_primary_masks(
            frames_dir=frames_dir,
            output_dir=str(out),
            mask_dir=mask_dir,
            gs2mesh_workdir=gs2mesh_workdir,
            gs2mesh_output=gs2mesh_output,
            settings=settings,
            report=_report,
        )
        _report(79.0, "Preparing SAM2 masks for TSDF fusion")
        _materialize_tsdf_masks(gs2mesh_output, mask_dir)

    # Step 3: GPU TSDF fusion (replaces gs2mesh CPU TSDF).
    _report(80.0, "GPU TSDF fusion")

    gpu_mesh_path = _run_gpu_tsdf_with_fallbacks(
        settings=settings,
        gs2mesh_output=gs2mesh_output,
        stereo_model=stereo_model,
        silhouette_extra_views_path=silhouette_extra_views_path,
        report=_report,
    )

    # Step 4: Copy output mesh
    _report(95.0, "Collecting output mesh")
    final_mesh = object_mesh_path(out)
    shutil.copy2(gpu_mesh_path, str(final_mesh))
    print(f"GPU TSDF output mesh: {final_mesh}")

    _report(100.0, "gs2mesh reconstruction complete")
    return str(final_mesh)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _find_recon_dir(colmap_sparse_dir: str) -> Path:
    """Find the first valid COLMAP reconstruction subdirectory."""
    colmap_sparse = Path(colmap_sparse_dir)
    for d in sorted(colmap_sparse.iterdir()):
        if d.is_dir() and (d / "images.bin").exists():
            return d
    raise RuntimeError(
        "No valid COLMAP reconstruction found in " + colmap_sparse_dir
    )


def _ensure_sparse_0(sparse_dir: Path) -> None:
    """Ensure sparse/ has a 0/ subdirectory (COLMAP convention).

    The undistorter puts files directly in sparse/; downstream tools
    expect sparse/0/.
    """
    if sparse_dir.is_dir() and not (sparse_dir / "0").is_dir():
        tmp = sparse_dir.parent / "_sparse_tmp"
        sparse_dir.rename(tmp)
        sparse_dir.mkdir()
        tmp.rename(sparse_dir / "0")


def _setup_gs2mesh_dirs(gs2mesh_data: Path, undistorted_dir: Path) -> None:
    """Create gs2mesh data directory with symlinks to undistorted output."""
    gs2mesh_data.mkdir(parents=True, exist_ok=True)
    for name in ("images", "sparse"):
        link = gs2mesh_data / name
        target = undistorted_dir / name
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(target.resolve())


def _ensure_gs_model_output_link(
    runtime_model_dir: Path,
    persistent_model_dir: Path,
) -> None:
    persistent_model_dir.mkdir(parents=True, exist_ok=True)
    runtime_model_dir.parent.mkdir(parents=True, exist_ok=True)

    if runtime_model_dir.is_symlink():
        if runtime_model_dir.resolve() == persistent_model_dir.resolve():
            return
        runtime_model_dir.unlink()
    elif runtime_model_dir.exists():
        shutil.rmtree(runtime_model_dir)

    runtime_model_dir.symlink_to(persistent_model_dir, target_is_directory=True)
    print(f"3DGS model output -> {persistent_model_dir}")


def _patch_gs2mesh_renderer_utils(renderer_utils_path: Path) -> bool:
    """Patch gs2mesh renderer for newer gaussian-splatting builds."""
    text = renderer_utils_path.read_text(encoding="utf-8")
    updated = text

    if "from PIL import Image" not in updated:
        anchor = "import copy\n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: copy import anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "from PIL import Image\n",
            1,
        )

    if "depths=''" not in updated:
        anchor = "                         images='images', \n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: images anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "                         depths='', \n",
            1,
        )

    if "train_test_exp=False" not in updated:
        anchor = "                         eval=False, \n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: eval anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "                         train_test_exp=False, \n",
            1,
        )

    if "antialiasing=False" not in updated:
        anchor = "                         debug=False, \n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: debug anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "                         antialiasing=False, \n",
            1,
        )

    old_camera_call = (
        '                view = cameras.Camera(0, R, T, FoVx, FoVy, '
        'torch.rand(3,h,w), None, "abcd", 0)\n'
    )
    new_camera_call = """                dummy_image = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
                view = cameras.Camera(
                    (w, h),
                    0,
                    R,
                    T,
                    FoVx,
                    FoVy,
                    None,
                    dummy_image,
                    None,
                    f"{camera_name}",
                    camera_number,
                    data_device=self.device,
                )
"""
    if old_camera_call in updated:
        updated = updated.replace(old_camera_call, new_camera_call, 1)

    if updated == text:
        return False

    renderer_utils_path.write_text(updated, encoding="utf-8")
    return True


def _patch_gaussian_renderer_init(renderer_init_path: Path) -> bool:
    """Patch gaussian_renderer to retry without unsupported rasterizer kwargs."""
    text = renderer_init_path.read_text(encoding="utf-8")
    updated = text

    helper_anchor = "from utils.sh_utils import eval_sh\n"
    raster_helper_body = """
def _build_raster_settings(**kwargs):
    try:
        return GaussianRasterizationSettings(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        retry_kwargs = dict(kwargs)
        for key in ("antialiasing", "train_test_exp"):
            if key in retry_kwargs and key in str(exc):
                retry_kwargs.pop(key, None)
                return _build_raster_settings(**retry_kwargs)
        raise

"""
    unpack_helper_body = """
def _unpack_rasterizer_output(output):
    if not isinstance(output, (tuple, list)):
        raise TypeError(
            "Unexpected rasterizer return type: "
            f"{type(output).__name__}"
        )
    if len(output) == 3:
        return output
    if len(output) == 2:
        rendered_image, radii = output
        return rendered_image, radii, None
    raise ValueError(
        "Unexpected rasterizer return arity: "
        f"{len(output)}"
    )

"""
    if "def _build_raster_settings(**kwargs):" not in updated:
        if helper_anchor not in updated:
            raise RuntimeError(
                "Could not patch gaussian_renderer: import anchor missing"
            )
        updated = updated.replace(
            helper_anchor,
            helper_anchor + raster_helper_body,
            1,
        )

    if "def _unpack_rasterizer_output(output):" not in updated:
        helper_insert_anchor = "def _build_raster_settings(**kwargs):"
        helper_insert_body = raster_helper_body + unpack_helper_body
        if helper_insert_body in updated:
            pass
        elif helper_insert_anchor in updated:
            updated = updated.replace(
                raster_helper_body,
                raster_helper_body + unpack_helper_body,
                1,
            )
        elif helper_anchor in updated:
            updated = updated.replace(
                helper_anchor,
                helper_anchor + unpack_helper_body,
                1,
            )
        else:
            raise RuntimeError(
                "Could not patch gaussian_renderer: helper insert anchor missing"
            )

    old_block = """    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
"""
    new_block = """    raster_settings_kwargs = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    if hasattr(pipe, "antialiasing"):
        raster_settings_kwargs["antialiasing"] = pipe.antialiasing
    if hasattr(pipe, "train_test_exp"):
        raster_settings_kwargs["train_test_exp"] = pipe.train_test_exp
    raster_settings = _build_raster_settings(**raster_settings_kwargs)
"""
    if old_block in updated:
        updated = updated.replace(old_block, new_block, 1)

    old_rasterizer_block = """    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
"""
    new_rasterizer_block = """    if separate_sh:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    rendered_image, radii, depth_image = _unpack_rasterizer_output(
        rasterizer_output
    )
"""
    if old_rasterizer_block in updated:
        updated = updated.replace(
            old_rasterizer_block,
            new_rasterizer_block,
            1,
        )

    if updated == text:
        return False

    renderer_init_path.write_text(updated, encoding="utf-8")
    return True


def _ensure_gs2mesh_renderer_compat() -> None:
    renderer_utils_path = _GS2MESH_BASE / "gs2mesh_utils" / "renderer_utils.py"
    if _patch_gs2mesh_renderer_utils(renderer_utils_path):
        print(
            "Patched gs2mesh renderer args for gaussian-splatting "
            "compatibility"
        )


def _ensure_gaussian_renderer_compat(
    gaussian_splatting_root: Path = _GAUSSIAN_SPLATTING_ACCEL,
) -> None:
    renderer_init_path = gaussian_splatting_root / "gaussian_renderer" / "__init__.py"
    if _patch_gaussian_renderer_init(renderer_init_path):
        print(
            "Patched gaussian_renderer to retry without unsupported "
            "rasterizer kwargs"
        )


def _normalize_runtime_profile(runtime_profile: str | None) -> str:
    candidate = str(runtime_profile or "").strip().lower()
    if candidate in GS2MESH_RUNTIME_PROFILES:
        return candidate
    return GS2MESH_RUNTIME_PROFILE


def _sparse_adam_available() -> bool:
    try:
        rasterizer = importlib.import_module("diff_gaussian_rasterization")
    except ImportError:
        return False
    return hasattr(rasterizer, "SparseGaussianAdam")


def _resolve_optimizer_type(
    runtime_profile: str,
) -> tuple[str, str | None]:
    if runtime_profile == "compat":
        return "default", None
    if _sparse_adam_available():
        return "sparse_adam", None
    return "default", "SparseGaussianAdam unavailable"


def _build_gs_train_args(
    source_path: Path,
    model_path: Path,
    iterations: int,
    runtime_profile: str,
    *,
    python_executable: str = "python3",
    gaussian_splatting_root: Path = _GAUSSIAN_SPLATTING_ACCEL,
) -> tuple[list[str], str, str, str | None]:
    resolved_profile = _normalize_runtime_profile(runtime_profile)
    optimizer_type, fallback_reason = _resolve_optimizer_type(
        resolved_profile
    )
    # train.py always appends the final iteration to save_iterations.
    # Use a sentinel just beyond the training horizon to suppress
    # intermediate saves while still keeping the final checkpoint.
    save_sentinel = max(iterations + 1, 1)
    args = [
        python_executable,
        "-u",
        str(gaussian_splatting_root / "train.py"),
        "--source_path",
        str(source_path),
        "--model_path",
        str(model_path),
        "--iterations",
        str(iterations),
        "--disable_viewer",
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(save_sentinel),
        "--optimizer_type",
        optimizer_type,
    ]
    return args, resolved_profile, optimizer_type, fallback_reason


def _run_colmap_cmd(
    cmd: list[str],
    step_name: str,
    output_dir: Path | None,
    register_process=None,
    unregister_process=None,
) -> None:
    """Run a COLMAP command, streaming output and checking exit code."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"COLMAP {step_name}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    if register_process:
        register_process(proc)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [COLMAP] {line}")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"COLMAP {step_name} failed (exit {proc.returncode})"
            )
    finally:
        if unregister_process:
            unregister_process(proc)


def _run_subprocess(
    args: list[str],
    *,
    cwd: str | None = None,
    prefix: str = "",
    progress_fn=None,
    register_process=None,
    unregister_process=None,
    error_msg: str = "subprocess failed",
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    capture_output: bool = False,
) -> str:
    """Run a subprocess, streaming prefixed output with optional capture."""
    output_lines: list[str] = []
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        env=env,
    )
    if register_process:
        register_process(proc)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(f"  [{prefix}] {line}")
            if log_file is not None:
                log_file.write(line + "\n")
                log_file.flush()
            if capture_output:
                output_lines.append(line)
            if progress_fn:
                progress_fn(line)
        proc.wait()
        output_text = "\n".join(output_lines)
        if proc.returncode != 0:
            raise _SubprocessFailure(
                f"{error_msg} (exit {proc.returncode})",
                args=args,
                exit_code=proc.returncode,
                output_text=output_text,
            )
        return output_text
    finally:
        if log_file is not None:
            log_file.close()
        if unregister_process:
            unregister_process(proc)


def _parse_gs_progress(
    line: str,
    total_iterations: int,
    report,
) -> None:
    """Parse 3DGS training progress from output lines."""
    match = re.search(r"\[ITER\s+(\d+)\]", line, re.IGNORECASE)
    if match:
        current = int(match.group(1))
        pct = 5.0 + (current / total_iterations) * 45.0  # 5% -> 50%
        report(pct, f"3DGS training ({current}/{total_iterations})")


def _parse_gs2mesh_progress(line: str, report) -> None:
    """Parse gs2mesh progress from output lines.

    With --skip_TSDF, gs2mesh only runs rendering + stereo (55%-80%).
    GPU TSDF fusion is handled separately afterwards.
    """
    lower = line.lower()
    if "stereo" in lower or "disparity" in lower:
        match = re.search(r"(\d+)%", line)
        if match:
            pct = 55.0 + int(match.group(1)) * 0.25  # 55% -> 80%
            report(pct, "Stereo depth matching")


def _find_output_mesh(mesh_dir: Path) -> Path | None:
    """Find the output mesh in gs2mesh output directory."""
    if not mesh_dir.is_dir():
        return None

    # gs2mesh outputs {TSDF_string}_cleaned_mesh.ply
    matches = list(mesh_dir.glob("*_cleaned_mesh.ply"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_size)

    # Fallback: any PLY or OBJ
    for pattern in ["*.ply", "**/*.ply", "*.obj", "**/*.obj"]:
        matches = list(mesh_dir.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_size)

    return None


def _resolve_training_runtime_stack(
    runtime_profile: str,
) -> _Gs2meshRuntimeStack:
    if runtime_profile == "compat":
        return _compat_runtime_stack(required=True)
    return _accel_runtime_stack()


def _build_stereo_runtime_stacks(
    runtime_profile: str,
) -> list[_Gs2meshRuntimeStack]:
    if runtime_profile == "compat":
        return [_compat_runtime_stack(required=True, diagnostics=True)]

    stacks = [_accel_runtime_stack()]
    compat_stack = _compat_runtime_stack(required=False, diagnostics=True)
    if compat_stack is not None:
        stacks.append(compat_stack)
    else:
        print("gs2mesh compat runtime unavailable; no stereo fallback stack")
    return stacks


def _accel_runtime_stack(
    *,
    diagnostics: bool = False,
) -> _Gs2meshRuntimeStack:
    env = {"PYTHONFAULTHANDLER": "1"}
    if diagnostics:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    return _Gs2meshRuntimeStack(
        name="accel",
        python_executable="python3",
        gaussian_splatting_root=_GAUSSIAN_SPLATTING_ACCEL,
        env_overrides=env,
    )


def _compat_runtime_stack(
    *,
    required: bool,
    diagnostics: bool = False,
) -> _Gs2meshRuntimeStack | None:
    missing: list[str] = []
    if not _GS_COMPAT_SITE_PACKAGES.is_dir():
        missing.append(str(_GS_COMPAT_SITE_PACKAGES))
    if not _GAUSSIAN_SPLATTING_COMPAT.is_dir():
        missing.append(str(_GAUSSIAN_SPLATTING_COMPAT))
    if missing:
        if required:
            raise RuntimeError(
                "gs2mesh compat runtime is unavailable: " + ", ".join(missing)
            )
        return None

    env = {
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTHONFAULTHANDLER": "1",
    }
    if diagnostics:
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    return _Gs2meshRuntimeStack(
        name="compat",
        python_executable="python3",
        gaussian_splatting_root=_GAUSSIAN_SPLATTING_COMPAT,
        env_overrides=env,
        extra_pythonpaths=(str(_GS_COMPAT_SITE_PACKAGES),),
    )


def _build_runtime_env(
    gaussian_splatting_root: Path,
    env_overrides: dict[str, str] | None = None,
    extra_pythonpaths: tuple[str, ...] = (),
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(
        gaussian_splatting_root,
        extra_pythonpaths=extra_pythonpaths,
    )
    if env_overrides:
        env.update(env_overrides)
    return env


def _build_pythonpath(
    gaussian_splatting_root: Path,
    *,
    extra_pythonpaths: tuple[str, ...] = (),
) -> str:
    preferred = [
        "/app",
        str(_GS2MESH_BASE),
        str(gaussian_splatting_root),
        "/opt/sam2",
    ]
    preferred = [*extra_pythonpaths, *preferred]
    existing = [
        part
        for part in os.environ.get("PYTHONPATH", "").split(":")
        if part
    ]
    filtered: list[str] = []
    ignored = {
        str(_GAUSSIAN_SPLATTING_ACCEL),
        str(_GAUSSIAN_SPLATTING_COMPAT),
    }
    for part in existing:
        if part in ignored or part in preferred or part in filtered:
            continue
        filtered.append(part)
    return ":".join(preferred + filtered)


def _ensure_gs2mesh_gaussian_splatting_link(target: Path) -> None:
    if not target.is_dir():
        raise RuntimeError(f"gaussian-splatting target missing: {target}")

    if _GS2MESH_GS_LINK.is_symlink():
        if _GS2MESH_GS_LINK.resolve() == target.resolve():
            return
        _GS2MESH_GS_LINK.unlink()
    elif _GS2MESH_GS_LINK.exists():
        raise RuntimeError(
            f"Expected symlink at {_GS2MESH_GS_LINK}, found directory/file"
        )

    _GS2MESH_GS_LINK.symlink_to(target, target_is_directory=True)
    print(f"gs2mesh gaussian-splatting target -> {target}")


def _run_gs2mesh_stereo_with_retries(
    *,
    scene_name: str,
    stereo_model: str,
    gs_iterations: int,
    tsdf_voxel: int,
    tsdf_depth_trunc: float,
    runtime_profile: str,
    gs2mesh_output: Path,
    debug_dir: Path,
    report,
    register_process=None,
    unregister_process=None,
) -> None:
    attempts = _build_stereo_runtime_stacks(runtime_profile)
    gs2mesh_args_tail = [
        "-u",
        str(_GS2MESH_BASE / "run_single.py"),
        "--colmap_name",
        scene_name,
        "--dataset_name",
        "custom",
        "--experiment_folder_name",
        "clip2mesh",
        "--skip_video_extraction",
        "--skip_colmap",
        "--skip_GS",
        "--skip_masking",
        "--skip_TSDF",
        "--GS_iterations",
        str(gs_iterations),
        "--stereo_model",
        stereo_model,
        "--TSDF_voxel",
        str(tsdf_voxel),
        "--TSDF_sdf_trunc",
        str(tsdf_depth_trunc),
    ]

    for idx, stack in enumerate(attempts, start=1):
        args = [stack.python_executable, *gs2mesh_args_tail]
        env = _build_runtime_env(
            stack.gaussian_splatting_root,
            stack.env_overrides,
            extra_pythonpaths=stack.extra_pythonpaths,
        )
        log_path = debug_dir / f"stereo_attempt_{idx:02d}_{stack.name}.log"
        meta_path = debug_dir / f"stereo_attempt_{idx:02d}_{stack.name}.json"

        report(55.0, f"Running gs2mesh stereo depth estimation ({stack.name})")
        _cleanup_gs2mesh_stereo_outputs(gs2mesh_output)
        _ensure_gs2mesh_gaussian_splatting_link(stack.gaussian_splatting_root)
        _ensure_gs2mesh_renderer_compat()
        _ensure_gaussian_renderer_compat(stack.gaussian_splatting_root)

        try:
            _run_subprocess(
                args,
                cwd=str(_GS2MESH_BASE),
                prefix="gs2mesh",
                progress_fn=lambda line: _parse_gs2mesh_progress(line, report),
                register_process=register_process,
                unregister_process=unregister_process,
                error_msg="gs2mesh stereo estimation failed",
                env=env,
                log_path=log_path,
                capture_output=True,
            )
            _validate_gs2mesh_stereo_outputs(gs2mesh_output, stereo_model)
            _write_attempt_metadata(
                meta_path,
                stack=stack,
                args=args,
                env=env,
                log_path=log_path,
                success=True,
                failure_reason=None,
                exit_code=0,
            )
            print(f"gs2mesh stereo attempt succeeded: {stack.name}")
            return
        except _SubprocessFailure as exc:
            failure_reason = _classify_gs2mesh_failure(exc.output_text)
            retrying = (
                idx < len(attempts)
                and failure_reason in _RETRYABLE_GS2MESH_FAILURES
            )
            _write_attempt_metadata(
                meta_path,
                stack=stack,
                args=args,
                env=env,
                log_path=log_path,
                success=False,
                failure_reason=failure_reason,
                exit_code=exc.exit_code,
            )
            print(
                f"gs2mesh stereo attempt failed: stack={stack.name} "
                f"reason={failure_reason}"
            )
            if retrying:
                print("Retrying gs2mesh stereo with fallback runtime stack")
                continue
            raise RuntimeError(
                _format_gs2mesh_failure_message(
                    failure_reason,
                    log_path,
                    debug_dir,
                )
            ) from exc
        except RuntimeError as exc:
            failure_reason = _classify_gs2mesh_failure(str(exc))
            retrying = (
                idx < len(attempts)
                and failure_reason in _RETRYABLE_GS2MESH_FAILURES
            )
            _write_attempt_metadata(
                meta_path,
                stack=stack,
                args=args,
                env=env,
                log_path=log_path,
                success=False,
                failure_reason=failure_reason,
                exit_code=None,
            )
            print(
                f"gs2mesh stereo attempt validation failed: stack={stack.name} "
                f"reason={failure_reason}"
            )
            if retrying:
                print("Retrying gs2mesh stereo after validation failure")
                continue
            raise RuntimeError(
                _format_gs2mesh_failure_message(
                    failure_reason,
                    log_path,
                    debug_dir,
                )
            ) from exc

    raise RuntimeError("gs2mesh stereo estimation failed without attempts")


def _cleanup_gs2mesh_stereo_outputs(gs2mesh_output: Path) -> None:
    if gs2mesh_output.exists():
        shutil.rmtree(gs2mesh_output)


def _iter_camera_string_fields(value: object) -> list[str]:
    """Collect nested string fields from a camera-data entry."""
    results: list[str] = []
    if isinstance(value, str):
        results.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            results.extend(_iter_camera_string_fields(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            results.extend(_iter_camera_string_fields(nested))
    return results


def _extract_frame_index_from_text(text: str) -> int | None:
    path = Path(text)
    search_space = [path.stem, path.name, text]
    for candidate_text in search_space:
        matches = re.findall(r"(?<!\d)(\d{5})(?!\d)", candidate_text)
        if len(matches) == 1:
            return int(matches[0])
    return None


def _load_pose_frame_index_lookup(
    mask_root: Path,
) -> list[tuple[int, "np.ndarray"]]:
    """Load frame-indexed COLMAP poses adjacent to the SAM2 mask directory."""
    import numpy as np

    for parent in (mask_root, *mask_root.parents):
        pose_path = parent / "p2_colmap" / "camera_poses.json"
        if not pose_path.is_file():
            continue
        with pose_path.open("r", encoding="utf-8") as f:
            pose_data = json.load(f)
        if not isinstance(pose_data, list):
            continue

        lookup: list[tuple[int, np.ndarray]] = []
        for record in pose_data:
            if not isinstance(record, dict):
                continue
            frame_name = record.get("frame_name")
            if not isinstance(frame_name, str):
                continue
            frame_idx = _extract_frame_index_from_text(frame_name)
            matrix = record.get("transform_matrix")
            if frame_idx is None or not isinstance(matrix, list):
                continue
            pose = np.asarray(matrix, dtype=np.float64)
            if pose.shape == (4, 4):
                lookup.append((frame_idx, pose))
        if lookup:
            return lookup
    return []


def _resolve_camera_frame_index_by_pose(
    camera_entry: dict[str, object],
    pose_lookup: list[tuple[int, "np.ndarray"]],
) -> int | None:
    """Resolve a frame index by matching gs2mesh camera pose to COLMAP poses."""
    if not pose_lookup:
        return None

    import numpy as np

    extrinsic = camera_entry.get("extrinsic")
    if not isinstance(extrinsic, list):
        return None
    camera_pose = np.asarray(extrinsic, dtype=np.float64)
    if camera_pose.shape != (4, 4):
        return None

    camera_pos = camera_pose[:3, 3]
    best_idx: int | None = None
    best_dist = float("inf")
    second_dist = float("inf")
    for frame_idx, pose in pose_lookup:
        dist = float(np.linalg.norm(camera_pos - pose[:3, 3]))
        if dist < best_dist:
            second_dist = best_dist
            best_dist = dist
            best_idx = frame_idx
        elif dist < second_dist:
            second_dist = dist

    if best_idx is None:
        return None
    if best_dist <= 1e-3:
        return best_idx
    if best_dist <= 5e-2 and best_dist * 10.0 < second_dist:
        return best_idx
    return None


def _resolve_camera_frame_index(
    camera_entry: dict[str, object],
    pose_lookup: list[tuple[int, "np.ndarray"]] | None = None,
) -> int:
    """Resolve the original 5-digit frame index from gs2mesh camera metadata."""
    priority_keys = (
        "image_name",
        "image_path",
        "file_name",
        "filename",
        "path",
        "name",
    )
    candidate_strings: list[str] = []
    for key in priority_keys:
        value = camera_entry.get(key)
        if isinstance(value, str):
            candidate_strings.append(value)
    for value in camera_entry.values():
        candidate_strings.extend(_iter_camera_string_fields(value))

    candidates: list[int] = []
    seen: set[int] = set()
    for text in candidate_strings:
        value = _extract_frame_index_from_text(text)
        if value is not None and value not in seen:
            candidates.append(value)
            seen.add(value)
        if candidates:
            break

    if len(candidates) == 1:
        return candidates[0]
    if not candidates and pose_lookup:
        pose_match = _resolve_camera_frame_index_by_pose(
            camera_entry,
            pose_lookup,
        )
        if pose_match is not None:
            return pose_match
    if not candidates:
        raise RuntimeError(
            "Could not resolve frame index from gs2mesh camera_data entry"
        )
    raise RuntimeError(
        "Ambiguous frame index in gs2mesh camera_data entry: "
        + ", ".join(str(value) for value in candidates)
    )


def _materialize_tsdf_masks(gs2mesh_output: Path, mask_dir: str | Path) -> None:
    """Convert canonical SAM2 masks into per-view ``left_mask.npy`` files."""
    import cv2
    import numpy as np

    mask_root = Path(mask_dir)
    camera_data_path = gs2mesh_output / "camera_data.json"
    if not camera_data_path.is_file():
        raise RuntimeError(
            "Missing camera_data.json while preparing TSDF masks"
        )

    with camera_data_path.open("r", encoding="utf-8") as f:
        camera_data = json.load(f)
    if not isinstance(camera_data, list):
        raise RuntimeError("camera_data.json must contain a list of camera entries")

    pose_lookup = _load_pose_frame_index_lookup(mask_root)
    for cam_idx, camera_record in enumerate(camera_data):
        if not isinstance(camera_record, dict):
            raise RuntimeError(
                f"camera_data.json entry {cam_idx} is not an object"
            )
        left_entry = camera_record.get("left")
        if not isinstance(left_entry, dict):
            raise RuntimeError(
                f"camera_data.json entry {cam_idx} is missing a 'left' object"
            )
        frame_idx = _resolve_camera_frame_index(left_entry, pose_lookup)
        mask_path = mask_root / f"{frame_idx:05d}.png"
        if not mask_path.is_file():
            raise RuntimeError(
                f"Missing canonical SAM2 mask for frame {frame_idx:05d}: {mask_path}"
            )

        view_dir = gs2mesh_output / f"{cam_idx:03d}"
        left_image_path = view_dir / "left.png"
        if not left_image_path.is_file():
            raise RuntimeError(f"Missing rendered left image: {left_image_path}")

        left_image = cv2.imread(str(left_image_path), cv2.IMREAD_COLOR)
        if left_image is None:
            raise RuntimeError(f"Could not read rendered left image: {left_image_path}")
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError(f"Could not read canonical SAM2 mask: {mask_path}")
        if mask_img.shape[:2] != left_image.shape[:2]:
            mask_img = cv2.resize(
                mask_img,
                (left_image.shape[1], left_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        np.save(view_dir / "left_mask.npy", mask_img > 0)


def _validate_gs2mesh_stereo_outputs(
    gs2mesh_output: Path,
    stereo_model: str,
) -> None:
    camera_data = gs2mesh_output / "camera_data.json"
    if not camera_data.is_file():
        raise RuntimeError("Missing stereo outputs: camera_data.json")
    depth_matches = sorted(
        gs2mesh_output.glob(f"*/out_{stereo_model}/depth.npy")
    )
    if not depth_matches:
        raise RuntimeError(
            f"Missing stereo outputs: */out_{stereo_model}/depth.npy"
        )


def _write_attempt_metadata(
    meta_path: Path,
    *,
    stack: _Gs2meshRuntimeStack,
    args: list[str],
    env: dict[str, str],
    log_path: Path,
    success: bool,
    failure_reason: str | None,
    exit_code: int | None,
) -> None:
    payload = {
        "stack": stack.name,
        "python_executable": stack.python_executable,
        "gaussian_splatting_root": str(stack.gaussian_splatting_root),
        "command": args,
        "log_path": str(log_path),
        "success": success,
        "failure_reason": failure_reason,
        "exit_code": exit_code,
        "env_overrides": {
            key: env[key]
            for key in sorted(stack.env_overrides)
            if key in env
        },
        "extra_pythonpaths": list(stack.extra_pythonpaths),
        "pythonpath": env.get("PYTHONPATH", ""),
    }
    meta_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _classify_gs2mesh_failure(text: str) -> str:
    lower = text.lower()
    if "illegal memory access" in lower:
        return "cuda_illegal_memory_access"
    if "diff_gaussian_rasterization" in lower and "cuda error" in lower:
        return "cuda_rasterizer_failure"
    if (
        "unexpected keyword argument" in lower
        or "camera.__init__" in lower
        or "train_test_exp" in lower
        or "antialiasing" in lower
        or "depths" in lower
        or "not enough values to unpack" in lower
    ):
        return "renderer_signature_mismatch"
    if (
        "no module named 'diff_gaussian_rasterization'" in lower
        or "modulenotfounderror: no module named 'diff_gaussian_rasterization'"
        in lower
    ):
        return "missing_rasterizer_extension"
    if "missing stereo outputs" in lower or "camera_data.json" in lower:
        return "missing_output_artifacts"
    return "unclassified"


def _format_gs2mesh_failure_message(
    failure_reason: str,
    log_path: Path,
    debug_dir: Path,
) -> str:
    return (
        "gs2mesh stereo estimation failed "
        f"({failure_reason}). See {log_path} and {debug_dir} for attempt "
        "logs and metadata."
    )
