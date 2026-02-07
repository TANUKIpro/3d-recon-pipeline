"""Stage 2: Pi3X 3D reconstruction with triple filtering.

Runs Pi3X on full (unmasked) images for optimal pose estimation,
then applies confidence + depth-edge + SAM2 mask filtering.

Adapted from im2pc/host/pi3x_sam2_cli.py.
"""

import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from vram_utils import (
    cleanup_pytorch_vram,
    get_free_vram_mb,
    log_vram,
    offload_module,
)

# Estimated Pi3X model weight size on GPU (encoder + decoder + heads)
_ESTIMATED_MODEL_MB = 5500

# OOM strategy: keep image detail first, then reduce resolution as a last resort.
_OOM_FRAME_REDUCTION_FACTOR = 0.80
_OOM_REDUCTION_FACTOR = 0.70
_MIN_INFER_FRAMES = 12
_MIN_PIXEL_LIMIT = 50_000
_MAX_OOM_RETRIES = 7
_USE_CHUNK_FALLBACK = True

# (effective_free_vram_mb_lower, frame_pixel_budget)
# budget = N_frames * pixel_limit
_VRAM_PROFILES = [
    (14000, 12_750_000),  # 50f*255K, or 85f*150K
    (12000, 10_000_000),  # 50f*200K, or 67f*150K
    (10000,  7_500_000),  # 50f*150K, or 75f*100K
    ( 8000,  4_800_000),  # 40f*120K, or 60f*80K
    (    0,  2_400_000),  # 30f*80K,  or 48f*50K
]

# Clamp per-chunk similarity scale to avoid catastrophic drift.
_CHUNK_SCALE_MIN = 0.85
_CHUNK_SCALE_MAX = 1.15

# Camera-orbit plane alignment (Stage 2 output coordinate normalization)
_ALIGN_MIN_FRAMES = 4
_ALIGN_MIN_LINE_RATIO = 1e-3
_ALIGN_MAX_PLANAR_RATIO = 0.35
_ALIGN_MIN_RADIUS = 1e-4
_ALIGN_ORIENTATION_SCORE_EPS = 1e-4
_TARGET_PLANE_NORMAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)

ProgressCallback = Callable[[float, str | None], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle to a 3x3 rotation matrix."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    K = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_matrix_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return rotation matrix R such that R @ src ~= dst."""
    src = src / max(np.linalg.norm(src), 1e-12)
    dst = dst / max(np.linalg.norm(dst), 1e-12)
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))

    if cross_norm < 1e-10:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180° turn around any axis orthogonal to src.
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            aux = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.cross(src, aux)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return _axis_angle_to_rotation(axis, np.pi)

    axis = cross / cross_norm
    angle = float(np.arctan2(cross_norm, dot))
    return _axis_angle_to_rotation(axis, angle)


def _rotation_matrix_about_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _matrix_to_list(mat: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in mat]


def _vector_to_list(vec: np.ndarray) -> list[float]:
    return [float(v) for v in vec]


def _safe_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _auto_scale_frame_target(
    requested_frames: int,
    pixel_limit: int,
) -> int:
    """Scale down frame count from VRAM budget while preserving pixel detail.

    Subtracts estimated model weight size from free VRAM before profile
    selection, since this function is called before model loading.
    Only reduces the frame count (never increases beyond caller's request).
    """
    free = get_free_vram_mb()
    if free is None:
        return requested_frames

    effective_free = free - _ESTIMATED_MODEL_MB
    print(
        f"VRAM auto-scale: {free}MB free, "
        f"{effective_free}MB effective (minus ~{_ESTIMATED_MODEL_MB}MB model), "
        f"requested {requested_frames} frames @ pixel_limit={pixel_limit}"
    )

    for threshold, budget in _VRAM_PROFILES:
        if effective_free >= threshold:
            safe_frames = max(2, budget // max(pixel_limit, 1))
            new_frames = min(requested_frames, safe_frames)
            if new_frames != requested_frames:
                print(
                    f"VRAM auto-scale: effective {effective_free}MB, "
                    f"budget {budget} / pixel_limit {pixel_limit} → "
                    f"frames {requested_frames}→{new_frames}"
                )
            else:
                print(
                    f"VRAM auto-scale: effective {effective_free}MB, "
                    f"budget {budget} / pixel_limit {pixel_limit} → "
                    f"no adjustment needed"
                )
            return new_frames

    return requested_frames


def _pick_frame_subset(
    frame_files: list[Path],
    target_frames: int,
) -> tuple[list[Path], list[int]]:
    """Pick evenly distributed frames and return (paths, source_indices)."""
    n_total = len(frame_files)
    if target_frames >= n_total:
        indices = list(range(n_total))
    else:
        # Even temporal coverage with deterministic selection.
        indices = [int(i) for i in np.linspace(0, n_total - 1, num=target_frames, dtype=np.int64)]
    return [frame_files[i] for i in indices], indices


def _load_images_from_paths(frame_paths: list[Path], pixel_limit: int) -> torch.Tensor:
    """Load selected JPEGs as float tensor (N,3,H,W), matching Pi3 resize policy."""
    if not frame_paths:
        return torch.zeros((0, 3, 0, 0), dtype=torch.float32)

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")

    first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    h, w = first.shape[:2]
    scale = min((pixel_limit / max(h * w, 1)) ** 0.5, 1.0)
    if scale < 1.0:
        new_h = max(int(h * scale) // 14 * 14, 14)
        new_w = max(int(w * scale) // 14 * 14, 14)
    else:
        new_h, new_w = h, w

    imgs = []
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if scale < 1.0:
            img = cv2.resize(img, (new_w, new_h))
        imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

    return torch.stack(imgs, dim=0)


def _frame_id_from_path(path: Path, fallback: int) -> int:
    """Resolve original frame index from filename stem, with safe fallback."""
    try:
        return int(path.stem)
    except ValueError:
        return int(fallback)


def _run_inference_memory_efficient(
    model, imgs: torch.Tensor, device: torch.device, dtype: torch.dtype | None
) -> dict:
    """Run Pi3X inference with stage-sequential parameter offloading.

    Splits the monolithic ``model(imgs)`` call into encode → decode →
    forward_head, offloading each stage's parameters to CPU after use.
    This frees ~2.1 GB of VRAM before the most memory-hungry head stage.

    The computation is **identical** to ``model(imgs[None])`` — only the
    parameter residency changes.
    """
    N, C, H, W = imgs.shape
    patch_h, patch_w = H // 14, W // 14
    imgs_batch = imgs[None]  # (1, N, 3, H, W)

    # Normalize images in full precision (matching Pi3X.forward() which does
    # this outside autocast context)
    imgs_batch = (imgs_batch - model.image_mean) / model.image_std

    use_autocast = dtype is not None and device.type == "cuda"

    with torch.no_grad():
        ctx = torch.amp.autocast("cuda", dtype=dtype) if use_autocast else _nullcontext()
        with ctx:
            # --- Stage 1: Encode (DINOv2 ViT-L ~600 MB) ---
            hidden, poses_, use_depth_mask, use_pose_mask, norm_factor = (
                model.encode(imgs_batch)
            )
            log_vram("after encode")

            # Offload encoder to CPU (~600 MB freed)
            offload_module(model.encoder)
            if hasattr(model, "depth_encoder"):
                offload_module(model.depth_encoder)
            log_vram("after encoder offload")

            # --- Stage 2: Decode (36-layer transformer ~1.5 GB) ---
            B = 1
            hidden = hidden.reshape(B, N, -1, model.dec_embed_dim)
            hidden, pos = model.decode(hidden, N, H, W, poses_, use_pose_mask)
            log_vram("after decode")

            # Offload decoder + pose injection to CPU (~1.5 GB freed)
            offload_module(model.decoder)
            if hasattr(model, "pose_inject_blk"):
                offload_module(model.pose_inject_blk)
            log_vram("after decoder offload")

            # --- Stage 3: Forward head (~400 MB, now has ~2.1 GB extra room) ---
            results = model.forward_head(
                hidden, pos, B, N, H, W, patch_h, patch_w
            )
            log_vram("after forward_head")

    return results


def _run_inference_chunked(
    model,
    imgs: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
    chunk_size: int = 20,
    overlap: int = 5,
    progress_cb: ProgressCallback | None = None,
    progress_start: float = 35.0,
    progress_end: float = 60.0,
) -> dict:
    """Fallback: process frames in overlapping windows when a single pass OOMs.

    Each chunk is processed via ``_run_inference_memory_efficient``, then
    results are merged using Procrustes alignment on overlapping frames.
    """
    N = imgs.shape[0]
    if N <= chunk_size:
        return _run_inference_memory_efficient(model, imgs, device, dtype)

    print(f"Chunked inference: {N} frames, chunk_size={chunk_size}, overlap={overlap}")
    all_chunks = []  # list of (start, end, results_dict)
    starts = list(range(0, N, chunk_size - overlap))
    total_chunks = max(1, len(starts))

    for chunk_idx, start in enumerate(starts):
        end = min(start + chunk_size, N)
        if end - start < 2:
            break
        print(f"  Processing chunk [{start}:{end}] ({end - start} frames)")
        _emit_progress(
            progress_cb,
            progress_start + ((chunk_idx / total_chunks) * (progress_end - progress_start)),
            f"Pi3X chunk inference ({chunk_idx + 1}/{total_chunks})",
        )

        # Reload full model to GPU for this chunk
        model.to(device)
        chunk_results = _run_inference_memory_efficient(
            model, imgs[start:end], device, dtype,
        )
        all_chunks.append((start, end, chunk_results))
        _emit_progress(
            progress_cb,
            progress_start + (((chunk_idx + 1) / total_chunks) * (progress_end - progress_start)),
            f"Pi3X chunk inference ({chunk_idx + 1}/{total_chunks})",
        )

        # Offload entire model between chunks
        model.cpu()
        torch.cuda.empty_cache()

        if end >= N:
            break

    if len(all_chunks) == 1:
        return all_chunks[0][2]

    return _merge_chunk_results(all_chunks, overlap, N)


def _merge_chunk_results(
    chunks: list[tuple[int, int, dict]],
    overlap: int,
    total_frames: int,
) -> dict:
    """Merge overlapping chunk results via Procrustes alignment.

    For each pair of adjacent chunks, the overlapping camera poses are used
    to estimate a rigid transform (rotation + translation + scale) that
    aligns the second chunk to the first.
    """
    ref_start, ref_end, merged = chunks[0]
    # Collect per-frame results for the reference chunk
    B = 1

    for i in range(1, len(chunks)):
        cur_start, cur_end, cur = chunks[i]
        # Overlap region in global frame indices
        ovlp_start = cur_start
        ovlp_end = min(ref_end, cur_end)
        n_ovlp = ovlp_end - ovlp_start
        if n_ovlp < 1:
            print(f"  Warning: no overlap between chunk {i-1} and {i}, concatenating as-is")
            merged = _concat_results(merged, cur, 0)
            ref_end = cur_end
            continue

        # Indices within each chunk's result tensors
        ref_ovlp_idx = slice(ovlp_start - ref_start, ovlp_end - ref_start)
        cur_ovlp_idx = slice(0, n_ovlp)

        # Camera poses: (B, N_chunk, 4, 4) — use positions (translation column)
        ref_poses = merged["camera_poses"][0, ref_ovlp_idx, :3, 3].cpu().numpy()
        cur_poses = cur["camera_poses"][0, cur_ovlp_idx, :3, 3].cpu().numpy()

        # Procrustes: find s*R@cur + t ≈ ref (scale can be noisy across chunks)
        R, t, s = _procrustes(cur_poses, ref_poses)
        s_clamped = float(np.clip(s, _CHUNK_SCALE_MIN, _CHUNK_SCALE_MAX))
        if abs(s_clamped - s) > 1e-6:
            print(
                f"  Warning: chunk {i} scale {s:.3f} out of range, "
                f"clamped to {s_clamped:.3f}"
            )
        s = s_clamped

        # Transform current chunk's world-space points and poses
        cur_aligned = _apply_rigid_transform(cur, R, t, s)

        # Merge: take non-overlapping frames from the aligned chunk
        n_new = cur_end - ovlp_end
        if n_new > 0:
            merged = _concat_results(merged, cur_aligned, n_ovlp)

        ref_end = cur_end

    return merged


def _procrustes(
    src: np.ndarray, tgt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute rigid + scale alignment: s*R@src + t ≈ tgt.

    Uses SVD-based Procrustes analysis.
    Returns (R [3x3], t [3], s [scalar]).
    """
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    # Scale
    s_src = np.sqrt((src_c ** 2).sum() / len(src_c))
    s_tgt = np.sqrt((tgt_c ** 2).sum() / len(tgt_c))
    s = s_tgt / max(s_src, 1e-8)

    # Rotation via SVD
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ S @ U.T

    t = tgt_mean - s * R @ src_mean
    return R, t, s


def _apply_rigid_transform(
    results: dict,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> dict:
    """Apply rigid transform (s*R@p + t) to points and camera poses in results."""
    device = results["points"].device
    R_t = torch.from_numpy(R).float().to(device)
    t_t = torch.from_numpy(t).float().to(device)

    out = {}
    for k, v in results.items():
        if k == "points":
            # (B, N, H, W, 3)
            shape = v.shape
            pts = v.reshape(-1, 3)
            pts = s * (pts @ R_t.T) + t_t
            out[k] = pts.reshape(shape)
        elif k == "camera_poses":
            # (B, N, 4, 4) — transform the translation column and rotation
            poses = v.clone()
            for b in range(poses.shape[0]):
                for n in range(poses.shape[1]):
                    pose = poses[b, n]
                    # Transform rotation
                    pose[:3, :3] = R_t @ pose[:3, :3]
                    # Transform translation
                    pose[:3, 3] = s * (R_t @ pose[:3, 3]) + t_t
            out[k] = poses
        else:
            out[k] = v
    return out


def _concat_results(
    a: dict, b: dict, skip_first_n: int
) -> dict:
    """Concatenate two result dicts along the frame (N) dimension.

    Skips the first ``skip_first_n`` frames of ``b`` (overlap region).
    """
    out = {}
    for k in a:
        va = a[k]
        vb = b[k]
        if k in ("points", "local_points", "rays", "conf"):
            # Shape: (B, N, H, W, ...)
            out[k] = torch.cat([va, vb[:, skip_first_n:]], dim=1)
        elif k == "camera_poses":
            # Shape: (B, N, 4, 4)
            out[k] = torch.cat([va, vb[:, skip_first_n:]], dim=1)
        elif k == "metric":
            # Shape: (B,) — keep the first chunk's metric
            out[k] = va
        else:
            out[k] = va
    return out


def _infer_forward_axis_sign_from_results(results: dict) -> tuple[int, dict]:
    """Infer whether camera forward is +col2 or -col2 from reconstructed points.

    Uses the per-frame reconstructed world points: whichever axis direction
    places more points in front of each camera is treated as forward.
    """
    meta: dict[str, object] = {
        "source": "points_in_front",
        "inferred_forward_axis": "camera +Z (col 2)",
        "selected_sign": 1,
    }

    if "camera_poses" not in results or "points" not in results:
        meta["reason"] = "missing_camera_poses_or_points"
        return 1, meta

    poses = results["camera_poses"][0]
    points = results["points"][0]
    if poses.shape[0] == 0 or points.shape[0] == 0:
        meta["reason"] = "empty_camera_poses_or_points"
        return 1, meta

    sample_per_frame = 2048
    mean_dot_plus_list: list[float] = []
    mean_dot_minus_list: list[float] = []
    frac_plus_positive_list: list[float] = []
    frac_minus_positive_list: list[float] = []

    for n in range(min(int(poses.shape[0]), int(points.shape[0]))):
        pts = points[n].reshape(-1, 3)
        if pts.shape[0] == 0:
            continue
        step = max(int(pts.shape[0] // sample_per_frame), 1)
        pts = pts[::step][:sample_per_frame]

        cam_center = poses[n, :3, 3]
        z_axis = poses[n, :3, 2]
        vec = pts - cam_center[None, :]
        dot = (vec * z_axis[None, :]).sum(dim=1)
        if dot.numel() == 0:
            continue

        mean_dot = float(dot.mean().item())
        mean_dot_plus_list.append(mean_dot)
        mean_dot_minus_list.append(-mean_dot)
        frac_plus_positive_list.append(float((dot > 0).float().mean().item()))
        frac_minus_positive_list.append(float((dot < 0).float().mean().item()))

    if not mean_dot_plus_list:
        meta["reason"] = "no_valid_samples"
        return 1, meta

    score_plus = float(np.mean(mean_dot_plus_list))
    score_minus = float(np.mean(mean_dot_minus_list))
    frac_plus = float(np.mean(frac_plus_positive_list))
    frac_minus = float(np.mean(frac_minus_positive_list))
    sign = 1 if score_plus >= score_minus else -1

    meta.update({
        "n_frames_evaluated": int(len(mean_dot_plus_list)),
        "sample_per_frame": sample_per_frame,
        "score_col_z": score_plus,
        "score_neg_col_z": score_minus,
        "positive_fraction_col_z": frac_plus,
        "positive_fraction_neg_col_z": frac_minus,
        "selected_sign": sign,
        "inferred_forward_axis": (
            "camera +Z (col 2)" if sign > 0 else "camera -Z (-col 2)"
        ),
    })
    return sign, meta


def _estimate_camera_plane_transform(
    camera_poses: np.ndarray,
    forward_axis_sign: int = 1,
    forward_axis_meta: dict | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Estimate world alignment from camera orbit plane via PCA.

    Aligns the best-fit camera orbit normal to +Y so the orbit plane becomes
    parallel to the default XZ reference surface.
    """
    meta: dict[str, object] = {
        "enabled": True,
        "applied": False,
        "method": "camera_plane_pca",
        "target_plane_normal": _vector_to_list(_TARGET_PLANE_NORMAL),
    }

    if camera_poses.shape[0] < _ALIGN_MIN_FRAMES:
        meta["reason"] = (
            f"insufficient_frames: need >= {_ALIGN_MIN_FRAMES}, "
            f"got {camera_poses.shape[0]}"
        )
        return None, None, meta

    centers = camera_poses[:, :3, 3].astype(np.float64)
    centroid = centers.mean(axis=0)
    centered = centers - centroid

    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    normal = eigvecs[:, 2]
    normal_raw = normal.copy()
    rotations = camera_poses[:, :3, :3].astype(np.float64)
    col2 = rotations[:, :, 2]
    sign = 1 if int(forward_axis_sign) >= 0 else -1
    forwards = col2 * sign
    downs = rotations[:, :, 1]     # OpenCV camera +Y (down)

    mean_forward = forwards.mean(axis=0)
    mean_down = downs.mean(axis=0)
    mean_forward_norm = float(np.linalg.norm(mean_forward))
    mean_down_norm = float(np.linalg.norm(mean_down))
    if mean_forward_norm > 1e-10:
        mean_forward = mean_forward / mean_forward_norm
    if mean_down_norm > 1e-10:
        mean_down = mean_down / mean_down_norm

    # Forward-axis sanity check: camera forward should generally point toward
    # the orbit centroid for turntable capture.
    to_center = _safe_normalize_rows(centroid[None, :] - centers)
    forward_plus_score = float(np.mean(np.sum(col2 * to_center, axis=1)))
    forward_minus_score = float(np.mean(np.sum((-col2) * to_center, axis=1)))

    line_ratio = float(eigvals[1] / max(eigvals[0], 1e-12))
    planar_ratio = float(eigvals[2] / max(eigvals[1], 1e-12))
    radius = float(np.sqrt(max((eigvals[0] + eigvals[1]) * 0.5, 0.0)))

    forward_dot = float(np.dot(mean_forward, normal)) if mean_forward_norm > 1e-10 else 0.0
    down_dot = float(np.dot(mean_down, normal)) if mean_down_norm > 1e-10 else 0.0

    # Disambiguate PCA normal sign with camera orientation assumption:
    # the object is below the camera, so camera forward/down should point
    # opposite to the chosen +Y plane normal on average.
    orientation_score = 0.0
    orientation_weight = 0.0
    if mean_forward_norm > 1e-10:
        orientation_score += 0.7 * forward_dot
        orientation_weight += 0.7
    if mean_down_norm > 1e-10:
        orientation_score += 0.3 * down_dot
        orientation_weight += 0.3
    if orientation_weight > 0:
        orientation_score /= orientation_weight

    normal_flipped = False
    if orientation_weight > 0 and orientation_score > _ALIGN_ORIENTATION_SCORE_EPS:
        normal = -normal
        forward_dot = -forward_dot
        down_dot = -down_dot
        normal_flipped = True

    meta.update({
        "eigenvalues": [float(v) for v in eigvals],
        "line_ratio": line_ratio,
        "planar_ratio": planar_ratio,
        "orbit_radius_estimate": radius,
        "camera_centroid": _vector_to_list(centroid),
        "mean_camera_forward": _vector_to_list(mean_forward),
        "mean_camera_down": _vector_to_list(mean_down),
        "forward_to_orbit_center_score_col_z": forward_plus_score,
        "forward_to_orbit_center_score_neg_col_z": forward_minus_score,
        "inferred_forward_axis": (
            "camera +Z (col 2)" if sign > 0 else "camera -Z (-col 2)"
        ),
        "forward_axis_source": (
            "points_in_front" if forward_axis_meta is not None else "default"
        ),
        "forward_axis_inference": forward_axis_meta or {},
        "normal_sign_score": float(orientation_score),
        "normal_flipped_from_orientation": normal_flipped,
        "normal_sign_policy": (
            "object_below_camera => mean(camera_forward/down · plane_normal) <= 0"
        ),
        "forward_dot_plane_normal": float(forward_dot),
        "down_dot_plane_normal": float(down_dot),
        "camera_plane_normal_raw": _vector_to_list(normal_raw),
        "camera_plane_normal_before": _vector_to_list(normal),
    })

    if line_ratio < _ALIGN_MIN_LINE_RATIO:
        meta["reason"] = (
            f"camera_path_too_linear: line_ratio={line_ratio:.6f} < "
            f"{_ALIGN_MIN_LINE_RATIO:.6f}"
        )
        return None, None, meta

    if planar_ratio > _ALIGN_MAX_PLANAR_RATIO:
        meta["reason"] = (
            f"camera_path_not_planar_enough: planar_ratio={planar_ratio:.6f} > "
            f"{_ALIGN_MAX_PLANAR_RATIO:.6f}"
        )
        return None, None, meta

    if radius < _ALIGN_MIN_RADIUS:
        meta["reason"] = (
            f"orbit_radius_too_small: radius={radius:.6f} < "
            f"{_ALIGN_MIN_RADIUS:.6f}"
        )
        return None, None, meta

    # Step 1: align orbit normal to +Y.
    R_normal = _rotation_matrix_from_vectors(normal, _TARGET_PLANE_NORMAL)

    # Step 2: stabilize yaw by sending the major in-plane axis to +X.
    major_rot = R_normal @ major
    major_xz = major_rot.copy()
    major_xz[1] = 0.0
    major_xz_norm = np.linalg.norm(major_xz)

    if major_xz_norm > 1e-10:
        major_xz = major_xz / major_xz_norm
        yaw = float(np.arctan2(major_xz[2], major_xz[0]))
        R_yaw = _rotation_matrix_about_y(-yaw)
    else:
        yaw = 0.0
        R_yaw = np.eye(3, dtype=np.float64)

    R = R_yaw @ R_normal
    # Rotate around camera centroid (preserves global placement scale).
    t = centroid - (R @ centroid)

    normal_after = R @ normal
    meta.update({
        "applied": True,
        "yaw_correction_rad": yaw,
        "rotation": _matrix_to_list(R),
        "translation": _vector_to_list(t),
        "camera_plane_normal_after": _vector_to_list(normal_after),
    })
    return R, t, meta


def _align_results_to_camera_plane(results: dict) -> tuple[dict, dict]:
    """Normalize world axes from camera orbit geometry."""
    if "camera_poses" not in results:
        return results, {
            "enabled": True,
            "applied": False,
            "method": "camera_plane_pca",
            "reason": "camera_poses_missing",
            "target_plane_normal": _vector_to_list(_TARGET_PLANE_NORMAL),
        }

    forward_axis_sign, forward_axis_meta = _infer_forward_axis_sign_from_results(results)
    poses_np = results["camera_poses"][0].detach().cpu().numpy()
    R, t, meta = _estimate_camera_plane_transform(
        poses_np,
        forward_axis_sign=forward_axis_sign,
        forward_axis_meta=forward_axis_meta,
    )
    if R is None or t is None:
        return results, meta

    aligned = _apply_rigid_transform(results, R, t, s=1.0)
    return aligned, meta


class _nullcontext:
    """Minimal no-op context manager (for Python <3.10 compat)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def run_pi3x_inference(
    frames_dir: str,
    output_dir: str,
    pixel_limit: int | None = None,
    max_frames: int | None = None,
    conf_threshold: float | None = None,
    edge_rtol: float | None = None,
    align_camera_plane: bool | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[Path, Path, Path]:
    """Run Pi3X inference and save full (conf+edge filtered) point cloud.

    This is the first half of the reconstruction: runs the model and applies
    confidence + depth-edge filtering only (no SAM2 masks). Saves intermediate
    cache for later mask application via ``apply_sam2_masks()``.

    Args:
        frames_dir: Directory of JPEG frames.
        output_dir: Output directory.
        pixel_limit: Max pixels per frame for resize.
        max_frames: Maximum frames for Pi3X.
        conf_threshold: Confidence threshold for filtering.
        edge_rtol: Depth edge relative tolerance.
        align_camera_plane: If True, align camera-orbit plane to the default
            reference surface (XZ) by rotating world coordinates.

    Returns:
        Tuple of (ply_full_path, poses_path, cache_path).
    """
    if pixel_limit is None:
        pixel_limit = int(os.environ.get("PIXEL_LIMIT", "255000"))
    if max_frames is None:
        max_frames = int(os.environ.get("MAX_FRAMES", "50"))
    if conf_threshold is None:
        conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.1"))
    if edge_rtol is None:
        edge_rtol = float(os.environ.get("EDGE_RTOL", "0.03"))
    if align_camera_plane is None:
        align_camera_plane = _env_flag("ALIGN_CAMERA_PLANE", True)

    from pi3.utils.basic import write_ply
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import depth_edge

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_cb, 3.0, "Scanning input frames")

    # --- Count frames and apply frame budget before inference ---
    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    num_frames = len(frame_files)
    if num_frames < 2:
        raise RuntimeError("Need at least 2 frames for reconstruction.")

    target_frames = min(max_frames, num_frames)
    target_frames = _auto_scale_frame_target(target_frames, pixel_limit)
    _emit_progress(progress_cb, 8.0, f"Selected up to {target_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if capability >= 8 else torch.float16
    else:
        dtype = None

    # --- Load model once (outside retry loop) ---
    log_vram("before Pi3X load")
    print("Loading Pi3X model...")
    _emit_progress(progress_cb, 12.0, "Loading Pi3X model")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    log_vram("after Pi3X load")
    _emit_progress(progress_cb, 22.0, "Pi3X model loaded")

    # --- OOM retry loop: reduce frame count first, resolution second ---
    current_target_frames = max(2, target_frames)
    current_pixel_limit = pixel_limit
    selected_files = frame_files
    selected_source_indices = list(range(num_frames))
    selected_frame_ids = list(range(num_frames))
    used_chunk_fallback = False
    retry_count = 0

    while True:
        selected_files, selected_source_indices = _pick_frame_subset(
            frame_files, current_target_frames
        )
        selected_frame_ids = [
            _frame_id_from_path(p, src_idx)
            for p, src_idx in zip(selected_files, selected_source_indices)
        ]

        imgs = _load_images_from_paths(selected_files, current_pixel_limit)
        if imgs.numel() == 0 or imgs.shape[0] < 2:
            raise RuntimeError("Need at least 2 frames for reconstruction.")

        N, _, H, W = imgs.shape
        print(
            f"Pi3X input: {N}/{num_frames} frames, {H}x{W} "
            f"(pixel_limit={current_pixel_limit})"
        )
        imgs = imgs.to(device)

        print(f"Running Pi3X inference on {N} frames...")
        _emit_progress(
            progress_cb,
            30.0,
            f"Running Pi3X inference ({N} frames, attempt {retry_count + 1})",
        )
        try:
            model.to(device)
            results = _run_inference_memory_efficient(model, imgs, device, dtype)
            _emit_progress(progress_cb, 60.0, "Pi3X inference complete")
            break  # success
        except torch.cuda.OutOfMemoryError:
            model.cpu()
            torch.cuda.empty_cache()
            retry_count += 1
            if retry_count > _MAX_OOM_RETRIES:
                del imgs
                raise RuntimeError(
                    f"CUDA OOM after {retry_count} retries "
                    f"(frames={current_target_frames}, pixel_limit={current_pixel_limit})."
                )

            reduced = False

            if current_target_frames > _MIN_INFER_FRAMES:
                new_target = max(
                    _MIN_INFER_FRAMES,
                    int(current_target_frames * _OOM_FRAME_REDUCTION_FACTOR),
                )
                if new_target < current_target_frames:
                    print(
                        f"CUDA OOM: reducing frame count {current_target_frames}→{new_target} "
                        f"(retry {retry_count}/{_MAX_OOM_RETRIES})"
                    )
                    current_target_frames = new_target
                    reduced = True
                    _emit_progress(
                        progress_cb,
                        24.0,
                        f"OOM fallback: reducing frames to {new_target} (retry {retry_count})",
                    )

            if not reduced and current_pixel_limit > _MIN_PIXEL_LIMIT:
                new_limit = max(
                    int(current_pixel_limit * _OOM_REDUCTION_FACTOR),
                    _MIN_PIXEL_LIMIT,
                )
                if new_limit < current_pixel_limit:
                    print(
                        f"CUDA OOM: reducing resolution {current_pixel_limit}→{new_limit} "
                        f"(retry {retry_count}/{_MAX_OOM_RETRIES})"
                    )
                    current_pixel_limit = new_limit
                    reduced = True
                    _emit_progress(
                        progress_cb,
                        24.0,
                        f"OOM fallback: reducing pixel limit to {new_limit} (retry {retry_count})",
                    )

            if reduced:
                del imgs
                continue

            if _USE_CHUNK_FALLBACK and N > 12 and not used_chunk_fallback:
                chunk_size = max(10, min(20, N))
                overlap = max(3, min(6, chunk_size // 3))
                print(
                    f"CUDA OOM persists, trying chunk fallback "
                    f"(chunk_size={chunk_size}, overlap={overlap})"
                )
                _emit_progress(progress_cb, 32.0, "OOM fallback: switching to chunked inference")
                model.to(device)
                try:
                    results = _run_inference_chunked(
                        model,
                        imgs,
                        device,
                        dtype,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        progress_cb=progress_cb,
                        progress_start=35.0,
                        progress_end=60.0,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    del imgs
                    raise RuntimeError("CUDA OOM even with chunk fallback.") from e
                used_chunk_fallback = True
                break

            del imgs
            raise RuntimeError(
                "CUDA OOM and no further fallback available "
                f"(frames={current_target_frames}, pixel_limit={current_pixel_limit})."
            )

    print("Pi3X inference complete")
    if used_chunk_fallback:
        print("Warning: Chunk fallback was used. Verify pose consistency in Pi3X preview.")
    log_vram("after Pi3X inference")

    # --- Release model before filtering ---
    del model
    cleanup_pytorch_vram()

    # --- Optional world alignment from camera orbit plane ---
    alignment_meta: dict[str, object] = {
        "enabled": bool(align_camera_plane),
        "applied": False,
        "method": "camera_plane_pca",
        "target_plane_normal": _vector_to_list(_TARGET_PLANE_NORMAL),
    }
    if align_camera_plane:
        _emit_progress(progress_cb, 66.0, "Aligning world axes from camera orbit")
        results, alignment_meta = _align_results_to_camera_plane(results)
        if alignment_meta.get("applied"):
            before = alignment_meta.get("camera_plane_normal_before")
            after = alignment_meta.get("camera_plane_normal_after")
            flipped = alignment_meta.get("normal_flipped_from_orientation")
            inferred_forward_axis = alignment_meta.get("inferred_forward_axis")
            print("Applied camera-plane world alignment:")
            print(f"  Inferred forward axis: {inferred_forward_axis}")
            print(f"  Normal sign flip (orientation): {flipped}")
            print(f"  Normal before: {before}")
            print(f"  Normal after:  {after}")
        else:
            print(f"Skipped camera-plane alignment: {alignment_meta.get('reason', 'unknown')}")
    else:
        alignment_meta["reason"] = "disabled"
        print("Camera-plane alignment disabled (ALIGN_CAMERA_PLANE=0).")

    # --- Confidence + edge filtering ---
    _emit_progress(progress_cb, 70.0, "Applying confidence and edge filters")
    # Filter 1: Confidence
    conf_mask = torch.sigmoid(results["conf"][..., 0]) > conf_threshold
    conf_count = conf_mask[0].sum().item()

    # Filter 2: Depth edge
    non_edge = ~depth_edge(results["local_points"][..., 2], rtol=edge_rtol)
    conf_edge_mask = conf_mask & non_edge
    conf_edge_count = conf_edge_mask[0].sum().item()

    total = N * H * W
    print("Filtering stats (conf+edge):")
    print(f"  Total pixels:     {total:>10,}")
    print(f"  After conf:       {conf_count:>10,} ({100*conf_count/total:.1f}%)")
    print(f"  After conf+edge:  {conf_edge_count:>10,} ({100*conf_edge_count/total:.1f}%)")

    # --- Save full (conf+edge filtered) point cloud ---
    _emit_progress(progress_cb, 80.0, "Saving point cloud")
    ply_full_path = output_path / "object_full.ply"
    if conf_edge_count > 0:
        full_points = results["points"][0][conf_edge_mask[0]].cpu()
        full_colors = imgs.permute(0, 2, 3, 1)[conf_edge_mask[0]]
        write_ply(full_points, full_colors, str(ply_full_path))
        print(f"Saved full PLY: {ply_full_path} ({conf_edge_count:,} points)")
        del full_points, full_colors
    else:
        write_ply(torch.zeros(0, 3), torch.zeros(0, 3), str(ply_full_path))
        print("Warning: No points after conf+edge filtering.")

    # --- Save camera poses ---
    _emit_progress(progress_cb, 90.0, "Saving camera poses")
    poses = results["camera_poses"][0].cpu().numpy()
    poses_path = output_path / "camera_poses.json"
    frame_indices_used = selected_frame_ids[:N]
    source_indices_used = selected_source_indices[:N]
    with open(poses_path, "w") as f:
        json.dump(
            {
                "poses": [pose.tolist() for pose in poses],
                "frame_indices": frame_indices_used,
                "source_indices": source_indices_used,
                "alignment": alignment_meta,
            },
            f, indent=2,
        )
    print(f"Saved camera poses: {poses_path}")

    # --- Cache intermediate data for apply_sam2_masks() ---
    _emit_progress(progress_cb, 96.0, "Saving Pi3X cache")
    cache_path = output_path / "pi3x_cache.npz"
    np.savez_compressed(
        str(cache_path),
        points=results["points"][0].cpu().half().numpy(),
        conf_edge_mask=conf_edge_mask[0].cpu().numpy(),
        colors=imgs.permute(0, 2, 3, 1).cpu().numpy(),
        frame_indices=np.array(frame_indices_used, dtype=np.int32),
        source_indices=np.array(source_indices_used, dtype=np.int32),
        N=np.array(N),
        H=np.array(H),
        W=np.array(W),
    )
    print(f"Saved Pi3X cache: {cache_path}")

    # Cleanup remaining GPU tensors
    del results, imgs, conf_mask, non_edge, conf_edge_mask
    cleanup_pytorch_vram()
    _emit_progress(progress_cb, 100.0, "Pi3X stage complete")

    return ply_full_path, poses_path, cache_path


def apply_sam2_masks(
    cache_path: str,
    mask_dir: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Apply SAM2 masks to cached Pi3X results and save filtered point cloud.

    Args:
        cache_path: Path to pi3x_cache.npz from run_pi3x_inference().
        mask_dir: Directory of SAM2 mask PNGs.
        output_dir: Output directory.

    Returns:
        Path to the filtered PLY file (object.ply).
    """
    from pi3.utils.basic import write_ply

    output_path = Path(output_dir)
    mask_dir = Path(mask_dir)
    _emit_progress(progress_cb, 5.0, "Loading Pi3X cache")

    # Load cache
    cache = np.load(str(cache_path))
    points = cache["points"]  # (N, H, W, 3) float16
    conf_edge_mask = cache["conf_edge_mask"]  # (N, H, W) bool
    colors = cache["colors"]  # (N, H, W, 3) float
    N = int(cache["N"])
    H = int(cache["H"])
    W = int(cache["W"])
    if "frame_indices" in cache.files:
        frame_indices = cache["frame_indices"].astype(np.int64).tolist()
    elif "source_indices" in cache.files:
        frame_indices = cache["source_indices"].astype(np.int64).tolist()
    else:
        frame_indices = list(range(N))
    if len(frame_indices) != N:
        print(
            f"Warning: cache frame_indices length {len(frame_indices)} != N={N}. "
            "Falling back to 0..N-1."
        )
        frame_indices = list(range(N))
    _emit_progress(progress_cb, 15.0, f"Loaded cache for {N} frames")

    # Load SAM2 masks
    mask_files = sorted(mask_dir.glob("*.png"))
    if len(mask_files) < N:
        print(f"Warning: {len(mask_files)} masks available for {N} Pi3X frames.")

    sam2_masks = []
    missing_masks = 0
    emit_every = max(1, N // 40)
    for i, frame_idx in enumerate(frame_indices):
        mask_path = mask_dir / f"{int(frame_idx):05d}.png"
        if not mask_path.exists() and 0 <= int(frame_idx) < len(mask_files):
            # Backward-compat fallback: treat as positional index.
            mask_path = mask_files[int(frame_idx)]

        if not mask_path.exists():
            missing_masks += 1
            sam2_masks.append(np.zeros((H, W), dtype=bool))
        else:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                missing_masks += 1
                sam2_masks.append(np.zeros((H, W), dtype=bool))
            else:
                mask_resized = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
                sam2_masks.append(mask_resized > 127)
        if i % emit_every == 0 or i == N - 1:
            ratio = (i + 1) / max(N, 1)
            _emit_progress(
                progress_cb,
                15.0 + ratio * 55.0,
                f"Loading SAM2 masks ({i + 1}/{N})",
            )

    if missing_masks > 0:
        print(f"Warning: {missing_masks}/{N} masks missing or unreadable.")

    sam2_mask = np.stack(sam2_masks)
    final_mask = conf_edge_mask & sam2_mask
    final_count = final_mask.sum()
    _emit_progress(progress_cb, 78.0, "Applying SAM2 mask filter")

    total = N * H * W
    conf_edge_count = conf_edge_mask.sum()
    print("Filtering stats (+SAM2 mask):")
    print(f"  After conf+edge:  {conf_edge_count:>10,} ({100*conf_edge_count/total:.1f}%)")
    print(f"  After +SAM2 mask: {final_count:>10,} ({100*final_count/total:.1f}%)")

    ply_path = output_path / "object.ply"
    if final_count == 0:
        print("Warning: No points after filtering.")
        write_ply(torch.zeros(0, 3), torch.zeros(0, 3), str(ply_path))
        _emit_progress(progress_cb, 100.0, "Mask filtering complete (no points)")
        return ply_path

    # Extract points and colors using final mask
    pts = torch.from_numpy(points[final_mask].astype(np.float32))
    cols = torch.from_numpy(colors[final_mask].astype(np.float32))

    # Bounding box
    pts_np = pts.numpy()
    bbox_size = pts_np.max(axis=0) - pts_np.min(axis=0)
    print(f"  Bounding box (m): {bbox_size[0]:.3f} x {bbox_size[1]:.3f} x {bbox_size[2]:.3f}")

    write_ply(pts, cols, str(ply_path))
    print(f"Saved PLY: {ply_path} ({final_count:,} points)")
    _emit_progress(progress_cb, 100.0, "Mask filtering complete")

    return ply_path


def run_pi3x(
    frames_dir: str,
    mask_dir: str,
    output_dir: str,
    pixel_limit: int | None = None,
    max_frames: int | None = None,
    conf_threshold: float | None = None,
    edge_rtol: float | None = None,
    align_camera_plane: bool | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[Path, Path]:
    """Run Pi3X inference and save filtered point cloud + camera poses.

    Backward-compatible wrapper that calls run_pi3x_inference() followed
    by apply_sam2_masks().

    Args:
        frames_dir: Directory of JPEG frames.
        mask_dir: Directory of SAM2 mask PNGs.
        output_dir: Output directory.
        pixel_limit: Max pixels per frame for resize.
        max_frames: Maximum frames for Pi3X.
        conf_threshold: Confidence threshold for filtering.
        edge_rtol: Depth edge relative tolerance.
        align_camera_plane: If True, align camera-orbit plane to the default
            reference surface (XZ) by rotating world coordinates.

    Returns:
        Tuple of (ply_path, poses_path).
    """
    _ply_full, poses_path, cache_path = run_pi3x_inference(
        frames_dir, output_dir,
        pixel_limit=pixel_limit,
        max_frames=max_frames,
        conf_threshold=conf_threshold,
        edge_rtol=edge_rtol,
        align_camera_plane=align_camera_plane,
        progress_cb=progress_cb,
    )
    ply_path = apply_sam2_masks(cache_path, mask_dir, output_dir, progress_cb=progress_cb)
    return ply_path, poses_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pi3X reconstruction")
    parser.add_argument("frames_dir", help="Directory of JPEG frames")
    parser.add_argument("mask_dir", help="Directory of mask PNGs")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    run_pi3x(args.frames_dir, args.mask_dir, args.output_dir)
