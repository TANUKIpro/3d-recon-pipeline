"""Stage 3: Pi3X 3D reconstruction with triple filtering.

Runs Pi3X on full (unmasked) images for optimal pose estimation,
then applies confidence + depth-edge + SAM2 mask filtering.

Adapted from im2pc/host/pi3x_sam2_cli.py.
"""

import json
import os
from pathlib import Path

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

# (effective_free_vram_mb_lower, pixel_limit, max_frames)
_VRAM_PROFILES = [
    (14000, 255000, 50),
    (12000, 200000, 35),
    (10000, 150000, 20),
    ( 8000, 120000, 15),
    (    0,  80000, 10),
]


def _auto_scale_params(
    pixel_limit: int, max_frames: int
) -> tuple[int, int]:
    """Scale down pixel_limit/max_frames based on available VRAM.

    Subtracts estimated model weight size from free VRAM before profile
    selection, since this function is called before model loading.
    Only reduces values (never increases beyond the caller's request).
    """
    free = get_free_vram_mb()
    if free is None:
        return pixel_limit, max_frames

    effective_free = free - _ESTIMATED_MODEL_MB
    print(
        f"VRAM auto-scale: {free}MB free, "
        f"{effective_free}MB effective (minus ~{_ESTIMATED_MODEL_MB}MB model)"
    )

    for threshold, prof_pixel, prof_frames in _VRAM_PROFILES:
        if effective_free >= threshold:
            new_pixel = min(pixel_limit, prof_pixel)
            new_frames = min(max_frames, prof_frames)
            if new_pixel != pixel_limit or new_frames != max_frames:
                print(
                    f"VRAM auto-scale: effective {effective_free}MB → "
                    f"PIXEL_LIMIT {pixel_limit}→{new_pixel}, "
                    f"MAX_FRAMES {max_frames}→{new_frames}"
                )
            else:
                print(
                    f"VRAM auto-scale: effective {effective_free}MB → "
                    f"no adjustment needed"
                )
            return new_pixel, new_frames

    return pixel_limit, max_frames


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

    for start in range(0, N, chunk_size - overlap):
        end = min(start + chunk_size, N)
        if end - start < 2:
            break
        print(f"  Processing chunk [{start}:{end}] ({end - start} frames)")

        # Reload full model to GPU for this chunk
        model.to(device)
        chunk_results = _run_inference_memory_efficient(
            model, imgs[start:end], device, dtype,
        )
        all_chunks.append((start, end, chunk_results))

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

        # Procrustes: find R, t, s such that s*R@cur + t ≈ ref
        R, t, s = _procrustes(cur_poses, ref_poses)

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


class _nullcontext:
    """Minimal no-op context manager (for Python <3.10 compat)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def run_pi3x(
    frames_dir: str,
    mask_dir: str,
    output_dir: str,
    pixel_limit: int | None = None,
    max_frames: int | None = None,
    conf_threshold: float | None = None,
    edge_rtol: float | None = None,
) -> tuple[Path, Path]:
    """Run Pi3X inference and save filtered point cloud + camera poses.

    Args:
        frames_dir: Directory of JPEG frames.
        mask_dir: Directory of SAM2 mask PNGs.
        output_dir: Output directory.
        pixel_limit: Max pixels per frame for resize.
        max_frames: Maximum frames for Pi3X.
        conf_threshold: Confidence threshold for filtering.
        edge_rtol: Depth edge relative tolerance.

    Returns:
        Tuple of (ply_path, poses_path).
    """
    if pixel_limit is None:
        pixel_limit = int(os.environ.get("PIXEL_LIMIT", "255000"))
    if max_frames is None:
        max_frames = int(os.environ.get("MAX_FRAMES", "50"))
    if conf_threshold is None:
        conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.1"))
    if edge_rtol is None:
        edge_rtol = float(os.environ.get("EDGE_RTOL", "0.03"))

    pixel_limit, max_frames = _auto_scale_params(pixel_limit, max_frames)

    from pi3.utils.basic import load_images_as_tensor, write_ply
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import depth_edge

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(mask_dir)

    # --- Load frames ---
    log_vram("before Pi3X load")
    imgs = load_images_as_tensor(
        str(frames_dir), interval=1, PIXEL_LIMIT=pixel_limit,
    )

    if imgs.numel() == 0 or imgs.shape[0] < 2:
        raise RuntimeError("Need at least 2 frames for reconstruction.")
    if imgs.shape[0] > max_frames:
        print(f"Limiting to {max_frames} frames (from {imgs.shape[0]})")
        imgs = imgs[:max_frames]

    N, _, H, W = imgs.shape
    print(f"Pi3X input: {N} frames, {H}x{W} pixels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = imgs.to(device)

    # --- Load and run model ---
    print("Loading Pi3X model...")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    log_vram("after Pi3X load")

    print(f"Running Pi3X inference on {N} frames...")
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if capability >= 8 else torch.float16
    else:
        dtype = None

    results = _run_inference_memory_efficient(model, imgs, device, dtype)

    print("Pi3X inference complete")
    log_vram("after Pi3X inference")

    # --- Release model before filtering ---
    del model
    cleanup_pytorch_vram()

    # --- Triple filter ---
    # Filter 1: Confidence
    conf_mask = torch.sigmoid(results["conf"][..., 0]) > conf_threshold
    conf_count = conf_mask[0].sum().item()

    # Filter 2: Depth edge
    non_edge = ~depth_edge(results["local_points"][..., 2], rtol=edge_rtol)
    conf_edge_mask = conf_mask & non_edge
    conf_edge_count = conf_edge_mask[0].sum().item()

    # Filter 3: SAM2 masks
    mask_files = sorted(mask_dir.glob("*.png"))
    if len(mask_files) != N:
        print(f"Warning: {len(mask_files)} masks vs {N} frames. Using min.")

    sam2_masks = []
    for i in range(N):
        if i < len(mask_files):
            mask_img = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
            sam2_masks.append(mask_resized > 127)
        else:
            sam2_masks.append(np.zeros((H, W), dtype=bool))

    sam2_mask_tensor = torch.from_numpy(np.stack(sam2_masks)).to(device)
    final_mask = conf_edge_mask[0] & sam2_mask_tensor
    final_count = final_mask.sum().item()

    # Stats
    total = N * H * W
    print("Filtering stats:")
    print(f"  Total pixels:     {total:>10,}")
    print(f"  After conf:       {conf_count:>10,} ({100*conf_count/total:.1f}%)")
    print(f"  After conf+edge:  {conf_edge_count:>10,} ({100*conf_edge_count/total:.1f}%)")
    print(f"  After +SAM2 mask: {final_count:>10,} ({100*final_count/total:.1f}%)")

    if final_count == 0:
        print("Warning: No points after filtering.")
        ply_path = output_path / "object.ply"
        write_ply(torch.zeros(0, 3), torch.zeros(0, 3), str(ply_path))
        poses_path = output_path / "camera_poses.json"
        with open(poses_path, "w") as f:
            json.dump({"poses": [], "frame_indices": []}, f)
        return ply_path, poses_path

    # Extract points and colors
    points = results["points"][0][final_mask].cpu()
    colors = imgs.permute(0, 2, 3, 1)[final_mask]

    # Bounding box
    pts_np = points.numpy()
    bbox_size = pts_np.max(axis=0) - pts_np.min(axis=0)
    print(f"  Bounding box (m): {bbox_size[0]:.3f} x {bbox_size[1]:.3f} x {bbox_size[2]:.3f}")

    # Save PLY
    ply_path = output_path / "object.ply"
    write_ply(points, colors, str(ply_path))
    print(f"Saved PLY: {ply_path} ({final_count:,} points)")

    # Save camera poses
    poses = results["camera_poses"][0].cpu().numpy()
    poses_path = output_path / "camera_poses.json"
    with open(poses_path, "w") as f:
        json.dump(
            {"poses": [pose.tolist() for pose in poses], "frame_indices": list(range(N))},
            f, indent=2,
        )
    print(f"Saved camera poses: {poses_path}")

    # Cleanup remaining GPU tensors
    del results, imgs, conf_mask, non_edge, conf_edge_mask, sam2_mask_tensor, final_mask
    cleanup_pytorch_vram()

    return ply_path, poses_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pi3X reconstruction")
    parser.add_argument("frames_dir", help="Directory of JPEG frames")
    parser.add_argument("mask_dir", help="Directory of mask PNGs")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    run_pi3x(args.frames_dir, args.mask_dir, args.output_dir)
