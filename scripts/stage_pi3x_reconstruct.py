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

from vram_utils import cleanup_pytorch_vram, get_free_vram_mb, log_vram

# (free_vram_mb_lower, pixel_limit, max_frames)
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

    Only reduces values (never increases beyond the caller's request).
    """
    free = get_free_vram_mb()
    if free is None:
        return pixel_limit, max_frames

    for threshold, prof_pixel, prof_frames in _VRAM_PROFILES:
        if free >= threshold:
            new_pixel = min(pixel_limit, prof_pixel)
            new_frames = min(max_frames, prof_frames)
            if new_pixel != pixel_limit or new_frames != max_frames:
                print(
                    f"VRAM auto-scale: {free}MB free → "
                    f"PIXEL_LIMIT {pixel_limit}→{new_pixel}, "
                    f"MAX_FRAMES {max_frames}→{new_frames}"
                )
            else:
                print(f"VRAM auto-scale: {free}MB free → no adjustment needed")
            return new_pixel, new_frames

    return pixel_limit, max_frames


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
    with torch.no_grad():
        if device.type == "cuda":
            capability = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if capability >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                results = model(imgs[None])
        else:
            results = model(imgs[None])

    print("Pi3X inference complete")
    log_vram("after Pi3X inference")

    # --- Release model before filtering ---
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

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
