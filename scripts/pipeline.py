#!/usr/bin/env python3
"""clip2mesh: Full 3D reconstruction pipeline orchestrator.

RGB video → Textured 3D mesh (OBJ)

Pipeline stages:
  1. Frame extraction (CPU)
  2. COLMAP SfM (GPU-accelerated feature matching)
  3. SAM2 interactive segmentation (GPU, Gradio UI)
  4. gs2mesh reconstruction (GPU — 3DGS + stereo depth + TSDF)
  5. Texture baking (GPU when available, CPU fallback)
"""

import argparse
import time
from pathlib import Path

from scripts.config_defaults import _VRAM_GATE_MIN_FREE_MB
from scripts.vram_utils import cleanup_pytorch_vram, ensure_vram_available, log_vram


def main():
    parser = argparse.ArgumentParser(
        description="RGB video → Textured 3D mesh (OBJ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py /data/input/video.mp4
  python pipeline.py /data/input/video.mp4 --output-dir /data/output
  python pipeline.py /data/input/video.mp4 --skip-to 3  # Resume from stage 3
        """,
    )
    parser.add_argument("video_path", help="Path to input video (.mp4)")
    parser.add_argument("--output-dir", default="/data/output", help="Output directory")
    parser.add_argument("--skip-to", type=int, default=1, choices=range(1, 6),
                        help="Skip to stage N (for resuming after interruption)")
    args = parser.parse_args()

    output_dir = args.output_dir
    skip_to = args.skip_to

    print("=" * 60)
    print("clip2mesh: RGB Video → Textured 3D Mesh")
    print("=" * 60)
    print(f"Input:  {args.video_path}")
    print(f"Output: {output_dir}")
    print()
    log_vram("startup")

    start_time = time.time()
    frames_dir = str(Path(output_dir) / "frames")
    mask_dir = str(Path(output_dir) / "masks")

    # =====================================================================
    # Stage 1: Frame Extraction
    # =====================================================================
    if skip_to <= 1:
        print("\n" + "=" * 60)
        print("Stage 1/5: Frame Extraction")
        print("=" * 60)
        from scripts.stage_extract_frames import extract_frames

        frames_dir = str(extract_frames(args.video_path, output_dir))
        print(f"  → {frames_dir}")

    # =====================================================================
    # Stage 2: COLMAP SfM
    # =====================================================================
    if skip_to <= 2:
        print("\n" + "=" * 60)
        print("Stage 2/5: COLMAP Structure-from-Motion")
        print("=" * 60)
        from scripts.stage_colmap_sfm import run_colmap_sfm

        poses_path, colmap_sparse_dir = run_colmap_sfm(frames_dir, output_dir)
        print(f"  → Poses: {poses_path}")
        print(f"  → Sparse: {colmap_sparse_dir}")
    else:
        poses_path = str(Path(output_dir) / "camera_poses.json")
        colmap_sparse_dir = str(Path(output_dir) / "colmap_sparse")

    # =====================================================================
    # Stage 3: SAM2 Interactive Segmentation
    # =====================================================================
    if skip_to <= 3:
        print("\n" + "=" * 60)
        print("Stage 3/5: SAM2 Interactive Segmentation")
        print("=" * 60)
        print(">>> Open http://localhost:7860 to select the object <<<")
        from scripts.stage_sam2_ui import run_sam2_interactive

        mask_dir = str(run_sam2_interactive(frames_dir, output_dir))
        cleanup_pytorch_vram()
        print(f"  → {mask_dir}")

    # =====================================================================
    # Stage 4: gs2mesh Reconstruction
    # =====================================================================
    if skip_to <= 4:
        print("\n" + "=" * 60)
        print("Stage 4/5: gs2mesh Reconstruction (3DGS + Stereo + TSDF)")
        print("=" * 60)
        ensure_vram_available(min_free_mb=_VRAM_GATE_MIN_FREE_MB, stage_name="before gs2mesh")

        from scripts.stage_gs2mesh_reconstruct import run_gs2mesh

        mesh_ply = run_gs2mesh(
            frames_dir,
            colmap_sparse_dir,
            mask_dir,
            output_dir,
        )
        cleanup_pytorch_vram()
        print(f"  → Mesh: {mesh_ply}")
    else:
        mesh_ply = str(Path(output_dir) / "object_mesh.ply")

    # =====================================================================
    # Stage 5: Texture Baking
    # =====================================================================
    if skip_to <= 5:
        print("\n" + "=" * 60)
        print("Stage 5/5: Texture Baking")
        print("=" * 60)
        from scripts.stage_texture_bake import bake_texture

        obj_path = bake_texture(
            str(mesh_ply), str(poses_path), frames_dir, mask_dir, output_dir,
        )
        print(f"  → {obj_path}")

    # =====================================================================
    # Summary
    # =====================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print()

    out = Path(output_dir)
    for name in ["textured_mesh.obj", "textured_mesh.mtl", "texture.png",
                  "object_mesh.ply",
                  "camera_poses.json", "intrinsics.json"]:
        p = out / name
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  {name}: {size_mb:.1f} MB")

    print()
    print("View the textured mesh:")
    print(f"  {out / 'textured_mesh.obj'}")


if __name__ == "__main__":
    main()
