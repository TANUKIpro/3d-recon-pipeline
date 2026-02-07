#!/usr/bin/env python3
"""im2pc-pipeline: Full 3D reconstruction pipeline orchestrator.

RGB video → Textured 3D mesh (OBJ)

Pipeline stages:
  1. Frame extraction (CPU)
  2. SAM2 interactive segmentation (GPU, Gradio UI)
  3. Pi3X 3D reconstruction + triple filtering (GPU)
  4. Point cloud denoising (CPU)
  5. DiffCD mesh reconstruction (GPU/JAX subprocess)
  6. Texture baking (CPU)
"""

import argparse
import sys
import time
from pathlib import Path

# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from vram_utils import cleanup_pytorch_vram, ensure_vram_available, log_vram


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
    parser.add_argument("--skip-to", type=int, default=1, choices=range(1, 7),
                        help="Skip to stage N (for resuming after interruption)")
    args = parser.parse_args()

    output_dir = args.output_dir
    skip_to = args.skip_to

    print("=" * 60)
    print("im2pc-pipeline: RGB Video → Textured 3D Mesh")
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
        print("Stage 1/6: Frame Extraction")
        print("=" * 60)
        from stage_extract_frames import extract_frames

        frames_dir = str(extract_frames(args.video_path, output_dir))
        print(f"  → {frames_dir}")

    # =====================================================================
    # Stage 2: SAM2 Interactive Segmentation
    # =====================================================================
    if skip_to <= 2:
        print("\n" + "=" * 60)
        print("Stage 2/6: SAM2 Interactive Segmentation")
        print("=" * 60)
        print(">>> Open http://localhost:7860 to select the object <<<")
        from stage_sam2_ui import run_sam2_interactive

        mask_dir = str(run_sam2_interactive(frames_dir, output_dir))
        cleanup_pytorch_vram()
        print(f"  → {mask_dir}")

    # VRAM gate: ensure sufficient free VRAM before Pi3X (only after SAM2 ran)
    if skip_to <= 2:
        ensure_vram_available(min_free_mb=12000, stage_name="before Pi3X")

    # =====================================================================
    # Stage 3: Pi3X 3D Reconstruction
    # =====================================================================
    if skip_to <= 3:
        print("\n" + "=" * 60)
        print("Stage 3/6: Pi3X 3D Reconstruction")
        print("=" * 60)
        from stage_pi3x_reconstruct import run_pi3x

        ply_path, poses_path = run_pi3x(frames_dir, mask_dir, output_dir)
        cleanup_pytorch_vram()
        print(f"  → PLY: {ply_path}")
        print(f"  → Poses: {poses_path}")
    else:
        ply_path = Path(output_dir) / "object.ply"
        poses_path = Path(output_dir) / "camera_poses.json"

    # =====================================================================
    # Stage 4: Point Cloud Denoising
    # =====================================================================
    if skip_to <= 4:
        print("\n" + "=" * 60)
        print("Stage 4/6: Point Cloud Denoising")
        print("=" * 60)
        from stage_denoise import denoise

        denoised_ply = denoise(str(ply_path), output_dir)
        print(f"  → {denoised_ply}")
    else:
        denoised_ply = Path(output_dir) / "object_denoised.ply"

    # =====================================================================
    # Stage 5: DiffCD Mesh Reconstruction
    # =====================================================================
    if skip_to <= 5:
        print("\n" + "=" * 60)
        print("Stage 5/6: DiffCD Mesh Reconstruction")
        print("=" * 60)
        from stage_diffcd_mesh import run_diffcd

        mesh_ply = run_diffcd(str(denoised_ply), output_dir)
        print(f"  → {mesh_ply}")
    else:
        mesh_ply = Path(output_dir) / "object_mesh.ply"

    # =====================================================================
    # Stage 6: Texture Baking
    # =====================================================================
    if skip_to <= 6:
        print("\n" + "=" * 60)
        print("Stage 6/6: Texture Baking")
        print("=" * 60)
        from stage_texture_bake import bake_texture

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
                  "object_mesh.ply", "object_denoised.ply", "object.ply",
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
