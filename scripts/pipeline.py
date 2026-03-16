#!/usr/bin/env python3
"""clip2mesh: Full 3D reconstruction pipeline orchestrator.

RGB video → Textured 3D mesh (OBJ)

Pipeline stages:
  1. Frame extraction (CPU)
  2. COLMAP SfM (GPU-accelerated feature matching)
  3. SAM2 interactive segmentation (GPU, Gradio UI)
  4. gs2mesh reconstruction (GPU — 3DGS + stereo depth + TSDF)
  5. Texture baking (GPU when available, CPU fallback)
  6. Post-texture contact cleanup
"""

import argparse
import json
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
  python pipeline.py /data/input/video.mp4 --post-texture-cleanup-selection-json cleanup.json
        """,
    )
    parser.add_argument("video_path", help="Path to input video (.mp4)")
    parser.add_argument("--output-dir", default="/data/output", help="Output directory")
    parser.add_argument("--skip-to", type=int, default=1, choices=range(1, 7),
                        help="Skip to stage N (for resuming after interruption)")
    parser.add_argument(
        "--post-texture-cleanup-selection-json",
        default="",
        help="JSON file containing {\"decision\": \"apply\"|\"skip\"} for Stage 6 review",
    )
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
        print("Stage 1/6: Frame Extraction")
        print("=" * 60)
        from scripts.stage_extract_frames import extract_frames

        frames_dir = str(extract_frames(args.video_path, output_dir))
        print(f"  → {frames_dir}")

    # =====================================================================
    # Stage 2: COLMAP SfM
    # =====================================================================
    if skip_to <= 2:
        print("\n" + "=" * 60)
        print("Stage 2/6: COLMAP Structure-from-Motion")
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
        print("Stage 3/6: SAM2 Interactive Segmentation")
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
        print("Stage 4/6: gs2mesh Reconstruction (3DGS + Stereo + TSDF)")
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
        print("Stage 5/6: Texture Baking")
        print("=" * 60)
        from scripts.stage_texture_bake import bake_texture

        obj_path = bake_texture(
            str(mesh_ply), str(poses_path), frames_dir, mask_dir, output_dir,
        )
        print(f"  → {obj_path}")
    else:
        obj_path = str(Path(output_dir) / "textured_mesh.obj")

    # =====================================================================
    # Stage 6: Post-texture Cleanup
    # =====================================================================
    if skip_to <= 6:
        print("\n" + "=" * 60)
        print("Stage 6/6: Post-texture Contact Cleanup")
        print("=" * 60)
        from scripts.stage_post_texture_contact_cleanup import (
            apply_cleanup_proposal,
            copy_stage5_as_cleaned,
            prepare_cleanup_review,
        )

        ground_plane_path = Path(output_dir) / "ground_plane.json"
        if not ground_plane_path.is_file():
            ground_masks_dir = Path(output_dir) / "masks_ground"
            if ground_masks_dir.is_dir() and any(ground_masks_dir.glob("*.png")):
                from scripts.ground_plane_extraction import extract_ground_plane_from_mesh
                print("  Extracting ground plane from mesh + ground masks...")
                extract_ground_plane_from_mesh(
                    str(Path(output_dir) / "object_mesh.ply"),
                    str(ground_masks_dir),
                    output_dir,
                    str(poses_path),
                    str(Path(output_dir) / "intrinsics.json"),
                    object_mask_dir=mask_dir,
                )
        proposal = prepare_cleanup_review(
            str(obj_path),
            output_dir,
            ground_plane_path=str(ground_plane_path) if ground_plane_path.is_file() else None,
            poses_path=str(poses_path),
            intrinsics_path=str(Path(output_dir) / "intrinsics.json"),
            masks_dir=mask_dir,
            ground_masks_dir=(
                str(Path(output_dir) / "masks_ground")
                if (Path(output_dir) / "masks_ground").is_dir()
                else None
            ),
        )
        decision = str(proposal.get("recommended_decision") or "skip").strip().lower()
        selection_json = str(args.post_texture_cleanup_selection_json or "").strip()
        if selection_json:
            selection_path = Path(selection_json)
            raw_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            decision = str(raw_selection.get("decision") or "").strip().lower()
            if decision not in {"apply", "skip"}:
                raise ValueError(
                    "post-texture cleanup selection JSON must contain "
                    '{"decision": "apply"|"skip"}'
                )
        elif proposal.get("requires_review") is True:
            proposal_path = Path(output_dir) / "post_texture_contact_cleanup" / "proposal.json"
            print(f"  proposal written: {proposal_path}")
            print("  no selection JSON provided; defaulting to skip")
            decision = "skip"

        if decision == "apply":
            cleaned_obj_path = Path(apply_cleanup_proposal(output_dir))
        else:
            cleaned_obj_path = Path(copy_stage5_as_cleaned(output_dir))
        print(f"  → Decision: {decision}")
        print(f"  → {cleaned_obj_path}")
        if decision == "apply":
            print(f"  → Cap texture: {Path(output_dir) / 'texture_cap.png'}")
    else:
        cleaned_obj_path = Path(output_dir) / "textured_mesh_cleaned.obj"

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
    for name in ["textured_mesh.obj", "textured_mesh.mtl", "textured_mesh_cleaned.obj", "textured_mesh_cleaned.mtl",
                  "texture.png", "texture_cap.png",
                  "object_mesh.ply",
                  "camera_poses.json", "intrinsics.json"]:
        p = out / name
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  {name}: {size_mb:.1f} MB")

    print()
    print("View the textured mesh:")
    final_mesh = cleaned_obj_path if cleaned_obj_path.exists() else out / "textured_mesh.obj"
    print(f"  {final_mesh}")


if __name__ == "__main__":
    main()
