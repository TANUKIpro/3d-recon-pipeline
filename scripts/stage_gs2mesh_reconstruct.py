"""Stage 4: gs2mesh Reconstruction.

Runs gs2mesh pipeline: 3DGS training → stereo depth → TSDF fusion → mesh.
Takes COLMAP output + optional SAM2 masks, produces object_mesh.ply.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def run_gs2mesh(
    frames_dir: str,
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: str,
    gs_iterations: int = 30000,
    stereo_model: str = "DLNR",
    tsdf_voxel_size: float = 0.005,
    tsdf_depth_trunc: float = 0.04,
    use_masks: bool = True,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    """gs2mesh pipeline: 3DGS training → stereo depth → TSDF → mesh.

    Returns path to object_mesh.ply.
    """
    out = Path(output_dir)
    gs2mesh_workdir = out / "gs2mesh_workspace"
    gs2mesh_workdir.mkdir(parents=True, exist_ok=True)

    def _report(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if cancel_cb:
            cancel_cb()

    # Find the COLMAP reconstruction subdir (0/, 1/, etc.)
    colmap_sparse = Path(colmap_sparse_dir)
    recon_dir = None
    for d in sorted(colmap_sparse.iterdir()):
        if d.is_dir() and (d / "images.bin").exists():
            recon_dir = d
            break
    if recon_dir is None:
        raise RuntimeError("No valid COLMAP reconstruction found in " + colmap_sparse_dir)

    # Undistort images: 3DGS only supports PINHOLE/SIMPLE_PINHOLE cameras,
    # but COLMAP defaults to SIMPLE_RADIAL.
    _report(2.0, "Undistorting images for 3DGS")
    undistorted_dir = gs2mesh_workdir / "undistorted"
    _run_undistort(
        frames_dir, recon_dir, undistorted_dir,
        register_process, unregister_process,
    )

    # Step 1: Train 3D Gaussian Splatting model
    _report(5.0, "Training 3D Gaussian Splatting model")
    gs_model_dir = gs2mesh_workdir / "gs_model"
    gs_model_dir.mkdir(parents=True, exist_ok=True)

    train_args = [
        "python3", "-u", "/opt/gaussian-splatting/train.py",
        "--source_path", str(out),
        "--model_path", str(gs_model_dir),
        "--iterations", str(gs_iterations),
    ]

    # Prepare COLMAP-compatible directory structure for gaussian-splatting
    # using undistorted images and PINHOLE camera model
    _prepare_gs_input(out, undistorted_dir / "images", undistorted_dir / "sparse" / "0")

    proc = subprocess.Popen(
        train_args,
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
                print(f"  [3DGS] {line}")
                # Parse iteration progress
                match = re.search(r"iteration\s+(\d+)/(\d+)", line, re.IGNORECASE)
                if not match:
                    match = re.search(r"\[ITER\s+(\d+)\]", line, re.IGNORECASE)
                if match:
                    current = int(match.group(1))
                    total = gs_iterations
                    pct = 5.0 + (current / total) * 45.0  # 5% -> 50%
                    _report(pct, f"3DGS training ({current}/{total})")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"3DGS training failed (exit {proc.returncode})")
    finally:
        if unregister_process:
            unregister_process(proc)

    if cancel_cb:
        cancel_cb()

    # Step 2: Run gs2mesh reconstruction (stereo + TSDF)
    _report(55.0, "Running gs2mesh stereo depth + TSDF fusion")

    gs2mesh_args = [
        "python3", "-u", "/opt/gs2mesh/run.py",
        "--gs_model_path", str(gs_model_dir),
        "--colmap_path", str(undistorted_dir / "sparse"),
        "--output_path", str(gs2mesh_workdir / "mesh_output"),
        "--stereo_model", stereo_model,
        "--tsdf_voxel_size", str(tsdf_voxel_size),
        "--tsdf_depth_trunc", str(tsdf_depth_trunc),
    ]

    if use_masks and mask_dir and Path(mask_dir).is_dir():
        gs2mesh_args.extend(["--mask_path", mask_dir])

    proc = subprocess.Popen(
        gs2mesh_args,
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
                print(f"  [gs2mesh] {line}")
                # Parse progress from gs2mesh output
                match = re.search(r"(\d+)%", line)
                if match:
                    pct = 55.0 + int(match.group(1)) * 0.40  # 55% -> 95%
                    _report(pct, line[:100])
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"gs2mesh reconstruction failed (exit {proc.returncode})")
    finally:
        if unregister_process:
            unregister_process(proc)

    # Step 3: Find and copy output mesh
    _report(95.0, "Collecting output mesh")
    mesh_output_dir = gs2mesh_workdir / "mesh_output"
    output_mesh = _find_output_mesh(mesh_output_dir)
    if output_mesh is None:
        raise RuntimeError(f"No mesh output found in {mesh_output_dir}")

    final_mesh = out / "object_mesh.ply"
    import shutil
    shutil.copy2(str(output_mesh), str(final_mesh))
    print(f"gs2mesh output mesh: {final_mesh}")

    _report(100.0, "gs2mesh reconstruction complete")
    return str(final_mesh)


def _prepare_gs_input(output_dir: Path, frames_dir: Path, recon_dir: Path) -> None:
    """Create COLMAP-compatible directory structure for gaussian-splatting.

    gaussian-splatting expects:
      source_path/sparse/0/  — COLMAP binary files
      source_path/images/    — input images
    """
    import shutil

    # Create sparse/0/ with symlinks to COLMAP files
    sparse_target = output_dir / "sparse" / "0"
    sparse_target.mkdir(parents=True, exist_ok=True)
    for f in recon_dir.iterdir():
        if f.is_file():
            shutil.copy2(str(f), str(sparse_target / f.name))

    # Create images/ symlink to frames directory
    images_link = output_dir / "images"
    if images_link.is_symlink():
        images_link.unlink()
    if not images_link.exists():
        images_link.symlink_to(frames_dir.resolve())


def _run_undistort(
    frames_dir: str,
    recon_dir: Path,
    output_dir: Path,
    register_process=None,
    unregister_process=None,
) -> None:
    """Run COLMAP image_undistorter to convert camera model to PINHOLE.

    3DGS only supports PINHOLE/SIMPLE_PINHOLE cameras. COLMAP defaults to
    SIMPLE_RADIAL, so we undistort images and convert the camera model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(frames_dir),
        "--input_path", str(recon_dir),
        "--output_path", str(output_dir),
        "--output_type", "COLMAP",
    ]
    print(f"COLMAP image_undistorter: {' '.join(cmd)}")
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
                f"COLMAP image_undistorter failed (exit {proc.returncode})"
            )
    finally:
        if unregister_process:
            unregister_process(proc)

    # Restructure sparse/ → sparse/0/ to match COLMAP convention.
    # The undistorter outputs files directly in sparse/, but 3DGS and
    # gs2mesh expect the sparse/0/ subdirectory structure.
    sparse_dir = output_dir / "sparse"
    if sparse_dir.is_dir() and not (sparse_dir / "0").is_dir():
        tmp = output_dir / "_sparse_tmp"
        sparse_dir.rename(tmp)
        sparse_dir.mkdir()
        tmp.rename(sparse_dir / "0")


def _find_output_mesh(mesh_dir: Path) -> Path | None:
    """Find the mesh PLY/OBJ in gs2mesh output directory."""
    if not mesh_dir.is_dir():
        return None

    # gs2mesh typically outputs mesh.ply or mesh.obj
    for pattern in ["*.ply", "**/*.ply", "*.obj", "**/*.obj"]:
        matches = list(mesh_dir.glob(pattern))
        if matches:
            # Prefer the largest file (most likely the full mesh)
            return max(matches, key=lambda p: p.stat().st_size)

    return None
