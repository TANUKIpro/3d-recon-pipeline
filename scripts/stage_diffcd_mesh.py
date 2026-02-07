"""Stage 5: DiffCD mesh reconstruction via subprocess.

Converts denoised PLY to NPY, runs DiffCD fit_implicit.py as a subprocess
(to avoid PyTorch/JAX CUDA context conflicts), then post-processes the mesh.

Based on im2pc/colab_diffcd.md and tmp/DiffCD.ipynb patterns.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from vram_utils import prepare_for_jax, log_vram

ProgressCallback = Callable[[float, str | None], None]
_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_ITER_FRACTION_RE = re.compile(
    r"(?:batch|iter|step|epoch)[^\d]{0,16}(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
_GENERIC_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


def _parse_progress_ratio(line: str) -> float | None:
    """Parse DiffCD training progress from a log line."""
    m = _TQDM_PERCENT_RE.search(line)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return pct / 100.0

    m = _ITER_FRACTION_RE.search(line)
    if not m:
        m = _GENERIC_FRACTION_RE.search(line)
    if not m:
        return None

    current = int(m.group(1))
    total = int(m.group(2))
    if total <= 0:
        return None
    ratio = current / total
    if ratio < 0 or ratio > 1.5:
        return None
    return min(1.0, max(0.0, ratio))


def _downsample_to_npy(ply_path: str, npy_path: str, target_points: int = 1_000_000) -> Path:
    """Convert PLY to downsampled NPY (xyz float32) for DiffCD.

    Args:
        ply_path: Input PLY file.
        npy_path: Output NPY path.
        target_points: Target number of points after downsampling.

    Returns:
        Path to the NPY file.
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_original = len(pcd.points)
    print(f"Original: {n_original:,} points")

    if n_original > target_points:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        volume = np.prod(bbox_extent)
        voxel_size = (volume / target_points) ** (1 / 3) * 0.9

        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Iterative adjustment to get within ±10% of target
        for _ in range(10):
            n = len(pcd_down.points)
            if target_points * 0.9 <= n <= target_points * 1.1:
                break
            if n > target_points * 1.1:
                voxel_size *= 1.1
            else:
                voxel_size *= 0.9
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        print(f"Downsampled: {len(pcd_down.points):,} points (voxel_size={voxel_size:.5f})")
    else:
        pcd_down = pcd
        print(f"No downsampling needed: {len(pcd_down.points):,} points")

    points = np.asarray(pcd_down.points, dtype=np.float32)
    np.save(str(npy_path), points)
    print(f"Saved: {npy_path} ({points.shape})")
    return Path(npy_path)


def run_diffcd(
    denoised_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run DiffCD mesh reconstruction as a subprocess.

    Args:
        denoised_ply: Path to denoised PLY file.
        output_dir: Output directory.

    Returns:
        Path to the final mesh PLY file.
    """
    batch_size = int(os.environ.get("DIFFCD_BATCH_SIZE", "3000"))
    n_batches = int(os.environ.get("DIFFCD_N_BATCHES", "25000"))
    resolution = int(os.environ.get("DIFFCD_RESOLUTION", "384"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    diffcd_dir = output_path / "diffcd"
    diffcd_dir.mkdir(exist_ok=True)

    # Step 1: Convert PLY to NPY
    npy_path = output_path / "object_points.npy"
    print("=== DiffCD: Preparing point cloud ===")
    _emit_progress(progress_cb, 5.0, "Preparing point cloud for DiffCD")
    _downsample_to_npy(denoised_ply, str(npy_path))
    _emit_progress(progress_cb, 15.0, "Point cloud prepared")

    # Step 2: Run DiffCD as subprocess
    print("\n=== DiffCD: Running implicit surface fitting ===")
    _emit_progress(progress_cb, 20.0, "Running DiffCD fitting")
    prepare_for_jax()
    log_vram("before DiffCD")

    cmd = [
        sys.executable, "/opt/diffcd/fit_implicit.py",
        "--output-dir", str(diffcd_dir),
        f"--dataset.path", str(npy_path),
        "--batch-size", str(batch_size),
        "--n-batches", str(n_batches),
        "--final-mesh-points-per-axis", str(resolution),
    ]

    print(f"Running: {' '.join(cmd)}")
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
    }

    process = subprocess.Popen(
        cmd,
        env=env,
        cwd="/opt/diffcd",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_reported = 20.0
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            ratio = _parse_progress_ratio(line)
            if ratio is None:
                continue
            stage_pct = 20.0 + ratio * 65.0
            if stage_pct - last_reported >= 0.5:
                _emit_progress(
                    progress_cb,
                    stage_pct,
                    f"DiffCD fitting ({int(ratio * 100)}%)",
                )
                last_reported = stage_pct

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"DiffCD failed with return code {return_code}")

    log_vram("after DiffCD")
    _emit_progress(progress_cb, 87.0, "Collecting DiffCD outputs")

    # Step 3: Find output mesh
    # DiffCD puts output in a timestamped subdir: {out}/experiment.../meshes/mesh_N.ply
    # and also {out}/experiment.../mesh_final_N.ply
    mesh_files = sorted(diffcd_dir.rglob("mesh_final_*.ply"))
    if not mesh_files:
        mesh_files = sorted(diffcd_dir.rglob("mesh_final.ply"))
    if not mesh_files:
        mesh_files = sorted(diffcd_dir.rglob("meshes/mesh_*.ply"))
    if not mesh_files:
        raise FileNotFoundError(f"DiffCD output mesh not found in {diffcd_dir}")

    raw_mesh_path = mesh_files[-1]
    print(f"\nDiffCD output: {raw_mesh_path}")

    # Step 4: Post-process (Laplacian smoothing)
    print("Applying Laplacian smoothing...")
    _emit_progress(progress_cb, 94.0, "Applying mesh smoothing")
    mesh = trimesh.load(str(raw_mesh_path))
    print(f"  Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.faces):,}")

    mesh_smooth = trimesh.smoothing.filter_laplacian(mesh, iterations=2)

    final_path = output_path / "object_mesh.ply"
    mesh_smooth.export(str(final_path))
    print(f"Saved: {final_path}")
    print(f"  Vertices: {len(mesh_smooth.vertices):,}, Faces: {len(mesh_smooth.faces):,}")
    _emit_progress(progress_cb, 100.0, "DiffCD stage complete")

    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiffCD mesh reconstruction")
    parser.add_argument("input_ply", help="Denoised PLY file")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    run_diffcd(args.input_ply, args.output_dir)
