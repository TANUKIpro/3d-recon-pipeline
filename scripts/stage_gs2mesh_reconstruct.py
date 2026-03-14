"""Stage 4: gs2mesh Reconstruction.

Runs gs2mesh pipeline: 3DGS training → stereo depth → TSDF fusion → mesh.
Takes COLMAP output + optional SAM2 masks, produces object_mesh.ply.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

_GS2MESH_BASE = Path("/opt/gs2mesh")


def run_gs2mesh(
    frames_dir: str,
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: str,
    gs_iterations: int = 30000,
    stereo_model: str = "DLNR_Middlebury",
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

    # Map shortened stereo model names for backward compatibility
    if stereo_model == "DLNR":
        stereo_model = "DLNR_Middlebury"

    def _report(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if cancel_cb:
            cancel_cb()

    # Find the COLMAP reconstruction subdir (0/, 1/, etc.)
    recon_dir = _find_recon_dir(colmap_sparse_dir)

    # Step 0a: Undistort images — 3DGS requires PINHOLE cameras,
    # but COLMAP defaults to SIMPLE_RADIAL.
    _report(1.0, "Undistorting images for 3DGS")
    undistorted_dir = gs2mesh_workdir / "undistorted"
    _run_colmap_cmd(
        ["colmap", "image_undistorter",
         "--image_path", str(frames_dir),
         "--input_path", str(recon_dir),
         "--output_path", str(undistorted_dir),
         "--output_type", "COLMAP"],
        "image_undistorter", undistorted_dir,
        register_process, unregister_process,
    )
    _ensure_sparse_0(undistorted_dir / "sparse")

    # Step 0b: Convert binary COLMAP model → text format.
    # gs2mesh's Renderer reads images.txt / cameras.txt, not binary files.
    _report(2.0, "Converting COLMAP model to text format")
    undist_model = undistorted_dir / "sparse" / "0"
    _run_colmap_cmd(
        ["colmap", "model_converter",
         "--input_path", str(undist_model),
         "--output_path", str(undist_model),
         "--output_type", "TXT"],
        "model_converter (binary→text)", None,
        register_process, unregister_process,
    )

    # Step 0c: Set up gs2mesh data directory with symlinks to undistorted output.
    # gs2mesh expects: {base}/data/custom/{scene}/images/ + sparse/0/
    scene_name = out.name
    gs2mesh_data = _GS2MESH_BASE / "data" / "custom" / scene_name
    _setup_gs2mesh_dirs(gs2mesh_data, undistorted_dir)

    # Step 1: Train 3D Gaussian Splatting model.
    # We train separately (rather than letting gs2mesh call os.system) for
    # better progress tracking and error handling.
    _report(5.0, "Training 3D Gaussian Splatting model")
    splatting_string = f"custom_nw_iterations{gs_iterations}"
    gs_model_dir = (
        _GS2MESH_BASE / "splatting_output" / splatting_string / scene_name
    )
    gs_model_dir.mkdir(parents=True, exist_ok=True)

    _run_subprocess(
        ["python3", "-u",
         str(_GS2MESH_BASE / "third_party/gaussian-splatting/train.py"),
         "--source_path", str(gs2mesh_data),
         "--model_path", str(gs_model_dir),
         "--iterations", str(gs_iterations)],
        prefix="3DGS",
        progress_fn=lambda line: _parse_gs_progress(
            line, gs_iterations, _report,
        ),
        register_process=register_process,
        unregister_process=unregister_process,
        error_msg="3DGS training failed",
    )

    if cancel_cb:
        cancel_cb()

    # Step 2: Run gs2mesh (rendering + stereo depth + TSDF fusion).
    # GS training is skipped since we did it above.
    _report(55.0, "Running gs2mesh stereo depth + TSDF fusion")
    tsdf_voxel = max(1, round(tsdf_voxel_size * 512))

    gs2mesh_args = [
        "python3", "-u", str(_GS2MESH_BASE / "run_single.py"),
        "--colmap_name", scene_name,
        "--dataset_name", "custom",
        "--experiment_folder_name", "clip2mesh",
        "--skip_video_extraction",
        "--skip_colmap",
        "--skip_GS",
        "--skip_masking",
        "--GS_iterations", str(gs_iterations),
        "--stereo_model", stereo_model,
        "--TSDF_voxel", str(tsdf_voxel),
        "--TSDF_sdf_trunc", str(tsdf_depth_trunc),
    ]

    _run_subprocess(
        gs2mesh_args,
        cwd=str(_GS2MESH_BASE),
        prefix="gs2mesh",
        progress_fn=lambda line: _parse_gs2mesh_progress(line, _report),
        register_process=register_process,
        unregister_process=unregister_process,
        error_msg="gs2mesh reconstruction failed",
    )

    # Step 3: Find and copy output mesh
    _report(95.0, "Collecting output mesh")
    gs2mesh_output = _GS2MESH_BASE / "output" / "clip2mesh" / scene_name
    output_mesh = _find_output_mesh(gs2mesh_output)
    if output_mesh is None:
        raise RuntimeError(f"No mesh output found in {gs2mesh_output}")

    final_mesh = out / "object_mesh.ply"
    shutil.copy2(str(output_mesh), str(final_mesh))
    print(f"gs2mesh output mesh: {final_mesh}")

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
) -> None:
    """Run a subprocess, streaming prefixed output with progress parsing."""
    proc = subprocess.Popen(
        args,
        cwd=cwd,
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
                print(f"  [{prefix}] {line}")
                if progress_fn:
                    progress_fn(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"{error_msg} (exit {proc.returncode})")
    finally:
        if unregister_process:
            unregister_process(proc)


def _parse_gs_progress(
    line: str, total_iterations: int, report,
) -> None:
    """Parse 3DGS training progress from output lines."""
    match = re.search(r"\[ITER\s+(\d+)\]", line, re.IGNORECASE)
    if match:
        current = int(match.group(1))
        pct = 5.0 + (current / total_iterations) * 45.0  # 5% → 50%
        report(pct, f"3DGS training ({current}/{total_iterations})")


def _parse_gs2mesh_progress(line: str, report) -> None:
    """Parse gs2mesh progress from output lines."""
    lower = line.lower()
    if "stereo" in lower or "disparity" in lower:
        match = re.search(r"(\d+)%", line)
        if match:
            pct = 55.0 + int(match.group(1)) * 0.25  # 55% → 80%
            report(pct, "Stereo depth matching")
    elif "tsdf" in lower or "fusion" in lower:
        report(85.0, "TSDF fusion")
    elif "clean" in lower and "mesh" in lower:
        report(92.0, "Cleaning mesh")


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
