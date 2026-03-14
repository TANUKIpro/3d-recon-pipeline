"""Stage 4: gs2mesh Reconstruction.

Runs gs2mesh pipeline: 3DGS training → stereo depth → GPU TSDF fusion → mesh.
Takes COLMAP output + optional SAM2 masks, produces object_mesh.ply.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path

from scripts.config_defaults import (
    GS2MESH_RUNTIME_PROFILE,
    GS2MESH_RUNTIME_PROFILES,
)

_GS2MESH_BASE = Path("/opt/gs2mesh")


def run_gs2mesh(
    frames_dir: str,
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: str,
    gs_iterations: int = 30000,
    runtime_profile: str = GS2MESH_RUNTIME_PROFILE,
    stereo_model: str = "DLNR_Middlebury",
    tsdf_voxel_size: float = 0.005,
    tsdf_depth_trunc: float = 0.04,
    use_masks: bool = True,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    """gs2mesh pipeline: 3DGS training → stereo depth → GPU TSDF → mesh.

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
    gs_train_args, resolved_profile, optimizer_type, fallback_reason = (
        _build_gs_train_args(
            gs2mesh_data,
            gs_model_dir,
            gs_iterations,
            runtime_profile,
        )
    )
    print(f"3DGS runtime profile: {resolved_profile}")
    print(f"3DGS optimizer: {optimizer_type}")
    if fallback_reason:
        print(f"3DGS optimizer fallback: {fallback_reason}")

    _run_subprocess(
        gs_train_args,
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

    # Step 2: Run gs2mesh (rendering + stereo depth only; TSDF on GPU below).
    # GS training is skipped since we did it above.
    _report(55.0, "Running gs2mesh stereo depth estimation")
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
        "--skip_TSDF",
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
        error_msg="gs2mesh stereo estimation failed",
    )

    if cancel_cb:
        cancel_cb()

    # Step 3: GPU TSDF fusion (replaces gs2mesh CPU TSDF).
    _report(80.0, "GPU TSDF fusion")
    gs2mesh_output = _GS2MESH_BASE / "output" / "clip2mesh" / scene_name

    from scripts.gpu_tsdf import gpu_tsdf_reconstruct

    gpu_mesh_path = gpu_tsdf_reconstruct(
        output_dir_root=str(gs2mesh_output),
        stereo_model=stereo_model,
        tsdf_voxel=tsdf_voxel,
        tsdf_sdf_trunc=tsdf_depth_trunc,
        tsdf_use_mask=use_masks,
        progress_cb=lambda pct, msg: _report(80.0 + pct * 15.0, msg),
    )

    # Step 4: Copy output mesh
    _report(95.0, "Collecting output mesh")
    final_mesh = out / "object_mesh.ply"
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
        "python3", "-u",
        str(_GS2MESH_BASE / "third_party/gaussian-splatting/train.py"),
        "--source_path", str(source_path),
        "--model_path", str(model_path),
        "--iterations", str(iterations),
        "--disable_viewer",
        "--test_iterations", "-1",
        "--save_iterations", str(save_sentinel),
        "--optimizer_type", optimizer_type,
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
    """Parse gs2mesh progress from output lines.

    With --skip_TSDF, gs2mesh only runs rendering + stereo (55%-80%).
    GPU TSDF fusion is handled separately afterwards.
    """
    lower = line.lower()
    if "stereo" in lower or "disparity" in lower:
        match = re.search(r"(\d+)%", line)
        if match:
            pct = 55.0 + int(match.group(1)) * 0.25  # 55% → 80%
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
