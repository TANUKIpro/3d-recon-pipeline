"""Stage 4: gs2mesh Reconstruction.

Runs gs2mesh pipeline: 3DGS training -> stereo depth -> GPU TSDF fusion -> mesh.
Takes COLMAP output + optional SAM2 masks, produces object_mesh.ply.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.config_defaults import (
    GS2MESH_RUNTIME_PROFILE,
    GS2MESH_RUNTIME_PROFILES,
)

_GS2MESH_BASE = Path("/opt/gs2mesh")
_GAUSSIAN_SPLATTING_ACCEL = Path("/opt/gaussian-splatting")
# The compat retry reuses the same gaussian-splatting source tree and only
# overrides the rasterizer extension from a separate site-packages overlay.
_GAUSSIAN_SPLATTING_COMPAT = _GAUSSIAN_SPLATTING_ACCEL
_GS2MESH_GS_LINK = _GS2MESH_BASE / "third_party" / "gaussian-splatting"
_GS_COMPAT_SITE_PACKAGES = Path("/opt/gs-compat-site")
_RETRYABLE_GS2MESH_FAILURES = frozenset(
    {
        "cuda_illegal_memory_access",
        "cuda_rasterizer_failure",
        "renderer_signature_mismatch",
        "missing_rasterizer_extension",
        "missing_output_artifacts",
    }
)


@dataclass(frozen=True)
class _Gs2meshRuntimeStack:
    name: str
    python_executable: str
    gaussian_splatting_root: Path
    env_overrides: dict[str, str]
    extra_pythonpaths: tuple[str, ...] = ()


class _SubprocessFailure(RuntimeError):
    """Structured subprocess failure with captured output for classification."""

    def __init__(
        self,
        message: str,
        *,
        args: list[str],
        exit_code: int,
        output_text: str,
    ) -> None:
        super().__init__(message)
        self.args_run = list(args)
        self.exit_code = exit_code
        self.output_text = output_text


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
    """gs2mesh pipeline: 3DGS training -> stereo depth -> GPU TSDF -> mesh.

    Returns path to object_mesh.ply.
    """
    out = Path(output_dir)
    gs2mesh_workdir = out / "gs2mesh_workspace"
    debug_dir = gs2mesh_workdir / "debug"
    gs2mesh_workdir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Map shortened stereo model names for backward compatibility
    if stereo_model == "DLNR":
        stereo_model = "DLNR_Middlebury"

    def _report(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if cancel_cb:
            cancel_cb()

    resolved_profile = _normalize_runtime_profile(runtime_profile)
    training_stack = _resolve_training_runtime_stack(resolved_profile)

    # Find the COLMAP reconstruction subdir (0/, 1/, etc.)
    recon_dir = _find_recon_dir(colmap_sparse_dir)

    # Step 0a: Undistort images. 3DGS requires PINHOLE cameras, but
    # COLMAP defaults to SIMPLE_RADIAL.
    _report(1.0, "Undistorting images for 3DGS")
    undistorted_dir = gs2mesh_workdir / "undistorted"
    _run_colmap_cmd(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(frames_dir),
            "--input_path",
            str(recon_dir),
            "--output_path",
            str(undistorted_dir),
            "--output_type",
            "COLMAP",
        ],
        "image_undistorter",
        undistorted_dir,
        register_process,
        unregister_process,
    )
    _ensure_sparse_0(undistorted_dir / "sparse")

    # Step 0b: Convert binary COLMAP model -> text format.
    # gs2mesh's Renderer reads images.txt / cameras.txt, not binary files.
    _report(2.0, "Converting COLMAP model to text format")
    undist_model = undistorted_dir / "sparse" / "0"
    _run_colmap_cmd(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(undist_model),
            "--output_path",
            str(undist_model),
            "--output_type",
            "TXT",
        ],
        "model_converter (binary->text)",
        None,
        register_process,
        unregister_process,
    )

    # Step 0c: Set up gs2mesh data directory with symlinks to undistorted output.
    # gs2mesh expects: {base}/data/custom/{scene}/images/ + sparse/0/
    scene_name = out.name
    gs2mesh_data = _GS2MESH_BASE / "data" / "custom" / scene_name
    _setup_gs2mesh_dirs(gs2mesh_data, undistorted_dir)
    _ensure_gs2mesh_renderer_compat()
    _ensure_gaussian_renderer_compat(training_stack.gaussian_splatting_root)

    # Step 1: Train 3D Gaussian Splatting model.
    # We train separately for better progress tracking and error handling.
    _report(5.0, "Training 3D Gaussian Splatting model")
    splatting_string = f"custom_nw_iterations{gs_iterations}"
    gs_model_dir = (
        _GS2MESH_BASE / "splatting_output" / splatting_string / scene_name
    )
    persistent_gs_model_dir = (
        gs2mesh_workdir / "splatting_output" / splatting_string / scene_name
    )
    _ensure_gs_model_output_link(gs_model_dir, persistent_gs_model_dir)
    final_point_cloud = (
        persistent_gs_model_dir
        / "point_cloud"
        / f"iteration_{gs_iterations}"
        / "point_cloud.ply"
    )
    gs_train_args, resolved_profile, optimizer_type, fallback_reason = (
        _build_gs_train_args(
            gs2mesh_data,
            gs_model_dir,
            gs_iterations,
            resolved_profile,
            python_executable=training_stack.python_executable,
            gaussian_splatting_root=training_stack.gaussian_splatting_root,
        )
    )
    print(f"3DGS runtime profile: {resolved_profile}")
    print(f"3DGS runtime stack: {training_stack.name}")
    print(f"3DGS optimizer: {optimizer_type}")
    if fallback_reason:
        print(f"3DGS optimizer fallback: {fallback_reason}")

    if final_point_cloud.is_file():
        print(f"Reusing existing 3DGS checkpoint: {final_point_cloud}")
        _report(50.0, "Reusing existing 3D Gaussian Splatting checkpoint")
    else:
        _run_subprocess(
            gs_train_args,
            prefix="3DGS",
            progress_fn=lambda line: _parse_gs_progress(
                line, gs_iterations, _report,
            ),
            register_process=register_process,
            unregister_process=unregister_process,
            error_msg="3DGS training failed",
            env=_build_runtime_env(
                training_stack.gaussian_splatting_root,
                training_stack.env_overrides,
                extra_pythonpaths=training_stack.extra_pythonpaths,
            ),
        )

    if cancel_cb:
        cancel_cb()

    # Step 2: Run gs2mesh (rendering + stereo depth only; TSDF on GPU below).
    _report(55.0, "Running gs2mesh stereo depth estimation")
    tsdf_voxel = max(1, round(tsdf_voxel_size * 512))
    gs2mesh_output = _GS2MESH_BASE / "output" / "clip2mesh" / scene_name

    _run_gs2mesh_stereo_with_retries(
        scene_name=scene_name,
        stereo_model=stereo_model,
        gs_iterations=gs_iterations,
        tsdf_voxel=tsdf_voxel,
        tsdf_depth_trunc=tsdf_depth_trunc,
        runtime_profile=resolved_profile,
        gs2mesh_output=gs2mesh_output,
        debug_dir=debug_dir,
        report=_report,
        register_process=register_process,
        unregister_process=unregister_process,
    )

    if cancel_cb:
        cancel_cb()

    if use_masks and mask_dir is not None:
        _report(79.0, "Materializing SAM2 masks for TSDF")
        _materialize_tsdf_masks(gs2mesh_output, mask_dir)

    # Step 3: GPU TSDF fusion (replaces gs2mesh CPU TSDF).
    _report(80.0, "GPU TSDF fusion")

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


def _ensure_gs_model_output_link(
    runtime_model_dir: Path,
    persistent_model_dir: Path,
) -> None:
    persistent_model_dir.mkdir(parents=True, exist_ok=True)
    runtime_model_dir.parent.mkdir(parents=True, exist_ok=True)

    if runtime_model_dir.is_symlink():
        if runtime_model_dir.resolve() == persistent_model_dir.resolve():
            return
        runtime_model_dir.unlink()
    elif runtime_model_dir.exists():
        shutil.rmtree(runtime_model_dir)

    runtime_model_dir.symlink_to(persistent_model_dir, target_is_directory=True)
    print(f"3DGS model output -> {persistent_model_dir}")


def _patch_gs2mesh_renderer_utils(renderer_utils_path: Path) -> bool:
    """Patch gs2mesh renderer for newer gaussian-splatting builds."""
    text = renderer_utils_path.read_text(encoding="utf-8")
    updated = text

    if "from PIL import Image" not in updated:
        anchor = "import copy\n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: copy import anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "from PIL import Image\n",
            1,
        )

    if "depths=''" not in updated:
        anchor = "                         images='images', \n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: images anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "                         depths='', \n",
            1,
        )

    if "train_test_exp=False" not in updated:
        anchor = "                         eval=False, \n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: eval anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "                         train_test_exp=False, \n",
            1,
        )

    if "antialiasing=False" not in updated:
        anchor = "                         debug=False, \n"
        if anchor not in updated:
            raise RuntimeError(
                "Could not patch gs2mesh renderer: debug anchor missing"
            )
        updated = updated.replace(
            anchor,
            anchor + "                         antialiasing=False, \n",
            1,
        )

    old_camera_call = (
        '                view = cameras.Camera(0, R, T, FoVx, FoVy, '
        'torch.rand(3,h,w), None, "abcd", 0)\n'
    )
    new_camera_call = """                dummy_image = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
                view = cameras.Camera(
                    (w, h),
                    0,
                    R,
                    T,
                    FoVx,
                    FoVy,
                    None,
                    dummy_image,
                    None,
                    f"{camera_name}",
                    camera_number,
                    data_device=self.device,
                )
"""
    if old_camera_call in updated:
        updated = updated.replace(old_camera_call, new_camera_call, 1)

    if updated == text:
        return False

    renderer_utils_path.write_text(updated, encoding="utf-8")
    return True


def _patch_gaussian_renderer_init(renderer_init_path: Path) -> bool:
    """Patch gaussian_renderer to retry without unsupported rasterizer kwargs."""
    text = renderer_init_path.read_text(encoding="utf-8")
    updated = text

    helper_anchor = "from utils.sh_utils import eval_sh\n"
    raster_helper_body = """
def _build_raster_settings(**kwargs):
    try:
        return GaussianRasterizationSettings(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        retry_kwargs = dict(kwargs)
        for key in ("antialiasing", "train_test_exp"):
            if key in retry_kwargs and key in str(exc):
                retry_kwargs.pop(key, None)
                return _build_raster_settings(**retry_kwargs)
        raise

"""
    unpack_helper_body = """
def _unpack_rasterizer_output(output):
    if not isinstance(output, (tuple, list)):
        raise TypeError(
            "Unexpected rasterizer return type: "
            f"{type(output).__name__}"
        )
    if len(output) == 3:
        return output
    if len(output) == 2:
        rendered_image, radii = output
        return rendered_image, radii, None
    raise ValueError(
        "Unexpected rasterizer return arity: "
        f"{len(output)}"
    )

"""
    if "def _build_raster_settings(**kwargs):" not in updated:
        if helper_anchor not in updated:
            raise RuntimeError(
                "Could not patch gaussian_renderer: import anchor missing"
            )
        updated = updated.replace(
            helper_anchor,
            helper_anchor + raster_helper_body,
            1,
        )

    if "def _unpack_rasterizer_output(output):" not in updated:
        helper_insert_anchor = "def _build_raster_settings(**kwargs):"
        helper_insert_body = raster_helper_body + unpack_helper_body
        if helper_insert_body in updated:
            pass
        elif helper_insert_anchor in updated:
            updated = updated.replace(
                raster_helper_body,
                raster_helper_body + unpack_helper_body,
                1,
            )
        elif helper_anchor in updated:
            updated = updated.replace(
                helper_anchor,
                helper_anchor + unpack_helper_body,
                1,
            )
        else:
            raise RuntimeError(
                "Could not patch gaussian_renderer: helper insert anchor missing"
            )

    old_block = """    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
"""
    new_block = """    raster_settings_kwargs = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    if hasattr(pipe, "antialiasing"):
        raster_settings_kwargs["antialiasing"] = pipe.antialiasing
    if hasattr(pipe, "train_test_exp"):
        raster_settings_kwargs["train_test_exp"] = pipe.train_test_exp
    raster_settings = _build_raster_settings(**raster_settings_kwargs)
"""
    if old_block in updated:
        updated = updated.replace(old_block, new_block, 1)

    old_rasterizer_block = """    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
"""
    new_rasterizer_block = """    if separate_sh:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    rendered_image, radii, depth_image = _unpack_rasterizer_output(
        rasterizer_output
    )
"""
    if old_rasterizer_block in updated:
        updated = updated.replace(
            old_rasterizer_block,
            new_rasterizer_block,
            1,
        )

    if updated == text:
        return False

    renderer_init_path.write_text(updated, encoding="utf-8")
    return True


def _ensure_gs2mesh_renderer_compat() -> None:
    renderer_utils_path = _GS2MESH_BASE / "gs2mesh_utils" / "renderer_utils.py"
    if _patch_gs2mesh_renderer_utils(renderer_utils_path):
        print(
            "Patched gs2mesh renderer args for gaussian-splatting "
            "compatibility"
        )


def _ensure_gaussian_renderer_compat(
    gaussian_splatting_root: Path = _GAUSSIAN_SPLATTING_ACCEL,
) -> None:
    renderer_init_path = gaussian_splatting_root / "gaussian_renderer" / "__init__.py"
    if _patch_gaussian_renderer_init(renderer_init_path):
        print(
            "Patched gaussian_renderer to retry without unsupported "
            "rasterizer kwargs"
        )


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
    *,
    python_executable: str = "python3",
    gaussian_splatting_root: Path = _GAUSSIAN_SPLATTING_ACCEL,
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
        python_executable,
        "-u",
        str(gaussian_splatting_root / "train.py"),
        "--source_path",
        str(source_path),
        "--model_path",
        str(model_path),
        "--iterations",
        str(iterations),
        "--disable_viewer",
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(save_sentinel),
        "--optimizer_type",
        optimizer_type,
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
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    capture_output: bool = False,
) -> str:
    """Run a subprocess, streaming prefixed output with optional capture."""
    output_lines: list[str] = []
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        env=env,
    )
    if register_process:
        register_process(proc)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(f"  [{prefix}] {line}")
            if log_file is not None:
                log_file.write(line + "\n")
                log_file.flush()
            if capture_output:
                output_lines.append(line)
            if progress_fn:
                progress_fn(line)
        proc.wait()
        output_text = "\n".join(output_lines)
        if proc.returncode != 0:
            raise _SubprocessFailure(
                f"{error_msg} (exit {proc.returncode})",
                args=args,
                exit_code=proc.returncode,
                output_text=output_text,
            )
        return output_text
    finally:
        if log_file is not None:
            log_file.close()
        if unregister_process:
            unregister_process(proc)


def _parse_gs_progress(
    line: str,
    total_iterations: int,
    report,
) -> None:
    """Parse 3DGS training progress from output lines."""
    match = re.search(r"\[ITER\s+(\d+)\]", line, re.IGNORECASE)
    if match:
        current = int(match.group(1))
        pct = 5.0 + (current / total_iterations) * 45.0  # 5% -> 50%
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
            pct = 55.0 + int(match.group(1)) * 0.25  # 55% -> 80%
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


def _resolve_training_runtime_stack(
    runtime_profile: str,
) -> _Gs2meshRuntimeStack:
    if runtime_profile == "compat":
        return _compat_runtime_stack(required=True)
    return _accel_runtime_stack()


def _build_stereo_runtime_stacks(
    runtime_profile: str,
) -> list[_Gs2meshRuntimeStack]:
    if runtime_profile == "compat":
        return [_compat_runtime_stack(required=True, diagnostics=True)]

    stacks = [_accel_runtime_stack()]
    compat_stack = _compat_runtime_stack(required=False, diagnostics=True)
    if compat_stack is not None:
        stacks.append(compat_stack)
    else:
        print("gs2mesh compat runtime unavailable; no stereo fallback stack")
    return stacks


def _accel_runtime_stack(
    *,
    diagnostics: bool = False,
) -> _Gs2meshRuntimeStack:
    env = {"PYTHONFAULTHANDLER": "1"}
    if diagnostics:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    return _Gs2meshRuntimeStack(
        name="accel",
        python_executable="python3",
        gaussian_splatting_root=_GAUSSIAN_SPLATTING_ACCEL,
        env_overrides=env,
    )


def _compat_runtime_stack(
    *,
    required: bool,
    diagnostics: bool = False,
) -> _Gs2meshRuntimeStack | None:
    missing: list[str] = []
    if not _GS_COMPAT_SITE_PACKAGES.is_dir():
        missing.append(str(_GS_COMPAT_SITE_PACKAGES))
    if not _GAUSSIAN_SPLATTING_COMPAT.is_dir():
        missing.append(str(_GAUSSIAN_SPLATTING_COMPAT))
    if missing:
        if required:
            raise RuntimeError(
                "gs2mesh compat runtime is unavailable: " + ", ".join(missing)
            )
        return None

    env = {
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTHONFAULTHANDLER": "1",
    }
    if diagnostics:
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    return _Gs2meshRuntimeStack(
        name="compat",
        python_executable="python3",
        gaussian_splatting_root=_GAUSSIAN_SPLATTING_COMPAT,
        env_overrides=env,
        extra_pythonpaths=(str(_GS_COMPAT_SITE_PACKAGES),),
    )


def _build_runtime_env(
    gaussian_splatting_root: Path,
    env_overrides: dict[str, str] | None = None,
    extra_pythonpaths: tuple[str, ...] = (),
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(
        gaussian_splatting_root,
        extra_pythonpaths=extra_pythonpaths,
    )
    if env_overrides:
        env.update(env_overrides)
    return env


def _build_pythonpath(
    gaussian_splatting_root: Path,
    *,
    extra_pythonpaths: tuple[str, ...] = (),
) -> str:
    preferred = [
        "/app",
        str(_GS2MESH_BASE),
        str(gaussian_splatting_root),
        "/opt/sam2",
    ]
    preferred = [*extra_pythonpaths, *preferred]
    existing = [
        part
        for part in os.environ.get("PYTHONPATH", "").split(":")
        if part
    ]
    filtered: list[str] = []
    ignored = {
        str(_GAUSSIAN_SPLATTING_ACCEL),
        str(_GAUSSIAN_SPLATTING_COMPAT),
    }
    for part in existing:
        if part in ignored or part in preferred or part in filtered:
            continue
        filtered.append(part)
    return ":".join(preferred + filtered)


def _ensure_gs2mesh_gaussian_splatting_link(target: Path) -> None:
    if not target.is_dir():
        raise RuntimeError(f"gaussian-splatting target missing: {target}")

    if _GS2MESH_GS_LINK.is_symlink():
        if _GS2MESH_GS_LINK.resolve() == target.resolve():
            return
        _GS2MESH_GS_LINK.unlink()
    elif _GS2MESH_GS_LINK.exists():
        raise RuntimeError(
            f"Expected symlink at {_GS2MESH_GS_LINK}, found directory/file"
        )

    _GS2MESH_GS_LINK.symlink_to(target, target_is_directory=True)
    print(f"gs2mesh gaussian-splatting target -> {target}")


def _run_gs2mesh_stereo_with_retries(
    *,
    scene_name: str,
    stereo_model: str,
    gs_iterations: int,
    tsdf_voxel: int,
    tsdf_depth_trunc: float,
    runtime_profile: str,
    gs2mesh_output: Path,
    debug_dir: Path,
    report,
    register_process=None,
    unregister_process=None,
) -> None:
    attempts = _build_stereo_runtime_stacks(runtime_profile)
    gs2mesh_args_tail = [
        "-u",
        str(_GS2MESH_BASE / "run_single.py"),
        "--colmap_name",
        scene_name,
        "--dataset_name",
        "custom",
        "--experiment_folder_name",
        "clip2mesh",
        "--skip_video_extraction",
        "--skip_colmap",
        "--skip_GS",
        "--skip_masking",
        "--skip_TSDF",
        "--GS_iterations",
        str(gs_iterations),
        "--stereo_model",
        stereo_model,
        "--TSDF_voxel",
        str(tsdf_voxel),
        "--TSDF_sdf_trunc",
        str(tsdf_depth_trunc),
    ]

    for idx, stack in enumerate(attempts, start=1):
        args = [stack.python_executable, *gs2mesh_args_tail]
        env = _build_runtime_env(
            stack.gaussian_splatting_root,
            stack.env_overrides,
            extra_pythonpaths=stack.extra_pythonpaths,
        )
        log_path = debug_dir / f"stereo_attempt_{idx:02d}_{stack.name}.log"
        meta_path = debug_dir / f"stereo_attempt_{idx:02d}_{stack.name}.json"

        report(55.0, f"Running gs2mesh stereo depth estimation ({stack.name})")
        _cleanup_gs2mesh_stereo_outputs(gs2mesh_output)
        _ensure_gs2mesh_gaussian_splatting_link(stack.gaussian_splatting_root)
        _ensure_gs2mesh_renderer_compat()
        _ensure_gaussian_renderer_compat(stack.gaussian_splatting_root)

        try:
            _run_subprocess(
                args,
                cwd=str(_GS2MESH_BASE),
                prefix="gs2mesh",
                progress_fn=lambda line: _parse_gs2mesh_progress(line, report),
                register_process=register_process,
                unregister_process=unregister_process,
                error_msg="gs2mesh stereo estimation failed",
                env=env,
                log_path=log_path,
                capture_output=True,
            )
            _validate_gs2mesh_stereo_outputs(gs2mesh_output, stereo_model)
            _write_attempt_metadata(
                meta_path,
                stack=stack,
                args=args,
                env=env,
                log_path=log_path,
                success=True,
                failure_reason=None,
                exit_code=0,
            )
            print(f"gs2mesh stereo attempt succeeded: {stack.name}")
            return
        except _SubprocessFailure as exc:
            failure_reason = _classify_gs2mesh_failure(exc.output_text)
            retrying = (
                idx < len(attempts)
                and failure_reason in _RETRYABLE_GS2MESH_FAILURES
            )
            _write_attempt_metadata(
                meta_path,
                stack=stack,
                args=args,
                env=env,
                log_path=log_path,
                success=False,
                failure_reason=failure_reason,
                exit_code=exc.exit_code,
            )
            print(
                f"gs2mesh stereo attempt failed: stack={stack.name} "
                f"reason={failure_reason}"
            )
            if retrying:
                print("Retrying gs2mesh stereo with fallback runtime stack")
                continue
            raise RuntimeError(
                _format_gs2mesh_failure_message(
                    failure_reason,
                    log_path,
                    debug_dir,
                )
            ) from exc
        except RuntimeError as exc:
            failure_reason = _classify_gs2mesh_failure(str(exc))
            retrying = (
                idx < len(attempts)
                and failure_reason in _RETRYABLE_GS2MESH_FAILURES
            )
            _write_attempt_metadata(
                meta_path,
                stack=stack,
                args=args,
                env=env,
                log_path=log_path,
                success=False,
                failure_reason=failure_reason,
                exit_code=None,
            )
            print(
                f"gs2mesh stereo attempt validation failed: stack={stack.name} "
                f"reason={failure_reason}"
            )
            if retrying:
                print("Retrying gs2mesh stereo after validation failure")
                continue
            raise RuntimeError(
                _format_gs2mesh_failure_message(
                    failure_reason,
                    log_path,
                    debug_dir,
                )
            ) from exc

    raise RuntimeError("gs2mesh stereo estimation failed without attempts")


def _cleanup_gs2mesh_stereo_outputs(gs2mesh_output: Path) -> None:
    if gs2mesh_output.exists():
        shutil.rmtree(gs2mesh_output)


def _iter_camera_string_fields(value: object) -> list[str]:
    """Collect nested string fields from a camera-data entry."""
    results: list[str] = []
    if isinstance(value, str):
        results.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            results.extend(_iter_camera_string_fields(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            results.extend(_iter_camera_string_fields(nested))
    return results


def _resolve_camera_frame_index(camera_entry: dict[str, object]) -> int:
    """Resolve the original 5-digit frame index from gs2mesh camera metadata."""
    priority_keys = (
        "image_name",
        "image_path",
        "file_name",
        "filename",
        "path",
        "name",
    )
    candidate_strings: list[str] = []
    for key in priority_keys:
        value = camera_entry.get(key)
        if isinstance(value, str):
            candidate_strings.append(value)
    for value in camera_entry.values():
        candidate_strings.extend(_iter_camera_string_fields(value))

    candidates: list[int] = []
    seen: set[int] = set()
    for text in candidate_strings:
        path = Path(text)
        search_space = [path.stem, path.name, text]
        for candidate_text in search_space:
            matches = re.findall(r"(?<!\d)(\d{5})(?!\d)", candidate_text)
            for match in matches:
                value = int(match)
                if value not in seen:
                    candidates.append(value)
                    seen.add(value)
        if candidates:
            break

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "Could not resolve frame index from gs2mesh camera_data entry"
        )
    raise RuntimeError(
        "Ambiguous frame index in gs2mesh camera_data entry: "
        + ", ".join(str(value) for value in candidates)
    )


def _materialize_tsdf_masks(gs2mesh_output: Path, mask_dir: str | Path) -> None:
    """Convert canonical SAM2 masks into per-view ``left_mask.npy`` files."""
    import cv2
    import numpy as np

    mask_root = Path(mask_dir)
    camera_data_path = gs2mesh_output / "camera_data.json"
    if not camera_data_path.is_file():
        raise RuntimeError(
            "Missing camera_data.json while preparing TSDF masks"
        )

    with camera_data_path.open("r", encoding="utf-8") as f:
        camera_data = json.load(f)
    if not isinstance(camera_data, list):
        raise RuntimeError("camera_data.json must contain a list of camera entries")

    for cam_idx, camera_record in enumerate(camera_data):
        if not isinstance(camera_record, dict):
            raise RuntimeError(
                f"camera_data.json entry {cam_idx} is not an object"
            )
        left_entry = camera_record.get("left")
        if not isinstance(left_entry, dict):
            raise RuntimeError(
                f"camera_data.json entry {cam_idx} is missing a 'left' object"
            )
        frame_idx = _resolve_camera_frame_index(left_entry)
        mask_path = mask_root / f"{frame_idx:05d}.png"
        if not mask_path.is_file():
            raise RuntimeError(
                f"Missing canonical SAM2 mask for frame {frame_idx:05d}: {mask_path}"
            )

        view_dir = gs2mesh_output / f"{cam_idx:03d}"
        left_image_path = view_dir / "left.png"
        if not left_image_path.is_file():
            raise RuntimeError(f"Missing rendered left image: {left_image_path}")

        left_image = cv2.imread(str(left_image_path), cv2.IMREAD_COLOR)
        if left_image is None:
            raise RuntimeError(f"Could not read rendered left image: {left_image_path}")
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError(f"Could not read canonical SAM2 mask: {mask_path}")
        if mask_img.shape[:2] != left_image.shape[:2]:
            mask_img = cv2.resize(
                mask_img,
                (left_image.shape[1], left_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        np.save(view_dir / "left_mask.npy", mask_img > 0)


def _validate_gs2mesh_stereo_outputs(
    gs2mesh_output: Path,
    stereo_model: str,
) -> None:
    camera_data = gs2mesh_output / "camera_data.json"
    if not camera_data.is_file():
        raise RuntimeError("Missing stereo outputs: camera_data.json")
    depth_matches = sorted(
        gs2mesh_output.glob(f"*/out_{stereo_model}/depth.npy")
    )
    if not depth_matches:
        raise RuntimeError(
            f"Missing stereo outputs: */out_{stereo_model}/depth.npy"
        )


def _write_attempt_metadata(
    meta_path: Path,
    *,
    stack: _Gs2meshRuntimeStack,
    args: list[str],
    env: dict[str, str],
    log_path: Path,
    success: bool,
    failure_reason: str | None,
    exit_code: int | None,
) -> None:
    payload = {
        "stack": stack.name,
        "python_executable": stack.python_executable,
        "gaussian_splatting_root": str(stack.gaussian_splatting_root),
        "command": args,
        "log_path": str(log_path),
        "success": success,
        "failure_reason": failure_reason,
        "exit_code": exit_code,
        "env_overrides": {
            key: env[key]
            for key in sorted(stack.env_overrides)
            if key in env
        },
        "extra_pythonpaths": list(stack.extra_pythonpaths),
        "pythonpath": env.get("PYTHONPATH", ""),
    }
    meta_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _classify_gs2mesh_failure(text: str) -> str:
    lower = text.lower()
    if "illegal memory access" in lower:
        return "cuda_illegal_memory_access"
    if "diff_gaussian_rasterization" in lower and "cuda error" in lower:
        return "cuda_rasterizer_failure"
    if (
        "unexpected keyword argument" in lower
        or "camera.__init__" in lower
        or "train_test_exp" in lower
        or "antialiasing" in lower
        or "depths" in lower
        or "not enough values to unpack" in lower
    ):
        return "renderer_signature_mismatch"
    if (
        "no module named 'diff_gaussian_rasterization'" in lower
        or "modulenotfounderror: no module named 'diff_gaussian_rasterization'"
        in lower
    ):
        return "missing_rasterizer_extension"
    if "missing stereo outputs" in lower or "camera_data.json" in lower:
        return "missing_output_artifacts"
    return "unclassified"


def _format_gs2mesh_failure_message(
    failure_reason: str,
    log_path: Path,
    debug_dir: Path,
) -> str:
    return (
        "gs2mesh stereo estimation failed "
        f"({failure_reason}). See {log_path} and {debug_dir} for attempt "
        "logs and metadata."
    )
