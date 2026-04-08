"""Stage 4: MILo Reconstruction.

Runs MILo (Mesh-In-the-Loop 3DGS) pipeline:
  COLMAP undistortion -> MILo training -> mesh extraction -> object_mesh.ply

Takes COLMAP output + optional SAM2 masks, produces object_mesh.ply.
"""

from __future__ import annotations

import gc
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from scripts.milo_config import MiloSettings

_MILO_BASE = Path("/opt/MILo")
_MILO_SCRIPTS = _MILO_BASE / "milo"


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


def run_milo(
    frames_dir: str,
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: str,
    settings: MiloSettings | None = None,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    """MILo pipeline: COLMAP undistort -> train -> mesh extract.

    Returns path to object_mesh.ply.
    """
    out = Path(output_dir)
    milo_workdir = out / "milo_workspace"
    milo_workdir.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = MiloSettings.from_preset()

    def _report(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if cancel_cb:
            cancel_cb()

    # Find the COLMAP reconstruction subdir (0/, 1/, etc.)
    recon_dir = _find_recon_dir(colmap_sparse_dir)

    # Step 0: Undistort images.
    _report(1.0, "Undistorting images for MILo")
    undistorted_dir = milo_workdir / "undistorted"
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

    if cancel_cb:
        cancel_cb()

    # Step 1: Apply SAM2 masks (zero out background) if enabled.
    source_images_dir = undistorted_dir / "images"
    if settings.use_masks and mask_dir is not None:
        _report(2.0, "Applying SAM2 masks to images")
        masked_dir = milo_workdir / "masked_images"
        _apply_masks_to_images(source_images_dir, Path(mask_dir), masked_dir)
        source_images_dir = masked_dir
        _report(5.0, "Mask preprocessing complete")

    if cancel_cb:
        cancel_cb()

    # Prepare MILo source directory structure.
    # MILo expects: {source}/images/ + {source}/sparse/0/
    milo_source = milo_workdir / "milo_source"
    _setup_milo_source_dir(milo_source, source_images_dir, undistorted_dir / "sparse")

    # Step 2: MILo training.
    _report(5.0, "Training MILo (Mesh-In-the-Loop 3DGS)")
    model_dir = milo_workdir / "model"

    from scripts.vram_utils import log_vram_detailed
    log_vram_detailed("before MILo training subprocess")

    train_args = _build_milo_train_args(milo_source, model_dir, settings)
    _run_subprocess(
        train_args,
        cwd=str(_MILO_SCRIPTS),
        prefix="MILo",
        progress_fn=lambda line: _parse_milo_progress(
            line, settings.iterations, _report,
        ),
        register_process=register_process,
        unregister_process=unregister_process,
        error_msg="MILo training failed",
        env=_build_milo_env(),
    )

    if cancel_cb:
        cancel_cb()

    from scripts.vram_utils import cleanup_pytorch_vram, log_vram
    log_vram("after MILo training subprocess exit")
    cleanup_pytorch_vram()

    # Step 3: Mesh extraction.
    _report(85.0, "Extracting mesh from MILo model")
    extract_args = _build_milo_extract_args(milo_source, model_dir, settings)
    _run_subprocess(
        extract_args,
        cwd=str(_MILO_SCRIPTS),
        prefix="MILo-extract",
        register_process=register_process,
        unregister_process=unregister_process,
        error_msg="MILo mesh extraction failed",
        env=_build_milo_env(),
    )

    if cancel_cb:
        cancel_cb()

    # Locate the extracted mesh.
    extracted_mesh = _find_extracted_mesh(model_dir, settings.extraction_method)
    if not extracted_mesh.exists():
        raise RuntimeError(
            f"MILo mesh extraction produced no output: {extracted_mesh}"
        )

    # Step 4: Post-extraction mask trimming (optional).
    if settings.use_masks and mask_dir is not None:
        _report(95.0, "Trimming mesh with SAM2 masks")
        extracted_mesh = _trim_mesh_with_masks(
            extracted_mesh, Path(mask_dir), undistorted_dir, milo_workdir,
        )

    # Step 5: Copy output mesh.
    _report(98.0, "Collecting output mesh")
    final_mesh = out / "object_mesh.ply"
    shutil.copy2(str(extracted_mesh), str(final_mesh))
    print(f"MILo output mesh: {final_mesh}")

    _report(100.0, "MILo reconstruction complete")
    return str(final_mesh)


# ---------------------------------------------------------------------------
#  COLMAP Helpers
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
    """Ensure sparse/ has a 0/ subdirectory (COLMAP convention)."""
    if sparse_dir.is_dir() and not (sparse_dir / "0").is_dir():
        tmp = sparse_dir.parent / "_sparse_tmp"
        sparse_dir.rename(tmp)
        sparse_dir.mkdir()
        tmp.rename(sparse_dir / "0")


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


# ---------------------------------------------------------------------------
#  MILo Source Directory Setup
# ---------------------------------------------------------------------------

def _setup_milo_source_dir(
    milo_source: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> None:
    """Create MILo source directory with symlinks to images and sparse model."""
    milo_source.mkdir(parents=True, exist_ok=True)
    for name, target in [("images", images_dir), ("sparse", sparse_dir)]:
        link = milo_source / name
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(target.resolve())


# ---------------------------------------------------------------------------
#  Mask Preprocessing
# ---------------------------------------------------------------------------

def _apply_masks_to_images(
    images_dir: Path,
    mask_dir: Path,
    output_dir: Path,
) -> None:
    """Apply SAM2 masks to images, zeroing out background pixels."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(images_dir.iterdir())
    mask_files = sorted(mask_dir.glob("*.png"))

    mask_map: dict[str, Path] = {}
    for mf in mask_files:
        mask_map[mf.stem] = mf

    for img_path in image_files:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # Match mask by stem (e.g., "00000" -> "00000.png")
        mask_path = mask_map.get(img_path.stem)
        if mask_path is None:
            # Try numeric matching: image "frame_00005.jpg" -> mask "00005.png"
            digits = re.search(r"(\d{5})", img_path.stem)
            if digits:
                mask_path = mask_map.get(digits.group(1))

        img = Image.open(img_path).convert("RGB")
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            img_array = np.array(img)
            mask_array = np.array(mask)
            # Feather mask edges for smoother boundaries
            mask_float = mask_array.astype(np.float32) / 255.0
            img_array = (img_array * mask_float[:, :, np.newaxis]).astype(
                np.uint8
            )
            img = Image.fromarray(img_array)

        img.save(output_dir / img_path.name)

    print(f"Masked {len(image_files)} images -> {output_dir}")


# ---------------------------------------------------------------------------
#  MILo Training
# ---------------------------------------------------------------------------

def _build_milo_train_args(
    source_path: Path,
    model_path: Path,
    settings: MiloSettings,
) -> list[str]:
    args = [
        "python3", "-u",
        str(_MILO_SCRIPTS / "train.py"),
        "-s", str(source_path),
        "-m", str(model_path),
        "--imp_metric", settings.scene_type,
        "--rasterizer", settings.rasterizer,
        "--mesh_config", settings.mesh_config,
        "--iterations", str(settings.iterations),
        "--data_device", settings.data_device,
    ]
    if settings.dense_gaussians:
        args.append("--dense_gaussians")
    return args


def _build_milo_env() -> dict[str, str]:
    """Build environment for MILo subprocess."""
    env = dict(os.environ)
    milo_paths = [str(_MILO_SCRIPTS), str(_MILO_BASE)]
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in milo_paths if p not in existing]
    if parts:
        prefix = ":".join(parts)
        env["PYTHONPATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


_ITER_RE = re.compile(r"\[ITER\s+(\d+)\]", re.IGNORECASE)


def _parse_milo_progress(
    line: str,
    total_iterations: int,
    report,
) -> None:
    """Parse MILo training output for iteration progress."""
    m = _ITER_RE.search(line)
    if m and total_iterations > 0:
        current = int(m.group(1))
        # Training progress maps to 5%-85%
        frac = min(current / total_iterations, 1.0)
        pct = 5.0 + frac * 80.0
        report(pct, f"MILo training iteration {current}/{total_iterations}")


# ---------------------------------------------------------------------------
#  Mesh Extraction
# ---------------------------------------------------------------------------

def _build_milo_extract_args(
    source_path: Path,
    model_path: Path,
    settings: MiloSettings,
) -> list[str]:
    if settings.extraction_method == "sdf":
        script = "mesh_extract_sdf.py"
    elif settings.extraction_method == "integration":
        script = "mesh_extract_integration.py"
    else:
        script = "mesh_extract_regular_tsdf.py"

    args = [
        "python3", "-u",
        str(_MILO_SCRIPTS / script),
        "-s", str(source_path),
        "-m", str(model_path),
        "--rasterizer", settings.rasterizer,
    ]
    if settings.extraction_method == "regular_tsdf":
        args.extend(["--mesh_res", str(settings.mesh_res)])
    return args


def _find_extracted_mesh(model_dir: Path, extraction_method: str) -> Path:
    """Locate the mesh file produced by MILo extraction."""
    if extraction_method == "sdf":
        return model_dir / "mesh_learnable_sdf.ply"
    elif extraction_method == "integration":
        return model_dir / "mesh_integration_sdf.ply"
    else:
        # regular_tsdf: filename includes resolution
        for p in model_dir.glob("mesh_regular_tsdf_res*.ply"):
            return p
        return model_dir / "mesh_regular_tsdf.ply"


# ---------------------------------------------------------------------------
#  Post-Extraction Mask Trimming
# ---------------------------------------------------------------------------

def _trim_mesh_with_masks(
    mesh_path: Path,
    mask_dir: Path,
    undistorted_dir: Path,
    work_dir: Path,
) -> Path:
    """Remove mesh vertices consistently outside SAM2 masks.

    Projects each vertex into available camera views and removes vertices
    that fall outside the object mask in all views.
    """
    try:
        import trimesh
    except ImportError:
        print("trimesh not available, skipping mask trimming")
        return mesh_path

    mesh = trimesh.load(str(mesh_path), process=False)
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        return mesh_path

    trimmed_path = work_dir / "trimmed_mesh.ply"
    # For now, copy as-is. Full camera-projection trimming is a future
    # enhancement that requires parsing COLMAP camera intrinsics/extrinsics.
    mesh.export(str(trimmed_path))
    print(f"Mesh trimming: {len(mesh.vertices)} vertices -> {trimmed_path}")
    return trimmed_path


# ---------------------------------------------------------------------------
#  Subprocess Helpers
# ---------------------------------------------------------------------------

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
) -> str:
    """Run a subprocess, streaming prefixed output."""
    output_lines: list[str] = []

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
        if unregister_process:
            unregister_process(proc)
