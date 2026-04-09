"""Stage 4: GaussianWrapping Reconstruction.

Runs GaussianWrapping pipeline:
  COLMAP undistortion -> GaussianWrapping training -> mesh extraction -> object_mesh.ply

Takes COLMAP output + optional SAM2 masks, produces object_mesh.ply.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from scripts.colmap_sparse_filter import (
    SparseFilterSettings,
    filter_colmap_sparse_model,
)
from scripts.gwrapping_config import GWrappingSettings

_GW_BASE = Path("/opt/GaussianWrapping")
_GW_TRAIN = _GW_BASE / "gaussian_wrapping" / "train.py"
_GW_EXTRACT = _GW_BASE / "gaussian_wrapping" / "pivot_based_mesh_extraction.py"
_GW_TEXTURE = _GW_BASE / "gaussian_wrapping" / "texture_mesh.py"
_GW_OURS_REGULARIZATION_FROM_ITER = 7000
_GW_RADEGS_REGULARIZATION_FROM_ITER = 15000


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


def run_gwrapping(
    frames_dir: str,
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: str,
    settings: GWrappingSettings | None = None,
    sparse_filter_settings: SparseFilterSettings | None = None,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    """GaussianWrapping pipeline: COLMAP undistort -> train -> mesh extract.

    Returns path to object_mesh.ply.
    """
    out = Path(output_dir)
    gw_workdir = out / "gwrapping_workspace"

    if settings is None:
        settings = GWrappingSettings.from_preset()
    if sparse_filter_settings is None:
        sparse_filter_settings = SparseFilterSettings()

    def _report(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if cancel_cb:
            cancel_cb()

    _reset_gwrapping_outputs(out)
    gw_workdir.mkdir(parents=True, exist_ok=True)

    # Find the COLMAP reconstruction subdir (0/, 1/, etc.) and optionally
    # filter sparse points against SAM2 object masks.
    recon_dir = _prepare_sparse_model(
        colmap_sparse_dir,
        mask_dir,
        out,
        settings,
        sparse_filter_settings,
        _report,
    )

    # Step 0: Undistort images.
    _report(1.0, "Undistorting images for GaussianWrapping")
    undistorted_dir = gw_workdir / "undistorted"
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
        masked_dir = gw_workdir / "masked_images"
        _apply_masks_to_images(source_images_dir, Path(mask_dir), masked_dir)
        source_images_dir = masked_dir
        _report(5.0, "Mask preprocessing complete")

    if cancel_cb:
        cancel_cb()

    # Prepare GaussianWrapping source directory structure.
    # GW expects: {source}/images/ + {source}/sparse/0/
    gw_source = gw_workdir / "gw_source"
    _setup_source_dir(gw_source, source_images_dir, undistorted_dir / "sparse")

    # Step 2: GaussianWrapping training + mesh extraction.
    _report(5.0, "Training GaussianWrapping (3DGS surface reconstruction)")
    model_dir = gw_workdir / "model"

    from scripts.vram_utils import log_vram_detailed
    log_vram_detailed("before GaussianWrapping training subprocess")

    _run_gwrapping_pipeline(
        gw_source,
        model_dir,
        settings,
        report=_report,
        register_process=register_process,
        unregister_process=unregister_process,
    )

    if cancel_cb:
        cancel_cb()

    from scripts.vram_utils import cleanup_pytorch_vram, log_vram
    log_vram("after GaussianWrapping subprocess exit")
    cleanup_pytorch_vram()

    # Locate the extracted mesh.
    _report(90.0, "Collecting output mesh")
    extracted_mesh = _find_extracted_mesh(model_dir, settings)
    if not extracted_mesh.exists():
        raise RuntimeError(
            f"GaussianWrapping produced no output mesh: {extracted_mesh}"
        )

    print(f"Selected GaussianWrapping mesh: {extracted_mesh}")

    # Step 3: Copy output mesh.
    _report(98.0, "Collecting output mesh")
    final_mesh = out / "object_mesh.ply"
    shutil.copy2(str(extracted_mesh), str(final_mesh))
    print(f"GaussianWrapping output mesh: {final_mesh}")

    _report(100.0, "GaussianWrapping reconstruction complete")
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


def _prepare_sparse_model(
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: Path,
    settings: GWrappingSettings,
    sparse_filter_settings: SparseFilterSettings,
    report,
) -> Path:
    """Return the reconstruction directory GaussianWrapping should consume."""
    recon_dir = _find_recon_dir(colmap_sparse_dir)
    if mask_dir is None or not settings.use_masks:
        return recon_dir
    if not sparse_filter_settings.enabled:
        return recon_dir

    report(0.5, "Filtering COLMAP sparse points with SAM2 masks")
    filtered_root = output_dir / "colmap_sparse_filtered"
    try:
        result = filter_colmap_sparse_model(
            recon_dir,
            Path(mask_dir),
            filtered_root,
            sparse_filter_settings,
        )
    except Exception as exc:
        print(
            "COLMAP sparse filter failed; using original sparse model: "
            f"{exc}"
        )
        report(0.8, "Filtered sparse unavailable, using original COLMAP sparse")
        return recon_dir

    if result.warning:
        print(f"COLMAP sparse filter fallback: {result.warning}")
    if result.used_filtered:
        print(
            f"COLMAP sparse filter kept "
            f"{result.kept_points}/{result.total_points} points across "
            f"{result.matched_images} masked images -> {result.filtered_recon_dir}"
        )
        return result.selected_recon_dir

    report(0.8, "Filtered sparse unavailable, using original COLMAP sparse")
    return recon_dir


def _reset_gwrapping_outputs(output_dir: Path) -> None:
    """Remove Stage 4 artifacts so each run starts from a clean workspace."""
    for path in (
        output_dir / "gwrapping_workspace",
        output_dir / "colmap_sparse_filtered",
        output_dir / "object_mesh.ply",
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


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
#  Source Directory Setup
# ---------------------------------------------------------------------------

def _setup_source_dir(
    gw_source: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> None:
    """Create GaussianWrapping source directory with symlinks."""
    gw_source.mkdir(parents=True, exist_ok=True)
    for name, target in [("images", images_dir), ("sparse", sparse_dir)]:
        link = gw_source / name
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
        mask_path = mask_map.get(img_path.stem)
        if mask_path is None:
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
            mask_float = mask_array.astype(np.float32) / 255.0
            img_array = (img_array * mask_float[:, :, np.newaxis]).astype(
                np.uint8
            )
            img = Image.fromarray(img_array)

        img.save(output_dir / img_path.name)

    print(f"Masked {len(image_files)} images -> {output_dir}")


# ---------------------------------------------------------------------------
#  GaussianWrapping Training + Extraction
# ---------------------------------------------------------------------------

def _build_train_args(
    source_path: Path,
    model_path: Path,
    settings: GWrappingSettings,
) -> list[str]:
    """Build the GaussianWrapping training command."""
    _require_supported_extraction_method(settings)
    args = [
        "python3",
        "-u",
        str(_GW_TRAIN),
        "-s", str(source_path),
        "-m", str(model_path),
        "-r", str(settings.resolution),
        "--iterations", str(settings.iterations),
        "--rasterizer", str(settings.rasterizer),
        "--regularization_from_iter", str(_resolve_regularization_start(settings)),
        "--data_device", "cpu",
        "--N_max_gaussians", "5000000",
    ]
    if settings.rasterizer == "radegs":
        args.extend([
            "--multiview_config", "fast_late",
            "--multiview_factor", "0.05",
            "--use_max_size_threshold",
        ])
    else:
        args.extend([
            "--feature_dc_lr", "0.0013",
            "--feature_rest_lr", "0.00011",
            "--exposure_compensation",
        ])
    return args


def _build_extract_args(
    source_path: Path,
    model_path: Path,
    settings: GWrappingSettings,
) -> list[str]:
    """Build the GaussianWrapping mesh extraction command."""
    _require_supported_extraction_method(settings)
    args = [
        "python3",
        "-u",
        str(_GW_EXTRACT),
        "-s", str(source_path),
        "-m", str(model_path),
        "-r", str(settings.resolution),
        "--iteration", str(settings.iterations),
        "--dtype", "int32",
        "--isosurface_value", "0.0",
        "--n_binary_steps", "10",
        "--postprocess",
        "--data_device", "cpu",
        "--rasterizer", str(settings.rasterizer),
    ]
    if settings.use_masks:
        args.append("--use_valid_mask")
    if settings.rasterizer == "radegs":
        args.extend([
            "--sdf_mode", "exact_computation",
            "--std_factor", "3.33",
            "--use_searched_pivots",
            "--search_iter", "5",
            "--search_step_size", "0.33",
        ])
    else:
        args.extend([
            "--sdf_mode", "ours",
            "--filter_large_edges",
        ])
    return args


def _build_texture_args(
    source_path: Path,
    model_path: Path,
    mesh_path: Path,
    settings: GWrappingSettings,
) -> list[str]:
    """Build the best-effort texture refinement command."""
    return [
        "python3",
        "-u",
        str(_GW_TEXTURE),
        "-s", str(source_path),
        "-m", str(model_path),
        "-r", str(settings.resolution),
        "--iteration", str(settings.iterations),
        "--rasterizer", str(settings.rasterizer),
        "--mesh", str(mesh_path),
    ]


def _resolve_regularization_start(settings: GWrappingSettings) -> int:
    """Map hidden use_depth to the training regularization schedule."""
    if not settings.use_depth:
        return settings.iterations + 1
    if settings.rasterizer == "radegs":
        return _GW_RADEGS_REGULARIZATION_FROM_ITER
    return _GW_OURS_REGULARIZATION_FROM_ITER


def _require_supported_extraction_method(settings: GWrappingSettings) -> None:
    if settings.extraction_method != "pivot":
        raise RuntimeError(
            "Unsupported GaussianWrapping extraction method: "
            f"{settings.extraction_method}"
        )


def _build_env() -> dict[str, str]:
    """Build environment for GaussianWrapping subprocess."""
    env = dict(os.environ)
    gw_paths = [
        str(_GW_BASE),
        str(_GW_BASE / "gaussian_wrapping"),
    ]
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in gw_paths if p not in existing]
    if parts:
        prefix = ":".join(parts)
        env["PYTHONPATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


_ITER_RE = re.compile(r"\[ITER\s+(\d+)\]", re.IGNORECASE)


def _parse_progress(
    line: str,
    total_iterations: int,
    report,
) -> None:
    """Parse GaussianWrapping training output for iteration progress."""
    m = _ITER_RE.search(line)
    if m and total_iterations > 0:
        current = int(m.group(1))
        frac = min(current / total_iterations, 1.0)
        pct = 5.0 + frac * 80.0
        report(pct, f"GaussianWrapping training iteration {current}/{total_iterations}")


def _find_extracted_mesh(
    model_dir: Path,
    settings: GWrappingSettings,
) -> Path:
    """Locate the highest-priority mesh file produced by GaussianWrapping."""
    base_name = _expected_mesh_basename(settings)
    preferred: list[Path] = [model_dir / f"{base_name}_post.ply"]
    preferred.extend(sorted(model_dir.glob(f"{base_name}_texture_refined_*.ply")))
    preferred.append(model_dir / f"{base_name}.ply")
    for candidate in preferred:
        if candidate.exists():
            return candidate

    fallback_candidates = [*model_dir.rglob("*.ply")]
    mesh_candidates = [
        path
        for path in fallback_candidates
        if "mesh" in path.stem.lower() or "extracted" in path.stem.lower()
    ]
    if mesh_candidates:
        selected = max(mesh_candidates, key=lambda path: path.stat().st_mtime)
        print(
            "GaussianWrapping output selection fell back to the most recent "
            f"mesh-like artifact: {selected}"
        )
        return selected
    if fallback_candidates:
        selected = max(fallback_candidates, key=lambda path: path.stat().st_mtime)
        print(
            "GaussianWrapping output selection fell back to the most recent PLY "
            f"artifact: {selected}"
        )
        return selected
    return model_dir / f"{base_name}.ply"


def _expected_mesh_basename(settings: GWrappingSettings) -> str:
    _require_supported_extraction_method(settings)
    if settings.rasterizer == "radegs":
        return "mesh_exact_computation_2pivots_searched"
    return "mesh_ours_2pivots"


# ---------------------------------------------------------------------------
#  Pipeline Helpers
# ---------------------------------------------------------------------------

def _run_gwrapping_pipeline(
    source_path: Path,
    model_path: Path,
    settings: GWrappingSettings,
    *,
    report,
    register_process=None,
    unregister_process=None,
) -> None:
    """Run GaussianWrapping training, extraction, and best-effort refinement."""
    env = _build_env()
    _run_subprocess(
        _build_train_args(source_path, model_path, settings),
        cwd=str(_GW_BASE),
        prefix="GWrapping",
        progress_fn=lambda line: _parse_progress(
            line, settings.iterations, report,
        ),
        register_process=register_process,
        unregister_process=unregister_process,
        error_msg="GaussianWrapping training failed",
        env=env,
    )

    report(86.0, "Extracting mesh from GaussianWrapping model")
    _run_subprocess(
        _build_extract_args(source_path, model_path, settings),
        cwd=str(_GW_BASE),
        prefix="GWrapping",
        register_process=register_process,
        unregister_process=unregister_process,
        error_msg="GaussianWrapping mesh extraction failed",
        env=env,
    )

    raw_mesh = model_path / f"{_expected_mesh_basename(settings)}.ply"
    if not raw_mesh.exists():
        print(
            "Skipping GaussianWrapping texture refinement because the expected "
            f"raw mesh is missing: {raw_mesh}"
        )
        return

    report(89.0, "Refining extracted mesh colors")
    try:
        _run_subprocess(
            _build_texture_args(source_path, model_path, raw_mesh, settings),
            cwd=str(_GW_BASE),
            prefix="GWrapping",
            register_process=register_process,
            unregister_process=unregister_process,
            error_msg="GaussianWrapping texture refinement failed",
            env=env,
        )
    except _SubprocessFailure as exc:
        print(f"GaussianWrapping texture refinement failed; continuing: {exc}")


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
