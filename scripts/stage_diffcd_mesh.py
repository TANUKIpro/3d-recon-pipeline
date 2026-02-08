"""Stage 5: DiffCD mesh reconstruction via subprocess.

Converts denoised PLY to NPY, runs DiffCD fit_implicit.py as a subprocess
(to avoid PyTorch/JAX CUDA context conflicts), then post-processes the mesh.

Based on im2pc/colab_diffcd.md and tmp/DiffCD.ipynb patterns.
"""

import math
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from vram_utils import (
    get_gpu_inventory,
    log_vram,
    pick_gpu_with_most_free_vram,
    prepare_for_jax,
)

ProgressCallback = Callable[[float, str | None], None]
_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_ITER_FRACTION_RE = re.compile(
    r"(?:batch|iter|step|epoch)[^\d]{0,16}(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
_GENERIC_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_OOM_MARKERS = (
    "out of memory",
    "resource exhausted",
    "cuda_error_out_of_memory",
    "cudnn_status_alloc_failed",
    "std::bad_alloc",
)

_DEFAULT_DIFFCD_BATCH_SIZE = 5000
_DEFAULT_DIFFCD_N_BATCHES = 30000
_DEFAULT_DIFFCD_RESOLUTION = 512
_MIN_DIFFCD_BATCH_SIZE = 500
_DIFFCD_BATCH_STEP = 100
_MIN_DIFFCD_N_BATCHES = 1000
_DEFAULT_AUTO_MIN_N_BATCHES = 10000
_MESH_SMOOTH_METHODS = {"laplacian", "taubin"}
_DEFAULT_MESH_SMOOTH_METHOD = "laplacian"
_DEFAULT_MESH_SMOOTH_ITERATIONS = 2
_DEFAULT_MESH_SMOOTH_LAMBDA = 0.5
_DEFAULT_MESH_SMOOTH_TAUBIN_NU = -0.53


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _extract_int(row: dict[str, str | int | None], key: str) -> int | None:
    value = row.get(key)
    return value if isinstance(value, int) else None


def _round_batch_size(value: float) -> int:
    rounded = int(round(value / _DIFFCD_BATCH_STEP) * _DIFFCD_BATCH_STEP)
    return max(_MIN_DIFFCD_BATCH_SIZE, rounded)


def _gpu_summary(gpu: dict[str, str | int | None] | None) -> str:
    if not gpu:
        return "nvidia-smi unavailable"
    idx = gpu.get("index")
    name = gpu.get("name") or "unknown"
    total = gpu.get("memory_total_mb")
    free = gpu.get("memory_free_mb")
    used = gpu.get("memory_used_mb")
    util = gpu.get("utilization_gpu_pct")
    return (
        f"GPU[{idx}] {name} "
        f"(used={used}MB, free={free}MB, total={total}MB, util={util}%)"
    )


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


def _load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = tuple(
            geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)
        )
        if not geoms:
            raise ValueError(f"No mesh geometry found: {mesh_path}")
        loaded = trimesh.util.concatenate(geoms)

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {type(loaded).__name__}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh is empty: {mesh_path}")
    return loaded


def mesh_vertex_face_count(mesh_path: str | Path) -> tuple[int, int]:
    mesh = _load_mesh(mesh_path)
    return len(mesh.vertices), len(mesh.faces)


def smooth_mesh_file(
    input_mesh_path: str | Path,
    output_mesh_path: str | Path,
    *,
    method: str = _DEFAULT_MESH_SMOOTH_METHOD,
    iterations: int = _DEFAULT_MESH_SMOOTH_ITERATIONS,
    lamb: float = _DEFAULT_MESH_SMOOTH_LAMBDA,
    taubin_nu: float = _DEFAULT_MESH_SMOOTH_TAUBIN_NU,
) -> Path:
    """Apply mesh smoothing and export as PLY."""
    method_norm = str(method or _DEFAULT_MESH_SMOOTH_METHOD).strip().lower()
    if method_norm not in _MESH_SMOOTH_METHODS:
        allowed = ", ".join(sorted(_MESH_SMOOTH_METHODS))
        raise ValueError(f"Unknown smoothing method '{method_norm}'. Expected one of: {allowed}")

    in_path = Path(input_mesh_path)
    out_path = Path(output_mesh_path)
    smoothed = _load_mesh(in_path).copy()
    iters = max(0, int(iterations))

    if iters > 0:
        if method_norm == "laplacian":
            filtered = trimesh.smoothing.filter_laplacian(
                smoothed,
                lamb=float(lamb),
                iterations=iters,
            )
        else:
            taubin_filter = getattr(trimesh.smoothing, "filter_taubin", None)
            if taubin_filter is None:
                raise RuntimeError("Taubin smoothing is not available in this trimesh version.")
            filtered = taubin_filter(
                smoothed,
                lamb=float(lamb),
                nu=float(taubin_nu),
                iterations=iters,
            )
        if isinstance(filtered, trimesh.Trimesh):
            smoothed = filtered

    out_path.parent.mkdir(parents=True, exist_ok=True)
    smoothed.export(str(out_path))
    return out_path


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


def _downsample_to_npy(
    ply_path: str,
    npy_path: str,
    target_points: int = 1_000_000,
) -> tuple[Path, int]:
    """Convert PLY to downsampled NPY (xyz float32) for DiffCD.

    Args:
        ply_path: Input PLY file.
        npy_path: Output NPY path.
        target_points: Target number of points after downsampling.

    Returns:
        Tuple of output NPY path and number of points.
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

        # Iterative adjustment to get within +-10% of target.
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
    return Path(npy_path), int(points.shape[0])


def _looks_like_oom(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in _OOM_MARKERS)


def _auto_tune_diffcd_params(
    batch_size: int,
    n_batches: int,
    resolution: int,
    gpu: dict[str, str | int | None] | None,
) -> tuple[int, int, str]:
    """Auto-tune DiffCD parameters from hardware while preserving fidelity."""
    if not _env_flag("DIFFCD_AUTO_TUNE", True):
        return batch_size, n_batches, "auto-tune disabled"

    if _env_flag("DIFFCD_AUTO_TUNE_RESPECT_MANUAL", True):
        if (
            batch_size != _DEFAULT_DIFFCD_BATCH_SIZE
            or n_batches != _DEFAULT_DIFFCD_N_BATCHES
            or resolution != _DEFAULT_DIFFCD_RESOLUTION
        ):
            return batch_size, n_batches, "manual DiffCD parameters detected"

    total_mb = _extract_int(gpu or {}, "memory_total_mb")
    free_mb = _extract_int(gpu or {}, "memory_free_mb")
    if total_mb is None or free_mb is None:
        return batch_size, n_batches, "GPU memory metrics unavailable"

    # Conservative heuristic from VRAM capacity and free memory.
    scale_total = max(0.75, (total_mb / 16_384) ** 0.9)
    scale_free = max(0.75, (free_mb / 12_000) ** 0.7)
    scale = min(scale_total, scale_free)

    if resolution >= 512:
        scale *= 0.92
    elif resolution <= 256:
        scale *= 1.05

    scale = min(
        _env_float("DIFFCD_AUTO_MAX_BATCH_SCALE", 2.4),
        max(_env_float("DIFFCD_AUTO_MIN_BATCH_SCALE", 0.75), scale),
    )

    tuned_batch = _round_batch_size(batch_size * scale)
    if tuned_batch <= int(batch_size * 1.08):
        return batch_size, n_batches, "GPU profile suggests default batch size"

    tuned_n_batches = n_batches
    if _env_flag("DIFFCD_AUTO_KEEP_EFFECTIVE_SAMPLES", True):
        baseline_samples = max(1, batch_size) * max(1, n_batches)
        computed = int(math.ceil(baseline_samples / max(1, tuned_batch)))
        min_n_batches = max(
            _MIN_DIFFCD_N_BATCHES,
            int(os.environ.get("DIFFCD_AUTO_MIN_N_BATCHES", _DEFAULT_AUTO_MIN_N_BATCHES)),
        )
        tuned_n_batches = min(n_batches, max(min_n_batches, computed))

    detail = (
        f"auto-tuned batch {batch_size}->{tuned_batch}, "
        f"n_batches {n_batches}->{tuned_n_batches} "
        f"(free={free_mb}MB, total={total_mb}MB, scale={scale:.2f})"
    )
    return tuned_batch, tuned_n_batches, detail


def _build_diffcd_attempts(
    primary_batch: int,
    primary_n_batches: int,
    baseline_batch: int,
    baseline_n_batches: int,
) -> list[tuple[int, int, str]]:
    attempts: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def _add(batch: int, n_batches: int, reason: str) -> None:
        b = _round_batch_size(batch)
        n = max(_MIN_DIFFCD_N_BATCHES, int(n_batches))
        key = (b, n)
        if key in seen:
            return
        seen.add(key)
        attempts.append((b, n, reason))

    _add(primary_batch, primary_n_batches, "auto/baseline")
    _add(baseline_batch, baseline_n_batches, "baseline")

    for ratio in (0.85, 0.70, 0.55):
        fallback_batch = int(baseline_batch * ratio)
        if fallback_batch < baseline_batch:
            _add(fallback_batch, baseline_n_batches, f"OOM fallback x{ratio:.2f}")

    return attempts


def _select_runtime_env(
    gpu_inventory: list[dict[str, str | int | None]],
    preferred_gpu: dict[str, str | int | None] | None,
) -> tuple[dict[str, str], str]:
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}

    # If user specifies explicit XLA fraction, honor it. Otherwise scale by headroom.
    mem_fraction = os.environ.get("DIFFCD_XLA_MEM_FRACTION")
    mem_fraction = mem_fraction.strip() if mem_fraction is not None else None
    if not mem_fraction:
        total_mb = _extract_int(preferred_gpu or {}, "memory_total_mb")
        free_mb = _extract_int(preferred_gpu or {}, "memory_free_mb")
        ratio = (free_mb / total_mb) if total_mb and free_mb else 0.0
        if ratio >= 0.90:
            mem_fraction = "0.92"
        elif ratio >= 0.75:
            mem_fraction = "0.88"
        else:
            mem_fraction = "0.80"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = mem_fraction

    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        "/root/.cache/jax_compilation_cache",
    )
    if cache_dir:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            env["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        except OSError:
            pass

    gpu_note = "using existing CUDA_VISIBLE_DEVICES"
    explicit_gpu = os.environ.get("DIFFCD_GPU_INDEX")
    if explicit_gpu:
        env["CUDA_VISIBLE_DEVICES"] = explicit_gpu.strip()
        gpu_note = f"forced CUDA_VISIBLE_DEVICES={explicit_gpu.strip()}"
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower()
        if (
            _env_flag("DIFFCD_AUTO_SELECT_GPU", True)
            and len(gpu_inventory) > 1
            and visible in {"", "all"}
        ):
            gpu = preferred_gpu or pick_gpu_with_most_free_vram(gpu_inventory)
            gpu_idx = _extract_int(gpu or {}, "index")
            if gpu_idx is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                gpu_note = f"auto-selected CUDA_VISIBLE_DEVICES={gpu_idx}"

    return env, gpu_note


def _run_diffcd_subprocess(
    cmd: list[str],
    env: dict[str, str],
    progress_cb: ProgressCallback | None = None,
    attempt_idx: int = 1,
    total_attempts: int = 1,
) -> tuple[int, bool, list[str]]:
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
    saw_oom = False
    tail_lines: deque[str] = deque(maxlen=80)

    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            tail_lines.append(line.rstrip())
            if _looks_like_oom(line):
                saw_oom = True

            ratio = _parse_progress_ratio(line)
            if ratio is None:
                continue

            stage_pct = 20.0 + ratio * 65.0
            if stage_pct - last_reported >= 0.5:
                suffix = (
                    f", attempt {attempt_idx}/{total_attempts}"
                    if total_attempts > 1
                    else ""
                )
                _emit_progress(
                    progress_cb,
                    stage_pct,
                    f"DiffCD fitting ({int(ratio * 100)}%{suffix})",
                )
                last_reported = stage_pct

    return process.wait(), saw_oom, list(tail_lines)


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
    batch_size = int(os.environ.get("DIFFCD_BATCH_SIZE", "5000"))
    n_batches = int(os.environ.get("DIFFCD_N_BATCHES", "30000"))
    resolution = int(os.environ.get("DIFFCD_RESOLUTION", "512"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    diffcd_dir = output_path / "diffcd"
    diffcd_dir.mkdir(exist_ok=True)

    # Step 1: Convert PLY to NPY
    npy_path = output_path / "object_points.npy"
    print("=== DiffCD: Preparing point cloud ===")
    _emit_progress(progress_cb, 5.0, "Preparing point cloud for DiffCD")
    npy_file, n_points = _downsample_to_npy(denoised_ply, str(npy_path))
    _emit_progress(progress_cb, 15.0, f"Point cloud prepared ({n_points:,} points)")

    # Step 2: Run DiffCD as subprocess
    print("\n=== DiffCD: Running implicit surface fitting ===")
    _emit_progress(progress_cb, 20.0, "Running DiffCD fitting")
    prepare_for_jax()
    log_vram("before DiffCD")

    gpu_inventory = get_gpu_inventory()
    preferred_gpu = pick_gpu_with_most_free_vram(gpu_inventory)
    print(f"DiffCD hardware: {_gpu_summary(preferred_gpu)}")

    tuned_batch, tuned_n_batches, tune_detail = _auto_tune_diffcd_params(
        batch_size,
        n_batches,
        resolution,
        preferred_gpu,
    )
    print(f"DiffCD tuning: {tune_detail}")

    attempts = _build_diffcd_attempts(
        tuned_batch,
        tuned_n_batches,
        batch_size,
        n_batches,
    )
    env, gpu_note = _select_runtime_env(gpu_inventory, preferred_gpu)
    print(
        "DiffCD runtime: "
        f"{gpu_note}, "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={env.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}, "
        f"JAX_COMPILATION_CACHE_DIR={env.get('JAX_COMPILATION_CACHE_DIR', '(disabled)')}"
    )

    success_dir: Path | None = None
    last_error_tail: list[str] = []
    total_attempts = len(attempts)
    for i, (try_batch, try_n_batches, reason) in enumerate(attempts, start=1):
        attempt_dir = diffcd_dir / f"run_{i:02d}_b{try_batch}_n{try_n_batches}"
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "/opt/diffcd/fit_implicit.py",
            "--output-dir",
            str(attempt_dir),
            "--dataset.path",
            str(npy_file),
            "--batch-size",
            str(try_batch),
            "--n-batches",
            str(try_n_batches),
            "--final-mesh-points-per-axis",
            str(resolution),
        ]

        print(
            f"\nDiffCD attempt {i}/{total_attempts} "
            f"({reason}; batch={try_batch}, n_batches={try_n_batches})"
        )
        print(f"Running: {' '.join(cmd)}")
        _emit_progress(
            progress_cb,
            20.0,
            f"Running DiffCD fitting (attempt {i}/{total_attempts})",
        )

        return_code, saw_oom, tail_lines = _run_diffcd_subprocess(
            cmd,
            env,
            progress_cb=progress_cb,
            attempt_idx=i,
            total_attempts=total_attempts,
        )

        if return_code == 0:
            success_dir = attempt_dir
            batch_size = try_batch
            n_batches = try_n_batches
            break

        last_error_tail = tail_lines
        if saw_oom and i < total_attempts:
            print(
                f"DiffCD attempt {i} failed with OOM-like error. "
                "Retrying with a safer batch size..."
            )
            continue

        tail_preview = "\n".join(last_error_tail[-20:])
        raise RuntimeError(
            f"DiffCD failed with return code {return_code} on attempt {i}/{total_attempts}.\n"
            f"{tail_preview}"
        )

    if success_dir is None:
        tail_preview = "\n".join(last_error_tail[-20:])
        raise RuntimeError(f"DiffCD failed after retries.\n{tail_preview}")

    print(
        f"DiffCD complete (batch={batch_size}, n_batches={n_batches}, resolution={resolution})"
    )

    log_vram("after DiffCD")
    _emit_progress(progress_cb, 87.0, "Collecting DiffCD outputs")

    # Step 3: Find output mesh
    # DiffCD puts output in a timestamped subdir: {out}/experiment.../meshes/mesh_N.ply
    # and also {out}/experiment.../mesh_final_N.ply
    mesh_files = sorted(success_dir.rglob("mesh_final_*.ply"))
    if not mesh_files:
        mesh_files = sorted(success_dir.rglob("mesh_final.ply"))
    if not mesh_files:
        mesh_files = sorted(success_dir.rglob("meshes/mesh_*.ply"))
    if not mesh_files:
        raise FileNotFoundError(f"DiffCD output mesh not found in {success_dir}")

    raw_mesh_path = mesh_files[-1]
    print(f"\nDiffCD output: {raw_mesh_path}")

    # Step 4: Post-process (smoothing)
    raw_copy_path = output_path / "object_mesh_raw.ply"
    shutil.copy2(raw_mesh_path, raw_copy_path)
    raw_vertices, raw_faces = mesh_vertex_face_count(raw_copy_path)
    print(f"Saved raw mesh: {raw_copy_path}")
    print(f"  Raw vertices: {raw_vertices:,}, Faces: {raw_faces:,}")

    smooth_method = os.environ.get(
        "DIFFCD_SMOOTH_METHOD",
        _DEFAULT_MESH_SMOOTH_METHOD,
    ).strip().lower()
    if smooth_method not in _MESH_SMOOTH_METHODS:
        print(
            f"Unknown DIFFCD_SMOOTH_METHOD='{smooth_method}', "
            f"fallback to '{_DEFAULT_MESH_SMOOTH_METHOD}'"
        )
        smooth_method = _DEFAULT_MESH_SMOOTH_METHOD
    smooth_iterations = max(
        0,
        _env_int("DIFFCD_SMOOTH_ITERATIONS", _DEFAULT_MESH_SMOOTH_ITERATIONS),
    )
    smooth_lambda = _env_float("DIFFCD_SMOOTH_LAMBDA", _DEFAULT_MESH_SMOOTH_LAMBDA)
    smooth_taubin_nu = _env_float("DIFFCD_SMOOTH_TAUBIN_NU", _DEFAULT_MESH_SMOOTH_TAUBIN_NU)

    detail = (
        f"Applying {smooth_method} smoothing "
        f"(iterations={smooth_iterations}, lambda={smooth_lambda:.3f}"
    )
    if smooth_method == "taubin":
        detail += f", nu={smooth_taubin_nu:.3f}"
    detail += ")..."
    print(detail)
    _emit_progress(progress_cb, 94.0, "Applying mesh smoothing")

    final_path = output_path / "object_mesh.ply"
    smooth_mesh_file(
        raw_copy_path,
        final_path,
        method=smooth_method,
        iterations=smooth_iterations,
        lamb=smooth_lambda,
        taubin_nu=smooth_taubin_nu,
    )
    smooth_vertices, smooth_faces = mesh_vertex_face_count(final_path)
    print(f"Saved: {final_path}")
    print(f"  Smoothed vertices: {smooth_vertices:,}, Faces: {smooth_faces:,}")
    _emit_progress(progress_cb, 100.0, "DiffCD stage complete")

    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiffCD mesh reconstruction")
    parser.add_argument("input_ply", help="Denoised PLY file")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    run_diffcd(args.input_ply, args.output_dir)
