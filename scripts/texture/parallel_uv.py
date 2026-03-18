"""Spatial-partition parallel xatlas UV atlas generation.

Splits a mesh into N partitions along the longest axis, runs
xatlas chart generation in parallel processes, then tiles all
partitions into a single atlas via grid-based UV placement.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial partitioning
# ---------------------------------------------------------------------------

def _spatial_partition_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_partitions: int,
) -> list[np.ndarray]:
    """Recursively bisect faces along the longest centroid axis.

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) face indices.
        n_partitions: Target number of partitions (rounded up to next power of 2).

    Returns:
        List of face-index arrays (each a 1-D ``int64`` array).
    """
    # Round to next power of two
    n = 1
    while n < n_partitions:
        n *= 2
    n_partitions = n

    centroids = (
        vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]
    ) / 3.0

    def _bisect(indices: np.ndarray, depth: int) -> list[np.ndarray]:
        if depth == 0 or len(indices) <= 1:
            return [indices]
        ctrs = centroids[indices]
        # Longest axis
        axis = int(np.argmax(ctrs.max(axis=0) - ctrs.min(axis=0)))
        median = np.median(ctrs[:, axis])
        left_mask = ctrs[:, axis] <= median
        left = indices[left_mask]
        right = indices[~left_mask]
        # Avoid empty splits
        if len(left) == 0 or len(right) == 0:
            return [indices]
        return _bisect(left, depth - 1) + _bisect(right, depth - 1)

    import math
    depth = int(math.log2(n_partitions))
    all_indices = np.arange(len(faces), dtype=np.int64)
    return _bisect(all_indices, depth)


# ---------------------------------------------------------------------------
# Sub-mesh extraction
# ---------------------------------------------------------------------------

def _extract_partition_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a compact sub-mesh for the given face subset.

    Returns:
        (local_vertices, local_faces, local_normals, vertex_remap)
        where ``vertex_remap[local_idx] == global_idx``.
    """
    sub_faces = faces[face_indices]
    unique_verts, inverse = np.unique(sub_faces.ravel(), return_inverse=True)
    local_faces = inverse.reshape(-1, 3)
    local_vertices = vertices[unique_verts]
    local_normals = normals[unique_verts]
    return local_vertices, local_faces, local_normals, unique_verts


# ---------------------------------------------------------------------------
# Worker (top-level, pickle-safe)
# ---------------------------------------------------------------------------

def _xatlas_worker(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    pack_opts_dict: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run xatlas chart generation + packing on a single sub-mesh.

    Must be a module-level function so ``ProcessPoolExecutor`` can pickle it.
    """
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        vertices.astype(np.float32),
        faces.astype(np.uint32),
        normals=normals.astype(np.float32),
    )
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    for k, v in pack_opts_dict.items():
        setattr(pack_options, k, v)
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, new_faces, uvs = atlas[0]
    return vmapping, new_faces, uvs


# ---------------------------------------------------------------------------
# Tile & combine (grid-based UV placement)
# ---------------------------------------------------------------------------

def _tile_and_combine(
    parallel_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertex_remaps: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile per-partition UVs into a single atlas using grid placement.

    Each partition's UVs are scaled and offset into a grid cell, avoiding
    the fragile triple-indirection of ``add_uv_mesh`` repacking.
    Only one vmapping hop is needed: partition UV vert → global mesh vert.
    """
    import math

    n = len(parallel_results)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    # Leave a small margin within each cell to avoid bleeding
    margin = 0.005
    inner_w = cell_w * (1.0 - 2.0 * margin)
    inner_h = cell_h * (1.0 - 2.0 * margin)

    all_vmapping = []
    all_new_faces = []
    all_uvs = []
    uv_vertex_offset = 0

    for i, (per_part_vm, per_part_nf, per_part_uv) in enumerate(parallel_results):
        # Map partition UV vertex → global mesh vertex (single hop)
        chained_vm = vertex_remaps[i][per_part_vm]
        all_vmapping.append(chained_vm)

        # Offset face indices into the combined UV vertex array
        all_new_faces.append(per_part_nf + uv_vertex_offset)

        # Scale UVs into this partition's grid cell
        col = i % cols
        row = i // cols
        scaled_uv = per_part_uv.copy()
        scaled_uv[:, 0] = scaled_uv[:, 0] * inner_w + col * cell_w + cell_w * margin
        scaled_uv[:, 1] = scaled_uv[:, 1] * inner_h + row * cell_h + cell_h * margin
        all_uvs.append(scaled_uv)

        uv_vertex_offset += len(per_part_uv)

    return (
        np.concatenate(all_vmapping),
        np.concatenate(all_new_faces),
        np.concatenate(all_uvs),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parallel_xatlas_generate(
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    vert_normals: np.ndarray,
    n_workers: int | None = None,
    progress_cb: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Generate UV atlas using spatial-partition parallelism.

    Returns ``(vmapping, new_faces, uvs)`` or ``None`` if parallel
    execution should be skipped (too few faces, disabled, error).
    """
    from scripts.config_defaults import (
        _TEXTURE_UV_PARALLEL_MAX_WORKERS,
        _TEXTURE_UV_PARALLEL_MIN_TOTAL_FACES,
    )
    from scripts.texture.config import _resolve_parallel_uv

    mode = _resolve_parallel_uv()
    if mode == "off":
        return None

    n_faces = len(uv_faces)
    if n_faces < _TEXTURE_UV_PARALLEL_MIN_TOTAL_FACES:
        logger.info(
            "Parallel UV: skipping (%d faces < %d threshold)",
            n_faces,
            _TEXTURE_UV_PARALLEL_MIN_TOTAL_FACES,
        )
        return None

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, _TEXTURE_UV_PARALLEL_MAX_WORKERS)
    if n_workers < 2:
        return None

    n_partitions = n_workers

    pack_opts_dict = {"blockAlign": True, "padding": 1}

    try:
        t_total = time.monotonic()

        # Step 1: spatial partition
        t0 = time.monotonic()
        partitions = _spatial_partition_faces(uv_vertices, uv_faces, n_partitions)
        actual_parts = len(partitions)
        dt_partition = time.monotonic() - t0
        part_sizes = [len(p) for p in partitions]
        logger.info(
            "Parallel UV: %d faces → %d partitions (%d workers) [%.2fs]",
            n_faces,
            actual_parts,
            n_workers,
            dt_partition,
        )
        logger.info(
            "  Partition sizes: min=%d, max=%d, avg=%d",
            min(part_sizes),
            max(part_sizes),
            sum(part_sizes) // len(part_sizes),
        )
        print(
            f"  Parallel UV: {n_faces:,} faces → {actual_parts} partitions "
            f"(workers={n_workers}, sizes={min(part_sizes):,}..{max(part_sizes):,})"
        )

        # Step 2: extract sub-meshes
        t0 = time.monotonic()
        sub_meshes = []
        vertex_remaps = []
        for part_indices in partitions:
            local_v, local_f, local_n, v_remap = _extract_partition_mesh(
                uv_vertices, uv_faces, vert_normals, part_indices,
            )
            sub_meshes.append((local_v, local_f, local_n))
            vertex_remaps.append(v_remap)
        dt_extract = time.monotonic() - t0
        logger.info("  Sub-mesh extraction: %.2fs", dt_extract)

        # Step 3: parallel xatlas generation
        t0 = time.monotonic()
        effective_workers = min(n_workers, actual_parts)
        print(
            f"  Running xatlas chart+pack on {actual_parts} partitions "
            f"({effective_workers} parallel workers)..."
        )
        parallel_results: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [
            None
        ] * len(sub_meshes)
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            future_to_idx = {
                pool.submit(_xatlas_worker, v, f, n, pack_opts_dict): i
                for i, (v, f, n) in enumerate(sub_meshes)
            }
            done_count = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                parallel_results[idx] = fut.result()
                done_count += 1
                vm, nf, uv = parallel_results[idx]  # type: ignore[misc]
                dt_so_far = time.monotonic() - t0
                logger.info(
                    "  Partition %d/%d done: %d faces → %d UV verts, %d UV faces [%.1fs elapsed]",
                    done_count,
                    actual_parts,
                    len(sub_meshes[idx][1]),
                    len(uv),
                    len(nf),
                    dt_so_far,
                )
                print(
                    f"  Partition {done_count}/{actual_parts} complete "
                    f"({len(sub_meshes[idx][1]):,} faces → {len(uv):,} UV verts) "
                    f"[{dt_so_far:.1f}s]"
                )
                if progress_cb is not None:
                    # Map partition progress to 58..68% range (UV atlas stage)
                    from scripts.texture.progress import _emit_progress
                    ratio = done_count / actual_parts
                    _emit_progress(
                        progress_cb,
                        58.0 + ratio * 8.0,
                        f"UV atlas: partition {done_count}/{actual_parts}",
                    )
        dt_generate = time.monotonic() - t0
        print(f"  All partitions done in {dt_generate:.1f}s")
        logger.info("  Parallel xatlas generation: %.2fs", dt_generate)

        # Step 4: tile partitions into single atlas (grid-based UV placement)
        t0 = time.monotonic()
        print("  Tiling partitions into single atlas...")
        vmapping, new_faces, uvs = _tile_and_combine(
            parallel_results, vertex_remaps,  # type: ignore[arg-type]
        )
        dt_repack = time.monotonic() - t0
        dt_total = time.monotonic() - t_total

        logger.info(
            "Parallel UV: done — %d UV vertices, %d faces [repack=%.2fs, total=%.2fs]",
            len(uvs),
            len(new_faces),
            dt_repack,
            dt_total,
        )
        print(
            f"  UV atlas complete: {len(uvs):,} UV vertices, {len(new_faces):,} faces "
            f"[repack={dt_repack:.1f}s, total={dt_total:.1f}s]"
        )
        if progress_cb is not None:
            from scripts.texture.progress import _emit_progress
            _emit_progress(progress_cb, 68.0, "UV atlas generation complete")
        return vmapping, new_faces, uvs

    except Exception:
        warnings.warn(
            "Parallel UV atlas generation failed; falling back to single-threaded",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.exception("Parallel UV failed")
        return None
