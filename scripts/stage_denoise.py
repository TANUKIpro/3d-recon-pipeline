"""Stage 4: Point cloud denoising with selectable algorithms/presets.

CPU-only stage, no VRAM needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from config_defaults import (
    DENOISE_ALGORITHM_STEPS as ALGORITHM_STEPS,
    DENOISE_PRESET_DEFAULTS as PRESET_DEFAULTS,
)

ProgressCallback = Callable[[float, str | None], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


def _load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load PLY file. Returns (points (N,3), colors (N,3) or None)."""
    ply_data = PlyData.read(str(path))
    vertex = ply_data["vertex"]
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    colors = None
    if "red" in vertex.data.dtype.names:
        colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)
    return points, colors


def _save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Save PLY file."""
    n = len(points)
    if colors is not None:
        if colors.dtype != np.uint8:
            colors = (
                (np.clip(colors, 0, 1) * 255).astype(np.uint8)
                if colors.max() <= 1.0
                else colors.astype(np.uint8)
            )
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        data = np.empty(n, dtype=dtype)
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
        data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        data = np.empty(n, dtype=dtype)
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]

    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def _median_bbox_extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1e-6
    extent = points.max(axis=0) - points.min(axis=0)
    return max(float(np.median(extent)), 1e-6)


def _voxel_downsample(points: np.ndarray, colors: np.ndarray | None, voxel_size: float):
    """Voxel grid downsampling."""
    if voxel_size <= 0:
        return points, colors, np.arange(len(points))

    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict: dict[tuple[int, int, int], list[int]] = {}
    for i, vid in enumerate(map(tuple, voxel_indices)):
        voxel_dict.setdefault(vid, []).append(i)

    ds_points: list[np.ndarray] = []
    ds_colors: list[np.ndarray] | None = [] if colors is not None else None
    rep_idx: list[int] = []
    for indices_list in voxel_dict.values():
        idx_arr = np.array(indices_list)
        ds_points.append(points[idx_arr].mean(axis=0))
        rep_idx.append(indices_list[0])
        if ds_colors is not None and colors is not None:
            ds_colors.append(colors[idx_arr].mean(axis=0))

    result_pts = np.array(ds_points, dtype=np.float32)
    result_cols = np.array(ds_colors, dtype=colors.dtype) if ds_colors is not None and colors is not None else None
    return result_pts, result_cols, np.array(rep_idx)


def _dbscan_largest_cluster(
    points: np.ndarray,
    colors: np.ndarray | None,
    eps: float,
    min_samples: int = 10,
    max_points: int = 500000,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Extract largest DBSCAN cluster."""
    min_samples = max(1, int(min_samples))
    if len(points) < min_samples:
        return points, colors, 0

    original_pts, original_cols = points, colors
    mapping = None
    max_points = max(1000, int(max_points))

    if len(points) > max_points:
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        volume = float(np.prod(bbox_extent))
        if volume > 0:
            voxel_size = max((volume / max_points) ** (1 / 3) * 0.8, 1e-6)
            print(f"  Downsampling {len(points):,} to ~{max_points:,} for DBSCAN...")
            points, colors, mapping = _voxel_downsample(points, colors, voxel_size)
            print(f"  Downsampled to {len(points):,} points")

    eps = max(float(eps), 1e-6)
    print(f"  Running DBSCAN (eps={eps:.6f}, min_samples={min_samples})...")
    labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree", n_jobs=-1).fit_predict(points)

    cluster_labels = np.unique(labels)
    cluster_labels = cluster_labels[cluster_labels >= 0]
    if len(cluster_labels) == 0:
        return original_pts, original_cols, 0

    largest = max(cluster_labels, key=lambda l: np.sum(labels == l))

    if mapping is not None:
        tree = cKDTree(points)
        _, nearest = tree.query(original_pts, k=1)
        inlier = labels[nearest] == largest
        return original_pts[inlier], original_cols[inlier] if original_cols is not None else None, len(cluster_labels)

    inlier = labels == largest
    return points[inlier], colors[inlier] if colors is not None else None, len(cluster_labels)


def _sor(
    points: np.ndarray,
    colors: np.ndarray | None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Statistical Outlier Removal."""
    nb_neighbors = max(2, int(nb_neighbors))
    std_ratio = max(0.1, float(std_ratio))
    if len(points) <= nb_neighbors:
        return points, colors
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    inlier = mean_dists < threshold
    return points[inlier], colors[inlier] if colors is not None else None


def _radius_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray | None,
    radius: float,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Radius-based outlier removal."""
    radius = max(float(radius), 1e-6)
    min_neighbors = max(1, int(min_neighbors))
    if len(points) <= min_neighbors:
        return points, colors

    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, radius)
    counts = np.array([max(0, len(idx) - 1) for idx in neighbor_lists], dtype=np.int32)
    inlier = counts >= min_neighbors
    return points[inlier], colors[inlier] if colors is not None else None


def _resolve_denoise_config(
    preset: str,
    algorithm: str | None,
    dbscan_eps: float | None,
    dbscan_eps_ratio: float | None,
    dbscan_min_samples: int | None,
    dbscan_max_points: int | None,
    sor_neighbors: int | None,
    sor_std_ratio: float | None,
    radius_neighbors: int | None,
    radius_ratio: float | None,
) -> dict[str, float | int | str]:
    preset_key = preset if preset in PRESET_DEFAULTS else "balanced"
    cfg = dict(PRESET_DEFAULTS[preset_key])
    if algorithm in ALGORITHM_STEPS:
        cfg["algorithm"] = algorithm
    if dbscan_eps is not None:
        cfg["dbscan_eps"] = max(float(dbscan_eps), 0.0)
    if dbscan_eps_ratio is not None:
        cfg["dbscan_eps_ratio"] = max(float(dbscan_eps_ratio), 0.0001)
    if dbscan_min_samples is not None:
        cfg["dbscan_min_samples"] = max(int(dbscan_min_samples), 1)
    if dbscan_max_points is not None:
        cfg["dbscan_max_points"] = max(int(dbscan_max_points), 1000)
    if sor_neighbors is not None:
        cfg["sor_neighbors"] = max(int(sor_neighbors), 2)
    if sor_std_ratio is not None:
        cfg["sor_std_ratio"] = max(float(sor_std_ratio), 0.1)
    if radius_neighbors is not None:
        cfg["radius_neighbors"] = max(int(radius_neighbors), 1)
    if radius_ratio is not None:
        cfg["radius_ratio"] = max(float(radius_ratio), 0.0001)
    return cfg


def denoise(
    ply_path: str,
    output_dir: str,
    *,
    preset: str = "balanced",
    algorithm: str | None = None,
    dbscan_eps: float | None = None,
    dbscan_eps_ratio: float | None = None,
    dbscan_min_samples: int | None = None,
    dbscan_max_points: int | None = None,
    sor_neighbors: int | None = None,
    sor_std_ratio: float | None = None,
    radius_neighbors: int | None = None,
    radius_ratio: float | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Denoise point cloud with configurable algorithm chain."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {ply_path}")
    _emit_progress(progress_cb, 5.0, "Loading point cloud")
    points, colors = _load_ply(ply_path)
    original_count = len(points)
    print(f"  {original_count:,} points loaded")
    _emit_progress(progress_cb, 20.0, f"Loaded {original_count:,} points")

    if original_count == 0:
        out = output_path / "object_denoised.ply"
        _save_ply(out, points, colors)
        _emit_progress(progress_cb, 100.0, "No points to denoise")
        return out

    cfg = _resolve_denoise_config(
        preset,
        algorithm,
        dbscan_eps,
        dbscan_eps_ratio,
        dbscan_min_samples,
        dbscan_max_points,
        sor_neighbors,
        sor_std_ratio,
        radius_neighbors,
        radius_ratio,
    )
    algo = str(cfg["algorithm"])
    steps = ALGORITHM_STEPS.get(algo, ALGORITHM_STEPS["dbscan_sor"])
    print(f"Denoise config: preset={preset}, algorithm={algo}")

    step_start = 28.0
    step_end = 88.0
    step_span = max(1.0, step_end - step_start)
    step_count = max(1, len(steps))

    for idx, step in enumerate(steps):
        start = step_start + (step_span * idx / step_count)
        end = step_start + (step_span * (idx + 1) / step_count)

        if step == "dbscan":
            eps = float(cfg["dbscan_eps"])
            if eps <= 0:
                eps = _median_bbox_extent(points) * float(cfg["dbscan_eps_ratio"])
                print(f"Auto-tuned DBSCAN eps: {eps:.6f}")
            _emit_progress(progress_cb, start, "Running DBSCAN clustering")
            points, colors, n_clusters = _dbscan_largest_cluster(
                points,
                colors,
                eps=eps,
                min_samples=int(cfg["dbscan_min_samples"]),
                max_points=int(cfg["dbscan_max_points"]),
            )
            print(f"DBSCAN: {n_clusters} clusters, kept {len(points):,} points")
            _emit_progress(progress_cb, end, f"DBSCAN complete ({n_clusters} clusters)")
        elif step == "sor":
            before = len(points)
            _emit_progress(progress_cb, start, "Running statistical outlier removal")
            points, colors = _sor(
                points,
                colors,
                nb_neighbors=int(cfg["sor_neighbors"]),
                std_ratio=float(cfg["sor_std_ratio"]),
            )
            removed = before - len(points)
            print(f"SOR: {len(points):,} points ({removed:,} removed)")
            _emit_progress(progress_cb, end, f"SOR complete ({removed:,} removed)")
        elif step == "radius":
            before = len(points)
            radius = _median_bbox_extent(points) * float(cfg["radius_ratio"])
            _emit_progress(progress_cb, start, "Running radius outlier removal")
            points, colors = _radius_outlier_removal(
                points,
                colors,
                radius=radius,
                min_neighbors=int(cfg["radius_neighbors"]),
            )
            removed = before - len(points)
            print(f"Radius outlier: radius={radius:.6f}, kept {len(points):,} ({removed:,} removed)")
            _emit_progress(progress_cb, end, f"Radius outlier complete ({removed:,} removed)")

    _emit_progress(progress_cb, 95.0, "Saving denoised point cloud")
    removed = original_count - len(points)
    ratio = (100.0 * removed / original_count) if original_count > 0 else 0.0
    print(f"\nSummary: {original_count:,} -> {len(points):,} ({removed:,} removed, {ratio:.1f}%)")

    out = output_path / "object_denoised.ply"
    _save_ply(out, points, colors)
    print(f"Saved: {out}")
    _emit_progress(progress_cb, 100.0, "Denoise complete")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Denoise point cloud")
    parser.add_argument("input_ply", help="Input PLY file")
    parser.add_argument("--output-dir", default="/data/output")
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--dbscan-eps", type=float, default=None)
    parser.add_argument("--dbscan-eps-ratio", type=float, default=None)
    parser.add_argument("--dbscan-min-samples", type=int, default=None)
    parser.add_argument("--dbscan-max-points", type=int, default=None)
    parser.add_argument("--sor-neighbors", type=int, default=None)
    parser.add_argument("--sor-std-ratio", type=float, default=None)
    parser.add_argument("--radius-neighbors", type=int, default=None)
    parser.add_argument("--radius-ratio", type=float, default=None)
    args = parser.parse_args()

    denoise(
        args.input_ply,
        args.output_dir,
        preset=args.preset,
        algorithm=args.algorithm,
        dbscan_eps=args.dbscan_eps,
        dbscan_eps_ratio=args.dbscan_eps_ratio,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_max_points=args.dbscan_max_points,
        sor_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        radius_neighbors=args.radius_neighbors,
        radius_ratio=args.radius_ratio,
    )
