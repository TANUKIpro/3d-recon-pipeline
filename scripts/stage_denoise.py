"""Stage 4: Point cloud denoising (DBSCAN + SOR).

Directly reuses logic from im2pc/host/denoise_ply.py.
CPU-only stage, no VRAM needed.
"""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def _load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load PLY file. Returns (points (N,3), colors (N,3) or None)."""
    ply_data = PlyData.read(str(path))
    vertex = ply_data["vertex"]
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    colors = None
    if "red" in vertex.data.dtype.names:
        colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)
    return points, colors


def _save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None):
    """Save PLY file."""
    n = len(points)
    if colors is not None:
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8) if colors.max() <= 1.0 else colors.astype(np.uint8)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        data = np.empty(n, dtype=dtype)
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
        data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        data = np.empty(n, dtype=dtype)
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]

    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def _voxel_downsample(points, colors, voxel_size):
    """Voxel grid downsampling."""
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict: dict[tuple, list[int]] = {}
    for i, vid in enumerate(map(tuple, voxel_indices)):
        voxel_dict.setdefault(vid, []).append(i)

    ds_points, ds_colors, rep_idx = [], [] if colors is not None else None, []
    for indices_list in voxel_dict.values():
        idx_arr = np.array(indices_list)
        ds_points.append(points[idx_arr].mean(axis=0))
        rep_idx.append(indices_list[0])
        if ds_colors is not None:
            ds_colors.append(colors[idx_arr].mean(axis=0))

    result_pts = np.array(ds_points, dtype=np.float32)
    result_cols = np.array(ds_colors, dtype=colors.dtype) if ds_colors is not None else None
    return result_pts, result_cols, np.array(rep_idx)


def _dbscan_largest_cluster(points, colors, eps, min_samples=10, max_points=500000):
    """Extract largest DBSCAN cluster."""
    if len(points) < min_samples:
        return points, colors, 0

    original_pts, original_cols = points, colors
    mapping = None

    if len(points) > max_points:
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(bbox_extent)
        voxel_size = (volume / max_points) ** (1 / 3) * 0.8
        print(f"  Downsampling {len(points):,} to ~{max_points:,} for DBSCAN...")
        points, colors, mapping = _voxel_downsample(points, colors, voxel_size)
        print(f"  Downsampled to {len(points):,} points")

    print(f"  Running DBSCAN (eps={eps:.6f}, min_samples={min_samples})...")
    labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree', n_jobs=-1).fit_predict(points)

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


def _sor(points, colors, nb_neighbors=20, std_ratio=2.0):
    """Statistical Outlier Removal."""
    if len(points) <= nb_neighbors:
        return points, colors
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    inlier = mean_dists < threshold
    return points[inlier], colors[inlier] if colors is not None else None


def denoise(ply_path: str, output_dir: str) -> Path:
    """Denoise point cloud using DBSCAN + SOR.

    Args:
        ply_path: Path to input PLY.
        output_dir: Output directory.

    Returns:
        Path to denoised PLY.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {ply_path}")
    points, colors = _load_ply(ply_path)
    original_count = len(points)
    print(f"  {original_count:,} points loaded")

    if original_count == 0:
        out = output_path / "object_denoised.ply"
        _save_ply(out, points, colors)
        return out

    # Auto-tune DBSCAN eps
    bbox_extent = points.max(axis=0) - points.min(axis=0)
    eps = np.median(bbox_extent) * 0.02
    print(f"Auto-tuned DBSCAN eps: {eps:.6f}")

    # DBSCAN
    points, colors, n_clusters = _dbscan_largest_cluster(points, colors, eps)
    print(f"DBSCAN: {n_clusters} clusters, kept {len(points):,} points")

    # SOR
    before_sor = len(points)
    print("Running SOR...")
    points, colors = _sor(points, colors)
    print(f"SOR: {len(points):,} points ({before_sor - len(points):,} removed)")

    # Summary
    removed = original_count - len(points)
    print(f"\nSummary: {original_count:,} → {len(points):,} ({removed:,} removed, {100*removed/original_count:.1f}%)")

    out = output_path / "object_denoised.ply"
    _save_ply(out, points, colors)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Denoise point cloud")
    parser.add_argument("input_ply", help="Input PLY file")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    denoise(args.input_ply, args.output_dir)
