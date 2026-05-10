"""Symmetric chamfer-distance comparison between two meshes.

Usage:
    python -m scripts.eval_chamfer \\
        --gt   /path/to/textured_mesh_cleaned.obj \\
        --pred /path/to/object_mesh.ply \\
        [--samples 100000] [--align icp|none] [--out report.json]

The lito_integration plan (§14) sets the final acceptance criterion at
chamfer-distance ≤ 30% of the gs2mesh pseudo-GT (i.e. lito mesh is at
least as close as 70% of the gs2mesh baseline). This script:

  1. Loads both meshes via trimesh (handles .obj/.ply uniformly).
  2. Optionally runs Open3D ICP to remove residual Sim(3) drift between
     the two reconstructions before evaluation.
  3. Samples points uniformly on each surface.
  4. Reports symmetric chamfer distance plus its normalised form
     (chamfer / gt_diagonal) so the 70% threshold is interpretable
     across object scales.

The script is deliberately self-contained — it imports Open3D only when
``--align icp`` is requested, so the trimesh-only path stays lightweight
for batch evaluation runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChamferReport:
    gt_path: str
    pred_path: str
    samples: int
    alignment: str
    chamfer_pred_to_gt: float
    chamfer_gt_to_pred: float
    chamfer_symmetric: float
    gt_diagonal: float
    pred_diagonal: float
    chamfer_normalised: float
    pass_threshold_normalised: float
    passed: bool


def _load_mesh_points(path: str, n_samples: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly from a mesh surface.

    Returns (points, aabb_extent) in mesh-native units.
    """
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty or mesh.faces.shape[0] == 0:
        raise ValueError(f"mesh {path} has no faces")
    pts, _ = trimesh.sample.sample_surface_even(mesh, n_samples, seed=seed)
    if pts.shape[0] < n_samples:
        # sample_surface_even may under-sample; top up with regular sampling.
        extra, _ = trimesh.sample.sample_surface(mesh, n_samples - pts.shape[0])
        pts = np.vstack([pts, extra])
    return pts.astype(np.float64), mesh.bounds


def _icp_align_pred_to_gt(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """ICP-align pred → gt and return transformed pred points."""
    import open3d as o3d

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(pred_points)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(gt_points)

    diag = float(np.linalg.norm(gt_points.max(0) - gt_points.min(0)))
    if threshold is None:
        threshold = max(diag * 0.05, 1e-3)

    init = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        threshold,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    H = np.asarray(result.transformation)
    homog = np.hstack([pred_points, np.ones((pred_points.shape[0], 1))])
    aligned = (H @ homog.T).T[:, :3]
    return aligned


def _nearest_distances(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-query nearest-neighbour distance to target (Euclidean)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    dists, _ = tree.query(query, k=1)
    return dists


def chamfer(
    gt_path: str,
    pred_path: str,
    *,
    samples: int = 100_000,
    alignment: str = "icp",
    pass_threshold_normalised: float = 0.30,
) -> ChamferReport:
    """Compute symmetric chamfer distance between gt and pred meshes."""
    gt_pts, gt_bounds = _load_mesh_points(gt_path, samples, seed=0)
    pred_pts, pred_bounds = _load_mesh_points(pred_path, samples, seed=1)

    gt_diag = float(np.linalg.norm(gt_bounds[1] - gt_bounds[0]))
    pred_diag = float(np.linalg.norm(pred_bounds[1] - pred_bounds[0]))

    if alignment == "icp":
        pred_pts = _icp_align_pred_to_gt(pred_pts, gt_pts)
    elif alignment != "none":
        raise ValueError(f"unknown alignment mode: {alignment}")

    d_pred_to_gt = float(_nearest_distances(pred_pts, gt_pts).mean())
    d_gt_to_pred = float(_nearest_distances(gt_pts, pred_pts).mean())
    sym = 0.5 * (d_pred_to_gt + d_gt_to_pred)
    normalised = sym / gt_diag if gt_diag > 0 else float("inf")

    return ChamferReport(
        gt_path=str(gt_path),
        pred_path=str(pred_path),
        samples=int(samples),
        alignment=alignment,
        chamfer_pred_to_gt=d_pred_to_gt,
        chamfer_gt_to_pred=d_gt_to_pred,
        chamfer_symmetric=sym,
        gt_diagonal=gt_diag,
        pred_diagonal=pred_diag,
        chamfer_normalised=normalised,
        pass_threshold_normalised=pass_threshold_normalised,
        passed=normalised <= pass_threshold_normalised,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, help="ground-truth (or pseudo-GT) mesh")
    p.add_argument("--pred", required=True, help="prediction mesh")
    p.add_argument("--samples", type=int, default=100_000)
    p.add_argument("--align", choices=("icp", "none"), default="icp")
    p.add_argument("--threshold", type=float, default=0.30,
                   help="pass threshold on chamfer / gt_diagonal "
                        "(default: 0.30 → lito ≥ 70%% of gs2mesh)")
    p.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = p.parse_args()

    report = chamfer(
        args.gt,
        args.pred,
        samples=args.samples,
        alignment=args.align,
        pass_threshold_normalised=args.threshold,
    )
    payload = asdict(report)
    print(json.dumps(payload, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
