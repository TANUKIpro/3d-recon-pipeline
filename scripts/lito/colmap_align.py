"""Sim(3) alignment of LiTo canonical mesh to the COLMAP world frame.

LiTo emits its 3D Gaussians (and the mesh derived from them in Phase 3)
in an object-centric canonical frame whose origin and scale are not
related to the COLMAP reconstruction. To make the result usable by
Stage 5 texture baking we need a 7-DOF transform (rotation + translation
+ uniform scale) into the COLMAP world.

Strategy (.claude/plans/lito_integration.md §6.4 Step 5):
  1. Source point cloud = LiTo Gaussian centres (sub-sampled).
  2. Target point cloud = COLMAP sparse 3D points that project inside the
     SAM2 mask of the input frame chosen by the frame selector.
  3. Initial Sim(3) from per-axis PCA + centroid + standard-deviation
     ratios (no point correspondence required).
  4. Optional point-to-point ICP refinement via Open3D — rotation and
     translation only; scale is held at the PCA estimate.
  5. Apply the transform to the canonical mesh and emit the world-frame
     PLY at the standard Stage 4 location.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Sim3Transform:
    """7-DOF similarity transform: world ← canonical."""

    rotation: np.ndarray  # (3, 3) world ← canonical (orthonormal)
    translation: np.ndarray  # (3,) world ← canonical
    scale: float
    residual_rms: float
    inlier_count: int

    def matrix(self) -> np.ndarray:
        H = np.eye(4, dtype=np.float64)
        H[:3, :3] = self.scale * self.rotation
        H[:3, 3] = self.translation
        return H

    def apply_points(self, points_xyz: np.ndarray) -> np.ndarray:
        return (self.scale * (self.rotation @ points_xyz.T)).T + self.translation


@dataclass(frozen=True)
class AlignmentResult:
    """Phase 4 output: world-frame mesh + transform metadata."""

    world_mesh_path: str
    transform: Sim3Transform
    n_source_points: int
    n_target_points: int


# ---------------------------------------------------------------------------
# COLMAP sparse readers (extending scripts/stage_colmap_sfm.py:_read_images_binary
# to also return per-image 2D→3D correspondences and the 3D point cloud).
# ---------------------------------------------------------------------------

def _read_images_with_points(path: Path) -> dict[int, dict]:
    """Read images.bin, keeping the (xy → point3D_id) correspondences."""
    images: dict[int, dict] = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # null-terminated name
            name_chars: list[str] = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            name = "".join(name_chars)

            num_points2d = struct.unpack("<Q", f.read(8))[0]
            xys = np.empty((num_points2d, 2), dtype=np.float64)
            point3d_ids = np.empty(num_points2d, dtype=np.int64)
            for i in range(num_points2d):
                x, y, p3d = struct.unpack("<dd q", f.read(24))
                xys[i, 0] = x
                xys[i, 1] = y
                point3d_ids[i] = p3d
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": xys,
                "point3d_ids": point3d_ids,
            }
    return images


def _read_points3d(path: Path) -> dict[int, np.ndarray]:
    """Read points3D.bin → {point3d_id: xyz}.

    COLMAP layout per record: id u64, xyz 3×float64, rgb 3×uint8,
    error float64, track_length u64, then `track_length` (u32, u32) entries.
    """
    out: dict[int, np.ndarray] = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            p3d_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)  # rgb (uint8 ×3)
            f.read(8)  # error (double)
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)  # (image_id u32, point2d_idx u32) per entry
            out[int(p3d_id)] = np.array(xyz, dtype=np.float64)
    return out


def _find_sparse_subdir(sparse_dir: Path) -> Path:
    """Return the first sparse reconstruction sub-folder (0/, 1/, …)."""
    for child in sorted(sparse_dir.iterdir()):
        if (child / "images.bin").exists() and (child / "points3D.bin").exists():
            return child
    raise FileNotFoundError(
        f"COLMAP sparse model not found under {sparse_dir} "
        f"(expected images.bin + points3D.bin)"
    )


def _load_foreground_3d_points(
    sparse_dir: str,
    frame_path: str,
    mask_path: str,
) -> np.ndarray:
    """Pick COLMAP 3D points that project into the input frame's SAM2 mask."""
    from PIL import Image

    sparse = _find_sparse_subdir(Path(sparse_dir))
    images = _read_images_with_points(sparse / "images.bin")
    points3d = _read_points3d(sparse / "points3D.bin")

    frame_name = Path(frame_path).name
    image_entry = next(
        (img for img in images.values() if img["name"] == frame_name),
        None,
    )
    if image_entry is None:
        raise ValueError(
            f"frame {frame_name} not found in COLMAP images.bin under {sparse}"
        )

    mask = np.array(Image.open(mask_path).convert("L"))
    h, w = mask.shape
    xs = image_entry["xys"]
    ids = image_entry["point3d_ids"]
    out: list[np.ndarray] = []
    for (x, y), p3d_id in zip(xs, ids):
        if p3d_id < 0:
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        if mask[yi, xi] == 0:
            continue
        if p3d_id not in points3d:
            continue
        out.append(points3d[p3d_id])
    if not out:
        raise RuntimeError(
            f"no foreground 3D points found for frame {frame_name} "
            f"(mask coverage may be too tight or sparse model lacks coverage)"
        )
    return np.stack(out, axis=0)


def _load_gaussian_centres(canonical_ply: str, max_points: int = 5000) -> np.ndarray:
    """Read xyz from the LiTo Gaussians PLY, sub-sampled to keep ICP cheap."""
    from plyfile import PlyData

    plydata = PlyData.read(canonical_ply)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"], dtype=np.float64),
            np.asarray(plydata.elements[0]["y"], dtype=np.float64),
            np.asarray(plydata.elements[0]["z"], dtype=np.float64),
        ),
        axis=1,
    )
    # Drop opacity-zero points if the field is present (LiTo writes opacity
    # as logits in its PLY; ultra-low values correspond to "off" Gaussians).
    if any(p.name == "opacity" for p in plydata.elements[0].properties):
        opacity = np.asarray(plydata.elements[0]["opacity"], dtype=np.float64)
        keep = opacity > -10.0  # logit(1e-5) ≈ -11.5
        if int(keep.sum()) > 0:
            xyz = xyz[keep]
    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
    return xyz


# ---------------------------------------------------------------------------
# PCA initial Sim(3) and ICP refinement
# ---------------------------------------------------------------------------

def _pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA on a centred point cloud. Returns (centroid, principal_axes_3x3, std_3)."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    cov = centred.T @ centred / max(centred.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    # Reorder to descending so axes[:, 0] is the dominant axis.
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    std = np.sqrt(np.clip(eigvals, 0.0, None))
    return centroid, eigvecs, std


def _best_rotation_from_axes(
    src_axes: np.ndarray, tgt_axes: np.ndarray
) -> np.ndarray:
    """Brute-force pick the sign-flip of axes that minimises Frobenius residual.

    Each principal axis has a sign ambiguity. We test the 8 possible
    sign combinations (2³) and keep the rotation matrix with the smallest
    deviation from a proper rotation (det = +1, orthogonal).
    """
    best_R = np.eye(3)
    best_err = float("inf")
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                S = np.diag([sx, sy, sz])
                R = tgt_axes @ S @ src_axes.T
                # Force orthogonal proper rotation via SVD.
                U, _, Vt = np.linalg.svd(R)
                if np.linalg.det(U @ Vt) < 0:
                    Vt[-1, :] *= -1
                R_ortho = U @ Vt
                err = np.linalg.norm(R - R_ortho, "fro")
                if err < best_err:
                    best_err = err
                    best_R = R_ortho
    return best_R


def pca_initial_sim3(source: np.ndarray, target: np.ndarray) -> Sim3Transform:
    """PCA-based 7-DOF similarity that maps `source` onto `target` roughly."""
    if source.shape[0] < 3 or target.shape[0] < 3:
        raise ValueError(
            f"pca_initial_sim3 needs ≥3 points per cloud "
            f"(got source={source.shape[0]}, target={target.shape[0]})"
        )
    src_c, src_axes, src_std = _pca_axes(source)
    tgt_c, tgt_axes, tgt_std = _pca_axes(target)
    R = _best_rotation_from_axes(src_axes, tgt_axes)

    src_scale = float(np.mean(src_std))
    tgt_scale = float(np.mean(tgt_std))
    if src_scale < 1e-12:
        scale = 1.0
    else:
        scale = tgt_scale / src_scale
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    t = tgt_c - scale * (R @ src_c)

    aligned = (scale * (R @ source.T)).T + t
    # Per-source residual to the nearest target point — coarse but useful.
    # Use a kd-tree only if target is large; otherwise brute-force.
    if target.shape[0] <= 4096:
        d2 = ((aligned[:, None, :] - target[None, :, :]) ** 2).sum(-1)
        nn = d2.min(axis=1)
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(target)
        nn, _ = tree.query(aligned, k=1)
        nn = nn ** 2
    rms = float(math.sqrt(nn.mean()))
    return Sim3Transform(
        rotation=R,
        translation=t,
        scale=scale,
        residual_rms=rms,
        inlier_count=int(source.shape[0]),
    )


def _icp_refine(
    source: np.ndarray,
    target: np.ndarray,
    init: Sim3Transform,
    *,
    max_iters: int,
    threshold: float,
) -> Sim3Transform:
    """Open3D point-to-point ICP refinement; scale held at init.scale."""
    import open3d as o3d

    src_pre = (init.scale * (init.rotation @ source.T)).T + init.translation
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_pre)
    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(target)

    reg = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_tgt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iters)
        ),
    )
    delta = np.asarray(reg.transformation, dtype=np.float64)
    R_delta = delta[:3, :3]
    t_delta = delta[:3, 3]
    R_total = R_delta @ init.rotation
    t_total = R_delta @ init.translation + t_delta
    aligned = (init.scale * (R_total @ source.T)).T + t_total
    if target.shape[0] <= 4096:
        d2 = ((aligned[:, None, :] - target[None, :, :]) ** 2).sum(-1)
        nn = d2.min(axis=1)
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(target)
        nn, _ = tree.query(aligned, k=1)
        nn = nn ** 2
    correspondences = int(np.asarray(reg.correspondence_set).shape[0])
    return Sim3Transform(
        rotation=R_total,
        translation=t_total,
        scale=init.scale,
        residual_rms=float(math.sqrt(nn.mean())),
        inlier_count=correspondences,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def align_canonical_mesh_to_world(
    canonical_ply_path: str,
    canonical_mesh_path: str,
    sparse_dir: str,
    frame_path: str,
    mask_path: str,
    out_world_mesh_path: str,
    *,
    icp_max_iters: int = 50,
    icp_threshold: float | None = None,
    workspace: str | None = None,
) -> AlignmentResult:
    """Compute Sim(3) and write the world-frame mesh to out_world_mesh_path."""
    src = _load_gaussian_centres(canonical_ply_path)
    tgt = _load_foreground_3d_points(sparse_dir, frame_path, mask_path)

    init = pca_initial_sim3(src, tgt)
    print(
        f"[lito-align] PCA init: scale={init.scale:.4f} residual_rms={init.residual_rms:.4f}"
    )

    threshold = icp_threshold
    if threshold is None:
        # 5 % of target diameter — a rough but defensible default.
        diam = float(np.linalg.norm(tgt.max(0) - tgt.min(0)))
        threshold = max(diam * 0.05, 1e-3)

    try:
        refined = _icp_refine(
            src, tgt, init, max_iters=icp_max_iters, threshold=threshold
        )
        print(
            f"[lito-align] ICP refine: residual_rms={refined.residual_rms:.4f}"
        )
    except ImportError as exc:
        print(f"[lito-align] ICP unavailable ({exc}); using PCA init")
        refined = init
    except Exception as exc:  # pragma: no cover
        print(f"[lito-align] ICP failed ({exc}); falling back to PCA init")
        refined = init

    # Apply transform to the canonical mesh and write world-frame PLY.
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(canonical_mesh_path)
    H = refined.matrix()
    mesh.transform(H)
    Path(out_world_mesh_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_world_mesh_path), mesh)

    if workspace is not None:
        Path(workspace).mkdir(parents=True, exist_ok=True)
        (Path(workspace) / "alignment.json").write_text(
            json.dumps(
                {
                    "rotation": refined.rotation.tolist(),
                    "translation": refined.translation.tolist(),
                    "scale": refined.scale,
                    "residual_rms": refined.residual_rms,
                    "n_source_points": int(src.shape[0]),
                    "n_target_points": int(tgt.shape[0]),
                    "icp_threshold": threshold,
                },
                indent=2,
            )
        )

    return AlignmentResult(
        world_mesh_path=str(out_world_mesh_path),
        transform=refined,
        n_source_points=int(src.shape[0]),
        n_target_points=int(tgt.shape[0]),
    )
