"""Gaussians → mesh in the LiTo canonical frame.

This is Phase 3 of the lito backend (see .claude/plans/lito_integration.md
§6.4 Step 4). The flow is:

    1. Sample Fibonacci sphere views (+ the input view as #0).
    2. Subprocess into /opt/ml-lito/.venv to render gsplat depth/rgb/alpha
       per pose.
    3. Compute per-view confidence from alpha and the angular distance to
       the input view (closer view = more trustworthy). Confidence is a
       hard-mask threshold for tsdf_core.fuse_tsdf.
    4. Fuse into a TSDF and return the canonical-frame mesh.

The mesh stays in the LiTo canonical frame; Phase 4 (Sim(3) alignment)
maps it into COLMAP world coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CanonicalMeshResult:
    """Output of gaussians_to_canonical_mesh."""

    mesh_path: str
    n_views_used: int
    n_views_skipped: int
    triangle_count: int
    vertex_count: int


def _angular_confidence(
    view_position: np.ndarray,
    input_position: np.ndarray,
    *,
    decay: float = 1.0,
) -> float:
    """Cosine-similarity-based per-view confidence.

    `decay` controls how quickly trust falls away from the input view:
      * decay=0 -> all views weighted equally (cosine collapses to 1.0)
      * decay=1 -> standard cosine (back of object = 0)
      * decay>1 -> sharper falloff
    """
    a = view_position / (np.linalg.norm(view_position) + 1e-12)
    b = input_position / (np.linalg.norm(input_position) + 1e-12)
    cos = float(np.clip(a @ b, -1.0, 1.0))
    raw = max(0.0, 0.5 * (cos + 1.0))  # remap [-1, 1] → [0, 1]
    return float(raw ** max(decay, 1e-6))


def _load_view_artefacts(
    out_dir: Path, index: int
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    rgb_path = out_dir / f"rgb_{index:04d}.png"
    depth_path = out_dir / f"depth_{index:04d}.npy"
    alpha_path = out_dir / f"alpha_{index:04d}.npy"
    meta_path = out_dir / f"meta_{index:04d}.json"
    if not (rgb_path.exists() and depth_path.exists() and alpha_path.exists()):
        return None
    from PIL import Image

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    alpha = np.load(alpha_path).astype(np.float32)
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    return rgb, depth, alpha, meta


def gaussians_to_canonical_mesh(
    canonical_ply_path: str,
    workspace: str,
    *,
    num_synth_views: int = 60,
    sphere_radius: float = 2.0,
    fov_y_deg: float = 40.0,
    image_size: tuple[int, int] = (512, 512),
    input_view_position: tuple[float, float, float] = (0.0, 0.0, 2.0),
    voxel_size: float = 0.01,
    sdf_trunc: float | None = None,
    depth_min: float = 0.3,
    depth_max: float = 5.0,
    confidence_min: float = 0.05,
    confidence_decay: float = 0.7,
    alpha_floor: float = 0.05,
    cleaning_threshold: int = 100,
    target_faces: int = 200_000,
    smooth_iters: int = 5,
    device: str = "CUDA:0",
    render_device: str = "cuda:0",
) -> CanonicalMeshResult:
    """Render Gaussians multi-view, fuse into TSDF, return canonical-frame mesh."""
    from scripts.lito.lito_runner import run_lito_render
    from scripts.lito.tsdf_core import TsdfFusionParams, TsdfView, fuse_tsdf
    from scripts.lito.view_sampler import sample_canonical_views, save_views_json

    workspace_p = Path(workspace)
    workspace_p.mkdir(parents=True, exist_ok=True)
    sdf_trunc = sdf_trunc if sdf_trunc is not None else 4.0 * voxel_size

    views = sample_canonical_views(
        n=num_synth_views,
        radius=sphere_radius,
        fov_y_deg=fov_y_deg,
        image_size=image_size,
        input_view_position=input_view_position,
    )
    views_json_path = workspace_p / "render_views.json"
    save_views_json(views, views_json_path)

    render_dir = workspace_p / "tsdf_views"
    run_lito_render(
        in_ply_path=canonical_ply_path,
        views_json_path=str(views_json_path),
        out_dir=str(render_dir),
        device=render_device,
    )

    input_pos_arr = np.array(input_view_position, dtype=np.float64)

    tsdf_views: list[TsdfView] = []
    skipped = 0
    for v in views:
        loaded = _load_view_artefacts(render_dir, v.index)
        if loaded is None:
            skipped += 1
            continue
        rgb, depth, alpha, meta = loaded

        valid_mask = (alpha > alpha_floor) & (depth > 0.0)
        if int(valid_mask.sum()) == 0:
            skipped += 1
            continue
        # Zero out invalid pixels so depth_min/max gating in fuse_tsdf works.
        depth = np.where(valid_mask, depth, np.float32(0.0))

        ang = _angular_confidence(
            np.array(v.position, dtype=np.float64),
            input_pos_arr,
            decay=confidence_decay,
        )
        confidence = (alpha * ang).astype(np.float32)

        # T_cw expected by tsdf_core is world→camera. view_sampler stores
        # camera→world (H_c2w), so invert here.
        T_cw = np.linalg.inv(v.H_c2w)

        tsdf_views.append(
            TsdfView(
                rgb=rgb,
                depth=depth,
                K=v.K,
                T_cw=T_cw,
                confidence=confidence,
                confidence_min=confidence_min,
            )
        )

    if not tsdf_views:
        raise RuntimeError(
            "lito Gaussians rendering produced no usable views — all were "
            "skipped after the alpha/depth gate"
        )

    params = TsdfFusionParams(
        voxel_size=float(voxel_size),
        sdf_trunc=float(sdf_trunc),
        depth_min=float(depth_min),
        depth_max=float(depth_max),
        device=device,
        block_count=100_000,
        cleaning_threshold=int(cleaning_threshold),
        tsdf_scale=1.0,
        smooth_iters=int(smooth_iters),
        target_faces=int(target_faces),
    )
    mesh = fuse_tsdf(tsdf_views, params)
    if mesh is None:
        raise RuntimeError(
            "lito Gaussians → mesh: TSDF integration produced no output. "
            "Check synth-view depth ranges (depth_min/depth_max) and "
            "alpha_floor."
        )

    canonical_mesh = workspace_p / "mesh_canonical.ply"
    import open3d as o3d

    o3d.io.write_triangle_mesh(str(canonical_mesh), mesh)
    return CanonicalMeshResult(
        mesh_path=str(canonical_mesh),
        n_views_used=len(tsdf_views),
        n_views_skipped=skipped,
        triangle_count=len(mesh.triangles),
        vertex_count=len(mesh.vertices),
    )
