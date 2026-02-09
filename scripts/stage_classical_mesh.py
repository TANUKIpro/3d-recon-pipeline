"""Stage 5 (classical): normals + Screened Poisson mesh reconstruction."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d

from stage_diffcd_mesh import mesh_vertex_face_count, smooth_mesh_file

ProgressCallback = Callable[[float, str | None], None]

_DEFAULT_NORMAL_RADIUS_RATIO = 0.02
_DEFAULT_NORMAL_MAX_NN = 32
_DEFAULT_NORMAL_ORIENT_K = 24
_DEFAULT_POISSON_DEPTH = 9
_DEFAULT_POISSON_SCALE = 1.08
_DEFAULT_POISSON_LINEAR_FIT = False
_DEFAULT_DENSITY_TRIM_QUANTILE = 0.02
_DEFAULT_CROP_SCALE = 1.03

_DEFAULT_SMOOTH_METHOD = "laplacian"
_DEFAULT_SMOOTH_ITERATIONS = 2
_DEFAULT_SMOOTH_LAMBDA = 0.5
_DEFAULT_SMOOTH_TAUBIN_NU = -0.53
_DEFAULT_AUTO_SMOOTH = False


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _resolve_smooth_method() -> str:
    method = (
        os.environ.get("CLASSICAL_SMOOTH_METHOD")
        or os.environ.get("DIFFCD_SMOOTH_METHOD")
        or _DEFAULT_SMOOTH_METHOD
    )
    method = method.strip().lower()
    if method not in {"laplacian", "taubin"}:
        print(
            f"Unknown smoothing method '{method}', "
            f"fallback to '{_DEFAULT_SMOOTH_METHOD}'."
        )
        return _DEFAULT_SMOOTH_METHOD
    return method


def _safe_bbox_diag(points: np.ndarray) -> float:
    if points.size == 0:
        return 1e-6
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return max(diag, 1e-6)


def run_classical_mesh(
    denoised_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run normals estimation + Screened Poisson mesh reconstruction."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    classical_dir = output_path / "classical_mesh"
    classical_dir.mkdir(parents=True, exist_ok=True)

    normal_ratio = max(
        _env_float("POISSON_NORMAL_RADIUS_RATIO", _DEFAULT_NORMAL_RADIUS_RATIO),
        1e-5,
    )
    normal_max_nn = max(_env_int("POISSON_NORMAL_MAX_NN", _DEFAULT_NORMAL_MAX_NN), 8)
    normal_orient_k = max(_env_int("POISSON_NORMAL_ORIENT_K", _DEFAULT_NORMAL_ORIENT_K), 8)
    poisson_depth = max(_env_int("POISSON_DEPTH", _DEFAULT_POISSON_DEPTH), 6)
    poisson_scale = max(_env_float("POISSON_SCALE", _DEFAULT_POISSON_SCALE), 1.0)
    poisson_linear_fit = _env_bool("POISSON_LINEAR_FIT", _DEFAULT_POISSON_LINEAR_FIT)
    density_trim_q = min(
        max(_env_float("POISSON_DENSITY_TRIM_QUANTILE", _DEFAULT_DENSITY_TRIM_QUANTILE), 0.0),
        0.49,
    )
    crop_scale = max(_env_float("POISSON_CROP_SCALE", _DEFAULT_CROP_SCALE), 1.0)

    print("=== Classical Mesh: Loading point cloud ===")
    _emit_progress(progress_cb, 5.0, "Loading denoised point cloud")
    pcd = o3d.io.read_point_cloud(str(denoised_ply))
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud has no points: {denoised_ply}")
    print(f"Loaded: {points.shape[0]:,} points")

    bbox_diag = _safe_bbox_diag(points)
    normal_radius = max(bbox_diag * normal_ratio, 1e-5)
    print(
        "Normal estimation params: "
        f"radius={normal_radius:.6f}, max_nn={normal_max_nn}, orient_k={normal_orient_k}"
    )

    _emit_progress(progress_cb, 18.0, "Estimating normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )
    pcd.normalize_normals()

    if points.shape[0] >= 16:
        orient_k = min(max(8, normal_orient_k), int(points.shape[0]) - 1)
        try:
            pcd.orient_normals_consistent_tangent_plane(orient_k)
        except RuntimeError as e:
            print(f"Normal orientation fallback (consistent tangent plane failed): {e}")
            look_at = np.mean(points, axis=0) + np.array([0.0, 0.0, bbox_diag * 2.0])
            pcd.orient_normals_towards_camera_location(look_at)

    normals_path = output_path / "object_points_with_normals.ply"
    try:
        o3d.io.write_point_cloud(
            str(normals_path),
            pcd,
            write_ascii=False,
            compressed=False,
        )
    except TypeError:
        o3d.io.write_point_cloud(str(normals_path), pcd)
    print(f"Saved normals point cloud: {normals_path}")
    _emit_progress(progress_cb, 40.0, "Point normals estimated")

    print("=== Classical Mesh: Screened Poisson Reconstruction ===")
    print(
        "Poisson params: "
        f"depth={poisson_depth}, scale={poisson_scale:.3f}, linear_fit={poisson_linear_fit}"
    )
    _emit_progress(progress_cb, 52.0, "Running Screened Poisson reconstruction")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=poisson_depth,
            width=0,
            scale=poisson_scale,
            linear_fit=poisson_linear_fit,
            n_threads=0,
        )
    except TypeError:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=poisson_depth,
            width=0,
            scale=poisson_scale,
            linear_fit=poisson_linear_fit,
        )

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced an empty mesh.")

    density_arr = np.asarray(densities)
    if density_arr.size > 0 and density_trim_q > 0.0:
        threshold = float(np.quantile(density_arr, density_trim_q))
        drop_mask = density_arr < threshold
        if 0 < int(drop_mask.sum()) < int(drop_mask.size):
            mesh.remove_vertices_by_mask(drop_mask)
            print(
                f"Density trim: q={density_trim_q:.3f}, "
                f"removed={int(drop_mask.sum()):,}/{int(drop_mask.size):,}"
            )

    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(crop_scale, bbox.get_center())
    cropped = mesh.crop(bbox)
    if len(cropped.vertices) > 0 and len(cropped.triangles) > 0:
        mesh = cropped

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    _emit_progress(progress_cb, 78.0, "Poisson mesh reconstructed")

    poisson_raw_path = classical_dir / "object_mesh_poisson_raw.ply"
    try:
        o3d.io.write_triangle_mesh(
            str(poisson_raw_path),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
        )
    except TypeError:
        o3d.io.write_triangle_mesh(str(poisson_raw_path), mesh)

    raw_copy_path = output_path / "object_mesh_raw.ply"
    try:
        o3d.io.write_triangle_mesh(
            str(raw_copy_path),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
        )
    except TypeError:
        o3d.io.write_triangle_mesh(str(raw_copy_path), mesh)
    raw_vertices, raw_faces = mesh_vertex_face_count(raw_copy_path)
    print(f"Saved raw poisson mesh: {raw_copy_path}")
    print(f"  Raw vertices: {raw_vertices:,}, Faces: {raw_faces:,}")
    final_path = output_path / "object_mesh.ply"
    auto_smooth = _env_bool("CLASSICAL_AUTO_SMOOTH", _DEFAULT_AUTO_SMOOTH)

    if auto_smooth:
        _emit_progress(progress_cb, 88.0, "Applying mesh smoothing")
        smooth_method = _resolve_smooth_method()
        smooth_iterations = max(
            0,
            _env_int("CLASSICAL_SMOOTH_ITERATIONS", _DEFAULT_SMOOTH_ITERATIONS),
        )
        smooth_lambda = _env_float("CLASSICAL_SMOOTH_LAMBDA", _DEFAULT_SMOOTH_LAMBDA)
        smooth_taubin_nu = _env_float("CLASSICAL_SMOOTH_TAUBIN_NU", _DEFAULT_SMOOTH_TAUBIN_NU)
        print(
            "Smoothing params: "
            f"method={smooth_method}, iterations={smooth_iterations}, "
            f"lambda={smooth_lambda:.3f}"
            + (f", nu={smooth_taubin_nu:.3f}" if smooth_method == "taubin" else "")
        )
        smooth_mesh_file(
            raw_copy_path,
            final_path,
            method=smooth_method,
            iterations=smooth_iterations,
            lamb=smooth_lambda,
            taubin_nu=smooth_taubin_nu,
        )
        final_vertices, final_faces = mesh_vertex_face_count(final_path)
        print(f"  Smoothed vertices: {final_vertices:,}, Faces: {final_faces:,}")
    else:
        _emit_progress(progress_cb, 88.0, "Skipping mesh smoothing (optional)")
        shutil.copyfile(raw_copy_path, final_path)
        final_vertices, final_faces = mesh_vertex_face_count(final_path)
        print("Smoothing skipped (optional).")
        print(f"  Final vertices: {final_vertices:,}, Faces: {final_faces:,}")

    print(f"Saved final mesh: {final_path}")

    _emit_progress(progress_cb, 100.0, "Classical mesh stage complete")
    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classical mesh reconstruction (normals + Screened Poisson)"
    )
    parser.add_argument("input_ply", help="Denoised PLY file")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    run_classical_mesh(args.input_ply, args.output_dir)
