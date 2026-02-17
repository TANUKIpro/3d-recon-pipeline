"""Stage 5 (classical): preprocess + Poisson + postprocess + downsampling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d

from stage_diffcd_mesh import mesh_vertex_face_count, smooth_mesh_file

ProgressCallback = Callable[[float, str | None], None]

_DEFAULT_PREPROCESS_ENABLED = True
_DEFAULT_PREPROCESS_VOXEL_RATIO = 0.003
_DEFAULT_PREPROCESS_MAX_POINTS = 700_000
_DEFAULT_PREPROCESS_SOR_NEIGHBORS = 20
_DEFAULT_PREPROCESS_SOR_STD_RATIO = 3.2

_DEFAULT_NORMAL_RADIUS_RATIO = 0.02
_DEFAULT_NORMAL_MAX_NN = 32
_DEFAULT_NORMAL_ORIENT_K = 24
_DEFAULT_POISSON_DEPTH = 9
_DEFAULT_POISSON_SCALE = 1.08
_DEFAULT_POISSON_LINEAR_FIT = False
_DEFAULT_DENSITY_TRIM_QUANTILE = 0.005
_DEFAULT_CROP_SCALE = 1.06

_DEFAULT_POST_MIN_COMPONENT_TRIANGLES = 150
_DEFAULT_POST_MIN_COMPONENT_RATIO = 0.005

_DEFAULT_SMOOTH_METHOD = "laplacian"
_DEFAULT_SMOOTH_ITERATIONS = 2
_DEFAULT_SMOOTH_LAMBDA = 0.5
_DEFAULT_SMOOTH_TAUBIN_NU = -0.53
_DEFAULT_AUTO_SMOOTH = False

_DEFAULT_DOWNSAMPLE_ENABLED = True
_DEFAULT_DOWNSAMPLE_TARGET_FACES = 120_000
_DEFAULT_DOWNSAMPLE_TRIGGER_FACES = 170_000


@dataclass(frozen=True)
class ClassicalMeshParams:
    preprocess_enabled: bool
    preprocess_voxel_ratio: float
    preprocess_max_points: int
    preprocess_sor_neighbors: int
    preprocess_sor_std_ratio: float
    normal_ratio: float
    normal_max_nn: int
    normal_orient_k: int
    poisson_depth: int
    poisson_scale: float
    poisson_linear_fit: bool
    density_trim_q: float
    crop_scale: float
    post_min_component_triangles: int
    post_min_component_ratio: float
    auto_smooth: bool
    smooth_method: str
    smooth_iterations: int
    smooth_lambda: float
    smooth_taubin_nu: float
    downsample_enabled: bool
    downsample_target_faces: int
    downsample_trigger_faces: int


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


def _resolve_params() -> ClassicalMeshParams:
    preprocess_enabled = _env_bool(
        "CLASSICAL_PREPROCESS_ENABLED",
        _DEFAULT_PREPROCESS_ENABLED,
    )
    preprocess_voxel_ratio = max(
        _env_float("CLASSICAL_PREPROCESS_VOXEL_RATIO", _DEFAULT_PREPROCESS_VOXEL_RATIO),
        0.0,
    )
    preprocess_max_points = max(
        _env_int("CLASSICAL_PREPROCESS_MAX_POINTS", _DEFAULT_PREPROCESS_MAX_POINTS),
        50_000,
    )
    preprocess_sor_neighbors = max(
        _env_int("CLASSICAL_PREPROCESS_SOR_NEIGHBORS", _DEFAULT_PREPROCESS_SOR_NEIGHBORS),
        2,
    )
    preprocess_sor_std_ratio = max(
        _env_float("CLASSICAL_PREPROCESS_SOR_STD_RATIO", _DEFAULT_PREPROCESS_SOR_STD_RATIO),
        0.1,
    )

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

    post_min_component_triangles = max(
        _env_int(
            "CLASSICAL_POST_MIN_COMPONENT_TRIANGLES",
            _DEFAULT_POST_MIN_COMPONENT_TRIANGLES,
        ),
        0,
    )
    post_min_component_ratio = min(
        max(
            _env_float(
                "CLASSICAL_POST_MIN_COMPONENT_RATIO",
                _DEFAULT_POST_MIN_COMPONENT_RATIO,
            ),
            0.0,
        ),
        0.5,
    )

    auto_smooth = _env_bool("CLASSICAL_AUTO_SMOOTH", _DEFAULT_AUTO_SMOOTH)
    smooth_method = _resolve_smooth_method()
    smooth_iterations = max(
        0,
        _env_int("CLASSICAL_SMOOTH_ITERATIONS", _DEFAULT_SMOOTH_ITERATIONS),
    )
    smooth_lambda = _env_float("CLASSICAL_SMOOTH_LAMBDA", _DEFAULT_SMOOTH_LAMBDA)
    smooth_taubin_nu = _env_float("CLASSICAL_SMOOTH_TAUBIN_NU", _DEFAULT_SMOOTH_TAUBIN_NU)

    downsample_enabled = _env_bool(
        "CLASSICAL_DOWNSAMPLE_ENABLED",
        _DEFAULT_DOWNSAMPLE_ENABLED,
    )
    downsample_target_faces = max(
        _env_int("CLASSICAL_DOWNSAMPLE_TARGET_FACES", _DEFAULT_DOWNSAMPLE_TARGET_FACES),
        1000,
    )
    downsample_trigger_faces = max(
        _env_int("CLASSICAL_DOWNSAMPLE_TRIGGER_FACES", _DEFAULT_DOWNSAMPLE_TRIGGER_FACES),
        downsample_target_faces,
    )
    return ClassicalMeshParams(
        preprocess_enabled=preprocess_enabled,
        preprocess_voxel_ratio=preprocess_voxel_ratio,
        preprocess_max_points=preprocess_max_points,
        preprocess_sor_neighbors=preprocess_sor_neighbors,
        preprocess_sor_std_ratio=preprocess_sor_std_ratio,
        normal_ratio=normal_ratio,
        normal_max_nn=normal_max_nn,
        normal_orient_k=normal_orient_k,
        poisson_depth=poisson_depth,
        poisson_scale=poisson_scale,
        poisson_linear_fit=poisson_linear_fit,
        density_trim_q=density_trim_q,
        crop_scale=crop_scale,
        post_min_component_triangles=post_min_component_triangles,
        post_min_component_ratio=post_min_component_ratio,
        auto_smooth=auto_smooth,
        smooth_method=smooth_method,
        smooth_iterations=smooth_iterations,
        smooth_lambda=smooth_lambda,
        smooth_taubin_nu=smooth_taubin_nu,
        downsample_enabled=downsample_enabled,
        downsample_target_faces=downsample_target_faces,
        downsample_trigger_faces=downsample_trigger_faces,
    )


def _safe_bbox_diag(points: np.ndarray) -> float:
    if points.size == 0:
        return 1e-6
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return max(diag, 1e-6)


def _write_point_cloud_safe(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    try:
        o3d.io.write_point_cloud(
            str(path),
            pcd,
            write_ascii=False,
            compressed=False,
        )
    except TypeError:
        o3d.io.write_point_cloud(str(path), pcd)


def _write_mesh_safe(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    try:
        o3d.io.write_triangle_mesh(
            str(path),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
        )
    except TypeError:
        o3d.io.write_triangle_mesh(str(path), mesh)


def _preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    *,
    bbox_diag: float,
    enabled: bool,
    voxel_ratio: float,
    max_points: int,
    sor_neighbors: int,
    sor_std_ratio: float,
) -> tuple[o3d.geometry.PointCloud, dict[str, float | int | bool]]:
    processed = pcd
    source_points = int(len(np.asarray(pcd.points)))
    metrics: dict[str, float | int | bool] = {
        "enabled": enabled,
        "source_points": source_points,
        "voxel_size": 0.0,
        "after_voxel_points": source_points,
        "after_sor_points": source_points,
        "final_points": source_points,
    }
    if not enabled or source_points == 0:
        return processed, metrics

    if voxel_ratio > 0.0 and source_points > max_points:
        voxel_size = max(bbox_diag * voxel_ratio, 1e-6)
        candidate = processed.voxel_down_sample(voxel_size=voxel_size)
        if len(candidate.points) > 0:
            processed = candidate
            metrics["voxel_size"] = float(voxel_size)
        metrics["after_voxel_points"] = int(len(np.asarray(processed.points)))

    if sor_neighbors >= 2 and len(processed.points) > sor_neighbors:
        filtered, _ = processed.remove_statistical_outlier(
            nb_neighbors=sor_neighbors,
            std_ratio=sor_std_ratio,
        )
        if len(filtered.points) > 0:
            processed = filtered
        metrics["after_sor_points"] = int(len(np.asarray(processed.points)))

    metrics["final_points"] = int(len(np.asarray(processed.points)))
    return processed, metrics


def _clean_small_components(
    mesh: o3d.geometry.TriangleMesh,
    *,
    min_triangles: int,
    min_ratio: float,
) -> int:
    if len(mesh.triangles) == 0:
        return 0

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(triangle_clusters)
    cluster_sizes = np.asarray(cluster_n_triangles)
    if cluster_ids.size == 0 or cluster_sizes.size <= 1:
        return 0

    largest = int(cluster_sizes.max())
    threshold = max(min_triangles, int(largest * min_ratio))
    remove_mask = cluster_sizes[cluster_ids] < threshold
    remove_count = int(remove_mask.sum())
    if 0 < remove_count < int(len(mesh.triangles)):
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
        return remove_count
    return 0


def _downsample_mesh_if_needed(
    mesh: o3d.geometry.TriangleMesh,
    *,
    enabled: bool,
    trigger_faces: int,
    target_faces: int,
) -> tuple[o3d.geometry.TriangleMesh, bool, int, int]:
    before_faces = int(len(mesh.triangles))
    if not enabled or before_faces == 0:
        return mesh, False, before_faces, before_faces

    effective_target = min(max(4, target_faces), max(4, before_faces - 1))
    effective_trigger = max(trigger_faces, effective_target)
    if before_faces <= effective_trigger:
        return mesh, False, before_faces, before_faces

    simplified = mesh.simplify_quadric_decimation(effective_target)
    if len(simplified.vertices) == 0 or len(simplified.triangles) == 0:
        return mesh, False, before_faces, before_faces

    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_non_manifold_edges()
    simplified.remove_unreferenced_vertices()
    simplified.compute_vertex_normals()

    after_faces = int(len(simplified.triangles))
    if after_faces == 0:
        return mesh, False, before_faces, before_faces
    return simplified, True, before_faces, after_faces


def _prepare_output_dirs(output_dir: str) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    classical_dir = output_path / "classical_mesh"
    classical_dir.mkdir(parents=True, exist_ok=True)
    return output_path, classical_dir


def run_classical_preprocess(
    denoised_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    params = _resolve_params()
    output_path, classical_dir = _prepare_output_dirs(output_dir)
    print("=== Classical Mesh: Preprocess point cloud ===")
    _emit_progress(progress_cb, 8.0, "Classical/Preprocess: loading denoised point cloud")
    pcd = o3d.io.read_point_cloud(str(denoised_ply))
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud has no points: {denoised_ply}")
    print(f"Loaded: {points.shape[0]:,} points")

    bbox_diag = _safe_bbox_diag(points)
    _emit_progress(
        progress_cb,
        12.0,
        "Classical/Preprocess: filtering and resampling points",
    )
    pcd, pre_metrics = _preprocess_point_cloud(
        pcd,
        bbox_diag=bbox_diag,
        enabled=params.preprocess_enabled,
        voxel_ratio=params.preprocess_voxel_ratio,
        max_points=params.preprocess_max_points,
        sor_neighbors=params.preprocess_sor_neighbors,
        sor_std_ratio=params.preprocess_sor_std_ratio,
    )
    pre_points = np.asarray(pcd.points)
    if pre_points.shape[0] == 0:
        raise RuntimeError("Preprocess produced zero points.")

    print(
        "Preprocess params: "
        f"enabled={params.preprocess_enabled}, voxel_ratio={params.preprocess_voxel_ratio:.5f}, "
        f"max_points={params.preprocess_max_points:,}, sor_neighbors={params.preprocess_sor_neighbors}, "
        f"sor_std={params.preprocess_sor_std_ratio:.3f}"
    )
    print(
        "Preprocess stats: "
        f"source={int(pre_metrics['source_points']):,}, "
        f"after_voxel={int(pre_metrics['after_voxel_points']):,}, "
        f"after_sor={int(pre_metrics['after_sor_points']):,}, "
        f"final={int(pre_metrics['final_points']):,}"
    )

    preprocess_path = output_path / "object_mesh_input.ply"
    _write_point_cloud_safe(preprocess_path, pcd)
    _write_point_cloud_safe(classical_dir / "object_mesh_input.ply", pcd)
    print(f"Saved preprocessed point cloud: {preprocess_path}")
    _emit_progress(progress_cb, 100.0, "Classical/Preprocess: complete")
    return preprocess_path


def run_classical_main(
    preprocess_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    params = _resolve_params()
    output_path, classical_dir = _prepare_output_dirs(output_dir)
    print("=== Classical Mesh: Main (normals + Screened Poisson) ===")
    pcd = o3d.io.read_point_cloud(str(preprocess_ply))
    pre_points = np.asarray(pcd.points)
    if pre_points.shape[0] == 0:
        raise ValueError(f"Point cloud has no points: {preprocess_ply}")
    pre_bbox_diag = _safe_bbox_diag(pre_points)
    normal_radius = max(pre_bbox_diag * params.normal_ratio, 1e-5)
    print(
        "Normal estimation params: "
        f"radius={normal_radius:.6f}, max_nn={params.normal_max_nn}, orient_k={params.normal_orient_k}"
    )

    _emit_progress(progress_cb, 10.0, "Classical/Main: estimating normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=params.normal_max_nn,
        )
    )
    pcd.normalize_normals()

    if pre_points.shape[0] >= 16:
        orient_k = min(max(8, params.normal_orient_k), int(pre_points.shape[0]) - 1)
        try:
            pcd.orient_normals_consistent_tangent_plane(orient_k)
        except RuntimeError as e:
            print(f"Normal orientation fallback (consistent tangent plane failed): {e}")
            look_at = np.mean(pre_points, axis=0) + np.array([0.0, 0.0, pre_bbox_diag * 2.0])
            pcd.orient_normals_towards_camera_location(look_at)

    normals_path = output_path / "object_points_with_normals.ply"
    _write_point_cloud_safe(normals_path, pcd)
    print(f"Saved normals point cloud: {normals_path}")
    _emit_progress(progress_cb, 45.0, "Classical/Main: point normals estimated")

    print(
        "Poisson params: "
        f"depth={params.poisson_depth}, scale={params.poisson_scale:.3f}, linear_fit={params.poisson_linear_fit}"
    )
    _emit_progress(progress_cb, 58.0, "Classical/Main: running Screened Poisson reconstruction")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=params.poisson_depth,
            width=0,
            scale=params.poisson_scale,
            linear_fit=params.poisson_linear_fit,
            n_threads=0,
        )
    except TypeError:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=params.poisson_depth,
            width=0,
            scale=params.poisson_scale,
            linear_fit=params.poisson_linear_fit,
        )

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced an empty mesh.")

    density_arr = np.asarray(densities)
    if density_arr.size > 0 and params.density_trim_q > 0.0:
        threshold = float(np.quantile(density_arr, params.density_trim_q))
        drop_mask = density_arr < threshold
        if 0 < int(drop_mask.sum()) < int(drop_mask.size):
            mesh.remove_vertices_by_mask(drop_mask)
            print(
                f"Density trim: q={params.density_trim_q:.3f}, "
                f"removed={int(drop_mask.sum()):,}/{int(drop_mask.size):,}"
            )

    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(params.crop_scale, bbox.get_center())
    cropped = mesh.crop(bbox)
    if len(cropped.vertices) > 0 and len(cropped.triangles) > 0:
        mesh = cropped

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    poisson_raw_path = classical_dir / "object_mesh_poisson_raw.ply"
    raw_copy_path = output_path / "object_mesh_raw.ply"
    _write_mesh_safe(poisson_raw_path, mesh)
    _write_mesh_safe(raw_copy_path, mesh)
    raw_vertices, raw_faces = mesh_vertex_face_count(raw_copy_path)
    print(f"Saved raw poisson mesh: {raw_copy_path}")
    print(f"  Raw vertices: {raw_vertices:,}, Faces: {raw_faces:,}")
    _emit_progress(progress_cb, 100.0, "Classical/Main: raw Poisson mesh saved")
    return raw_copy_path


def run_classical_postprocess(
    raw_mesh_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    params = _resolve_params()
    output_path, classical_dir = _prepare_output_dirs(output_dir)
    print("=== Classical Mesh: Postprocess mesh ===")
    mesh = o3d.io.read_triangle_mesh(str(raw_mesh_ply))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh is empty: {raw_mesh_ply}")
    _emit_progress(progress_cb, 12.0, "Classical/Postprocess: cleaning mesh topology")
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    removed_triangles = _clean_small_components(
        mesh,
        min_triangles=params.post_min_component_triangles,
        min_ratio=params.post_min_component_ratio,
    )
    if removed_triangles > 0:
        print(
            "Component cleanup: "
            f"removed {removed_triangles:,} small triangles "
            f"(min_triangles={params.post_min_component_triangles}, "
            f"min_ratio={params.post_min_component_ratio:.3f})"
        )
    mesh.compute_vertex_normals()

    post_path = output_path / "object_mesh_postprocessed.ply"
    _write_mesh_safe(post_path, mesh)
    _write_mesh_safe(classical_dir / "object_mesh_postprocessed.ply", mesh)

    if params.auto_smooth and params.smooth_iterations > 0:
        _emit_progress(progress_cb, 66.0, "Classical/Postprocess: applying smoothing")
        print(
            "Smoothing params: "
            f"method={params.smooth_method}, iterations={params.smooth_iterations}, "
            f"lambda={params.smooth_lambda:.3f}"
            + (f", nu={params.smooth_taubin_nu:.3f}" if params.smooth_method == "taubin" else "")
        )
        smooth_mesh_file(
            post_path,
            post_path,
            method=params.smooth_method,
            iterations=params.smooth_iterations,
            lamb=params.smooth_lambda,
            taubin_nu=params.smooth_taubin_nu,
        )
    else:
        print("Smoothing skipped (optional).")

    post_vertices, post_faces = mesh_vertex_face_count(post_path)
    print(f"Saved postprocessed mesh: {post_path}")
    print(f"  Postprocessed vertices: {post_vertices:,}, Faces: {post_faces:,}")
    _emit_progress(progress_cb, 100.0, "Classical/Postprocess: complete")
    return post_path


def run_classical_downsample(
    post_mesh_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    params = _resolve_params()
    output_path, classical_dir = _prepare_output_dirs(output_dir)
    print("=== Classical Mesh: Downsample mesh ===")
    downsample_mesh = o3d.io.read_triangle_mesh(str(post_mesh_ply))
    if len(downsample_mesh.vertices) == 0 or len(downsample_mesh.triangles) == 0:
        raise ValueError(f"Mesh is empty: {post_mesh_ply}")
    _emit_progress(progress_cb, 15.0, "Classical/Downsample: reducing face count")
    downsample_mesh, downsampled, before_faces, after_faces = _downsample_mesh_if_needed(
        downsample_mesh,
        enabled=params.downsample_enabled,
        trigger_faces=params.downsample_trigger_faces,
        target_faces=params.downsample_target_faces,
    )
    if downsampled:
        print(
            "Downsample applied: "
            f"{before_faces:,} -> {after_faces:,} faces "
            f"(target={params.downsample_target_faces:,}, trigger={params.downsample_trigger_faces:,})"
        )
    else:
        print(
            "Downsample skipped: "
            f"faces={before_faces:,}, target={params.downsample_target_faces:,}, "
            f"trigger={params.downsample_trigger_faces:,}, enabled={params.downsample_enabled}"
        )
        _emit_progress(progress_cb, 55.0, "Classical/Downsample: skipped")

    final_path = output_path / "object_mesh.ply"
    _write_mesh_safe(final_path, downsample_mesh)
    _write_mesh_safe(classical_dir / "object_mesh_final.ply", downsample_mesh)
    final_vertices, final_faces = mesh_vertex_face_count(final_path)
    print(f"Saved final mesh: {final_path}")
    print(f"  Final vertices: {final_vertices:,}, Faces: {final_faces:,}")
    _emit_progress(progress_cb, 100.0, "Classical/Downsample: complete")
    return final_path


def _map_progress(
    progress_cb: ProgressCallback | None,
    *,
    start: float,
    end: float,
) -> ProgressCallback:
    span = max(0.0, end - start)

    def _mapped(local_progress: float, detail: str | None = None) -> None:
        ratio = max(0.0, min(100.0, float(local_progress))) / 100.0
        _emit_progress(progress_cb, start + ratio * span, detail)

    return _mapped


def run_classical_mesh(
    denoised_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run classical mesh as a 4-step chain: pre -> main -> post -> downsample."""
    preprocess_path = run_classical_preprocess(
        denoised_ply,
        output_dir,
        progress_cb=_map_progress(progress_cb, start=0.0, end=24.0),
    )
    raw_path = run_classical_main(
        str(preprocess_path),
        output_dir,
        progress_cb=_map_progress(progress_cb, start=24.0, end=72.0),
    )
    post_path = run_classical_postprocess(
        str(raw_path),
        output_dir,
        progress_cb=_map_progress(progress_cb, start=72.0, end=92.0),
    )
    final_path = run_classical_downsample(
        str(post_path),
        output_dir,
        progress_cb=_map_progress(progress_cb, start=92.0, end=100.0),
    )
    _emit_progress(progress_cb, 100.0, "Classical mesh stage complete")
    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classical mesh reconstruction (pre -> main -> post -> downsample)"
    )
    parser.add_argument("input_ply", help="Denoised PLY file")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    run_classical_mesh(args.input_ply, args.output_dir)
