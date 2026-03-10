"""Stage 6: mesh wrapping via iterative Poisson reconstruction or CGAL alpha wrap.

This stage creates a closed "outer shell" mesh from the Stage 5 mesh.
The wrapped output is used as texture baking input to improve UV robustness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d

from config_defaults import (
    MESHWRAP_ALPHA_RATIO as _DEFAULT_WRAP_ALPHA_RATIO,
    MESHWRAP_CROP_SCALE as _DEFAULT_WRAP_CROP_SCALE,
    MESHWRAP_DENSITY_TRIM_Q as _DEFAULT_WRAP_DENSITY_TRIM_Q,
    MESHWRAP_ITERATIONS as _DEFAULT_WRAP_ITERATIONS,
    MESHWRAP_NORMAL_RADIUS_RATIO as _DEFAULT_WRAP_NORMAL_RADIUS_RATIO,
    MESHWRAP_OFFSET_RATIO as _DEFAULT_WRAP_OFFSET_RATIO,
    MESHWRAP_POISSON_DEPTH as _DEFAULT_WRAP_POISSON_DEPTH,
    MESHWRAP_POISSON_SCALE as _DEFAULT_WRAP_POISSON_SCALE,
    MESHWRAP_QUALITY_THRESHOLD as _DEFAULT_WRAP_QUALITY_THRESHOLD,
    MESHWRAP_SAMPLE_POINTS as _DEFAULT_WRAP_SAMPLE_POINTS,
    MESHWRAP_SMOOTH_ITERATIONS as _DEFAULT_WRAP_SMOOTH_ITERATIONS,
    MESHWRAP_TARGET_FACE_RATIO as _DEFAULT_WRAP_TARGET_FACE_RATIO,
    _MESHWRAP_ENABLED as _DEFAULT_WRAP_ENABLED,
    _MESHWRAP_KEEP_LARGEST_COMPONENT as _DEFAULT_WRAP_KEEP_LARGEST_COMPONENT,
    _MESHWRAP_MAX_FACES as _DEFAULT_WRAP_MAX_FACES,
    _MESHWRAP_METHOD as _DEFAULT_WRAP_METHOD,
    _MESHWRAP_METHODS as _VALID_METHODS,
    _MESHWRAP_MIN_FACES as _DEFAULT_WRAP_MIN_FACES,
    _MESHWRAP_NORMAL_MAX_NN as _DEFAULT_WRAP_NORMAL_MAX_NN,
    _MESHWRAP_NORMAL_ORIENT_K as _DEFAULT_WRAP_NORMAL_ORIENT_K,
    _MESHWRAP_POISSON_LINEAR_FIT as _DEFAULT_WRAP_POISSON_LINEAR_FIT,
    _MESHWRAP_PRESERVE_INPUT_ON_FAILURE as _DEFAULT_WRAP_PRESERVE_INPUT_ON_FAILURE,
    _MESHWRAP_QUALITY_SAMPLE_POINTS as _DEFAULT_WRAP_QUALITY_SAMPLE_POINTS,
    _MESHWRAP_SMOOTH_LAMBDA as _DEFAULT_WRAP_SMOOTH_LAMBDA,
    _MESHWRAP_SMOOTH_NU as _DEFAULT_WRAP_SMOOTH_NU,
)
from mesh_orientation import orient_faces_outward

ProgressCallback = Callable[[float, str | None], None]


@dataclass(frozen=True)
class MeshWrapParams:
    enabled: bool
    method: str
    iterations: int
    sample_points: int
    normal_radius_ratio: float
    normal_max_nn: int
    normal_orient_k: int
    poisson_depth: int
    poisson_scale: float
    poisson_linear_fit: bool
    density_trim_q: float
    crop_scale: float
    keep_largest_component: bool
    target_face_ratio: float
    min_faces: int
    max_faces: int
    preserve_input_on_failure: bool
    smooth_iterations: int
    quality_threshold: float
    alpha_ratio: float
    offset_ratio: float


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
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _resolve_params(overrides: dict | None = None) -> MeshWrapParams:
    """Build MeshWrapParams with priority: overrides (UI) > env vars > defaults."""
    ov = overrides or {}

    # Method resolution: override > env > default
    if "method" in ov and ov["method"] is not None:
        method = str(ov["method"]).strip().lower()
    else:
        method = str(os.environ.get("MESH_WRAP_METHOD", _DEFAULT_WRAP_METHOD)).strip().lower()
    # Accept legacy "ipsr" as alias for "poisson_iterative"
    if method == "ipsr":
        method = "poisson_iterative"
    if method not in _VALID_METHODS:
        print(
            f"Unknown MESH_WRAP_METHOD='{method}', fallback to '{_DEFAULT_WRAP_METHOD}'."
        )
        method = _DEFAULT_WRAP_METHOD

    def _val_int(key: str, env_name: str, default: int) -> int:
        if key in ov and ov[key] is not None:
            try:
                return int(ov[key])
            except (TypeError, ValueError):
                pass
        return _env_int(env_name, default)

    def _val_float(key: str, env_name: str, default: float) -> float:
        if key in ov and ov[key] is not None:
            try:
                return float(ov[key])
            except (TypeError, ValueError):
                pass
        return _env_float(env_name, default)

    return MeshWrapParams(
        enabled=_env_bool("MESH_WRAP_ENABLED", _DEFAULT_WRAP_ENABLED),
        method=method,
        iterations=max(1, _val_int("iterations", "MESH_WRAP_ITERATIONS", _DEFAULT_WRAP_ITERATIONS)),
        sample_points=max(50_000, _val_int("sample_points", "MESH_WRAP_SAMPLE_POINTS", _DEFAULT_WRAP_SAMPLE_POINTS)),
        normal_radius_ratio=max(
            1e-6,
            _val_float("normal_radius_ratio", "MESH_WRAP_NORMAL_RADIUS_RATIO", _DEFAULT_WRAP_NORMAL_RADIUS_RATIO),
        ),
        normal_max_nn=max(8, _env_int("MESH_WRAP_NORMAL_MAX_NN", _DEFAULT_WRAP_NORMAL_MAX_NN)),
        normal_orient_k=max(
            8,
            _env_int("MESH_WRAP_NORMAL_ORIENT_K", _DEFAULT_WRAP_NORMAL_ORIENT_K),
        ),
        poisson_depth=max(6, _val_int("poisson_depth", "MESH_WRAP_POISSON_DEPTH", _DEFAULT_WRAP_POISSON_DEPTH)),
        poisson_scale=max(
            1.0,
            _val_float("poisson_scale", "MESH_WRAP_POISSON_SCALE", _DEFAULT_WRAP_POISSON_SCALE),
        ),
        poisson_linear_fit=_env_bool(
            "MESH_WRAP_POISSON_LINEAR_FIT",
            _DEFAULT_WRAP_POISSON_LINEAR_FIT,
        ),
        density_trim_q=min(
            0.49,
            max(0.0, _val_float("density_trim_q", "MESH_WRAP_DENSITY_TRIM_Q", _DEFAULT_WRAP_DENSITY_TRIM_Q)),
        ),
        crop_scale=max(1.0, _val_float("crop_scale", "MESH_WRAP_CROP_SCALE", _DEFAULT_WRAP_CROP_SCALE)),
        keep_largest_component=_env_bool(
            "MESH_WRAP_KEEP_LARGEST_COMPONENT",
            _DEFAULT_WRAP_KEEP_LARGEST_COMPONENT,
        ),
        target_face_ratio=min(
            3.0,
            max(
                0.2,
                _val_float("target_face_ratio", "MESH_WRAP_TARGET_FACE_RATIO", _DEFAULT_WRAP_TARGET_FACE_RATIO),
            ),
        ),
        min_faces=max(1000, _env_int("MESH_WRAP_MIN_FACES", _DEFAULT_WRAP_MIN_FACES)),
        max_faces=max(1000, _val_int("max_faces", "MESH_WRAP_MAX_FACES", _DEFAULT_WRAP_MAX_FACES)),
        preserve_input_on_failure=_env_bool(
            "MESH_WRAP_PRESERVE_INPUT_ON_FAILURE",
            _DEFAULT_WRAP_PRESERVE_INPUT_ON_FAILURE,
        ),
        smooth_iterations=max(
            0,
            _val_int("smooth_iterations", "MESH_WRAP_SMOOTH_ITERATIONS", _DEFAULT_WRAP_SMOOTH_ITERATIONS),
        ),
        quality_threshold=max(
            0.0,
            _val_float("quality_threshold", "MESH_WRAP_QUALITY_THRESHOLD", _DEFAULT_WRAP_QUALITY_THRESHOLD),
        ),
        alpha_ratio=max(
            0.001,
            _val_float("alpha_ratio", "MESH_WRAP_ALPHA_RATIO", _DEFAULT_WRAP_ALPHA_RATIO),
        ),
        offset_ratio=max(
            0.01,
            _val_float("offset_ratio", "MESH_WRAP_OFFSET_RATIO", _DEFAULT_WRAP_OFFSET_RATIO),
        ),
    )


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


def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _bbox_diag(mesh: o3d.geometry.TriangleMesh) -> float:
    if len(mesh.vertices) == 0:
        return 1e-6
    vs = np.asarray(mesh.vertices)
    mins = vs.min(axis=0)
    maxs = vs.max(axis=0)
    return max(float(np.linalg.norm(maxs - mins)), 1e-6)


def _keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(triangle_clusters)
    cluster_sizes = np.asarray(cluster_n_triangles)
    if cluster_ids.size == 0 or cluster_sizes.size <= 1:
        return mesh
    keep_cluster = int(np.argmax(cluster_sizes))
    remove_mask = cluster_ids != keep_cluster
    if np.any(remove_mask):
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
    return mesh


def _estimate_orient_normals(
    pcd: o3d.geometry.PointCloud,
    *,
    radius: float,
    max_nn: int,
    orient_k: int,
    center: np.ndarray,
) -> None:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    pcd.normalize_normals()

    n = int(len(np.asarray(pcd.points)))
    if n >= max(orient_k + 1, 16):
        orient_k = min(max(8, orient_k), n - 1)
        try:
            pcd.orient_normals_consistent_tangent_plane(orient_k)
            return
        except RuntimeError as e:
            print(f"Normal orientation fallback (consistent tangent plane failed): {e}")

    fallback = np.asarray(center, dtype=np.float64) + np.array([0.0, 0.0, radius * 4.0])
    pcd.orient_normals_towards_camera_location(fallback)


def _has_valid_normals(pcd: o3d.geometry.PointCloud) -> bool:
    """Check if a point cloud has valid (non-zero, non-NaN) normals."""
    if not pcd.has_normals():
        return False
    normals = np.asarray(pcd.normals)
    if normals.shape[0] == 0:
        return False
    if np.any(np.isnan(normals)):
        return False
    norms = np.linalg.norm(normals, axis=1)
    zero_ratio = float(np.sum(norms < 1e-8)) / float(norms.shape[0])
    return zero_ratio < 0.1


def _apply_taubin_smoothing(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int,
    lam: float = _DEFAULT_WRAP_SMOOTH_LAMBDA,
    nu: float = _DEFAULT_WRAP_SMOOTH_NU,
) -> o3d.geometry.TriangleMesh:
    """Apply Taubin smoothing to reduce ringing without volume loss."""
    if iterations <= 0:
        return mesh
    smoothed = mesh.filter_smooth_taubin(
        number_of_iterations=iterations,
        lambda_filter=lam,
        mu=nu,
    )
    print(f"Mesh Wrap: applied Taubin smoothing ({iterations} iterations, lambda={lam}, nu={nu})")
    return smoothed


def _compute_chamfer_distance(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
) -> float:
    """Compute symmetric mean Chamfer distance between two meshes."""
    n_sample = _DEFAULT_WRAP_QUALITY_SAMPLE_POINTS
    try:
        pcd_a = mesh_a.sample_points_uniformly(number_of_points=n_sample)
        pcd_b = mesh_b.sample_points_uniformly(number_of_points=n_sample)
    except Exception:
        return 0.0

    if len(pcd_a.points) == 0 or len(pcd_b.points) == 0:
        return 0.0

    dist_a_to_b = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    dist_b_to_a = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
    return float((dist_a_to_b.mean() + dist_b_to_a.mean()) / 2.0)


def _run_alpha_wrap(
    source_mesh: o3d.geometry.TriangleMesh,
    params: MeshWrapParams,
    progress_cb: ProgressCallback | None,
) -> o3d.geometry.TriangleMesh:
    """Wrap mesh using CGAL 3D alpha wrapping via seagullmesh."""
    _emit_progress(progress_cb, 20.0, "Mesh Wrap: loading seagullmesh (CGAL)")
    try:
        from seagullmesh import Mesh3
    except ImportError:
        raise RuntimeError(
            "seagullmesh not installed. Install with: "
            "pip install git+https://github.com/darikg/seagullmesh.git  "
            "Use method='poisson_iterative' as fallback."
        )

    vertices = np.asarray(source_mesh.vertices)
    triangles = np.asarray(source_mesh.triangles)
    diag = _bbox_diag(source_mesh)
    alpha = diag * params.alpha_ratio
    offset = alpha * params.offset_ratio

    print(
        f"Mesh Wrap alpha_wrap: alpha={alpha:.6f} (ratio={params.alpha_ratio}), "
        f"offset={offset:.6f} (ratio={params.offset_ratio}), diag={diag:.4f}"
    )

    _emit_progress(progress_cb, 30.0, "Mesh Wrap: running CGAL alpha wrapping")

    # seagullmesh wraps CGAL alpha_wrap_3 — accepts vertices+faces or just vertices
    try:
        sgm_input = Mesh3.from_polygon_soup(vertices, triangles)
        wrapped_sgm = Mesh3.alpha_wrap(sgm_input, alpha, offset)
    except (TypeError, AttributeError):
        # Fallback: try point-cloud-only API variant
        try:
            wrapped_sgm = Mesh3.from_alpha_wrapping(vertices, alpha, offset)
        except AttributeError:
            raise RuntimeError(
                "seagullmesh API mismatch — neither Mesh3.alpha_wrap() nor "
                "Mesh3.from_alpha_wrapping() found. Check seagullmesh version."
            )

    _emit_progress(progress_cb, 70.0, "Mesh Wrap: converting alpha wrap result")

    w_verts, w_faces = wrapped_sgm.to_vertices_and_faces()
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.asarray(w_verts, dtype=np.float64))
    result.triangles = o3d.utility.Vector3iVector(np.asarray(w_faces, dtype=np.int32))
    result = _clean_mesh(result)

    if params.keep_largest_component:
        result = _keep_largest_component(result)
        result = _clean_mesh(result)

    print(
        f"Mesh Wrap alpha_wrap result: "
        f"{len(result.vertices):,} verts, {len(result.triangles):,} faces"
    )

    # Face count adjustment (mirror _run_poisson_wrap pattern)
    source_faces = int(len(source_mesh.triangles))
    target_faces = int(round(float(source_faces) * params.target_face_ratio))
    target_faces = max(params.min_faces, target_faces)
    target_faces = min(params.max_faces, target_faces)
    current_faces = int(len(result.triangles))
    if current_faces > target_faces:
        _emit_progress(progress_cb, 78.0, "Mesh Wrap: decimating alpha wrap result")
        effective_target = min(max(4, target_faces), max(4, current_faces - 1))
        simplified = result.simplify_quadric_decimation(effective_target)
        if len(simplified.vertices) > 0 and len(simplified.triangles) > 0:
            result = _clean_mesh(simplified)
            print(
                f"Mesh Wrap alpha_wrap face trim: {current_faces:,} -> "
                f"{len(result.triangles):,} (target={target_faces:,})"
            )

    # Taubin smoothing (same as _run_poisson_wrap)
    result = _apply_taubin_smoothing(result, params.smooth_iterations)
    result.compute_vertex_normals()

    return result


def _run_poisson_wrap(
    source_mesh: o3d.geometry.TriangleMesh,
    params: MeshWrapParams,
    progress_cb: ProgressCallback | None,
) -> o3d.geometry.TriangleMesh:
    wrapped = source_mesh
    source_obb = wrapped.get_oriented_bounding_box()
    source_obb_center = np.asarray(source_obb.center).copy()
    source_obb_R = np.asarray(source_obb.R).copy()
    source_obb_extent = np.asarray(source_obb.extent).copy()
    source_diag = _bbox_diag(wrapped)
    source_faces = int(len(wrapped.triangles))

    for iter_idx in range(params.iterations):
        iter_start = 18.0 + (iter_idx / max(params.iterations, 1)) * 62.0
        iter_end = 18.0 + ((iter_idx + 1) / max(params.iterations, 1)) * 62.0
        sample_n = max(
            50_000,
            min(params.sample_points, max(50_000, int(len(wrapped.triangles)) * 4)),
        )

        _emit_progress(
            progress_cb,
            iter_start,
            f"Mesh Wrap: sampling surface points ({iter_idx + 1}/{params.iterations})",
        )
        has_triangle_normals = False
        try:
            pcd = wrapped.sample_points_uniformly(
                number_of_points=sample_n,
                use_triangle_normal=True,
            )
            has_triangle_normals = True
        except TypeError:
            pcd = wrapped.sample_points_uniformly(number_of_points=sample_n)
        points = np.asarray(pcd.points)
        if points.shape[0] < 100:
            raise RuntimeError("Mesh wrap sampling produced too few points.")

        normal_radius = max(source_diag * params.normal_radius_ratio, 1e-5)
        _emit_progress(
            progress_cb,
            iter_start + (iter_end - iter_start) * 0.20,
            f"Mesh Wrap: processing normals ({iter_idx + 1}/{params.iterations})",
        )

        # Phase 1 fix: preserve triangle normals when available and valid
        if has_triangle_normals and _has_valid_normals(pcd):
            pcd.normalize_normals()
            print(
                f"Mesh Wrap iter {iter_idx + 1}: using triangle normals "
                f"(skipping KDTree re-estimation)"
            )
        else:
            if has_triangle_normals:
                print(
                    f"Mesh Wrap iter {iter_idx + 1}: triangle normals invalid, "
                    f"falling back to KDTree estimation"
                )
            _estimate_orient_normals(
                pcd,
                radius=normal_radius,
                max_nn=params.normal_max_nn,
                orient_k=params.normal_orient_k,
                center=source_obb_center,
            )

        _emit_progress(
            progress_cb,
            iter_start + (iter_end - iter_start) * 0.40,
            f"Mesh Wrap: running Poisson reconstruction ({iter_idx + 1}/{params.iterations})",
        )
        try:
            mesh_new, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=params.poisson_depth,
                width=0,
                scale=params.poisson_scale,
                linear_fit=params.poisson_linear_fit,
                n_threads=0,
            )
        except TypeError:
            mesh_new, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=params.poisson_depth,
                width=0,
                scale=params.poisson_scale,
                linear_fit=params.poisson_linear_fit,
            )

        if len(mesh_new.vertices) == 0 or len(mesh_new.triangles) == 0:
            raise RuntimeError("Poisson wrap produced an empty mesh.")

        density_arr = np.asarray(densities)
        if density_arr.size > 0 and params.density_trim_q > 0.0:
            th = float(np.quantile(density_arr, params.density_trim_q))
            drop_mask = density_arr < th
            if 0 < int(drop_mask.sum()) < int(drop_mask.size):
                mesh_new.remove_vertices_by_mask(drop_mask)
                print(
                    f"Mesh Wrap iter {iter_idx + 1}: "
                    f"density trim q={params.density_trim_q:.3f}, "
                    f"removed={int(drop_mask.sum()):,}/{int(drop_mask.size):,}"
                )

        crop_bbox = o3d.geometry.OrientedBoundingBox(
            source_obb_center,
            source_obb_R,
            source_obb_extent * params.crop_scale,
        )
        cropped = mesh_new.crop(crop_bbox)
        if len(cropped.vertices) > 0 and len(cropped.triangles) > 0:
            mesh_new = cropped

        _emit_progress(
            progress_cb,
            iter_start + (iter_end - iter_start) * 0.70,
            f"Mesh Wrap: cleaning wrapped mesh ({iter_idx + 1}/{params.iterations})",
        )
        mesh_new = _clean_mesh(mesh_new)
        if params.keep_largest_component:
            mesh_new = _keep_largest_component(mesh_new)
            mesh_new = _clean_mesh(mesh_new)

        if len(mesh_new.vertices) == 0 or len(mesh_new.triangles) == 0:
            raise RuntimeError("Mesh wrap cleanup produced an empty mesh.")

        wrapped = mesh_new
        print(
            f"Mesh Wrap iter {iter_idx + 1}/{params.iterations}: "
            f"{len(wrapped.vertices):,} verts, {len(wrapped.triangles):,} faces"
        )
        _emit_progress(
            progress_cb,
            iter_end,
            f"Mesh Wrap: iteration {iter_idx + 1}/{params.iterations} complete",
        )

    _emit_progress(progress_cb, 83.0, "Mesh Wrap: adjusting final face count")
    target_faces = int(round(float(source_faces) * params.target_face_ratio))
    target_faces = max(params.min_faces, target_faces)
    target_faces = min(params.max_faces, target_faces)
    current_faces = int(len(wrapped.triangles))
    if current_faces > target_faces:
        effective_target = min(max(4, target_faces), max(4, current_faces - 1))
        simplified = wrapped.simplify_quadric_decimation(effective_target)
        if len(simplified.vertices) > 0 and len(simplified.triangles) > 0:
            wrapped = _clean_mesh(simplified)
            print(
                f"Mesh Wrap face trim: {current_faces:,} -> {len(wrapped.triangles):,} "
                f"(target={target_faces:,})"
            )

    # Taubin smoothing (Phase 1)
    wrapped = _apply_taubin_smoothing(wrapped, params.smooth_iterations)

    wrapped.compute_vertex_normals()
    return wrapped


def run_mesh_wrap(
    mesh_ply: str,
    output_dir: str,
    progress_cb: ProgressCallback | None = None,
    *,
    poisson_depth: int | None = None,
    poisson_scale: float | None = None,
    density_trim_q: float | None = None,
    target_face_ratio: float | None = None,
    iterations: int | None = None,
    crop_scale: float | None = None,
    sample_points: int | None = None,
    normal_radius_ratio: float | None = None,
    smooth_iterations: int | None = None,
    quality_threshold: float | None = None,
    method: str | None = None,
    alpha_ratio: float | None = None,
    offset_ratio: float | None = None,
) -> Path:
    """Wrap the input mesh with a Poisson shell or CGAL alpha wrap for downstream texturing."""
    overrides: dict = {}
    for key, val in [
        ("poisson_depth", poisson_depth),
        ("poisson_scale", poisson_scale),
        ("density_trim_q", density_trim_q),
        ("target_face_ratio", target_face_ratio),
        ("iterations", iterations),
        ("crop_scale", crop_scale),
        ("sample_points", sample_points),
        ("normal_radius_ratio", normal_radius_ratio),
        ("smooth_iterations", smooth_iterations),
        ("quality_threshold", quality_threshold),
        ("method", method),
        ("alpha_ratio", alpha_ratio),
        ("offset_ratio", offset_ratio),
    ]:
        if val is not None:
            overrides[key] = val
    params = _resolve_params(overrides if overrides else None)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wrap_dir = out / "mesh_wrap"
    wrap_dir.mkdir(parents=True, exist_ok=True)
    wrapped_path = out / "object_mesh_wrapped.ply"

    print("=== Mesh Wrap: Iterative Poisson shell ===")
    _emit_progress(progress_cb, 4.0, "Mesh Wrap: loading mesh")
    source_mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    if len(source_mesh.vertices) == 0 or len(source_mesh.triangles) == 0:
        raise ValueError(f"Mesh is empty: {mesh_ply}")
    source_mesh = _clean_mesh(source_mesh)
    source_diag = _bbox_diag(source_mesh)
    print(
        f"Mesh Wrap params: enabled={params.enabled}, method={params.method}, "
        f"iterations={params.iterations}, sample_points={params.sample_points:,}, "
        f"depth={params.poisson_depth}, scale={params.poisson_scale:.3f}, "
        f"density_trim_q={params.density_trim_q:.3f}, smooth_iters={params.smooth_iterations}, "
        f"quality_threshold={params.quality_threshold:.4f}"
    )
    print(
        f"Input mesh: {len(source_mesh.vertices):,} verts, "
        f"{len(source_mesh.triangles):,} faces"
    )

    if not params.enabled:
        print("Mesh Wrap disabled. Copying Stage 5 mesh as wrapped mesh.")
        _emit_progress(progress_cb, 85.0, "Mesh Wrap: disabled (copying source mesh)")
        wrapped = source_mesh
    else:
        try:
            if params.method == "alpha_wrap":
                wrapped = _run_alpha_wrap(source_mesh, params, progress_cb)
            else:
                wrapped = _run_poisson_wrap(source_mesh, params, progress_cb)
        except Exception as e:
            if not params.preserve_input_on_failure:
                raise
            print(
                "Mesh Wrap failed, preserving source mesh. "
                f"(reason: {e})"
            )
            wrapped = source_mesh

        # Quality gate (Phase 2): auto-bypass on quality degradation
        # Skip for alpha_wrap — its offset is intentional, not degradation
        if wrapped is not source_mesh and params.quality_threshold > 0.0 and params.method != "alpha_wrap":
            _emit_progress(progress_cb, 88.0, "Mesh Wrap: quality gate check")
            chamfer = _compute_chamfer_distance(source_mesh, wrapped)
            chamfer_ratio = chamfer / source_diag if source_diag > 0 else 0.0
            if chamfer_ratio > params.quality_threshold:
                print(
                    f"Mesh Wrap quality gate FAILED: "
                    f"chamfer_ratio={chamfer_ratio:.4f} > threshold={params.quality_threshold:.4f} "
                    f"(chamfer={chamfer:.6f}, diag={source_diag:.4f}). "
                    f"Auto-bypassing to source mesh."
                )
                wrapped = source_mesh
            else:
                print(
                    f"Mesh Wrap quality gate PASSED: "
                    f"chamfer_ratio={chamfer_ratio:.4f} <= threshold={params.quality_threshold:.4f}"
                )

    wrapped_vertices = np.asarray(wrapped.vertices)
    wrapped_faces = np.asarray(wrapped.triangles)
    oriented_faces, flipped, ratio_before, ratio_after = orient_faces_outward(
        wrapped_vertices,
        wrapped_faces,
        min_outward_ratio=0.5,
    )
    if flipped:
        wrapped.triangles = o3d.utility.Vector3iVector(oriented_faces.astype(np.int32))
        wrapped.compute_vertex_normals()
        print(
            "Mesh Wrap orientation fix: flipped face winding "
            f"(outward_ratio {ratio_before:.3f} -> {ratio_after:.3f})"
        )

    _emit_progress(progress_cb, 94.0, "Mesh Wrap: saving wrapped mesh")
    _write_mesh_safe(wrapped_path, wrapped)
    _write_mesh_safe(wrap_dir / "object_mesh_wrapped.ply", wrapped)
    print(f"Saved wrapped mesh: {wrapped_path}")
    print(f"  Wrapped vertices: {len(wrapped.vertices):,}, Faces: {len(wrapped.triangles):,}")
    _emit_progress(progress_cb, 100.0, "Mesh Wrap: complete")
    return wrapped_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mesh wrap via iterative Poisson shell")
    parser.add_argument("mesh_ply", help="Input mesh PLY file")
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()
    run_mesh_wrap(args.mesh_ply, args.output_dir)
