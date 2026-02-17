"""Stage 6: mesh wrapping via iterative Poisson reconstruction.

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

ProgressCallback = Callable[[float, str | None], None]

_DEFAULT_WRAP_ENABLED = True
_DEFAULT_WRAP_METHOD = "poisson_iterative"  # reserved: "ipsr"
_DEFAULT_WRAP_ITERATIONS = 2
_DEFAULT_WRAP_SAMPLE_POINTS = 300_000
_DEFAULT_WRAP_NORMAL_RADIUS_RATIO = 0.035
_DEFAULT_WRAP_NORMAL_MAX_NN = 32
_DEFAULT_WRAP_NORMAL_ORIENT_K = 24
_DEFAULT_WRAP_POISSON_DEPTH = 9
_DEFAULT_WRAP_POISSON_SCALE = 1.18
_DEFAULT_WRAP_POISSON_LINEAR_FIT = False
_DEFAULT_WRAP_DENSITY_TRIM_Q = 0.06
_DEFAULT_WRAP_CROP_SCALE = 1.08
_DEFAULT_WRAP_KEEP_LARGEST_COMPONENT = True
_DEFAULT_WRAP_TARGET_FACE_RATIO = 1.50
_DEFAULT_WRAP_MIN_FACES = 25_000
_DEFAULT_WRAP_MAX_FACES = 200_000
_DEFAULT_WRAP_PRESERVE_INPUT_ON_FAILURE = True


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

    method = str(os.environ.get("MESH_WRAP_METHOD", _DEFAULT_WRAP_METHOD)).strip().lower()
    if method not in {"poisson_iterative", "ipsr"}:
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


def _run_poisson_wrap(
    source_mesh: o3d.geometry.TriangleMesh,
    params: MeshWrapParams,
    progress_cb: ProgressCallback | None,
) -> o3d.geometry.TriangleMesh:
    wrapped = source_mesh
    source_bbox = wrapped.get_axis_aligned_bounding_box()
    source_diag = _bbox_diag(wrapped)
    source_faces = int(len(wrapped.triangles))

    if params.method == "ipsr":
        # True iPSR from the paper requires a dedicated solver/library not bundled here.
        print(
            "MESH_WRAP_METHOD='ipsr' requested. "
            "Using Poisson iterative wrap fallback (compatible in current environment)."
        )

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
        try:
            pcd = wrapped.sample_points_uniformly(
                number_of_points=sample_n,
                use_triangle_normal=True,
            )
        except TypeError:
            pcd = wrapped.sample_points_uniformly(number_of_points=sample_n)
        points = np.asarray(pcd.points)
        if points.shape[0] < 100:
            raise RuntimeError("Mesh wrap sampling produced too few points.")

        normal_radius = max(source_diag * params.normal_radius_ratio, 1e-5)
        _emit_progress(
            progress_cb,
            iter_start + (iter_end - iter_start) * 0.20,
            f"Mesh Wrap: estimating normals ({iter_idx + 1}/{params.iterations})",
        )
        _estimate_orient_normals(
            pcd,
            radius=normal_radius,
            max_nn=params.normal_max_nn,
            orient_k=params.normal_orient_k,
            center=source_bbox.get_center(),
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

        crop_bbox = source_bbox.scale(params.crop_scale, source_bbox.get_center())
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
) -> Path:
    """Wrap the input mesh with a Poisson shell for downstream texturing."""
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
    print(
        f"Mesh Wrap params: enabled={params.enabled}, method={params.method}, "
        f"iterations={params.iterations}, sample_points={params.sample_points:,}, "
        f"depth={params.poisson_depth}, scale={params.poisson_scale:.3f}, "
        f"density_trim_q={params.density_trim_q:.3f}"
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
            wrapped = _run_poisson_wrap(source_mesh, params, progress_cb)
        except Exception as e:
            if not params.preserve_input_on_failure:
                raise
            print(
                "Mesh Wrap failed, preserving source mesh. "
                f"(reason: {e})"
            )
            wrapped = source_mesh

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
