"""Mesh I/O helpers for contact-hole repair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


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


def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()


def _load_mesh_arrays(mesh_ply: str) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh is empty: {mesh_ply}")
    _clean_mesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh is empty after cleanup: {mesh_ply}")
    return mesh, vertices, faces


def _prepare_repaired_mesh(
    repaired_vertices: np.ndarray,
    repaired_faces: np.ndarray,
    *,
    remove_non_manifold: bool = True,
    cap_vertex_ids: set[int] | None = None,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    repaired_mesh = o3d.geometry.TriangleMesh()
    repaired_mesh.vertices = o3d.utility.Vector3dVector(repaired_vertices)
    repaired_mesh.triangles = o3d.utility.Vector3iVector(repaired_faces.astype(np.int32))

    if cap_vertex_ids:
        n = len(repaired_vertices)
        colors = np.full((n, 3), 0.82, dtype=np.float64)  # light grey default
        cap_idx = np.array(sorted(cap_vertex_ids), dtype=np.int64)
        cap_idx = cap_idx[cap_idx < n]
        colors[cap_idx] = [1.0, 0.55, 0.0]  # orange highlight for cap
        repaired_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    repaired_mesh.remove_degenerate_triangles()
    repaired_mesh.remove_duplicated_triangles()
    repaired_mesh.remove_duplicated_vertices()
    if remove_non_manifold:
        repaired_mesh.remove_non_manifold_edges()
    repaired_mesh.remove_unreferenced_vertices()
    repaired_mesh.compute_vertex_normals()

    final_vertices = np.asarray(repaired_mesh.vertices, dtype=np.float64).copy()
    final_faces = np.asarray(repaired_mesh.triangles, dtype=np.int64).copy()
    return repaired_mesh, final_vertices, final_faces


def _write_repaired_outputs(
    repaired_vertices: np.ndarray,
    repaired_faces: np.ndarray,
    repaired_path: Path,
    repaired_copy_path: Path,
    *,
    remove_non_manifold: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    repaired_mesh, final_vertices, final_faces = _prepare_repaired_mesh(
        repaired_vertices,
        repaired_faces,
        remove_non_manifold=remove_non_manifold,
    )

    _write_mesh_safe(repaired_path, repaired_mesh)
    _write_mesh_safe(repaired_copy_path, repaired_mesh)
    return final_vertices, final_faces
