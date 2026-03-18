"""Tests for spatial-partition parallel UV atlas generation."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from scripts.texture.parallel_uv import (
    _extract_partition_mesh,
    _spatial_partition_faces,
    parallel_xatlas_generate,
)


def _make_grid_mesh(
    rows: int = 10,
    cols: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple grid mesh for testing.

    Returns:
        (vertices, faces) — a (rows+1)*(cols+1) vertex grid with 2*rows*cols faces.
    """
    xs = np.linspace(0.0, 1.0, cols + 1)
    ys = np.linspace(0.0, 1.0, rows + 1)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros((rows + 1) * (cols + 1))])

    faces = []
    for r in range(rows):
        for c in range(cols):
            v0 = r * (cols + 1) + c
            v1 = v0 + 1
            v2 = v0 + (cols + 1)
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces


class SpatialPartitionTests(unittest.TestCase):
    def test_all_faces_assigned(self) -> None:
        """Every face must appear in exactly one partition."""
        verts, faces = _make_grid_mesh(10, 10)
        for n in (2, 4, 8):
            parts = _spatial_partition_faces(verts, faces, n)
            all_indices = np.concatenate(parts)
            self.assertEqual(
                len(all_indices),
                len(faces),
                f"n_partitions={n}: face count mismatch",
            )
            self.assertEqual(
                len(np.unique(all_indices)),
                len(faces),
                f"n_partitions={n}: duplicate face indices",
            )

    def test_partitions_roughly_balanced(self) -> None:
        """Partition sizes should be within 2x of each other."""
        verts, faces = _make_grid_mesh(20, 20)
        parts = _spatial_partition_faces(verts, faces, 4)
        sizes = [len(p) for p in parts]
        self.assertGreaterEqual(min(sizes) * 2, max(sizes))

    def test_rounds_to_power_of_two(self) -> None:
        """Requesting 3 partitions should produce 4."""
        verts, faces = _make_grid_mesh(10, 10)
        parts = _spatial_partition_faces(verts, faces, 3)
        self.assertEqual(len(parts), 4)

    def test_single_partition(self) -> None:
        """n_partitions=1 should return all faces in one group."""
        verts, faces = _make_grid_mesh(5, 5)
        parts = _spatial_partition_faces(verts, faces, 1)
        self.assertEqual(len(parts), 1)
        self.assertEqual(len(parts[0]), len(faces))

    def test_single_face_mesh(self) -> None:
        """Mesh with one face should not crash on partitioning."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        parts = _spatial_partition_faces(verts, faces, 4)
        all_indices = np.concatenate(parts)
        self.assertEqual(len(all_indices), 1)


class ExtractPartitionMeshTests(unittest.TestCase):
    def test_vertex_remap_correctness(self) -> None:
        """Remapped vertices should match the originals at the global indices."""
        verts, faces = _make_grid_mesh(5, 5)
        normals = np.random.default_rng(42).standard_normal(verts.shape)

        parts = _spatial_partition_faces(verts, faces, 2)
        for part_idx, face_indices in enumerate(parts):
            local_v, local_f, local_n, v_remap = _extract_partition_mesh(
                verts, faces, normals, face_indices,
            )
            # vertex_remap maps local → global
            np.testing.assert_array_almost_equal(
                local_v,
                verts[v_remap],
                err_msg=f"Partition {part_idx}: vertex data mismatch",
            )
            np.testing.assert_array_almost_equal(
                local_n,
                normals[v_remap],
                err_msg=f"Partition {part_idx}: normal data mismatch",
            )
            # Local faces should reference valid local indices
            self.assertTrue(np.all(local_f >= 0))
            self.assertTrue(np.all(local_f < len(local_v)))

    def test_local_faces_reconstruct_global(self) -> None:
        """Mapping local faces back through vertex_remap should reproduce global faces."""
        verts, faces = _make_grid_mesh(4, 4)
        normals = np.zeros_like(verts)
        face_indices = np.array([0, 1, 2, 3], dtype=np.int64)

        local_v, local_f, _, v_remap = _extract_partition_mesh(
            verts, faces, normals, face_indices,
        )
        global_faces_reconstructed = v_remap[local_f]
        np.testing.assert_array_equal(
            np.sort(global_faces_reconstructed, axis=1),
            np.sort(faces[face_indices], axis=1),
        )


class ParallelXatlasGenerateTests(unittest.TestCase):
    def test_returns_none_for_small_mesh(self) -> None:
        """Meshes below the face threshold should return None."""
        verts, faces = _make_grid_mesh(5, 5)  # 50 faces
        normals = np.zeros_like(verts)
        normals[:, 2] = 1.0
        with patch.dict(os.environ, {"TEXTURE_UV_PARALLEL": "auto"}, clear=False):
            result = parallel_xatlas_generate(verts, faces, normals)
        self.assertIsNone(result)

    def test_returns_none_when_disabled(self) -> None:
        """TEXTURE_UV_PARALLEL=off should always return None."""
        verts, faces = _make_grid_mesh(100, 100)  # 20000 faces
        normals = np.zeros_like(verts)
        normals[:, 2] = 1.0
        with patch.dict(os.environ, {"TEXTURE_UV_PARALLEL": "off"}, clear=False):
            result = parallel_xatlas_generate(verts, faces, normals)
        self.assertIsNone(result)

    def test_returns_none_single_worker(self) -> None:
        """n_workers=1 should return None (no benefit from parallelism)."""
        verts, faces = _make_grid_mesh(100, 100)
        normals = np.zeros_like(verts)
        normals[:, 2] = 1.0
        with patch.dict(os.environ, {"TEXTURE_UV_PARALLEL": "auto"}, clear=False):
            result = parallel_xatlas_generate(verts, faces, normals, n_workers=1)
        self.assertIsNone(result)

    def test_vmapping_values_are_valid_vertex_indices(self) -> None:
        """All vmapping values must be valid indices into the input vertices."""
        verts, faces = _make_grid_mesh(80, 80)  # 12800 faces
        normals = np.zeros_like(verts)
        normals[:, 2] = 1.0
        with patch.dict(os.environ, {"TEXTURE_UV_PARALLEL": "auto"}, clear=False):
            result = parallel_xatlas_generate(verts, faces, normals, n_workers=2)
        if result is None:
            self.skipTest("xatlas not available in this environment")
        vmapping, new_faces, uvs = result
        self.assertTrue(np.all(vmapping >= 0))
        self.assertTrue(np.all(vmapping < len(verts)))
        # UV coordinates should be in [0, 1]
        self.assertTrue(np.all(uvs >= 0.0))
        self.assertTrue(np.all(uvs <= 1.0))
        # new_faces should reference valid UV vertex indices
        self.assertTrue(np.all(new_faces >= 0))
        self.assertTrue(np.all(new_faces < len(uvs)))

    def test_output_face_count_matches_input(self) -> None:
        """Parallel result should have the same number of faces as the input."""
        verts, faces = _make_grid_mesh(80, 80)
        normals = np.zeros_like(verts)
        normals[:, 2] = 1.0
        with patch.dict(os.environ, {"TEXTURE_UV_PARALLEL": "auto"}, clear=False):
            result = parallel_xatlas_generate(verts, faces, normals, n_workers=2)
        if result is None:
            self.skipTest("xatlas not available in this environment")
        _, new_faces, _ = result
        self.assertEqual(len(new_faces), len(faces))

    def test_graceful_fallback_on_error(self) -> None:
        """If xatlas worker raises, parallel_xatlas_generate should return None."""
        verts, faces = _make_grid_mesh(80, 80)
        normals = np.zeros_like(verts)
        normals[:, 2] = 1.0
        with patch.dict(os.environ, {"TEXTURE_UV_PARALLEL": "auto"}, clear=False):
            with patch(
                "scripts.texture.parallel_uv._xatlas_worker",
                side_effect=RuntimeError("mock failure"),
            ):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = parallel_xatlas_generate(verts, faces, normals, n_workers=2)
        self.assertIsNone(result)
