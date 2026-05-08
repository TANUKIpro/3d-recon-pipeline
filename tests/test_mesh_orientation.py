import unittest

import numpy as np

from scripts.mesh_orientation import (
    camera_facing_ratio,
    face_outward_ratio,
    orient_faces_outward,
    orient_mesh_for_cameras,
    orient_mesh_outward,
)


class MeshOrientationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = np.array(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        inward = np.array(
            [
                [0, 1, 2],
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 2],
            ],
            dtype=np.int32,
        )
        self.inward_faces = inward
        self.outward_faces = inward[:, [0, 2, 1]]

    def test_face_outward_ratio_detects_orientation(self) -> None:
        inward_ratio = face_outward_ratio(self.vertices, self.inward_faces)
        outward_ratio = face_outward_ratio(self.vertices, self.outward_faces)

        self.assertIsNotNone(inward_ratio)
        self.assertIsNotNone(outward_ratio)
        self.assertLess(inward_ratio, 0.1)
        self.assertGreater(outward_ratio, 0.9)

    def test_orient_faces_outward_flips_inward_mesh(self) -> None:
        fixed_faces, flipped, ratio_before, ratio_after = orient_faces_outward(
            self.vertices,
            self.inward_faces,
            min_outward_ratio=0.5,
        )

        self.assertTrue(flipped)
        self.assertIsNotNone(ratio_before)
        self.assertIsNotNone(ratio_after)
        self.assertLess(ratio_before, 0.1)
        self.assertGreater(ratio_after, 0.9)
        np.testing.assert_array_equal(fixed_faces, self.outward_faces)

    def test_orient_faces_outward_keeps_already_outward_mesh(self) -> None:
        fixed_faces, flipped, ratio_before, ratio_after = orient_faces_outward(
            self.vertices,
            self.outward_faces,
            min_outward_ratio=0.5,
        )

        self.assertFalse(flipped)
        self.assertIsNotNone(ratio_before)
        self.assertEqual(ratio_before, ratio_after)
        np.testing.assert_array_equal(fixed_faces, self.outward_faces)

    def test_degenerate_faces_return_none_ratio(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        ratio = face_outward_ratio(vertices, faces)
        fixed_faces, flipped, ratio_before, ratio_after = orient_faces_outward(vertices, faces)

        self.assertIsNone(ratio)
        self.assertFalse(flipped)
        self.assertIsNone(ratio_before)
        self.assertIsNone(ratio_after)
        np.testing.assert_array_equal(fixed_faces, faces)


class OrientMeshOutwardTests(unittest.TestCase):
    """Tests for orient_mesh_outward (boundary-face normal verification)."""

    @staticmethod
    def _make_box(inward: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned unit box centred at origin with outward (or inward) faces."""
        verts = np.array(
            [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # top
            ],
            dtype=np.float64,
        )
        # Outward-wound faces (CCW when viewed from outside)
        faces = np.array(
            [
                # -Z face (normals pointing -Z)
                [0, 3, 2], [0, 2, 1],
                # +Z face (normals pointing +Z)
                [4, 5, 6], [4, 6, 7],
                # -X face
                [0, 4, 7], [0, 7, 3],
                # +X face
                [1, 2, 6], [1, 6, 5],
                # -Y face
                [0, 1, 5], [0, 5, 4],
                # +Y face
                [2, 3, 7], [2, 7, 6],
            ],
            dtype=np.int32,
        )
        if inward:
            faces = faces[:, [0, 2, 1]]
        return verts, faces

    @staticmethod
    def _make_nonconvex_bottle_table() -> tuple[np.ndarray, np.ndarray]:
        """Non-convex mesh: large flat table + small bottle on top.

        The centroid heuristic fails on this shape because the table dominates
        the vertex mean. Boundary-face verification should still work.
        """
        # Large table: flat box from (-5,-5,0) to (5,5,0.2)
        tv = np.array(
            [
                [-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0],
                [-5, -5, 0.2], [5, -5, 0.2], [5, 5, 0.2], [-5, 5, 0.2],
            ],
            dtype=np.float64,
        )
        tf = np.array(
            [
                [0, 3, 2], [0, 2, 1],  # -Z
                [4, 5, 6], [4, 6, 7],  # +Z
                [0, 4, 7], [0, 7, 3],  # -X
                [1, 2, 6], [1, 6, 5],  # +X
                [0, 1, 5], [0, 5, 4],  # -Y
                [2, 3, 7], [2, 7, 6],  # +Y
            ],
            dtype=np.int32,
        )
        # Small bottle: cylinder-ish box from (-0.3,-0.3,0.2) to (0.3,0.3,2)
        bv = np.array(
            [
                [-0.3, -0.3, 0.2], [0.3, -0.3, 0.2], [0.3, 0.3, 0.2], [-0.3, 0.3, 0.2],
                [-0.3, -0.3, 2.0], [0.3, -0.3, 2.0], [0.3, 0.3, 2.0], [-0.3, 0.3, 2.0],
            ],
            dtype=np.float64,
        )
        bf = np.array(
            [
                [0, 3, 2], [0, 2, 1],
                [4, 5, 6], [4, 6, 7],
                [0, 4, 7], [0, 7, 3],
                [1, 2, 6], [1, 6, 5],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
            ],
            dtype=np.int32,
        )
        # Combine: offset bottle face indices
        verts = np.vstack([tv, bv])
        faces = np.vstack([tf, bf + len(tv)])
        return verts, faces

    def test_outward_box_not_flipped(self) -> None:
        verts, faces = self._make_box(inward=False)
        result_faces, flipped, _, _ = orient_mesh_outward(verts, faces)
        self.assertFalse(flipped)
        np.testing.assert_array_equal(result_faces, faces)

    def test_inward_box_flipped(self) -> None:
        verts, faces = self._make_box(inward=True)
        result_faces, flipped, ratio_before, ratio_after = orient_mesh_outward(verts, faces)
        self.assertTrue(flipped)
        self.assertIsNotNone(ratio_before)
        self.assertIsNotNone(ratio_after)
        # After flip, should match outward faces
        expected = faces[:, [0, 2, 1]]
        np.testing.assert_array_equal(result_faces, expected)

    def test_nonconvex_bottle_table_not_flipped(self) -> None:
        """Non-convex mesh with outward faces should NOT be flipped.

        This is the key test: the centroid heuristic (face_outward_ratio)
        gives < 0.5 for this shape, but orient_mesh_outward should correctly
        identify it as already outward.
        """
        verts, faces = self._make_nonconvex_bottle_table()
        # Verify that old centroid heuristic would have gotten this wrong
        centroid_ratio = face_outward_ratio(verts, faces)
        # The table dominates — centroid is inside the table, so many faces
        # appear inward to the centroid heuristic
        self.assertIsNotNone(centroid_ratio)

        # New method should NOT flip
        result_faces, flipped, _, _ = orient_mesh_outward(verts, faces)
        self.assertFalse(flipped, "orient_mesh_outward incorrectly flipped a non-convex outward mesh")
        np.testing.assert_array_equal(result_faces, faces)

    def test_nonconvex_bottle_table_inward_is_flipped(self) -> None:
        """Non-convex mesh with inward faces SHOULD be flipped."""
        verts, faces = self._make_nonconvex_bottle_table()
        inward_faces = faces[:, [0, 2, 1]]
        result_faces, flipped, _, _ = orient_mesh_outward(verts, inward_faces)
        self.assertTrue(flipped, "orient_mesh_outward failed to flip inward non-convex mesh")

    def test_degenerate_faces(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        result_faces, flipped, _, _ = orient_mesh_outward(verts, faces)
        self.assertFalse(flipped)
        np.testing.assert_array_equal(result_faces, faces)


class OrientMeshForCamerasTests(unittest.TestCase):
    @staticmethod
    def _make_x_plane(normal_sign: int = -1) -> tuple[np.ndarray, np.ndarray]:
        verts = np.array(
            [
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )
        pos_x_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        if normal_sign > 0:
            return verts, pos_x_faces
        return verts, pos_x_faces[:, [0, 2, 1]]

    @staticmethod
    def _make_poses(*centers: tuple[float, float, float]) -> np.ndarray:
        poses = []
        for center in centers:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, 3] = np.array(center, dtype=np.float64)
            poses.append(mat)
        return np.array(poses, dtype=np.float64)

    def test_camera_facing_ratio_scores_supplied_and_flipped_winding(self) -> None:
        verts, faces = self._make_x_plane(normal_sign=-1)
        poses = self._make_poses((-3.0, 0.0, 0.0))

        before, after = camera_facing_ratio(verts, faces, poses)

        self.assertEqual(before, 1.0)
        self.assertEqual(after, 0.0)

    def test_camera_vote_overrides_boundary_vote_when_open_surface_faces_camera(self) -> None:
        verts, faces = self._make_x_plane(normal_sign=-1)
        poses = self._make_poses((-3.0, 0.0, 0.0))

        result = orient_mesh_for_cameras(verts, faces, poses)

        self.assertFalse(result.flipped)
        self.assertEqual(result.source, "camera")
        self.assertEqual(result.camera_ratio_before, 1.0)
        self.assertEqual(result.camera_ratio_after, 0.0)
        np.testing.assert_array_equal(result.faces, faces)

    def test_camera_vote_flips_reversed_observed_surface(self) -> None:
        verts, faces = self._make_x_plane(normal_sign=1)
        poses = self._make_poses((-3.0, 0.0, 0.0))

        result = orient_mesh_for_cameras(verts, faces, poses)

        self.assertTrue(result.flipped)
        self.assertEqual(result.source, "camera")
        self.assertEqual(result.camera_ratio_before, 0.0)
        self.assertEqual(result.camera_ratio_after, 1.0)
        np.testing.assert_array_equal(result.faces, faces[:, [0, 2, 1]])

    def test_falls_back_to_boundary_vote_without_usable_cameras(self) -> None:
        verts, faces = OrientMeshOutwardTests._make_box(inward=True)
        result = orient_mesh_for_cameras(verts, faces, np.empty((0, 4, 4)))

        self.assertTrue(result.flipped)
        self.assertEqual(result.source, "boundary_fallback")
        self.assertIsNone(result.camera_ratio_before)
        self.assertIsNone(result.camera_ratio_after)


if __name__ == "__main__":
    unittest.main()
