import unittest

import numpy as np

from scripts.mesh_orientation import face_outward_ratio, orient_faces_outward


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


if __name__ == "__main__":
    unittest.main()

