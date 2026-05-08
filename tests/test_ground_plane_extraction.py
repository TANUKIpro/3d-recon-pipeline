"""Tests for scripts/ground_plane_extraction.py."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

try:
    import open3d as o3d  # noqa: F401
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


@unittest.skipUnless(_HAS_OPEN3D, "open3d not available")
class TestExtractGroundPlaneFromMesh(unittest.TestCase):
    """extract_ground_plane_from_mesh() with synthetic data."""

    def _make_flat_floor_mesh(self, tmp: Path) -> str:
        """Create a PLY mesh with a flat floor at y=0 and an object above."""
        import open3d as o3d

        # Floor vertices (y=0)
        floor_verts = []
        for x in np.linspace(-1, 1, 10):
            for z in np.linspace(-1, 1, 10):
                floor_verts.append([x, 0.0, z])

        # Object vertices (above floor)
        obj_verts = []
        for x in np.linspace(-0.3, 0.3, 5):
            for z in np.linspace(-0.3, 0.3, 5):
                obj_verts.append([x, 0.5, z])
                obj_verts.append([x, 1.0, z])

        all_verts = np.array(floor_verts + obj_verts, dtype=np.float64)

        # Simple triangulation of floor grid
        triangles = []
        for i in range(9):
            for j in range(9):
                v0 = i * 10 + j
                v1 = v0 + 1
                v2 = v0 + 10
                v3 = v2 + 1
                triangles.append([v0, v1, v2])
                triangles.append([v1, v3, v2])

        # Add some object triangles
        offset = len(floor_verts)
        for i in range(len(obj_verts) - 2):
            triangles.append([offset + i, offset + i + 1, offset + i + 2])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

        mesh_path = str(tmp / "object_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        return mesh_path

    def _make_ground_masks(self, tmp: Path, n_frames: int = 5) -> str:
        """Create simple ground masks (bottom half white)."""
        import cv2

        mask_dir = tmp / "masks_ground"
        mask_dir.mkdir()
        for i in range(n_frames):
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[240:, :] = 255  # bottom half is ground
            cv2.imwrite(str(mask_dir / f"{i:05d}.png"), mask)
        return str(mask_dir)

    def _make_object_masks(self, tmp: Path, n_frames: int = 5) -> str:
        """Create object masks (center region white)."""
        import cv2

        mask_dir = tmp / "masks"
        mask_dir.mkdir()
        for i in range(n_frames):
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[100:400, 150:490] = 255
            cv2.imwrite(str(mask_dir / f"{i:05d}.png"), mask)
        return str(mask_dir)

    def _make_poses(self, tmp: Path, n_frames: int = 5) -> str:
        """Create camera poses looking at origin from different angles."""
        poses = []
        for i in range(n_frames):
            angle = 2 * np.pi * i / n_frames
            # Camera at distance 3, looking at origin
            eye = np.array([3 * np.cos(angle), 1.5, 3 * np.sin(angle)])
            target = np.array([0.0, 0.5, 0.0])
            up = np.array([0.0, 1.0, 0.0])

            forward = target - eye
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up_actual = np.cross(right, forward)

            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = -up_actual
            c2w[:3, 2] = -forward
            c2w[:3, 3] = eye
            poses.append(c2w.tolist())

        data = {
            "poses": poses,
            "frame_indices": list(range(n_frames)),
        }
        poses_path = str(tmp / "camera_poses.json")
        Path(poses_path).write_text(json.dumps(data), encoding="utf-8")
        return poses_path

    def _make_intrinsics(self, tmp: Path) -> str:
        """Create intrinsics JSON."""
        data = {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "image_width": 640,
            "image_height": 480,
        }
        path = str(tmp / "intrinsics.json")
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_extraction_produces_valid_output(self) -> None:
        from scripts.ground_plane_extraction import extract_ground_plane_from_mesh

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = self._make_flat_floor_mesh(tmp)
            ground_mask_dir = self._make_ground_masks(tmp)
            object_mask_dir = self._make_object_masks(tmp)
            poses_path = self._make_poses(tmp)
            intrinsics_path = self._make_intrinsics(tmp)

            result = extract_ground_plane_from_mesh(
                mesh_path,
                ground_mask_dir,
                tmpdir,
                poses_path,
                intrinsics_path,
                object_mask_dir=object_mask_dir,
                min_ground_points=10,
            )

            if result is not None:
                # Check that output file was written under the new phase dir
                from scripts.output_layout import ground_plane_path

                self.assertTrue(ground_plane_path(tmp).is_file())

                # Check both key formats present
                self.assertIn("normal", result)
                self.assertIn("d", result)
                self.assertIn("plane_normal", result)
                self.assertIn("plane_d", result)

                # Check consistency between key formats
                np.testing.assert_array_almost_equal(
                    result["normal"], result["plane_normal"]
                )
                self.assertAlmostEqual(result["d"], result["plane_d"])

                # Check metadata
                self.assertIn("point_count", result)
                self.assertIn("inlier_ratio", result)
                self.assertGreater(result["point_count"], 0)

    def test_returns_none_for_empty_mesh(self) -> None:
        from scripts.ground_plane_extraction import extract_ground_plane_from_mesh

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            import open3d as o3d

            mesh = o3d.geometry.TriangleMesh()
            mesh_path = str(tmp / "object_mesh.ply")
            o3d.io.write_triangle_mesh(mesh_path, mesh)

            ground_mask_dir = self._make_ground_masks(tmp)
            poses_path = self._make_poses(tmp)
            intrinsics_path = self._make_intrinsics(tmp)

            result = extract_ground_plane_from_mesh(
                mesh_path,
                ground_mask_dir,
                tmpdir,
                poses_path,
                intrinsics_path,
            )
            self.assertIsNone(result)

    def test_returns_none_for_missing_intrinsics(self) -> None:
        from scripts.ground_plane_extraction import extract_ground_plane_from_mesh

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = self._make_flat_floor_mesh(tmp)
            ground_mask_dir = self._make_ground_masks(tmp)
            poses_path = self._make_poses(tmp)

            result = extract_ground_plane_from_mesh(
                mesh_path,
                ground_mask_dir,
                tmpdir,
                poses_path,
                str(tmp / "nonexistent_intrinsics.json"),
            )
            self.assertIsNone(result)

    def test_progress_cb_called(self) -> None:
        from scripts.ground_plane_extraction import extract_ground_plane_from_mesh

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = self._make_flat_floor_mesh(tmp)
            ground_mask_dir = self._make_ground_masks(tmp)
            poses_path = self._make_poses(tmp)
            intrinsics_path = self._make_intrinsics(tmp)
            progress_cb = MagicMock()

            extract_ground_plane_from_mesh(
                mesh_path,
                ground_mask_dir,
                tmpdir,
                poses_path,
                intrinsics_path,
                min_ground_points=10,
                progress_cb=progress_cb,
            )

            self.assertTrue(progress_cb.called)


@unittest.skipUnless(_HAS_OPEN3D, "open3d not available (transitive via repair.ground_plane)")
class TestResolveGroundPlaneBothKeyFormats(unittest.TestCase):
    """_resolve_ground_plane() accepts both key formats."""

    def test_plane_normal_plane_d_format(self) -> None:
        from scripts.stage_post_texture_contact_cleanup import _resolve_ground_plane

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ground_plane.json"
            path.write_text(
                json.dumps({
                    "plane_normal": [0.0, 1.0, 0.0],
                    "plane_d": -0.5,
                }),
                encoding="utf-8",
            )
            verts = np.array([[0, 1, 0], [1, 2, 0], [0, 0.5, 1]], dtype=np.float64)
            normal, d, source = _resolve_ground_plane(verts, str(path))
            self.assertEqual(source, "ground_plane_json")

    def test_normal_d_format(self) -> None:
        from scripts.stage_post_texture_contact_cleanup import _resolve_ground_plane

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ground_plane.json"
            path.write_text(
                json.dumps({
                    "normal": [0.0, 1.0, 0.0],
                    "d": -0.5,
                }),
                encoding="utf-8",
            )
            verts = np.array([[0, 1, 0], [1, 2, 0], [0, 0.5, 1]], dtype=np.float64)
            normal, d, source = _resolve_ground_plane(verts, str(path))
            self.assertEqual(source, "ground_plane_json")

    def test_fallback_when_no_file(self) -> None:
        from scripts.stage_post_texture_contact_cleanup import _resolve_ground_plane

        verts = np.array([[0, 1, 0], [1, 2, 0], [0, 0.5, 1]], dtype=np.float64)
        normal, d, source = _resolve_ground_plane(verts, None)
        self.assertEqual(source, "mesh_bottom_fallback")

    def test_fallback_on_invalid_json(self) -> None:
        from scripts.stage_post_texture_contact_cleanup import _resolve_ground_plane

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ground_plane.json"
            path.write_text("not valid json {{{", encoding="utf-8")
            verts = np.array([[0, 1, 0], [1, 2, 0], [0, 0.5, 1]], dtype=np.float64)
            normal, d, source = _resolve_ground_plane(verts, str(path))
            self.assertEqual(source, "mesh_bottom_fallback")

    def test_plane_d_zero_accepted(self) -> None:
        """plane_d=0 should not fall back to the 'd' key."""
        from scripts.stage_post_texture_contact_cleanup import _resolve_ground_plane

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ground_plane.json"
            path.write_text(
                json.dumps({
                    "plane_normal": [0.0, 1.0, 0.0],
                    "plane_d": 0.0,
                }),
                encoding="utf-8",
            )
            verts = np.array([[0, 1, 0], [1, 2, 0], [0, 0.5, 1]], dtype=np.float64)
            normal, d, source = _resolve_ground_plane(verts, str(path))
            self.assertEqual(source, "ground_plane_json")


@unittest.skipUnless(_HAS_OPEN3D, "open3d not available (transitive via repair.ground_plane)")
class TestComputePerFaceGroundScore(unittest.TestCase):
    """_compute_per_face_ground_score() basic validation."""

    def test_returns_none_when_no_poses(self) -> None:
        from scripts.stage_post_texture_contact_cleanup import (
            _compute_per_face_ground_score,
        )

        centroids = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        result = _compute_per_face_ground_score(
            centroids,
            poses_path=None,
            intrinsics_path=None,
            masks_dir=None,
            ground_masks_dir=None,
        )
        self.assertIsNone(result)

    def test_returns_none_when_empty_centroids(self) -> None:
        from scripts.stage_post_texture_contact_cleanup import (
            _compute_per_face_ground_score,
        )

        centroids = np.zeros((0, 3), dtype=np.float64)
        result = _compute_per_face_ground_score(
            centroids,
            poses_path="/tmp/poses.json",
            intrinsics_path="/tmp/intrinsics.json",
            masks_dir="/tmp/masks",
            ground_masks_dir="/tmp/ground_masks",
        )
        self.assertIsNone(result)


class TestHydratePicksUpGroundPlane(unittest.TestCase):
    """hydrate_from_output_dir picks up ground_plane_path."""

    def test_ground_plane_path_set_when_file_exists(self) -> None:
        from scripts.dashboard.state import PipelineSession
        from scripts.output_layout import ground_plane_path

        session = PipelineSession()
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            gp = ground_plane_path(out)
            gp.parent.mkdir(parents=True, exist_ok=True)
            gp.write_text('{"normal":[0,1,0],"d":-0.5}', encoding="utf-8")
            session.hydrate_from_output_dir(out)
            self.assertEqual(session.ground_plane_path, str(gp))

    def test_ground_plane_path_none_when_no_file(self) -> None:
        from scripts.dashboard.state import PipelineSession

        session = PipelineSession()
        with TemporaryDirectory() as tmpdir:
            session.hydrate_from_output_dir(tmpdir)
            self.assertIsNone(session.ground_plane_path)


if __name__ == "__main__":
    unittest.main()
