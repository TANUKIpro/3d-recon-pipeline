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
                # Check that output file was written
                self.assertTrue((tmp / "ground_plane.json").is_file())

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


@unittest.skipUnless(_HAS_OPEN3D, "open3d not available")
class TestExtractGroundPlaneFromColmapSparse(unittest.TestCase):
    """extract_ground_plane() via COLMAP sparse points."""

    def _make_colmap_model(
        self,
        tmp: Path,
        n_frames: int = 5,
        n_ground_points: int = 200,
        n_object_points: int = 100,
    ) -> str:
        """Create a synthetic COLMAP binary model with ground + object points.

        Ground points are placed near y=0, object points above y=0.5.
        2D observations are synthesised so that ground-point observations
        land in the bottom half of the image (matching ground masks) and
        object-point observations land in the top half.
        """
        from scripts.colmap_sparse_filter import (
            ColmapCamera,
            ColmapImage,
            ColmapModel,
            ColmapPoint3D,
            write_colmap_model,
        )

        rng = np.random.default_rng(42)
        sparse_dir = tmp / "colmap_sparse" / "0"
        sparse_dir.mkdir(parents=True)

        # Single SIMPLE_PINHOLE camera
        cameras = {
            1: ColmapCamera(
                camera_id=1,
                model_id=0,  # SIMPLE_PINHOLE
                width=640,
                height=480,
                params=(500.0, 320.0, 240.0),
            )
        }

        # Images
        images: dict[int, ColmapImage] = {}
        total_points = n_ground_points + n_object_points
        for i in range(1, n_frames + 1):
            # Ground-point observations: bottom half (y 240..479)
            ground_xs = rng.uniform(10, 630, size=n_ground_points)
            ground_ys = rng.uniform(260, 470, size=n_ground_points)
            # Object-point observations: top half (y 10..220)
            obj_xs = rng.uniform(100, 540, size=n_object_points)
            obj_ys = rng.uniform(10, 220, size=n_object_points)
            xys = np.column_stack([
                np.concatenate([ground_xs, obj_xs]),
                np.concatenate([ground_ys, obj_ys]),
            ])
            point3D_ids = np.arange(1, total_points + 1, dtype=np.int64)
            images[i] = ColmapImage(
                image_id=i,
                qvec=(1.0, 0.0, 0.0, 0.0),
                tvec=(0.0, 0.0, 0.0),
                camera_id=1,
                name=f"{i - 1:05d}.png",
                xys=xys,
                point3D_ids=point3D_ids,
            )

        # 3D points
        points3D: dict[int, ColmapPoint3D] = {}
        pid = 1
        # Ground points near y=0
        for _ in range(n_ground_points):
            xyz = (
                float(rng.uniform(-2, 2)),
                float(rng.uniform(-0.05, 0.05)),
                float(rng.uniform(-2, 2)),
            )
            track = tuple((img_id, pid - 1) for img_id in range(1, n_frames + 1))
            points3D[pid] = ColmapPoint3D(
                point3D_id=pid, xyz=xyz, rgb=(128, 128, 128),
                error=0.5, track=track,
            )
            pid += 1
        # Object points above ground
        for _ in range(n_object_points):
            xyz = (
                float(rng.uniform(-0.5, 0.5)),
                float(rng.uniform(0.5, 1.5)),
                float(rng.uniform(-0.5, 0.5)),
            )
            track = tuple((img_id, pid - 1) for img_id in range(1, n_frames + 1))
            points3D[pid] = ColmapPoint3D(
                point3D_id=pid, xyz=xyz, rgb=(200, 50, 50),
                error=0.5, track=track,
            )
            pid += 1

        model = ColmapModel(cameras=cameras, images=images, points3D=points3D)
        write_colmap_model(model, sparse_dir)
        return str(tmp / "colmap_sparse")

    def _make_ground_masks(self, tmp: Path, n_frames: int = 5) -> str:
        """Bottom half white = ground."""
        import cv2

        mask_dir = tmp / "masks_ground"
        mask_dir.mkdir()
        for i in range(n_frames):
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[240:, :] = 255
            cv2.imwrite(str(mask_dir / f"{i:05d}.png"), mask)
        return str(mask_dir)

    def test_colmap_sparse_extraction(self) -> None:
        from scripts.ground_plane_extraction import extract_ground_plane

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            colmap_dir = self._make_colmap_model(tmp)
            ground_mask_dir = self._make_ground_masks(tmp)

            result = extract_ground_plane(
                ground_mask_dir,
                tmpdir,
                colmap_sparse_dir=colmap_dir,
                min_ground_points=10,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["source"], "colmap_sparse")
            self.assertTrue((tmp / "ground_plane.json").is_file())

            # Check all expected keys
            for key in ("normal", "d", "plane_normal", "plane_d",
                        "center", "point_count", "inlier_ratio"):
                self.assertIn(key, result)

            # Normal should be roughly [0, 1, 0] (pointing up from ground)
            normal = np.array(result["normal"])
            self.assertGreater(abs(normal[1]), 0.8,
                               "Expected predominantly Y-axis normal for flat ground")

    def test_colmap_sparse_insufficient_falls_back_to_mesh(self) -> None:
        """When COLMAP has too few ground points, fall back to mesh."""
        from scripts.ground_plane_extraction import extract_ground_plane

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create COLMAP model with zero ground points
            colmap_dir = self._make_colmap_model(
                tmp, n_ground_points=0, n_object_points=50,
            )
            ground_mask_dir = self._make_ground_masks(tmp)

            # No mesh fallback provided either → should return None
            result = extract_ground_plane(
                ground_mask_dir,
                tmpdir,
                colmap_sparse_dir=colmap_dir,
                min_ground_points=10,
            )
            self.assertIsNone(result)

    def test_no_colmap_dir_returns_none_without_mesh(self) -> None:
        from scripts.ground_plane_extraction import extract_ground_plane

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ground_mask_dir = self._make_ground_masks(tmp)

            result = extract_ground_plane(
                ground_mask_dir,
                tmpdir,
                colmap_sparse_dir=None,
            )
            self.assertIsNone(result)


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

        session = PipelineSession()
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            gp = out / "ground_plane.json"
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
