from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from plyfile import PlyData

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

if "open3d" not in sys.modules:
    open3d_stub = types.ModuleType("open3d")
    open3d_stub.geometry = types.SimpleNamespace(TriangleMesh=object)
    open3d_stub.utility = types.SimpleNamespace(Vector3dVector=object, Vector3iVector=object)
    open3d_stub.io = types.SimpleNamespace(read_triangle_mesh=None, write_triangle_mesh=None)
    sys.modules["open3d"] = open3d_stub

from config_defaults import REPAIR_MAX_DIAMETER_RATIO, REPAIR_Y_BAND_RATIO
from stage_contact_hole_repair import (
    GroundPlaneProbe,
    GroundPlaneSearchResult,
    GroundPlaneValidation,
    _candidate_from_path,
    _evaluate_loop,
    _extract_boundary_edges,
    _extract_boundary_paths,
    _extract_closed_section_loops,
    _clip_mesh_at_plane,
    _find_ground_plane_shift,
    _find_optimal_clip_offset,
    _normalize_plane,
    _project_vertices_to_plane,
    _probe_ground_plane_shift,
    _resolve_params,
    _validate_ground_plane_output,
    run_contact_hole_repair,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "coffee01"


def _load_fixture_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    faces = np.asarray([list(row) for row in ply["face"].data["vertex_indices"]], dtype=np.int64)
    return vertices, faces


class StageContactHoleRepairDefaultsTests(unittest.TestCase):
    def test_fixture_bottom_loop_is_auto_selected_with_current_defaults(self) -> None:
        vertices, faces = _load_fixture_mesh(FIXTURE_DIR / "object_mesh_wrapped.ply")
        boundary_edges = _extract_boundary_edges(faces)
        raw_paths = _extract_boundary_paths(boundary_edges)
        params = _resolve_params({})
        mesh_center = vertices.mean(axis=0)
        mesh_diag = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-6)
        mesh_min_y = float(vertices[:, 1].min())

        candidate_ids: list[int] = []
        for loop_id, path in enumerate(raw_paths):
            stats = types.SimpleNamespace(
                skipped_small=0,
                skipped_open=0,
                skipped_triangulation=0,
                closed_loops=0,
                skipped_degenerate=0,
            )
            if _candidate_from_path(loop_id, path, vertices, mesh_center, stats) is None:
                continue
            loop_eval = _evaluate_loop(
                loop_id,
                path[:-1],
                vertices,
                mesh_center,
                mesh_diag,
                mesh_min_y,
                params,
            )
            if loop_eval.candidate:
                candidate_ids.append(loop_id)

        self.assertEqual(REPAIR_MAX_DIAMETER_RATIO, 0.46)
        self.assertEqual(candidate_ids, [0])

    def test_upper_loop_is_rejected_even_when_it_is_inside_contact_band(self) -> None:
        params = _resolve_params({})
        bottom_loop = np.asarray(
            [
                [-0.6, 0.02, -0.6],
                [0.6, 0.02, -0.6],
                [0.6, 0.02, 0.6],
                [-0.6, 0.02, 0.6],
            ],
            dtype=np.float64,
        )
        upper_loop = np.asarray(
            [
                [-0.6, 0.12, -0.6],
                [0.6, 0.12, -0.6],
                [0.6, 0.12, 0.6],
                [-0.6, 0.12, 0.6],
            ],
            dtype=np.float64,
        )
        support_vertices = np.asarray(
            [[0.0, 0.0, 0.0]] * 20 + [[1.4, 1.0, -1.4], [-1.4, 1.0, 1.4]],
            dtype=np.float64,
        )
        vertices = np.vstack((bottom_loop, upper_loop, support_vertices))
        mesh_center = vertices.mean(axis=0)
        mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        mesh_min_y = float(vertices[:, 1].min())

        bottom_eval = _evaluate_loop(
            0,
            [0, 1, 2, 3],
            vertices,
            mesh_center,
            mesh_diag,
            mesh_min_y,
            params,
        )
        upper_eval = _evaluate_loop(
            1,
            [4, 5, 6, 7],
            vertices,
            mesh_center,
            mesh_diag,
            mesh_min_y,
            params,
        )

        self.assertEqual(REPAIR_Y_BAND_RATIO, 0.06)
        self.assertTrue(bottom_eval.candidate)
        self.assertEqual(upper_eval.reason, "not_downward_facing")
        self.assertFalse(upper_eval.candidate)


class FindOptimalClipOffsetTests(unittest.TestCase):
    """Tests for _find_optimal_clip_offset."""

    def _make_open_bottom_box(
        self, bottom_y: float = 0.0, top_y: float = 1.0, half: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a box open at the bottom (y=bottom_y).

        The box has 5 faces (top + 4 sides) and the bottom is a hole,
        producing a boundary loop at y=bottom_y.
        """
        # 8 vertices of a box
        vertices = np.array(
            [
                [-half, bottom_y, -half],  # 0  bottom
                [half, bottom_y, -half],   # 1  bottom
                [half, bottom_y, half],    # 2  bottom
                [-half, bottom_y, half],   # 3  bottom
                [-half, top_y, -half],     # 4  top
                [half, top_y, -half],      # 5  top
                [half, top_y, half],       # 6  top
                [-half, top_y, half],      # 7  top
            ],
            dtype=np.float64,
        )
        # Faces: top face + 4 side faces (each side = 2 triangles), no bottom face
        faces = np.array(
            [
                # Top
                [4, 5, 6],
                [4, 6, 7],
                # Front (z = -half)
                [0, 5, 4],
                [0, 1, 5],
                # Right (x = half)
                [1, 6, 5],
                [1, 2, 6],
                # Back (z = half)
                [2, 7, 6],
                [2, 3, 7],
                # Left (x = -half)
                [3, 4, 7],
                [3, 0, 4],
            ],
            dtype=np.int64,
        )
        return vertices, faces

    def test_open_bottom_box_plane_below(self) -> None:
        """Ground plane below the box: offset should push above boundary vertices."""
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        # Ground plane at y=0 (normal pointing up = toward the object)
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0  # plane equation: y + 0 = 0, i.e. y = 0

        offset = _find_optimal_clip_offset(vertices, faces, plane_normal, plane_d)

        # Bottom boundary vertices are at y=0.05, so signed dist = 0.05
        # The offset should be at least 0.05 + epsilon (0.002)
        self.assertGreaterEqual(offset, 0.05 + 0.002 - 1e-9)
        # But not excessively large
        self.assertLess(offset, 0.30)

    def test_watertight_mesh_returns_epsilon(self) -> None:
        """A watertight mesh (no boundary edges) returns epsilon."""
        vertices, faces = self._make_open_bottom_box(bottom_y=0.0, top_y=1.0)
        # Add the bottom face to make it watertight
        bottom_faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
        faces = np.vstack([faces, bottom_faces])

        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0
        offset = _find_optimal_clip_offset(vertices, faces, plane_normal, plane_d)
        self.assertAlmostEqual(offset, 0.002, places=6)

    def test_watertight_mesh_above_plane(self) -> None:
        """A watertight mesh above the plane clips at model bottom, not ground."""
        vertices, faces = self._make_open_bottom_box(bottom_y=1.3, top_y=1.8)
        # Add bottom face to make it watertight
        bottom_faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
        faces = np.vstack([faces, bottom_faces])

        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0
        offset = _find_optimal_clip_offset(vertices, faces, plane_normal, plane_d)

        # dist_min = 1.3, so offset should be 1.3 + 0.002
        self.assertGreaterEqual(offset, 1.3 + 0.002 - 1e-9)

    def test_model_entirely_above_plane(self) -> None:
        """Model sits entirely above the ground plane (all dists positive).

        This is the case where the ground plane normal was flipped because
        all vertices were on one side.  The bottom boundary loop should
        still be detected and the offset should push past it.
        """
        # Box bottom at y=1.3, top at y=1.8, plane at y=0
        vertices, faces = self._make_open_bottom_box(bottom_y=1.3, top_y=1.8)
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0

        offset = _find_optimal_clip_offset(vertices, faces, plane_normal, plane_d)

        # Bottom boundary vertices at y=1.3, signed dist=1.3
        # Offset must be >= 1.3 + epsilon
        self.assertGreaterEqual(offset, 1.3 + 0.002 - 1e-9)
        self.assertLess(offset, 2.0)

    def test_no_bottom_loops_with_two_holes(self) -> None:
        """Only holes near the mesh bottom are considered; upper holes are ignored."""
        # Create a box open at both top and bottom
        half = 0.5
        vertices = np.array(
            [
                [-half, 0.0, -half],   # 0  bottom
                [half, 0.0, -half],    # 1
                [half, 0.0, half],     # 2
                [-half, 0.0, half],    # 3
                [-half, 1.0, -half],   # 4  top
                [half, 1.0, -half],    # 5
                [half, 1.0, half],     # 6
                [-half, 1.0, half],    # 7
            ],
            dtype=np.float64,
        )
        # 4 sides only — both top and bottom are open
        faces = np.array(
            [
                [0, 5, 4], [0, 1, 5],
                [1, 6, 5], [1, 2, 6],
                [2, 7, 6], [2, 3, 7],
                [3, 4, 7], [3, 0, 4],
            ],
            dtype=np.int64,
        )
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0

        offset = _find_optimal_clip_offset(vertices, faces, plane_normal, plane_d)

        # Bottom loop at y=0 (dist=0), top loop at y=1 (dist=1).
        # Only the bottom loop (within 20% of bbox_height=1.0) should be used.
        # Bottom loop max dist = 0, so offset = 0 + epsilon = epsilon.
        self.assertAlmostEqual(offset, 0.002, places=6)


class GroundPlaneShiftSearchTests(unittest.TestCase):
    def _make_open_bottom_box(
        self, bottom_y: float = 0.0, top_y: float = 1.0, half: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.array(
            [
                [-half, bottom_y, -half],
                [half, bottom_y, -half],
                [half, bottom_y, half],
                [-half, bottom_y, half],
                [-half, top_y, -half],
                [half, top_y, -half],
                [half, top_y, half],
                [-half, top_y, half],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [4, 5, 6],
                [4, 6, 7],
                [0, 5, 4],
                [0, 1, 5],
                [1, 6, 5],
                [1, 2, 6],
                [2, 7, 6],
                [2, 3, 7],
                [3, 4, 7],
                [3, 0, 4],
            ],
            dtype=np.int64,
        )
        return vertices, faces

    def _rotate_x(self, vertices: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
        radians = np.deg2rad(degrees)
        c = float(np.cos(radians))
        s = float(np.sin(radians))
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float64,
        )
        return vertices @ rotation.T, rotation

    def test_search_raises_plane_until_bottom_section_becomes_closable(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0

        result = _find_ground_plane_shift(vertices, faces, plane_normal, plane_d)

        self.assertIsNotNone(result.selected_shift)
        assert result.selected_shift is not None
        self.assertGreater(result.selected_shift, 0.0498)
        self.assertLess(result.selected_shift, 0.06)
        probe = _probe_ground_plane_shift(vertices, faces, plane_normal, plane_d, result.selected_shift)
        self.assertTrue(probe.valid)
        self.assertGreater(probe.matched_boundary_area_before, 0.0)
        self.assertEqual(probe.matched_boundary_area_after, 0.0)
        self.assertGreater(probe.cap_added, 0)

    def test_probe_rejects_hole_that_is_only_near_plane(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0

        probe = _probe_ground_plane_shift(vertices, faces, plane_normal, plane_d, 0.0483)

        self.assertFalse(probe.valid)
        self.assertEqual(probe.section_loop_count, 0)
        self.assertEqual(probe.cap_added, 0)
        self.assertEqual(probe.matched_boundary_area_before, 0.0)

    def test_probe_rejects_tiny_section_loop_relative_to_mesh_scale(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0, half=0.01)
        vertices = np.vstack(
            (
                vertices,
                np.array(
                    [
                        [2.0, 0.0, 2.0],
                        [-2.0, 1.0, -2.0],
                    ],
                    dtype=np.float64,
                ),
            )
        )
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 0.0

        probe = _probe_ground_plane_shift(
            vertices,
            faces,
            plane_normal,
            plane_d,
            0.0501,
            min_section_area=0.01,
        )

        self.assertFalse(probe.valid)
        self.assertEqual(probe.section_loop_count, 1)
        self.assertGreater(probe.selected_loop_area, 0.0)
        self.assertEqual(probe.cap_added, 0)

    def test_search_lowers_already_closed_plane_to_lowest_valid_shift(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_height = 0.08
        plane_d = -plane_height

        zero_probe = _probe_ground_plane_shift(vertices, faces, plane_normal, plane_d, 0.0)
        self.assertTrue(zero_probe.valid)

        result = _find_ground_plane_shift(vertices, faces, plane_normal, plane_d)

        self.assertIsNotNone(result.selected_shift)
        assert result.selected_shift is not None
        self.assertLess(result.selected_shift, 0.0)
        self.assertGreater(result.selected_shift, -0.04)
        self.assertLess(result.selected_shift, -0.02)
        probe = _probe_ground_plane_shift(vertices, faces, plane_normal, plane_d, result.selected_shift)
        self.assertTrue(probe.valid)
        self.assertEqual(probe.matched_boundary_area_after, 0.0)
        self.assertGreater(probe.cap_added, 0)

    def test_search_handles_tilted_ground_plane_normal(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        rotated_vertices, rotation = self._rotate_x(vertices, 27.0)
        plane_normal = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        plane_d = 0.0

        result = _find_ground_plane_shift(rotated_vertices, faces, plane_normal, plane_d)

        self.assertIsNotNone(result.selected_shift)
        assert result.selected_shift is not None
        self.assertGreater(result.selected_shift, 0.049)
        self.assertLess(result.selected_shift, 0.055)
        probe = _probe_ground_plane_shift(
            rotated_vertices,
            faces,
            plane_normal,
            plane_d,
            result.selected_shift,
        )
        self.assertTrue(probe.valid)
        self.assertEqual(probe.matched_boundary_area_after, 0.0)

    def test_search_is_invariant_to_plane_normal_scaling(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        unit_result = _find_ground_plane_shift(
            vertices,
            faces,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            0.0,
        )
        scaled_result = _find_ground_plane_shift(
            vertices,
            faces,
            np.array([0.0, 5.0, 0.0], dtype=np.float64),
            0.0,
        )

        self.assertIsNotNone(unit_result.selected_shift)
        self.assertIsNotNone(scaled_result.selected_shift)
        assert unit_result.selected_shift is not None
        assert scaled_result.selected_shift is not None
        self.assertAlmostEqual(unit_result.selected_shift, scaled_result.selected_shift, places=6)

    def test_search_requires_stable_second_valid_probe(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        def _fake_probe(*args, **kwargs):
            shift = float(args[4])
            if 0.4 < shift < 0.6:
                return GroundPlaneProbe(
                    shift=shift,
                    section_loop_count=1,
                    selected_loop_area=1.0,
                    matched_boundary_area_before=1.0,
                    matched_boundary_area_after=0.0,
                    cap_added=2,
                    valid=True,
                )
            if shift > 0.9:
                return GroundPlaneProbe(
                    shift=shift,
                    section_loop_count=0,
                    selected_loop_area=0.0,
                    matched_boundary_area_before=0.0,
                    matched_boundary_area_after=0.0,
                    cap_added=0,
                    valid=False,
                )
            return GroundPlaneProbe(
                shift=shift,
                section_loop_count=0,
                selected_loop_area=0.0,
                matched_boundary_area_before=0.0,
                matched_boundary_area_after=0.0,
                cap_added=0,
                valid=False,
            )

        with mock.patch("stage_contact_hole_repair._probe_ground_plane_shift", side_effect=_fake_probe):
            result = _find_ground_plane_shift(vertices, faces, plane_normal, 0.0)

        self.assertIsNone(result.selected_shift)

    def test_extract_closed_section_loops_prefers_largest_loop(self) -> None:
        small_vertices, small_faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0, half=0.25)
        large_vertices, large_faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0, half=0.60)
        large_vertices = large_vertices + np.array([2.0, 0.0, 0.0], dtype=np.float64)
        large_faces = large_faces + small_vertices.shape[0]

        vertices = np.vstack((small_vertices, large_vertices))
        faces = np.vstack((small_faces, large_faces))
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        shift = 0.0501
        actual_plane_d = -shift

        loops = _extract_closed_section_loops(vertices, faces, plane_normal, actual_plane_d)

        self.assertEqual(len(loops), 2)
        selected_loop = max(loops, key=lambda loop: loop.area)
        self.assertGreater(selected_loop.area, 1.3)
        probe = _probe_ground_plane_shift(vertices, faces, plane_normal, 0.0, shift)
        self.assertTrue(probe.valid)
        self.assertEqual(probe.section_loop_count, 2)
        self.assertAlmostEqual(probe.selected_loop_area, selected_loop.area, places=6)

    def test_search_returns_none_when_only_tiny_section_loops_exist(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0, half=0.01)
        vertices = np.vstack(
            (
                vertices,
                np.array(
                    [
                        [2.0, 0.0, 2.0],
                        [-2.0, 1.0, -2.0],
                    ],
                    dtype=np.float64,
                ),
            )
        )

        result = _find_ground_plane_shift(
            vertices,
            faces,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            0.0,
        )

        self.assertIsNone(result.selected_shift)


class GroundPlaneValidationTests(unittest.TestCase):
    def _make_open_bottom_box(
        self, bottom_y: float = 0.0, top_y: float = 1.0, half: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.array(
            [
                [-half, bottom_y, -half],
                [half, bottom_y, -half],
                [half, bottom_y, half],
                [-half, bottom_y, half],
                [-half, top_y, -half],
                [half, top_y, -half],
                [half, top_y, half],
                [-half, top_y, half],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [4, 5, 6],
                [4, 6, 7],
                [0, 5, 4],
                [0, 1, 5],
                [1, 6, 5],
                [1, 2, 6],
                [2, 7, 6],
                [2, 3, 7],
                [3, 4, 7],
                [3, 0, 4],
            ],
            dtype=np.int64,
        )
        return vertices, faces

    def test_project_vertices_to_plane_snaps_selected_vertices(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.10, 0.0],
                [1.0, 0.20, 0.0],
                [0.0, 0.30, 1.0],
            ],
            dtype=np.float64,
        )
        projected = _project_vertices_to_plane(
            vertices,
            {0, 2},
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            0.0,
        )

        self.assertAlmostEqual(projected[0, 1], 0.0, places=9)
        self.assertAlmostEqual(projected[2, 1], 0.0, places=9)
        self.assertAlmostEqual(projected[1, 1], 0.20, places=9)

    def test_validate_ground_plane_output_detects_open_plane_boundary(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.0, top_y=1.0)

        validation = _validate_ground_plane_output(
            vertices,
            faces,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            0.0,
        )

        self.assertIsInstance(validation, GroundPlaneValidation)
        self.assertFalse(validation.valid)
        self.assertGreater(validation.plane_boundary_edges, 0)
        self.assertTrue(validation.matched_loop_present)

    def test_validate_ground_plane_output_accepts_closed_bottom(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.0, top_y=1.0)
        bottom_faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
        closed_faces = np.vstack([faces, bottom_faces])

        validation = _validate_ground_plane_output(
            vertices,
            closed_faces,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            0.0,
        )

        self.assertTrue(validation.valid)
        self.assertEqual(validation.plane_boundary_edges, 0)
        self.assertFalse(validation.matched_loop_present)

    def test_validate_ground_plane_output_uses_target_loop_matching(self) -> None:
        vertices, faces = self._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        plane_d = -0.10
        clipped_vertices, clipped_faces = _clip_mesh_at_plane(
            vertices,
            faces,
            plane_normal,
            0.0,
            offset=0.10,
        )
        target_loop = max(
            _extract_closed_section_loops(vertices, faces, plane_normal, plane_d),
            key=lambda loop: loop.area,
        )

        validation = _validate_ground_plane_output(
            clipped_vertices,
            clipped_faces,
            plane_normal,
            plane_d,
            target_loop=target_loop,
        )

        self.assertFalse(validation.valid)
        self.assertTrue(validation.matched_loop_present)
        self.assertGreater(validation.matched_loop_area, 0.0)


class GroundPlaneRunTests(unittest.TestCase):
    def test_run_contact_hole_repair_raises_when_no_closed_section_loop_exists(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        ground_plane = {"normal": [0.0, 1.0, 0.0], "d": 0.0}
        search = GroundPlaneSearchResult(
            selected_shift=None,
            scanned_count=3,
            first_valid_shift=None,
            lowest_valid_shift=None,
            selected_loop_area=0.0,
        )

        with mock.patch(
            "stage_contact_hole_repair._load_mesh_arrays",
            return_value=(object(), vertices, faces),
        ), mock.patch(
            "stage_contact_hole_repair._find_ground_plane_shift",
            return_value=search,
        ):
            with self.assertRaisesRegex(ValueError, "could not find a stable closed section loop"):
                run_contact_hole_repair(
                    "dummy_mesh.ply",
                    str(Path("/tmp") / "repair-fallback-test"),
                    ground_plane=ground_plane,
                )

    def test_run_contact_hole_repair_passes_selected_target_loop_to_finalize(self) -> None:
        vertices, faces = GroundPlaneShiftSearchTests()._make_open_bottom_box(bottom_y=0.05, top_y=1.0)
        ground_plane = {"normal": [0.0, 1.0, 0.0], "d": 0.0}
        search = GroundPlaneSearchResult(
            selected_shift=0.0501,
            scanned_count=2,
            first_valid_shift=0.525,
            lowest_valid_shift=0.0501,
            selected_loop_area=0.9998,
        )
        probe = GroundPlaneProbe(
            shift=0.0501,
            section_loop_count=1,
            selected_loop_area=0.9998,
            matched_boundary_area_before=0.9998,
            matched_boundary_area_after=0.0,
            cap_added=2,
            valid=True,
        )
        finalized_validation = GroundPlaneValidation(
            plane_loop_count=0,
            plane_boundary_edges=0,
            matched_loop_present=False,
            matched_loop_area=0.0,
            valid=True,
        )

        with mock.patch(
            "stage_contact_hole_repair._load_mesh_arrays",
            return_value=(object(), vertices, faces),
        ), mock.patch(
            "stage_contact_hole_repair._find_ground_plane_shift",
            return_value=search,
        ), mock.patch(
            "stage_contact_hole_repair._probe_ground_plane_shift",
            return_value=probe,
        ), mock.patch(
            "stage_contact_hole_repair._clip_mesh_at_plane",
            return_value=(vertices, faces),
        ), mock.patch(
            "stage_contact_hole_repair._cap_boundary_at_plane",
            return_value=(vertices, np.vstack((faces, faces[:2])), {0, 1, 2, 3}, 0.9998),
        ), mock.patch(
            "stage_contact_hole_repair._finalize_ground_plane_outputs",
            return_value=(
                vertices,
                faces,
                "flat-cap+preserve-cap",
                finalized_validation,
            ),
        ) as finalize_mock:
            repaired = run_contact_hole_repair(
                "dummy_mesh.ply",
                str(Path("/tmp") / "repair-run-test"),
                ground_plane=ground_plane,
            )

        finalize_kwargs = finalize_mock.call_args.kwargs
        self.assertIn("target_loop", finalize_kwargs)
        self.assertGreater(finalize_kwargs["target_loop"].area, 0.0)
        self.assertEqual(repaired.name, "object_mesh_repaired.ply")
