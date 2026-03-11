from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

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
    _candidate_from_path,
    _evaluate_loop,
    _extract_boundary_edges,
    _extract_boundary_paths,
    _resolve_params,
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
