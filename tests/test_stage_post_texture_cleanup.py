"""Tests for the multi-pass post-texture cleanup functions."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from scripts.stage_post_texture_contact_cleanup import (
        _COMPONENT_MIN_FACE_RATIO,
        _CONVERGENCE_MAX_ITERATIONS,
        _FLIPPED_NORMAL_NEAR_GROUND_BBOX_RATIO,
        _MASK_ABOVE_PLANE_REMOVAL_THRESHOLD,
        _MASK_LOWER_HALF_REMOVAL_THRESHOLD,
        _NEIGHBOR_REMOVAL_RATIO_THRESHOLD,
        _aggregate_pass_stats,
        _filter_small_components,
        _pass_above_plane_noise,
        _pass_flipped_ground_normals,
        _pass_floating_island_removal,
        _pass_geometric_clip,
        _pass_lower_half_mask_noise,
        _pass_mask_boundary_refinement,
        _pass_neighbor_erosion,
    )
    from scripts.texture.conflict_region import _build_face_adjacency

    _IMPORTABLE = True
except ImportError:
    _IMPORTABLE = False


def _make_test_mesh():
    """Build a small test mesh: two quads (4 triangles) sharing an edge + 1 isolated triangle.

    Layout (top view, Y-up):

        v0---v1---v4      v6
        |  / |  / |      / |
        | /  | /  |     /  |
        v2---v3---v5   v7--v8

    Faces (0-indexed):
      Quad A: f0=(v0,v2,v1), f1=(v1,v2,v3)
      Quad B: f2=(v1,v3,v4), f3=(v4,v3,v5)  — shares edge v1-v3 with quad A
      Isolated: f4=(v6,v7,v8)
    """
    vertices = np.array([
        [0.0, 1.0, 0.0],   # v0 — above plane
        [1.0, 1.0, 0.0],   # v1 — above plane
        [0.0, 0.0, 0.0],   # v2 — on plane
        [1.0, 0.0, 0.0],   # v3 — on plane
        [2.0, 1.0, 0.0],   # v4 — above plane
        [2.0, 0.0, 0.0],   # v5 — on plane
        [5.0, 2.0, 0.0],   # v6 — above plane (isolated)
        [5.0, 1.0, 0.0],   # v7 — above plane (isolated)
        [6.0, 1.0, 0.0],   # v8 — above plane (isolated)
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1],  # f0
        [1, 2, 3],  # f1
        [1, 3, 4],  # f2
        [4, 3, 5],  # f3
        [6, 7, 8],  # f4 — isolated
    ], dtype=np.int64)
    adjacency = _build_face_adjacency(faces)
    return vertices, faces, adjacency


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestFilterSmallComponentsWithAdjacency(unittest.TestCase):
    """Step 1: _filter_small_components with pre-built adjacency."""

    def test_same_result_with_and_without_adjacency(self):
        _, faces, adjacency = _make_test_mesh()

        mask_a = np.ones(len(faces), dtype=bool)
        mask_b = mask_a.copy()

        stats_a = _filter_small_components(mask_a, faces)
        stats_b = _filter_small_components(mask_b, faces, adjacency=adjacency)

        np.testing.assert_array_equal(mask_a, mask_b)
        assert stats_a["removed_faces"] == stats_b["removed_faces"]

    def test_isolated_triangle_removed_with_higher_ratio(self):
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        # min_faces = max(1, int(0.5 * 5)) = 2, so isolated 1-face component is removed
        stats = _filter_small_components(mask, faces, min_face_ratio=0.5, adjacency=adjacency)
        assert not mask[4]  # isolated triangle removed
        assert stats["removed_faces"] == 1


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestPassGeometricClip(unittest.TestCase):
    """Step 2: _pass_geometric_clip removes below-plane faces."""

    def test_removes_below_plane(self):
        vertices, faces, adjacency = _make_test_mesh()
        # Plane: y = 0.5 → plane_normal = [0, 1, 0], plane_d = -0.5
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = -0.5
        orig_face_dist = vertices[faces] @ plane_normal + plane_d

        mask = np.ones(len(faces), dtype=bool)
        result = _pass_geometric_clip(mask, orig_face_dist, adjacency, faces)

        # Faces with any vertex at y=0 have dist = -0.5 for those vertices
        assert result["geometric_removed"] > 0
        assert "component_stats" in result

    def test_all_above_plane_no_removal(self):
        vertices, faces, adjacency = _make_test_mesh()
        # Plane well below all vertices: y + 1.0 > 0 for all y >= 0
        plane_normal = np.array([0.0, 1.0, 0.0])
        plane_d = 1.0
        orig_face_dist = vertices[faces] @ plane_normal + plane_d

        mask = np.ones(len(faces), dtype=bool)
        result = _pass_geometric_clip(mask, orig_face_dist, adjacency, faces)
        assert result["geometric_removed"] == 0
        assert mask.all()


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestPassMaskBoundaryRefinement(unittest.TestCase):
    """Step 3: _pass_mask_boundary_refinement."""

    def test_preserves_low_score_faces(self):
        _, faces, adjacency = _make_test_mesh()
        mask = np.array([True, False, True, True, True], dtype=bool)
        outside = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        ground = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        near_plane = np.array([True, True, True, True, True])

        result = _pass_mask_boundary_refinement(
            mask, outside, ground, near_plane, adjacency, faces,
        )
        assert result["mask_preserved_count"] >= 1
        assert mask[1]  # f1 should be restored

    def test_removes_high_score_faces(self):
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.array([0.1, 0.8, 0.1, 0.1, 0.1])
        ground = np.array([0.1, 0.8, 0.1, 0.1, 0.1])
        near_plane = np.array([True, True, True, True, True])

        result = _pass_mask_boundary_refinement(
            mask, outside, ground, near_plane, adjacency, faces,
        )
        assert result["mask_removed_count"] >= 1
        assert not mask[1]  # f1 should be removed

    def test_component_filter_runs(self):
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.zeros(len(faces))
        ground = np.zeros(len(faces))
        near_plane = np.zeros(len(faces), dtype=bool)

        result = _pass_mask_boundary_refinement(
            mask, outside, ground, near_plane, adjacency, faces,
        )
        assert "component_stats" in result
        assert result["component_stats"]["applied"]


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestPassAbovePlaneNoise(unittest.TestCase):
    """Step 4: _pass_above_plane_noise."""

    def test_removes_above_plane_high_outside(self):
        vertices, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
        near_plane = np.array([True, True, True, True, False])
        min_face_dist = np.array([0.5, 0.5, 0.5, 0.5, 1.5])

        result = _pass_above_plane_noise(
            mask, outside, near_plane, min_face_dist, adjacency, faces,
        )
        assert result["above_plane_removed"] >= 1
        assert not mask[4]  # isolated high-outside face removed

    def test_keeps_near_plane_faces(self):
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.ones(len(faces)) * 0.9
        near_plane = np.ones(len(faces), dtype=bool)
        min_face_dist = np.ones(len(faces)) * 0.5

        result = _pass_above_plane_noise(
            mask, outside, near_plane, min_face_dist, adjacency, faces,
        )
        assert result["above_plane_removed"] == 0


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestPassLowerHalfMaskNoise(unittest.TestCase):
    """Pass 3b: _pass_lower_half_mask_noise."""

    def test_removes_lower_half_high_outside(self):
        """Faces in the lower half with outside_ratio > 0 are removed."""
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        # centroid_heights: faces 0-3 in lower half, face 4 in upper half
        centroid_heights = np.array([0.1, 0.1, 0.1, 0.1, 2.0])
        half_height = 0.5

        result = _pass_lower_half_mask_noise(
            mask, outside, centroid_heights, half_height, adjacency, faces,
        )
        assert result["applied"] is True
        assert result["lower_half_removed"] == 4  # faces 0-3 removed
        assert not mask[0] and not mask[1] and not mask[2] and not mask[3]

    def test_keeps_upper_half_high_outside(self):
        """Faces in the upper half are not removed even with high outside_ratio."""
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        centroid_heights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        half_height = 0.5

        result = _pass_lower_half_mask_noise(
            mask, outside, centroid_heights, half_height, adjacency, faces,
        )
        assert result["lower_half_removed"] == 0

    def test_keeps_lower_half_zero_outside(self):
        """Faces with outside_ratio == 0 (fully inside mask) are kept."""
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        centroid_heights = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        half_height = 0.5

        result = _pass_lower_half_mask_noise(
            mask, outside, centroid_heights, half_height, adjacency, faces,
        )
        assert result["lower_half_removed"] == 0
        assert mask.all()

    def test_component_filter_runs(self):
        """Component filter executes after lower-half removal."""
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)
        outside = np.array([0.5, 0.5, 0.5, 0.0, 0.0])
        centroid_heights = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        half_height = 0.5

        result = _pass_lower_half_mask_noise(
            mask, outside, centroid_heights, half_height, adjacency, faces,
        )
        assert "component_stats" in result


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestPassNeighborErosion(unittest.TestCase):
    """Step 5: _pass_neighbor_erosion."""

    def test_removes_peninsula(self):
        """A face whose majority of neighbors are removed should be eroded."""
        _, faces, adjacency = _make_test_mesh()
        mask = np.array([False, True, False, False, False], dtype=bool)

        result = _pass_neighbor_erosion(
            mask, adjacency, faces, removal_ratio_threshold=0.5,
            min_face_ratio=0.0,
        )
        assert result["total_eroded"] >= 1
        assert not mask[1]

    def test_convergence_stops_early(self):
        """When no faces are eroded, iteration stops."""
        _, faces, adjacency = _make_test_mesh()
        mask = np.ones(len(faces), dtype=bool)

        result = _pass_neighbor_erosion(mask, adjacency, faces)
        assert result["total_eroded"] == 0
        assert result["iterations"] == 1

    def test_max_iterations_respected(self):
        """Erosion should not exceed max_iterations."""
        n = 10
        verts_list = [[float(i), 0.0, 0.0] for i in range(n + 2)]
        vertices = np.array(verts_list, dtype=np.float64)
        faces_list = [[i, i + 1, i + 2] for i in range(n)]
        faces = np.array(faces_list, dtype=np.int64)
        adjacency = _build_face_adjacency(faces)

        mask = np.ones(len(faces), dtype=bool)
        mask[0] = False

        result = _pass_neighbor_erosion(
            mask, adjacency, faces,
            removal_ratio_threshold=0.4,
            max_iterations=2,
            min_face_ratio=0.0,
        )
        assert result["iterations"] <= 2

    def test_snapshot_based_erosion_is_order_independent(self):
        """Erosion result should not depend on face index ordering."""
        # Build a symmetric cross: center face with 3 neighbors, 2 removed
        # The snapshot approach ensures all faces are evaluated against the
        # same state, so the result is deterministic regardless of index order.
        n = 10
        verts = np.array([[float(i), 0.0, 0.0] for i in range(n + 2)], dtype=np.float64)
        fwd = np.array([[i, i + 1, i + 2] for i in range(n)], dtype=np.int64)
        adj_fwd = _build_face_adjacency(fwd)
        mask_fwd = np.ones(n, dtype=bool)
        mask_fwd[0] = False
        mask_fwd[n - 1] = False

        # Reverse face order
        rev = fwd[::-1].copy()
        adj_rev = _build_face_adjacency(rev)
        mask_rev = np.ones(n, dtype=bool)
        mask_rev[0] = False
        mask_rev[n - 1] = False

        r_fwd = _pass_neighbor_erosion(
            mask_fwd, adj_fwd, fwd,
            removal_ratio_threshold=0.4, max_iterations=3, min_face_ratio=0.0,
        )
        r_rev = _pass_neighbor_erosion(
            mask_rev, adj_rev, rev,
            removal_ratio_threshold=0.4, max_iterations=3, min_face_ratio=0.0,
        )
        # Same total eroded regardless of face ordering
        assert r_fwd["total_eroded"] == r_rev["total_eroded"]


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestAggregatePassStats(unittest.TestCase):
    """Step 6: _aggregate_pass_stats produces correct structure."""

    def test_structure_matches_proposal_payload(self):
        pass1 = {"geometric_removed": 5, "component_stats": {"removed_faces": 2, "applied": True}}
        pass2 = {
            "applied": True,
            "mask_removed_count": 3,
            "mask_preserved_count": 1,
            "near_plane_faces": 10,
            "component_stats": {"removed_faces": 1, "applied": True},
        }
        pass3 = {
            "applied": True,
            "above_plane_removed": 4,
            "threshold": 0.7,
            "component_stats": {"removed_faces": 0, "applied": True},
        }
        pass3b = {
            "applied": True,
            "lower_half_removed": 6,
            "threshold": 0.0,
            "half_height": 0.5,
            "component_stats": {"removed_faces": 1, "applied": True},
        }
        pass4 = {
            "total_eroded": 2,
            "iterations": 2,
            "component_stats": {"removed_faces": 1, "applied": True},
        }
        pass5 = {"removed_islands": 1, "removed_faces": 50}
        outside = np.array([0.5, 0.8, 0.2, 0.9, 0.1])
        ground = np.array([0.3, 0.6, 0.1, 0.7, 0.05])
        keep = np.array([True, False, True, False, True])

        mf_stats, comp_stats = _aggregate_pass_stats(
            pass1, pass2, pass3, pass3b, pass4, pass5, outside, ground, keep,
        )

        # Keys expected by _proposal_payload
        assert mf_stats["applied"] is True
        assert "above_plane_filtering" in mf_stats
        assert mf_stats["above_plane_filtering"]["removed_count"] == 4
        assert "lower_half_mask_noise" in mf_stats
        assert mf_stats["lower_half_mask_noise"]["removed_count"] == 6
        assert "component_filtering" in mf_stats
        assert mf_stats["component_filtering"]["removed_faces"] == 2 + 1 + 0 + 1 + 1 + 50
        assert mf_stats["component_filtering"]["removed_components"] is not None
        assert "neighbor_erosion" in mf_stats
        assert mf_stats["neighbor_erosion"]["total_eroded"] == 2
        assert "floating_island_removal" in mf_stats
        assert mf_stats["floating_island_removal"]["removed_islands"] == 1
        assert comp_stats["removed_faces"] == 2 + 1 + 0 + 1 + 1 + 50
        assert "total_components" in comp_stats
        assert "largest_component_faces" in comp_stats

    def test_no_mask_passes_returns_not_applied(self):
        pass1 = {"geometric_removed": 5, "component_stats": {"removed_faces": 0}}
        pass2 = {"applied": False}
        pass3 = {"applied": False}
        pass3b = {"applied": False}
        pass4 = {"total_eroded": 0, "iterations": 1, "component_stats": {"removed_faces": 0}}
        pass5 = {"removed_islands": 0, "removed_faces": 0}
        keep = np.ones(5, dtype=bool)

        mf_stats, _ = _aggregate_pass_stats(
            pass1, pass2, pass3, pass3b, pass4, pass5, None, None, keep,
        )
        assert mf_stats["applied"] is False

    def test_floating_islands_propagate_even_without_mask_passes(self):
        """If only pass5 removes faces, stats should still reflect applied=True."""
        pass1 = {"geometric_removed": 0, "component_stats": {"removed_faces": 0}}
        pass2 = {"applied": False}
        pass3 = {"applied": False}
        pass3b = {"applied": False}
        pass4 = {"total_eroded": 0, "iterations": 1, "component_stats": {"removed_faces": 0}}
        pass5 = {"removed_islands": 2, "removed_faces": 100}
        keep = np.ones(5, dtype=bool)

        mf_stats, comp_stats = _aggregate_pass_stats(
            pass1, pass2, pass3, pass3b, pass4, pass5, None, None, keep,
        )
        assert mf_stats["applied"] is True
        assert mf_stats["floating_island_removal"]["removed_islands"] == 2
        assert comp_stats["removed_faces"] == 100


@unittest.skipUnless(_IMPORTABLE, "required modules not available")
class TestPassFloatingIslandRemoval(unittest.TestCase):
    """Pass 5: _pass_floating_island_removal."""

    def _make_island_mesh(self):
        """Main body (4 tris) + distant island (1 tri) + nearby small component (1 tri).

            Main body (y=0..1):         Island (y=10):      Near (y=0.5):
            v0--v1    v1--v4            v9--v10             v12--v13
            | / |     | / |            / |                  / |
            v2--v3    v3--v5          v11                  v14
        """
        vertices = np.array([
            # Main body
            [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0], [2, 1, 0], [2, 0, 0],
            # Distant island (at y=10, far from main body)
            [0, 10, 0], [1, 10, 0], [0, 9, 0],
            # Near small component (at y=0.5, within main body bbox)
            [3, 0.5, 0], [4, 0.5, 0], [3, 0, 0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 2, 1], [1, 2, 3], [1, 3, 4], [4, 3, 5],  # main body (4 tris)
            [6, 8, 7],                                       # distant island (1 tri)
            [9, 11, 10],                                     # near small (1 tri)
        ], dtype=np.int64)
        adjacency = _build_face_adjacency(faces)
        return vertices, faces, adjacency

    def test_removes_distant_island(self):
        verts, faces, adj = self._make_island_mesh()
        mask = np.ones(len(faces), dtype=bool)

        result = _pass_floating_island_removal(
            mask, verts, faces, adj,
            island_face_ratio=0.5,  # require 50% of main body → both small comps fail size
            gap_ratio=0.5,
        )
        # Distant island (f4) at y=9-10 should be removed — far from main bbox [0,2]x[0,1]
        assert not mask[4]
        assert result["removed_islands"] >= 1
        assert result["removed_faces"] >= 1

    def test_removes_small_component_inside_bbox(self):
        """Small components inside the main body's bbox are removed by size criterion."""
        verts, faces, adj = self._make_island_mesh()
        mask = np.ones(len(faces), dtype=bool)

        result = _pass_floating_island_removal(
            mask, verts, faces, adj,
            island_face_ratio=0.5,  # 50% of 4 = 2 → 1-face components removed
            gap_ratio=1.0,          # very large gap threshold → spatial criterion won't fire
        )
        # Near component (f5) is inside bbox but has only 1 face (< 2)
        assert not mask[5]
        assert result["removed_islands"] >= 1

    def test_keeps_large_component(self):
        """Components large enough relative to main body are kept regardless of distance."""
        verts, faces, adj = self._make_island_mesh()
        mask = np.ones(len(faces), dtype=bool)

        result = _pass_floating_island_removal(
            mask, verts, faces, adj,
            island_face_ratio=0.01,  # very low → all comps are "large enough"
            gap_ratio=100.0,          # very large → spatial criterion won't fire
        )
        assert mask.all()
        assert result["removed_islands"] == 0

    def test_removed_faces_not_in_split_faces(self):
        """Faces removed by pass 2-5 must NOT appear in split_faces.

        Regression test for a bug where the exclusion set for split_faces was
        built from *kept* merged faces only (``merged_faces[keep_merged_mask]``).
        This caused faces removed by above-plane noise / floating-island passes
        to be re-added as cap faces in the final OBJ.

        The fix uses **all** merged faces as the exclusion set so that
        ``split_faces`` contains only genuinely new faces produced by plane
        clipping (those with interpolated vertex indices).
        """
        # Build a minimal mesh:
        #   - 2 main-body triangles (faces 0-1)
        #   - 1 floating-island triangle (face 2) above the ground plane
        merged_faces = np.array([
            [0, 1, 2],  # main body
            [1, 2, 3],  # main body
            [4, 5, 6],  # floating island (above ground, removed by pass 4/5)
        ], dtype=np.int64)

        # Simulate keep_merged_mask: island face is removed
        keep_merged_mask = np.array([True, True, False], dtype=bool)

        # clipped_faces == merged_faces (all above ground, nothing clipped)
        clipped_faces = merged_faces.copy()

        # --- Fixed logic (using all merged faces) ---
        all_merged_face_keys = {
            tuple(sorted((int(f[0]), int(f[1]), int(f[2]))))
            for f in merged_faces
        }
        split_faces = np.asarray(
            [
                f for f in clipped_faces
                if tuple(sorted((int(f[0]), int(f[1]), int(f[2])))) not in all_merged_face_keys
            ],
            dtype=np.int64,
        )
        if split_faces.size == 0:
            split_faces = np.zeros((0, 3), dtype=np.int64)

        # The floating island face must NOT appear in split_faces
        self.assertEqual(split_faces.shape[0], 0,
                         "Removed island face leaked into split_faces")

        # --- Old (buggy) logic for comparison: would produce 1 leaked face ---
        kept_face_keys = {
            tuple(sorted((int(f[0]), int(f[1]), int(f[2]))))
            for f in merged_faces[keep_merged_mask]
        }
        buggy_split = [
            f for f in clipped_faces
            if tuple(sorted((int(f[0]), int(f[1]), int(f[2])))) not in kept_face_keys
        ]
        self.assertEqual(len(buggy_split), 1,
                         "Sanity check: old logic should leak 1 face")

    def test_split_faces_keeps_interpolated_boundary(self):
        """Genuine boundary faces (with interpolated vertices) survive the fix.

        When clipping splits a face, the new sub-triangle references at least
        one interpolated vertex whose index is >= len(merged_vertices), so its
        sorted key will never match any merged face.  Split faces from kept
        source faces pass through; those from removed sources are excluded.
        """
        merged_faces = np.array([
            [0, 1, 2],
            [1, 2, 3],
        ], dtype=np.int64)

        # Clipping produced: face 0 kept as-is, face 1 kept as-is,
        # plus a boundary fragment from source face 1 (interpolated vertex 10)
        clipped_faces = np.array([
            [0, 1, 2],
            [1, 2, 3],
            [1, 10, 2],  # boundary fragment from face 1
        ], dtype=np.int64)
        clipped_source = np.array([0, 1, 1], dtype=np.int64)

        keep_merged_mask = np.array([True, True], dtype=bool)

        all_merged_face_keys = {
            tuple(sorted((int(f[0]), int(f[1]), int(f[2]))))
            for f in merged_faces
        }
        split_faces = np.asarray(
            [
                f for cf_idx, f in enumerate(clipped_faces)
                if tuple(sorted((int(f[0]), int(f[1]), int(f[2])))) not in all_merged_face_keys
                and keep_merged_mask[clipped_source[cf_idx]]
            ],
            dtype=np.int64,
        )
        self.assertEqual(split_faces.shape[0], 1)
        np.testing.assert_array_equal(split_faces[0], [1, 10, 2])

        # Now mark source face 1 as removed — the split fragment should be excluded
        keep_merged_mask[1] = False
        split_faces_after = np.asarray(
            [
                f for cf_idx, f in enumerate(clipped_faces)
                if tuple(sorted((int(f[0]), int(f[1]), int(f[2])))) not in all_merged_face_keys
                and keep_merged_mask[clipped_source[cf_idx]]
            ],
            dtype=np.int64,
        )
        if split_faces_after.size == 0:
            split_faces_after = np.zeros((0, 3), dtype=np.int64)
        self.assertEqual(split_faces_after.shape[0], 0)

    def test_split_faces_from_removed_source_excluded(self):
        """Split fragments from a removed merged face must not appear in output.

        A merged face straddling the ground plane produces split fragments with
        interpolated vertices.  When that source face is removed by cleanup
        (keep_merged_mask=False), the fragments must be filtered out.
        """
        # 3 merged faces: face 0 is fully above, face 1 straddles (removed),
        # face 2 is fully above.
        merged_faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ], dtype=np.int64)

        # After clipping: face 0 kept, face 1 split into 2 fragments, face 2 kept
        clipped_faces = np.array([
            [0, 1, 2],       # from face 0
            [6, 7, 8],       # from face 2
            [3, 4, 20],      # split from face 1 (vertex 20 interpolated)
            [4, 21, 20],     # split from face 1 (vertex 21 interpolated)
        ], dtype=np.int64)
        clipped_source = np.array([0, 2, 1, 1], dtype=np.int64)

        # Face 1 was removed by cleanup
        keep_merged_mask = np.array([True, False, True], dtype=bool)

        all_merged_face_keys = {
            tuple(sorted((int(f[0]), int(f[1]), int(f[2]))))
            for f in merged_faces
        }
        split_faces = np.asarray(
            [
                f for cf_idx, f in enumerate(clipped_faces)
                if tuple(sorted((int(f[0]), int(f[1]), int(f[2])))) not in all_merged_face_keys
                and keep_merged_mask[clipped_source[cf_idx]]
            ],
            dtype=np.int64,
        )
        if split_faces.size == 0:
            split_faces = np.zeros((0, 3), dtype=np.int64)
        # Both split fragments from face 1 should be excluded
        self.assertEqual(split_faces.shape[0], 0)

    def test_single_component_noop(self):
        """Mesh with only one component should be a no-op."""
        _, faces, adjacency = _make_test_mesh()
        verts = np.array([
            [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0],
            [2, 1, 0], [2, 0, 0],
            [5, 2, 0], [5, 1, 0], [6, 1, 0],
        ], dtype=np.float64)
        # Use only the connected faces (f0-f3), skip isolated f4
        connected_faces = faces[:4]
        connected_adj = _build_face_adjacency(connected_faces)
        mask = np.ones(len(connected_faces), dtype=bool)

        result = _pass_floating_island_removal(
            mask, verts, connected_faces, connected_adj,
        )
        assert mask.all()
        assert result["removed_islands"] == 0


@unittest.skipUnless(_IMPORTABLE, "cleanup module not importable")
class TestPassFlippedGroundNormals(unittest.TestCase):
    """Tests for Pass 6: flipped normals at ground plane."""

    def test_removes_downward_facing_near_ground(self):
        """Face near the ground with a downward normal should be removed."""
        # Ground plane: y-up, at y=0
        plane_normal = np.array([0.0, 1.0, 0.0])
        # Two triangles: face 0 is upward-facing near ground, face 1 is
        # downward-facing near ground (flipped winding).
        verts = np.array([
            [0, 0.01, 0],   # v0 — just above ground
            [1, 0.01, 0],   # v1
            [0.5, 0.01, 1], # v2
            [0, 0.01, 2],   # v3
            [1, 0.01, 2],   # v4
            [0.5, 0.01, 3], # v5
        ], dtype=np.float64)
        faces = np.array([
            [0, 2, 1],  # normal = +y (upward)
            [3, 4, 5],  # normal = -y (downward, flipped winding)
        ], dtype=np.int64)

        plane_d = 0.0
        orig_face_dist = verts[faces] @ plane_normal + plane_d
        adjacency = _build_face_adjacency(faces)
        mask = np.ones(len(faces), dtype=bool)

        result = _pass_flipped_ground_normals(
            mask, verts, faces, plane_normal, orig_face_dist, adjacency,
            near_ground_bbox_ratio=1.0,  # generous threshold for test
        )
        self.assertTrue(mask[0], "Upward-facing face should be kept")
        self.assertFalse(mask[1], "Downward-facing face should be removed")
        self.assertEqual(result["flipped_removed"], 1)

    def test_keeps_downward_facing_far_from_ground(self):
        """Downward-facing face far from ground should NOT be removed."""
        plane_normal = np.array([0.0, 1.0, 0.0])
        verts = np.array([
            [0, 5.0, 0],
            [1, 5.0, 0],
            [0.5, 5.0, 1],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)  # downward normal at y=5

        plane_d = 0.0
        orig_face_dist = verts[faces] @ plane_normal + plane_d
        adjacency = _build_face_adjacency(faces)
        mask = np.ones(len(faces), dtype=bool)

        result = _pass_flipped_ground_normals(
            mask, verts, faces, plane_normal, orig_face_dist, adjacency,
            near_ground_bbox_ratio=0.01,
        )
        self.assertTrue(mask[0], "Far-from-ground face should be kept")
        self.assertEqual(result["flipped_removed"], 0)

    def test_split_faces_flipped_normal_filtered(self):
        """Split face fragments with flipped normals are removed."""
        # Ground plane: y-up at y=0
        plane_normal = np.array([0.0, 1.0, 0.0])

        # Simulate split faces — one upward, one downward
        verts = np.array([
            [0, 0.01, 0],
            [1, 0.01, 0],
            [0.5, 0.01, 1],
            [0, 0.01, 2],
            [1, 0.01, 2],
            [0.5, 0.01, 3],
        ], dtype=np.float64)
        split_faces = np.array([
            [0, 2, 1],  # upward normal (+y)
            [3, 4, 5],  # downward normal (-y, flipped)
        ], dtype=np.int64)

        sv0 = verts[split_faces[:, 0]]
        sv1 = verts[split_faces[:, 1]]
        sv2 = verts[split_faces[:, 2]]
        split_dot = np.cross(sv1 - sv0, sv2 - sv0) @ plane_normal
        filtered = split_faces[split_dot >= 0.0]

        self.assertEqual(filtered.shape[0], 1)
        np.testing.assert_array_equal(filtered[0], [0, 2, 1])
