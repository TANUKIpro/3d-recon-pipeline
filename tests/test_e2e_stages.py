"""End-to-end smoke tests for each pipeline stage.

Every test uses fixture data from ``tests/fixtures/coffee01/`` and verifies
that the stage function runs without error and produces non-empty output files.

All tests require GPU access and are marked accordingly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "coffee01"

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Stage 1: extract_frames
# ---------------------------------------------------------------------------


class TestExtractFramesE2E:
    def test_extracts_frames_from_synthetic_video(
        self, synthetic_video: Path, tmp_path: Path
    ) -> None:
        from scripts.stage_extract_frames import extract_frames

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        frames_path = extract_frames(
            str(synthetic_video),
            str(output_dir),
            frame_interval=1,
            max_frames=5,
        )
        assert frames_path.exists()
        jpgs = list(frames_path.glob("*.jpg"))
        assert len(jpgs) >= 1


# ---------------------------------------------------------------------------
# Stage 2: Pi3X reconstruction
# ---------------------------------------------------------------------------


class TestPi3xReconstructionE2E:
    def test_produces_point_cloud_from_fixture_frames(
        self, tmp_path: Path
    ) -> None:
        from scripts.stage_pi3x_reconstruct import run_pi3x_inference

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        ply_path, poses_path, cache_path = run_pi3x_inference(
            str(FIXTURE_DIR / "frames"),
            str(output_dir),
            pixel_limit=64000,
            max_frames=5,
        )
        assert ply_path.exists()
        assert ply_path.stat().st_size > 0
        assert poses_path.exists()


# ---------------------------------------------------------------------------
# Stage 4: denoise
# ---------------------------------------------------------------------------


class TestDenoiseE2E:
    def test_denoises_fixture_point_cloud(self, tmp_path: Path) -> None:
        from scripts.stage_denoise import denoise

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = denoise(
            str(FIXTURE_DIR / "object_full.ply"),
            str(output_dir),
            preset="balanced",
        )
        assert result.exists()
        assert result.stat().st_size > 0


# ---------------------------------------------------------------------------
# Stage 5: classical_mesh
# ---------------------------------------------------------------------------


class TestClassicalMeshE2E:
    def test_creates_mesh_from_denoised_ply(self, tmp_path: Path) -> None:
        from scripts.stage_classical_mesh import run_classical_mesh

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = run_classical_mesh(
            str(FIXTURE_DIR / "object_denoised.ply"),
            str(output_dir),
        )
        assert result.exists()
        assert result.stat().st_size > 0


# ---------------------------------------------------------------------------
# Stage 6: mesh_wrap
# ---------------------------------------------------------------------------


class TestMeshWrapE2E:
    def test_wraps_fixture_mesh(self, tmp_path: Path) -> None:
        from scripts.stage_mesh_wrap import run_mesh_wrap

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = run_mesh_wrap(
            str(FIXTURE_DIR / "object_mesh.ply"),
            str(output_dir),
        )
        assert result.exists()
        assert result.stat().st_size > 0


# ---------------------------------------------------------------------------
# Stage 7: contact_hole_repair
# ---------------------------------------------------------------------------


class TestContactHoleRepairE2E:
    def test_repairs_wrapped_mesh(self, tmp_path: Path) -> None:
        from scripts.stage_contact_hole_repair import run_contact_hole_repair

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = run_contact_hole_repair(
            str(FIXTURE_DIR / "object_mesh_wrapped.ply"),
            str(output_dir),
        )
        assert result.exists()
        assert result.stat().st_size > 0


# ---------------------------------------------------------------------------
# Stage 8: texture_bake
# ---------------------------------------------------------------------------


class TestTextureBakeE2E:
    def test_bakes_texture_onto_fixture_mesh(self, tmp_path: Path) -> None:
        from scripts.stage_texture_bake import bake_texture

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Copy fixture mesh to output_dir as bake_texture expects it there
        shutil.copy(
            FIXTURE_DIR / "object_mesh_repaired.ply",
            output_dir / "object_mesh_repaired.ply",
        )
        result = bake_texture(
            str(output_dir / "object_mesh_repaired.ply"),
            str(FIXTURE_DIR / "camera_poses.json"),
            str(FIXTURE_DIR / "frames"),
            str(FIXTURE_DIR / "masks"),
            str(output_dir),
        )
        assert result.exists()
        # Check that OBJ and texture PNG were created
        obj_files = list(output_dir.glob("*.obj"))
        assert len(obj_files) >= 1
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) >= 1
