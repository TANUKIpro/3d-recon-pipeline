"""End-to-end smoke tests for each pipeline stage.

Every test uses fixture data from ``tests/fixtures/coffee01/`` and verifies
that the stage function runs without error and produces non-empty output files.

All tests require GPU access and are marked accordingly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.conftest import FIXTURE_DIR

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
# Stage 5: texture_bake
# ---------------------------------------------------------------------------


class TestTextureBakeE2E:
    def test_bakes_texture_onto_fixture_mesh(self, tmp_path: Path) -> None:
        from scripts.stage_texture_bake import bake_texture

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Copy fixture files to output_dir as bake_texture expects them there
        shutil.copy(
            FIXTURE_DIR / "object_mesh.ply",
            output_dir / "object_mesh.ply",
        )
        result = bake_texture(
            str(output_dir / "object_mesh.ply"),
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
