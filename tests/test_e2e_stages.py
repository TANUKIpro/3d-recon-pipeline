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
        from scripts.output_layout import object_mesh_path
        from scripts.stage_texture_bake import bake_texture

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Copy fixture files to output_dir as bake_texture expects them there
        from scripts.output_layout import (
            camera_poses_path,
            frames_dir,
            masks_dir,
        )
        mesh_dst = object_mesh_path(output_dir)
        mesh_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(object_mesh_path(FIXTURE_DIR), mesh_dst)
        result = bake_texture(
            str(mesh_dst),
            str(camera_poses_path(FIXTURE_DIR)),
            str(frames_dir(FIXTURE_DIR)),
            str(masks_dir(FIXTURE_DIR)),
            str(output_dir),
        )
        assert result.exists()
        # Check that OBJ and texture PNG were created under the new phase dirs
        obj_files = list(output_dir.rglob("*.obj"))
        assert len(obj_files) >= 1
        png_files = list(output_dir.rglob("*.png"))
        assert len(png_files) >= 1
