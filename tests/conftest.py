"""Common pytest fixtures for clip2mesh tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "coffee01"


@pytest.fixture
def fixture_dir() -> Path:
    """Return the path to the shared test fixture directory."""
    return FIXTURE_DIR


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a small test video (5 frames, 64x64) and return its path."""
    import cv2
    import numpy as np

    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (64, 64))
    for i in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Different color per frame for variety
        frame[:, :, i % 3] = 200
        writer.write(frame)
    writer.release()
    return video_path
