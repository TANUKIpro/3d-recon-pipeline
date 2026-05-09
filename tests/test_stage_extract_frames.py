from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from scripts.output_layout import frame_manifest_path
from scripts.stage_extract_frames import extract_frames


class TestExtractFramesManifest(unittest.TestCase):
    def test_extract_frames_writes_source_frame_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "input.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (8, 8),
            )
            for idx in range(6):
                frame = np.full((8, 8, 3), idx * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            extract_frames(
                str(video_path),
                str(root / "out"),
                frame_interval=2,
                max_frames=3,
            )

            manifest = json.loads(
                frame_manifest_path(root / "out").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["video_path"], str(video_path))
            self.assertEqual(manifest["frame_interval"], 2)
            self.assertEqual(manifest["max_frames"], 3)
            self.assertEqual(manifest["saved_frame_count"], 3)
            self.assertEqual(
                [item["source_frame_index"] for item in manifest["saved_frames"]],
                [0, 2, 4],
            )


if __name__ == "__main__":
    unittest.main()
