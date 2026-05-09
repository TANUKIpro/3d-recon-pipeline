from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


class _NoopInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            synchronize=lambda: None,
        ),
        inference_mode=lambda: _NoopInferenceMode(),
    ),
)

from scripts.output_layout import sam2_clicks_path
from scripts.stage_sam2_ui import (
    _compose_final_mask,
    _load_click_manifest,
    _restore_click_points,
    _save_click_points,
)


class TestComposeFinalMask(unittest.TestCase):
    def test_ground_mask_is_subtracted_from_object_mask(self) -> None:
        object_mask = np.array(
            [[True, True, False], [False, True, True]],
            dtype=bool,
        )
        ground_mask = np.array(
            [[False, True, False], [False, False, True]],
            dtype=bool,
        )

        final_mask = _compose_final_mask(object_mask, ground_mask)

        expected = np.array(
            [[True, False, False], [False, True, False]],
            dtype=bool,
        )
        self.assertTrue(np.array_equal(final_mask, expected))

    def test_missing_ground_mask_keeps_object_mask(self) -> None:
        object_mask = np.array([[True, False], [True, True]], dtype=bool)

        final_mask = _compose_final_mask(object_mask, None)

        self.assertTrue(np.array_equal(final_mask, object_mask))


class TestSam2ClickPersistence(unittest.TestCase):
    def _fake_session(self, output_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(
            output_dir=output_dir,
            frames_dir=output_dir / "p1_frames",
            model_type="small",
            frame_files=[Path("00000.jpg"), Path("00001.jpg")],
            img_w=640,
            img_h=480,
            click_points=[(0.25, 0.5), (0.75, 0.8)],
            click_labels=[1, 0],
            ground_click_points=[(0.1, 0.2)],
            ground_click_labels=[1],
        )

    def test_save_click_points_writes_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            session = self._fake_session(out)

            path = _save_click_points(session)

            self.assertEqual(path, sam2_clicks_path(out))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["model_type"], "small")
            self.assertEqual(data["frame_count"], 2)
            self.assertEqual(data["image_width"], 640)
            self.assertEqual(
                data["object"]["click_points"],
                [[0.25, 0.5], [0.75, 0.8]],
            )
            self.assertEqual(data["object"]["click_labels"], [1, 0])
            self.assertEqual(data["ground"]["click_points"], [[0.1, 0.2]])
            self.assertEqual(data["ground"]["click_labels"], [1])

    def test_restore_click_points_loads_object_and_ground(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            original = self._fake_session(out)
            _save_click_points(original)

            manifest = _load_click_manifest(out)
            restored = self._fake_session(out)
            restored.click_points = []
            restored.click_labels = []
            restored.ground_click_points = []
            restored.ground_click_labels = []
            _restore_click_points(restored, manifest)

            self.assertEqual(restored.click_points, original.click_points)
            self.assertEqual(restored.click_labels, original.click_labels)
            self.assertEqual(
                restored.ground_click_points,
                original.ground_click_points,
            )
            self.assertEqual(
                restored.ground_click_labels,
                original.ground_click_labels,
            )


if __name__ == "__main__":
    unittest.main()
