from __future__ import annotations

import sys
import types
import unittest

import numpy as np

sys.modules.setdefault("torch", types.SimpleNamespace())

from scripts.stage_sam2_ui import _compose_final_mask


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


if __name__ == "__main__":
    unittest.main()
