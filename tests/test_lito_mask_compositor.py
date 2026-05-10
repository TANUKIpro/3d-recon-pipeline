"""Tests for scripts.lito.mask_compositor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.lito.mask_compositor import compose_lito_input


def _write_inputs(
    tmp_path: Path,
    *,
    rgb_size=(256, 384),
    mask_box=(80, 100, 200, 280),
) -> tuple[str, str]:
    h, w = rgb_size
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = mask_box
    mask[y0:y1, x0:x1] = 255
    fp = tmp_path / "frame.jpg"
    mp = tmp_path / "mask.png"
    Image.fromarray(rgb).save(fp, quality=95)
    Image.fromarray(mask).save(mp)
    return str(fp), str(mp)


class TestComposeLitoInput:
    def test_writes_square_rgba_at_resolution(self, tmp_path: Path):
        fp, mp = _write_inputs(tmp_path)
        out = tmp_path / "out.png"
        meta = compose_lito_input(fp, mp, str(out), resolution=518)
        assert out.exists()
        img = np.array(Image.open(out))
        assert img.shape == (518, 518, 4)
        assert meta["resolution"] == 518
        assert meta["src_width"] == 384
        assert meta["src_height"] == 256

    def test_alpha_carries_mask(self, tmp_path: Path):
        fp, mp = _write_inputs(tmp_path)
        out = tmp_path / "out.png"
        compose_lito_input(fp, mp, str(out), resolution=128)
        img = np.array(Image.open(out))
        alpha = img[..., 3]
        # Alpha must contain both opaque (foreground) and transparent (background) pixels.
        assert int((alpha == 0).sum()) > 0
        assert int((alpha > 200).sum()) > 0

    def test_lower_resolution_supported(self, tmp_path: Path):
        fp, mp = _write_inputs(tmp_path)
        out = tmp_path / "out64.png"
        meta = compose_lito_input(fp, mp, str(out), resolution=64)
        assert meta["resolution"] == 64
        assert np.array(Image.open(out)).shape == (64, 64, 4)

    def test_empty_mask_raises(self, tmp_path: Path):
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        fp = tmp_path / "empty_frame.jpg"
        mp = tmp_path / "empty_mask.png"
        Image.fromarray(rgb).save(fp, quality=95)
        Image.fromarray(mask).save(mp)
        with pytest.raises(ValueError, match="empty mask"):
            compose_lito_input(str(fp), str(mp), str(tmp_path / "out.png"))

    def test_bbox_clamped_to_frame(self, tmp_path: Path):
        # Mask spans entire frame; bbox-with-padding may extend outside.
        # The compositor must clamp to image bounds and still produce a
        # square letterbox.
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        mask = np.full((100, 200), 255, dtype=np.uint8)
        fp = tmp_path / "fullframe.jpg"
        mp = tmp_path / "fullmask.png"
        Image.fromarray(rgb).save(fp, quality=95)
        Image.fromarray(mask).save(mp)
        out = tmp_path / "out_clamped.png"
        meta = compose_lito_input(str(fp), str(mp), str(out), resolution=128)
        assert meta["bbox"][0] == 0
        assert meta["bbox"][1] == 0
        assert meta["letterbox_side"] >= 200
