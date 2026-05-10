"""Tests for scripts.lito.frame_selector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.lito.config import FrameSelectionConfig
from scripts.lito.frame_selector import (
    evaluate_gate,
    measure_frame,
    select_best_frame,
)


def _make_default_cfg(**overrides) -> FrameSelectionConfig:
    base = dict(
        weight_mask_coverage=0.7,
        weight_triangulation=0.0,
        weight_sharpness=0.3,
        manual_index=None,
        gate_min_bbox_short_px=8,
        gate_min_mask_coverage=0.05,
        gate_max_mask_coverage=0.80,
        gate_max_connected_components=1,
    )
    base.update(overrides)
    return FrameSelectionConfig(**base)


def _write_pair(
    tmp_path: Path,
    idx: int,
    *,
    rgb_size=(256, 256),
    mask_box=(64, 64, 192, 192),
    sharpness="checker",
) -> tuple[Path, Path]:
    frames = tmp_path / "frames"
    masks = tmp_path / "masks"
    frames.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    h, w = rgb_size
    if sharpness == "blur":
        rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    else:
        # Sharp checker pattern → high Laplacian variance
        y, x = np.mgrid[0:h, 0:w]
        rgb = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = mask_box
    mask[y0:y1, x0:x1] = 255
    fp = frames / f"{idx:05d}.jpg"
    mp = masks / f"{idx:05d}.png"
    Image.fromarray(rgb).save(fp, quality=95)
    Image.fromarray(mask).save(mp)
    return fp, mp


class TestMeasureFrame:
    def test_basic_metrics(self, tmp_path: Path):
        fp, mp = _write_pair(tmp_path, 0)
        m = measure_frame(0, fp, mp)
        assert m.frame_index == 0
        assert 0.20 < m.mask_coverage < 0.30  # 128*128 / 256*256 = 0.25
        assert m.sharpness > 0
        assert m.bbox == (64, 64, 192, 192)
        assert m.mask_components == 1


class TestEvaluateGate:
    def test_pass_under_default_cfg(self, tmp_path: Path):
        fp, mp = _write_pair(tmp_path, 0)
        m = measure_frame(0, fp, mp)
        out = evaluate_gate(m, _make_default_cfg())
        assert out.passed, out.reasons

    def test_reject_low_coverage(self, tmp_path: Path):
        fp, mp = _write_pair(tmp_path, 0, mask_box=(0, 0, 4, 4))
        m = measure_frame(0, fp, mp)
        out = evaluate_gate(m, _make_default_cfg())
        assert not out.passed
        assert any("mask_coverage" in r for r in out.reasons)

    def test_reject_high_coverage(self, tmp_path: Path):
        fp, mp = _write_pair(tmp_path, 0, mask_box=(0, 0, 256, 256))
        m = measure_frame(0, fp, mp)
        out = evaluate_gate(m, _make_default_cfg())
        assert not out.passed
        assert any("mask_coverage" in r for r in out.reasons)

    def test_reject_small_bbox(self, tmp_path: Path):
        fp, mp = _write_pair(tmp_path, 0, mask_box=(60, 60, 70, 70))
        m = measure_frame(0, fp, mp)
        cfg = _make_default_cfg(gate_min_bbox_short_px=32)
        out = evaluate_gate(m, cfg)
        assert not out.passed
        assert any("bbox_short" in r for r in out.reasons)

    def test_reject_multi_component(self, tmp_path: Path):
        fp, mp = _write_pair(tmp_path, 0, mask_box=(40, 40, 80, 80))
        # paint a second blob
        m_arr = np.array(Image.open(mp))
        m_arr[150:200, 150:200] = 255
        Image.fromarray(m_arr).save(mp)
        m = measure_frame(0, fp, mp)
        out = evaluate_gate(m, _make_default_cfg())
        assert not out.passed
        assert any("connected_components" in r for r in out.reasons)


class TestSelectBestFrame:
    def test_picks_highest_score(self, tmp_path: Path):
        # frame 0: small object (coverage 0.0625)
        _write_pair(tmp_path, 0, mask_box=(64, 64, 128, 128))
        # frame 1: bigger object (coverage 0.25) → higher score
        _write_pair(tmp_path, 1, mask_box=(64, 64, 192, 192))
        sel = select_best_frame(
            str(tmp_path / "frames"),
            str(tmp_path / "masks"),
            _make_default_cfg(),
        )
        assert sel.frame_index == 1
        assert sel.score > 0

    def test_manual_index_overrides(self, tmp_path: Path):
        _write_pair(tmp_path, 0, mask_box=(64, 64, 96, 96))
        _write_pair(tmp_path, 1, mask_box=(64, 64, 192, 192))
        sel = select_best_frame(
            str(tmp_path / "frames"),
            str(tmp_path / "masks"),
            _make_default_cfg(manual_index=0),
        )
        assert sel.frame_index == 0

    def test_no_pairs_raises(self, tmp_path: Path):
        (tmp_path / "frames").mkdir()
        (tmp_path / "masks").mkdir()
        with pytest.raises(ValueError, match="no paired frames"):
            select_best_frame(
                str(tmp_path / "frames"),
                str(tmp_path / "masks"),
                _make_default_cfg(),
            )

    def test_all_frames_fail_gates_raises(self, tmp_path: Path):
        # both frames have empty masks — fail bbox gate
        _write_pair(tmp_path, 0, mask_box=(0, 0, 1, 1))
        _write_pair(tmp_path, 1, mask_box=(0, 0, 1, 1))
        with pytest.raises(ValueError, match="no frame passes quality gates"):
            select_best_frame(
                str(tmp_path / "frames"),
                str(tmp_path / "masks"),
                _make_default_cfg(),
            )

    def test_manual_index_passes_even_if_gates_fail(self, tmp_path: Path, capsys):
        _write_pair(tmp_path, 0, mask_box=(0, 0, 4, 4))  # fails coverage
        cfg = _make_default_cfg(manual_index=0)
        sel = select_best_frame(
            str(tmp_path / "frames"),
            str(tmp_path / "masks"),
            cfg,
        )
        assert sel.frame_index == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
