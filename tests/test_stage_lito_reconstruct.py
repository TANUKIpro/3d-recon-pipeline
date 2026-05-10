"""Smoke tests for ``scripts.stage_lito_reconstruct.run_lito``.

These tests stub out the heavy lito subprocess and Open3D-backed steps
(rendering, fusion, alignment) so the orchestration layer can be
exercised on hosts that have no GPU and no ``/opt/ml-lito/.venv``. The
goal is end-to-end coverage of the Stage 4 wiring:

    frame_selector → mask_compositor → run_lito_inference (mocked)
                  → gaussians_to_canonical_mesh (mocked)
                  → align_canonical_mesh_to_world (mocked)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.config_defaults import LITO_ACCEPT_RESEARCH_LICENSE_ENV


def _make_frame_pair(frames_dir: Path, masks_dir: Path, idx: int) -> None:
    """Write a sharp checker frame + a centred square mask."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = (((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8)
    rgb = np.repeat(rgb[..., None], 3, axis=2)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[128:384, 128:384] = 255
    Image.fromarray(rgb).save(frames_dir / f"{idx:05d}.jpg", quality=95)
    Image.fromarray(mask).save(masks_dir / f"{idx:05d}.png")


def _write_minimal_ply(path: Path, n_points: int = 64) -> None:
    """Write a tiny ASCII PLY so colmap_align._load_gaussian_centres works."""
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(n_points, 3)).astype(np.float32)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    body = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)
    path.write_text(header + body + "\n")


def _make_lito_workspace(tmp_path: Path) -> dict[str, Path]:
    """Lay out frames/masks/output dirs that ``run_lito`` expects."""
    frames_dir = tmp_path / "p1_frames"
    masks_dir = tmp_path / "p3_masks" / "masks"
    sparse_dir = tmp_path / "p2_colmap" / "colmap_sparse"
    output_dir = tmp_path
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _make_frame_pair(frames_dir, masks_dir, i)
    return {
        "frames_dir": frames_dir,
        "masks_dir": masks_dir,
        "sparse_dir": sparse_dir,
        "output_dir": output_dir,
    }


@pytest.fixture
def accept_license(monkeypatch):
    monkeypatch.setenv(LITO_ACCEPT_RESEARCH_LICENSE_ENV, "1")


def _stub_inference_writer(out_ply_path: str):
    """Drop a tiny canonical PLY where the bridge would have."""
    _write_minimal_ply(Path(out_ply_path))


def test_run_lito_requires_research_license_acknowledgement(tmp_path):
    """Without the env flag, run_lito must refuse to proceed."""
    from scripts import stage_lito_reconstruct

    layout = _make_lito_workspace(tmp_path)
    os.environ.pop(LITO_ACCEPT_RESEARCH_LICENSE_ENV, None)
    with pytest.raises(RuntimeError, match="research-only license"):
        stage_lito_reconstruct.run_lito(
            str(layout["frames_dir"]),
            str(layout["sparse_dir"]),
            str(layout["masks_dir"]),
            str(layout["output_dir"]),
        )


def test_run_lito_smoke_e2e_mocked(tmp_path, monkeypatch, accept_license):
    """End-to-end orchestration with subprocess + heavy compute mocked.

    Verifies the dataflow contract: frame selection produces a frame_score.json,
    mask compositor writes the 518×518 RGBA, the (mocked) bridge produces a
    canonical PLY, the (mocked) mesh fusion / alignment is invoked, and the
    final ``object_mesh.ply`` path is returned.
    """
    from scripts import stage_lito_reconstruct
    from scripts.lito.gaussian_to_mesh import CanonicalMeshResult

    layout = _make_lito_workspace(tmp_path)

    # Mock 1: lito inference subprocess. Writes a small canonical PLY in
    # place of the real bridge.
    inference_calls: list[dict] = []

    def fake_run_lito_inference(in_rgba_path, out_ply_path, **kwargs):
        inference_calls.append({"in_rgba": in_rgba_path, "out_ply": out_ply_path})
        _stub_inference_writer(out_ply_path)
        from scripts.lito.lito_runner import LitoInferenceResult
        return LitoInferenceResult(
            out_ply=out_ply_path,
            meta={"gaussian_count": 64, "timings_s": {"total": 0.01}},
        )

    monkeypatch.setattr(
        "scripts.lito.lito_runner.run_lito_inference",
        fake_run_lito_inference,
    )

    # Mock 2: Gaussians → canonical mesh. The real implementation drives a
    # second subprocess + Open3D TSDF; we just write a degenerate canonical
    # mesh PLY and report it.
    fusion_calls: list[dict] = []

    def fake_gaussians_to_canonical_mesh(canonical_ply_path, workspace, **kwargs):
        fusion_calls.append({"canonical_ply_path": canonical_ply_path})
        canonical_mesh_path = Path(workspace) / "mesh_raw.ply"
        _write_minimal_ply(canonical_mesh_path, n_points=8)
        return CanonicalMeshResult(
            mesh_path=str(canonical_mesh_path),
            n_views_used=1,
            n_views_skipped=0,
            triangle_count=0,
            vertex_count=8,
        )

    monkeypatch.setattr(
        "scripts.lito.gaussian_to_mesh.gaussians_to_canonical_mesh",
        fake_gaussians_to_canonical_mesh,
    )

    # Mock 3: Sim(3) alignment. Just copy the canonical mesh bytes to the
    # world-mesh path and return a plausible AlignmentResult.
    align_calls: list[dict] = []

    def fake_align(
        canonical_ply_path,
        canonical_mesh_path,
        sparse_dir,
        frame_path,
        mask_path,
        out_world_mesh_path,
        **kwargs,
    ):
        align_calls.append(
            {
                "canonical_ply_path": canonical_ply_path,
                "canonical_mesh_path": canonical_mesh_path,
                "sparse_dir": sparse_dir,
                "frame_path": frame_path,
                "mask_path": mask_path,
                "out_world_mesh_path": out_world_mesh_path,
            }
        )
        Path(out_world_mesh_path).parent.mkdir(parents=True, exist_ok=True)
        # Reuse the canonical mesh as the "world" mesh for the smoke test.
        Path(out_world_mesh_path).write_bytes(Path(canonical_mesh_path).read_bytes())
        from scripts.lito.colmap_align import AlignmentResult, Sim3Transform

        identity = Sim3Transform(
            rotation=np.eye(3),
            translation=np.zeros(3),
            scale=1.0,
            residual_rms=0.0,
            inlier_count=128,
        )
        return AlignmentResult(
            world_mesh_path=str(out_world_mesh_path),
            transform=identity,
            n_source_points=64,
            n_target_points=128,
        )

    monkeypatch.setattr(
        "scripts.lito.colmap_align.align_canonical_mesh_to_world",
        fake_align,
    )

    out_path = stage_lito_reconstruct.run_lito(
        str(layout["frames_dir"]),
        str(layout["sparse_dir"]),
        str(layout["masks_dir"]),
        str(layout["output_dir"]),
    )

    # Output contract: object_mesh.ply written under p4_mesh/ and returned.
    expected_mesh = layout["output_dir"] / "p4_mesh" / "object_mesh.ply"
    assert Path(out_path) == expected_mesh
    assert expected_mesh.is_file()

    # Each step ran exactly once.
    assert len(inference_calls) == 1
    assert len(fusion_calls) == 1
    assert len(align_calls) == 1

    # Workspace artefacts exist for debugging / reproducibility.
    workspace = expected_mesh.parent / "lito_workspace"
    assert (workspace / "frame_score.json").is_file()
    assert (workspace / "selected_frame.png").is_file()
    assert (workspace / "compose_meta.json").is_file()
    assert (workspace / "LICENSE_RESEARCH_ONLY.txt").is_file()
    score = json.loads((workspace / "frame_score.json").read_text())
    assert "frame_index" in score
    assert score["mask_path"].endswith(".png")

    # The bridge stub received the composed RGBA, not the raw frame.
    assert inference_calls[0]["in_rgba"] == str(workspace / "selected_frame.png")
    assert inference_calls[0]["out_ply"] == str(workspace / "gaussians_canonical.ply")

    # Alignment received the selected frame/mask + sparse_dir for re-projection.
    assert align_calls[0]["sparse_dir"] == str(layout["sparse_dir"])
    assert align_calls[0]["frame_path"].startswith(str(layout["frames_dir"]))
    assert align_calls[0]["mask_path"].startswith(str(layout["masks_dir"]))


def test_run_lito_smoke_propagates_inference_failure(
    tmp_path, monkeypatch, accept_license
):
    """If the bridge subprocess raises, run_lito surfaces the error verbatim."""
    from scripts import stage_lito_reconstruct
    from scripts.lito.lito_runner import LitoSubprocessError

    layout = _make_lito_workspace(tmp_path)

    def fake_run_lito_inference(in_rgba_path, out_ply_path, **kwargs):
        raise LitoSubprocessError("bridge failed: cuda OOM")

    monkeypatch.setattr(
        "scripts.lito.lito_runner.run_lito_inference",
        fake_run_lito_inference,
    )

    with pytest.raises(LitoSubprocessError, match="bridge failed"):
        stage_lito_reconstruct.run_lito(
            str(layout["frames_dir"]),
            str(layout["sparse_dir"]),
            str(layout["masks_dir"]),
            str(layout["output_dir"]),
        )
