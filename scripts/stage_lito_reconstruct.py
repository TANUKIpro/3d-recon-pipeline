"""Stage 4 (lito backend): 3D reconstruction via Apple ml-lito.

Drop-in alternative to scripts.stage_gs2mesh_reconstruct.run_gs2mesh.
The function signature mirrors run_gs2mesh exactly so pipeline.py can
swap backends behind --reconstructor.

Pipeline (matches `.claude/plans/lito_integration.md` §6.4):

    1. Frame selection      — compound score + quality gates
    2. RGBA composition     — SAM2 mask + RGB → 518×518 RGBA letterbox
    3. LiTo subprocess      — image → 3D Gaussians (PLY, canonical frame)
    4. Gaussians → mesh     — opacity-weighted multi-view TSDF (Phase 3)
    5. Sim(3) alignment     — umeyama + ICP into COLMAP world (Phase 4)

Phase 2 (this revision): steps 1–3 are wired up. Step 4–5 raise
NotImplementedError until those phases land.

Research license: invoking this backend requires acknowledging the Apple
ML Research Model License (LICENSE_MODEL of apple/ml-lito). Set
CLIP2MESH_ACCEPT_LITO_RESEARCH_LICENSE=1 to skip the interactive prompt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.config_defaults import (
    LITO_ACCEPT_RESEARCH_LICENSE_ENV,
    LITO_INPUT_RESOLUTION,
)
from scripts.output_layout import object_mesh_path


_LICENSE_NOTICE = """
========================================================================
LiTo (Apple ml-lito) Research License Acknowledgement
------------------------------------------------------------------------
The lito reconstruction backend uses model weights licensed under the
Apple ML Research Model License — RESEARCH PURPOSES ONLY. Commercial
exploitation, product development, or use in any commercial product or
service is NOT permitted.

See: https://github.com/apple/ml-lito/blob/main/LICENSE_MODEL

To proceed non-interactively, set:
    {env_var}=1
========================================================================
""".strip()


def _ensure_research_license_acknowledged() -> None:
    """Block lito backend unless the research-license env flag is set."""
    if os.environ.get(LITO_ACCEPT_RESEARCH_LICENSE_ENV) == "1":
        return
    print(_LICENSE_NOTICE.format(env_var=LITO_ACCEPT_RESEARCH_LICENSE_ENV))
    raise RuntimeError(
        f"lito backend requires {LITO_ACCEPT_RESEARCH_LICENSE_ENV}=1 "
        f"to acknowledge the research-only license"
    )


def _drop_license_marker(workspace: Path) -> None:
    """Drop a LICENSE_RESEARCH_ONLY.txt next to lito outputs."""
    marker = workspace / "LICENSE_RESEARCH_ONLY.txt"
    marker.write_text(
        "Outputs in this directory derive from Apple ml-lito model weights\n"
        "licensed under the Apple ML Research Model License (LICENSE_MODEL).\n"
        "Research purposes only. No commercial use.\n"
        "See: https://github.com/apple/ml-lito/blob/main/LICENSE_MODEL\n"
    )


def run_lito(
    frames_dir: str,
    sparse_dir: str,
    mask_dir: str,
    output_dir: str,
) -> str:
    """Run Stage 4 reconstruction via the LiTo backend.

    Mirrors scripts.stage_gs2mesh_reconstruct.run_gs2mesh signature.

    Returns:
        Absolute path to {output_dir}/p4_mesh/object_mesh.ply (PLY, COLMAP world frame).
    """
    _ensure_research_license_acknowledged()

    output_root = Path(output_dir)
    mesh_ply = object_mesh_path(output_root)
    workspace = mesh_ply.parent / "lito_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    _drop_license_marker(workspace)

    print(f"[lito] frames_dir={frames_dir}")
    print(f"[lito] sparse_dir={sparse_dir}")
    print(f"[lito] mask_dir={mask_dir}")
    print(f"[lito] workspace={workspace}")

    # ---- Step 1: Frame selection -----------------------------------------
    from scripts.lito.frame_selector import select_best_frame

    selection = select_best_frame(frames_dir, mask_dir)
    print(
        f"[lito] selected frame {selection.frame_index} "
        f"score={selection.score:.4f} "
        f"mask_coverage={selection.metric.mask_coverage:.3f} "
        f"sharpness={selection.metric.sharpness:.1f}"
    )
    (workspace / "frame_score.json").write_text(
        json.dumps(
            {
                "frame_index": selection.frame_index,
                "frame_path": selection.frame_path,
                "mask_path": selection.mask_path,
                "score": selection.score,
                "breakdown": selection.breakdown,
                "metric": {
                    "mask_coverage": selection.metric.mask_coverage,
                    "sharpness": selection.metric.sharpness,
                    "triangulation_count": selection.metric.triangulation_count,
                    "bbox": list(selection.metric.bbox) if selection.metric.bbox else None,
                    "mask_components": selection.metric.mask_components,
                },
            },
            indent=2,
        )
    )

    # ---- Step 2: RGBA composition ---------------------------------------
    from scripts.lito.mask_compositor import compose_lito_input

    selected_rgba = workspace / "selected_frame.png"
    compose_meta = compose_lito_input(
        selection.frame_path,
        selection.mask_path,
        str(selected_rgba),
        resolution=LITO_INPUT_RESOLUTION,
    )
    (workspace / "compose_meta.json").write_text(json.dumps(compose_meta, indent=2))
    print(f"[lito] composed input → {selected_rgba}")

    # ---- Step 3: LiTo subprocess ----------------------------------------
    from scripts.lito.lito_runner import run_lito_inference

    canonical_ply = workspace / "gaussians_canonical.ply"
    result = run_lito_inference(
        in_rgba_path=str(selected_rgba),
        out_ply_path=str(canonical_ply),
    )
    print(
        f"[lito] inference complete: gaussians={result.meta.get('gaussian_count')} "
        f"timings={result.meta.get('timings_s')}"
    )

    # ---- Step 4: Gaussians → canonical mesh (Phase 3) -------------------
    from scripts.lito.gaussian_to_mesh import gaussians_to_canonical_mesh

    canonical = gaussians_to_canonical_mesh(
        canonical_ply_path=str(canonical_ply),
        workspace=str(workspace),
    )
    print(
        f"[lito] canonical mesh: {canonical.triangle_count} tris, "
        f"{canonical.vertex_count} verts "
        f"({canonical.n_views_used} views used, {canonical.n_views_skipped} skipped) "
        f"→ {canonical.mesh_path}"
    )

    # ---- Step 5: Sim(3) alignment to COLMAP world (Phase 4) -------------
    raise NotImplementedError(
        "lito backend Phase 3 produces canonical-frame mesh at "
        f"{canonical.mesh_path}. Phase 4 (Sim(3) alignment to COLMAP world) "
        "is not yet implemented. See .claude/plans/lito_integration.md §6.4 step 5."
    )
