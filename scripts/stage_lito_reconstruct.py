"""Stage 4 (lito backend): 3D reconstruction via Apple ml-lito.

Drop-in alternative to scripts.stage_gs2mesh_reconstruct.run_gs2mesh.
The function signature mirrors run_gs2mesh exactly so pipeline.py can swap
backends behind a configuration flag.

Phase 1 (this revision): skeleton only. The function validates inputs,
prepares the lito_workspace directory, emits a clear NotImplementedError,
and otherwise leaves Stage 4 untouched. Phases 2–4 fill in the body
(frame selection → LiTo subprocess → Gaussians → mesh → Sim(3) align).

Output contract (final): writes {output_dir}/p4_mesh/object_mesh.ply in the
COLMAP world frame, identical schema to gs2mesh output (see docs/lito_reconstruct.md
once Phase 6 lands).

Research license: invoking this backend requires acknowledging the Apple
ML Research Model License (LICENSE_MODEL of apple/ml-lito). Set
CLIP2MESH_ACCEPT_LITO_RESEARCH_LICENSE=1 to skip the interactive prompt.
"""

from __future__ import annotations

import os
from pathlib import Path

from scripts.config_defaults import LITO_ACCEPT_RESEARCH_LICENSE_ENV
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

    Raises:
        RuntimeError: research license not acknowledged.
        NotImplementedError: until Phases 2–4 land.
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
    print(f"[lito] target mesh_ply={mesh_ply}")

    raise NotImplementedError(
        "lito backend skeleton only — Phase 2 (inference), Phase 3 (mesh extraction), "
        "and Phase 4 (Sim(3) alignment) not yet implemented. "
        "See .claude/plans/lito_integration.md for the implementation plan."
    )
