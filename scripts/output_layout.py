"""Per-phase output directory layout for the 3D-recon pipeline.

Each pipeline stage writes its artifacts under a dedicated ``pN_<name>/``
subdirectory of the per-object output root. Subpath names within each phase
match the legacy top-level names so existing artifact filenames stay stable.
"""

from __future__ import annotations

from pathlib import Path

# Phase directory names ── one per stage, ordered by stage number.
P1_FRAMES = "p1_frames"
P2_COLMAP = "p2_colmap"
P3_MASKS = "p3_masks"
P4_MESH = "p4_mesh"
P5_TEXTURE = "p5_texture"
P6_CLEANUP = "p6_cleanup"

PHASE_DIRS: tuple[str, ...] = (
    P1_FRAMES,
    P2_COLMAP,
    P3_MASKS,
    P4_MESH,
    P5_TEXTURE,
    P6_CLEANUP,
)

# Inner artifact names (kept identical to the legacy top-level names so
# rename churn stays minimal — only the wrapping phase prefix is added).
FRAMES_DIR = P1_FRAMES  # Stage 1 has a single artifact, so the phase dir
                        # itself is the frames container (no inner nesting).

COLMAP_SPARSE_DIRNAME = "colmap_sparse"
COLMAP_WORKSPACE_DIRNAME = "colmap_workspace"
COLMAP_SPARSE_POINTS_FILENAME = "colmap_sparse_points.ply"
CAMERA_POSES_FILENAME = "camera_poses.json"
INTRINSICS_FILENAME = "intrinsics.json"

MASKS_DIRNAME = "masks"
MASKS_OBJECT_RAW_DIRNAME = "masks_object_raw"
MASKS_GROUND_DIRNAME = "masks_ground"
GROUND_PLANE_FILENAME = "ground_plane.json"

OBJECT_MESH_FILENAME = "object_mesh.ply"
GS2MESH_WORKSPACE_DIRNAME = "gs2mesh_workspace"

TEXTURED_MESH_OBJ_FILENAME = "textured_mesh.obj"
TEXTURED_MESH_MTL_FILENAME = "textured_mesh.mtl"
TEXTURE_PNG_FILENAME = "texture.png"

CLEANUP_PROPOSAL_DIRNAME = "post_texture_contact_cleanup"


# ── Phase root directories ────────────────────────────────────────

def frames_dir(out: str | Path) -> Path:
    return Path(out) / P1_FRAMES


def colmap_phase_dir(out: str | Path) -> Path:
    return Path(out) / P2_COLMAP


def masks_phase_dir(out: str | Path) -> Path:
    return Path(out) / P3_MASKS


def mesh_phase_dir(out: str | Path) -> Path:
    return Path(out) / P4_MESH


def texture_phase_dir(out: str | Path) -> Path:
    return Path(out) / P5_TEXTURE


def cleanup_phase_dir(out: str | Path) -> Path:
    return Path(out) / P6_CLEANUP


# ── Stage 2 (COLMAP) artifacts ───────────────────────────────────

def colmap_sparse_dir(out: str | Path) -> Path:
    return colmap_phase_dir(out) / COLMAP_SPARSE_DIRNAME


def colmap_workspace_dir(out: str | Path) -> Path:
    return colmap_phase_dir(out) / COLMAP_WORKSPACE_DIRNAME


def colmap_sparse_points_path(out: str | Path) -> Path:
    return colmap_phase_dir(out) / COLMAP_SPARSE_POINTS_FILENAME


def camera_poses_path(out: str | Path) -> Path:
    return colmap_phase_dir(out) / CAMERA_POSES_FILENAME


def intrinsics_path(out: str | Path) -> Path:
    return colmap_phase_dir(out) / INTRINSICS_FILENAME


# ── Stage 3 (SAM2) artifacts ─────────────────────────────────────

def masks_dir(out: str | Path) -> Path:
    return masks_phase_dir(out) / MASKS_DIRNAME


def masks_object_raw_dir(out: str | Path) -> Path:
    return masks_phase_dir(out) / MASKS_OBJECT_RAW_DIRNAME


def masks_ground_dir(out: str | Path) -> Path:
    return masks_phase_dir(out) / MASKS_GROUND_DIRNAME


def ground_plane_path(out: str | Path) -> Path:
    return masks_phase_dir(out) / GROUND_PLANE_FILENAME


# ── Stage 4 (gs2mesh) artifacts ──────────────────────────────────

def object_mesh_path(out: str | Path) -> Path:
    return mesh_phase_dir(out) / OBJECT_MESH_FILENAME


def gs2mesh_workspace_dir(out: str | Path) -> Path:
    return mesh_phase_dir(out) / GS2MESH_WORKSPACE_DIRNAME


# ── Stage 5 (texture) artifacts ──────────────────────────────────

def textured_mesh_obj_path(out: str | Path) -> Path:
    return texture_phase_dir(out) / TEXTURED_MESH_OBJ_FILENAME


def textured_mesh_mtl_path(out: str | Path) -> Path:
    return texture_phase_dir(out) / TEXTURED_MESH_MTL_FILENAME


def texture_png_path(out: str | Path) -> Path:
    return texture_phase_dir(out) / TEXTURE_PNG_FILENAME


# ── Stage 6 (cleanup) artifacts ──────────────────────────────────

def cleanup_proposal_dir(out: str | Path) -> Path:
    return cleanup_phase_dir(out) / CLEANUP_PROPOSAL_DIRNAME


def cleanup_final_dir(out: str | Path) -> Path:
    """Subfolder that collects the final deliverables (named after the object)."""
    root = Path(out)
    return cleanup_phase_dir(root) / root.name
