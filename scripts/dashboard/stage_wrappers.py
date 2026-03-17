"""Thin wrappers around external stage modules.

Each function delegates to the actual stage implementation, providing a
uniform call signature that ``pipeline_runner._run_stage`` expects.
Extracted from ``pipeline_runner.py`` to keep orchestration logic separate
from per-stage wiring.
"""

from __future__ import annotations

def _stage_extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int,
    max_frames: int,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del register_process, unregister_process
    from scripts.stage_extract_frames import extract_frames
    extract_frames(
        video_path,
        output_dir,
        frame_interval=frame_interval,
        max_frames=max_frames,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def _stage_colmap_sfm(
    frames_dir: str,
    output_dir: str,
    matcher: str = "exhaustive",
    max_features: int = 32768,
    image_size: int = 1024,
    use_gpu: bool = False,
    dsp_sift: bool = True,
    first_octave: int = -1,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> tuple[str, str]:
    from scripts.stage_colmap_sfm import run_colmap_sfm
    return run_colmap_sfm(
        frames_dir,
        output_dir,
        matcher=matcher,
        max_features=max_features,
        image_size=image_size,
        use_gpu=use_gpu,
        dsp_sift=dsp_sift,
        first_octave=first_octave,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        register_process=register_process,
        unregister_process=unregister_process,
    )


def _stage_gs2mesh_reconstruct(
    frames_dir: str,
    colmap_sparse_dir: str,
    mask_dir: str | None,
    output_dir: str,
    gs_iterations: int = 5000,
    runtime_profile: str = "auto",
    stereo_model: str = "DLNR_Middlebury",
    tsdf_voxel_size: float = 0.005,
    tsdf_depth_trunc: float = 0.04,
    use_masks: bool = True,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    from scripts.stage_gs2mesh_reconstruct import run_gs2mesh
    from scripts.vram_utils import cleanup_pytorch_vram
    result = run_gs2mesh(
        frames_dir,
        colmap_sparse_dir,
        mask_dir,
        output_dir,
        gs_iterations=gs_iterations,
        runtime_profile=runtime_profile,
        stereo_model=stereo_model,
        tsdf_voxel_size=tsdf_voxel_size,
        tsdf_depth_trunc=tsdf_depth_trunc,
        use_masks=use_masks,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        register_process=register_process,
        unregister_process=unregister_process,
    )
    cleanup_pytorch_vram()
    return result


def _stage_texture_bake(
    mesh_ply: str, poses_path: str, frames_dir: str,
    mask_dir: str, output_dir: str, texture_size: int,
    texture_view_assign_mode: str = "region_gc",
    texture_quality_boost: bool = True,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del cancel_cb, register_process, unregister_process
    from scripts.stage_texture_bake import bake_texture
    bake_texture(
        mesh_ply,
        poses_path,
        frames_dir,
        mask_dir,
        output_dir,
        tex_size=texture_size,
        view_assign_mode=texture_view_assign_mode,
        quality_boost=texture_quality_boost,
        progress_cb=progress_cb,
    )


def _stage_extract_ground_plane(
    mesh_ply: str,
    ground_mask_dir: str,
    output_dir: str,
    poses_path: str,
    intrinsics_path: str,
    object_mask_dir: str | None = None,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> dict | None:
    del register_process, unregister_process
    from scripts.ground_plane_extraction import extract_ground_plane_from_mesh
    return extract_ground_plane_from_mesh(
        mesh_ply,
        ground_mask_dir,
        output_dir,
        poses_path,
        intrinsics_path,
        object_mask_dir=object_mask_dir,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def _stage_post_texture_contact_cleanup_prepare(
    textured_obj: str,
    output_dir: str,
    poses_path: str | None = None,
    intrinsics_path: str | None = None,
    masks_dir: str | None = None,
    ground_masks_dir: str | None = None,
    ground_plane_path: str | None = None,
    cleanup_lower_half_threshold: float | None = None,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> dict:
    del register_process, unregister_process
    from scripts.stage_post_texture_contact_cleanup import prepare_cleanup_review
    return prepare_cleanup_review(
        textured_obj,
        output_dir,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        masks_dir=masks_dir,
        ground_masks_dir=ground_masks_dir,
        ground_plane_path=ground_plane_path,
        sam2_only=True,
        cleanup_lower_half_threshold=cleanup_lower_half_threshold,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def _stage_post_texture_contact_cleanup_apply(
    output_dir: str,
    cleanup_lower_half_threshold: float | None = None,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    del register_process, unregister_process
    from scripts.stage_post_texture_contact_cleanup import apply_cleanup_proposal
    return apply_cleanup_proposal(
        output_dir,
        sam2_only=True,
        cleanup_lower_half_threshold=cleanup_lower_half_threshold,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def _stage_post_texture_contact_cleanup_skip(
    output_dir: str,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> str:
    del register_process, unregister_process
    from scripts.stage_post_texture_contact_cleanup import copy_stage5_as_cleaned
    return copy_stage5_as_cleaned(
        output_dir,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def _vram_gate() -> None:
    from scripts.config_defaults import _VRAM_GATE_MIN_FREE_MB
    from scripts.vram_utils import cleanup_pytorch_vram, ensure_vram_available
    cleanup_pytorch_vram()
    ensure_vram_available(min_free_mb=_VRAM_GATE_MIN_FREE_MB, stage_name="before gs2mesh")
