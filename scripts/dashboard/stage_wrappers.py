"""Thin wrappers around external stage modules.

Each function delegates to the actual stage implementation, providing a
uniform call signature that ``pipeline_runner._run_stage`` expects.
Extracted from ``pipeline_runner.py`` to keep orchestration logic separate
from per-stage wiring.
"""

from __future__ import annotations

from scripts.dashboard.log_capture import stage_log_scope
from scripts.dashboard.state import PipelineStage


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
    from stage_extract_frames import extract_frames
    extract_frames(
        video_path,
        output_dir,
        frame_interval=frame_interval,
        max_frames=max_frames,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def _stage_pi3x_inference(
    frames_dir: str, output_dir: str,
    pixel_limit: int, pi3x_frame_target: int,
    conf_threshold: float, edge_rtol: float,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del register_process, unregister_process
    from stage_pi3x_reconstruct import run_pi3x_inference
    from vram_utils import cleanup_pytorch_vram
    run_pi3x_inference(
        frames_dir, output_dir,
        pixel_limit=pixel_limit,
        max_frames=pi3x_frame_target,
        conf_threshold=conf_threshold,
        edge_rtol=edge_rtol,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    cleanup_pytorch_vram()


def _stage_apply_masks(
    cache_path: str,
    mask_dir: str,
    output_dir: str,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del register_process, unregister_process
    from stage_pi3x_reconstruct import apply_sam2_masks
    apply_sam2_masks(
        cache_path,
        mask_dir,
        output_dir,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )


def _stage_denoise(
    ply_path: str,
    output_dir: str,
    denoise_preset: str,
    denoise_algorithm: str,
    denoise_dbscan_eps: float,
    denoise_dbscan_eps_ratio: float,
    denoise_dbscan_min_samples: int,
    denoise_dbscan_max_points: int,
    denoise_sor_neighbors: int,
    denoise_sor_std_ratio: float,
    denoise_radius_neighbors: int,
    denoise_radius_radius_ratio: float,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del cancel_cb, register_process, unregister_process
    from stage_denoise import denoise
    denoise(
        ply_path,
        output_dir,
        preset=denoise_preset,
        algorithm=denoise_algorithm,
        dbscan_eps=denoise_dbscan_eps,
        dbscan_eps_ratio=denoise_dbscan_eps_ratio,
        dbscan_min_samples=denoise_dbscan_min_samples,
        dbscan_max_points=denoise_dbscan_max_points,
        sor_neighbors=denoise_sor_neighbors,
        sor_std_ratio=denoise_sor_std_ratio,
        radius_neighbors=denoise_radius_neighbors,
        radius_ratio=denoise_radius_radius_ratio,
        progress_cb=progress_cb,
    )


def _stage_diffcd(
    denoised_ply: str,
    output_dir: str,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    from stage_diffcd_mesh import run_diffcd
    run_diffcd(
        denoised_ply,
        output_dir,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        register_process=register_process,
        unregister_process=unregister_process,
    )


def _stage_classical_mesh(
    denoised_ply: str,
    output_dir: str,
    preprocess_enabled: bool = True,
    poisson_depth: int = 9,
    density_trim_q: float = 0.005,
    auto_smooth: bool = False,
    smooth_iterations: int = 2,
    downsample_enabled: bool = True,
    downsample_target_faces: int = 120_000,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del cancel_cb, register_process, unregister_process
    from stage_classical_mesh import run_classical_mesh

    run_classical_mesh(
        denoised_ply,
        output_dir,
        progress_cb=progress_cb,
        preprocess_enabled=preprocess_enabled,
        poisson_depth=poisson_depth,
        density_trim_q=density_trim_q,
        auto_smooth=auto_smooth,
        smooth_iterations=smooth_iterations,
        downsample_enabled=downsample_enabled,
        downsample_target_faces=downsample_target_faces,
    )


def _stage_mesh_wrap(
    mesh_ply: str,
    output_dir: str,
    poisson_depth: int = 8,
    poisson_scale: float = 1.18,
    density_trim_q: float = 0.003,
    target_face_ratio: float = 2.20,
    iterations: int = 1,
    crop_scale: float = 1.08,
    sample_points: int = 400_000,
    normal_radius_ratio: float = 0.02,
    smooth_iterations: int = 2,
    quality_threshold: float = 0.02,
    method: str = "alpha_wrap",
    alpha_ratio: float = 0.02,
    offset_ratio: float = 0.3,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del cancel_cb, register_process, unregister_process
    from stage_mesh_wrap import run_mesh_wrap

    run_mesh_wrap(
        mesh_ply,
        output_dir,
        progress_cb=progress_cb,
        poisson_depth=poisson_depth,
        poisson_scale=poisson_scale,
        density_trim_q=density_trim_q,
        target_face_ratio=target_face_ratio,
        iterations=iterations,
        crop_scale=crop_scale,
        sample_points=sample_points,
        normal_radius_ratio=normal_radius_ratio,
        smooth_iterations=smooth_iterations,
        quality_threshold=quality_threshold,
        method=method,
        alpha_ratio=alpha_ratio,
        offset_ratio=offset_ratio,
    )


def _stage_mesh_repair(
    mesh_ply: str,
    output_dir: str,
    mesh_repair_enabled: bool = True,
    mesh_repair_max_diameter_ratio: float = 0.08,
    mesh_repair_y_band_ratio: float = 0.06,
    mesh_repair_smooth_iters: int = 3,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del register_process, unregister_process
    from stage_contact_hole_repair import run_contact_hole_repair

    run_contact_hole_repair(
        mesh_ply,
        output_dir,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        enabled=mesh_repair_enabled,
        max_diameter_ratio=mesh_repair_max_diameter_ratio,
        y_band_ratio=mesh_repair_y_band_ratio,
        smooth_iters=mesh_repair_smooth_iters,
    )


def _stage_mesh_repair_analyze(
    mesh_ply: str,
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    from stage_contact_hole_repair import analyze_contact_hole_candidates

    with stage_log_scope(int(PipelineStage.MESH_REPAIR)):
        analysis = analyze_contact_hole_candidates(
            mesh_ply,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    return analysis.to_dict()


def _stage_mesh_repair_selected(
    mesh_ply: str,
    output_dir: str,
    selected_loop_ids: list[int],
    mesh_repair_enabled: bool = True,
    mesh_repair_max_diameter_ratio: float = 0.08,
    mesh_repair_y_band_ratio: float = 0.06,
    mesh_repair_smooth_iters: int = 3,
    progress_cb=None,
    cancel_cb=None,
) -> None:
    from stage_contact_hole_repair import run_selected_contact_hole_repair

    with stage_log_scope(int(PipelineStage.MESH_REPAIR)):
        run_selected_contact_hole_repair(
            mesh_ply,
            output_dir,
            selected_loop_ids=selected_loop_ids,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            enabled=mesh_repair_enabled,
            max_diameter_ratio=mesh_repair_max_diameter_ratio,
            y_band_ratio=mesh_repair_y_band_ratio,
            smooth_iters=mesh_repair_smooth_iters,
        )


def _stage_texture_bake(
    mesh_ply: str, poses_path: str, frames_dir: str,
    mask_dir: str, output_dir: str, texture_size: int,
    texture_view_assign_mode: str = "legacy",
    texture_quality_boost: bool = False,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> None:
    del cancel_cb, register_process, unregister_process
    from stage_texture_bake import bake_texture
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


def _vram_gate() -> None:
    from scripts.config_defaults import _VRAM_GATE_MIN_FREE_MB
    from vram_utils import cleanup_pytorch_vram, ensure_vram_available
    cleanup_pytorch_vram()
    ensure_vram_available(min_free_mb=_VRAM_GATE_MIN_FREE_MB, stage_name="before Pi3X")
