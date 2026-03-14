"""Mesh repair and post-processing routes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from scripts.dashboard.configuration import (
    env_int,
    parse_bool,
    parse_choice,
    parse_float,
    parse_int,
)
from scripts.dashboard.dependencies import MESH_POSTPROCESS_METHODS, get_state
from scripts.dashboard.object_store import STAGE_RESET_PATHS, reset_outputs_from_stage
from scripts.dashboard.pipeline_runner import broadcast
from scripts.dashboard.state import PipelineStage

router = APIRouter()


@router.get("/api/mesh-repair/candidates")
async def mesh_repair_candidates():
    session = get_state().session
    if not session.running:
        return JSONResponse({"error": "Pipeline is not running"}, status_code=409)
    if not session.mesh_repair_ready:
        return JSONResponse({"error": "Mesh repair selection is not ready"}, status_code=409)

    source_rel: str | None = None
    source_path = session.mesh_repair_source_mesh_path
    if source_path:
        try:
            out = get_state().active_output_dir().resolve()
            source_obj = Path(source_path).resolve()
            source_rel = str(source_obj.relative_to(out))
        except Exception:
            source_rel = None

    return JSONResponse(
        {
            "status": "ok",
            "source_mesh_path": source_path,
            "source_mesh_relpath": source_rel,
            "loop_count": len(session.mesh_repair_candidates),
            "loops": session.mesh_repair_candidates,
            "analysis": session.mesh_repair_analysis,
            "color_scheme": {
                "candidate": "#f4d03f",
                "selected": "#e74c3c",
                "confirmed": "#2ecc71",
            },
        }
    )


@router.post("/api/mesh-repair/confirm")
async def mesh_repair_confirm(body: dict | None = None):
    session = get_state().session
    if not session.running:
        return JSONResponse({"error": "Pipeline is not running"}, status_code=409)
    if not session.mesh_repair_ready:
        return JSONResponse({"error": "Mesh repair selection is not ready"}, status_code=409)

    raw = body or {}
    selected_raw = raw.get("selected_loop_ids")
    if not isinstance(selected_raw, list):
        return JSONResponse({"error": "selected_loop_ids must be a list"}, status_code=400)

    selected: list[int] = []
    seen: set[int] = set()
    for value in selected_raw:
        try:
            loop_id = int(value)
        except (TypeError, ValueError):
            return JSONResponse({"error": f"Invalid loop id: {value!r}"}, status_code=400)
        if loop_id in seen:
            continue
        selected.append(loop_id)
        seen.add(loop_id)

    available_ids = {
        int(item.get("loop_id"))
        for item in session.mesh_repair_candidates
        if isinstance(item, dict) and item.get("loop_id") is not None
    }
    missing = sorted(loop_id for loop_id in selected if loop_id not in available_ids)
    if missing:
        return JSONResponse({"error": f"Unknown loop ids: {missing}"}, status_code=400)

    session.mesh_repair_selected_loop_ids = selected
    session.mesh_repair_confirm_event.set()
    return JSONResponse(
        {
            "status": "confirmed",
            "mode": "skip" if len(selected) == 0 else "repair",
            "selected_loop_ids": selected,
            "selected_count": len(selected),
        }
    )


@router.post("/api/mesh/postprocess")
async def mesh_postprocess(body: dict | None = None):
    session = get_state().session
    if session.running:
        waiting_stage5 = (
            session.next_stage_confirmation_required
            and session.next_stage_confirmation_from == int(PipelineStage.DIFFCD_MESH)
        )
        if not waiting_stage5:
            return JSONResponse(
                {
                    "error": (
                        "Mesh post-process is available only when the pipeline is idle "
                        "or paused after Stage 5."
                    )
                },
                status_code=409,
            )

    raw = body or {}
    method = parse_choice(raw.get("method"), MESH_POSTPROCESS_METHODS, "laplacian")
    iterations = max(0, min(100, parse_int(raw.get("iterations"), 6)))
    lamb = max(0.01, min(1.5, parse_float(raw.get("lamb"), 0.5)))
    taubin_nu = max(-1.5, min(-0.01, parse_float(raw.get("taubin_nu"), -0.53)))
    downsample_enabled = parse_bool(
        raw.get("downsample_enabled"),
        parse_bool(os.environ.get("CLASSICAL_DOWNSAMPLE_ENABLED"), True),
    )
    downsample_target_faces = max(
        1000,
        parse_int(
            raw.get("downsample_target_faces"),
            env_int("CLASSICAL_DOWNSAMPLE_TARGET_FACES", 120000),
        ),
    )
    downsample_trigger_faces = max(
        parse_int(
            raw.get("downsample_trigger_faces"),
            env_int("CLASSICAL_DOWNSAMPLE_TRIGGER_FACES", 170000),
        ),
        downsample_target_faces,
    )
    source = str(raw.get("source") or "raw").strip().lower()
    if source not in {"raw", "current"}:
        source = "raw"
    invalidate_texture = parse_bool(raw.get("invalidate_texture"), True)

    out = get_state().active_output_dir()
    mesh_path = out / "object_mesh.ply"
    if not mesh_path.is_file():
        return JSONResponse({"error": "object_mesh.ply not found"}, status_code=404)

    source_path = out / "object_mesh_raw.ply" if source == "raw" else mesh_path
    if not source_path.is_file():
        source_path = mesh_path
        source = "current"

    def _apply() -> tuple[int, int, bool]:
        from scripts.stage_classical_mesh import generate_preview_mesh
        from scripts.stage_diffcd_mesh import mesh_vertex_face_count, smooth_mesh_file
        import open3d as o3d

        smooth_mesh_file(
            source_path,
            mesh_path,
            method=method,
            iterations=iterations,
            lamb=lamb,
            taubin_nu=taubin_nu,
        )
        downsample_applied = False
        if downsample_enabled:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            face_count = int(len(mesh.triangles))
            if face_count > downsample_trigger_faces:
                target_faces = min(max(4, downsample_target_faces), max(4, face_count - 1))
                simplified = mesh.simplify_quadric_decimation(target_faces)
                if len(simplified.vertices) > 0 and len(simplified.triangles) > 0:
                    simplified.remove_degenerate_triangles()
                    simplified.remove_duplicated_triangles()
                    simplified.remove_duplicated_vertices()
                    simplified.remove_non_manifold_edges()
                    simplified.remove_unreferenced_vertices()
                    simplified.compute_vertex_normals()
                    try:
                        o3d.io.write_triangle_mesh(
                            str(mesh_path),
                            simplified,
                            write_ascii=False,
                            compressed=False,
                            write_vertex_normals=True,
                        )
                    except TypeError:
                        o3d.io.write_triangle_mesh(str(mesh_path), simplified)
                    downsample_applied = True

        preview_mesh_path = out / "object_mesh_preview.ply"
        try:
            generate_preview_mesh(mesh_path, preview_mesh_path)
        except Exception as preview_err:
            print(f"Preview mesh generation failed after post-process (non-fatal): {preview_err}")

        vertices, faces = mesh_vertex_face_count(mesh_path)
        return vertices, faces, downsample_applied

    try:
        vertices, faces, downsample_applied = await asyncio.to_thread(_apply)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    texture_invalidated = False
    if invalidate_texture:
        invalidate_from = int(PipelineStage.MESH_WRAP)
        downstream_files: tuple[str, ...] = tuple(
            rel
            for stage_id in range(invalidate_from, int(PipelineStage.TEXTURE_BAKE) + 1)
            for rel in STAGE_RESET_PATHS.get(stage_id, {}).get("files", ())
        )
        if any((out / rel).is_file() for rel in downstream_files):
            reset_outputs_from_stage(out, invalidate_from)
            texture_invalidated = True

    session.mesh_ply = str(mesh_path)
    if not session.running:
        session.hydrate_from_output_dir(out)
        await broadcast(session, {"type": "status", **session.to_status_dict()})

    return JSONResponse(
        {
            "status": "ok",
            "method": method,
            "iterations": iterations,
            "lamb": lamb,
            "taubin_nu": taubin_nu,
            "source": source,
            "mesh_path": str(mesh_path.relative_to(out)),
            "vertices": vertices,
            "faces": faces,
            "downsample_enabled": downsample_enabled,
            "downsample_target_faces": downsample_target_faces,
            "downsample_trigger_faces": downsample_trigger_faces,
            "downsample_applied": downsample_applied,
            "texture_invalidated": texture_invalidated,
        }
    )
