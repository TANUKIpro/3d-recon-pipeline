"""Preview and file serving routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from scripts.dashboard.dependencies import get_state
from scripts.dashboard.object_store import (
    list_preview_files,
    object_dir,
    resolve_output_root,
    validate_object_name,
)

router = APIRouter()


@router.get("/api/preview/object-file/{object_name}/{path:path}")
async def preview_object_file(object_name: str, path: str, branch: str | None = None):
    """Serve a file from any object's directory (not just active session).

    The optional ``?branch=`` query parameter selects the branch namespace.
    Defaults to the current branch.
    """
    try:
        object_name = validate_object_name(object_name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    state = get_state()
    slug = branch or state.branch_slug
    base_output = resolve_output_root(state.output_dir)
    out = object_dir(object_name, base_output, slug)
    if not out.is_dir():
        return JSONResponse({"error": "Object not found"}, status_code=404)

    target = (out / path).resolve()
    if not str(target).startswith(str(out.resolve())):
        return JSONResponse({"error": "Access denied"}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    media_types = {
        ".ply": "application/octet-stream",
        ".obj": "text/plain",
        ".mtl": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".json": "application/json",
    }
    mt = media_types.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(
        str(target),
        media_type=mt,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/preview/outputs")
async def preview_outputs():
    """List output files available for preview."""
    out = get_state().active_output_dir()
    files = list_preview_files(out)
    for f in files:
        f.pop("size_bytes", None)
    return JSONResponse({"files": files})


@router.get("/api/preview/file/{path:path}")
async def preview_file(path: str):
    """Serve an output file by relative path."""
    out = get_state().active_output_dir()
    target = (out / path).resolve()
    # Security: ensure target is within output directory
    if not str(target).startswith(str(out.resolve())):
        return JSONResponse({"error": "Access denied"}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    media_types = {
        ".ply": "application/octet-stream",
        ".obj": "text/plain",
        ".mtl": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".json": "application/json",
    }
    mt = media_types.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(
        str(target),
        media_type=mt,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/preview/crop-obb")
async def preview_crop_obb():
    """Return OBB (center, extent, rotation) for the object mesh."""
    out = get_state().active_output_dir()
    mesh_path = out / "object_mesh.ply"
    if not mesh_path.is_file():
        return JSONResponse({"error": "object_mesh.ply not found"}, status_code=404)
    try:
        import open3d as o3d
        import numpy as np
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        obb = mesh.get_oriented_bounding_box()
        return JSONResponse({
            "center": np.asarray(obb.center).tolist(),
            "extent": np.asarray(obb.extent).tolist(),
            "rotation": np.asarray(obb.R).tolist(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
