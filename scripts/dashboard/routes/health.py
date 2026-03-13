"""Health and VRAM monitoring routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/vram")
async def vram_info():
    """Return current VRAM usage."""
    try:
        from scripts.vram_utils import get_free_vram_mb
        free = get_free_vram_mb()
        return JSONResponse({"free_mb": free})
    except Exception:
        return JSONResponse({"free_mb": None})
