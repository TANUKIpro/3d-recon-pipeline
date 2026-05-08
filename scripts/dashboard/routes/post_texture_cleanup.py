"""Review routes for post-texture contact cleanup."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from scripts.dashboard.dependencies import get_state
from scripts.dashboard.state import PipelineStage
from scripts.output_layout import cleanup_proposal_dir

router = APIRouter()


@router.get("/api/post-texture-cleanup/proposal")
async def post_texture_cleanup_proposal():
    session = get_state().session
    proposal_path = (
        Path(session.cleanup_proposal_path)
        if session.cleanup_proposal_path
        else cleanup_proposal_dir(session.config.output_dir) / "proposal.json"
    )
    if not proposal_path.is_file():
        return JSONResponse({"error": "Cleanup proposal not found"}, status_code=404)
    try:
        payload = proposal_path.read_text(encoding="utf-8")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"proposal": payload and json.loads(payload)})


@router.post("/api/post-texture-cleanup/confirm")
async def post_texture_cleanup_confirm(body: dict | None = None):
    session = get_state().session
    if not session.running:
        return JSONResponse({"error": "No pipeline running"}, status_code=409)
    if int(session.current_stage) != int(PipelineStage.POST_TEXTURE_CONTACT_CLEANUP):
        return JSONResponse({"error": "Cleanup review is not active"}, status_code=409)

    decision = str((body or {}).get("decision", "")).strip().lower()
    if decision not in {"apply", "skip"}:
        return JSONResponse({"error": "decision must be 'apply' or 'skip'"}, status_code=400)

    session.cleanup_decision = decision
    session.cleanup_review_event.set()
    return JSONResponse({"status": "confirmed", "decision": decision})
