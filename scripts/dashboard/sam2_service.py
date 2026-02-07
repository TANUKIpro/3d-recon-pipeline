"""Headless SAM2 service for the dashboard REST API.

Re-uses SAM2Session, _run_single_frame_inference, and _create_mask_overlay
from stage_sam2_ui.py — no logic is duplicated.

Heavy imports (cv2, numpy, torch, stage_sam2_ui) are deferred to method calls
so the dashboard can start even without GPU libraries.
"""

from __future__ import annotations

import threading
from typing import Any


class SAM2Service:
    """Thread-safe wrapper around SAM2Session for REST API usage."""

    def __init__(self) -> None:
        self._session: Any = None
        self._lock = threading.Lock()
        self._current_mask: Any = None

    @property
    def initialized(self) -> bool:
        return self._session is not None

    def initialize(self, frames_dir: str, output_dir: str, model_type: str) -> dict:
        """Load SAM2 model and prepare inference state.

        Returns dict with frame metadata.
        """
        from stage_sam2_ui import SAM2Session

        with self._lock:
            session = SAM2Session(frames_dir, output_dir, model_type)
            session._load_model()
            session._init_inference_state()
            self._session = session
            self._current_mask = None
            return {
                "frame_count": len(session.frame_files),
                "width": session.img_w,
                "height": session.img_h,
            }

    def add_click(self, norm_x: float, norm_y: float, label: int) -> bytes:
        """Add a click point and return the mask overlay as PNG bytes."""
        from stage_sam2_ui import _run_single_frame_inference

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            s.click_points.append((norm_x, norm_y))
            s.click_labels.append(label)
            self._current_mask = _run_single_frame_inference(s)
            return self._render_overlay_png()

    def undo_click(self) -> bytes:
        """Remove last click and return updated overlay PNG."""
        from stage_sam2_ui import _run_single_frame_inference

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            if s.click_points:
                s.click_points.pop()
                s.click_labels.pop()
            if s.click_points:
                s.predictor.reset_state(s.inference_state)
                self._current_mask = _run_single_frame_inference(s)
            else:
                s.predictor.reset_state(s.inference_state)
                self._current_mask = None
            return self._render_overlay_png()

    def clear_clicks(self) -> bytes:
        """Clear all clicks and return clean frame PNG."""
        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            s.click_points.clear()
            s.click_labels.clear()
            s.predictor.reset_state(s.inference_state)
            self._current_mask = None
            return self._render_overlay_png()

    def propagate_and_save(self, progress_callback=None) -> str:
        """Propagate masks to all frames and save.

        Returns path to mask directory.
        """
        import cv2
        import numpy as np
        import torch

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            if not s.click_points:
                raise ValueError("No click points to propagate")

            with torch.inference_mode():
                num_frames = s.inference_state["num_frames"]
                for frame_idx, _, masks_tensor in s.predictor.propagate_in_video(
                    s.inference_state
                ):
                    mask = (masks_tensor[0] > 0).cpu().numpy().squeeze()
                    mask_path = s.mask_dir / f"{frame_idx:05d}.png"
                    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                    if progress_callback:
                        progress_callback(frame_idx, num_frames)

            return str(s.mask_dir)

    def release(self) -> None:
        """Release model and free VRAM."""
        with self._lock:
            if self._session is not None:
                self._session.release_model()
                self._session = None
            self._current_mask = None

    def get_frame_jpeg(self, idx: int) -> bytes:
        """Return frame at index as JPEG bytes."""
        import cv2

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            if idx < 0 or idx >= len(s.frame_files):
                raise IndexError(f"Frame index {idx} out of range")
            img = cv2.imread(str(s.frame_files[idx]))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes()

    def get_mask_png(self, idx: int) -> bytes | None:
        """Return saved mask at index as PNG bytes, or None."""
        import cv2

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            mask_path = s.mask_dir / f"{idx:05d}.png"
            if not mask_path.exists():
                return None
            img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            _, buf = cv2.imencode(".png", img)
            return buf.tobytes()

    def _render_overlay_png(self) -> bytes:
        """Render current state as overlay PNG (must hold lock)."""
        import cv2
        from stage_sam2_ui import _create_mask_overlay

        s = self._session
        vis = _create_mask_overlay(
            s.first_frame,
            self._current_mask,
            s.click_points,
            s.click_labels,
        )
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", vis_bgr)
        return buf.tobytes()
