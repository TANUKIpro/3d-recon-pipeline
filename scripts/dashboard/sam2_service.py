"""Headless SAM2 service for the dashboard REST API.

Re-uses SAM2Session, _run_single_frame_inference_obj, and _create_mask_overlay
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
        self._current_ground_mask: Any = None
        self._segmentation_mode: str = "object"

    @property
    def initialized(self) -> bool:
        return self._session is not None

    @property
    def segmentation_mode(self) -> str:
        return self._segmentation_mode

    @property
    def has_ground_clicks(self) -> bool:
        s = self._session
        return s is not None and bool(s.ground_click_points)

    def set_mode(self, mode: str) -> None:
        if mode not in ("object", "ground"):
            raise ValueError(f"Invalid mode: {mode!r}")
        self._segmentation_mode = mode

    def initialize(self, frames_dir: str, output_dir: str, model_type: str) -> dict:
        """Load SAM2 model and prepare inference state.

        Returns dict with frame metadata.
        """
        from scripts.stage_sam2_ui import SAM2Session

        with self._lock:
            session = SAM2Session(frames_dir, output_dir, model_type)
            session._load_model()
            session._init_inference_state()
            self._session = session
            self._current_mask = None
            self._current_ground_mask = None
            self._segmentation_mode = "object"
            return {
                "frame_count": len(session.frame_files),
                "width": session.img_w,
                "height": session.img_h,
            }

    def add_click(self, norm_x: float, norm_y: float, label: int) -> bytes:
        """Add a click point and return the mask overlay as PNG bytes."""
        from scripts.stage_sam2_ui import _run_single_frame_inference_obj, _sanitize_normalized_point

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            nx, ny = _sanitize_normalized_point(norm_x, norm_y)

            if self._segmentation_mode == "ground":
                s.ground_click_points.append((nx, ny))
                s.ground_click_labels.append(label)
                self._current_ground_mask = _run_single_frame_inference_obj(
                    s, obj_id=2,
                    click_points=s.ground_click_points,
                    click_labels=s.ground_click_labels,
                )
            else:
                s.click_points.append((nx, ny))
                s.click_labels.append(label)
                self._current_mask = _run_single_frame_inference_obj(
                    s, obj_id=1,
                    click_points=s.click_points,
                    click_labels=s.click_labels,
                )
            return self._render_overlay_png()

    def undo_click(self) -> bytes:
        """Remove last click from active mode and return updated overlay PNG."""
        from scripts.stage_sam2_ui import _run_single_frame_inference_obj

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")

            if self._segmentation_mode == "ground":
                if s.ground_click_points:
                    s.ground_click_points.pop()
                    s.ground_click_labels.pop()
                if s.ground_click_points:
                    self._current_ground_mask = _run_single_frame_inference_obj(
                        s, obj_id=2,
                        click_points=s.ground_click_points,
                        click_labels=s.ground_click_labels,
                    )
                else:
                    self._current_ground_mask = None
            else:
                if s.click_points:
                    s.click_points.pop()
                    s.click_labels.pop()
                if s.click_points:
                    self._current_mask = _run_single_frame_inference_obj(
                        s, obj_id=1,
                        click_points=s.click_points,
                        click_labels=s.click_labels,
                    )
                else:
                    self._current_mask = None
            return self._render_overlay_png()

    def clear_clicks(self) -> bytes:
        """Clear clicks for the active mode only and return updated overlay PNG."""
        from scripts.stage_sam2_ui import _run_single_frame_inference_obj

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")

            if self._segmentation_mode == "ground":
                s.ground_click_points.clear()
                s.ground_click_labels.clear()
                self._current_ground_mask = None
            else:
                s.click_points.clear()
                s.click_labels.clear()
                self._current_mask = None

            # Reset predictor state and re-add remaining points
            s.predictor.reset_state(s.inference_state)
            if s.click_points:
                self._current_mask = _run_single_frame_inference_obj(
                    s, obj_id=1,
                    click_points=s.click_points,
                    click_labels=s.click_labels,
                )
            if s.ground_click_points:
                self._current_ground_mask = _run_single_frame_inference_obj(
                    s, obj_id=2,
                    click_points=s.ground_click_points,
                    click_labels=s.ground_click_labels,
                )

            return self._render_overlay_png()

    def propagate_and_save(self, progress_callback=None) -> tuple[str, str | None]:
        """Propagate masks to all frames and save.

        Returns (mask_dir, ground_mask_dir or None).
        """
        import cv2
        import numpy as np
        import torch
        from scripts.stage_sam2_ui import _run_single_frame_inference_obj

        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("SAM2 not initialized")
            if not s.click_points:
                raise ValueError("No click points to propagate")

            has_ground = bool(s.ground_click_points)

            # Clear stale masks from both directories
            for stale_path in s.mask_dir.glob("*.png"):
                try:
                    stale_path.unlink()
                except OSError:
                    pass
            # Always clean ground masks (prevent stale files on redo skip)
            for stale_path in s.ground_mask_dir.glob("*.png"):
                try:
                    stale_path.unlink()
                except OSError:
                    pass

            # Ensure frame 0 masks are on disk
            if self._current_mask is None:
                self._current_mask = _run_single_frame_inference_obj(
                    s, obj_id=1,
                    click_points=s.click_points,
                    click_labels=s.click_labels,
                )
            if self._current_mask is not None:
                mask0_path = s.mask_dir / "00000.png"
                cv2.imwrite(str(mask0_path), (self._current_mask * 255).astype(np.uint8))

            if has_ground:
                if self._current_ground_mask is None:
                    self._current_ground_mask = _run_single_frame_inference_obj(
                        s, obj_id=2,
                        click_points=s.ground_click_points,
                        click_labels=s.ground_click_labels,
                    )
                if self._current_ground_mask is not None:
                    gmask0_path = s.ground_mask_dir / "00000.png"
                    cv2.imwrite(str(gmask0_path), (self._current_ground_mask * 255).astype(np.uint8))

            # Propagate all objects in video
            with torch.inference_mode():
                num_frames = s.inference_state["num_frames"]
                for frame_idx, obj_ids, masks_tensor in s.predictor.propagate_in_video(
                    s.inference_state
                ):
                    # masks_tensor shape: (num_objects, 1, H, W)
                    for i, oid in enumerate(obj_ids):
                        mask = (masks_tensor[i] > 0).cpu().numpy().squeeze()
                        if oid == 1:
                            mask_path = s.mask_dir / f"{frame_idx:05d}.png"
                            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                        elif oid == 2 and has_ground:
                            mask_path = s.ground_mask_dir / f"{frame_idx:05d}.png"
                            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                    if progress_callback:
                        progress_callback(frame_idx, num_frames)

            ground_dir = str(s.ground_mask_dir) if has_ground else None
            return str(s.mask_dir), ground_dir

    def release(self) -> None:
        """Release model and free VRAM."""
        with self._lock:
            if self._session is not None:
                self._session.release_model()
                self._session = None
            self._current_mask = None
            self._current_ground_mask = None
            self._segmentation_mode = "object"

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
        from scripts.stage_sam2_ui import _create_mask_overlay

        s = self._session
        vis = _create_mask_overlay(
            s.first_frame,
            self._current_mask,
            s.click_points,
            s.click_labels,
            ground_mask=self._current_ground_mask,
            ground_points=s.ground_click_points,
            ground_labels=s.ground_click_labels,
        )
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", vis_bgr)
        return buf.tobytes()
