"""Stage 2: SAM2 interactive segmentation with Gradio UI.

Provides a minimal Gradio UI for click-based object selection,
then propagates masks to all frames.
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from config_defaults import (
    SAM2_MODEL_CONFIGS,
    _SAM2_CIRCLE_OUTLINE_WIDTH,
    _SAM2_CIRCLE_RADIUS,
    _SAM2_GROUND_MASK_COLOR,
    _SAM2_GROUND_MASK_OVERLAY_ALPHA,
    _SAM2_MASK_COLOR,
    _SAM2_MASK_OVERLAY_ALPHA,
)
from vram_utils import log_vram

_NORM_UPPER = float(np.nextafter(np.float32(1.0), np.float32(0.0)))


def _sanitize_normalized_point(nx: float, ny: float) -> tuple[float, float]:
    """Clamp normalized coordinates into [0, 1) and handle non-finite input."""
    if not np.isfinite(nx):
        nx = 0.0
    if not np.isfinite(ny):
        ny = 0.0
    nx = float(np.clip(nx, 0.0, _NORM_UPPER))
    ny = float(np.clip(ny, 0.0, _NORM_UPPER))
    return nx, ny


class SAM2Session:
    """Manages SAM2 state for the Gradio UI session."""

    def __init__(self, frames_dir: str, output_dir: str, model_type: str):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.mask_dir = self.output_dir / "masks"
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        # Load first frame
        frame_files = sorted(self.frames_dir.glob("*.jpg"))
        if not frame_files:
            raise FileNotFoundError(f"No JPEG frames in {self.frames_dir}")
        self.frame_files = frame_files
        self.first_frame = cv2.cvtColor(
            cv2.imread(str(frame_files[0])), cv2.COLOR_BGR2RGB
        )
        self.img_h, self.img_w = self.first_frame.shape[:2]

        # Click state — object (obj_id=1)
        self.click_points: list[tuple[float, float]] = []
        self.click_labels: list[int] = []

        # Click state — ground plane (obj_id=2)
        self.ground_click_points: list[tuple[float, float]] = []
        self.ground_click_labels: list[int] = []
        self.ground_mask_dir = self.output_dir / "masks_ground"
        self.ground_mask_dir.mkdir(parents=True, exist_ok=True)

        # Completion signal
        self.done_event = threading.Event()

        # SAM2 model (loaded on demand)
        self.predictor = None
        self.inference_state = None

    def _load_model(self):
        """Load SAM2 model (always large)."""
        from sam2.build_sam import build_sam2_video_predictor_hf

        config = SAM2_MODEL_CONFIGS["large"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading SAM2 (large) on {device}...")
        log_vram("before SAM2 load")

        self.predictor = build_sam2_video_predictor_hf(
            model_id=config["hf_model_id"],
            device=device,
        )
        print("SAM2 loaded")
        log_vram("after SAM2 load")

    def _init_inference_state(self):
        """Initialize inference state for video frames."""
        with torch.inference_mode():
            self.inference_state = self.predictor.init_state(
                video_path=str(self.frames_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
        print("SAM2 inference state initialized")
        log_vram("after inference state init")

    def release_model(self):
        """Release SAM2 model and free VRAM."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        if self.inference_state is not None:
            del self.inference_state
            self.inference_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        log_vram("after SAM2 release")


def _create_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray | None,
    points: list[tuple[float, float]],
    labels: list[int],
    alpha: float = _SAM2_MASK_OVERLAY_ALPHA,
    ground_mask: np.ndarray | None = None,
    ground_points: list[tuple[float, float]] | None = None,
    ground_labels: list[int] | None = None,
) -> np.ndarray:
    """Draw segmentation mask overlay and click points on frame."""
    vis = frame.copy().astype(np.float32)

    # Object mask in blue
    color = np.array(_SAM2_MASK_COLOR, dtype=np.float32)
    if mask is not None and mask.any():
        vis[mask] = vis[mask] * (1 - alpha) + color * alpha

    # Ground mask in orange
    ground_color = np.array(_SAM2_GROUND_MASK_COLOR, dtype=np.float32)
    ground_alpha = _SAM2_GROUND_MASK_OVERLAY_ALPHA
    if ground_mask is not None and ground_mask.any():
        vis[ground_mask] = vis[ground_mask] * (1 - ground_alpha) + ground_color * ground_alpha

    vis = np.clip(vis, 0, 255).astype(np.uint8)

    h, w = vis.shape[:2]

    # Object click points (green positive, red negative)
    for (nx, ny), label in zip(points, labels):
        nx, ny = _sanitize_normalized_point(nx, ny)
        px = int(np.clip(round(nx * w), 0, w - 1))
        py = int(np.clip(round(ny * h), 0, h - 1))
        c = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(vis, (px, py), _SAM2_CIRCLE_RADIUS, c, -1)
        cv2.circle(vis, (px, py), _SAM2_CIRCLE_RADIUS + _SAM2_CIRCLE_OUTLINE_WIDTH, (255, 255, 255), _SAM2_CIRCLE_OUTLINE_WIDTH)

    # Ground click points (orange positive, dark red negative)
    if ground_points and ground_labels:
        for (nx, ny), label in zip(ground_points, ground_labels):
            nx, ny = _sanitize_normalized_point(nx, ny)
            px = int(np.clip(round(nx * w), 0, w - 1))
            py = int(np.clip(round(ny * h), 0, h - 1))
            c = (255, 165, 0) if label == 1 else (180, 0, 0)
            cv2.circle(vis, (px, py), _SAM2_CIRCLE_RADIUS, c, -1)
            cv2.circle(vis, (px, py), _SAM2_CIRCLE_RADIUS + _SAM2_CIRCLE_OUTLINE_WIDTH, (200, 200, 200), _SAM2_CIRCLE_OUTLINE_WIDTH)

    return vis


def _run_single_frame_inference_obj(
    session: SAM2Session,
    obj_id: int,
    click_points: list[tuple[float, float]],
    click_labels: list[int],
) -> np.ndarray:
    """Run SAM2 inference on frame 0 for a specific object with explicit click lists."""
    points_px = []
    for p in click_points:
        nx, ny = _sanitize_normalized_point(p[0], p[1])
        points_px.append([nx * session.img_w, ny * session.img_h])
    points_np = np.array(points_px, dtype=np.float32)
    labels_np = np.array(click_labels, dtype=np.int32)

    with torch.inference_mode():
        _, obj_ids_out, masks_out = session.predictor.add_new_points_or_box(
            inference_state=session.inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=points_np,
            labels=labels_np,
            clear_old_points=True,
            normalize_coords=True,
        )
    obj_idx = list(obj_ids_out).index(obj_id)
    mask = (masks_out[obj_idx] > 0).cpu().numpy().squeeze()
    return mask


def _run_single_frame_inference(session: SAM2Session) -> np.ndarray:
    """Run SAM2 inference on frame 0 with all accumulated click points (obj_id=1)."""
    return _run_single_frame_inference_obj(
        session, obj_id=1,
        click_points=session.click_points,
        click_labels=session.click_labels,
    )


def run_sam2_interactive(
    frames_dir: str,
    output_dir: str,
    model_type: str | None = None,
) -> Path:
    """Launch Gradio UI for SAM2 segmentation, block until complete.

    Args:
        frames_dir: Directory of JPEG frames.
        output_dir: Parent output directory.
        model_type: SAM2 model variant. Default from env SAM2_MODEL or "large".

    Returns:
        Path to masks directory.
    """
    import gradio as gr

    if model_type is None:
        model_type = os.environ.get("SAM2_MODEL", "large")

    session = SAM2Session(frames_dir, output_dir, model_type)

    # Pre-load model and initialize inference state before UI
    session._load_model()
    session._init_inference_state()

    def handle_click(image, evt: gr.SelectData, is_negative: bool):
        """Handle click: accumulate point, run SAM2 inference, show overlay."""
        x, y = evt.index[0], evt.index[1]
        norm_x = x / session.img_w
        norm_y = y / session.img_h
        session.click_points.append((norm_x, norm_y))
        session.click_labels.append(0 if is_negative else 1)

        mask = _run_single_frame_inference(session)

        vis = _create_mask_overlay(
            session.first_frame, mask,
            session.click_points, session.click_labels,
        )
        pos = sum(1 for l in session.click_labels if l == 1)
        neg = sum(1 for l in session.click_labels if l == 0)
        status = f"Points: {pos} positive, {neg} negative"
        return vis, status

    def clear_clicks():
        session.click_points.clear()
        session.click_labels.clear()
        session.predictor.reset_state(session.inference_state)
        return session.first_frame, "Clicks cleared"

    def propagate_and_finish(progress=gr.Progress()):
        """Propagate masks to all frames, then signal completion."""
        if not session.click_points:
            return "Add at least one click point first"

        try:
            progress(0.3, desc="Propagating masks...")
            with torch.inference_mode():
                num_frames = session.inference_state["num_frames"]
                for frame_idx, _, masks_tensor in session.predictor.propagate_in_video(
                    session.inference_state
                ):
                    mask = (masks_tensor[0] > 0).cpu().numpy().squeeze()
                    mask_path = session.mask_dir / f"{frame_idx:05d}.png"
                    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                    pct = 0.3 + 0.6 * (frame_idx + 1) / num_frames
                    progress(pct, desc=f"Frame {frame_idx+1}/{num_frames}")

            progress(0.95, desc="Releasing SAM2 model...")
            session.release_model()

            progress(1.0, desc="Done!")
            msg = (
                f"Propagated masks to {num_frames} frames\n"
                f"Masks saved to: {session.mask_dir}\n\n"
                "Pipeline will continue automatically..."
            )

            # Signal completion after a short delay to let UI update
            def _signal():
                import time
                time.sleep(2)
                session.done_event.set()
            threading.Thread(target=_signal, daemon=True).start()

            return msg

        except Exception as e:
            session.release_model()
            return f"Error: {e}"

    # Build Gradio UI
    with gr.Blocks(title="SAM2 Segmentation") as demo:
        gr.Markdown("# SAM2 Object Segmentation")
        gr.Markdown(
            "1. Click on the object to segment (green=include)\n"
            "2. Toggle 'Negative mode' and click to exclude regions (red)\n"
            "   → Segmentation preview updates on each click\n"
            "3. Click **Confirm & Propagate** to process all frames"
        )

        with gr.Row():
            with gr.Column(scale=2):
                image_display = gr.Image(
                    label="Click to select object",
                    value=session.first_frame,
                    interactive=False,
                    type="numpy",
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear Clicks", size="sm")
                    negative_mode = gr.Checkbox(
                        label="Negative mode (exclude)", value=False
                    )

            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    label="Status", interactive=False, lines=2,
                    value=f"Frame: {session.img_w}x{session.img_h}, "
                          f"{len(session.frame_files)} frames",
                )
                propagate_btn = gr.Button(
                    "Confirm & Propagate", variant="primary", size="lg"
                )
                propagate_status = gr.Textbox(
                    label="Propagation Result", interactive=False, lines=6
                )

        image_display.select(
            fn=handle_click,
            inputs=[image_display, negative_mode],
            outputs=[image_display, status_text],
        )
        clear_btn.click(fn=clear_clicks, outputs=[image_display, status_text])
        propagate_btn.click(fn=propagate_and_finish, outputs=propagate_status)

    # Launch in a thread, wait for completion
    print("Starting SAM2 segmentation UI on port 7860...")
    print("Open http://localhost:7860 in your browser to select the object.")

    server = demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,
    )

    # Block until propagation is complete
    session.done_event.wait()
    print("SAM2 segmentation complete, shutting down UI...")
    demo.close()

    # --- Full VRAM cleanup ---
    # Save return value before destroying session
    mask_dir_result = session.mask_dir

    # Ensure model is released (idempotent)
    session.release_model()

    # Break closure references that Gradio may still hold
    del handle_click, clear_clicks, propagate_and_finish
    del session

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log_vram("after SAM2 full cleanup")

    return mask_dir_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM2 interactive segmentation")
    parser.add_argument("frames_dir", help="Directory of JPEG frames")
    parser.add_argument("--output-dir", default="/data/output")
    parser.add_argument("--model-type", default=None)
    args = parser.parse_args()

    run_sam2_interactive(args.frames_dir, args.output_dir, args.model_type)
