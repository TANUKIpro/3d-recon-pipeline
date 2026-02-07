"""Stage 2: SAM2 interactive segmentation with Gradio UI.

Provides a minimal Gradio UI for click-based object selection,
then propagates masks to all frames.

Adapted from im2pc/host/pi3x_sam2_cli.py and im2pc/app/main.py.
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr

from vram_utils import log_vram

# SAM2 model configurations
SAM2_MODEL_CONFIGS = {
    "tiny": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-tiny",
    },
    "small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-small",
    },
    "base": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-base-plus",
    },
    "large": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-large",
    },
}


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

        # Click state
        self.click_points: list[tuple[float, float]] = []
        self.click_labels: list[int] = []

        # Completion signal
        self.done_event = threading.Event()

        # SAM2 model (loaded on demand)
        self.predictor = None
        self.inference_state = None

    def _load_model(self):
        """Load SAM2 model."""
        from sam2.build_sam import build_sam2_video_predictor_hf

        config = SAM2_MODEL_CONFIGS[self.model_type]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading SAM2 ({self.model_type}) on {device}...")
        log_vram("before SAM2 load")

        self.predictor = build_sam2_video_predictor_hf(
            model_id=config["hf_model_id"],
            device=device,
        )
        print("SAM2 loaded")
        log_vram("after SAM2 load")

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


def _create_click_visualization(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    labels: list[int],
) -> np.ndarray:
    """Draw click points on frame image."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    for (nx, ny), label in zip(points, labels):
        px, py = int(nx * w), int(ny * h)
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(vis, (px, py), 8, color, -1)
        cv2.circle(vis, (px, py), 10, (255, 255, 255), 2)
    return vis


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
    if model_type is None:
        model_type = os.environ.get("SAM2_MODEL", "large")

    session = SAM2Session(frames_dir, output_dir, model_type)

    def on_click(image, evt: gr.SelectData):
        """Handle left-click (positive point)."""
        x, y = evt.index[0], evt.index[1]
        norm_x = x / session.img_w
        norm_y = y / session.img_h
        session.click_points.append((norm_x, norm_y))
        session.click_labels.append(1)

        vis = _create_click_visualization(
            session.first_frame, session.click_points, session.click_labels
        )
        pos = sum(1 for l in session.click_labels if l == 1)
        neg = sum(1 for l in session.click_labels if l == 0)
        status = f"Points: {pos} positive, {neg} negative"
        return vis, status

    def on_click_negative(image, evt: gr.SelectData):
        """Handle click in negative mode."""
        x, y = evt.index[0], evt.index[1]
        norm_x = x / session.img_w
        norm_y = y / session.img_h
        session.click_points.append((norm_x, norm_y))
        session.click_labels.append(0)

        vis = _create_click_visualization(
            session.first_frame, session.click_points, session.click_labels
        )
        pos = sum(1 for l in session.click_labels if l == 1)
        neg = sum(1 for l in session.click_labels if l == 0)
        status = f"Points: {pos} positive, {neg} negative"
        return vis, status

    def clear_clicks():
        session.click_points.clear()
        session.click_labels.clear()
        return session.first_frame, "Clicks cleared"

    def handle_click(image, evt: gr.SelectData, is_negative: bool):
        if is_negative:
            return on_click_negative(image, evt)
        return on_click(image, evt)

    def propagate_and_finish(progress=gr.Progress()):
        """Load SAM2, propagate masks, then signal completion."""
        if not session.click_points:
            return "Add at least one click point first"

        try:
            progress(0, desc="Loading SAM2 model...")
            session._load_model()

            progress(0.1, desc="Initializing video...")
            with torch.inference_mode():
                session.inference_state = session.predictor.init_state(
                    video_path=str(session.frames_dir),
                    offload_video_to_cpu=True,
                )

                # Convert normalized coords to pixel coords for SAM2
                points_px = [
                    [p[0] * session.img_w, p[1] * session.img_h]
                    for p in session.click_points
                ]
                points_np = np.array(points_px, dtype=np.float32)
                labels_np = np.array(session.click_labels, dtype=np.int32)

                progress(0.2, desc="Adding prompt...")
                _, _, masks_out = session.predictor.add_new_points_or_box(
                    inference_state=session.inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points_np,
                    labels=labels_np,
                    clear_old_points=True,
                    normalize_coords=False,
                )

                prompt_mask = (masks_out[0] > 0).cpu().numpy().squeeze()
                prompt_pixels = int(prompt_mask.sum())

                progress(0.3, desc="Propagating masks...")
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
                f"Prompt mask: {prompt_pixels:,} pixels\n"
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

    return session.mask_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM2 interactive segmentation")
    parser.add_argument("frames_dir", help="Directory of JPEG frames")
    parser.add_argument("--output-dir", default="/data/output")
    parser.add_argument("--model-type", default=None)
    args = parser.parse_args()

    run_sam2_interactive(args.frames_dir, args.output_dir, args.model_type)
