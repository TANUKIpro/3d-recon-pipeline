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

from scripts.config_defaults import (
    SAM2_MODEL_CONFIGS,
    _SAM2_CIRCLE_OUTLINE_WIDTH,
    _SAM2_CIRCLE_RADIUS,
    _SAM2_GROUND_MASK_COLOR,
    _SAM2_GROUND_MASK_OVERLAY_ALPHA,
    _SAM2_MASK_COLOR,
    _SAM2_MASK_OVERLAY_ALPHA,
)
from scripts.vram_utils import log_vram

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
        self.object_mask_raw_dir = self.output_dir / "masks_object_raw"
        self.object_mask_raw_dir.mkdir(parents=True, exist_ok=True)
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
        import gc

        # reset_state() is the only API that actually drops the GPU tensors
        # (maskmem_features, maskmem_pos_enc, obj_ptr, ...) that propagate_in_video
        # accumulates inside inference_state. Plain `del` on the dict leaves them
        # alive because the predictor still holds internal references.
        if self.predictor is not None and self.inference_state is not None:
            log_vram("before SAM2 reset_state")
            try:
                self.predictor.reset_state(self.inference_state)
            except Exception as exc:
                print(f"SAM2 reset_state failed during release: {exc}")
            log_vram("after SAM2 reset_state")
        if self.inference_state is not None:
            del self.inference_state
            self.inference_state = None
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        # gc.collect() MUST run before empty_cache(): nn.Module has cyclic
        # references that prevent deallocation on `del` alone.  gc breaks
        # those cycles, then empty_cache() returns the freed GPU blocks to
        # the CUDA driver.  A second pass catches weak-ref callbacks.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_vram("after SAM2 release")


def _clear_saved_masks(session: SAM2Session) -> None:
    """Remove stale raw/final mask PNGs before a fresh propagation."""
    for mask_dir in (
        session.mask_dir,
        session.object_mask_raw_dir,
        session.ground_mask_dir,
    ):
        for stale_path in mask_dir.glob("*.png"):
            try:
                stale_path.unlink()
            except OSError:
                pass


def _mask_to_bool(mask: np.ndarray) -> np.ndarray:
    """Normalize a propagated mask into a boolean array."""
    return np.asarray(mask).astype(bool)


def _write_mask_png(mask_path: Path, mask: np.ndarray) -> None:
    """Persist a boolean mask as an 8-bit PNG."""
    cv2.imwrite(str(mask_path), (_mask_to_bool(mask).astype(np.uint8) * 255))


def _prediction_to_mask(mask_like) -> np.ndarray:
    """Normalize a predictor output slice into a boolean mask."""
    mask = mask_like > 0
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    if hasattr(mask, "squeeze"):
        mask = mask.squeeze()
    return _mask_to_bool(mask)


def _compose_final_mask(
    object_mask: np.ndarray,
    ground_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Ground-priority composition used by all downstream stages."""
    object_bool = _mask_to_bool(object_mask)
    if ground_mask is None:
        return object_bool
    ground_bool = _mask_to_bool(ground_mask)
    if ground_bool.shape != object_bool.shape:
        raise ValueError(
            "Ground mask shape does not match object mask shape: "
            f"{ground_bool.shape} != {object_bool.shape}"
        )
    return object_bool & ~ground_bool


def _restore_interactive_masks(
    session: SAM2Session,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reset predictor state and rebuild current interactive masks."""
    session.predictor.reset_state(session.inference_state)
    object_mask = None
    ground_mask = None
    if session.click_points:
        object_mask = _run_single_frame_inference_obj(
            session,
            obj_id=1,
            click_points=session.click_points,
            click_labels=session.click_labels,
        )
    if session.ground_click_points:
        ground_mask = _run_single_frame_inference_obj(
            session,
            obj_id=2,
            click_points=session.ground_click_points,
            click_labels=session.ground_click_labels,
        )
    return object_mask, ground_mask


def _save_final_masks(
    session: SAM2Session,
    object_masks: dict[int, np.ndarray],
    ground_masks: dict[int, np.ndarray] | None,
    *,
    num_frames: int,
) -> None:
    """Materialize canonical final masks in ``masks/``."""
    for frame_idx in range(num_frames):
        object_mask = object_masks.get(frame_idx)
        if object_mask is None:
            raise RuntimeError(
                f"Missing propagated object mask for frame {frame_idx:05d}"
            )
        ground_mask = None if ground_masks is None else ground_masks.get(frame_idx)
        final_mask = _compose_final_mask(object_mask, ground_mask)
        _write_mask_png(session.mask_dir / f"{frame_idx:05d}.png", final_mask)


def _propagate_masks(
    session: SAM2Session,
    *,
    current_object_mask: np.ndarray | None = None,
    current_ground_mask: np.ndarray | None = None,
    progress_callback=None,
) -> tuple[Path, Path | None]:
    """Propagate SAM2 masks, save raw masks, and compose canonical finals."""
    if not session.click_points:
        raise ValueError("Add at least one click point first")

    has_ground = bool(session.ground_click_points)
    _clear_saved_masks(session)

    if current_object_mask is None:
        current_object_mask = _run_single_frame_inference_obj(
            session,
            obj_id=1,
            click_points=session.click_points,
            click_labels=session.click_labels,
        )

    object_masks: dict[int, np.ndarray] = {0: _mask_to_bool(current_object_mask)}
    _write_mask_png(session.object_mask_raw_dir / "00000.png", object_masks[0])

    ground_masks: dict[int, np.ndarray] | None = None
    if has_ground:
        if current_ground_mask is None:
            current_ground_mask = _run_single_frame_inference_obj(
                session,
                obj_id=2,
                click_points=session.ground_click_points,
                click_labels=session.ground_click_labels,
            )
        ground_masks = {0: _mask_to_bool(current_ground_mask)}
        _write_mask_png(session.ground_mask_dir / "00000.png", ground_masks[0])

    with torch.inference_mode():
        num_frames = int(session.inference_state["num_frames"])
        masks_tensor = None
        for frame_idx, obj_ids, masks_tensor in session.predictor.propagate_in_video(
            session.inference_state
        ):
            for i, oid in enumerate(obj_ids):
                mask = _prediction_to_mask(masks_tensor[i])
                if oid == 1:
                    object_masks[frame_idx] = mask
                    _write_mask_png(
                        session.object_mask_raw_dir / f"{frame_idx:05d}.png",
                        mask,
                    )
                elif oid == 2 and ground_masks is not None:
                    ground_masks[frame_idx] = mask
                    _write_mask_png(
                        session.ground_mask_dir / f"{frame_idx:05d}.png",
                        mask,
                    )
            if progress_callback:
                progress_callback(frame_idx, num_frames)
        del masks_tensor

    _save_final_masks(session, object_masks, ground_masks, num_frames=num_frames)
    return session.mask_dir, (session.ground_mask_dir if has_ground else None)


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
    return _prediction_to_mask(masks_out[obj_idx])


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
    current_object_mask: np.ndarray | None = None
    current_ground_mask: np.ndarray | None = None
    ui_state = {"ground_phase": False}

    def _status_text(target: str = "object") -> str:
        obj_pos = sum(1 for label in session.click_labels if label == 1)
        obj_neg = sum(1 for label in session.click_labels if label == 0)
        ground_pos = sum(1 for label in session.ground_click_labels if label == 1)
        ground_neg = sum(1 for label in session.ground_click_labels if label == 0)
        phase = "ground" if ui_state["ground_phase"] else "object"
        status = (
            f"Phase: {phase} | Active target: {target}\n"
            f"Object: {obj_pos}+ {obj_neg}- | "
            f"Ground: {ground_pos}+ {ground_neg}-"
        )
        if ui_state["ground_phase"]:
            status += "\nGround clicks will be subtracted from the final mask."
        return status

    def _render_overlay() -> np.ndarray:
        return _create_mask_overlay(
            session.first_frame,
            current_object_mask,
            session.click_points,
            session.click_labels,
            ground_mask=current_ground_mask,
            ground_points=session.ground_click_points,
            ground_labels=session.ground_click_labels,
        )

    def handle_click(image, evt: gr.SelectData, is_negative: bool, target: str):
        """Handle click for the currently selected target."""
        del image
        nonlocal current_object_mask, current_ground_mask

        x, y = evt.index[0], evt.index[1]
        norm_x = x / session.img_w
        norm_y = y / session.img_h
        label = 0 if is_negative else 1
        if target == "ground":
            session.ground_click_points.append((norm_x, norm_y))
            session.ground_click_labels.append(label)
            current_ground_mask = _run_single_frame_inference_obj(
                session,
                obj_id=2,
                click_points=session.ground_click_points,
                click_labels=session.ground_click_labels,
            )
        else:
            session.click_points.append((norm_x, norm_y))
            session.click_labels.append(label)
            current_object_mask = _run_single_frame_inference_obj(
                session,
                obj_id=1,
                click_points=session.click_points,
                click_labels=session.click_labels,
            )

        return _render_overlay(), _status_text(target)

    def undo_click(target: str):
        """Undo the last click for the active target."""
        nonlocal current_object_mask, current_ground_mask

        if target == "ground":
            if session.ground_click_points:
                session.ground_click_points.pop()
                session.ground_click_labels.pop()
            if session.ground_click_points:
                current_ground_mask = _run_single_frame_inference_obj(
                    session,
                    obj_id=2,
                    click_points=session.ground_click_points,
                    click_labels=session.ground_click_labels,
                )
            else:
                current_ground_mask = None
        else:
            if session.click_points:
                session.click_points.pop()
                session.click_labels.pop()
            if session.click_points:
                current_object_mask = _run_single_frame_inference_obj(
                    session,
                    obj_id=1,
                    click_points=session.click_points,
                    click_labels=session.click_labels,
                )
            else:
                current_object_mask = None

        return _render_overlay(), _status_text(target)

    def clear_clicks(target: str):
        """Clear clicks for the active target and rebuild remaining masks."""
        nonlocal current_object_mask, current_ground_mask

        if target == "ground":
            session.ground_click_points.clear()
            session.ground_click_labels.clear()
        else:
            session.click_points.clear()
            session.click_labels.clear()

        current_object_mask, current_ground_mask = _restore_interactive_masks(
            session
        )
        return _render_overlay(), _status_text(target)

    def _signal_completion() -> None:
        import time

        time.sleep(2)
        session.done_event.set()

    def propagate_and_finish(
        progress=gr.Progress(),
        *,
        status_target: str = "ground",
    ):
        """Propagate masks to all frames, then signal completion."""
        nonlocal current_object_mask, current_ground_mask

        try:
            progress(0.3, desc="Propagating masks...")
            num_frames = int(session.inference_state["num_frames"])

            def _progress(frame_idx: int, num_frames: int) -> None:
                pct = 0.3 + 0.6 * (frame_idx + 1) / max(num_frames, 1)
                progress(pct, desc=f"Frame {frame_idx + 1}/{num_frames}")

            mask_dir_result, ground_dir_result = _propagate_masks(
                session,
                current_object_mask=current_object_mask,
                current_ground_mask=current_ground_mask,
                progress_callback=_progress,
            )

            progress(0.95, desc="Releasing SAM2 model...")
            session.release_model()

            progress(1.0, desc="Done!")
            msg = (
                f"Propagated masks to {num_frames} frames\n"
                f"Final masks: {mask_dir_result}\n"
            )
            if ground_dir_result is not None:
                msg += f"Ground masks: {ground_dir_result}\n"
            msg += (
                f"Raw object masks: {session.object_mask_raw_dir}\n\n"
                "Pipeline will continue automatically..."
            )

            threading.Thread(target=_signal_completion, daemon=True).start()
            return (
                _render_overlay(),
                _status_text(status_target),
                gr.update(visible=False, value="object"),
                gr.update(value="Propagating...", interactive=False),
                gr.update(visible=False),
                msg,
            )

        except Exception as e:
            return (
                _render_overlay(),
                f"Error: {e}",
                gr.update(
                    visible=ui_state["ground_phase"],
                    value="ground" if ui_state["ground_phase"] else "object",
                ),
                gr.update(
                    value=(
                        "Confirm Ground & Propagate"
                        if ui_state["ground_phase"]
                        else "Confirm Object"
                    ),
                    interactive=True,
                ),
                gr.update(visible=ui_state["ground_phase"]),
                f"Error: {e}",
            )

    def enter_ground_phase():
        """Advance to ground selection without propagating yet."""
        if not session.click_points:
            return (
                _render_overlay(),
                "Add at least one object click before confirming.",
                gr.update(visible=False, value="object"),
                gr.update(value="Confirm Object", interactive=True),
                gr.update(visible=False),
                "",
            )

        ui_state["ground_phase"] = True
        return (
            _render_overlay(),
            _status_text("ground"),
            gr.update(visible=True, value="ground"),
            gr.update(value="Confirm Ground & Propagate", interactive=True),
            gr.update(visible=True),
            "",
        )

    def confirm_target(target: str, progress=gr.Progress()):
        """Either enter ground phase or propagate, depending on current phase."""
        if ui_state["ground_phase"]:
            return propagate_and_finish(progress, status_target=target)
        return enter_ground_phase()

    def skip_ground_and_propagate(progress=gr.Progress()):
        """Discard ground clicks and finish with object-only propagation."""
        nonlocal current_object_mask, current_ground_mask

        session.ground_click_points.clear()
        session.ground_click_labels.clear()
        current_object_mask, current_ground_mask = _restore_interactive_masks(
            session
        )
        return propagate_and_finish(progress, status_target="object")

    # Build Gradio UI
    with gr.Blocks(title="SAM2 Segmentation") as demo:
        gr.Markdown("# SAM2 Object Segmentation")
        gr.Markdown(
            "1. Click on the object to segment (green=include, red=exclude)\n"
            "2. Click **Confirm Object** to move to ground selection\n"
            "3. Optionally click the ground/contact surface to subtract it\n"
            "4. Click **Confirm Ground & Propagate** or skip ground"
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
                    undo_btn = gr.Button("Undo", size="sm")
                    clear_btn = gr.Button("Clear Clicks", size="sm")
                    negative_mode = gr.Checkbox(
                        label="Negative mode (exclude)", value=False
                    )
                target_selector = gr.Radio(
                    ["object", "ground"],
                    value="object",
                    label="Segmentation target",
                    visible=False,
                )

            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    label="Status", interactive=False, lines=2,
                    value=(
                        f"Frame: {session.img_w}x{session.img_h}, "
                        f"{len(session.frame_files)} frames\n"
                        "Phase: object | Active target: object"
                    ),
                )
                propagate_btn = gr.Button(
                    "Confirm Object", variant="primary", size="lg"
                )
                skip_ground_btn = gr.Button(
                    "Skip Ground & Propagate", variant="secondary", visible=False
                )
                propagate_status = gr.Textbox(
                    label="Propagation Result", interactive=False, lines=6
                )

        image_display.select(
            fn=handle_click,
            inputs=[image_display, negative_mode, target_selector],
            outputs=[image_display, status_text],
        )
        undo_btn.click(
            fn=undo_click,
            inputs=[target_selector],
            outputs=[image_display, status_text],
        )
        clear_btn.click(
            fn=clear_clicks,
            inputs=[target_selector],
            outputs=[image_display, status_text],
        )
        propagate_btn.click(
            fn=confirm_target,
            inputs=[target_selector],
            outputs=[
                image_display,
                status_text,
                target_selector,
                propagate_btn,
                skip_ground_btn,
                propagate_status,
            ],
        )
        skip_ground_btn.click(
            fn=skip_ground_and_propagate,
            outputs=[
                image_display,
                status_text,
                target_selector,
                propagate_btn,
                skip_ground_btn,
                propagate_status,
            ],
        )

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
