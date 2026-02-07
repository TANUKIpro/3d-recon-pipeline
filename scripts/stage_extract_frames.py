"""Stage 1: Extract frames from video as JPEGs.

Adapted from im2pc/host/pi3x_sam2_cli.py::extract_frames().
"""

import os
import subprocess
from pathlib import Path
from typing import Callable

import cv2

ProgressCallback = Callable[[float, str | None], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


def _detect_rotation(cap: cv2.VideoCapture, video_path: str) -> int:
    """Detect video rotation metadata in degrees (0, 90, 180, 270).

    Checks cv2 properties first, falls back to ffprobe.
    """
    # Check if OpenCV already auto-rotates
    try:
        auto_orient = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)
        if auto_orient == 1.0:
            return 0  # OpenCV handles rotation automatically
    except Exception:
        pass

    # Try cv2 orientation metadata property
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if rotation in (90, 180, 270):
            return rotation
    except Exception:
        pass

    # Fallback: ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream_tags=rotate",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            angle = int(result.stdout.strip())
            if angle in (90, 180, 270):
                return angle
    except Exception:
        pass

    return 0


def _rotation_to_cv2_code(rotation: int):
    """Convert rotation degrees to cv2.rotate code, or None if no rotation needed."""
    if rotation == 90:
        return cv2.ROTATE_90_CLOCKWISE
    elif rotation == 180:
        return cv2.ROTATE_180
    elif rotation == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int | None = None,
    max_frames: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Extract frames from a video at regular intervals.

    Args:
        video_path: Path to input video (.mp4).
        output_dir: Parent output directory.
        frame_interval: Extract every N-th frame. Default from env FRAME_INTERVAL or 10.
        max_frames: Maximum frames to extract. Default from env MAX_FRAMES or 50.

    Returns:
        Path to the frames directory containing JPEG files.
    """
    if frame_interval is None:
        frame_interval = int(os.environ.get("FRAME_INTERVAL", "10"))
    if max_frames is None:
        max_frames = int(os.environ.get("MAX_FRAMES", "50"))

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps")
    _emit_progress(progress_cb, 2.0, "Inspecting video metadata")

    # Detect and handle rotation metadata
    rotation = _detect_rotation(cap, str(video_path))
    rotate_code = _rotation_to_cv2_code(rotation)
    if rotation:
        print(f"Detected rotation: {rotation}\u00b0 \u2014 will correct frames")

    frame_idx = 0
    saved_idx = 0
    expected_reads = total_frames
    if frame_interval > 0 and max_frames > 0:
        expected_reads = min(total_frames, (max_frames - 1) * frame_interval + 1)
    expected_reads = max(expected_reads, 1)
    emit_every = max(1, expected_reads // 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            frame_path = frames_dir / f"{saved_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1
            if saved_idx >= max_frames:
                reads = frame_idx + 1
                pct = min(99.0, (reads / expected_reads) * 100.0)
                _emit_progress(
                    progress_cb,
                    pct,
                    f"Extracting frames ({saved_idx}/{max_frames})",
                )
                break
        reads = frame_idx + 1
        if reads % emit_every == 0 or reads >= expected_reads:
            pct = min(99.0, (reads / expected_reads) * 100.0)
            _emit_progress(
                progress_cb,
                pct,
                f"Extracting frames ({saved_idx}/{max_frames})",
            )
        frame_idx += 1

    cap.release()
    _emit_progress(progress_cb, 100.0, f"Extracted {saved_idx} frames")
    print(f"Extracted {saved_idx} frames to {frames_dir}")
    return frames_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output-dir", default="/data/output")
    parser.add_argument("--frame-interval", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.frame_interval, args.max_frames)
