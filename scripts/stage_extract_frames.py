"""Stage 1: Extract frames from video as JPEGs.

Adapted from im2pc/host/pi3x_sam2_cli.py::extract_frames().
"""

import os
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int | None = None,
    max_frames: int | None = None,
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

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_path = frames_dir / f"{saved_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1
            if saved_idx >= max_frames:
                break
        frame_idx += 1

    cap.release()
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
