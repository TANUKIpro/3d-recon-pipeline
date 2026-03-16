"""Stage 2: COLMAP Structure-from-Motion.

Runs COLMAP feature extraction, matching, and sparse reconstruction to produce
camera poses and a sparse point cloud from extracted video frames.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path


def run_colmap_sfm(
    frames_dir: str,
    output_dir: str,
    matcher: str = "sequential",
    max_features: int = 8192,
    image_size: int = 1024,
    use_gpu: bool = False,
    dsp_sift: bool = True,
    progress_cb=None,
    cancel_cb=None,
    register_process=None,
    unregister_process=None,
) -> tuple[str, str]:
    """COLMAP SfM: feature extraction → matching → mapping → pose export.

    Returns (poses_json_path, colmap_sparse_dir).
    """
    frames = Path(frames_dir)
    out = Path(output_dir)
    colmap_workspace = out / "colmap_workspace"
    colmap_db = colmap_workspace / "database.db"
    colmap_sparse = out / "colmap_sparse"

    colmap_workspace.mkdir(parents=True, exist_ok=True)
    colmap_sparse.mkdir(parents=True, exist_ok=True)

    def _report(pct: float, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if cancel_cb:
            cancel_cb()

    def _run_colmap(args: list[str], step_name: str) -> None:
        print(f"COLMAP {step_name}: {' '.join(args)}")
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        if register_process:
            register_process(proc)
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print(f"  [COLMAP] {line}")
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"COLMAP {step_name} failed (exit {proc.returncode})")
        finally:
            if unregister_process:
                unregister_process(proc)

    # Step 1: Feature extraction
    _report(5.0, "Running COLMAP feature extraction")
    extract_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(colmap_db),
        "--image_path", str(frames),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", str(max_features),
        "--SiftExtraction.max_image_size", str(image_size),
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
    ]
    if dsp_sift:
        extract_cmd += [
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ]
    _run_colmap(extract_cmd, "feature_extractor")

    if cancel_cb:
        cancel_cb()

    # Step 2: Feature matching
    _report(25.0, f"Running COLMAP {matcher} matcher")
    matcher_cmd = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    _run_colmap([
        "colmap", matcher_cmd,
        "--database_path", str(colmap_db),
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
    ], matcher_cmd)

    if cancel_cb:
        cancel_cb()

    # Step 3: Sparse reconstruction (mapper)
    _report(50.0, "Running COLMAP sparse reconstruction")
    _run_colmap([
        "colmap", "mapper",
        "--database_path", str(colmap_db),
        "--image_path", str(frames),
        "--output_path", str(colmap_sparse),
    ], "mapper")

    if cancel_cb:
        cancel_cb()

    # Step 4: Convert poses to camera_poses.json
    _report(80.0, "Exporting camera poses")

    # Find the reconstruction directory (mapper outputs to 0/, 1/, etc.)
    recon_dirs = sorted(colmap_sparse.iterdir())
    recon_dir = None
    for d in recon_dirs:
        if d.is_dir() and (d / "images.bin").exists():
            recon_dir = d
            break
    if recon_dir is None:
        raise RuntimeError("COLMAP mapper produced no valid reconstruction")

    poses = _read_colmap_poses(recon_dir)
    poses_path = str(out / "camera_poses.json")
    with open(poses_path, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    print(f"Exported {len(poses)} camera poses to {poses_path}")

    # Step 5: Export sparse point cloud as PLY
    _report(90.0, "Exporting sparse point cloud")
    sparse_ply = out / "colmap_sparse_points.ply"
    _run_colmap([
        "colmap", "model_converter",
        "--input_path", str(recon_dir),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY",
    ], "model_converter")

    _report(100.0, "COLMAP SfM complete")
    return poses_path, str(colmap_sparse)


def _read_colmap_poses(recon_dir: Path) -> list[dict]:
    """Read COLMAP binary images.bin and cameras.bin to produce pose dicts.

    Each pose dict: {frame_name, transform_matrix (4x4 row-major list)}.
    COLMAP stores world-to-camera transforms (qw,qx,qy,qz,tx,ty,tz).
    We convert to camera-to-world (c2w) 4x4 matrices.
    """
    images_path = recon_dir / "images.bin"
    images = _read_images_binary(images_path)

    poses = []
    for img in sorted(images.values(), key=lambda x: x["name"]):
        qw, qx, qy, qz = img["qvec"]
        tx, ty, tz = img["tvec"]

        # Quaternion to rotation matrix (world-to-camera)
        r = _qvec2rotmat(qw, qx, qy, qz)

        # World-to-camera → camera-to-world
        r_inv = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ]
        t_inv = [
            -(r_inv[0][0] * tx + r_inv[0][1] * ty + r_inv[0][2] * tz),
            -(r_inv[1][0] * tx + r_inv[1][1] * ty + r_inv[1][2] * tz),
            -(r_inv[2][0] * tx + r_inv[2][1] * ty + r_inv[2][2] * tz),
        ]

        c2w = [
            [r_inv[0][0], r_inv[0][1], r_inv[0][2], t_inv[0]],
            [r_inv[1][0], r_inv[1][1], r_inv[1][2], t_inv[1]],
            [r_inv[2][0], r_inv[2][1], r_inv[2][2], t_inv[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

        poses.append({
            "frame_name": img["name"],
            "transform_matrix": c2w,
        })

    return poses


def _qvec2rotmat(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return [
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qx * qw,
        ],
        [
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ]


def _read_images_binary(path: Path) -> dict[int, dict]:
    """Read COLMAP images.bin (binary format)."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # Read image name (null-terminated)
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            name = "".join(name_chars)
            # Read 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Each point2D: x, y, point3d_id
            f.read(num_points2d * 24)  # 2 doubles + 1 int64 = 24 bytes

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images
