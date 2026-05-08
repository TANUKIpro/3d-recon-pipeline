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

from scripts.output_layout import (
    camera_poses_path,
    colmap_phase_dir,
    colmap_sparse_dir as _colmap_sparse_dir,
    colmap_sparse_points_path,
    colmap_workspace_dir,
    intrinsics_path,
)


def run_colmap_sfm(
    frames_dir: str,
    output_dir: str,
    matcher: str = "sequential",
    max_features: int = 32768,
    image_size: int = 1024,
    use_gpu: bool = False,
    dsp_sift: bool = False,
    first_octave: int = -1,
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
    colmap_phase = colmap_phase_dir(out)
    colmap_workspace = colmap_workspace_dir(out)
    colmap_db = colmap_workspace / "database.db"
    colmap_sparse = _colmap_sparse_dir(out)

    colmap_phase.mkdir(parents=True, exist_ok=True)
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
        "--SiftExtraction.first_octave", str(first_octave),
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
    poses_path = str(camera_poses_path(out))
    with open(poses_path, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    print(f"Exported {len(poses)} camera poses to {poses_path}")

    # Export camera intrinsics so the texture stage can skip its grid search.
    try:
        intrinsics = _read_colmap_intrinsics(recon_dir)
    except Exception as exc:
        print(f"  Warning: failed to export COLMAP intrinsics ({exc}); texture stage will re-estimate")
    else:
        if intrinsics is not None:
            intrinsics_file = intrinsics_path(out)
            with open(intrinsics_file, "w", encoding="utf-8") as f:
                json.dump(intrinsics, f, indent=2)
            print(
                f"Exported intrinsics to {intrinsics_file}: "
                f"fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
                f"cx={intrinsics['cx']:.1f} cy={intrinsics['cy']:.1f}"
            )

    # Step 5: Export sparse point cloud as PLY
    _report(90.0, "Exporting sparse point cloud")
    sparse_ply = colmap_sparse_points_path(out)
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


# COLMAP camera model IDs → (name, num_params, (fx_idx, fy_idx, cx_idx, cy_idx)).
# fy_idx == fx_idx when the model shares a single focal length.
_COLMAP_CAMERA_MODELS: dict[int, tuple[str, int, tuple[int, int, int, int]]] = {
    0: ("SIMPLE_PINHOLE", 3, (0, 0, 1, 2)),
    1: ("PINHOLE", 4, (0, 1, 2, 3)),
    2: ("SIMPLE_RADIAL", 4, (0, 0, 1, 2)),
    3: ("RADIAL", 5, (0, 0, 1, 2)),
    4: ("OPENCV", 8, (0, 1, 2, 3)),
    5: ("OPENCV_FISHEYE", 8, (0, 1, 2, 3)),
    6: ("FULL_OPENCV", 12, (0, 1, 2, 3)),
    7: ("FOV", 5, (0, 1, 2, 3)),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, (0, 0, 1, 2)),
    9: ("RADIAL_FISHEYE", 5, (0, 0, 1, 2)),
    10: ("THIN_PRISM_FISHEYE", 12, (0, 1, 2, 3)),
}


# Distortion-parameter indices (within ``cam["params"]``) for OpenCV's 5-element
# ``(k1, k2, p1, p2, k3)`` coefficient layout. Missing slots stay at 0. Models
# not in this table contribute no distortion (e.g., PINHOLE).
_COLMAP_DISTORTION_INDICES: dict[int, tuple[int, int, int, int, int]] = {
    # SIMPLE_RADIAL: (f, cx, cy, k1)
    2: (3, -1, -1, -1, -1),
    # RADIAL: (f, cx, cy, k1, k2)
    3: (3, 4, -1, -1, -1),
    # OPENCV: (fx, fy, cx, cy, k1, k2, p1, p2)
    4: (4, 5, 6, 7, -1),
    # FULL_OPENCV: (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6) — k4..k6 unused by texture stage
    6: (4, 5, 6, 7, 8),
}


def _read_cameras_binary(path: Path) -> dict[int, dict]:
    """Read COLMAP cameras.bin and return {camera_id: {model, width, height, params}}."""
    cameras: dict[int, dict] = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            if model_id not in _COLMAP_CAMERA_MODELS:
                raise RuntimeError(
                    f"Unsupported COLMAP camera model_id={model_id} for camera {camera_id}"
                )
            _model_name, num_params, _idx = _COLMAP_CAMERA_MODELS[model_id]
            params = struct.unpack(f"<{num_params}d", f.read(num_params * 8))

            cameras[camera_id] = {
                "model_id": model_id,
                "width": int(width),
                "height": int(height),
                "params": list(params),
            }
    return cameras


def _read_colmap_intrinsics(recon_dir: Path) -> dict | None:
    """Extract fx/fy/cx/cy/distortion from the dominant COLMAP camera.

    Returns a dict in the same schema that ``scripts/texture/intrinsics.py``
    writes, or None if the cameras file is missing.

    The returned dict includes ``dist_coeffs`` — the OpenCV-layout 5-vector
    ``[k1, k2, p1, p2, k3]`` extracted from the COLMAP camera model. The
    texture stage uses these to undistort frames at load time so that its
    pure-pinhole projection matches the bundle-adjusted geometry. iPhone
    captures typically end up with SIMPLE_RADIAL k1≈0.01–0.05, which causes
    >2 px sub-pixel mismatch on cylindrical objects when ignored.
    """
    cameras_path = recon_dir / "cameras.bin"
    if not cameras_path.exists():
        return None

    cameras = _read_cameras_binary(cameras_path)
    if not cameras:
        return None

    images_path = recon_dir / "images.bin"
    dominant_camera_id: int | None = None
    if images_path.exists():
        images = _read_images_binary(images_path)
        counts: dict[int, int] = {}
        for img in images.values():
            cid = int(img["camera_id"])
            counts[cid] = counts.get(cid, 0) + 1
        if counts:
            dominant_camera_id = max(counts.items(), key=lambda kv: kv[1])[0]

    if dominant_camera_id is None or dominant_camera_id not in cameras:
        dominant_camera_id = next(iter(cameras))

    cam = cameras[dominant_camera_id]
    model_id = cam["model_id"]
    model_name, _num_params, (fx_i, fy_i, cx_i, cy_i) = _COLMAP_CAMERA_MODELS[model_id]
    params = cam["params"]
    fx = float(params[fx_i])
    fy = float(params[fy_i])
    cx = float(params[cx_i])
    cy = float(params[cy_i])
    img_w = int(cam["width"])
    img_h = int(cam["height"])

    dist = [0.0, 0.0, 0.0, 0.0, 0.0]
    if model_id in _COLMAP_DISTORTION_INDICES:
        for slot, idx in enumerate(_COLMAP_DISTORTION_INDICES[model_id]):
            if 0 <= idx < len(params):
                dist[slot] = float(params[idx])

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_width": img_w,
        "image_height": img_h,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "dist_coeffs": dist,
        "distortion_model": model_name,
        "source": f"colmap:{model_name}",
    }


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
