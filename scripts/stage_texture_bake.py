"""Stage 7: Texture baking from multiple views.

Estimates camera intrinsics, generates UV atlas, projects textures from
multiple views, and exports OBJ + MTL + PNG.

Adapted from im2pc/host/extract_intrinsics.py and im2pc/host/texture_mesh.py.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from plyfile import PlyData

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for GPU acceleration
    torch = None

ProgressCallback = Callable[[float, str | None], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    progress: float,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    progress_cb(max(0.0, min(100.0, float(progress))), detail)


# ---------------------------------------------------------------------------
# Intrinsics estimation
# ---------------------------------------------------------------------------

def _load_point_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load PLY, return (points (N,3), colors (N,3) in [0,1])."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    colors = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float64) / 255.0
    return points, colors


def _load_poses(path: str) -> tuple[np.ndarray, list[int]]:
    with open(path) as f:
        data = json.load(f)
    poses = np.array(data["poses"], dtype=np.float64)
    frame_indices = data.get("frame_indices")
    if frame_indices is None:
        frame_indices = list(range(len(poses)))
    else:
        frame_indices = [int(i) for i in frame_indices]
    if len(frame_indices) != len(poses):
        print(
            f"Warning: poses={len(poses)} but frame_indices={len(frame_indices)}. "
            "Using positional indices."
        )
        frame_indices = list(range(len(poses)))
    return poses, frame_indices


def _resolve_indexed_file(base_dir: str, idx: int, suffix: str) -> Path:
    """Resolve by numbered filename first, then by sorted positional index."""
    path = Path(base_dir) / f"{idx:05d}{suffix}"
    if path.is_file():
        return path

    files = _list_indexed_files(base_dir, suffix)
    if 0 <= idx < len(files):
        return Path(files[idx])
    raise FileNotFoundError(f"Indexed file not found: {base_dir} idx={idx} suffix={suffix}")


@lru_cache(maxsize=16)
def _list_indexed_files(base_dir: str, suffix: str) -> tuple[str, ...]:
    return tuple(str(p) for p in sorted(Path(base_dir).glob(f"*{suffix}")))


def _load_frame(frames_dir: str, idx: int) -> np.ndarray:
    path = _resolve_indexed_file(frames_dir, idx, ".jpg")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def _load_mask(masks_dir: str, idx: int) -> np.ndarray:
    path = _resolve_indexed_file(masks_dir, idx, ".png")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask > 127


def _make_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _project_points(pts, c2w, K, img_w, img_h):
    """Project 3D→2D. Returns (uv (M,2), valid (N,), depths (N,))."""
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    depths = cam[:, 2]
    valid = depths > 0.01
    pts_v = cam[valid]
    sz = np.maximum(pts_v[:, 2], 1e-10)
    u = K[0, 0] * pts_v[:, 0] / sz + K[0, 2]
    v = K[1, 1] * pts_v[:, 1] / sz + K[1, 2]
    in_bounds = (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)
    uv = np.column_stack([u[in_bounds], v[in_bounds]])
    valid_idx = np.where(valid)[0]
    full_valid = np.zeros(len(pts), dtype=bool)
    full_valid[valid_idx[in_bounds]] = True
    return uv, full_valid, depths


def _estimate_intrinsics(
    points, colors, poses, pose_frame_indices, frames_dir, masks_dir, img_w, img_h,
    num_eval_frames=10, subsample_points=50000,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    """Estimate camera intrinsics via grid search + Nelder-Mead."""
    from scipy.optimize import minimize

    # Subsample
    if len(points) > subsample_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), subsample_points, replace=False)
        pts_sub, col_sub = points[idx], colors[idx]
    else:
        pts_sub, col_sub = points, colors

    n_frames = len(poses)
    eval_pose_indices = np.linspace(0, n_frames - 1, num_eval_frames, dtype=int).tolist()
    print(f"  Evaluating {len(eval_pose_indices)} frames, {len(pts_sub)} points")

    cx_init, cy_init = img_w / 2.0, img_h / 2.0

    def color_score(K):
        total_err, total_cnt = 0.0, 0
        for pose_idx in eval_pose_indices:
            src_idx = int(pose_frame_indices[pose_idx])
            try:
                frame = _load_frame(frames_dir, src_idx)
                mask = _load_mask(masks_dir, src_idx)
            except FileNotFoundError:
                continue
            uv, valid, _ = _project_points(pts_sub, poses[pose_idx], K, img_w, img_h)
            if uv.shape[0] == 0:
                continue
            ui, vi = uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)
            m = mask[vi, ui]
            if m.sum() == 0:
                continue
            diff = frame[vi[m], ui[m]] - col_sub[valid][m]
            total_err += np.mean(diff**2) * m.sum()
            total_cnt += m.sum()
        return -(total_err / total_cnt) if total_cnt > 0 else -1.0

    # Grid search
    best_score, best_fov = -np.inf, 60
    print("  Grid search over FOV...")
    fov_values = list(range(35, 85, 2))
    for idx, fov in enumerate(fov_values):
        fx = img_w / (2.0 * np.tan(np.radians(fov) / 2.0))
        score = color_score(_make_K(fx, fx, cx_init, cy_init))
        if score > best_score:
            best_score, best_fov = score, fov
        if idx % 2 == 0 or idx == len(fov_values) - 1:
            ratio = (idx + 1) / len(fov_values)
            _emit_progress(
                progress_cb,
                ratio * 70.0,
                f"Estimating intrinsics (grid search {idx + 1}/{len(fov_values)})",
            )

    print(f"  Best FOV: {best_fov}° (score={best_score:.6f})")

    # Refine
    fx_init = img_w / (2.0 * np.tan(np.radians(best_fov) / 2.0))

    def objective(params):
        fx, fy, cx, cy = params
        if fx < 100 or fy < 100 or fx > 5000 or fy > 5000:
            return 1.0
        if cx < 0 or cx > img_w or cy < 0 or cy > img_h:
            return 1.0
        return -color_score(_make_K(fx, fy, cx, cy))

    result = minimize(objective, [fx_init, fx_init, cx_init, cy_init],
                      method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 0.5, "fatol": 1e-7, "adaptive": True})
    _emit_progress(progress_cb, 100.0, "Intrinsics optimization complete")

    fx, fy, cx, cy = result.x
    print(f"  Optimized: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    return {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "image_width": img_w, "image_height": img_h,
        "K": _make_K(fx, fy, cx, cy).tolist(),
    }


# ---------------------------------------------------------------------------
# Texture baking
# ---------------------------------------------------------------------------

def _bilinear_sample(img, x, y):
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    fx = (x - x0.astype(np.float64))[:, None]
    fy = (y - y0.astype(np.float64))[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x1]
            + (1 - fx) * fy * img[y1, x0] + fx * fy * img[y1, x1])


def _project_simple(pts, c2w, K):
    """Simple 3D→2D projection. Returns (uv (N,2), depths (N,))."""
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    d = cam[:, 2].copy()
    sz = np.maximum(d, 1e-10)
    u = K[0, 0] * cam[:, 0] / sz + K[0, 2]
    v = K[1, 1] * cam[:, 1] / sz + K[1, 2]
    return np.column_stack([u, v]), d


def _resolve_texture_device(requested: str | None = None) -> str:
    mode = (requested or os.environ.get("TEXTURE_DEVICE", "cuda")).strip().lower()
    if mode == "gpu":
        mode = "cuda"
    if mode not in {"auto", "cpu", "cuda"}:
        print(f"Warning: invalid TEXTURE_DEVICE='{mode}', falling back to cuda")
        mode = "cuda"

    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        print("Warning: TEXTURE_DEVICE=cuda requested but CUDA is unavailable; using CPU")
        return "cpu"
    # auto
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _bilinear_sample_torch(img, x, y):
    h, w = img.shape[:2]
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    x1 = torch.clamp(x0 + 1, 0, w - 1)
    y1 = torch.clamp(y0 + 1, 0, h - 1)
    x0 = torch.clamp(x0, 0, w - 1)
    y0 = torch.clamp(y0, 0, h - 1)
    fx = (x - x0.to(x.dtype)).unsqueeze(1)
    fy = (y - y0.to(y.dtype)).unsqueeze(1)
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    return (
        (1 - fx) * (1 - fy) * c00
        + fx * (1 - fy) * c10
        + (1 - fx) * fy * c01
        + fx * fy * c11
    )


def _project_simple_torch_w2c(pts, w2c, K):
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    d = cam[:, 2].clone()
    sz = torch.clamp(d, min=1e-10)
    u = K[0, 0] * cam[:, 0] / sz + K[0, 2]
    v = K[1, 1] * cam[:, 1] / sz + K[1, 2]
    return torch.stack((u, v), dim=1), d


def bake_texture(
    mesh_ply: str,
    poses_path: str,
    frames_dir: str,
    mask_dir: str,
    output_dir: str,
    tex_size: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Full texture baking pipeline: intrinsics → UV atlas → bake → export.

    Args:
        mesh_ply: Path to mesh PLY file.
        poses_path: Path to camera_poses.json.
        frames_dir: Path to JPEG frames directory.
        mask_dir: Path to mask PNGs directory.
        output_dir: Output directory for OBJ/MTL/PNG.
        tex_size: Texture resolution. Default from env TEXTURE_SIZE or 2048.

    Environment toggles for sharpness/quality:
        TEXTURE_DEVICE (str): 'cuda' (default), 'auto', or 'cpu' projection backend.
        TEXTURE_GPU_CHUNK (int): Texel chunk size per view for CUDA projection.
        TEXTURE_OVERSAMPLE (int): Internal supersampling multiplier (1=off, 2=default).
        TEXTURE_BLEND_MODE (str): 'weighted' (default) or 'max' to pick best single view.
        TEXTURE_MIN_COS (float): Minimum normal·view cosine to accept a sample (default 0.2).
        TEXTURE_ANGLE_EXP (float): Exponent on cosine weight to sharpen view selection.
        TEXTURE_DIST_POW (float): Distance falloff power for view weight (default 1.0).
        TEXTURE_SHARPEN (float): Unsharp mask amount (0=off, 0.1–0.4 typical).

    Returns:
        Path to the output OBJ file.
    """
    import xatlas

    if tex_size is None:
        tex_size = int(os.environ.get("TEXTURE_SIZE", "2048"))

    oversample = max(1, int(os.environ.get("TEXTURE_OVERSAMPLE", "2")))
    blend_mode = os.environ.get("TEXTURE_BLEND_MODE", "weighted").strip().lower()
    use_max_blend = blend_mode == "max"
    min_cos = float(os.environ.get("TEXTURE_MIN_COS", "0.2"))
    angle_exp = float(os.environ.get("TEXTURE_ANGLE_EXP", "2.0"))
    dist_pow = float(os.environ.get("TEXTURE_DIST_POW", "1.0"))
    sharpen_amt = float(os.environ.get("TEXTURE_SHARPEN", "0.15"))
    texture_device = _resolve_texture_device()
    texture_gpu_chunk = max(50_000, int(os.environ.get("TEXTURE_GPU_CHUNK", "750000")))

    tex_res = tex_size * oversample

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_cb, 4.0, "Loading mesh")

    # --- Load mesh ---
    print("Loading mesh...")
    ply = PlyData.read(mesh_ply)
    v = ply["vertex"]
    vertices = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    f = ply["face"]
    faces = np.vstack(f["vertex_indices"]).astype(np.int32)
    print(f"  {len(vertices)} verts, {len(faces)} faces")

    # --- Load camera data ---
    poses, pose_frame_indices = _load_poses(poses_path)

    if len(poses) == 0:
        raise RuntimeError("No poses found in camera_poses.json")

    # Determine image size from first pose-linked frame.
    first_frame = _load_frame(frames_dir, int(pose_frame_indices[0]))
    img_h, img_w = first_frame.shape[:2]
    print(
        f"  Image size: {img_w}x{img_h}, {len(poses)} poses "
        f"(frame range: {min(pose_frame_indices)}..{max(pose_frame_indices)})"
    )

    # --- Estimate intrinsics ---
    # Load denoised PLY for intrinsics estimation (original colored point cloud)
    denoised_ply = output_path / "object_denoised.ply"
    if denoised_ply.exists():
        pc_points, pc_colors = _load_point_cloud(str(denoised_ply))
    else:
        # Fallback: use object.ply
        object_ply = output_path / "object.ply"
        pc_points, pc_colors = _load_point_cloud(str(object_ply))

    print("Estimating camera intrinsics...")
    _emit_progress(progress_cb, 15.0, "Estimating camera intrinsics")
    _intrinsics_progress_last = {"value": 15.0}

    def _intrinsics_progress(local_progress: float, detail: str | None = None) -> None:
        stage_progress = 15.0 + (max(0.0, min(100.0, local_progress)) * 0.35)
        if stage_progress - _intrinsics_progress_last["value"] >= 0.5:
            _emit_progress(progress_cb, stage_progress, detail)
            _intrinsics_progress_last["value"] = stage_progress

    intrinsics = _estimate_intrinsics(
        pc_points, pc_colors, poses, pose_frame_indices, frames_dir, mask_dir, img_w, img_h,
        progress_cb=_intrinsics_progress,
    )
    K = np.array(intrinsics["K"], dtype=np.float64)
    print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    _emit_progress(progress_cb, 52.0, "Camera intrinsics estimated")

    # Save intrinsics
    with open(output_path / "intrinsics.json", "w") as f_out:
        json.dump(intrinsics, f_out, indent=2)

    # --- UV Atlas ---
    print("Generating UV atlas...")
    _emit_progress(progress_cb, 58.0, "Generating UV atlas")
    vmapping, new_faces, uvs = xatlas.parametrize(
        vertices.astype(np.float32), faces.astype(np.uint32)
    )
    new_vertices = vertices[vmapping]
    print(f"  {len(new_vertices)} verts, {len(new_faces)} faces")

    # Face normals
    v0 = new_vertices[new_faces[:, 0]]
    v1 = new_vertices[new_faces[:, 1]]
    v2 = new_vertices[new_faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(norms, 1e-10)

    # --- Build texel mapping ---
    print(
        "Building texel mapping "
        f"({tex_size}x{tex_size}, internal {tex_res}x{tex_res}, oversample x{oversample})"
    )
    print(
        "  Weighting: blend=%s, min_cos=%.2f, angle_exp=%.2f, dist_pow=%.2f, sharpen=%.2f"
        % ("max" if use_max_blend else "weighted", min_cos, angle_exp, dist_pow, sharpen_amt)
    )
    if texture_device == "cuda":
        print(f"  Projection backend: CUDA (chunk={texture_gpu_chunk})")
    else:
        print("  Projection backend: CPU")
    _emit_progress(progress_cb, 62.0, "Building texel mapping")
    face_id_buf = np.full((tex_res, tex_res), -1, dtype=np.int32)
    uv_scaled = uvs * tex_res

    emit_every_face = max(1, len(new_faces) // 40)
    for fi in range(len(new_faces)):
        i0, i1, i2 = new_faces[fi]
        pts = np.array([
            [uv_scaled[i0, 0], uv_scaled[i0, 1]],
            [uv_scaled[i1, 0], uv_scaled[i1, 1]],
            [uv_scaled[i2, 0], uv_scaled[i2, 1]],
        ], dtype=np.int32).reshape(3, 1, 2)
        cv2.fillConvexPoly(face_id_buf, pts, int(fi))
        if fi % emit_every_face == 0 or fi == len(new_faces) - 1:
            ratio = (fi + 1) / max(len(new_faces), 1)
            _emit_progress(
                progress_cb,
                62.0 + ratio * 10.0,
                f"Building texel mapping ({fi + 1}/{len(new_faces)} faces)",
            )

    valid_texels = face_id_buf >= 0
    ys, xs = np.where(valid_texels)
    fids = face_id_buf[ys, xs]

    # Barycentric coords
    px, py = xs + 0.5, ys + 0.5
    fi0, fi1, fi2 = new_faces[fids, 0], new_faces[fids, 1], new_faces[fids, 2]
    uv0, uv1, uv2 = uv_scaled[fi0], uv_scaled[fi1], uv_scaled[fi2]
    denom = (uv1[:, 1] - uv2[:, 1]) * (uv0[:, 0] - uv2[:, 0]) + (uv2[:, 0] - uv1[:, 0]) * (uv0[:, 1] - uv2[:, 1])
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    w0 = ((uv1[:, 1] - uv2[:, 1]) * (px - uv2[:, 0]) + (uv2[:, 0] - uv1[:, 0]) * (py - uv2[:, 1])) / denom
    w1 = ((uv2[:, 1] - uv0[:, 1]) * (px - uv2[:, 0]) + (uv0[:, 0] - uv2[:, 0]) * (py - uv2[:, 1])) / denom
    w2 = 1.0 - w0 - w1

    barys = np.column_stack([w0, w1, w2]).astype(np.float32)

    # 3D positions of texels
    tv0 = new_vertices[new_faces[fids, 0]]
    tv1 = new_vertices[new_faces[fids, 1]]
    tv2 = new_vertices[new_faces[fids, 2]]
    pos3d = barys[:, 0:1] * tv0 + barys[:, 1:2] * tv1 + barys[:, 2:3] * tv2
    normals = face_normals[fids]

    n_valid = len(pos3d)
    print(f"  {n_valid} valid texels")

    # --- Multi-view texture projection ---
    _emit_progress(progress_cb, 72.0, "Projecting textures from input views")
    emit_every_view = max(1, len(poses) // 40)

    if texture_device == "cuda":
        color_sum_t = torch.zeros((n_valid, 3), dtype=torch.float32, device="cuda")
        weight_sum_t = torch.zeros(n_valid, dtype=torch.float32, device="cuda")
        pos3d_t = torch.as_tensor(pos3d, dtype=torch.float32, device="cuda")
        normals_t = torch.as_tensor(normals, dtype=torch.float32, device="cuda")
        K_t = torch.as_tensor(K, dtype=torch.float32, device="cuda")

        for vidx in range(len(poses)):
            src_idx = int(pose_frame_indices[vidx])
            c2w = poses[vidx]

            try:
                frame = _load_frame(frames_dir, src_idx)
                mask_bool = _load_mask(mask_dir, src_idx)
            except FileNotFoundError:
                continue

            frame_t = torch.as_tensor(frame, dtype=torch.float32, device="cuda")
            mask_t = torch.as_tensor(mask_bool, dtype=torch.bool, device="cuda")
            c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device="cuda")
            w2c_t = torch.linalg.inv(c2w_t)
            cam_pos_t = c2w_t[:3, 3]

            n_ok_view = 0
            for start in range(0, n_valid, texture_gpu_chunk):
                end = min(start + texture_gpu_chunk, n_valid)
                pos_chunk = pos3d_t[start:end]
                normal_chunk = normals_t[start:end]

                view_dirs = cam_pos_t.unsqueeze(0) - pos_chunk
                dists = torch.linalg.norm(view_dirs, dim=1)
                view_dirs_n = view_dirs / torch.clamp(dists.unsqueeze(1), min=1e-10)
                cos_angle = torch.sum(normal_chunk * view_dirs_n, dim=1)
                facing = cos_angle > min_cos
                if not torch.any(facing).item():
                    continue

                facing_indices = torch.nonzero(facing, as_tuple=False).squeeze(1)
                uv2d, depths = _project_simple_torch_w2c(pos_chunk[facing], w2c_t, K_t)
                px_proj = uv2d[:, 0]
                py_proj = uv2d[:, 1]

                ok = (
                    (depths > 0.01)
                    & (px_proj >= 0)
                    & (px_proj < img_w - 1)
                    & (py_proj >= 0)
                    & (py_proj < img_h - 1)
                )
                if not torch.any(ok).item():
                    continue

                pxi = torch.clamp(px_proj.to(torch.int64), 0, img_w - 1)
                pyi = torch.clamp(py_proj.to(torch.int64), 0, img_h - 1)
                ok = ok & mask_t[pyi, pxi]
                if not torch.any(ok).item():
                    continue

                colors_sampled = _bilinear_sample_torch(frame_t, px_proj[ok], py_proj[ok])

                # View weight: sharper angle preference + distance falloff
                ang = torch.clamp(cos_angle[facing][ok], min=0.0).pow(angle_exp)
                dist_term = torch.clamp(dists[facing][ok], min=1e-6).pow(dist_pow)
                w = ang / dist_term
                final_indices = facing_indices[ok] + start
                n_ok_view += int(ok.sum().item())

                if use_max_blend:
                    # Winner-takes-all per texel
                    better = w > weight_sum_t[final_indices]
                    if torch.any(better).item():
                        better_idx = final_indices[better]
                        weight_sum_t[better_idx] = w[better]
                        color_sum_t[better_idx] = colors_sampled[better]
                else:
                    color_sum_t.index_add_(0, final_indices, colors_sampled * w.unsqueeze(1))
                    weight_sum_t.index_add_(0, final_indices, w)

            if vidx % 5 == 0:
                print(f"  View {vidx+1}/{len(poses)}: {n_ok_view} texels")
            if vidx % emit_every_view == 0 or vidx == len(poses) - 1:
                ratio = (vidx + 1) / max(len(poses), 1)
                _emit_progress(
                    progress_cb,
                    72.0 + ratio * 20.0,
                    f"Projecting textures ({vidx + 1}/{len(poses)} views)",
                )

        has_color_t = weight_sum_t > 0
        if not use_max_blend and torch.any(has_color_t).item():
            color_sum_t[has_color_t] = color_sum_t[has_color_t] / weight_sum_t[has_color_t].unsqueeze(1)

        color_sum = color_sum_t.detach().cpu().numpy().astype(np.float64, copy=False)
        weight_sum = weight_sum_t.detach().cpu().numpy().astype(np.float64, copy=False)
        del color_sum_t, weight_sum_t, pos3d_t, normals_t, K_t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        color_sum = np.zeros((n_valid, 3), dtype=np.float64)
        weight_sum = np.zeros(n_valid, dtype=np.float64)

        for vidx in range(len(poses)):
            src_idx = int(pose_frame_indices[vidx])
            c2w = poses[vidx]
            cam_pos = c2w[:3, 3]

            try:
                frame = _load_frame(frames_dir, src_idx)
            except FileNotFoundError:
                continue

            try:
                mask_bool = _load_mask(mask_dir, src_idx)
            except FileNotFoundError:
                continue

            # View direction
            view_dirs = cam_pos - pos3d
            dists = np.linalg.norm(view_dirs, axis=1, keepdims=True)
            view_dirs_n = view_dirs / np.maximum(dists, 1e-10)
            cos_angle = np.sum(normals * view_dirs_n, axis=1)
            facing = cos_angle > min_cos

            if facing.sum() == 0:
                continue

            uv2d, depths = _project_simple(pos3d[facing], c2w, K)
            px_proj, py_proj = uv2d[:, 0], uv2d[:, 1]

            ok = (depths > 0.01) & (px_proj >= 0) & (px_proj < img_w - 1) & (py_proj >= 0) & (py_proj < img_h - 1)

            pxi = np.clip(px_proj.astype(np.int32), 0, img_w - 1)
            pyi = np.clip(py_proj.astype(np.int32), 0, img_h - 1)
            ok = ok & mask_bool[pyi, pxi]

            n_ok = ok.sum()
            if n_ok == 0:
                continue

            colors_sampled = _bilinear_sample(frame, px_proj[ok], py_proj[ok])

            # View weight: sharper angle preference + distance falloff
            ang = np.maximum(cos_angle[facing][ok], 0.0) ** angle_exp
            dist_term = np.power(np.maximum(dists[facing][ok, 0], 1e-6), dist_pow)
            w = ang / dist_term

            if use_max_blend:
                # Winner-takes-all per texel
                facing_indices = np.where(facing)[0]
                final_indices = facing_indices[ok]
                better = w > weight_sum[final_indices]
                if np.any(better):
                    weight_sum[final_indices[better]] = w[better]
                    color_sum[final_indices[better]] = colors_sampled[better]
            else:
                facing_indices = np.where(facing)[0]
                final_indices = facing_indices[ok]
                color_sum[final_indices] += colors_sampled * w[:, None]
                weight_sum[final_indices] += w

            if vidx % 5 == 0:
                print(f"  View {vidx+1}/{len(poses)}: {n_ok} texels")
            if vidx % emit_every_view == 0 or vidx == len(poses) - 1:
                ratio = (vidx + 1) / max(len(poses), 1)
                _emit_progress(
                    progress_cb,
                    72.0 + ratio * 20.0,
                    f"Projecting textures ({vidx + 1}/{len(poses)} views)",
                )

    # Normalize
    has_color = weight_sum > 0
    if not use_max_blend:
        color_sum[has_color] /= weight_sum[has_color, None]

    texture = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    texture[ys, xs] = color_sum.astype(np.float32)

    cov = has_color.sum()
    print(f"Texture coverage: {cov}/{n_valid} ({100*cov/max(n_valid,1):.1f}%)")

    # --- Seam padding ---
    print("Padding seams...")
    _emit_progress(progress_cb, 92.0, "Padding UV seams")
    valid_pad = (face_id_buf >= 0) & (np.sum(texture, axis=2) > 0)
    result_tex = texture.copy()
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    for iter_idx in range(8):
        empty = ~valid_pad
        if not np.any(empty):
            break
        for c in range(3):
            ns = cv2.filter2D(result_tex[:, :, c], -1, kernel, borderType=cv2.BORDER_CONSTANT)
            nc = cv2.filter2D(valid_pad.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
            fill = empty & (nc > 0)
            if np.any(fill):
                result_tex[:, :, c][fill] = ns[fill] / nc[fill]
                valid_pad[fill] = True
        _emit_progress(
            progress_cb,
            92.0 + ((iter_idx + 1) / 8.0) * 5.0,
            f"Padding seams ({iter_idx + 1}/8)",
        )

    # Downsample from supersample grid and optionally sharpen
    final_tex = result_tex
    if oversample > 1:
        final_tex = cv2.resize(
            result_tex, (tex_size, tex_size), interpolation=cv2.INTER_AREA
        )
    if sharpen_amt > 0:
        blur = cv2.GaussianBlur(final_tex, (0, 0), sigmaX=1.0)
        final_tex = np.clip(final_tex + sharpen_amt * (final_tex - blur), 0.0, 1.0)

    # --- Export OBJ + MTL + PNG ---
    basename = "textured_mesh"
    obj_path = output_path / f"{basename}.obj"
    mtl_path = output_path / f"{basename}.mtl"
    tex_path = output_path / "texture.png"

    # Texture PNG (flip V for OBJ convention)
    tex_u8 = np.clip(final_tex * 255, 0, 255).astype(np.uint8)
    tex_bgr = cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR)[::-1]
    cv2.imwrite(str(tex_path), tex_bgr)
    print(f"Saved: {tex_path}")
    _emit_progress(progress_cb, 98.0, "Exporting textured mesh")

    # MTL
    with open(mtl_path, "w") as f_out:
        f_out.write(
            f"newmtl material_0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n"
            f"Ks 0.0 0.0 0.0\nd 1.0\nillum 1\nmap_Kd texture.png\n"
        )
    print(f"Saved: {mtl_path}")

    # OBJ
    with open(obj_path, "w") as f_out:
        f_out.write(f"mtllib {basename}.mtl\nusemtl material_0\n\n")
        for vert in new_vertices:
            f_out.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
        for uv in uvs:
            f_out.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in new_faces:
            a, b, c = face + 1
            f_out.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    print(f"Saved: {obj_path}")
    _emit_progress(progress_cb, 100.0, "Texture stage complete")

    return obj_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Texture baking")
    parser.add_argument("mesh_ply", help="Mesh PLY file")
    parser.add_argument("--poses", required=True, help="camera_poses.json")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", default="/data/output")
    args = parser.parse_args()

    bake_texture(args.mesh_ply, args.poses, args.frames_dir, args.masks_dir, args.output_dir)
