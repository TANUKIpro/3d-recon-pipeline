"""Stage 6: Texture baking from multiple views.

Estimates camera intrinsics, generates UV atlas, projects textures from
multiple views, and exports OBJ + MTL + PNG.

Adapted from im2pc/host/extract_intrinsics.py and im2pc/host/texture_mesh.py.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData


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


def _load_poses(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f)["poses"], dtype=np.float64)


def _load_frame(frames_dir: str, idx: int) -> np.ndarray:
    path = Path(frames_dir) / f"{idx:05d}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def _load_mask(masks_dir: str, idx: int) -> np.ndarray:
    path = Path(masks_dir) / f"{idx:05d}.png"
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
    points, colors, poses, frames_dir, masks_dir, img_w, img_h,
    num_eval_frames=10, subsample_points=50000,
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
    frame_indices = np.linspace(0, n_frames - 1, num_eval_frames, dtype=int).tolist()
    print(f"  Evaluating {len(frame_indices)} frames, {len(pts_sub)} points")

    cx_init, cy_init = img_w / 2.0, img_h / 2.0

    def color_score(K):
        total_err, total_cnt = 0.0, 0
        for idx in frame_indices:
            frame = _load_frame(frames_dir, idx)
            mask = _load_mask(masks_dir, idx)
            uv, valid, _ = _project_points(pts_sub, poses[idx], K, img_w, img_h)
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
    for fov in range(35, 85, 2):
        fx = img_w / (2.0 * np.tan(np.radians(fov) / 2.0))
        score = color_score(_make_K(fx, fx, cx_init, cy_init))
        if score > best_score:
            best_score, best_fov = score, fov

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


def bake_texture(
    mesh_ply: str,
    poses_path: str,
    frames_dir: str,
    mask_dir: str,
    output_dir: str,
    tex_size: int | None = None,
) -> Path:
    """Full texture baking pipeline: intrinsics → UV atlas → bake → export.

    Args:
        mesh_ply: Path to mesh PLY file.
        poses_path: Path to camera_poses.json.
        frames_dir: Path to JPEG frames directory.
        mask_dir: Path to mask PNGs directory.
        output_dir: Output directory for OBJ/MTL/PNG.
        tex_size: Texture resolution. Default from env TEXTURE_SIZE or 2048.

    Returns:
        Path to the output OBJ file.
    """
    import xatlas

    if tex_size is None:
        tex_size = int(os.environ.get("TEXTURE_SIZE", "2048"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load mesh ---
    print("Loading mesh...")
    ply = PlyData.read(mesh_ply)
    v = ply["vertex"]
    vertices = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    f = ply["face"]
    faces = np.vstack(f["vertex_indices"]).astype(np.int32)
    print(f"  {len(vertices)} verts, {len(faces)} faces")

    # --- Load camera data ---
    poses = _load_poses(poses_path)

    # Determine image size from first frame
    first_frame = cv2.imread(str(Path(frames_dir) / "00000.jpg"))
    if first_frame is None:
        raise FileNotFoundError("Cannot load first frame")
    img_h, img_w = first_frame.shape[:2]
    print(f"  Image size: {img_w}x{img_h}, {len(poses)} poses")

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
    intrinsics = _estimate_intrinsics(
        pc_points, pc_colors, poses, frames_dir, mask_dir, img_w, img_h,
    )
    K = np.array(intrinsics["K"], dtype=np.float64)
    print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    # Save intrinsics
    with open(output_path / "intrinsics.json", "w") as f_out:
        json.dump(intrinsics, f_out, indent=2)

    # --- UV Atlas ---
    print("Generating UV atlas...")
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
    print(f"Building texel mapping ({tex_size}x{tex_size})...")
    face_id_buf = np.full((tex_size, tex_size), -1, dtype=np.int32)
    uv_scaled = uvs * tex_size

    for fi in range(len(new_faces)):
        i0, i1, i2 = new_faces[fi]
        pts = np.array([
            [uv_scaled[i0, 0], uv_scaled[i0, 1]],
            [uv_scaled[i1, 0], uv_scaled[i1, 1]],
            [uv_scaled[i2, 0], uv_scaled[i2, 1]],
        ], dtype=np.int32).reshape(3, 1, 2)
        cv2.fillConvexPoly(face_id_buf, pts, int(fi))

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
    color_sum = np.zeros((n_valid, 3), dtype=np.float64)
    weight_sum = np.zeros(n_valid, dtype=np.float64)

    for vidx in range(len(poses)):
        c2w = poses[vidx]
        cam_pos = c2w[:3, 3]

        frame = cv2.imread(str(Path(frames_dir) / f"{vidx:05d}.jpg"))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        mask = cv2.imread(str(Path(mask_dir) / f"{vidx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_bool = mask > 127

        # View direction
        view_dirs = cam_pos - pos3d
        dists = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs_n = view_dirs / np.maximum(dists, 1e-10)
        cos_angle = np.sum(normals * view_dirs_n, axis=1)
        facing = cos_angle > 0.1

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
        w = cos_angle[facing][ok]

        facing_indices = np.where(facing)[0]
        final_indices = facing_indices[ok]
        color_sum[final_indices] += colors_sampled * w[:, None]
        weight_sum[final_indices] += w

        if vidx % 5 == 0:
            print(f"  View {vidx+1}/{len(poses)}: {n_ok} texels")

    # Normalize
    has_color = weight_sum > 0
    color_sum[has_color] /= weight_sum[has_color, None]

    texture = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    texture[ys, xs] = color_sum.astype(np.float32)

    cov = has_color.sum()
    print(f"Texture coverage: {cov}/{n_valid} ({100*cov/max(n_valid,1):.1f}%)")

    # --- Seam padding ---
    print("Padding seams...")
    valid_pad = (face_id_buf >= 0) & (np.sum(texture, axis=2) > 0)
    result_tex = texture.copy()
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    for _ in range(8):
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

    # --- Export OBJ + MTL + PNG ---
    basename = "textured_mesh"
    obj_path = output_path / f"{basename}.obj"
    mtl_path = output_path / f"{basename}.mtl"
    tex_path = output_path / "texture.png"

    # Texture PNG (flip V for OBJ convention)
    tex_u8 = np.clip(result_tex * 255, 0, 255).astype(np.uint8)
    tex_bgr = cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR)[::-1]
    cv2.imwrite(str(tex_path), tex_bgr)
    print(f"Saved: {tex_path}")

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
