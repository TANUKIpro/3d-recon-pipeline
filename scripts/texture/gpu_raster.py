"""GPU-accelerated rasterization via nvdiffrast + PyTorch."""

from __future__ import annotations

import numpy as np

# Lazy-loaded modules
_torch = None
_dr = None
_cuda_ctx = None
_deps_checked = False
_deps_available = False


def _ensure_deps() -> bool:
    """Lazily import torch and nvdiffrast. Returns True if available."""
    global _torch, _dr, _deps_checked, _deps_available
    if _deps_checked:
        return _deps_available
    _deps_checked = True
    try:
        import torch
        _torch = torch
        if not torch.cuda.is_available():
            _deps_available = False
            return False
        import nvdiffrast.torch as dr
        _dr = dr
        _deps_available = True
    except ImportError:
        _deps_available = False
    return _deps_available


def _get_cuda_context():
    """Return a singleton RasterizeCudaContext."""
    global _cuda_ctx
    if _cuda_ctx is None:
        if not _ensure_deps():
            raise RuntimeError("nvdiffrast/CUDA not available")
        _cuda_ctx = _dr.RasterizeCudaContext()
    return _cuda_ctx


def _pinhole_to_gl_proj(
    K: np.ndarray,
    img_w: int,
    img_h: int,
    near: float = 0.01,
    far: float = 100.0,
) -> np.ndarray:
    """Convert a 3x3 pinhole intrinsics matrix to a 4x4 OpenGL projection matrix.

    The existing pipeline uses a z-forward camera convention (positive depth = in
    front of camera).  nvdiffrast uses OpenGL clip-space where the camera looks
    down -Z, so we flip Z (and Y for NDC top-down → bottom-up).

    Returns a float32 (4, 4) projection matrix suitable for nvdiffrast.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # OpenGL NDC projection
    # Map [0, img_w] x [0, img_h] to [-1, 1] x [-1, 1]
    # Flip Y because OpenGL has Y-up while our images have Y-down
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0 * fx / img_w
    proj[0, 2] = 1.0 - 2.0 * cx / img_w
    proj[1, 1] = -(2.0 * fy / img_h)  # flip Y
    proj[1, 2] = -(1.0 - 2.0 * cy / img_h)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


# ---------------------------------------------------------------------------
# Mesh tensor cache (vertices/faces are immutable during a bake run)
# ---------------------------------------------------------------------------
_cached_verts_f32 = None
_cached_faces_i32 = None
_cached_mesh_id: int | None = None


def _get_mesh_tensors(vertices: np.ndarray, faces: np.ndarray):
    """Return (verts_gpu [V,3] float32, faces_gpu [F,3] int32), cached."""
    global _cached_verts_f32, _cached_faces_i32, _cached_mesh_id
    torch = _torch
    mesh_id = id(vertices)
    if _cached_mesh_id == mesh_id and _cached_verts_f32 is not None:
        return _cached_verts_f32, _cached_faces_i32
    _cached_verts_f32 = torch.as_tensor(
        vertices.astype(np.float32, copy=False), device="cuda"
    )
    _cached_faces_i32 = torch.as_tensor(
        faces.astype(np.int32, copy=False), device="cuda"
    )
    _cached_mesh_id = mesh_id
    return _cached_verts_f32, _cached_faces_i32


def gpu_rasterize_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """GPU depth rasterization using nvdiffrast.

    Returns an (img_h, img_w) float32 depth buffer (camera-space Z) with
    np.inf for background pixels — same contract as the CPU version.
    """
    torch = _torch
    dr = _dr
    ctx = _get_cuda_context()

    verts_gpu, faces_gpu = _get_mesh_tensors(vertices, faces)

    # World-to-camera transform
    w2c = np.linalg.inv(c2w).astype(np.float32)
    # Camera-space positions: cam_xyz = w2c[:3,:3] @ pos + w2c[:3,3]
    w2c_t = torch.as_tensor(w2c, device="cuda")
    cam_pos = (verts_gpu @ w2c_t[:3, :3].T) + w2c_t[:3, 3].unsqueeze(0)

    # Clip-space: apply projection matrix
    proj = _pinhole_to_gl_proj(K, img_w, img_h, near=0.005, far=200.0)
    proj_t = torch.as_tensor(proj, device="cuda")

    # Build homogeneous clip coords: [x, y, z, 1] in camera space → clip
    ones = torch.ones(cam_pos.shape[0], 1, device="cuda", dtype=torch.float32)
    cam_h = torch.cat([cam_pos, ones], dim=1)  # (V, 4)
    clip = (cam_h @ proj_t.T)  # (V, 4)

    # nvdiffrast expects (B, V, 4) and (F, 3)
    clip = clip.unsqueeze(0).contiguous()
    tri = faces_gpu.contiguous()

    rast_out, _ = dr.rasterize(ctx, clip, tri, resolution=[img_h, img_w])
    # rast_out shape: (1, H, W, 4) — [u, v, z_depth_in_ndc, face_id]

    # Interpolate camera-space Z (depth)
    cam_z = cam_pos[:, 2:3].contiguous()  # (V, 1)
    depth_interp, _ = dr.interpolate(cam_z.unsqueeze(0), rast_out, tri)
    # depth_interp: (1, H, W, 1)

    depth_np = depth_interp[0, :, :, 0].cpu().numpy()

    # Background pixels (face_id == 0 means no triangle) → np.inf
    face_ids = rast_out[0, :, :, 3].cpu().numpy()
    depth_np[face_ids == 0] = np.inf

    return depth_np.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 2: UV-space rasterization
# ---------------------------------------------------------------------------

def gpu_rasterize_uv_space(
    uvs: np.ndarray,
    faces: np.ndarray,
    tex_res: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the UV atlas to produce face_id_buf and barycentric weights.

    Args:
        uvs: (V, 2) float UV coordinates in [0, 1].
        faces: (F, 3) int32 face indices.
        tex_res: Texture resolution (square).

    Returns:
        face_id_buf: (tex_res, tex_res) int32, -1 for empty pixels.
        bary_buf: (tex_res, tex_res, 3) float32 barycentric weights (w0, w1, w2).
    """
    torch = _torch
    dr = _dr
    ctx = _get_cuda_context()

    # UV [0,1] → clip-space [-1,1], z=0, w=1
    uv_t = torch.as_tensor(uvs.astype(np.float32, copy=False), device="cuda")
    clip_x = uv_t[:, 0] * 2.0 - 1.0
    clip_y = uv_t[:, 1] * 2.0 - 1.0
    zeros = torch.zeros(uv_t.shape[0], device="cuda", dtype=torch.float32)
    ones = torch.ones(uv_t.shape[0], device="cuda", dtype=torch.float32)
    clip = torch.stack([clip_x, clip_y, zeros, ones], dim=1)  # (V, 4)
    clip = clip.unsqueeze(0).contiguous()

    tri = torch.as_tensor(faces.astype(np.int32, copy=False), device="cuda").contiguous()

    rast_out, _ = dr.rasterize(ctx, clip, tri, resolution=[tex_res, tex_res])
    # rast_out: (1, H, W, 4) — [bary_u, bary_v, z, face_id_1indexed]

    rast_np = rast_out[0].cpu().numpy()
    face_ids_1idx = rast_np[:, :, 3].astype(np.int32)

    # Convert 1-indexed → 0-indexed; background (0) → -1
    face_id_buf = np.where(face_ids_1idx > 0, face_ids_1idx - 1, -1).astype(np.int32)

    # Barycentric: nvdiffrast gives (u, v) where w0 = 1-u-v, w1 = u, w2 = v
    bary_u = rast_np[:, :, 0]
    bary_v = rast_np[:, :, 1]
    bary_w0 = 1.0 - bary_u - bary_v
    bary_buf = np.stack([bary_w0, bary_u, bary_v], axis=2).astype(np.float32)

    return face_id_buf, bary_buf


# ---------------------------------------------------------------------------
# Phase 3: GPU view scoring (projection + scoring)
# ---------------------------------------------------------------------------

# Cached tensors for pos3d / normals (immutable across views)
_cached_pos3d_gpu = None
_cached_normals_gpu = None
_cached_pos3d_id: int | None = None


def _cache_texel_tensors(pos3d: np.ndarray, normals: np.ndarray):
    """Upload pos3d and normals to GPU once."""
    global _cached_pos3d_gpu, _cached_normals_gpu, _cached_pos3d_id
    torch = _torch
    pid = id(pos3d)
    if _cached_pos3d_id == pid and _cached_pos3d_gpu is not None:
        return _cached_pos3d_gpu, _cached_normals_gpu
    _cached_pos3d_gpu = torch.as_tensor(
        pos3d.astype(np.float32, copy=False), device="cuda"
    )
    _cached_normals_gpu = torch.as_tensor(
        normals.astype(np.float32, copy=False), device="cuda"
    )
    _cached_pos3d_id = pid
    return _cached_pos3d_gpu, _cached_normals_gpu


def gpu_evaluate_view_samples(
    pos3d: np.ndarray,
    normals: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    mask_bool: np.ndarray,
    depth_buffer: np.ndarray,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated view sample evaluation. Same interface as CPU version."""
    torch = _torch

    pos_gpu, norm_gpu = _cache_texel_tensors(pos3d, normals)
    # If pos3d is a slice (different id), do a direct upload
    if id(pos3d) != _cached_pos3d_id:
        pos_gpu = torch.as_tensor(pos3d.astype(np.float32, copy=False), device="cuda")
        norm_gpu = torch.as_tensor(normals.astype(np.float32, copy=False), device="cuda")

    c2w_t = torch.as_tensor(c2w.astype(np.float32, copy=False), device="cuda")
    K_t = torch.as_tensor(K.astype(np.float32, copy=False), device="cuda")

    cam_pos = c2w_t[:3, 3]
    view_dirs = cam_pos.unsqueeze(0) - pos_gpu  # (N, 3)
    dists = torch.norm(view_dirs, dim=1)
    view_dirs_n = view_dirs / torch.clamp(dists.unsqueeze(1), min=1e-10)
    cos_angle = (norm_gpu * view_dirs_n).sum(dim=1)

    # Project
    w2c = torch.inverse(c2w_t)
    cam = (pos_gpu @ w2c[:3, :3].T) + w2c[:3, 3].unsqueeze(0)
    d = cam[:, 2].clone()
    sz = torch.clamp(d, min=1e-10)
    u = K_t[0, 0] * cam[:, 0] / sz + K_t[0, 2]
    v = K_t[1, 1] * cam[:, 1] / sz + K_t[1, 2]

    in_bounds = (d > 0.01) & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)

    # Transfer to CPU for mask/depth lookup
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    cos_np = cos_angle.cpu().numpy().astype(np.float64)
    dists_np = dists.cpu().numpy().astype(np.float64)
    in_bounds_np = in_bounds.cpu().numpy()

    pxi = np.clip(np.rint(u_np).astype(np.int32), 0, img_w - 1)
    pyi = np.clip(np.rint(v_np).astype(np.int32), 0, img_h - 1)
    mask_ok = mask_bool[pyi, pxi]

    depth_ref = depth_buffer[pyi, pxi].astype(np.float64)
    cos_safe = np.maximum(np.abs(cos_np), 0.1)
    visibility_eps = np.maximum(1e-4, 0.003 * depth_ref / cos_safe)
    d_np = d.cpu().numpy().astype(np.float64)
    visible = d_np <= (depth_ref + visibility_eps)

    facing = cos_np > min_cos
    valid = in_bounds_np & mask_ok & visible & facing

    score = np.zeros(len(pos3d), dtype=np.float64)
    if np.any(valid):
        ang = np.power(np.maximum(cos_np[valid], 0.0), angle_exp)
        dist_term = np.power(np.maximum(dists_np[valid], 1e-6), dist_pow)
        score[valid] = ang / np.maximum(dist_term, 1e-10)

    return valid, score, u_np, v_np


# ---------------------------------------------------------------------------
# Phase 4: GPU batch color scoring for intrinsics estimation
# ---------------------------------------------------------------------------

def gpu_batch_color_score(
    pts: np.ndarray,
    colors: np.ndarray,
    poses: list[np.ndarray],
    pose_indices: list[int],
    K: np.ndarray,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    img_w: int,
    img_h: int,
) -> float:
    """Batch-evaluate color matching score across frames on GPU.

    Returns negative MSE (higher is better), matching CPU convention.
    """
    torch = _torch

    pts_gpu = torch.as_tensor(pts.astype(np.float32, copy=False), device="cuda")
    cols_gpu = torch.as_tensor(colors.astype(np.float32, copy=False), device="cuda")
    K_t = torch.as_tensor(K.astype(np.float32, copy=False), device="cuda")

    total_err = 0.0
    total_cnt = 0

    for pose, frame, mask in zip(poses, frames, masks):
        w2c = np.linalg.inv(pose).astype(np.float32)
        w2c_t = torch.as_tensor(w2c, device="cuda")

        cam = (pts_gpu @ w2c_t[:3, :3].T) + w2c_t[:3, 3].unsqueeze(0)
        d = cam[:, 2]
        valid_d = d > 0.01
        sz = torch.clamp(d, min=1e-10)
        u = K_t[0, 0] * cam[:, 0] / sz + K_t[0, 2]
        v = K_t[1, 1] * cam[:, 1] / sz + K_t[1, 2]

        in_bounds = valid_d & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)
        if not torch.any(in_bounds):
            continue

        ui = torch.clamp(torch.round(u).long(), 0, img_w - 1)
        vi = torch.clamp(torch.round(v).long(), 0, img_h - 1)

        # Mask/frame lookups on CPU (textures not on GPU)
        ui_np = ui.cpu().numpy()
        vi_np = vi.cpu().numpy()
        in_b = in_bounds.cpu().numpy()

        m = mask[vi_np[in_b], ui_np[in_b]]
        if m.sum() == 0:
            continue

        frame_colors = frame[vi_np[in_b][m], ui_np[in_b][m]]
        # Use indices correctly
        in_b_idx = np.where(in_b)[0]
        pt_colors = colors[in_b_idx[m]]

        diff = frame_colors - pt_colors
        total_err += float(np.mean(diff ** 2) * m.sum())
        total_cnt += int(m.sum())

    return -(total_err / total_cnt) if total_cnt > 0 else -1.0


def clear_cache():
    """Release all cached GPU tensors."""
    global _cached_verts_f32, _cached_faces_i32, _cached_mesh_id
    global _cached_pos3d_gpu, _cached_normals_gpu, _cached_pos3d_id
    global _cuda_ctx
    _cached_verts_f32 = None
    _cached_faces_i32 = None
    _cached_mesh_id = None
    _cached_pos3d_gpu = None
    _cached_normals_gpu = None
    _cached_pos3d_id = None
    # Don't destroy the context — it's reusable
