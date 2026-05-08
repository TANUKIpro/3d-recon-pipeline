"""View scoring and selection for texture baking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.texture.intrinsics import _project_simple


@dataclass(slots=True)
class ViewEvalPacket:
    """Projection/scoring intermediates for a single view."""

    px_proj: np.ndarray
    py_proj: np.ndarray
    sample_valid: np.ndarray
    cos_angle: np.ndarray
    distances: np.ndarray


class ViewPacketShapeError(RuntimeError):
    """Raised when a view-evaluation packet length diverges from its input texels."""


def _validate_view_packet(packet: ViewEvalPacket, expected_len: int) -> None:
    """Ensure all packet arrays match the input texel count."""
    lengths = {
        "px_proj": int(packet.px_proj.shape[0]),
        "py_proj": int(packet.py_proj.shape[0]),
        "sample_valid": int(packet.sample_valid.shape[0]),
        "cos_angle": int(packet.cos_angle.shape[0]),
        "distances": int(packet.distances.shape[0]),
    }
    mismatched = {name: length for name, length in lengths.items() if length != expected_len}
    if not mismatched:
        return

    details = ", ".join(f"{name}={length}" for name, length in mismatched.items())
    raise ViewPacketShapeError(
        f"ViewEvalPacket length mismatch: expected {expected_len} samples, got {details}"
    )


def _rasterize_view_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    device: str = "cpu",
    dist_coeffs: np.ndarray | list | None = None,
) -> np.ndarray:
    """Rasterize mesh depth into camera image space for a simple z-test.

    Uses vectorized preprocessing to batch-filter faces before the per-face
    rasterization loop (Optimization B).  When *device* is ``"cuda"`` the
    function attempts GPU rasterization via nvdiffrast and falls back to CPU
    on failure.

    When ``dist_coeffs`` is supplied the projection applies the COLMAP camera
    model so the depth buffer lives in the same distorted-pixel coordinate
    frame as the unmodified source frames; callers can then sample those
    frames without first remapping them through ``cv2.undistort``.
    """
    if device == "cuda":
        try:
            from scripts.texture.gpu_raster import gpu_rasterize_depth
            return gpu_rasterize_depth(
                vertices, faces, c2w, K, img_w, img_h, dist_coeffs=dist_coeffs
            )
        except Exception:
            pass  # fall through to CPU path

    uv_all, depth_all = _project_simple(vertices, c2w, K, dist_coeffs=dist_coeffs)
    depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)
    inside_eps = 1e-5

    n_faces = len(faces)
    if n_faces == 0:
        return depth_buffer

    # --- Vectorized pre-processing ---
    i0s = faces[:, 0]
    i1s = faces[:, 1]
    i2s = faces[:, 2]

    z0s = depth_all[i0s]
    z1s = depth_all[i1s]
    z2s = depth_all[i2s]

    # Back-face culling: skip faces where all three depths <= 0.01
    visible = ~((z0s <= 0.01) & (z1s <= 0.01) & (z2s <= 0.01))

    x0s = uv_all[i0s, 0]
    y0s = uv_all[i0s, 1]
    x1s = uv_all[i1s, 0]
    y1s = uv_all[i1s, 1]
    x2s = uv_all[i2s, 0]
    y2s = uv_all[i2s, 1]

    # Bounding boxes (vectorized)
    bb_min_x = np.floor(np.minimum(np.minimum(x0s, x1s), x2s)).astype(np.int32)
    bb_max_x = np.ceil(np.maximum(np.maximum(x0s, x1s), x2s)).astype(np.int32)
    bb_min_y = np.floor(np.minimum(np.minimum(y0s, y1s), y2s)).astype(np.int32)
    bb_max_y = np.ceil(np.maximum(np.maximum(y0s, y1s), y2s)).astype(np.int32)

    # Check overlap with image BEFORE clipping (matches original asymmetric clip logic)
    bb_valid = (bb_max_x >= 0) & (bb_min_x <= img_w - 1) & (bb_max_y >= 0) & (bb_min_y <= img_h - 1)

    np.clip(bb_min_x, 0, img_w - 1, out=bb_min_x)
    np.clip(bb_max_x, 0, img_w - 1, out=bb_max_x)
    np.clip(bb_min_y, 0, img_h - 1, out=bb_min_y)
    np.clip(bb_max_y, 0, img_h - 1, out=bb_max_y)

    # Barycentric denominator (vectorized)
    denoms = (y1s - y2s) * (x0s - x2s) + (x2s - x1s) * (y0s - y2s)
    non_degenerate = np.abs(denoms) >= 1e-12

    # Combined filter
    keep = visible & bb_valid & non_degenerate
    keep_idx = np.where(keep)[0]

    # Build contiguous arrays for the inner loop
    _x0 = x0s[keep_idx]
    _y0 = y0s[keep_idx]
    _x1 = x1s[keep_idx]
    _y1 = y1s[keep_idx]
    _x2 = x2s[keep_idx]
    _y2 = y2s[keep_idx]
    _z0 = z0s[keep_idx]
    _z1 = z1s[keep_idx]
    _z2 = z2s[keep_idx]
    _denom = denoms[keep_idx]
    _bb_min_x = bb_min_x[keep_idx]
    _bb_max_x = bb_max_x[keep_idx]
    _bb_min_y = bb_min_y[keep_idx]
    _bb_max_y = bb_max_y[keep_idx]

    for i in range(len(keep_idx)):
        mn_x = int(_bb_min_x[i])
        mx_x = int(_bb_max_x[i])
        mn_y = int(_bb_min_y[i])
        mx_y = int(_bb_max_y[i])
        d = _denom[i]

        xs = np.arange(mn_x, mx_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(mn_y, mx_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        gx2 = grid_x - _x2[i]
        gy2 = grid_y - _y2[i]
        w0 = ((_y1[i] - _y2[i]) * gx2 + (_x2[i] - _x1[i]) * gy2) / d
        w1 = ((_y2[i] - _y0[i]) * gx2 + (_x0[i] - _x2[i]) * gy2) / d
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -inside_eps) & (w1 >= -inside_eps) & (w2 >= -inside_eps)
        if not np.any(inside):
            continue

        depth = w0 * _z0[i] + w1 * _z1[i] + w2 * _z2[i]
        valid = inside & (depth > 0.01)
        if not np.any(valid):
            continue

        tile = depth_buffer[mn_y:mx_y + 1, mn_x:mx_x + 1]
        candidate = np.where(valid, depth, np.inf).astype(np.float32, copy=False)
        np.minimum(tile, candidate, out=tile)

    return depth_buffer


def _evaluate_view_samples(
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
    device: str = "cpu",
    dist_coeffs: np.ndarray | list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    packet, valid, score = _evaluate_view_packet(
        pos3d=pos3d,
        normals=normals,
        c2w=c2w,
        K=K,
        img_w=img_w,
        img_h=img_h,
        mask_bool=mask_bool,
        depth_buffer=depth_buffer,
        min_cos=min_cos,
        angle_exp=angle_exp,
        dist_pow=dist_pow,
        device=device,
        dist_coeffs=dist_coeffs,
    )

    return valid, score, packet.px_proj, packet.py_proj


def _score_view_packet(
    packet: ViewEvalPacket,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply facing-angle thresholding and score computation to a packet."""
    facing = packet.cos_angle > min_cos
    valid = packet.sample_valid & facing

    score = np.zeros(packet.cos_angle.shape[0], dtype=np.float64)
    if np.any(valid):
        ang = np.power(np.maximum(packet.cos_angle[valid], 0.0), angle_exp)
        dist_term = np.power(np.maximum(packet.distances[valid], 1e-6), dist_pow)
        score[valid] = ang / np.maximum(dist_term, 1e-10)

    return valid, score


def _evaluate_view_packet(
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
    device: str = "cpu",
    dist_coeffs: np.ndarray | list | None = None,
) -> tuple[ViewEvalPacket, np.ndarray, np.ndarray]:
    """Evaluate which texels can be sampled from a view and return per-texel scores."""
    if device == "cuda":
        try:
            from scripts.texture.gpu_raster import gpu_evaluate_view_packet

            packet = gpu_evaluate_view_packet(
                pos3d, normals, c2w, K, img_w, img_h,
                mask_bool, depth_buffer,
                dist_coeffs=dist_coeffs,
            )
            _validate_view_packet(packet, int(pos3d.shape[0]))
            valid, score = _score_view_packet(packet, min_cos, angle_exp, dist_pow)
            return packet, valid, score
        except ViewPacketShapeError:
            raise
        except Exception:
            pass  # fall through to CPU path

    cam_pos = c2w[:3, 3]
    view_dirs = cam_pos[None, :] - pos3d
    dists = np.linalg.norm(view_dirs, axis=1)
    view_dirs_n = view_dirs / np.maximum(dists[:, None], 1e-10)
    cos_angle = np.sum(normals * view_dirs_n, axis=1)

    uv2d, depths = _project_simple(pos3d, c2w, K, dist_coeffs=dist_coeffs)
    px_proj = uv2d[:, 0]
    py_proj = uv2d[:, 1]
    in_bounds = (
        (depths > 0.01)
        & (px_proj >= 0)
        & (px_proj < img_w - 1)
        & (py_proj >= 0)
        & (py_proj < img_h - 1)
    )

    pxi = np.clip(np.rint(px_proj).astype(np.int32), 0, img_w - 1)
    pyi = np.clip(np.rint(py_proj).astype(np.int32), 0, img_h - 1)
    mask_ok = mask_bool[pyi, pxi]

    depth_ref = depth_buffer[pyi, pxi].astype(np.float64)
    cos_safe = np.maximum(np.abs(cos_angle), 0.1)
    visibility_eps = np.maximum(1e-4, 0.003 * depth_ref / cos_safe)
    visible = depths <= (depth_ref + visibility_eps)

    packet = ViewEvalPacket(
        px_proj=px_proj,
        py_proj=py_proj,
        sample_valid=(in_bounds & mask_ok & visible),
        cos_angle=cos_angle,
        distances=dists,
    )
    _validate_view_packet(packet, int(pos3d.shape[0]))
    valid, score = _score_view_packet(packet, min_cos, angle_exp, dist_pow)

    return packet, valid, score


def _update_topk_scores(
    best_scores: np.ndarray,
    best_views: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    vidx: int,
) -> None:
    """Update per-texel top-K view scores in-place.

    For texels where *valid* is True and *score* exceeds the current worst
    entry in the top-K list, the new sample is inserted into its sorted
    position and the descending-order invariant is preserved.

    Args:
        best_scores: (n_texels, K) float32 — scores kept in descending order.
        best_views: (n_texels, K) int32 — corresponding view indices.
        valid: (n_texels,) bool — which texels have a usable projection.
        score: (n_texels,) float64 — per-texel score for view *vidx*.
        vidx: View index to record.
    """
    K = best_scores.shape[1]
    if K <= 0:
        return

    candidates = valid & (score > best_scores[:, K - 1])
    idx = np.where(candidates)[0]
    if idx.size == 0:
        return

    rows_s = best_scores[idx]                       # (M, K), descending
    rows_v = best_views[idx]                        # (M, K)
    new_scores = score[idx].astype(np.float32)      # (M,)

    # best_scores is descending, so the first slot where the existing score
    # is strictly less than the new sample is the insertion index.  The
    # column-wise filter guarantees at least one such slot exists, so
    # argmax (which returns the first True) is well-defined.
    lt_mask = rows_s < new_scores[:, None]          # (M, K)
    pos = np.argmax(lt_mask, axis=1)                # (M,) ∈ [0, K-1]

    cols = np.arange(K)                             # (K,)
    col_bcast = cols[None, :]                       # (1, K)
    pos_bcast = pos[:, None]                        # (M, 1)

    # Right-shift source: slot j inherits from slot (j - 1); slot 0 stays put.
    shift_idx = np.broadcast_to(
        np.maximum(cols - 1, 0)[None, :], rows_s.shape
    )
    shifted_s = np.take_along_axis(rows_s, shift_idx, axis=1)
    shifted_v = np.take_along_axis(rows_v, shift_idx, axis=1)

    before_mask = col_bcast < pos_bcast             # keep original
    at_mask = col_bcast == pos_bcast                # insert new sample

    new_rows_s = np.where(before_mask, rows_s, shifted_s)
    new_rows_s = np.where(at_mask, new_scores[:, None], new_rows_s)
    new_rows_v = np.where(before_mask, rows_v, shifted_v)
    new_rows_v = np.where(at_mask, np.int32(vidx), new_rows_v)

    best_scores[idx] = new_rows_s
    best_views[idx] = new_rows_v


def _apply_view_hardening(
    best_scores: np.ndarray,
    best_views: np.ndarray,
    hard_ratio: float,
) -> int:
    """Zero out non-dominant views when top-1 clearly wins."""
    if best_scores.shape[1] <= 1 or hard_ratio <= 0:
        return 0
    s0 = best_scores[:, 0]
    s1 = best_scores[:, 1]
    dominant = (s0 > 0) & (s1 > 0) & (s0 > hard_ratio * s1)
    n_hard = int(dominant.sum())
    if n_hard > 0:
        best_scores[dominant, 1:] = -1.0
        best_views[dominant, 1:] = -1
    return n_hard
