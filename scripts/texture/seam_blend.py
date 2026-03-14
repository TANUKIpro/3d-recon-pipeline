"""Seam blending and detail refinement for texture baking."""

from collections import OrderedDict

import cv2
import numpy as np

from scripts.texture.image_utils import (
    _bilinear_sample,
    _masked_detail_advantage,
    _masked_gaussian,
    _masked_gradient_difference,
    _masked_gray_difference,
    _masked_rgb_difference,
    _rgb_to_gray,
    _translate_patch,
)
from scripts.texture.io_utils import _FrameCache
from scripts.texture.view_scoring import _evaluate_view_samples, _rasterize_view_depth

_TEXTURE_SEAM_LEVEL_BLEND = 0.30
_TEXTURE_SEAM_LEVEL_DILATE = 2
_TEXTURE_SEAM_LEVEL_SIGMA = 0.8
_TEXTURE_SEAM_LOW_SIGMA = 0.75
_TEXTURE_SEAM_SMOOTH_SIGMA = 1.65
_TEXTURE_SEAM_HQ_SIGMA = 1.25
_TEXTURE_SEAM_HQ_DILATE = 3
_TEXTURE_SEAM_HQ_DETAIL_GAIN = 1.0
_TEXTURE_SEAM_HQ_LOW_BLEND = 0.35
_TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS = 9
_TEXTURE_SEAM_HQ_SHIFT_LIMIT = 1.5
_TEXTURE_SEAM_HQ_RESPONSE_FLOOR = 0.50
_TEXTURE_SEAM_HQ_MAX_COMPANIONS = 3
_TEXTURE_SEAM_HQ_PATCH_PAD = 3
_TEXTURE_SEAM_HQ_MIN_OVERLAP_RATIO = 0.35
_TEXTURE_SEAM_HQ_ECC_ITERS = 60
_TEXTURE_SEAM_HQ_ECC_EPS = 1e-5
_TEXTURE_SEAM_HQ_COLOR_EPS = 1e-4
_TEXTURE_SEAM_HQ_GRADIENT_WEIGHT = 0.35
_TEXTURE_SEAM_HQ_DETAIL_WEIGHT = 0.18


def _compute_label_boundary(label_buffer: np.ndarray) -> np.ndarray:
    valid_labels = label_buffer >= 0
    boundary = np.zeros(label_buffer.shape, dtype=bool)
    diff_x = valid_labels[:, 1:] & valid_labels[:, :-1] & (label_buffer[:, 1:] != label_buffer[:, :-1])
    diff_y = valid_labels[1:, :] & valid_labels[:-1, :] & (label_buffer[1:, :] != label_buffer[:-1, :])
    boundary[:, 1:] |= diff_x
    boundary[:, :-1] |= diff_x
    boundary[1:, :] |= diff_y
    boundary[:-1, :] |= diff_y
    return boundary


def _normalize_candidate_patch_colors(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = _TEXTURE_SEAM_HQ_COLOR_EPS,
) -> np.ndarray:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return candidate_patch.copy()

    normalized = candidate_patch.astype(np.float32, copy=True)
    ref_low = _masked_gaussian(reference_patch.astype(np.float32, copy=False), valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    cand_low = _masked_gaussian(candidate_patch.astype(np.float32, copy=False), valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    cand_detail = normalized - cand_low
    ref_vals = ref_low[overlap].astype(np.float32, copy=False)
    cand_vals = cand_low[overlap].astype(np.float32, copy=False)
    ref_mean = ref_vals.mean(axis=0)
    cand_mean = cand_vals.mean(axis=0)
    ref_std = ref_vals.std(axis=0)
    cand_std = cand_vals.std(axis=0)
    scale = np.where(cand_std > eps, ref_std / np.maximum(cand_std, eps), 1.0)
    normalized_low = (cand_low - cand_mean[None, None, :]) * scale[None, None, :] + ref_mean[None, None, :]
    normalized = normalized_low + cand_detail
    return np.clip(normalized, 0.0, 1.0)


def _estimate_detail_band_shift(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
    max_shift: float = _TEXTURE_SEAM_HQ_SHIFT_LIMIT,
) -> tuple[float, float, float]:
    if reference_patch.size == 0 or candidate_patch.size == 0 or int(valid_mask.sum()) < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
        return 0.0, 0.0, 0.0

    ref_gray = (
        0.299 * reference_patch[:, :, 0]
        + 0.587 * reference_patch[:, :, 1]
        + 0.114 * reference_patch[:, :, 2]
    ).astype(np.float32, copy=False)
    cand_gray = (
        0.299 * candidate_patch[:, :, 0]
        + 0.587 * candidate_patch[:, :, 1]
        + 0.114 * candidate_patch[:, :, 2]
    ).astype(np.float32, copy=False)
    mask_f = valid_mask.astype(np.float32, copy=False)

    ref_hp = ref_gray - _masked_gaussian(ref_gray, valid_mask, sigma=1.0)
    cand_hp = cand_gray - _masked_gaussian(cand_gray, valid_mask, sigma=1.0)
    ref_hp *= mask_f
    cand_hp *= mask_f
    if float(np.std(ref_hp)) < 1e-4 or float(np.std(cand_hp)) < 1e-4:
        return 0.0, 0.0, 0.0

    try:
        shift, response = cv2.phaseCorrelate(ref_hp, cand_hp)
    except cv2.error:
        return 0.0, 0.0, 0.0
    dx, dy = float(shift[0]), float(shift[1])
    if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(response):
        return 0.0, 0.0, 0.0
    return (
        float(np.clip(dx, -max_shift, max_shift)),
        float(np.clip(dy, -max_shift, max_shift)),
        float(max(0.0, min(1.0, response))),
    )


def _refine_detail_band_shift(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
    init_dx: float = 0.0,
    init_dy: float = 0.0,
    max_shift: float = _TEXTURE_SEAM_HQ_SHIFT_LIMIT,
) -> tuple[float, float, float]:
    if reference_patch.size == 0 or candidate_patch.size == 0 or int(valid_mask.sum()) < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
        return 0.0, 0.0, 0.0

    ref_gray = _rgb_to_gray(reference_patch)
    cand_gray = _rgb_to_gray(candidate_patch)
    ref_hp = ref_gray - _masked_gaussian(ref_gray, valid_mask, sigma=1.0)
    cand_hp = cand_gray - _masked_gaussian(cand_gray, valid_mask, sigma=1.0)
    overlap = valid_mask.astype(bool, copy=False)
    ref_std = float(np.std(ref_hp[overlap])) if np.any(overlap) else 0.0
    cand_std = float(np.std(cand_hp[overlap])) if np.any(overlap) else 0.0
    if ref_std < 1e-4 or cand_std < 1e-4:
        return 0.0, 0.0, 0.0

    ref_hp = ref_hp.astype(np.float32, copy=False)
    cand_hp = cand_hp.astype(np.float32, copy=False)
    ref_hp[overlap] = (ref_hp[overlap] - float(ref_hp[overlap].mean())) / max(ref_std, 1e-4)
    cand_hp[overlap] = (cand_hp[overlap] - float(cand_hp[overlap].mean())) / max(cand_std, 1e-4)
    mask_u8 = (valid_mask.astype(np.uint8) * 255)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        _TEXTURE_SEAM_HQ_ECC_ITERS,
        _TEXTURE_SEAM_HQ_ECC_EPS,
    )

    seeds = [
        (float(init_dx), float(init_dy)),
        (-float(init_dx), -float(init_dy)),
        (0.0, 0.0),
    ]
    best_state: tuple[float, float, float, float, float, float] | None = None
    seen: set[tuple[int, int]] = set()
    for seed_dx, seed_dy in seeds:
        key = (int(round(seed_dx * 1000.0)), int(round(seed_dy * 1000.0)))
        if key in seen:
            continue
        seen.add(key)
        seed_shifted = _translate_patch(candidate_patch, seed_dx, seed_dy)
        seed_rgb_diff = _masked_rgb_difference(reference_patch, seed_shifted, valid_mask)
        seed_gray_diff = _masked_gray_difference(reference_patch, seed_shifted, valid_mask)
        seed_response = 0.5 if abs(seed_dx) > 1e-6 or abs(seed_dy) > 1e-6 else 0.0
        seed_state = (
            seed_rgb_diff,
            seed_gray_diff,
            -float(seed_response),
            float(seed_dx),
            float(seed_dy),
            float(seed_response),
        )
        if best_state is None or seed_state < best_state:
            best_state = seed_state
        warp = np.array([[1.0, 0.0, seed_dx], [0.0, 1.0, seed_dy]], dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(
                ref_hp,
                cand_hp,
                warp,
                cv2.MOTION_TRANSLATION,
                criteria,
                inputMask=mask_u8,
                gaussFiltSize=3,
            )
        except cv2.error:
            continue
        dx = float(np.clip(warp[0, 2], -max_shift, max_shift))
        dy = float(np.clip(warp[1, 2], -max_shift, max_shift))
        shifted = _translate_patch(candidate_patch, dx, dy)
        rgb_diff = _masked_rgb_difference(reference_patch, shifted, valid_mask)
        gray_diff = _masked_gray_difference(reference_patch, shifted, valid_mask)
        state = (rgb_diff, gray_diff, -float(cc), dx, dy, float(np.clip(cc, 0.0, 1.0)))
        if best_state is None or state < best_state:
            best_state = state

    if best_state is None:
        return 0.0, 0.0, 0.0
    rgb_diff, gray_diff, neg_cc, dx, dy, response = best_state
    del rgb_diff, gray_diff
    return dx, dy, float(np.clip(max(response, -neg_cc), 0.0, 1.0))


def _select_detail_companion_view_candidates(
    primary_views: np.ndarray,
    best_views: np.ndarray,
    best_scores: np.ndarray,
    max_candidates: int = _TEXTURE_SEAM_HQ_MAX_COMPANIONS,
) -> list[np.ndarray]:
    selections: list[np.ndarray] = []
    if max_candidates <= 0 or best_views.size == 0:
        return selections

    for _cand_idx in range(max_candidates):
        companion = np.full(primary_views.shape, -1, dtype=np.int32)
        for k in range(best_views.shape[1]):
            cand_views = best_views[:, k]
            cand_scores = best_scores[:, k]
            take = (
                (companion < 0)
                & (primary_views >= 0)
                & (cand_views >= 0)
                & (cand_scores > 0)
                & (cand_views != primary_views)
            )
            for prev in selections:
                take &= cand_views != prev
            companion[take] = cand_views[take].astype(np.int32, copy=False)
        if not np.any(companion >= 0):
            break
        selections.append(companion)
    return selections


def _select_detail_companion_views(
    primary_views: np.ndarray,
    best_views: np.ndarray,
    best_scores: np.ndarray,
) -> np.ndarray:
    candidates = _select_detail_companion_view_candidates(primary_views, best_views, best_scores, max_candidates=1)
    if candidates:
        return candidates[0]
    return np.full(primary_views.shape, -1, dtype=np.int32)


def _select_best_detail_candidate(
    reference_patch: np.ndarray,
    component_mask: np.ndarray,
    candidate_sources: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if reference_patch.size == 0 or not np.any(component_mask) or not candidate_sources:
        return None

    component_pixels = int(component_mask.sum())
    best_choice: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None

    for candidate_patch, candidate_valid_mask in candidate_sources:
        overlap_seed = component_mask & candidate_valid_mask
        if int(overlap_seed.sum()) < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
            continue

        normalized = _normalize_candidate_patch_colors(reference_patch, candidate_patch, overlap_seed)
        phase_dx, phase_dy, phase_response = _estimate_detail_band_shift(reference_patch, normalized, overlap_seed)
        ecc_dx, ecc_dy, ecc_response = _refine_detail_band_shift(
            reference_patch,
            normalized,
            overlap_seed,
            init_dx=phase_dx,
            init_dy=phase_dy,
        )

        proposals = [
            (0.0, 0.0, max(0.0, phase_response * 0.25)),
            (phase_dx, phase_dy, phase_response),
            (-phase_dx, -phase_dy, phase_response * 0.85),
            (ecc_dx, ecc_dy, max(phase_response, ecc_response)),
            (-ecc_dx, -ecc_dy, max(phase_response, ecc_response) * 0.85),
        ]
        seen_shifts: set[tuple[int, int]] = set()
        for shift_dx, shift_dy, response in proposals:
            shift_key = (int(round(shift_dx * 1000.0)), int(round(shift_dy * 1000.0)))
            if shift_key in seen_shifts:
                continue
            seen_shifts.add(shift_key)

            shifted_patch = _translate_patch(normalized, shift_dx, shift_dy)
            shifted_mask = _translate_patch(candidate_valid_mask.astype(np.uint8), shift_dx, shift_dy, is_mask=True) > 0
            merge_mask = component_mask & shifted_mask
            overlap_pixels = int(merge_mask.sum())
            if overlap_pixels < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
                continue

            overlap_ratio = overlap_pixels / max(component_pixels, 1)
            if overlap_ratio < _TEXTURE_SEAM_HQ_MIN_OVERLAP_RATIO:
                continue

            gray_diff = _masked_gray_difference(reference_patch, shifted_patch, merge_mask)
            gradient_diff = _masked_gradient_difference(reference_patch, shifted_patch, merge_mask)
            detail_advantage = _masked_detail_advantage(reference_patch, shifted_patch, merge_mask)
            score = (
                1.10 * overlap_ratio
                + 0.35 * max(0.0, min(1.0, response))
                + (_TEXTURE_SEAM_HQ_DETAIL_WEIGHT * detail_advantage)
                - gray_diff
                - (_TEXTURE_SEAM_HQ_GRADIENT_WEIGHT * gradient_diff)
            )
            choice = (score, shifted_patch, shifted_mask, merge_mask, max(overlap_ratio, response))
            if best_choice is None or choice[0] > best_choice[0]:
                best_choice = choice

    if best_choice is None:
        return None
    _score, shifted_patch, shifted_mask, merge_mask, confidence = best_choice
    return shifted_patch, shifted_mask, merge_mask, float(confidence)


def _merge_detail_component(
    base_low_patch: np.ndarray,
    base_detail_patch: np.ndarray,
    shifted_patch: np.ndarray,
    shifted_mask: np.ndarray,
    merge_mask: np.ndarray,
    confidence: float,
) -> np.ndarray:
    cand_low = _masked_gaussian(shifted_patch, shifted_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    cand_detail = shifted_patch - cand_low
    ref_mag = np.linalg.norm(base_detail_patch, axis=2)
    cand_mag = np.linalg.norm(cand_detail, axis=2)
    stronger = cand_mag > (ref_mag * 1.03)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    low_mix = min(0.60, _TEXTURE_SEAM_HQ_LOW_BLEND * max(confidence, 0.55))
    detail_gain = min(1.00, max(0.70, _TEXTURE_SEAM_HQ_DETAIL_GAIN * max(confidence, 0.70)))
    chosen_detail = np.where(stronger[:, :, None], cand_detail, base_detail_patch)
    merged = np.clip(
        (1.0 - low_mix) * base_low_patch
        + low_mix * cand_low
        + (1.0 - detail_gain) * base_detail_patch
        + detail_gain * chosen_detail,
        0.0,
        1.0,
    )
    merged = np.where(
        stronger[:, :, None],
        np.maximum(merged, shifted_patch),
        merged,
    )
    color_advantage = np.max(np.abs(shifted_patch - base_low_patch), axis=2) > 0.18
    if np.any(color_advantage & merge_mask):
        color_mix = min(0.85, max(0.60, confidence))
        boosted = np.clip(
            (1.0 - color_mix) * merged + color_mix * shifted_patch,
            0.0,
            1.0,
        )
        merged = np.where(color_advantage[:, :, None], boosted, merged)
    return merged


def _apply_narrow_seam_leveling(
    texture: np.ndarray,
    valid_mask: np.ndarray,
    label_buffer: np.ndarray,
    blend: float = _TEXTURE_SEAM_LEVEL_BLEND,
    dilate_iters: int = _TEXTURE_SEAM_LEVEL_DILATE,
    sigma: float = _TEXTURE_SEAM_LEVEL_SIGMA,
    detail_texture: np.ndarray | None = None,
    detail_valid_mask: np.ndarray | None = None,
    detail_textures: list[np.ndarray] | None = None,
    detail_valid_masks: list[np.ndarray] | None = None,
    quality_boost: bool = False,
) -> tuple[np.ndarray, int]:
    """Soften region-label seams while preserving local texture detail."""
    valid_labels = label_buffer >= 0
    if texture.size == 0 or not np.any(valid_mask & valid_labels):
        return texture, 0

    boundary = _compute_label_boundary(label_buffer)
    if not np.any(boundary):
        return texture, 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    band = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=max(1, int(dilate_iters))).astype(bool)
    editable = band & valid_mask & valid_labels
    if not np.any(editable):
        return texture, int(boundary.sum())

    result = texture.copy()
    local_low = _masked_gaussian(result, valid_mask, sigma=max(0.4, _TEXTURE_SEAM_LOW_SIGMA))
    smooth_low = _masked_gaussian(result, valid_mask, sigma=max(sigma, _TEXTURE_SEAM_SMOOTH_SIGMA))
    detail = result - local_low
    boundary_distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
    boundary_weight = np.clip(
        1.0 - (boundary_distance / max(float(dilate_iters) + 0.5, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    local_blend = (0.7 * blend * boundary_weight[editable]).astype(np.float32, copy=False)
    result[editable] = np.clip(
        (1.0 - local_blend[:, None]) * local_low[editable]
        + local_blend[:, None] * smooth_low[editable]
        + detail[editable],
        0.0,
        1.0,
    )

    if (
        quality_boost
    ):
        detail_sources: list[tuple[np.ndarray, np.ndarray]] = []
        if (
            detail_texture is not None
            and detail_valid_mask is not None
            and detail_texture.shape == texture.shape
            and detail_valid_mask.shape == valid_mask.shape
        ):
            detail_sources.append((detail_texture, detail_valid_mask))
        if detail_textures is not None and detail_valid_masks is not None:
            for tex_src, mask_src in zip(detail_textures, detail_valid_masks, strict=False):
                if tex_src.shape != texture.shape or mask_src.shape != valid_mask.shape:
                    continue
                detail_sources.append((tex_src, mask_src))

        if not detail_sources:
            return result, int(boundary.sum())

        detail_avail = np.zeros(valid_mask.shape, dtype=bool)
        for _detail_texture, detail_mask_src in detail_sources:
            detail_avail |= detail_mask_src
        detail_mask = editable & detail_avail
        if np.any(detail_mask):
            num_components, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                detail_mask.astype(np.uint8),
                connectivity=8,
            )
            refined = result.copy()
            base_low = _masked_gaussian(refined, valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
            base_detail = refined - base_low
            for comp_id in range(1, num_components):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
                    continue
                x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                y = int(stats[comp_id, cv2.CC_STAT_TOP])
                w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                pad = _TEXTURE_SEAM_HQ_PATCH_PAD
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(texture.shape[1], x + w + pad)
                y1 = min(texture.shape[0], y + h + pad)
                local_component = labels[y0:y1, x0:x1] == comp_id
                local_ref = refined[y0:y1, x0:x1]
                candidate_patches: list[tuple[np.ndarray, np.ndarray]] = []
                for candidate_texture, candidate_mask in detail_sources:
                    local_cand_mask = candidate_mask[y0:y1, x0:x1]
                    if not np.any(local_component & local_cand_mask):
                        continue
                    candidate_patches.append(
                        (
                            candidate_texture[y0:y1, x0:x1],
                            local_cand_mask,
                        )
                    )
                selected = _select_best_detail_candidate(local_ref, local_component, candidate_patches)
                if selected is None:
                    continue
                shifted_cand, shifted_mask, merge_mask, confidence = selected
                base_low_patch = base_low[y0:y1, x0:x1]
                base_detail_patch = base_detail[y0:y1, x0:x1]
                merged = _merge_detail_component(
                    base_low_patch=base_low_patch,
                    base_detail_patch=base_detail_patch,
                    shifted_patch=shifted_cand,
                    shifted_mask=shifted_mask,
                    merge_mask=merge_mask,
                    confidence=confidence,
                )
                patch = refined[y0:y1, x0:x1]
                patch[merge_mask] = merged[merge_mask]
                refined[y0:y1, x0:x1] = patch
            result = refined
    return result, int(boundary.sum())


def _sample_companion_texture(
    tex_shape: tuple[int, int, int],
    tidx: np.ndarray,
    companion_views: np.ndarray,
    tex_xs: np.ndarray,
    tex_ys: np.ndarray,
    pos3d: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
    pose_frame_indices: list[int],
    frames_dir: str,
    mask_dir: str,
    frame_cache: _FrameCache,
    depth_cache: OrderedDict[int, np.ndarray],
    new_vertices: np.ndarray,
    new_faces: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
) -> tuple[np.ndarray, np.ndarray]:
    companion_texture = np.zeros(tex_shape, dtype=np.float32)
    companion_valid = np.zeros(tex_shape[:2], dtype=bool)
    if tidx.size == 0:
        return companion_texture, companion_valid

    local_views = companion_views[tidx]
    usable = local_views >= 0
    if not np.any(usable):
        return companion_texture, companion_valid
    active_tidx = tidx[usable]
    active_views = local_views[usable]

    for vidx in np.unique(active_views):
        if vidx < 0:
            continue
        view_tidx = active_tidx[active_views == vidx]
        if view_tidx.size == 0:
            continue
        src_idx = int(pose_frame_indices[int(vidx)])
        c2w = poses[int(vidx)]
        try:
            frame = frame_cache.load_frame(frames_dir, src_idx)
            mask_bool = frame_cache.load_mask(mask_dir, src_idx)
        except FileNotFoundError:
            continue

        if int(vidx) in depth_cache:
            depth_buffer = depth_cache[int(vidx)]
        else:
            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
            depth_cache[int(vidx)] = depth_buffer
        valid, _score, px_proj, py_proj = _evaluate_view_samples(
            pos3d=pos3d[view_tidx],
            normals=normals[view_tidx],
            c2w=c2w,
            K=K,
            img_w=img_w,
            img_h=img_h,
            mask_bool=mask_bool,
            depth_buffer=depth_buffer,
            min_cos=min_cos,
            angle_exp=angle_exp,
            dist_pow=dist_pow,
        )
        if not np.any(valid):
            continue
        fill_tidx = view_tidx[valid]
        colors = _bilinear_sample(frame, px_proj[valid], py_proj[valid]).astype(np.float32)
        companion_texture[tex_ys[fill_tidx], tex_xs[fill_tidx]] = colors
        companion_valid[tex_ys[fill_tidx], tex_xs[fill_tidx]] = True
    return companion_texture, companion_valid


def _sample_companion_patch(
    patch_shape: tuple[int, int, int],
    global_tidx: np.ndarray,
    companion_views: np.ndarray,
    patch_x0: int,
    patch_y0: int,
    tex_xs: np.ndarray,
    tex_ys: np.ndarray,
    pos3d: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
    pose_frame_indices: list[int],
    frames_dir: str,
    mask_dir: str,
    frame_cache: _FrameCache,
    depth_cache: OrderedDict[int, np.ndarray],
    new_vertices: np.ndarray,
    new_faces: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
) -> tuple[np.ndarray, np.ndarray]:
    companion_patch = np.zeros(patch_shape, dtype=np.float32)
    companion_valid = np.zeros(patch_shape[:2], dtype=bool)
    if global_tidx.size == 0 or companion_views.size == 0:
        return companion_patch, companion_valid

    usable = companion_views >= 0
    if not np.any(usable):
        return companion_patch, companion_valid

    active_tidx = global_tidx[usable]
    active_views = companion_views[usable]
    for vidx in np.unique(active_views):
        if vidx < 0:
            continue
        view_tidx = active_tidx[active_views == vidx]
        if view_tidx.size == 0:
            continue
        src_idx = int(pose_frame_indices[int(vidx)])
        c2w = poses[int(vidx)]
        try:
            frame = frame_cache.load_frame(frames_dir, src_idx)
            mask_bool = frame_cache.load_mask(mask_dir, src_idx)
        except FileNotFoundError:
            continue

        if int(vidx) in depth_cache:
            depth_buffer = depth_cache[int(vidx)]
        else:
            depth_buffer = _rasterize_view_depth(new_vertices, new_faces, c2w, K, img_w, img_h)
            depth_cache[int(vidx)] = depth_buffer
        valid, _score, px_proj, py_proj = _evaluate_view_samples(
            pos3d=pos3d[view_tidx],
            normals=normals[view_tidx],
            c2w=c2w,
            K=K,
            img_w=img_w,
            img_h=img_h,
            mask_bool=mask_bool,
            depth_buffer=depth_buffer,
            min_cos=min_cos,
            angle_exp=angle_exp,
            dist_pow=dist_pow,
        )
        if not np.any(valid):
            continue
        fill_tidx = view_tidx[valid]
        local_x = tex_xs[fill_tidx] - int(patch_x0)
        local_y = tex_ys[fill_tidx] - int(patch_y0)
        inside = (
            (local_x >= 0)
            & (local_x < patch_shape[1])
            & (local_y >= 0)
            & (local_y < patch_shape[0])
        )
        if not np.any(inside):
            continue
        fill_tidx = fill_tidx[inside]
        local_x = local_x[inside]
        local_y = local_y[inside]
        colors = _bilinear_sample(frame, px_proj[valid][inside], py_proj[valid][inside]).astype(np.float32)
        companion_patch[local_y, local_x] = colors
        companion_valid[local_y, local_x] = True
    return companion_patch, companion_valid


def _apply_quality_boost_detail_refinement(
    texture: np.ndarray,
    valid_mask: np.ndarray,
    label_buffer: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    best_scores: np.ndarray,
    best_views: np.ndarray,
    locked_view_per_texel: np.ndarray,
    pos3d: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
    pose_frame_indices: list[int],
    frames_dir: str,
    mask_dir: str,
    frame_cache: _FrameCache,
    depth_cache: OrderedDict[int, np.ndarray],
    new_vertices: np.ndarray,
    new_faces: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_cos: float,
    angle_exp: float,
    dist_pow: float,
    dilate_iters: int = _TEXTURE_SEAM_HQ_DILATE,
) -> tuple[np.ndarray, int, int]:
    valid_labels = label_buffer >= 0
    if texture.size == 0 or not np.any(valid_mask & valid_labels):
        return texture, 0, 0

    boundary = _compute_label_boundary(label_buffer)
    if not np.any(boundary):
        return texture, 0, 0

    band = cv2.dilate(
        boundary.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=max(1, int(dilate_iters)),
    ).astype(bool)
    detail_mask = band & valid_mask & valid_labels
    if not np.any(detail_mask):
        return texture, int(boundary.sum()), 0

    num_components, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        detail_mask.astype(np.uint8),
        connectivity=8,
    )
    refined = texture.copy()
    base_low = _masked_gaussian(refined, valid_mask, sigma=max(0.7, _TEXTURE_SEAM_HQ_SIGMA))
    base_detail = refined - base_low
    refined_components = 0
    sampled_detail_pixels = 0

    for comp_id in range(1, num_components):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
            continue
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        pad = _TEXTURE_SEAM_HQ_PATCH_PAD
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(texture.shape[1], x + w + pad)
        y1 = min(texture.shape[0], y + h + pad)
        local_component = labels[y0:y1, x0:x1] == comp_id

        in_bbox = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
        if not np.any(in_bbox):
            continue
        bbox_tidx = np.where(in_bbox)[0]
        local_y = ys[bbox_tidx] - y0
        local_x = xs[bbox_tidx] - x0
        component_hits = local_component[local_y, local_x]
        component_tidx = bbox_tidx[component_hits]
        if component_tidx.size < _TEXTURE_SEAM_HQ_MIN_COMPONENT_PIXELS:
            continue

        candidate_view_maps = _select_detail_companion_view_candidates(
            primary_views=locked_view_per_texel[component_tidx],
            best_views=best_views[component_tidx],
            best_scores=best_scores[component_tidx],
        )
        if not candidate_view_maps:
            continue

        local_ref = refined[y0:y1, x0:x1]
        candidate_sources: list[tuple[np.ndarray, np.ndarray]] = []
        for companion_views in candidate_view_maps:
            cand_patch, cand_valid = _sample_companion_patch(
                patch_shape=(y1 - y0, x1 - x0, texture.shape[2]),
                global_tidx=component_tidx,
                companion_views=companion_views,
                patch_x0=x0,
                patch_y0=y0,
                tex_xs=xs,
                tex_ys=ys,
                pos3d=pos3d,
                normals=normals,
                poses=poses,
                pose_frame_indices=pose_frame_indices,
                frames_dir=frames_dir,
                mask_dir=mask_dir,
                frame_cache=frame_cache,
                depth_cache=depth_cache,
                new_vertices=new_vertices,
                new_faces=new_faces,
                K=K,
                img_w=img_w,
                img_h=img_h,
                min_cos=min_cos,
                angle_exp=angle_exp,
                dist_pow=dist_pow,
            )
            if not np.any(local_component & cand_valid):
                continue
            candidate_sources.append((cand_patch, cand_valid))

        if not candidate_sources:
            continue

        selected = _select_best_detail_candidate(local_ref, local_component, candidate_sources)
        if selected is None:
            continue
        shifted_patch, shifted_mask, merge_mask, confidence = selected
        base_low_patch = base_low[y0:y1, x0:x1]
        base_detail_patch = base_detail[y0:y1, x0:x1]
        merged = _merge_detail_component(
            base_low_patch=base_low_patch,
            base_detail_patch=base_detail_patch,
            shifted_patch=shifted_patch,
            shifted_mask=shifted_mask,
            merge_mask=merge_mask,
            confidence=confidence,
        )
        patch = refined[y0:y1, x0:x1]
        patch[merge_mask] = merged[merge_mask]
        refined[y0:y1, x0:x1] = patch
        refined_components += 1
        sampled_detail_pixels += int(merge_mask.sum())

    return refined, int(boundary.sum()), sampled_detail_pixels
