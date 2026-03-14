"""Image processing utilities for texture baking."""

import cv2
import numpy as np


def _bilinear_sample(img, x, y):
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    fx = (x - x0.astype(np.float64))[:, None]
    fy = (y - y0.astype(np.float64))[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x1]
            + (1 - fx) * fy * img[y1, x0] + fx * fy * img[y1, x1])


def _masked_gaussian(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    if image.ndim == 3:
        num = cv2.GaussianBlur(image * mask_f[:, :, None], (0, 0), sigmaX=sigma)
        den = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma)
        return num / np.maximum(den[:, :, None], 1e-6)
    num = cv2.GaussianBlur(image * mask_f, (0, 0), sigmaX=sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma)
    return num / np.maximum(den, 1e-6)


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    ).astype(np.float32, copy=False)


def _translate_patch(patch: np.ndarray, dx: float, dy: float, is_mask: bool = False) -> np.ndarray:
    if patch.size == 0 or (abs(dx) < 1e-6 and abs(dy) < 1e-6):
        return patch.copy()
    h, w = patch.shape[:2]
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(
        patch,
        warp,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _masked_gray_difference(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return float("inf")
    ref_gray = (
        0.299 * reference_patch[:, :, 0]
        + 0.587 * reference_patch[:, :, 1]
        + 0.114 * reference_patch[:, :, 2]
    )
    cand_gray = (
        0.299 * candidate_patch[:, :, 0]
        + 0.587 * candidate_patch[:, :, 1]
        + 0.114 * candidate_patch[:, :, 2]
    )
    diff = np.abs(ref_gray[overlap] - cand_gray[overlap])
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff))


def _masked_rgb_difference(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return float("inf")
    diff = np.abs(reference_patch[overlap] - candidate_patch[overlap])
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff))


def _masked_gradient_difference(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return float("inf")
    ref_gray = _rgb_to_gray(reference_patch)
    cand_gray = _rgb_to_gray(candidate_patch)
    ref_dx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    ref_dy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    cand_dx = cv2.Sobel(cand_gray, cv2.CV_32F, 1, 0, ksize=3)
    cand_dy = cv2.Sobel(cand_gray, cv2.CV_32F, 0, 1, ksize=3)
    ref_mag = np.sqrt(ref_dx * ref_dx + ref_dy * ref_dy)
    cand_mag = np.sqrt(cand_dx * cand_dx + cand_dy * cand_dy)
    diff = np.abs(ref_mag[overlap] - cand_mag[overlap])
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff))


def _masked_detail_advantage(
    reference_patch: np.ndarray,
    candidate_patch: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    overlap = valid_mask.astype(bool, copy=False)
    if reference_patch.size == 0 or candidate_patch.size == 0 or not np.any(overlap):
        return 0.0
    ref_gray = _rgb_to_gray(reference_patch)
    cand_gray = _rgb_to_gray(candidate_patch)
    ref_detail = ref_gray - _masked_gaussian(ref_gray, valid_mask, sigma=1.0)
    cand_detail = cand_gray - _masked_gaussian(cand_gray, valid_mask, sigma=1.0)
    ref_mag = np.abs(ref_detail[overlap])
    cand_mag = np.abs(cand_detail[overlap])
    if ref_mag.size == 0 or cand_mag.size == 0:
        return 0.0
    gain = np.maximum(cand_mag - ref_mag, 0.0)
    scale = max(float(np.mean(ref_mag)), 1e-4)
    return float(np.clip(np.mean(gain) / scale, 0.0, 1.5))
