"""Cap region detection and infill for texture baking."""

import cv2
import numpy as np


def _identify_cap_texels(
    vert_colors: np.ndarray,
    vmapping: np.ndarray,
    new_faces: np.ndarray,
    fids: np.ndarray,
    has_color: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return (indices of cap texels missing color or None, per-face cap mask)."""
    orange = np.array([1.0, 0.55, 0.0])
    orig_colors = vert_colors[vmapping]
    is_orange = np.linalg.norm(orig_colors - orange, axis=1) < 0.05
    cap_face = (
        is_orange[new_faces[:, 0]]
        & is_orange[new_faces[:, 1]]
        & is_orange[new_faces[:, 2]]
    )
    cap_missing = cap_face[fids] & ~has_color
    idx = np.where(cap_missing)[0]
    return (idx if idx.size > 0 else None), cap_face


def _seed_cap_border(
    texture: np.ndarray,
    valid_mask: np.ndarray,
    vmapping: np.ndarray,
    new_faces: np.ndarray,
    cap_face_mask: np.ndarray,
    uvs: np.ndarray,
    tex_res: int,
) -> int:
    """Seed cap border texels from mesh-adjacent body texels via shared vertices.

    xatlas places the cap in an isolated UV chart.  This bridges the gap by
    finding cap UV vertices that share an original PLY vertex with a body UV
    vertex and copying the body-side texture color to the cap-side position.
    """
    cap_verts = set(new_faces[cap_face_mask].ravel().tolist())
    body_verts = set(new_faces[~cap_face_mask].ravel().tolist())

    # original PLY vertex → list of xatlas UV vertices
    orig_to_uv: dict[int, list[int]] = {}
    for uv_idx, orig_idx in enumerate(vmapping):
        orig_to_uv.setdefault(int(orig_idx), []).append(uv_idx)

    uv_px = np.clip((uvs * tex_res).astype(np.int32), 0, tex_res - 1)
    seeded = 0
    search = 3  # half-size of neighborhood search around body UV vertex

    for cap_uv in cap_verts:
        peers = orig_to_uv.get(int(vmapping[cap_uv]), [])
        for peer in peers:
            if peer not in body_verts:
                continue
            bx, by = int(uv_px[peer, 0]), int(uv_px[peer, 1])
            # Search small neighborhood for a colored pixel
            found = False
            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < tex_res and 0 <= nx < tex_res and valid_mask[ny, nx]:
                        cx, cy = int(uv_px[cap_uv, 0]), int(uv_px[cap_uv, 1])
                        texture[cy, cx] = texture[ny, nx]
                        valid_mask[cy, cx] = True
                        seeded += 1
                        found = True
                        break
                if found:
                    break
            if found:
                break

    return seeded


def _fill_cap_region(
    texture: np.ndarray,
    cap_mask: np.ndarray,
    valid_mask: np.ndarray,
    max_iters: int = 300,
) -> int:
    """Fill cap region by iterative neighbor-averaging. Returns texels filled."""
    cy, cx = np.where(cap_mask)
    margin = 10
    y0 = max(0, int(cy.min()) - margin)
    y1 = min(texture.shape[0], int(cy.max()) + margin + 1)
    x0 = max(0, int(cx.min()) - margin)
    x1 = min(texture.shape[1], int(cx.max()) + margin + 1)

    tex_crop = texture[y0:y1, x0:x1].copy()
    cap_crop = cap_mask[y0:y1, x0:x1]
    val_crop = valid_mask[y0:y1, x0:x1].copy()

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    filled = 0

    for _ in range(max_iters):
        empty = cap_crop & ~val_crop
        if not np.any(empty):
            break
        nc = cv2.filter2D(
            val_crop.astype(np.float32), -1, kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        newly = empty & (nc > 0)
        if not np.any(newly):
            break
        for c in range(3):
            ns = cv2.filter2D(
                tex_crop[:, :, c], -1, kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            tex_crop[:, :, c][newly] = ns[newly] / nc[newly]
        val_crop[newly] = True
        filled += int(newly.sum())

    texture[y0:y1, x0:x1] = tex_crop
    valid_mask[y0:y1, x0:x1] = val_crop
    return filled
