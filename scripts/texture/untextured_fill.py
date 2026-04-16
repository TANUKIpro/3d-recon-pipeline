"""Detect and fill untextured faces in the final OBJ mesh.

Called as the last sub-step of Stage 6 (PostTextureContactCleanup),
after the cleaned OBJ has been written and before final progress is emitted.
Only ``material_0`` faces (body texture) are inspected; cap/strip faces are
intentionally excluded because they carry their own texture sources.

Fill strategy (in priority order):
1. **Face-adjacency BFS** — propagate per-face average colour from textured
   neighbours through the mesh topology, then smooth boundaries with a
   small ``cv2.inpaint`` pass.
2. **PLY vertex colours** — barycentric interpolation of MILo vertex colours
   (useful when PLY colours are brighter than the BFS source).
3. **cv2.inpaint** — pure texel-level TELEA inpainting (last resort).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.config_defaults import (
    _UNTEXTURED_FILL_EMPTY_FRACTION,
    _UNTEXTURED_FILL_EMPTY_THRESHOLD,
    _UNTEXTURED_FILL_INPAINT_RADIUS,
)


def _emit_progress(progress_cb, progress: float, detail: str) -> None:
    if progress_cb is not None:
        progress_cb(float(progress), str(detail))


# ------------------------------------------------------------------
# OBJ parser
# ------------------------------------------------------------------

def _parse_material0_faces(
    obj_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse OBJ and return arrays for material_0 faces.

    Returns
    -------
    vertices, texcoords, uv_faces, pos_faces
    """
    text = obj_path.read_text(encoding="utf-8")
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    uv_face_list: list[tuple[int, int, int]] = []
    pos_face_list: list[tuple[int, int, int]] = []
    current_material = "material_0"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("usemtl "):
            current_material = line.split(None, 1)[1].strip() or current_material
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            continue
        if line.startswith("vt "):
            parts = line.split()
            if len(parts) >= 3:
                texcoords.append([float(parts[1]), float(parts[2])])
            continue
        if line.startswith("f ") and current_material == "material_0":
            tokens = line.split()[1:]
            if len(tokens) < 3:
                continue
            parsed: list[tuple[int, int]] = []
            for token in tokens:
                sp = token.split("/")
                if len(sp) >= 2 and sp[0] and sp[1]:
                    parsed.append((int(sp[0]) - 1, int(sp[1]) - 1))
            for i in range(1, len(parsed) - 1):
                pos_face_list.append((parsed[0][0], parsed[i][0], parsed[i + 1][0]))
                uv_face_list.append((parsed[0][1], parsed[i][1], parsed[i + 1][1]))

    if not vertices or not texcoords or not uv_face_list:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
        )

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(texcoords, dtype=np.float64),
        np.asarray(uv_face_list, dtype=np.int32),
        np.asarray(pos_face_list, dtype=np.int32),
    )


# ------------------------------------------------------------------
# Face adjacency (via vertex position indices for correct topology)
# ------------------------------------------------------------------

def _build_pos_face_adjacency(pos_faces: np.ndarray) -> list[list[int]]:
    """Build face adjacency by shared edges on *position* indices."""
    n = pos_faces.shape[0]
    adj: list[set[int]] = [set() for _ in range(n)]
    edge_map: dict[tuple[int, int], int] = {}
    for fi in range(n):
        a, b, c = int(pos_faces[fi, 0]), int(pos_faces[fi, 1]), int(pos_faces[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            prev = edge_map.get(edge)
            if prev is not None:
                adj[fi].add(prev)
                adj[prev].add(fi)
            else:
                edge_map[edge] = fi
    return [sorted(s) for s in adj]


# ------------------------------------------------------------------
# BFS colour propagation
# ------------------------------------------------------------------

def _bfs_propagate_face_colors(
    face_avg_color: np.ndarray,
    textured_mask: np.ndarray,
    adjacency: list[list[int]],
) -> np.ndarray:
    """BFS from textured faces → untextured faces, propagating avg colour.

    Each untextured face receives the colour of its nearest textured
    neighbour (by topological distance).

    Returns per-face RGB float32 (N, 3) in [0, 255].
    """
    n = len(textured_mask)
    result = face_avg_color.copy()
    visited = textured_mask.copy()
    queue: deque[int] = deque()

    for fi in range(n):
        if textured_mask[fi]:
            queue.append(fi)

    while queue:
        fi = queue.popleft()
        for nb in adjacency[fi]:
            if not visited[nb]:
                visited[nb] = True
                result[nb] = result[fi]
                queue.append(nb)

    return result


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def fill_untextured_faces(
    obj_path: str | Path,
    texture_path: str | Path,
    *,
    ply_path: str | Path | None = None,
    empty_threshold: int = _UNTEXTURED_FILL_EMPTY_THRESHOLD,
    empty_fraction: float = _UNTEXTURED_FILL_EMPTY_FRACTION,
    inpaint_radius: int = _UNTEXTURED_FILL_INPAINT_RADIUS,
    progress_cb=None,
) -> dict[str, Any]:
    """Detect untextured faces and fill them.

    Returns dict with ``total_faces``, ``untextured_faces``,
    ``filled_texels``, ``method``.
    """
    obj_path = Path(obj_path)
    texture_path = Path(texture_path)
    stats: dict[str, Any] = {
        "total_faces": 0,
        "untextured_faces": 0,
        "filled_texels": 0,
        "method": "none",
    }

    # ---- 1. Parse OBJ (material_0 only) ----
    obj_verts, texcoords, uv_faces, pos_faces = _parse_material0_faces(obj_path)
    n_faces = uv_faces.shape[0]
    stats["total_faces"] = n_faces
    if n_faces == 0:
        return stats

    # ---- 2. Load texture (V-flip to align with UV coordinate system) ----
    tex_bgr = cv2.imread(str(texture_path))
    if tex_bgr is None:
        print(f"  untextured fill: WARNING texture not found at {texture_path}")
        return stats
    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)[::-1].copy()
    tex_h, tex_w = tex_rgb.shape[:2]
    tex_res = max(tex_h, tex_w)

    _emit_progress(progress_cb, 92.0, "Rasterising UV for untextured detection")

    # ---- 3. Rasterize UV space ----
    try:
        from scripts.texture.vertex_color_bake import _rasterize_uv
        face_id_buf, bary_buf = _rasterize_uv(texcoords, uv_faces, tex_res)
    except Exception as exc:
        print(f"  untextured fill: WARNING rasterisation failed: {exc}")
        return stats

    _emit_progress(progress_cb, 94.0, "Classifying face texture coverage")

    # ---- 4. Classify texels and faces ----
    valid_mask = face_id_buf >= 0
    ys, xs = np.where(valid_mask)
    if ys.size == 0:
        return stats
    fids = face_id_buf[ys, xs]

    ys_c = np.minimum(ys, tex_h - 1)
    xs_c = np.minimum(xs, tex_w - 1)

    texel_rgb = tex_rgb[ys_c, xs_c]  # (N, 3) uint8
    texel_rgb_sum = texel_rgb.astype(np.int32).sum(axis=1)
    texel_is_empty = texel_rgb_sum < empty_threshold

    face_texel_count = np.bincount(fids, minlength=n_faces)
    face_empty_count = np.bincount(fids[texel_is_empty], minlength=n_faces)

    with np.errstate(divide="ignore", invalid="ignore"):
        face_empty_frac = np.where(
            face_texel_count > 0,
            face_empty_count / face_texel_count,
            0.0,
        )

    # ---- 5. Compute per-face average colour ----
    face_color_sum = np.zeros((n_faces, 3), dtype=np.float64)
    np.add.at(face_color_sum, fids, texel_rgb.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        face_avg_color = np.where(
            face_texel_count[:, None] > 0,
            face_color_sum / face_texel_count[:, None],
            0.0,
        ).astype(np.float32)
    face_avg_brightness = face_avg_color.sum(axis=1)

    # Three-tier classification:
    #   bright  = well-textured with visible colour → BFS seed
    #   needs_fill = mostly empty OR dark → will be filled by BFS
    #   (anything else stays as-is)
    _BRIGHT_FLOOR = 80.0  # face avg RGB sum must exceed this to be a seed
    mostly_empty = (face_empty_frac >= empty_fraction) & (face_texel_count > 0)
    dim_textured = (~mostly_empty) & (face_avg_brightness < _BRIGHT_FLOOR) & (face_texel_count > 0)
    needs_fill = mostly_empty | dim_textured
    bright_seed = (~needs_fill) & (face_texel_count > 0)

    n_untextured = int(mostly_empty.sum())
    n_dim = int(dim_textured.sum())
    n_fill = int(needs_fill.sum())
    n_seed = int(bright_seed.sum())
    stats["untextured_faces"] = n_fill

    print(
        f"  untextured fill: tex={tex_w}x{tex_h} "
        f"faces={n_faces} empty={n_untextured} dim={n_dim} "
        f"fill_target={n_fill} seeds={n_seed}"
    )

    if n_fill == 0:
        return stats

    # ---- 6. BFS propagation from bright seeds ----
    _emit_progress(progress_cb, 95.0, f"Filling {n_fill} faces via BFS")
    stats["method"] = "bfs_adjacency"
    adjacency = _build_pos_face_adjacency(pos_faces)
    propagated = _bfs_propagate_face_colors(face_avg_color, bright_seed, adjacency)

    # Fallback: faces unreachable from any bright seed get the global
    # average of all bright-seed faces (better than staying black).
    global_bright_avg = face_avg_color[bright_seed].mean(axis=0) if n_seed > 0 else np.zeros(3)
    still_dark = needs_fill & (propagated.sum(axis=1) < _BRIGHT_FLOOR)
    n_fallback = int(still_dark.sum())
    if n_fallback > 0:
        propagated[still_dark] = global_bright_avg
        print(f"  untextured fill: {n_fallback} unreachable faces → global avg")

    # Write propagated colours into all needs_fill texels
    fill_face_ids = np.where(needs_fill)[0]
    texel_in_fill = np.isin(fids, fill_face_ids)
    fill_ys = ys_c[texel_in_fill]
    fill_xs = xs_c[texel_in_fill]
    fill_fids = fids[texel_in_fill]
    filled_texels = int(fill_ys.size)
    stats["filled_texels"] = filled_texels

    colors_u8 = np.clip(propagated[fill_fids], 0, 255).astype(np.uint8)
    tex_rgb[fill_ys, fill_xs] = colors_u8

    # ---- 7. Seam padding — dilate coloured texels into inter-chart black ----
    # Texture filtering samples across UV chart boundaries; black padding
    # causes dark seams.  Iterative 4-neighbour dilation pushes colours
    # outward, matching the seam-pad logic in bake.py / image_utils.py.
    _PAD_ITERS = 16
    colored = (tex_rgb.astype(np.int32).sum(axis=2) > 0).astype(np.float32)
    kern4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    tex_f = tex_rgb.astype(np.float32)
    for _ in range(_PAD_ITERS):
        empty = colored == 0
        if not empty.any():
            break
        nc = cv2.filter2D(colored, -1, kern4, borderType=cv2.BORDER_CONSTANT)
        fill_here = empty & (nc > 0)
        if not fill_here.any():
            break
        for ch in range(3):
            ns = cv2.filter2D(tex_f[:, :, ch], -1, kern4, borderType=cv2.BORDER_CONSTANT)
            tex_f[:, :, ch] = np.where(fill_here, ns / np.maximum(nc, 1), tex_f[:, :, ch])
        colored = np.where(fill_here, 1.0, colored)
    tex_rgb = np.clip(tex_f, 0, 255).astype(np.uint8)

    print(
        f"  untextured fill: {n_fill} faces ({n_untextured} empty + {n_dim} dim), "
        f"{filled_texels} texels filled (bfs_adjacency + {_PAD_ITERS}-iter pad)"
    )

    _emit_progress(progress_cb, 98.0, "Writing filled texture")

    # ---- 8. Write back (V-flip to OBJ convention) ----
    tex_bgr_out = cv2.cvtColor(tex_rgb[::-1], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(texture_path), tex_bgr_out)

    return stats
