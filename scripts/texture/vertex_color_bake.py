"""Bake MILo vertex colors into a UV-mapped texture.

Reads a PLY mesh with per-vertex RGB colors, generates a UV atlas via xatlas,
rasterises the vertex colours into a texture image, and writes the same
OBJ + MTL + texture.png bundle that the multi-view bake produces so that
downstream stages (post-texture cleanup, preview) work unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

from scripts.texture.progress import ProgressCallback, _emit_progress


def bake_vertex_color_texture(
    mesh_ply: str,
    output_dir: str,
    tex_size: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Bake vertex colors from a PLY mesh into a UV-mapped textured OBJ.

    Args:
        mesh_ply: Path to vertex-colored PLY mesh (from MILo).
        output_dir: Output directory for OBJ/MTL/PNG.
        tex_size: Texture resolution. 0 or None = auto (1024).
        progress_cb: Optional ``(progress, detail)`` callback.

    Returns:
        Path to the output OBJ file.
    """
    import xatlas

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load mesh ---
    _emit_progress(progress_cb, 5.0, "Loading vertex-colored mesh")
    print("Loading vertex-colored mesh...")
    ply = PlyData.read(mesh_ply)
    v = ply["vertex"]
    vertices = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    f = ply["face"]
    faces = np.vstack(f["vertex_indices"]).astype(np.int32)
    print(f"  {len(vertices)} verts, {len(faces)} faces")

    # Read vertex colors (required)
    try:
        vert_colors = np.column_stack(
            [v["red"], v["green"], v["blue"]]
        ).astype(np.float64)
        # Normalise to [0, 1] if stored as uint8
        if vert_colors.max() > 1.5:
            vert_colors = vert_colors / 255.0
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"PLY mesh does not contain vertex colors (red/green/blue): {mesh_ply}"
        ) from exc
    print(f"  Vertex colors: loaded ({vert_colors.shape[0]} vertices)")

    # --- Resolve texture size ---
    if tex_size is None or tex_size <= 0:
        tex_size = max(1024, int(os.environ.get("TEXTURE_SIZE", "0")))
        if tex_size <= 0:
            tex_size = 1024
    print(f"  Texture size: {tex_size}x{tex_size}")

    # --- UV Atlas ---
    _emit_progress(progress_cb, 15.0, "Generating UV atlas")
    print("Generating UV atlas with xatlas...")
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces)
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]
    new_vertices = vertices[vmapping]
    print(f"  UV atlas: {len(new_vertices)} verts, {len(new_faces)} faces")

    # Map new vertices back to original vertex colors.
    # xatlas splits vertices at UV seams; vmapping maps new→original indices.
    new_vert_colors = vert_colors[vmapping].astype(np.float32)

    # --- Rasterise UV triangles → face_id + barycentric buffer ---
    _emit_progress(progress_cb, 30.0, "Rasterising UV atlas")
    print("Rasterising UV atlas...")

    face_id_buf, bary_buf = _rasterize_uv(uvs, new_faces, tex_size, progress_cb)

    # --- Fill texture from vertex colors via barycentric interpolation ---
    _emit_progress(progress_cb, 70.0, "Interpolating vertex colors")
    valid_mask = face_id_buf >= 0
    ys, xs = np.where(valid_mask)
    fids = face_id_buf[ys, xs]
    barys = bary_buf[ys, xs]  # (N, 3)

    # Gather per-face vertex colors
    c0 = new_vert_colors[new_faces[fids, 0]]  # (N, 3)
    c1 = new_vert_colors[new_faces[fids, 1]]
    c2 = new_vert_colors[new_faces[fids, 2]]

    # Barycentric interpolation
    colors = barys[:, 0:1] * c0 + barys[:, 1:2] * c1 + barys[:, 2:3] * c2

    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    tex[ys, xs] = colors

    n_valid = int(valid_mask.sum())
    print(f"  {n_valid} valid texels")

    # Fill empty texels via inpainting
    _emit_progress(progress_cb, 80.0, "Filling texture gaps")
    tex_u8 = np.clip(tex * 255, 0, 255).astype(np.uint8)
    invalid_u8 = (~valid_mask).astype(np.uint8) * 255
    if invalid_u8.any():
        tex_bgr = cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR)
        tex_bgr = cv2.inpaint(tex_bgr, invalid_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        tex_u8 = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)

    # --- Export OBJ + MTL + PNG ---
    _emit_progress(progress_cb, 90.0, "Exporting textured mesh")
    basename = "textured_mesh"
    obj_path = output_path / f"{basename}.obj"
    mtl_path = output_path / f"{basename}.mtl"
    tex_path = output_path / "texture.png"

    # Texture PNG (flip V for OBJ convention)
    tex_bgr = cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR)[::-1]
    cv2.imwrite(str(tex_path), tex_bgr)
    print(f"Saved: {tex_path}")

    # MTL
    with open(mtl_path, "w") as f_out:
        f_out.write(
            "newmtl material_0\nKa 0.0 0.0 0.0\nKd 1.0 1.0 1.0\n"
            "Ks 0.0 0.0 0.0\nd 1.0\nillum 1\nmap_Kd texture.png\n"
        )
    print(f"Saved: {mtl_path}")

    # OBJ
    with open(obj_path, "w") as f_out:
        lines = [f"mtllib {basename}.mtl\nusemtl material_0\n\n"]
        for vert in new_vertices:
            lines.append(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in new_faces:
            a, b, c = face + 1
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        f_out.write("".join(lines))
    print(f"Saved: {obj_path}")

    # Vertex color mode does not estimate intrinsics — rely on COLMAP's
    # intrinsics.json written during Stage 2.  Stage 6 gracefully handles a
    # missing file (mask-based passes simply become no-ops).
    # Clean up any broken stub from earlier vertex_color_bake versions.
    intrinsics_path = output_path / "intrinsics.json"
    if intrinsics_path.exists():
        _intrinsics_ok = False
        try:
            _idata = json.loads(intrinsics_path.read_text(encoding="utf-8"))
            _iw = int(_idata.get("image_width") or _idata.get("width") or 0)
            _ih = int(_idata.get("image_height") or _idata.get("height") or 0)
            _intrinsics_ok = _iw > 0 and _ih > 0
        except Exception:
            pass
        if _intrinsics_ok:
            print(f"  intrinsics.json: using existing ({intrinsics_path})")
        else:
            intrinsics_path.unlink()
            print("  intrinsics.json: removed invalid stub (missing image dimensions)")
    else:
        print("  intrinsics.json: not found (Stage 6 mask projection will be skipped)")

    _emit_progress(progress_cb, 100.0, "Vertex color texture bake complete")
    print("Vertex color texture bake complete.")
    return obj_path


def _rasterize_uv(
    uvs: np.ndarray,
    faces: np.ndarray,
    tex_res: int,
    progress_cb: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize UV atlas → (face_id_buf, bary_buf).

    Tries GPU (nvdiffrast) first, falls back to CPU (cv2.fillConvexPoly +
    vectorized barycentric).  Same output format as ``gpu_raster.gpu_rasterize_uv_space``.
    """
    # --- GPU path ---
    try:
        from scripts.texture.gpu_raster import gpu_rasterize_uv_space
        face_id_buf, bary_buf = gpu_rasterize_uv_space(uvs, faces, tex_res)
        print("  UV rasterisation: GPU")
        return face_id_buf, bary_buf
    except Exception:
        pass

    # --- CPU fallback ---
    print("  UV rasterisation: CPU")
    face_id_buf = np.full((tex_res, tex_res), -1, dtype=np.int32)
    uv_scaled = uvs * tex_res

    tri_coords = np.stack([
        uv_scaled[faces[:, 0]],
        uv_scaled[faces[:, 1]],
        uv_scaled[faces[:, 2]],
    ], axis=1).astype(np.int32).reshape(-1, 3, 1, 2)

    emit_every = max(1, len(faces) // 20)
    for fi in range(len(faces)):
        cv2.fillConvexPoly(face_id_buf, tri_coords[fi], int(fi))
        if progress_cb and fi % emit_every == 0:
            ratio = (fi + 1) / max(len(faces), 1)
            _emit_progress(progress_cb, 30.0 + ratio * 35.0,
                           f"Rasterising UV ({fi + 1}/{len(faces)} faces)")

    # Vectorized barycentric coords for all valid texels
    valid = face_id_buf >= 0
    ys, xs = np.where(valid)
    fids = face_id_buf[ys, xs]
    px, py = xs + 0.5, ys + 0.5
    fi0, fi1, fi2 = faces[fids, 0], faces[fids, 1], faces[fids, 2]
    uv0, uv1, uv2 = uv_scaled[fi0], uv_scaled[fi1], uv_scaled[fi2]
    denom = ((uv1[:, 1] - uv2[:, 1]) * (uv0[:, 0] - uv2[:, 0])
             + (uv2[:, 0] - uv1[:, 0]) * (uv0[:, 1] - uv2[:, 1]))
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    w0 = ((uv1[:, 1] - uv2[:, 1]) * (px - uv2[:, 0])
           + (uv2[:, 0] - uv1[:, 0]) * (py - uv2[:, 1])) / denom
    w1 = ((uv2[:, 1] - uv0[:, 1]) * (px - uv2[:, 0])
           + (uv0[:, 0] - uv2[:, 0]) * (py - uv2[:, 1])) / denom
    w2 = 1.0 - w0 - w1

    bary_buf = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    bary_buf[ys, xs] = np.column_stack([w0, w1, w2]).astype(np.float32)

    return face_id_buf, bary_buf
