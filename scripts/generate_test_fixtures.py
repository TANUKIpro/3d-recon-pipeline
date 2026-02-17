"""Generate lightweight test fixtures from real pipeline output.

Downsamples ``data/output/objects/coffee01/`` into
``tests/fixtures/coffee01/`` so that backend and frontend tests can
exercise stage-detection, file I/O and 3D preview without shipping
hundreds of megabytes.

Usage (inside Docker or anywhere open3d + numpy + Pillow are available)::

    python3 scripts/generate_test_fixtures.py
    python3 scripts/generate_test_fixtures.py --source /path/to/src --dest /path/to/dst
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample_point_cloud(src: Path, dst: Path, n_points: int) -> None:
    """Read a PLY point cloud, keep *n_points* random vertices."""
    pcd = o3d.io.read_point_cloud(str(src))
    pts = np.asarray(pcd.points)
    if len(pts) <= n_points:
        shutil.copy2(src, dst)
        return
    idx = np.random.default_rng(42).choice(len(pts), size=n_points, replace=False)
    idx.sort()
    down = pcd.select_by_index(idx.tolist())
    o3d.io.write_point_cloud(str(dst), down, write_ascii=False)


def _simplify_mesh(src: Path, dst: Path, target_faces: int) -> None:
    """Read a PLY triangle mesh, simplify to *target_faces* via vertex clustering."""
    mesh = o3d.io.read_triangle_mesh(str(src))
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0:
        shutil.copy2(src, dst)
        return
    if len(faces) <= target_faces:
        shutil.copy2(src, dst)
        return
    # Estimate voxel size via binary search to hit target face count
    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    lo, hi = diag * 0.001, diag * 0.5
    best = mesh
    for _ in range(30):
        mid = (lo + hi) / 2
        simplified = mesh.simplify_vertex_clustering(mid)
        n = len(np.asarray(simplified.triangles))
        if n > target_faces:
            lo = mid
        else:
            hi = mid
            best = simplified
        if abs(n - target_faces) < target_faces * 0.15:
            best = simplified
            break
    o3d.io.write_triangle_mesh(str(dst), best, write_ascii=False)


def _simplify_obj(src: Path, dst: Path, target_faces: int) -> None:
    """Simplify a textured OBJ by keeping first *target_faces* faces."""
    lines = src.read_text(encoding="utf-8").splitlines()
    header: list[str] = []
    vertices: list[str] = []
    texcoords: list[str] = []
    faces: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("v "):
            vertices.append(line)
        elif stripped.startswith("vt "):
            texcoords.append(line)
        elif stripped.startswith("f "):
            faces.append(line)
        else:
            header.append(line)

    # Keep only first target_faces faces
    faces = faces[:target_faces]

    # Determine which vertex/texcoord indices are actually used
    used_v: set[int] = set()
    used_vt: set[int] = set()
    for f in faces:
        parts = f.strip().split()[1:]  # skip "f"
        for p in parts:
            indices = p.split("/")
            used_v.add(int(indices[0]))
            if len(indices) > 1 and indices[1]:
                used_vt.add(int(indices[1]))

    # Re-index vertices
    old_to_new_v: dict[int, int] = {}
    new_vertices: list[str] = []
    for old_idx in sorted(used_v):
        old_to_new_v[old_idx] = len(new_vertices) + 1
        new_vertices.append(vertices[old_idx - 1])

    old_to_new_vt: dict[int, int] = {}
    new_texcoords: list[str] = []
    for old_idx in sorted(used_vt):
        old_to_new_vt[old_idx] = len(new_texcoords) + 1
        new_texcoords.append(texcoords[old_idx - 1])

    # Rewrite faces with new indices
    new_faces: list[str] = []
    for f in faces:
        parts = f.strip().split()
        new_parts = [parts[0]]  # "f"
        for p in parts[1:]:
            indices = p.split("/")
            new_vi = str(old_to_new_v[int(indices[0])])
            if len(indices) > 1 and indices[1]:
                new_vti = str(old_to_new_vt[int(indices[1])])
                new_parts.append(f"{new_vi}/{new_vti}")
            else:
                new_parts.append(new_vi)
        new_faces.append(" ".join(new_parts))

    out_lines = header + [""] + new_vertices + [""] + new_texcoords + [""] + new_faces + [""]
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _resize_image(src: Path, dst: Path, size: tuple[int, int],
                  resample: int = Image.LANCZOS) -> None:
    """Resize an image to (width, height)."""
    with Image.open(src) as img:
        resized = img.resize(size, resample=resample)
        resized.save(dst)


def _pick_frame_indices(total: int, count: int) -> list[int]:
    """Pick *count* evenly-spaced indices from [0, total)."""
    if total <= count:
        return list(range(total))
    step = (total - 1) / (count - 1)
    return [round(i * step) for i in range(count)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(source: Path, dest: Path) -> None:
    """Generate all fixture files from *source* into *dest*."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Source: {source}")
    print(f"Dest:   {dest}")

    # ---- Frames & Masks (5 of 59, resized to 108x192) --------------------
    frame_dir = source / "frames"
    mask_dir = source / "masks"
    frame_files = sorted(frame_dir.glob("*.jpg"))
    mask_files = sorted(mask_dir.glob("*.png"))
    total_frames = len(frame_files)
    target_count = 5
    indices = _pick_frame_indices(total_frames, target_count)
    print(f"Frames: {total_frames} → {len(indices)} (indices: {indices})")

    (dest / "frames").mkdir(exist_ok=True)
    (dest / "masks").mkdir(exist_ok=True)
    new_size = (108, 192)  # width x height
    for new_idx, src_idx in enumerate(indices):
        fname = f"{new_idx:05d}.jpg"
        _resize_image(frame_files[src_idx], dest / "frames" / fname, new_size)
        mname = f"{new_idx:05d}.png"
        _resize_image(
            mask_files[src_idx], dest / "masks" / mname, new_size,
            resample=Image.NEAREST,
        )
    print("  frames/ and masks/ done")

    # ---- Point clouds -----------------------------------------------------
    pc_specs = [
        ("object_full.ply", 500),
        ("object.ply", 300),
        ("object_denoised.ply", 300),
    ]
    for name, n in pc_specs:
        src_path = source / name
        if src_path.exists():
            _subsample_point_cloud(src_path, dest / name, n)
            print(f"  {name} → {n} points")

    # ---- Meshes (PLY) -----------------------------------------------------
    mesh_specs = [
        ("object_mesh.ply", 150),
        ("object_mesh_preview.ply", 150),
        ("object_mesh_wrapped.ply", 80),
        ("object_mesh_repaired.ply", 80),
    ]
    for name, target in mesh_specs:
        src_path = source / name
        if src_path.exists():
            _simplify_mesh(src_path, dest / name, target)
            print(f"  {name} → ~{target} faces")

    # ---- Textured mesh (OBJ + MTL) ----------------------------------------
    obj_src = source / "textured_mesh.obj"
    if obj_src.exists():
        _simplify_obj(obj_src, dest / "textured_mesh.obj", 80)
        print("  textured_mesh.obj → ~80 faces")
    mtl_src = source / "textured_mesh.mtl"
    if mtl_src.exists():
        shutil.copy2(mtl_src, dest / "textured_mesh.mtl")
        print("  textured_mesh.mtl copied")

    # ---- Texture map -------------------------------------------------------
    tex_src = source / "texture.png"
    if tex_src.exists():
        _resize_image(tex_src, dest / "texture.png", (64, 64))
        print("  texture.png → 64x64")

    # ---- pi3x_cache.npz (stub with tiny arrays) --------------------------
    stub_N = target_count
    stub_H, stub_W = 4, 4
    np.savez(
        dest / "pi3x_cache.npz",
        points=np.zeros((stub_N, stub_H, stub_W, 3), dtype=np.float16),
        conf_edge_mask=np.zeros((stub_N, stub_H, stub_W), dtype=bool),
        colors=np.zeros((stub_N, stub_H, stub_W, 3), dtype=np.float32),
        frame_indices=np.arange(stub_N, dtype=np.int32),
        source_indices=np.arange(stub_N, dtype=np.int32),
        N=np.int64(stub_N),
        H=np.int64(stub_H),
        W=np.int64(stub_W),
    )
    print("  pi3x_cache.npz stub created")

    # ---- camera_poses.json (keep poses matching selected indices) ----------
    poses_src = source / "camera_poses.json"
    if poses_src.exists():
        with open(poses_src, encoding="utf-8") as f:
            poses_data = json.load(f)
        selected_poses = [poses_data["poses"][i] for i in indices]
        with open(dest / "camera_poses.json", "w", encoding="utf-8") as f:
            json.dump({"poses": selected_poses}, f, indent=2)
        print(f"  camera_poses.json → {len(selected_poses)} poses")

    # ---- JSON files (copy as-is) ------------------------------------------
    for name in ("intrinsics.json", "object_meta.json"):
        src_path = source / name
        if src_path.exists():
            shutil.copy2(src_path, dest / name)
            print(f"  {name} copied")

    # ---- Summary -----------------------------------------------------------
    total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"\nTotal fixture size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    if total_size > 2 * 1024 * 1024:
        print("WARNING: Exceeds 2 MB target!")
    else:
        print("OK: Under 2 MB limit.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test fixtures from pipeline output")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/output/objects/coffee01"),
        help="Source output directory",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("tests/fixtures/coffee01"),
        help="Destination fixture directory",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    generate(args.source, args.dest)


if __name__ == "__main__":
    main()
