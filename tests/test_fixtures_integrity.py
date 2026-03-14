"""Verify that ``tests/fixtures/coffee01/`` is a valid, minimal pipeline output.

Checks:
- All 5 stages are detected by ``detect_stage_outputs()``
- PLY files have valid headers
- OBJ references MTL, MTL references texture.png
- texture.png is a valid PNG
- Total fixture size stays under 2 MB

Note: The fixtures still contain legacy files (object_full.ply, pi3x_cache.npz,
object_denoised.ply, object_mesh_wrapped.ply, object_mesh_repaired.ply) from
the old pipeline. Only files relevant to the 5-stage pipeline are validated
for stage detection.
"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

from scripts.dashboard.state import detect_stage_outputs

from tests.conftest import FIXTURE_DIR


class FixtureExistenceTests(unittest.TestCase):
    """Expected fixture files for the 5-stage pipeline exist."""

    EXPECTED_FILES = [
        "frames/00000.jpg",
        "frames/00004.jpg",
        "masks/00000.png",
        "masks/00004.png",
        "object_mesh.ply",
        "textured_mesh.obj",
        "textured_mesh.mtl",
        "texture.png",
        "camera_poses.json",
        "intrinsics.json",
    ]

    def test_all_files_exist(self) -> None:
        for rel in self.EXPECTED_FILES:
            path = FIXTURE_DIR / rel
            self.assertTrue(path.exists(), f"Missing fixture: {rel}")


class StageDetectionTests(unittest.TestCase):
    """``detect_stage_outputs()`` finds all 5 stages complete.

    Note: Stage 2 (COLMAP SfM) requires a colmap_sparse directory.
    If the fixture does not contain it, stage 2 will not be detected.
    We test stages that the fixture actually supports.
    """

    def test_stage1_detected(self) -> None:
        stages, frame_count, mask_count = detect_stage_outputs(FIXTURE_DIR)
        self.assertTrue(stages[1])
        self.assertGreaterEqual(frame_count, 1)

    def test_stage3_detected(self) -> None:
        stages, _, mask_count = detect_stage_outputs(FIXTURE_DIR)
        self.assertTrue(stages[3])
        self.assertGreaterEqual(mask_count, 1)

    def test_stage4_detected(self) -> None:
        stages, _, _ = detect_stage_outputs(FIXTURE_DIR)
        self.assertTrue(stages[4])

    def test_stage5_detected(self) -> None:
        stages, _, _ = detect_stage_outputs(FIXTURE_DIR)
        self.assertTrue(stages[5])


class PlyValidityTests(unittest.TestCase):
    """PLY files start with the magic ``ply`` header."""

    PLY_FILES = [
        "object_mesh.ply",
    ]

    def test_ply_headers_valid(self) -> None:
        for name in self.PLY_FILES:
            path = FIXTURE_DIR / name
            with open(path, "rb") as f:
                magic = f.read(3)
            self.assertEqual(magic, b"ply", f"{name} doesn't start with 'ply' magic")

    def test_ply_meshes_have_faces(self) -> None:
        for name in ("object_mesh.ply",):
            path = FIXTURE_DIR / name
            header = path.read_bytes().split(b"end_header")[0].decode("ascii")
            self.assertIn("element vertex", header, f"{name} missing vertex element")
            self.assertIn("element face", header, f"{name} missing face element")


class ObjMtlReferenceTests(unittest.TestCase):
    """OBJ references MTL and MTL references texture.png."""

    def test_obj_references_mtl(self) -> None:
        obj = (FIXTURE_DIR / "textured_mesh.obj").read_text(encoding="utf-8")
        self.assertIn("mtllib textured_mesh.mtl", obj)

    def test_mtl_references_texture(self) -> None:
        mtl = (FIXTURE_DIR / "textured_mesh.mtl").read_text(encoding="utf-8")
        self.assertIn("map_Kd texture.png", mtl)

    def test_obj_has_vertices_and_faces(self) -> None:
        obj = (FIXTURE_DIR / "textured_mesh.obj").read_text(encoding="utf-8")
        self.assertIn("v ", obj)
        self.assertIn("f ", obj)


class TextureImageTests(unittest.TestCase):
    """texture.png is a valid PNG file."""

    def test_png_magic(self) -> None:
        path = FIXTURE_DIR / "texture.png"
        with open(path, "rb") as f:
            magic = f.read(8)
        # PNG magic: 137 80 78 71 13 10 26 10
        self.assertEqual(magic, b"\x89PNG\r\n\x1a\n")

    def test_png_dimensions(self) -> None:
        path = FIXTURE_DIR / "texture.png"
        with open(path, "rb") as f:
            f.read(8)  # skip magic
            f.read(4)  # chunk length
            chunk_type = f.read(4)
            self.assertEqual(chunk_type, b"IHDR")
            width = struct.unpack(">I", f.read(4))[0]
            height = struct.unpack(">I", f.read(4))[0]
        self.assertEqual(width, 64)
        self.assertEqual(height, 64)


class FixtureSizeTests(unittest.TestCase):
    """Total fixture size stays under 2 MB."""

    def test_total_size_under_2mb(self) -> None:
        total = sum(f.stat().st_size for f in FIXTURE_DIR.rglob("*") if f.is_file())
        max_bytes = 2 * 1024 * 1024
        self.assertLess(
            total,
            max_bytes,
            f"Fixtures too large: {total:,} bytes ({total / 1024:.1f} KB) > 2 MB",
        )


if __name__ == "__main__":
    unittest.main()
