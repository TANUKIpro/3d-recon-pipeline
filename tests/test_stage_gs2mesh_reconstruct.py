from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.stage_gs2mesh_reconstruct import (
    _build_gs_train_args,
    _build_pythonpath,
    _build_stereo_runtime_stacks,
    _classify_gs2mesh_failure,
    _patch_gaussian_renderer_init,
    _normalize_runtime_profile,
    _patch_gs2mesh_renderer_utils,
    _validate_gs2mesh_stereo_outputs,
)


class TestGsTrainArgs(unittest.TestCase):
    def test_invalid_runtime_profile_falls_back_to_auto(self) -> None:
        self.assertEqual(_normalize_runtime_profile("invalid"), "auto")

    @patch("scripts.stage_gs2mesh_reconstruct._sparse_adam_available")
    def test_auto_profile_uses_sparse_adam_when_available(
        self,
        mock_sparse_adam_available,
    ) -> None:
        mock_sparse_adam_available.return_value = True

        args, resolved_profile, optimizer_type, fallback_reason = (
            _build_gs_train_args(
                Path("/tmp/source"),
                Path("/tmp/model"),
                5000,
                "auto",
            )
        )

        self.assertEqual(resolved_profile, "auto")
        self.assertEqual(optimizer_type, "sparse_adam")
        self.assertIsNone(fallback_reason)
        self.assertIn("--disable_viewer", args)
        self.assertEqual(
            args[args.index("--test_iterations") + 1],
            "-1",
        )
        self.assertEqual(
            args[args.index("--save_iterations") + 1],
            "5001",
        )
        self.assertEqual(
            args[args.index("--optimizer_type") + 1],
            "sparse_adam",
        )

    @patch("scripts.stage_gs2mesh_reconstruct._sparse_adam_available")
    def test_auto_profile_falls_back_to_default_when_unavailable(
        self,
        mock_sparse_adam_available,
    ) -> None:
        mock_sparse_adam_available.return_value = False

        args, resolved_profile, optimizer_type, fallback_reason = (
            _build_gs_train_args(
                Path("/tmp/source"),
                Path("/tmp/model"),
                30000,
                "auto",
            )
        )

        self.assertEqual(resolved_profile, "auto")
        self.assertEqual(optimizer_type, "default")
        self.assertEqual(fallback_reason, "SparseGaussianAdam unavailable")
        self.assertEqual(
            args[args.index("--optimizer_type") + 1],
            "default",
        )

    @patch("scripts.stage_gs2mesh_reconstruct._sparse_adam_available")
    def test_compat_profile_forces_default_optimizer(
        self,
        mock_sparse_adam_available,
    ) -> None:
        mock_sparse_adam_available.return_value = True

        args, resolved_profile, optimizer_type, fallback_reason = (
            _build_gs_train_args(
                Path("/tmp/source"),
                Path("/tmp/model"),
                7000,
                "compat",
            )
        )

        self.assertEqual(resolved_profile, "compat")
        self.assertEqual(optimizer_type, "default")
        self.assertIsNone(fallback_reason)
        self.assertEqual(
            args[args.index("--optimizer_type") + 1],
            "default",
        )

    def test_custom_python_and_gaussian_splatting_root_are_used(self) -> None:
        args, _, _, _ = _build_gs_train_args(
            Path("/tmp/source"),
            Path("/tmp/model"),
            1234,
            "auto",
            python_executable="/opt/custom/bin/python",
            gaussian_splatting_root=Path("/opt/gaussian-splatting-compat"),
        )

        self.assertEqual(args[0], "/opt/custom/bin/python")
        self.assertEqual(
            args[2],
            "/opt/gaussian-splatting-compat/train.py",
        )


class TestGs2meshRendererPatch(unittest.TestCase):
    def test_renderer_patch_adds_missing_gaussian_splatting_args(self) -> None:
        original = """import copy

args = Namespace(compute_cov3D_python=False, 
                         convert_SHs_python=False, 
                         data_device=self.device, 
                         debug=False, 
                         eval=False, 
                         feature_dim=32, 
                         feature_model_path='', 
                         idx=0, 
                         images='images', 
                         init_from_3dgs_pcd=False)
                view = cameras.Camera(0, R, T, FoVx, FoVy, torch.rand(3,h,w), None, "abcd", 0)
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "renderer_utils.py"
            path.write_text(original, encoding="utf-8")

            changed = _patch_gs2mesh_renderer_utils(path)

            self.assertTrue(changed)
            patched = path.read_text(encoding="utf-8")
            self.assertIn("from PIL import Image", patched)
            self.assertIn("depths=''", patched)
            self.assertIn("train_test_exp=False", patched)
            self.assertIn("antialiasing=False", patched)
            self.assertIn("dummy_image = Image.fromarray", patched)
            self.assertIn("data_device=self.device", patched)

    def test_renderer_patch_is_idempotent(self) -> None:
        original = """import copy
from PIL import Image

args = Namespace(eval=False, 
                         train_test_exp=False, 
                         antialiasing=False, 
                         images='images', 
                         depths='', 
                         init_from_3dgs_pcd=False)
                dummy_image = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
                view = cameras.Camera(
                    (w, h),
                    0,
                    R,
                    T,
                    FoVx,
                    FoVy,
                    None,
                    dummy_image,
                    None,
                    f"{camera_name}",
                    camera_number,
                    data_device=self.device,
                )
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "renderer_utils.py"
            path.write_text(original, encoding="utf-8")

            changed = _patch_gs2mesh_renderer_utils(path)

            self.assertFalse(changed)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


class TestGaussianRendererPatch(unittest.TestCase):
    def test_gaussian_renderer_patch_adds_fallback_for_unsupported_kwargs(
        self,
    ) -> None:
        original = """import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "__init__.py"
            path.write_text(original, encoding="utf-8")

            changed = _patch_gaussian_renderer_init(path)

            self.assertTrue(changed)
            patched = path.read_text(encoding="utf-8")
            self.assertIn("def _build_raster_settings(**kwargs):", patched)
            self.assertIn("def _unpack_rasterizer_output(output):", patched)
            self.assertIn('if hasattr(pipe, "antialiasing")', patched)
            self.assertIn('if hasattr(pipe, "train_test_exp")', patched)
            self.assertIn("raster_settings = _build_raster_settings(**raster_settings_kwargs)", patched)
            self.assertIn("rasterizer_output = rasterizer(", patched)
            self.assertIn(
                "rendered_image, radii, depth_image = _unpack_rasterizer_output(",
                patched,
            )

    def test_gaussian_renderer_patch_is_idempotent(self) -> None:
        original = """import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def _build_raster_settings(**kwargs):
    try:
        return GaussianRasterizationSettings(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        retry_kwargs = dict(kwargs)
        for key in ("antialiasing", "train_test_exp"):
            if key in retry_kwargs and key in str(exc):
                retry_kwargs.pop(key, None)
                return _build_raster_settings(**retry_kwargs)
        raise

def _unpack_rasterizer_output(output):
    if not isinstance(output, (tuple, list)):
        raise TypeError(
            "Unexpected rasterizer return type: "
            f"{type(output).__name__}"
        )
    if len(output) == 3:
        return output
    if len(output) == 2:
        rendered_image, radii = output
        return rendered_image, radii, None
    raise ValueError(
        "Unexpected rasterizer return arity: "
        f"{len(output)}"
    )

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    raster_settings_kwargs = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    if hasattr(pipe, "antialiasing"):
        raster_settings_kwargs["antialiasing"] = pipe.antialiasing
    if hasattr(pipe, "train_test_exp"):
        raster_settings_kwargs["train_test_exp"] = pipe.train_test_exp
    raster_settings = _build_raster_settings(**raster_settings_kwargs)
    if separate_sh:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rasterizer_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    rendered_image, radii, depth_image = _unpack_rasterizer_output(
        rasterizer_output
    )
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "__init__.py"
            path.write_text(original, encoding="utf-8")

            changed = _patch_gaussian_renderer_init(path)

            self.assertFalse(changed)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


class TestGs2meshRuntimeHelpers(unittest.TestCase):
    def test_build_pythonpath_prefers_selected_runtime(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "PYTHONPATH": (
                    "/app:/opt/gs2mesh:/opt/gaussian-splatting:"
                    "/opt/sam2:/tmp/custom"
                )
            },
            clear=False,
        ):
            pythonpath = _build_pythonpath(
                Path("/opt/gaussian-splatting-compat")
            )

        self.assertTrue(
            pythonpath.startswith(
                "/app:/opt/gs2mesh:/opt/gaussian-splatting-compat:/opt/sam2"
            )
        )
        self.assertNotIn(
            "/opt/gaussian-splatting:/opt/sam2",
            pythonpath,
        )
        self.assertTrue(pythonpath.endswith("/tmp/custom"))

    @patch("scripts.stage_gs2mesh_reconstruct._compat_runtime_stack")
    def test_auto_profile_adds_compat_fallback_when_available(
        self,
        mock_compat_runtime_stack,
    ) -> None:
        mock_compat_runtime_stack.return_value = object()

        stacks = _build_stereo_runtime_stacks("auto")

        self.assertEqual(len(stacks), 2)
        self.assertEqual(stacks[0].name, "accel")

    @patch("scripts.stage_gs2mesh_reconstruct._compat_runtime_stack")
    def test_auto_profile_skips_compat_when_unavailable(
        self,
        mock_compat_runtime_stack,
    ) -> None:
        mock_compat_runtime_stack.return_value = None

        stacks = _build_stereo_runtime_stacks("auto")

        self.assertEqual([stack.name for stack in stacks], ["accel"])

    def test_classify_cuda_illegal_memory_access(self) -> None:
        reason = _classify_gs2mesh_failure(
            "RuntimeError: CUDA error: an illegal memory access was encountered"
        )

        self.assertEqual(reason, "cuda_illegal_memory_access")

    def test_classify_renderer_signature_mismatch(self) -> None:
        reason = _classify_gs2mesh_failure(
            "TypeError: Camera.__init__() got an unexpected keyword argument"
        )

        self.assertEqual(reason, "renderer_signature_mismatch")

    def test_classify_rasterizer_return_arity_mismatch(self) -> None:
        reason = _classify_gs2mesh_failure(
            "ValueError: not enough values to unpack (expected 3, got 2)"
        )

        self.assertEqual(reason, "renderer_signature_mismatch")

    def test_validate_stereo_outputs_requires_camera_data_and_depth(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "camera_data.json").write_text("{}", encoding="utf-8")
            out_dir = root / "000" / "out_DLNR_Middlebury"
            out_dir.mkdir(parents=True)
            (out_dir / "depth.npy").write_bytes(b"depth")

            _validate_gs2mesh_stereo_outputs(root, "DLNR_Middlebury")

    def test_validate_stereo_outputs_rejects_missing_depth(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "camera_data.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Missing stereo outputs"):
                _validate_gs2mesh_stereo_outputs(root, "DLNR_Middlebury")


if __name__ == "__main__":
    unittest.main()
