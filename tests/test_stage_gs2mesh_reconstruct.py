from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.stage_gs2mesh_reconstruct import (
    _Gs2meshRuntimeStack,
    _build_gs_train_args,
    _build_pythonpath,
    _build_stereo_runtime_stacks,
    _classify_gs2mesh_failure,
    _materialize_tsdf_masks,
    _patch_gaussian_renderer_init,
    _normalize_runtime_profile,
    _patch_gs2mesh_renderer_utils,
    _resolve_camera_frame_index,
    _validate_gs2mesh_stereo_outputs,
    run_gs2mesh,
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


class TestTsdfMaskMaterialization(unittest.TestCase):
    def test_resolve_camera_frame_index_from_image_name(self) -> None:
        frame_idx = _resolve_camera_frame_index(
            {"image_name": "frames/00017.jpg"}
        )

        self.assertEqual(frame_idx, 17)

    def test_resolve_camera_frame_index_raises_when_missing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Could not resolve frame index"):
            _resolve_camera_frame_index({"left_png": "rendered_left.png"})

    def test_materialize_tsdf_masks_writes_resized_left_mask(self) -> None:
        from PIL import Image

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks_dir = root / "masks"
            masks_dir.mkdir(parents=True)
            gs2mesh_output = root / "gs2mesh"
            view_dir = gs2mesh_output / "000"
            view_dir.mkdir(parents=True)

            Image.fromarray(
                np.array([[0, 255], [255, 255]], dtype=np.uint8)
            ).save(masks_dir / "00003.png")
            Image.fromarray(
                np.zeros((4, 6, 3), dtype=np.uint8)
            ).save(view_dir / "left.png")
            (gs2mesh_output / "camera_data.json").write_text(
                json.dumps(
                    [
                        {
                            "left": {
                                "image_name": "frames/00003.jpg",
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )

            _materialize_tsdf_masks(gs2mesh_output, masks_dir)

            saved = np.load(view_dir / "left_mask.npy")
            self.assertEqual(saved.shape, (4, 6))
            self.assertTrue(saved.dtype == np.bool_)
            self.assertTrue(saved.any())

    def test_materialize_tsdf_masks_requires_matching_mask_file(self) -> None:
        from PIL import Image

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks_dir = root / "masks"
            masks_dir.mkdir(parents=True)
            gs2mesh_output = root / "gs2mesh"
            view_dir = gs2mesh_output / "000"
            view_dir.mkdir(parents=True)
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
                view_dir / "left.png"
            )
            (gs2mesh_output / "camera_data.json").write_text(
                json.dumps([{"left": {"image_name": "frames/00009.jpg"}}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Missing canonical SAM2 mask"):
                _materialize_tsdf_masks(gs2mesh_output, masks_dir)


class TestRunGs2meshMaskIntegration(unittest.TestCase):
    @patch("scripts.stage_gs2mesh_reconstruct.shutil.copy2")
    @patch("scripts.gpu_tsdf.gpu_tsdf_reconstruct", return_value="/tmp/gpu_mesh.ply")
    @patch("scripts.stage_gs2mesh_reconstruct._materialize_tsdf_masks")
    @patch("scripts.stage_gs2mesh_reconstruct._run_gs2mesh_stereo_with_retries")
    @patch("scripts.stage_gs2mesh_reconstruct._ensure_gaussian_renderer_compat")
    @patch("scripts.stage_gs2mesh_reconstruct._ensure_gs2mesh_renderer_compat")
    @patch("scripts.stage_gs2mesh_reconstruct._run_subprocess")
    @patch("scripts.stage_gs2mesh_reconstruct._build_gs_train_args")
    @patch("scripts.stage_gs2mesh_reconstruct._ensure_gs_model_output_link")
    @patch("scripts.stage_gs2mesh_reconstruct._setup_gs2mesh_dirs")
    @patch("scripts.stage_gs2mesh_reconstruct._ensure_sparse_0")
    @patch("scripts.stage_gs2mesh_reconstruct._run_colmap_cmd")
    @patch("scripts.stage_gs2mesh_reconstruct._find_recon_dir")
    @patch("scripts.stage_gs2mesh_reconstruct._resolve_training_runtime_stack")
    def test_run_gs2mesh_materializes_masks_before_tsdf(
        self,
        mock_resolve_stack,
        mock_find_recon_dir,
        mock_run_colmap_cmd,
        mock_ensure_sparse_0,
        mock_setup_dirs,
        mock_output_link,
        mock_build_train_args,
        mock_run_subprocess,
        mock_renderer_compat,
        mock_gaussian_compat,
        mock_run_stereo,
        mock_materialize_masks,
        mock_gpu_tsdf,
        mock_copy2,
    ) -> None:
        del mock_run_colmap_cmd, mock_ensure_sparse_0, mock_setup_dirs
        del mock_output_link, mock_run_subprocess, mock_copy2
        del mock_renderer_compat, mock_gaussian_compat

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            frames_dir = Path(tmp) / "frames"
            colmap_sparse = Path(tmp) / "colmap_sparse"
            frames_dir.mkdir(parents=True)
            colmap_sparse.mkdir(parents=True)
            mock_find_recon_dir.return_value = colmap_sparse
            mock_resolve_stack.return_value = _Gs2meshRuntimeStack(
                name="accel",
                python_executable="python3",
                gaussian_splatting_root=Path("/tmp/gs"),
                env_overrides={},
            )
            mock_build_train_args.return_value = (
                ["python3", "train.py"],
                "auto",
                "default",
                None,
            )

            call_order: list[str] = []
            mock_materialize_masks.side_effect = (
                lambda *args, **kwargs: call_order.append("mask")
            )
            mock_gpu_tsdf.side_effect = (
                lambda *args, **kwargs: call_order.append("tsdf") or "/tmp/gpu_mesh.ply"
            )

            run_gs2mesh(
                str(frames_dir),
                str(colmap_sparse),
                str(Path(tmp) / "masks"),
                str(out),
                use_masks=True,
            )

            self.assertEqual(call_order, ["mask", "tsdf"])
            mock_run_stereo.assert_called_once()
            mock_materialize_masks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
