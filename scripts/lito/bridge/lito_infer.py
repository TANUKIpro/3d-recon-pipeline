#!/usr/bin/env python
"""LiTo image-to-3D bridge — runs inside /opt/ml-lito/.venv.

Invoked by scripts/lito/lito_runner.py via subprocess. The clip2mesh main
image (torch 2.5 + numpy 1.x) cannot import this file directly; ml-lito
needs torch 2.9 + numpy 2.x + xformers + flash-attn + PyTorch3D + gsplat,
which we install only in the isolated venv (Phase 0a §7.1).

Inputs:
  --in-rgba   <path>  Pre-processed 518×518 RGBA PNG (clip2mesh produced).
  --out-ply   <path>  Where to write the 3D Gaussians PLY.
  --meta      <path>  Where to write the timing/diagnostic JSON.
  --checkpoint-dir <path>  Local cache directory for ml-lito weights.
  --model     <name>  Checkpoint stem (default lito_dit_rgba).
  --steps     <int>   Heun sampling steps (default 20).
  --cfg       <float> CFG scale (default 3.0).
  --device    <cuda:0|cpu|mps>

Outputs:
  out_ply: 3D Gaussian splat PLY in LiTo's canonical (object-centric) frame.
  meta:    JSON with timings, gaussian count, peak VRAM, exit status.

Exit codes: 0 success, 1 invocation error, 2 inference error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

LITO_CHECKPOINT_BASE_URL = "https://ml-site.cdn-apple.com/models/lito"


def _resolve_checkpoint(model_name: str, checkpoint_dir: str) -> str:
    """Return the URL or local file path that load_model() should consume.

    If the checkpoint already exists in checkpoint_dir, prefer the local
    path so we skip the CDN redownload check on every run.
    """
    local_path = Path(checkpoint_dir) / f"{model_name}.ckpt"
    if local_path.exists():
        return str(local_path)
    return f"{LITO_CHECKPOINT_BASE_URL}/{model_name}.ckpt"


def _peak_vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:  # pragma: no cover
        pass
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="LiTo image-to-3D bridge")
    parser.add_argument("--in-rgba", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model", default="lito_dit_rgba")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    Path(args.out_ply).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)

    timings: dict = {}
    status = "ok"
    error_message: str | None = None
    gaussian_count = 0
    sh_degree = 0
    peak_vram_mb = 0.0

    try:
        # Heavy imports stay inside try/except so meta JSON gets written
        # even when the venv is mis-provisioned.
        import numpy as np
        import torch
        from PIL import Image

        # ml-lito modules
        from lito.eval_scripts.st_model_utils import load_model
        from plibs import gs_utils, sh_utils

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device={args.device} requested but CUDA is not available"
            )
        device = torch.device(args.device)

        t0 = time.time()
        ckpt = _resolve_checkpoint(args.model, args.checkpoint_dir)
        mdict = load_model(
            checkpoint_url=ckpt,
            download_dir_root=args.checkpoint_dir,
            overwrite=False,
            dtype=torch.float,
            device=device,
            load_params=True,
        )
        model = mdict["model"]
        model.to(device=device)
        model.eval()
        model.freeze()
        st_model = model.pretrained_tokenizer
        timings["load_model"] = time.time() - t0

        # Load pre-composed 518×518 RGBA. clip2mesh has already applied the
        # bbox crop + letterbox + resize, so we run preprocess with crop=False
        # and remove_bg=False to consume the alpha as-is.
        img = Image.open(args.in_rgba).convert("RGBA")
        rgba_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 4) [0,1]
        rgb = rgba_np[..., :3] * rgba_np[..., 3:4]  # premultiply
        cond_rgba = torch.from_numpy(rgba_np).float()  # (H, W, 4)

        # (b=1, q=1, H, W, 4)
        cond_rgba_b = cond_rgba.unsqueeze(0).unsqueeze(0).to(device=device)

        t0 = time.time()
        if device.type == "cuda":
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                out_dict = model.inference_sample_latent(
                    cond_rgba=cond_rgba_b,
                    ode_sampling_method="heun",
                    ode_num_steps=int(args.steps),
                    cfg_scale=float(args.cfg),
                    use_ema=True,
                )
        else:
            with torch.no_grad():
                out_dict = model.inference_sample_latent_mlx(
                    cond_rgba=cond_rgba_b,
                    ode_sampling_method="heun",
                    ode_num_steps=int(args.steps),
                    cfg_scale=float(args.cfg),
                    use_ema=True,
                    mlx_compute_dtype="float16",
                )
        timings["sampling"] = time.time() - t0

        t0 = time.time()
        init_coord_src = (
            "voxel_decoder" if st_model.voxel_decoder is not None else "sample_xyz"
        )
        if device.type == "cuda":
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True):
                gs_dicts = st_model.inference_estimate_gaussians(
                    fpoint_latent=out_dict["unnormalized_latent"],
                    init_coord_src=init_coord_src,
                    steps_for_sample_xyz=50,
                )
        else:
            with torch.no_grad():
                gs_dicts = st_model.inference_estimate_gaussians_mlx(
                    fpoint_latent=out_dict["unnormalized_latent"],
                    init_coord_src=init_coord_src,
                    steps_for_sample_xyz=50,
                    mlx_compute_dtype="float16",
                )
        gs_dict = gs_dicts[0]
        timings["decoding"] = time.time() - t0

        sh_degree = sh_utils.get_sh_degree_from_total_dim(gs_dict["rgb_sh"].size(-2))
        ngs = math.prod(gs_dict["xyz_w"].shape[:-1])
        gaussian_count = int(ngs)
        gs = gs_utils.Gaussians(
            sh_degree=sh_degree,
            xyz_w=gs_dict["xyz_w"].reshape(ngs, 3),
            rgb_sh=gs_dict["rgb_sh"].reshape(ngs, -1, 3),
            rgb_sh_dc=None,
            rgb_sh_rest=None,
            scaling_logit=None,
            quaternion_prenorm=None,
            opacity_logit=None,
            scaling=gs_dict["scaling"].reshape(ngs, 3),
            quaternion=gs_dict["quaternion"].reshape(ngs, 4),
            opacity=gs_dict["opacity"].reshape(ngs, 1),
            min_scaling=0,
            scaling_activation_type="none",
        )

        t0 = time.time()
        gs.save_ply(filename=str(args.out_ply))
        timings["save_ply"] = time.time() - t0
        peak_vram_mb = _peak_vram_mb()

    except SystemExit:
        raise
    except Exception:
        status = "error"
        error_message = traceback.format_exc()
        print(error_message, file=sys.stderr)

    meta = {
        "status": status,
        "model": args.model,
        "steps": args.steps,
        "cfg": args.cfg,
        "device": args.device,
        "timings_s": timings,
        "gaussian_count": gaussian_count,
        "sh_degree": sh_degree,
        "peak_vram_mb": peak_vram_mb,
        "out_ply": args.out_ply,
        "in_rgba": args.in_rgba,
        "error": error_message,
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
