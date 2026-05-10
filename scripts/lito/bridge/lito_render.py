#!/usr/bin/env python
"""LiTo Gaussians multi-view depth rendering — runs inside /opt/ml-lito/.venv.

Companion to lito_infer.py. Loads a Gaussian splats PLY produced by LiTo
image-to-3D inference and renders RGB / depth / alpha per pose using
gsplat (via plibs.gs_utils.render_3dgs_gsplat). The output feeds Phase 3's
TSDF fusion in the clip2mesh main image.

Inputs:
  --in-ply     <path>  Gaussians PLY in LiTo canonical frame.
  --views-json <path>  JSON written by scripts.lito.view_sampler.save_views_json.
  --out-dir    <path>  Directory to write per-view PNG/NPY artefacts.
  --device     <cuda:0|cpu>
  --sh-degree  <int>   Override SH degree (default: read from PLY).

Per-view outputs:
  rgb_NNNN.png      uint8 RGB (premultiplied straight from gsplat)
  depth_NNNN.npy    float32 expected z-depth in canonical-frame metres (0 = no hit)
  alpha_NNNN.npy    float32 [0, 1] alpha mask
  meta_NNNN.json    {"index", "label", "is_input_view", "depth_min", "depth_max"}
  summary.json      total render time, vram, error if any
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

# ml-lito's training-time imports (e.g. apple_fsspec, blender_rendering)
# are not on PyPI / not vendored. Prepend our sibling _lito_shims/ to
# sys.path so those imports resolve to inference-only stubs.
# See scripts/lito/bridge/_lito_shims/.
_SHIMS_DIR = Path(__file__).resolve().parent / "_lito_shims"
if str(_SHIMS_DIR) not in sys.path:
    sys.path.insert(0, str(_SHIMS_DIR))

# Match lito_infer.py: prefer the real Microsoft TRELLIS clone when
# present, fall back to the inference-only stub otherwise.
_REAL_TRELLIS = Path("/opt/ml-lito/third_party/TRELLIS")
if not _REAL_TRELLIS.exists():
    os.environ.setdefault("TRELLIS_REPO_DIR", str(_SHIMS_DIR / "trellis_repo"))

# gsplat JIT-compiles a CUDA extension via ninja; ninja ships in the
# ml-lito venv. Make sure the venv bin is on PATH even when the runner
# (loaded into the dashboard's uvicorn process) hasn't been refreshed
# with a venv-PATH env shim.
_VENV_BIN = "/opt/ml-lito/.venv/bin"
if Path(_VENV_BIN).exists() and _VENV_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _VENV_BIN + os.pathsep + os.environ.get("PATH", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="LiTo Gaussians multi-view renderer")
    parser.add_argument("--in-ply", required=True)
    parser.add_argument("--views-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sh-degree", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary: dict = {
        "status": "ok",
        "in_ply": args.in_ply,
        "views_json": args.views_json,
        "out_dir": str(out_dir),
        "n_views": 0,
        "total_seconds": 0.0,
        "peak_vram_mb": 0.0,
        "error": None,
    }

    try:
        import numpy as np
        import torch
        from PIL import Image

        from plibs import gs_utils

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device={args.device} requested but CUDA is not available"
            )
        device = torch.device(args.device)

        # Load Gaussians (returns plibs.gs_utils.Gaussians).
        gs = gs_utils.Gaussians.load_ply(
            filename=args.in_ply,
            sh_degree=args.sh_degree,
            device=device,
        )

        # Resolve activations to dense tensors expected by render_3dgs_gsplat.
        xyz_w = gs.xyz_w  # (n, 3)
        scaling = gs.scaling if gs.scaling is not None else gs.get_scaling()
        quaternion = gs.quaternion if gs.quaternion is not None else gs.get_rotation()
        opacity = gs.opacity if gs.opacity is not None else gs.get_opacity()
        rgb_sh = gs.get_rgb_sh
        sh_degree = gs.sh_degree

        with open(args.views_json) as fh:
            views_payload = json.load(fh)
        views = views_payload["views"]
        summary["n_views"] = len(views)

        t_start = time.time()
        for v in views:
            idx = int(v["index"])
            H_c2w = torch.tensor(v["H_c2w"], dtype=torch.float, device=device)
            K = torch.tensor(v["K"], dtype=torch.float, device=device)
            h, w = int(v["image_size"][0]), int(v["image_size"][1])

            with torch.no_grad():
                # plibs.gs_utils.render_3dgs_gsplat returns a dict with
                # premultiplied_rgb / premultiplied_depth / alpha tensors.
                render_out = gs_utils.render_3dgs_gsplat(
                    H_c2w=H_c2w,
                    intrinsic=K,
                    width_px=w,
                    height_px=h,
                    sh_degree=sh_degree,
                    xyz_w=xyz_w,
                    scaling=scaling,
                    quaternion=quaternion,
                    opacity=opacity,
                    rgb_sh=rgb_sh,
                    render_depth=True,
                    depth_mode="expectation",
                )
            rgb = render_out["premultiplied_rgb"]
            depth = render_out["premultiplied_depth"]
            alpha = render_out["alpha"]

            # gsplat returns (h, w, 3) for rgb (premultiplied), (h, w) for depth,
            # (h, w, 1) for alpha. Strip extra trailing dims and move to CPU.
            rgb_np = (rgb.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(
                "uint8"
            )
            depth_np = depth.detach().cpu().numpy().astype("float32")
            alpha_np = alpha.detach().cpu().numpy().astype("float32")
            if alpha_np.ndim == 3 and alpha_np.shape[-1] == 1:
                alpha_np = alpha_np[..., 0]

            Image.fromarray(rgb_np).save(out_dir / f"rgb_{idx:04d}.png")
            np.save(out_dir / f"depth_{idx:04d}.npy", depth_np)
            np.save(out_dir / f"alpha_{idx:04d}.npy", alpha_np)

            valid = depth_np[(alpha_np > 0.05) & (depth_np > 0)]
            d_min = float(valid.min()) if valid.size else 0.0
            d_max = float(valid.max()) if valid.size else 0.0
            (out_dir / f"meta_{idx:04d}.json").write_text(
                json.dumps(
                    {
                        "index": idx,
                        "label": v["label"],
                        "is_input_view": v["is_input_view"],
                        "depth_min": d_min,
                        "depth_max": d_max,
                        "alpha_coverage": float((alpha_np > 0.05).mean()),
                    }
                )
            )

        summary["total_seconds"] = time.time() - t_start
        if torch.cuda.is_available():
            summary["peak_vram_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )

    except SystemExit:
        raise
    except Exception:
        summary["status"] = "error"
        summary["error"] = traceback.format_exc()
        print(summary["error"], file=sys.stderr)

    summary_path.write_text(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
