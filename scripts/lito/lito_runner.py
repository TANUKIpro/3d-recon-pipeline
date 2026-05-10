"""Subprocess bridge to /opt/ml-lito/.venv (Apple ml-lito inference).

The clip2mesh main image runs torch 2.5 + CUDA 12.1 + numpy 1.x; ml-lito
requires torch 2.9 + CUDA 12.8 + numpy 2.x. We isolate ml-lito in its own
venv (Phase 0a §7.1) and invoke it as a subprocess. This module is the
clip2mesh-side caller — it ships the RGBA input, waits, parses the meta
JSON, and returns the path to the resulting Gaussians PLY.

The actual inference entry point lives in `scripts/lito/bridge/lito_infer.py`
and is COPYed into /opt/ml-lito-bridge inside the Docker image.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LitoInferenceResult:
    """Result payload from a successful subprocess invocation."""

    out_ply: str
    meta: dict


class LitoSubprocessError(RuntimeError):
    """Raised when the LiTo subprocess returns a non-zero exit code."""


def run_lito_inference(
    in_rgba_path: str,
    out_ply_path: str,
    *,
    venv_python: str | None = None,
    bridge_script: str | None = None,
    checkpoint_dir: str | None = None,
    model_name: str | None = None,
    inference_steps: int | None = None,
    cfg_scale: float | None = None,
    device: str = "cuda:0",
    timeout_s: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> LitoInferenceResult:
    """Invoke ml-lito's image-to-3D pipeline through the isolated venv.

    Defaults are pulled from scripts.config_defaults so callers only need
    to specify input/output paths in the common case.
    """
    from scripts.config_defaults import (
        LITO_BRIDGE_SCRIPT,
        LITO_CFG_SCALE,
        LITO_CHECKPOINT_DIR,
        LITO_INFERENCE_STEPS,
        LITO_MODEL_NAME,
        LITO_SUBPROCESS_TIMEOUT_S,
        LITO_VENV_PYTHON,
    )

    venv_python = venv_python or LITO_VENV_PYTHON
    bridge_script = bridge_script or LITO_BRIDGE_SCRIPT
    checkpoint_dir = checkpoint_dir or LITO_CHECKPOINT_DIR
    model_name = model_name or LITO_MODEL_NAME
    inference_steps = inference_steps if inference_steps is not None else LITO_INFERENCE_STEPS
    cfg_scale = cfg_scale if cfg_scale is not None else LITO_CFG_SCALE
    timeout_s = timeout_s if timeout_s is not None else LITO_SUBPROCESS_TIMEOUT_S

    if not Path(venv_python).exists():
        raise LitoSubprocessError(
            f"lito venv python not found at {venv_python}. "
            f"Run Phase 0c to provision /opt/ml-lito/.venv."
        )
    if not Path(bridge_script).exists():
        raise LitoSubprocessError(
            f"lito bridge script not found at {bridge_script}. "
            f"Ensure Dockerfile copies scripts/lito/bridge/lito_infer.py."
        )

    out_ply_path_p = Path(out_ply_path)
    out_ply_path_p.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_ply_path_p.with_suffix(".meta.json")

    cmd = [
        venv_python,
        bridge_script,
        "--in-rgba",
        in_rgba_path,
        "--out-ply",
        str(out_ply_path),
        "--meta",
        str(meta_path),
        "--checkpoint-dir",
        checkpoint_dir,
        "--model",
        model_name,
        "--steps",
        str(int(inference_steps)),
        "--cfg",
        str(float(cfg_scale)),
        "--device",
        device,
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"[lito] subprocess: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LitoSubprocessError(
            f"lito subprocess timed out after {timeout_s}s"
        ) from exc

    if proc.returncode != 0:
        raise LitoSubprocessError(
            f"lito subprocess failed with exit code {proc.returncode} "
            f"(see logs above)"
        )

    if not Path(out_ply_path).exists():
        raise LitoSubprocessError(
            f"lito subprocess returned 0 but {out_ply_path} was not produced"
        )

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[lito] WARNING: meta JSON unparseable ({exc}); ignoring")

    return LitoInferenceResult(out_ply=str(out_ply_path), meta=meta)


@dataclass(frozen=True)
class LitoRenderResult:
    """Result of a Gaussians multi-view rendering pass."""

    out_dir: str
    summary: dict


def run_lito_render(
    in_ply_path: str,
    views_json_path: str,
    out_dir: str,
    *,
    venv_python: str | None = None,
    bridge_script: str | None = None,
    device: str = "cuda:0",
    sh_degree: int | None = None,
    timeout_s: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> LitoRenderResult:
    """Render Gaussians from many canonical-frame views via the isolated venv.

    The bridge script (`scripts/lito/bridge/lito_render.py`) writes per-view
    rgb / depth / alpha files plus a summary JSON. clip2mesh's TSDF driver
    consumes those artefacts in the main image (torch 2.5 / numpy 1.x).
    """
    from scripts.config_defaults import (
        LITO_BRIDGE_SCRIPT,
        LITO_SUBPROCESS_TIMEOUT_S,
        LITO_VENV_PYTHON,
    )

    venv_python = venv_python or LITO_VENV_PYTHON
    if bridge_script is None:
        bridge_script = str(Path(LITO_BRIDGE_SCRIPT).with_name("lito_render.py"))
    timeout_s = timeout_s if timeout_s is not None else LITO_SUBPROCESS_TIMEOUT_S

    if not Path(venv_python).exists():
        raise LitoSubprocessError(
            f"lito venv python not found at {venv_python}. "
            f"Run Phase 0c to provision /opt/ml-lito/.venv."
        )
    if not Path(bridge_script).exists():
        raise LitoSubprocessError(
            f"lito render bridge script not found at {bridge_script}. "
            f"Ensure Dockerfile copies scripts/lito/bridge/lito_render.py."
        )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    cmd = [
        venv_python,
        bridge_script,
        "--in-ply",
        in_ply_path,
        "--views-json",
        views_json_path,
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    if sh_degree is not None:
        cmd.extend(["--sh-degree", str(int(sh_degree))])

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"[lito] render subprocess: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise LitoSubprocessError(
            f"lito render subprocess timed out after {timeout_s}s"
        ) from exc
    if proc.returncode != 0:
        raise LitoSubprocessError(
            f"lito render subprocess failed with exit code {proc.returncode}"
        )

    summary_path = out_dir_p / "summary.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[lito] WARNING: render summary unparseable ({exc})")
    if summary.get("status") and summary["status"] != "ok":
        raise LitoSubprocessError(
            f"lito render produced summary error: {summary.get('error', '<none>')}"
        )
    return LitoRenderResult(out_dir=str(out_dir), summary=summary)
