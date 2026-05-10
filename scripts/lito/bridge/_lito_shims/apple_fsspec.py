"""Inference-only stub for `apple_fsspec`.

Apple's ml-lito training-time dataset module
(`lito/datasets/obj_wdset.py`) imports `apple_fsspec` at module load
time, but the package is not on PyPI and is unused for inference. The
3d-recon-pipeline only invokes ml-lito for inference (`load_model` →
`inference_sample_latent` → `save_ply`), so a minimal stub is enough
to satisfy the import chain.

The stub raises a clear error if any attribute is actually accessed,
to surface the situation if a future code path starts depending on it.

Loaded by prepending `_lito_shims/` to PYTHONPATH when the lito
subprocess is launched (see `scripts/lito/lito_runner.py`).
"""

# Bare-module stub: ml-lito imports this at module load time but only
# uses it inside training-only dataset code paths. We keep the module
# empty (no custom __getattr__) so introspection tools (lightning's
# inspect.stack(), pickle, etc.) see standard module dunders. If a
# future code path actually touches an attribute, AttributeError will
# surface naturally and point at the missing real package.
