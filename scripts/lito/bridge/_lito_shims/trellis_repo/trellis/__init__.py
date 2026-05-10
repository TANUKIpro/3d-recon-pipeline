"""Inference-only stub package for `trellis`.

ml-lito's `lito.integrations.trellis` adds `$TRELLIS_REPO_DIR` (default
`<ml-lito>/third_party/TRELLIS`) to `sys.path`, then does
`import trellis.models`. The real Microsoft TRELLIS repo is not vendored
in our build (it would add gigabytes of weights for a code path the
3d-recon-pipeline never exercises during inference).

We point `TRELLIS_REPO_DIR` at this shim package via the lito bridge
runner; only module-import succeeds, and any actual call into
`trellis.*` raises a clear error.
"""
