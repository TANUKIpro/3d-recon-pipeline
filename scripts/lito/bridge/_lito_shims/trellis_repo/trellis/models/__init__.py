"""Inference-only stub for `trellis.models`."""


def from_pretrained(*_args, **_kwargs):
    raise RuntimeError(
        "trellis.models.from_pretrained is unavailable in this venv "
        "(inference-only stub). The 3d-recon-pipeline never trains the "
        "TRELLIS sparse-structure pipeline."
    )
