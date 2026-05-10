"""Inference-only stub for `blender_rendering`.

Apple's ml-lito plibs.data_utils / plibs.vis_utils import
`from blender_rendering import blender_open3d_utils, blender_plib_utils,
utils as blender_rendering_utils` at module load time. The real package
ships Apple-internal Blender automation that is never exercised during
3d-recon-pipeline inference, so we register stub submodules whose
attribute access raises a clear error if anything actually touches them.
"""

import sys as _sys
import types as _types


# Empty submodules: ml-lito imports these at module-load time but only
# invokes them inside training-only code paths. Keeping the stubs bare
# (no module-level `__getattr__`) lets `inspect`/`pickle`/lightning probe
# the usual dunders without surprises; if a future code path actually
# touches one of these symbols, `AttributeError` will surface naturally.
for _sub in ("blender_open3d_utils", "blender_plib_utils", "utils"):
    _full = f"{__name__}.{_sub}"
    if _full not in _sys.modules:
        _sys.modules[_full] = _types.ModuleType(_full)
