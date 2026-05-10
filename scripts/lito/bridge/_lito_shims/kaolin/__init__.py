"""Inference-only stub for `kaolin`.

NVIDIA Kaolin is a heavyweight Linux+CUDA package. The only symbol the
lito → TRELLIS → flexicubes import chain pulls is
`kaolin.utils.testing.check_tensor`, which is a debug-only shape/dtype
asserter. We stub it as a no-op so flexicubes imports cleanly without
the multi-GB Kaolin install.

ml-lito's own `lito.integrations.trellis.__init__` already does the
equivalent stub when running on macOS; this shim covers the Linux
inference path.
"""

import sys as _sys
import types as _types

_utils = _types.ModuleType(f"{__name__}.utils")
_testing = _types.ModuleType(f"{__name__}.utils.testing")


def _check_tensor(*_args, **_kwargs):
    return True


_testing.check_tensor = _check_tensor
_utils.testing = _testing

_sys.modules[f"{__name__}.utils"] = _utils
_sys.modules[f"{__name__}.utils.testing"] = _testing

utils = _utils
