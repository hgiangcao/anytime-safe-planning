"""Load numpy>=2.0 pickles under numpy 1.26 (and vice versa).

The recorded rollouts in results/data/*.pkl were written when this project's
environment carried numpy 2.x, which pickles array reconstructors under
`numpy._core.*`; the current interpreter has numpy 1.26, where the same modules
live under `numpy.core.*`.  Importing this module installs the aliases so the
archived data loads unchanged -- no regeneration, no silent difference between
the data the paper reports and the data on disk.
"""
from __future__ import annotations

import sys

import numpy as np

if not hasattr(np, "_core"):
    import numpy.core as _c
    sys.modules["numpy._core"] = _c
    for _sub in ("numeric", "multiarray", "umath", "_multiarray_umath",
                 "_internal", "overrides", "numerictypes", "_dtype",
                 "_methods", "fromnumeric", "shape_base", "einsumfunc"):
        try:
            sys.modules[f"numpy._core.{_sub}"] = __import__(
                f"numpy.core.{_sub}", fromlist=[_sub])
        except Exception:                                   # pragma: no cover
            pass
