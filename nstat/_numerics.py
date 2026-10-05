"""Private numeric primitives shared across nstat modules.

Not public API.  Helpers live here when more than one module must use the
*same* implementation so their numbers cannot drift apart:

- :func:`_sigmoid` — used by :mod:`nstat.cif` (re-exported there as
  ``nstat.cif._sigmoid``), :mod:`nstat.linear_cif` and
  :mod:`nstat.extras.continuous_cif`.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid σ(η) = e^η / (1 + e^η).

    Uses the two-branch form (``1/(1+e^{-η})`` for η≥0,
    ``e^η/(1+e^η)`` for η<0) to avoid overflow at large |η| without
    clipping.  Matches ``LinearCIF._link_inverse`` for the binomial
    fitType (audit finding H1 — unified sigmoid path between cif.py
    and linear_cif.py so the two CIF classes produce numerically
    identical lambda values for any η).
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        eta = float(arr)
        if eta >= 0.0:
            return np.asarray(1.0 / (1.0 + np.exp(-eta)), dtype=float)
        ez = np.exp(eta)
        return np.asarray(ez / (1.0 + ez), dtype=float)
    out = np.empty_like(arr)
    pos = arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    neg = ~pos
    ez = np.exp(arr[neg])
    out[neg] = ez / (1.0 + ez)
    return out
