r"""Spatial & spatiotemporal point processes — Python-only ``nstat.extras`` module.

This subpackage has **no MATLAB counterpart** and therefore no
``parity/manifest.yml`` entry — it lives in the opt-in ``extras/``
namespace so the core ``nstat`` MATLAB-parity contract is preserved.

Pure-NumPy/SciPy core (no optional dependency required):

- :mod:`~nstat.extras.spatial.lgcp` — log-Gaussian Cox process rate maps
  by the Laplace approximation (Newton/IRLS to the posterior mode; a
  log-normal credible band that widens in data-sparse cells).
  :func:`~nstat.extras.spatial.lgcp.lgcp_fit` →
  :class:`~nstat.extras.spatial.lgcp.LGCPResult` with ``.rate_map(level=...)``.
- :mod:`~nstat.extras.spatial.spatial_gof` — inhomogeneous second-order
  goodness-of-fit: the SOIRS-reweighted pair correlation
  :func:`~nstat.extras.spatial.spatial_gof.pair_correlation`, the
  inhomogeneous :func:`~nstat.extras.spatial.spatial_gof.k_inhom` /
  :func:`~nstat.extras.spatial.spatial_gof.l_function`, the empty-space /
  nearest-neighbour :func:`~nstat.extras.spatial.spatial_gof.nearest_neighbour_FGJ`,
  and the Monte-Carlo
  :func:`~nstat.extras.spatial.spatial_gof.global_envelope` (Myllymaki 2017).
  Three published edge-correction modes are selectable via the
  ``edge_correction`` keyword (Ripley 1976/1977; Ohser 1983;
  Baddeley-Rubak-Turner 2015).
- :mod:`~nstat.extras.spatial.basis` — tensor-product B-spline log-rate
  bases (:func:`~nstat.extras.spatial.basis.bspline_basis_1d`,
  :func:`~nstat.extras.spatial.basis.bspline_basis_2d`,
  :class:`~nstat.extras.spatial.basis.BSplineBasis2D`); the resulting
  design matrix is a valid ``x`` argument to
  :func:`nstat.glm.fit_poisson_glm`, and the P-spline second-difference
  penalty (Eilers-Marx 1996) is available via ``.gram()``.
- :mod:`~nstat.extras.spatial.marked_gof` — the discrete-time-rescaling KS
  correction (Haslinger-Pipa-Brown 2010) and marked goodness-of-fit:
  :func:`~nstat.extras.spatial.marked_gof.marked_time_rescaling`.

Optional bridges (lazy-import; fail gracefully with an install hint):

- :mod:`~nstat.extras.spatial.hawkes_bridge` — multivariate Hawkes via
  ``tick`` (``pip install nstat-toolbox[hawkes]``).
- :mod:`~nstat.extras.spatial.dpp_bridge` — DPP sampling via ``DPPy``
  (``pip install nstat-toolbox[dpp]``), with a dependency-free inline
  NumPy eigen-sampler fallback (``sample_l_ensemble``).

The heavier LGCP GP path (``gpflow``) is behind
``pip install nstat-toolbox[spatial-gp]``; the default ``lgcp_fit`` backend
is dependency-free.

Install
-------

.. code-block:: bash

    # core (lgcp, spatial_gof, marked_gof, basis) needs only numpy/scipy.
    pip install nstat-toolbox[spatial-gp]   # optional heavier GP path (gpflow)
    pip install nstat-toolbox[hawkes]       # tick (multivariate Hawkes)
    pip install nstat-toolbox[dpp]          # DPPy (DPP sampling)
"""
from __future__ import annotations

# Pure-core submodules import cleanly with only numpy/scipy — safe to
# re-export their public entry points at package import time.
from nstat.extras.spatial.basis import (
    BSplineBasis2D,
    bspline_basis_1d,
    bspline_basis_2d,
)
from nstat.extras.spatial.lgcp import (
    LGCPResult,
    MaternPrior,
    lgcp_fit,
    lgcp_fit_glm,
)
from nstat.extras.spatial.marked_gof import (
    MarkedGOFResult,
    corrected_rescaled,
    marked_time_rescaling,
    multivariate_time_rescaling,
    uncorrected_rescaled,
)
from nstat.extras.spatial.spatial_gof import (
    EnvelopeResult,
    global_envelope,
    k_inhom,
    l_function,
    nearest_neighbour_FGJ,
    pair_correlation,
)

# The optional-dep bridge submodules (hawkes_bridge, dpp_bridge) are NOT
# eagerly imported — they are reached via explicit submodule import so the
# package stays import-safe without tick / DPPy / gpflow installed:
#
#     from nstat.extras.spatial import dpp_bridge
#     idx = dpp_bridge.sample_l_ensemble(L)        # dependency-free
#     from nstat.extras.spatial import hawkes_bridge
#     fit = hawkes_bridge.fit_hawkes_exp(events)   # needs [hawkes]

__all__ = [
    # lgcp
    "lgcp_fit",
    "lgcp_fit_glm",
    "LGCPResult",
    "MaternPrior",
    # spatial_gof
    "pair_correlation",
    "k_inhom",
    "l_function",
    "nearest_neighbour_FGJ",
    "global_envelope",
    "EnvelopeResult",
    # basis
    "bspline_basis_1d",
    "bspline_basis_2d",
    "BSplineBasis2D",
    # marked_gof
    "marked_time_rescaling",
    "multivariate_time_rescaling",
    "uncorrected_rescaled",
    "corrected_rescaled",
    "MarkedGOFResult",
]
