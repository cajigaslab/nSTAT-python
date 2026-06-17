r"""Spatial & spatiotemporal point processes — Python-only ``nstat.extras`` module.

A Python-only companion covering spatial and spatiotemporal
point-process methods (Møller & Waagepetersen 2003; Diggle 2013;
Baddeley, Rubak & Turner 2015; Haslinger-Pipa-Brown 2010; Tao et
al. 2018).  This subpackage has **no MATLAB counterpart** and
therefore no ``parity/manifest.yml`` entry — it lives in the opt-in
``extras/`` namespace precisely so the core ``nstat`` MATLAB-parity
contract is preserved.

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
- :mod:`~nstat.extras.spatial.cluster_cox` — Thomas (1949), Matérn-cluster
  (Matérn 1986), and generic Neyman-Scott (1958) Cox-process simulators
  and closed-form pair correlations.  Pair with
  :mod:`~nstat.extras.spatial.inference` for minimum-contrast estimation.
- :mod:`~nstat.extras.spatial.inference` — minimum-contrast estimator
  (Diggle 2013 §6.2.1; Møller-Waagepetersen 2003 §4.2):
  :func:`~nstat.extras.spatial.inference.min_contrast_estimator` plus the
  convenience wrappers :func:`~nstat.extras.spatial.inference.fit_thomas` /
  :func:`~nstat.extras.spatial.inference.fit_matern_cluster`.
- :mod:`~nstat.extras.spatial.marked_gof` — the discrete-time-rescaling KS
  correction (Haslinger-Pipa-Brown 2010) and marked goodness-of-fit:
  :func:`~nstat.extras.spatial.marked_gof.marked_time_rescaling`.
- :mod:`~nstat.extras.spatial.wave_analysis` — pure-NumPy spatiotemporal
  wave diagnostics on top of a fitted Hawkes triggering matrix:
  :func:`~nstat.extras.spatial.wave_analysis.reconstruct_kernel`,
  :func:`~nstat.extras.spatial.wave_analysis.detect_wave_peaks`
  (returning :class:`~nstat.extras.spatial.wave_analysis.WaveAnalysisResult`).
  The companion frequency × wave-vector
  :func:`~nstat.extras.spatial.hawkes_bridge.bartlett_spectrum` lives in
  ``hawkes_bridge.py`` but is pure NumPy/SciPy (it does NOT require ``tick``).

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

The convenience symbols below re-export the pure-core entry points so
worked examples can ``from nstat.extras.spatial import lgcp_fit,
pair_correlation, global_envelope, marked_time_rescaling``.
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
from nstat.extras.spatial.bartlett import bartlett_density_from_pcf
from nstat.extras.spatial.cluster_cox import (
    MaternClusterProcess,
    NeymanScottCox,
    ThomasProcess,
    matern_cluster_pair_correlation,
    simulate_matern_cluster,
    simulate_neyman_scott,
    simulate_thomas,
    thomas_pair_correlation,
)
from nstat.extras.spatial.gibbs import (
    AreaInteractionProcess,
    GibbsFitResult,
    GibbsStrauss,
    HardcoreProcess,
    pseudo_likelihood_fit,
    simulate_hardcore_rejection,
    simulate_strauss_birth_death,
)
from nstat.extras.spatial.inference import (
    MinContrastResult,
    fit_matern_cluster,
    fit_thomas,
    min_contrast_estimator,
)
from nstat.extras.spatial.mark_gof import mark_correlation, mark_variogram
from nstat.extras.spatial.marked_gof import (
    CoupledMarkedGOFResult,
    MarkedGOFResult,
    RescaledACFResult,
    corrected_rescaled,
    marked_time_rescaling,
    multivariate_gof_with_coupling,
    multivariate_time_rescaling,
    rescaled_acf,
    uncorrected_rescaled,
)
from nstat.extras.spatial.residuals import pp_residuals_smoothed
from nstat.extras.spatial.spatial_gof import (
    EnvelopeResult,
    cross_k_inhom,
    cross_pair_correlation,
    global_envelope,
    k_inhom,
    l_function,
    nearest_neighbour_FGJ,
    pair_correlation,
)

# ``bartlett_spectrum`` is pure NumPy/SciPy — safe to eagerly re-export
# even though it lives in ``hawkes_bridge.py`` alongside ``fit_hawkes_exp``
# (which is NOT touched here and still requires the optional ``tick`` dep
# only when called).  Importing ``bartlett_spectrum`` does NOT import ``tick``.
from nstat.extras.spatial.hawkes_bridge import bartlett_spectrum
from nstat.extras.spatial.wave_analysis import (
    WaveAnalysisResult,
    detect_wave_peaks,
    reconstruct_kernel,
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
    "cross_k_inhom",
    "cross_pair_correlation",
    "EnvelopeResult",
    # basis
    "bspline_basis_1d",
    "bspline_basis_2d",
    "BSplineBasis2D",
    # marked_gof
    "marked_time_rescaling",
    "multivariate_time_rescaling",
    "multivariate_gof_with_coupling",
    "uncorrected_rescaled",
    "corrected_rescaled",
    "rescaled_acf",
    "MarkedGOFResult",
    "CoupledMarkedGOFResult",
    "RescaledACFResult",
    # mark_gof
    "mark_correlation",
    "mark_variogram",
    # residuals
    "pp_residuals_smoothed",
    # bartlett (spatial Bartlett density of the pair correlation)
    "bartlett_density_from_pcf",
    # cluster_cox (Thomas / Matern-cluster / Neyman-Scott Cox processes)
    "ThomasProcess",
    "MaternClusterProcess",
    "NeymanScottCox",
    "thomas_pair_correlation",
    "matern_cluster_pair_correlation",
    "simulate_thomas",
    "simulate_matern_cluster",
    "simulate_neyman_scott",
    # inference (minimum-contrast estimation; Diggle 2013 §6.2.1)
    "MinContrastResult",
    "min_contrast_estimator",
    "fit_thomas",
    "fit_matern_cluster",
    # gibbs (Gibbs interaction models + Berman-Turner pseudo-likelihood)
    "GibbsStrauss",
    "HardcoreProcess",
    "AreaInteractionProcess",
    "simulate_strauss_birth_death",
    "simulate_hardcore_rejection",
    "pseudo_likelihood_fit",
    "GibbsFitResult",
    # hawkes_bridge (pure-NumPy/SciPy Bartlett diagnostic; no [hawkes] needed)
    "bartlett_spectrum",
    # wave_analysis (pure NumPy/SciPy)
    "reconstruct_kernel",
    "detect_wave_peaks",
    "WaveAnalysisResult",
]
