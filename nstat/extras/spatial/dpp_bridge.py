r"""Determinantal point process (DPP) sampling — ``DPPy`` bridge + NumPy fallback.

The repulsive foil to the LGCP's clustering (Chapter 5, Rem. 5.1.3 /
Prop. 5.C.3): a discrete DPP on a finite ground set, where
:math:`\Pr(Y = A) = \det(L_A)/\det(L + I)` for a PSD :math:`L`-ensemble,
and the inclusion kernel is :math:`K = L(L+I)^{-1}`.  Used in the
curriculum's optional cell B5.2 to draw a repulsive pattern whose pair
correlation sits *below* the Poisson null (:math:`g(r) < 1`), mirroring
the LGCP draw above it.

Two paths:

- :func:`sample_dpp` — preferred ``DPPy`` backend when installed
  (``pip install nstat-toolbox[dpp]``), exposing its broader sampler
  catalogue.
- :func:`sample_l_ensemble` — a small **inline NumPy eigen-sampler**
  (Hough-Krishnapur-Peres-Virag 2006) that is always available — no
  optional dependency.  This is the exact spectral sampler the curriculum
  worked example uses, kept dependency-free so the foil runs anywhere.

References
----------
- Kulesza A, Taskar B (2012). *Determinantal Point Processes for Machine
  Learning.* Foundations and Trends in ML 5(2-3):123.
- Hough JB, Krishnapur M, Peres Y, Virag B (2006). *Determinantal
  processes and independence.* Probability Surveys 3:206.
"""
from __future__ import annotations

import numpy as np

from nstat.extras._lazy import require_optional


def sample_l_ensemble(
    L: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Exact DPP sample from an :math:`L`-ensemble (HKPV 2006 eigen-sampler).

    Pure NumPy — no optional dependency.  Implements the two-stage spectral
    algorithm: (1) include each eigenvector :math:`v_k` of :math:`L` with
    probability :math:`\eta_k/(1+\eta_k)`; (2) sample the elementary DPP on
    the chosen eigenvectors by iterative projection.

    Parameters
    ----------
    L
        ``(M, M)`` symmetric PSD :math:`L`-ensemble matrix over the ground
        set of ``M`` candidate sites.
    rng
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Sorted integer indices of the selected subset.
    """
    rng = np.random.default_rng() if rng is None else rng
    L = np.asarray(L, dtype=float)
    M = L.shape[0]
    evals, evecs = np.linalg.eigh(L)
    evals = np.clip(evals, 0.0, None)

    keep = rng.uniform(size=M) < evals / (1.0 + evals)
    V = evecs[:, keep]
    chosen: list[int] = []
    # Iterative elementary-DPP projection sampler.
    n_vec = V.shape[1]
    for _ in range(n_vec):
        if V.shape[1] == 0:
            break
        p = (V**2).sum(axis=1)
        total = p.sum()
        if total <= 0:
            break
        p = p / total
        i = int(rng.choice(M, p=p))
        chosen.append(i)
        # Project V onto the subspace orthogonal to e_i.
        vj = V[i] / (np.linalg.norm(V[i]) + 1e-12)
        V = V - np.outer(V @ vj, vj)
        q, _ = np.linalg.qr(V)
        V = q[:, : max(V.shape[1] - 1, 0)]
    return np.array(sorted(chosen), dtype=int)


def sample_dpp(
    L: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Sample a discrete DPP, preferring ``DPPy`` when available.

    Parameters
    ----------
    L
        ``(M, M)`` symmetric PSD :math:`L`-ensemble matrix.
    rng
        NumPy random generator (used by the NumPy fallback; ``DPPy`` uses
        its own seeding).
    backend
        ``"auto"`` (default) tries ``DPPy`` then falls back to the inline
        NumPy sampler; ``"dppy"`` forces the optional backend (raising the
        install hint if absent); ``"numpy"`` forces the inline sampler.

    Returns
    -------
    np.ndarray
        Sorted integer indices of the selected subset.
    """
    L = np.asarray(L, dtype=float)
    if backend == "numpy":
        return sample_l_ensemble(L, rng=rng)
    if backend == "dppy":
        return _sample_dppy(L)
    if backend != "auto":
        raise ValueError(f"backend must be 'auto'/'dppy'/'numpy'; got {backend!r}")
    # auto: try DPPy, fall back gracefully to the dependency-free sampler.
    try:
        return _sample_dppy(L)
    except ImportError:
        return sample_l_ensemble(L, rng=rng)


def _sample_dppy(L: np.ndarray) -> np.ndarray:
    """DPPy-backed exact sampler (lazy import)."""
    dppy = require_optional("dppy.finite_dpps", install_key="dpp")
    FiniteDPP = dppy.FiniteDPP
    dpp = FiniteDPP("likelihood", **{"L": np.asarray(L, dtype=float)})
    dpp.sample_exact()
    return np.array(sorted(int(i) for i in dpp.list_of_samples[-1]), dtype=int)


__all__ = ["sample_l_ensemble", "sample_dpp"]
