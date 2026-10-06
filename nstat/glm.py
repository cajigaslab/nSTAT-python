"""Pythonic Poisson and binomial GLM fits via IRLS.

This module provides the dependency-free IRLS (iteratively re-weighted
least squares) Newton solvers used by :class:`nstat.analysis.Analysis`
to fit GLMs.  They are exposed as standalone helpers so callers can fit
GLMs against arbitrary design matrices without instantiating a full
:class:`~nstat.trial.Trial`.

Exported symbols
----------------
- :class:`PoissonGLMResult` — frozen dataclass with ``intercept``,
  ``coefficients``, ``n_iter``, ``converged``, ``log_likelihood``, plus
  a :meth:`PoissonGLMResult.predict_rate` helper (canonical
  log-link inverse).
- :func:`fit_poisson_glm` — Newton-IRLS Poisson GLM solver.
- :class:`BinomialGLMResult` — analogous container for binomial fits
  with ``predict_probability`` and ``predict_rate`` helpers (logistic
  link).
- :func:`fit_binomial_glm` — Newton-IRLS binomial GLM solver.

No MATLAB counterpart — the MATLAB toolbox uses Stats Toolbox
``glmfit``.  All rates are in **Hz**; the binomial response must lie in
``[0, 1]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# The bound MATLAB's own ``glmfit`` log-link iterations use internally,
# via ``stattestlink.m`` (R2026a toolbox/stats/stats/private/stattestlink.m):
# ``tiny = realmin(class)^.25; bound = -log(tiny)``, i.e.
# ``ilink = @(eta) exp(constrain(eta, -bound, bound))``.  Computed from
# ``np.finfo(float)`` so it tracks MATLAB's double ``realmin`` exactly.
#
# This is passed as ``eta_bound`` by ``Analysis.GLMFit``'s poisson ('GLM')
# path ONLY, for ``fit_poisson_glm``'s *fitting iterations* (so the Newton
# walk reaches the basin MATLAB's glmfit reaches); it is a true MATLAB
# mirror there, read from stattestlink.m directly, not assumed. It is NOT
# used as the default here, and there is no binomial counterpart: MATLAB's
# BNLRCG algorithm ('Algorithm','BNLRCG') calls `Analysis.m`'s own nested
# `bnlrCG` (Demba Ba's truncated conjugate-gradient logistic fit), which
# computes `u = exp(n)./(1+exp(n))` with NO constrain at all -- stattestlink's
# 'logit' bound belongs to `glmfit`, which BNLRCG does not call. See
# parity/matlab_defects.yml ("glmfit-ilink-eta-bound-too-tight") for the
# full before/after and the six affected call sites; every caller of
# fit_poisson_glm / fit_binomial_glm with no MATLAB counterpart (the paper
# examples, extras, tutorials, docs figures) keeps the original Python-only
# +-20 default, unchanged, since widening it was shown to destabilize at
# least one such caller (a near-collinear tensor-product B-spline design;
# see tests/extras/test_spatial_basis.py) with no MATLAB basis for adding a
# line search to compensate.
_MATLAB_GLMFIT_POISSON_ETA_BOUND = -np.log(np.finfo(float).tiny ** 0.25)

# Python-only default: a stability guard with no MATLAB counterpart (neither
# glmfit's 'log'/'logit' links for the BNLRCG/bnlrCG path, which has no
# constrain at all, nor any of the non-GLMFit callers of these two
# functions). Kept at its original value so every caller that does not pass
# ``eta_bound`` explicitly is unaffected by the stattestlink.m correction
# above.
_DEFAULT_ETA_BOUND = 20.0


@dataclass(frozen=True)
class PoissonGLMResult:
    intercept: float
    coefficients: np.ndarray
    n_iter: int
    converged: bool
    log_likelihood: float

    def predict_rate(
        self, x: Sequence[Sequence[float]] | Sequence[float] | np.ndarray, offset: Sequence[float] | np.ndarray | None = None
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        eta = self.intercept + x_arr @ self.coefficients
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=float).reshape(-1)
        return np.exp(np.clip(eta, -_DEFAULT_ETA_BOUND, _DEFAULT_ETA_BOUND))


@dataclass(frozen=True)
class BinomialGLMResult:
    intercept: float
    coefficients: np.ndarray
    n_iter: int
    converged: bool
    log_likelihood: float

    def predict_probability(
        self, x: Sequence[Sequence[float]] | Sequence[float] | np.ndarray
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        eta = self.intercept + x_arr @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -_DEFAULT_ETA_BOUND, _DEFAULT_ETA_BOUND)))

    def predict_rate(
        self,
        x: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
        *,
        sample_rate: float,
    ) -> np.ndarray:
        return self.predict_probability(x) * float(sample_rate)


def fit_poisson_glm(
    x: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    offset: Sequence[float] | np.ndarray | None = None,
    include_intercept: bool = True,
    l2: float = 1e-6,
    max_iter: int = 120,
    tol: float = 1e-8,
    eta_bound: float = _DEFAULT_ETA_BOUND,
) -> PoissonGLMResult:
    """Fit a Poisson GLM (log link) by Newton-Raphson with an L2 ridge penalty.

    Maximises the L2-penalised Poisson log-likelihood
    ``sum(y * eta - exp(eta))`` with ``eta = X @ beta + offset``.  The linear
    predictor is clipped to ``[-eta_bound, eta_bound]`` before exponentiation
    for numerical stability.

    Parameters
    ----------
    x : array_like, shape (n_samples,) or (n_samples, n_features)
        Design matrix, or a single covariate vector (treated as one column).
    y : array_like, shape (n_samples,)
        Observed non-negative counts (for example spikes per bin).
    offset : array_like, shape (n_samples,), optional
        Per-sample additive offset to the linear predictor, for example
        ``log(bin_width)`` to model a rate.  Defaults to zero.
    include_intercept : bool, default True
        Prepend a constant column to ``x``.  The intercept is not penalised.
    l2 : float, default 1e-6
        Ridge penalty applied to all non-intercept coefficients.
    max_iter : int, default 120
        Maximum number of Newton iterations.
    tol : float, default 1e-8
        Convergence tolerance on the L2 norm of the coefficient update.
    eta_bound : float, default 20.0
        Clip bound for the linear predictor during the Newton iterations
        (a Python-only stability guard with no MATLAB counterpart at this
        default). ``Analysis.GLMFit``'s poisson (``'GLM'``) path passes
        MATLAB's own ``glmfit`` log-link bound here instead
        (``-log(realmin**0.25)``, from ``stattestlink.m``); every other
        caller keeps the default. This matches the BOUND glmfit's own
        iterations use, not its full numerical path: MATLAB's ``glmfit``
        initializes from ``startingVals(y)`` (a function of the data), while
        this Newton-IRLS starts from ``beta = 0`` unconditionally, so on a
        design whose MLE is far from 0 the two can still walk different
        paths (and, on a sufficiently extreme design, Python can diverge
        where MATLAB's better-initialized walk converges) even with the
        bound matched.

    Returns
    -------
    PoissonGLMResult
        Frozen dataclass with ``intercept``, ``coefficients``, ``n_iter``,
        ``converged`` and ``log_likelihood`` (the unpenalised Poisson
        log-likelihood, up to the ``log(y!)`` constant) fields.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different numbers of rows, or ``offset`` has
        a different length than ``y``.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have same row count")

    if offset is None:
        offset_arr = np.zeros_like(y_arr)
    else:
        offset_arr = np.asarray(offset, dtype=float).reshape(-1)
        if offset_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("offset size mismatch")

    n_samples, n_features = x_arr.shape
    if include_intercept:
        x_aug = np.column_stack([np.ones(n_samples), x_arr])
    else:
        x_aug = x_arr
    beta = np.zeros(x_aug.shape[1], dtype=float)

    eye = np.eye(x_aug.shape[1], dtype=float)
    if include_intercept and eye.size:
        eye[0, 0] = 0.0

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = x_aug @ beta + offset_arr
        lam = np.exp(np.clip(eta, -eta_bound, eta_bound))

        grad = x_aug.T @ (y_arr - lam) - l2 * (eye @ beta)
        hess_pos = x_aug.T @ (lam[:, None] * x_aug) + l2 * eye
        try:
            step = np.linalg.solve(hess_pos, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess_pos, grad, rcond=None)[0]

        beta_next = beta + step
        if np.linalg.norm(beta_next - beta, ord=2) < tol:
            beta = beta_next
            converged = True
            break
        beta = beta_next

    eta = x_aug @ beta + offset_arr
    lam = np.exp(np.clip(eta, -eta_bound, eta_bound))
    log_likelihood = float(np.sum(y_arr * np.log(np.maximum(lam, 1e-12)) - lam))

    return PoissonGLMResult(
        intercept=float(beta[0]) if include_intercept else 0.0,
        coefficients=beta[1:].copy() if include_intercept else beta.copy(),
        n_iter=n_iter,
        converged=converged,
        log_likelihood=log_likelihood,
    )


def fit_binomial_glm(
    x: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    include_intercept: bool = True,
    l2: float = 1e-6,
    max_iter: int = 120,
    tol: float = 1e-8,
    eta_bound: float = _DEFAULT_ETA_BOUND,
) -> BinomialGLMResult:
    """Fit a binomial GLM (logit link) by Newton-Raphson with an L2 ridge penalty.

    ``eta_bound`` : float, default 20.0
        Clip bound for the linear predictor during the Newton iterations --
        a Python-only stability guard.  Unlike :func:`fit_poisson_glm`,
        there is no MATLAB bound to adopt here: MATLAB's binomial path
        (``Algorithm == 'BNLRCG'``) calls ``Analysis.m``'s own nested
        ``bnlrCG`` (not ``glmfit``), which computes
        ``u = exp(n)./(1+exp(n))`` with no constrain at all. Every caller
        (including ``Analysis.GLMFit``'s ``'BNLRCG'`` path) keeps the
        default.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have same row count")
    if np.any((y_arr < 0.0) | (y_arr > 1.0)):
        raise ValueError("binomial GLM requires response values in [0, 1]")

    n_samples, n_features = x_arr.shape
    if include_intercept:
        x_aug = np.column_stack([np.ones(n_samples), x_arr])
    else:
        x_aug = x_arr
    beta = np.zeros(x_aug.shape[1], dtype=float)
    eye = np.eye(x_aug.shape[1], dtype=float)
    if include_intercept and eye.size:
        eye[0, 0] = 0.0

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = np.clip(x_aug @ beta, -eta_bound, eta_bound)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        grad = x_aug.T @ (y_arr - p) - l2 * (eye @ beta)
        hess_pos = x_aug.T @ (w[:, None] * x_aug) + l2 * eye
        try:
            step = np.linalg.solve(hess_pos, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess_pos, grad, rcond=None)[0]

        beta_next = beta + step
        if np.linalg.norm(beta_next - beta, ord=2) < tol:
            beta = beta_next
            converged = True
            break
        beta = beta_next

    eta = np.clip(x_aug @ beta, -eta_bound, eta_bound)
    p = 1.0 / (1.0 + np.exp(-eta))
    log_likelihood = float(np.sum(y_arr * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y_arr) * np.log(np.clip(1.0 - p, 1e-12, 1.0))))

    return BinomialGLMResult(
        intercept=float(beta[0]) if include_intercept else 0.0,
        coefficients=beta[1:].copy() if include_intercept else beta.copy(),
        n_iter=n_iter,
        converged=converged,
        log_likelihood=log_likelihood,
    )
