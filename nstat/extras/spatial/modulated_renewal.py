r"""Rate-modulated inhomogeneous renewal / conditional-ISI spiking model.

Pure-NumPy/SciPy, ``nstat.extras``-only (no MATLAB counterpart).  This is
the license-clean alternative to JAX-based non-Poisson neural renewal (NPNR)
methods: everything here is implemented directly from the published
mathematics below, reusing :func:`nstat.glm.fit_poisson_glm` and
:mod:`nstat.extras.spatial.basis` — no source code from any external
repository was read, ported, or paraphrased.

Model
-----
The conditional intensity function (CIF) is a product of a covariate-driven
*rate* term and a *renewal* term in elapsed time since the last spike:

.. math::

    \lambda(t \mid H_t) = \lambda_0(t) \cdot r(s(t); \theta), \qquad
    s(t) = \Lambda_0(t) - \Lambda_0(t^*(t)), \qquad
    \Lambda_0(t) = \int_0^t \lambda_0(u)\,du,

where :math:`\lambda_0(t) = \exp(x(t)^\top \beta)` is an ordinary Poisson-GLM
log-linear rate, :math:`t^*(t)` is the last spike time strictly before
:math:`t`, and :math:`r(\cdot; \theta)` is the hazard function of a renewal
ISI density with mean 1 and shape parameter :math:`\theta` (gamma or inverse
Gaussian).  :math:`s(t)` is elapsed time since the last spike measured in
:math:`\lambda_0`-*rescaled* ("operational") time — the classical
rate-modulated / doubly-stochastic renewal-process construction of Cox
(1955), used explicitly by Barbieri, Quirk, Frank, Wilson & Brown (2001) for
inhomogeneous inverse-Gaussian/gamma ISI models, and the same multiplicative
rate-times-history-term CIF form of Kass & Ventura's (2001) inhomogeneous
Markov interval (IMI) model.  :math:`\theta` controls the coefficient of
variation (CV) of the renewal ISI density — i.e. spike-timing *regularity*,
not just mean rate — and the exponential/Poisson limit (:math:`r \equiv 1`
identically, for *any* elapsed time) is exact at gamma shape :math:`k = 1`
(see :func:`fit_modulated_renewal`'s Poisson-collapse behaviour).

Working in operational time makes the model exactly simulatable and gives a
clean two-block penalized-MLE fit:

- **Simulation** (:func:`simulate_modulated_renewal`) draws i.i.d.
  operational-time increments :math:`u_k \sim f(\cdot;\theta)` (mean 1),
  partial-sums them, and inverts through :math:`\Lambda_0^{-1}` (linear
  interpolation on a dense grid) to get real spike times — the
  time-rescaling inverse method (Ogata 1988; Brown, Barbieri, Ventura, Kass
  & Frank 2002), generalized from the Poisson (Exp(1)) case to a general
  renewal density.
- **Fitting** (:func:`fit_modulated_renewal`) alternates: given
  :math:`\theta`, :math:`\beta` is fit by an ordinary Poisson-GLM
  (:func:`nstat.glm.fit_poisson_glm`) on a time-discretized design with the
  renewal hazard folded into the offset — the discretized point-process
  likelihood of Truccolo, Eden, Fellows, Donoghue & Brown (2005); given
  :math:`\beta` (hence :math:`\lambda_0` and :math:`\Lambda_0`),
  :math:`\theta` maximizes the renewal-density log-likelihood of the
  operational-time ISIs :math:`u_k = \Lambda_0(t_{k+1}) - \Lambda_0(t_k)` by
  1-D optimization.  Because the offset itself depends on :math:`\Lambda_0`
  (hence on the *current* :math:`\lambda_0` estimate), the beta-step is
  itself a short inner Picard fixed-point loop (recompute the operational
  elapsed time from the latest :math:`\lambda_0`, refit, repeat) — without
  it the outer alternation does not use a self-consistent operational time
  and can cycle rather than converge.
- **Goodness-of-fit tie**: under the true model the operational-time ISIs
  :math:`u_k` are i.i.d. :math:`f(\cdot;\theta)`, so the probability
  integral transform :math:`F(u_k;\theta)` (:func:`renewal_cdf`) is
  Uniform(0,1) — exactly the object a KS test (or
  :mod:`nstat.extras.spatial.marked_gof`) checks.

Literature
----------
- Barbieri R, Quirk MC, Frank LM, Wilson MA, Brown EN (2001). *Construction
  and analysis of non-Poisson stimulus-response models of neural spiking
  activity.* J Neurosci Methods 105:25-37.  (Inhomogeneous inverse-Gaussian
  / gamma renewal ISI models.)
- Kass RE, Ventura V (2001). *A spike-train probability model.* Neural
  Computation 13:1713-1720.  (The inhomogeneous Markov interval (IMI) model
  — a rate term times a function of time-since-last-spike, the same
  multiplicative CIF form used here.)
- Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005). *A point
  process framework for relating neural spiking activity to spiking
  history, neural ensemble, and extrinsic covariate effects.* J
  Neurophysiol 93:1074-1089.  (CIF/GLM foundation; the time-discretized
  Poisson-GLM likelihood used for the :math:`\beta`-step.)
- Cox DR (1955). *Some statistical methods connected with series of
  events.* J R Stat Soc B 17(2):129-164.  (Rate-modulated / doubly
  stochastic renewal processes via a time change.)
- Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002). *The
  time-rescaling theorem and its application to neural spike train data
  analysis.* Neural Computation 14(2):325-346.
- Chhikara RS, Folks JL (1974). *Estimation of the inverse Gaussian
  distribution function.* J. Amer. Statist. Assoc. 69(345):250-254 — the
  closed-form inverse-Gaussian CDF used here (see also their 1989 book
  *The Inverse Gaussian Distribution: Theory, Methodology, and
  Applications*, Marcel Dekker).  (Closed-form inverse
  Gaussian CDF used for :func:`renewal_cdf`.)

Reused patterns
----------------
- :func:`nstat.glm.fit_poisson_glm` — the IRLS Poisson-GLM solver used
  for the :math:`\beta`-step, called on the modulated design with the
  renewal-hazard folded into ``offset``.
- :mod:`nstat.extras.spatial.basis` — a caller may pass a pre-built
  B-spline design (e.g. :func:`~nstat.extras.spatial.basis.bspline_basis_1d`)
  as the ``basis`` keyword; it is used verbatim as the ``x`` argument to
  ``fit_poisson_glm`` (see that module's docstring: "the resulting design
  matrix is a valid ``x`` argument to ``fit_poisson_glm``").

Implementation note (performance)
----------------------------------
The renewal pdf/cdf/sf are implemented directly from their closed forms via
``scipy.special`` (regularized incomplete gamma for the gamma family;
the Chhikara-Folks (1974) closed-form CDF, via the standard normal CDF, for
the inverse Gaussian family) rather than through ``scipy.stats.gamma`` /
``scipy.stats.invgauss`` objects.  This matches ``scipy.stats`` to float
precision but is materially faster in the per-bin hot loop: ``scipy.stats``'s
generic ``logsf`` computes a median via a numerical inverse-CDF root-find
per element for its accuracy branch selection, which is the dominant cost
at the bin counts this module discretizes on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import special
from scipy.optimize import minimize_scalar

from nstat.glm import PoissonGLMResult, fit_poisson_glm

_VALID_RENEWALS = ("gamma", "inverse_gaussian")
_EPS = np.finfo(float).eps
_LOG_THETA_BOUNDS = (np.log(1e-3), np.log(200.0))  # CV in [~0.07, ~31.6]


def _validate_renewal(renewal: str) -> None:
    if renewal not in _VALID_RENEWALS:
        raise ValueError(
            f"renewal must be one of {_VALID_RENEWALS!r}; got {renewal!r}"
        )


# ----------------------------------------------------------------------
# Renewal density primitives (mean fixed at 1; shape parameter theta)
# ----------------------------------------------------------------------
#
# Both families are parametrized with mean 1 so theta alone controls the
# coefficient of variation: CV = 1/sqrt(theta) for BOTH families under this
# parametrization (gamma: Var = 1/k; inverse Gaussian: Var = 1/lambda).
#
# gamma(shape=k, scale=1/k):    mean = 1, CV = 1/sqrt(k).  k=1 is exactly
#                                Exponential(1), whose hazard is identically
#                                1 for all tau -- the exact Poisson limit.
# invgauss(mean=1, shape=lam):  mean = 1, CV = 1/sqrt(lam).  Standard
#                                Chhikara-Folks parametrization; verified
#                                against scipy.stats.invgauss(mu=1/lam,
#                                scale=lam) to float precision.


def _renewal_logpdf(u: np.ndarray, theta: float, renewal: str) -> np.ndarray:
    x = np.maximum(np.asarray(u, dtype=float), _EPS)
    if renewal == "gamma":
        return (
            theta * np.log(theta)
            - special.gammaln(theta)
            + (theta - 1.0) * np.log(x)
            - theta * x
        )
    if renewal == "inverse_gaussian":
        return 0.5 * np.log(theta) - 0.5 * np.log(2.0 * np.pi * x**3) - theta * (
            x - 1.0
        ) ** 2 / (2.0 * x)
    raise ValueError(f"renewal must be one of {_VALID_RENEWALS!r}; got {renewal!r}")


def _renewal_sf(u: np.ndarray, theta: float, renewal: str) -> np.ndarray:
    """Renewal survival function :math:`S(u;\\theta) = 1 - F(u;\\theta)`."""
    x = np.maximum(np.asarray(u, dtype=float), _EPS)
    if renewal == "gamma":
        return special.gammaincc(theta, theta * x)
    if renewal == "inverse_gaussian":
        # Chhikara & Folks (1974) closed-form CDF for IG(mean=1, shape=theta):
        #   F(x) = Phi(z1) + exp(2 theta) Phi(-z2)
        #   z1 = sqrt(theta/x) (x - 1),  z2 = sqrt(theta/x) (x + 1)
        # so S(x) = 1 - F(x) = Phi(-z1) - exp(2 theta) Phi(-z2).
        sqrt_term = np.sqrt(theta / x)
        z1 = sqrt_term * (x - 1.0)
        z2 = sqrt_term * (x + 1.0)
        return special.ndtr(-z1) - np.exp(2.0 * theta) * special.ndtr(-z2)
    raise ValueError(f"renewal must be one of {_VALID_RENEWALS!r}; got {renewal!r}")


def _renewal_cdf(u: np.ndarray, theta: float, renewal: str) -> np.ndarray:
    return 1.0 - _renewal_sf(u, theta, renewal)


def _renewal_hazard(u: np.ndarray, theta: float, renewal: str) -> np.ndarray:
    """Renewal hazard ``r(u;theta) = f(u;theta) / S(u;theta)``."""
    logpdf = _renewal_logpdf(u, theta, renewal)
    sf = np.maximum(_renewal_sf(u, theta, renewal), np.exp(-700.0))
    return np.exp(np.clip(logpdf - np.log(sf), -50.0, 50.0))


def renewal_hazard(
    tau: np.ndarray | float, shape_param: float, renewal: str = "inverse_gaussian"
) -> np.ndarray:
    """Public renewal-hazard evaluator :math:`r(\\tau;\\theta)`.

    Thin, documented wrapper around the gamma / inverse-Gaussian hazard
    used internally by :func:`fit_modulated_renewal` and
    :meth:`ModulatedRenewalResult.rate_fn`.  Exposed so a caller can
    inspect or plot the fitted renewal shape directly.

    Parameters
    ----------
    tau
        Elapsed time since the last spike, in :math:`\\lambda_0`-rescaled
        ("operational") time (see module docstring); scalar or array.
    shape_param
        Renewal shape :math:`\\theta` (mean is fixed at 1 in this
        parametrization; ``CV = 1/sqrt(theta)`` for both families).
    renewal
        ``"gamma"`` or ``"inverse_gaussian"``.

    Returns
    -------
    np.ndarray
    """
    _validate_renewal(renewal)
    return _renewal_hazard(np.asarray(tau, dtype=float), float(shape_param), renewal)


def renewal_cdf(
    u: np.ndarray | float, shape_param: float, renewal: str = "inverse_gaussian"
) -> np.ndarray:
    """Public renewal CDF :math:`F(u;\\theta)` — the probability-integral
    transform used for the time-rescaling goodness-of-fit tie.

    Under a correctly-fit model the operational-time ISIs
    (:attr:`ModulatedRenewalResult.rescaled_isis`) are i.i.d. draws from
    the renewal density with shape ``shape_param``, so
    ``renewal_cdf(rescaled_isis, shape_param, renewal)`` is ~Uniform(0,1)
    — the classical time-rescaling-theorem check (Brown et al. 2002),
    generalized from the Exp(1) Poisson case to a general renewal density.

    Parameters
    ----------
    u
        Operational-time inter-spike intervals; scalar or array.
    shape_param
        Renewal shape :math:`\\theta`.
    renewal
        ``"gamma"`` or ``"inverse_gaussian"``.

    Returns
    -------
    np.ndarray
    """
    _validate_renewal(renewal)
    return _renewal_cdf(np.asarray(u, dtype=float), float(shape_param), renewal)


def _fit_renewal_shape(u: np.ndarray, renewal: str) -> float:
    """1-D MLE of the renewal shape :math:`\\theta` given operational ISIs."""
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u) & (u > 0.0)]
    if u.size == 0:
        return 1.0

    def neg_ll(log_theta: float) -> float:
        theta = float(np.exp(log_theta))
        return -float(np.sum(_renewal_logpdf(u, theta, renewal)))

    res = minimize_scalar(neg_ll, bounds=_LOG_THETA_BOUNDS, method="bounded")
    return float(np.exp(res.x))


# ----------------------------------------------------------------------
# Time-grid helpers (uniform bins of width dt covering [0, n_bins*dt])
# ----------------------------------------------------------------------


def _piecewise_cumulative_rate(
    bin_edges: np.ndarray, lam0_bin: np.ndarray, t_query: np.ndarray
) -> np.ndarray:
    """Cumulative :math:`\\Lambda_0(t) = \\int_0^t \\lambda_0(u)\\,du` for a
    piecewise-constant-per-bin :math:`\\lambda_0`, evaluated at ``t_query``.

    ``t_query`` must lie within ``[bin_edges[0], bin_edges[-1]]``.
    """
    bin_widths = np.diff(bin_edges)
    cum_at_edges = np.concatenate([[0.0], np.cumsum(lam0_bin * bin_widths)])
    idx = np.clip(
        np.searchsorted(bin_edges, t_query, side="right") - 1, 0, len(lam0_bin) - 1
    )
    partial = (np.asarray(t_query, dtype=float) - bin_edges[idx]) * lam0_bin[idx]
    return cum_at_edges[idx] + partial


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModulatedRenewalResult:
    """Fitted modulated-renewal (conditional-ISI) point-process model.

    Attributes
    ----------
    beta
        Covariate coefficients of :math:`\\log \\lambda_0`.  ``beta[0]`` is
        the intercept; ``beta[1:]`` are the coefficients on the columns of
        the fitted design matrix (``basis`` if given at fit time, else
        ``covariates``).
    shape_param
        Fitted renewal shape :math:`\\theta` (mean-1 parametrization; see
        module docstring).
    renewal
        ``"gamma"`` or ``"inverse_gaussian"``.
    rescaled_isis
        Operational-time inter-spike intervals
        :math:`u_k = \\Lambda_0(t_{k+1}) - \\Lambda_0(t_k)`, length
        ``len(spike_times) - 1``.  Feed through :func:`renewal_cdf` for the
        time-rescaling goodness-of-fit check (see module docstring), or
        into :mod:`nstat.extras.spatial.marked_gof` via per-bin
        probabilities built from :meth:`rate_fn`.
    cv
        Coefficient of variation of the renewal ISI density,
        ``1 / sqrt(shape_param)`` (identical formula for both families
        under the mean-1 parametrization).
    log_likelihood
        Discretized (bin-Poisson) joint log-likelihood of the final
        ``(beta, shape_param)`` fit — the same time-discretized
        point-process likelihood of Truccolo et al. (2005), with the
        renewal hazard folded into the offset.
    n_iter
        Number of outer (beta, theta)-alternation iterations performed.
    converged
        Whether the alternation satisfied ``tol`` before ``max_iter``.
    spike_times
        The (sorted) spike times the model was fit on.
    dt
        Bin width (seconds) of the internal time-discretization grid.
    """

    beta: np.ndarray
    shape_param: float
    renewal: str
    rescaled_isis: np.ndarray
    cv: float
    log_likelihood: float
    n_iter: int
    converged: bool
    spike_times: np.ndarray
    dt: float
    _design: np.ndarray = field(repr=False, compare=False)

    def rate_fn(self) -> Callable[[np.ndarray | float], np.ndarray | float]:
        """Return a callable ``t -> lambda(t | H_t)`` (the fitted CIF).

        The covariate row at time ``t`` is looked up on the fixed
        ``dt``-grid the model was fit on (piecewise-constant / nearest-bin;
        times beyond the fitted grid clamp to the boundary row), which also
        fixes :math:`\\lambda_0` and hence :math:`\\Lambda_0` for the
        operational-time renewal term.  ``spike_times`` (the history the
        model was fit on) gives the last spike strictly before ``t``;
        before the first spike the renewal term is 1 (no history yet),
        matching the fit convention.

        Returns
        -------
        Callable[[np.ndarray | float], np.ndarray | float]
            Vectorized over array input; returns a scalar float for scalar
            input.
        """
        beta = self.beta
        design = self._design
        dt = self.dt
        n_bins = design.shape[0]
        bin_edges = np.arange(n_bins + 1, dtype=float) * dt
        eta_bins = beta[0] + (design @ beta[1:] if beta.size > 1 else np.zeros(n_bins))
        lam0_bin = np.exp(np.clip(eta_bins, -20.0, 20.0))
        spike_times = self.spike_times
        shape_param = self.shape_param
        renewal = self.renewal

        def f(t: np.ndarray | float) -> np.ndarray | float:
            scalar_input = np.ndim(t) == 0
            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            t_clamped = np.clip(t_arr, bin_edges[0], bin_edges[-1])
            idx = np.clip(np.floor(t_clamped / dt).astype(int), 0, n_bins - 1)
            x_t = design[idx]
            eta = beta[0] + (x_t @ beta[1:] if beta.size > 1 else 0.0)
            lam0 = np.exp(np.clip(eta, -20.0, 20.0))

            last_idx = np.searchsorted(spike_times, t_arr, side="right") - 1
            has_history = last_idx >= 0
            r = np.ones_like(lam0)
            if np.any(has_history):
                lambda0_t = _piecewise_cumulative_rate(
                    bin_edges, lam0_bin, t_clamped[has_history]
                )
                spk_clamped = np.clip(
                    spike_times[last_idx[has_history]], bin_edges[0], bin_edges[-1]
                )
                lambda0_spk = _piecewise_cumulative_rate(bin_edges, lam0_bin, spk_clamped)
                s_op = np.maximum(lambda0_t - lambda0_spk, _EPS)
                r[has_history] = _renewal_hazard(s_op, shape_param, renewal)

            out = lam0 * r
            return float(out[0]) if scalar_input else out

        return f


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------


def fit_modulated_renewal(
    spike_times: np.ndarray,
    covariates: np.ndarray | None,
    *,
    renewal: str = "inverse_gaussian",
    basis: np.ndarray | None = None,
    dt: float | None = None,
    penalty: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    n_inner: int = 12,
) -> ModulatedRenewalResult:
    r"""Penalized maximum-likelihood fit of a modulated-renewal CIF.

    Alternates two blocks to convergence (see module docstring for the
    operational-time renewal construction):

    1. **beta-step** — given :math:`\theta` and the current
       :math:`\lambda_0` estimate, discretize time into bins of width
       ``dt``, compute each bin's elapsed *operational* time since the last
       spike (:math:`\Lambda_0(\text{bin}) - \Lambda_0(\text{last
       spike})`, using the current :math:`\lambda_0`), and fit
       :math:`\beta` by :func:`nstat.glm.fit_poisson_glm` on
       ``x = basis if basis is not None else covariates`` with ``offset =
       log(r(s_op; theta)) + log(dt)`` (bins before the first spike use
       ``r = 1``, i.e. no history yet — Truccolo et al. 2005's discretized
       point-process likelihood).  Because the offset depends on
       :math:`\lambda_0` itself, this is repeated (recompute
       :math:`\Lambda_0` from the newly-fit :math:`\lambda_0`, refit) for
       up to ``n_inner`` Picard iterations per outer step, to a
       self-consistent :math:`(\beta, \lambda_0)` pair for the current
       :math:`\theta`.
    2. **theta-step** — given :math:`\beta` (hence :math:`\lambda_0` and
       :math:`\Lambda_0`), compute the operational-time ISIs
       :math:`u_k = \Lambda_0(t_{k+1}) - \Lambda_0(t_k)` and maximize the
       renewal-density log-likelihood :math:`\sum_k \log f(u_k;\theta)` by
       1-D optimization (``scipy.optimize.minimize_scalar``).

    Parameters
    ----------
    spike_times
        Sorted (or unsorted — sorted internally) spike times in seconds.
        Must contain at least 2 spikes.
    covariates
        ``(n_bins,)`` or ``(n_bins, n_features)`` array of covariate values
        sampled on a uniform time grid.  Used as the GLM design matrix
        unless ``basis`` is given (in which case ``basis`` takes
        precedence and ``covariates`` is ignored).  Must be non-empty
        unless ``basis`` is given.
    renewal
        ``"gamma"`` or ``"inverse_gaussian"`` (default).
    basis
        Optional pre-built design matrix, e.g. the output of
        :func:`nstat.extras.spatial.basis.bspline_basis_1d` /
        :func:`~nstat.extras.spatial.basis.bspline_basis_2d` evaluated on
        the same time grid as ``covariates``.  When given, it — not
        ``covariates`` — is used as the ``x`` argument to
        :func:`nstat.glm.fit_poisson_glm`.
    dt
        Bin width (seconds) of the internal discretization grid. If
        ``None`` (default), inferred as ``spike_times[-1] / n_bins`` where
        ``n_bins`` is the row count of the design matrix (i.e. the design
        is assumed to span ``[0, spike_times[-1]]``).  All spike times
        must lie within ``[0, n_bins * dt]``.
    penalty
        Ridge strength passed through to ``fit_poisson_glm``'s ``l2``
        (isotropic ridge on beta, excluding the intercept) at each
        beta-step.  ``0.0`` (default) uses ``fit_poisson_glm``'s own
        conservative default (``1e-6``); ``penalty > 0`` overrides it.
        Note: this is *not* the P-spline gram-matrix penalty of
        :meth:`~nstat.extras.spatial.basis.BSplineBasis2D.gram` —
        ``fit_poisson_glm`` only supports a scalar isotropic ridge.
    max_iter
        Maximum number of (beta, theta)-alternation iterations.
    tol
        Convergence tolerance on both the beta step (``||beta_new -
        beta||_2 < tol``, checked across a full outer iteration) and the
        theta step (``|theta_new - theta| < tol``); also the inner Picard
        loop's per-step stopping tolerance.
    n_inner
        Maximum number of inner Picard sub-iterations per beta-step (see
        above); each re-solves the GLM with the operational-time offset
        recomputed from the latest :math:`\lambda_0`.

    Returns
    -------
    ModulatedRenewalResult

    Raises
    ------
    ValueError
        If ``renewal`` is invalid, ``spike_times`` has fewer than 2
        spikes, both ``covariates`` and ``basis`` are ``None`` / empty, or
        the resulting time grid does not cover the full spike-time range.
    """
    _validate_renewal(renewal)

    spike_times = np.sort(np.asarray(spike_times, dtype=float).ravel())
    if spike_times.size < 2:
        raise ValueError(
            "fit_modulated_renewal needs at least 2 spikes to fit a "
            f"renewal model; got {spike_times.size}"
        )

    design_src = basis if basis is not None else covariates
    if design_src is None:
        raise ValueError(
            "covariates must be provided (directly, or via a pre-built "
            "design matrix passed as `basis`)"
        )
    design = np.asarray(design_src, dtype=float)
    if design.ndim == 1:
        design = design[:, None]
    if design.size == 0 or design.shape[0] == 0:
        raise ValueError(
            "covariates (or basis) must be a non-empty (n_bins, n_features) "
            f"array; got shape {design.shape}"
        )
    n_bins, n_features = design.shape

    if dt is None:
        dt_val = float(spike_times[-1]) / n_bins
        if dt_val <= 0.0:
            raise ValueError(
                "could not infer dt from spike_times[-1] / n_bins "
                f"(spike_times[-1]={spike_times[-1]!r}, n_bins={n_bins}); "
                "pass dt explicitly"
            )
    else:
        dt_val = float(dt)
        if dt_val <= 0.0:
            raise ValueError(f"dt must be positive; got {dt_val!r}")

    bin_edges = np.arange(n_bins + 1, dtype=float) * dt_val
    bin_centers = bin_edges[:-1] + 0.5 * dt_val
    tol_edge = 1e-9 * dt_val
    if spike_times[0] < bin_edges[0] - tol_edge or spike_times[-1] > bin_edges[-1] + tol_edge:
        raise ValueError(
            "spike_times must lie within the covariate time grid "
            f"[0, {bin_edges[-1]!r}] (n_bins={n_bins}, dt={dt_val!r}); "
            "widen dt/covariates or pass dt explicitly"
        )

    y_counts, _ = np.histogram(spike_times, bins=bin_edges)

    # IMPORTANT: look up "the last spike before this bin" using each bin's
    # LEFT edge, not its center.  Searching against the center would let a
    # bin that itself contains a spike (when that spike falls before the
    # bin's midpoint) treat ITS OWN spike as "the last spike" -- collapsing
    # the elapsed operational time for that (highest-leverage, y_i=1) bin
    # to ~0 instead of the full inter-spike interval, which silently
    # corrupts the offset and biases the fit (discovered empirically: it
    # made the discretized bin-Poisson log-likelihood disagree in *rank
    # order* with the exact closed-form renewal log-likelihood, i.e. an
    # unbounded/runaway objective rather than a small O(dt) discretization
    # gap).
    last_spike_idx = np.searchsorted(spike_times, bin_edges[:-1], side="right") - 1
    has_history = last_spike_idx >= 0

    l2 = penalty if penalty > 0.0 else 1e-6
    # Damping (successive under-relaxation) on the inner Picard update of
    # lambda_0.  The undamped fixed-point map (recompute the operational
    # elapsed time from the latest lambda_0, refit beta, update lambda_0)
    # is *not* a contraction here -- lambda_0 feeds into the operational
    # elapsed time (via Lambda_0) which feeds back into how strongly the
    # renewal hazard reshapes the offset, which can overshoot and produce
    # a sustained 2-cycle instead of converging (verified empirically: the
    # undamped loop oscillates indefinitely even at the *true* theta).
    # Damping to a convex combination with the previous iterate restores
    # convergence to the correct conditional optimum (verified against a
    # grid search of the exact discretized log-likelihood).
    _inner_damping = 0.5

    def _beta_step(
        theta: float, lam0_bin_init: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, PoissonGLMResult]:
        """Inner (damped) Picard loop: self-consistent (beta, lambda_0) given theta."""
        lam0_bin = lam0_bin_init
        beta_full: np.ndarray | None = None
        glm_res: PoissonGLMResult | None = None
        for _ in range(max(n_inner, 1)):
            lambda0_bins = _piecewise_cumulative_rate(bin_edges, lam0_bin, bin_centers)
            lambda0_spikes = _piecewise_cumulative_rate(bin_edges, lam0_bin, spike_times)
            s_op = np.full(n_bins, np.nan)
            s_op[has_history] = (
                lambda0_bins[has_history] - lambda0_spikes[last_spike_idx[has_history]]
            )
            r = np.ones(n_bins, dtype=float)
            r[has_history] = _renewal_hazard(
                np.maximum(s_op[has_history], _EPS), theta, renewal
            )
            offset = np.log(np.maximum(r, _EPS)) + np.log(dt_val)

            glm_res = fit_poisson_glm(
                design, y_counts, offset=offset, include_intercept=True, l2=l2
            )
            beta_full_new = np.concatenate([[glm_res.intercept], glm_res.coefficients])
            eta_bins = glm_res.intercept + design @ glm_res.coefficients
            lam0_bin_target = np.exp(np.clip(eta_bins, -20.0, 20.0))
            lam0_bin_new = (
                (1.0 - _inner_damping) * lam0_bin + _inner_damping * lam0_bin_target
            )

            inner_change = (
                float(np.linalg.norm(beta_full_new - beta_full))
                if beta_full is not None
                else np.inf
            )
            beta_full = beta_full_new
            lam0_bin = lam0_bin_new
            if inner_change < tol:
                break
        assert beta_full is not None and glm_res is not None  # n_inner >= 1
        return beta_full, lam0_bin, glm_res

    theta = 1.0
    beta_full = np.zeros(n_features + 1, dtype=float)
    lam0_bin = np.ones(n_bins, dtype=float)  # bootstrap: flat baseline rate
    converged = False
    n_iter = 0
    glm_res = None

    for it in range(max_iter):
        n_iter = it + 1
        beta_prev_outer = beta_full
        beta_full, lam0_bin, glm_res = _beta_step(theta, lam0_bin)

        lambda0_at_spikes = _piecewise_cumulative_rate(bin_edges, lam0_bin, spike_times)
        u = np.diff(lambda0_at_spikes)
        theta_new = _fit_renewal_shape(u, renewal)

        beta_change = float(np.linalg.norm(beta_full - beta_prev_outer))
        theta_change = abs(theta_new - theta)
        theta = theta_new

        if beta_change < tol and theta_change < tol:
            converged = True
            break

    # Final polish: one more self-consistent beta-step at the converged
    # theta, so the returned (beta, theta) pair, rescaled_isis, cv and
    # log_likelihood are mutually consistent (the loop above updates theta
    # strictly after the last beta-step).
    beta_full, lam0_bin, glm_res = _beta_step(theta, lam0_bin)
    lambda0_at_spikes = _piecewise_cumulative_rate(bin_edges, lam0_bin, spike_times)
    rescaled_isis = np.diff(lambda0_at_spikes)
    cv = 1.0 / np.sqrt(theta)

    return ModulatedRenewalResult(
        beta=beta_full,
        shape_param=float(theta),
        renewal=renewal,
        rescaled_isis=rescaled_isis,
        cv=float(cv),
        log_likelihood=float(glm_res.log_likelihood),
        n_iter=n_iter,
        converged=converged,
        spike_times=spike_times,
        dt=dt_val,
        _design=design,
    )


# ----------------------------------------------------------------------
# Simulate
# ----------------------------------------------------------------------


def simulate_modulated_renewal(
    rate_fn_or_beta: Callable[[np.ndarray], np.ndarray] | float,
    shape_param: float,
    *,
    T: float,
    renewal: str = "inverse_gaussian",
    rng: np.random.Generator,
    dt: float = 1e-3,
) -> np.ndarray:
    r"""Simulate an inhomogeneous modulated-renewal spike train.

    Time-rescaling inverse method, generalized from the Poisson (Exp(1))
    case to a general renewal density (Cox 1955; Brown et al. 2002; see
    module docstring): compute :math:`\Lambda_0(t) = \int_0^t
    \lambda_0(u)\,du` on a dense grid of width ``dt``, draw i.i.d.
    operational-time increments :math:`u_k \sim f(\cdot;\theta)` (mean 1),
    take partial sums :math:`S_k = \sum_{i \le k} u_i`, and invert through
    :math:`\Lambda_0^{-1}` (linear interpolation on the dense grid) to
    recover the real-time spike times :math:`t_k`.

    Parameters
    ----------
    rate_fn_or_beta
        Either a vectorized callable ``lambda_0(t_array) -> rate_array``
        (Hz), or a scalar constant baseline rate (Hz) for a homogeneous
        :math:`\lambda_0`.
    shape_param
        Renewal shape :math:`\theta` (mean-1 parametrization; ``CV =
        1/sqrt(theta)``).  Must be positive.
    T
        Total simulated duration (seconds); must be positive.
    renewal
        ``"gamma"`` or ``"inverse_gaussian"`` (default).
    rng
        ``np.random.Generator`` (e.g. ``np.random.default_rng(seed)``).
        Sampling uses ``rng.gamma`` for the gamma family and ``rng.wald``
        (numpy's inverse-Gaussian sampler; verified to match the
        mean/shape parametrization used throughout this module) for the
        inverse-Gaussian family — both direct NumPy Generator methods, no
        ``scipy.stats`` sampling overhead.
    dt
        Resolution of the dense internal integration/inversion grid
        (seconds); default ``1e-3``.

    Returns
    -------
    np.ndarray
        Sorted spike times in ``[0, T]``.  Empty if the cumulative rate is
        zero.

    Raises
    ------
    ValueError
        If ``renewal`` is invalid, ``T`` / ``shape_param`` are not
        positive, or ``rate_fn_or_beta`` returns a negative or non-finite
        rate.
    """
    _validate_renewal(renewal)
    if T <= 0.0:
        raise ValueError(f"T must be positive; got {T!r}")
    if shape_param <= 0.0:
        raise ValueError(f"shape_param must be positive; got {shape_param!r}")

    n_grid = max(int(np.ceil(T / dt)) + 1, 2)
    t_grid = np.linspace(0.0, float(T), n_grid)

    if callable(rate_fn_or_beta):
        lam0 = np.asarray(rate_fn_or_beta(t_grid), dtype=float)
        lam0 = np.broadcast_to(lam0, t_grid.shape).astype(float)
    else:
        lam0 = np.full(t_grid.shape, float(rate_fn_or_beta))

    if not np.all(np.isfinite(lam0)) or np.any(lam0 < 0.0):
        raise ValueError("rate_fn_or_beta must return finite, non-negative rates")

    seg = 0.5 * (lam0[:-1] + lam0[1:]) * np.diff(t_grid)
    lambda0 = np.concatenate([[0.0], np.cumsum(seg)])
    lambda0_max = float(lambda0[-1])
    if lambda0_max <= 0.0:
        return np.empty(0, dtype=float)

    def _draw(n: int) -> np.ndarray:
        if renewal == "gamma":
            return rng.gamma(shape=shape_param, scale=1.0 / shape_param, size=n)
        return rng.wald(mean=1.0, scale=shape_param, size=n)

    s_cum = 0.0
    spikes_s: list[float] = []
    batch = max(int(lambda0_max * 2) + 16, 16)
    while s_cum <= lambda0_max:
        draws = _draw(batch)
        cum = s_cum + np.cumsum(draws)
        spikes_s.extend(cum[cum <= lambda0_max].tolist())
        s_cum = float(cum[-1])
        batch *= 2

    if not spikes_s:
        return np.empty(0, dtype=float)

    spikes_s_arr = np.asarray(spikes_s, dtype=float)
    spike_times = np.interp(spikes_s_arr, lambda0, t_grid)
    return np.sort(spike_times)


__all__ = [
    "ModulatedRenewalResult",
    "fit_modulated_renewal",
    "simulate_modulated_renewal",
    "renewal_hazard",
    "renewal_cdf",
]
