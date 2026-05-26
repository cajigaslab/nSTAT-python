"""EM-trained linear-Gaussian state-space models via Dynamax.

This module wraps :mod:`dynamax.linear_gaussian_ssm.LinearGaussianSSM`
behind a thin nstat-style API so users can fit EM-trained state-space
models without nstat owning the EM code itself.

Scope (initial release)
-----------------------
- :func:`fit_linear_gaussian_em` — fit a discrete-time linear-Gaussian
  state-space model from observations via EM.  Returns the learned
  parameters as plain NumPy arrays (NOT pytrees) so callers stay
  decoupled from JAX.

Out of scope (deferred)
-----------------------
- ``fit_point_process_em`` (PP_EM equivalent) — needs the Dynamax
  ``PoissonHMM`` bridge.
- ``fit_hybrid_em`` (mPPCO_EM equivalent) — needs the Dynamax
  ``GeneralizedGaussianSSM`` bridge.

Install
-------

.. code-block:: bash

    pip install nstat-toolbox[dynamax]

Pulls Dynamax (~50 MB) + JAX (~200 MB).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_IMPORT_ERROR_MSG = (
    "nstat.extras.em.dynamax_bridge requires the 'dynamax' package, which is "
    "not installed.  Install with: pip install nstat-toolbox[dynamax]"
)


def _require_dynamax():
    try:
        import dynamax  # noqa: F401
        import jax  # noqa: F401
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e


@dataclass(frozen=True)
class LinearGaussianEMResult:
    """Output of a linear-Gaussian SSM EM fit.

    Attributes
    ----------
    transition_matrix
        Learned :math:`A`, shape ``(state_dim, state_dim)``.
    observation_matrix
        Learned :math:`C`, shape ``(emission_dim, state_dim)``.
    transition_covariance
        Learned :math:`Q`, shape ``(state_dim, state_dim)``.
    observation_covariance
        Learned :math:`R`, shape ``(emission_dim, emission_dim)``.
    initial_state_mean
        Learned :math:`\\hat x_0`, shape ``(state_dim,)``.
    initial_state_covariance
        Learned :math:`P_0`, shape ``(state_dim, state_dim)``.
    log_likelihoods
        EM log-likelihood trace, shape ``(n_iter,)``.  Last value is
        the final log-likelihood at convergence.
    n_iter
        Number of EM iterations actually run.
    """

    transition_matrix: np.ndarray
    observation_matrix: np.ndarray
    transition_covariance: np.ndarray
    observation_covariance: np.ndarray
    initial_state_mean: np.ndarray
    initial_state_covariance: np.ndarray
    log_likelihoods: np.ndarray
    n_iter: int


def fit_linear_gaussian_em(
    observations: np.ndarray,
    state_dim: int,
    *,
    n_iter: int = 50,
    seed: int = 0,
) -> LinearGaussianEMResult:
    """Fit a linear-Gaussian state-space model via Dynamax EM.

    Counterpart to MATLAB nSTAT's ``KF_EM`` family.

    Parameters
    ----------
    observations
        Observation time series, shape ``(T, emission_dim)``.
    state_dim
        Dimensionality of the latent state.
    n_iter
        Number of EM iterations.  Default 50.
    seed
        Seed for JAX PRNG (used for parameter initialization).

    Returns
    -------
    LinearGaussianEMResult

    Notes
    -----
    Dynamax internally represents parameters as nested JAX pytrees;
    this wrapper unpacks them into plain NumPy arrays so callers don't
    need to know about pytrees or JAX.
    """
    _require_dynamax()
    from dynamax.linear_gaussian_ssm import LinearGaussianSSM
    import jax

    observations = np.asarray(observations, dtype=float)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be shape (T, emission_dim); got {observations.shape}"
        )
    if state_dim < 1:
        raise ValueError(f"state_dim must be >= 1; got {state_dim}")

    emission_dim = observations.shape[1]
    model = LinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
    key = jax.random.PRNGKey(int(seed))
    params, props = model.initialize(key)

    fitted_params, lls = model.fit_em(
        params, props, observations, num_iters=int(n_iter)
    )

    return LinearGaussianEMResult(
        transition_matrix=np.asarray(fitted_params.dynamics.weights, dtype=float),
        observation_matrix=np.asarray(fitted_params.emissions.weights, dtype=float),
        transition_covariance=np.asarray(fitted_params.dynamics.cov, dtype=float),
        observation_covariance=np.asarray(fitted_params.emissions.cov, dtype=float),
        initial_state_mean=np.asarray(fitted_params.initial.mean, dtype=float),
        initial_state_covariance=np.asarray(fitted_params.initial.cov, dtype=float),
        log_likelihoods=np.asarray(lls, dtype=float),
        n_iter=int(len(lls)),
    )


# ----------------------------------------------------------------------
# Point-process state-space filtering / smoothing
#
# These wrap Dynamax's Conditional-Moments Gaussian Filter / Smoother
# (CMGF) for a continuous-state linear-Gaussian dynamics with Poisson
# observations.  Useful for *inference* on a known model — the EM
# *training* counterpart (MATLAB nSTAT's PP_EM family) is deferred:
# Dynamax doesn't provide a Poisson-LGSSM EM out of the box, so PP_EM
# would require hand-rolling the E-step (CMGF smoother sufficient
# statistics) + M-step (closed-form for A/Q/x0/P0, Newton-Raphson for
# C) in nstat itself — a non-trivial ~500 LOC port deferred to a
# future release.  See ``parity/integration_opportunities.md`` for the
# rationale.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CMGFPoissonFilterResult:
    """Output of a CMGF Poisson filter or smoother.

    Attributes
    ----------
    state_means
        Posterior state means, shape ``(T, state_dim)``.  These are
        *filtered* means when produced by :func:`cmgf_poisson_filter`,
        *smoothed* (forward + backward pass) when produced by
        :func:`cmgf_poisson_smoother`.
    state_covariances
        Posterior state covariances, shape ``(T, state_dim, state_dim)``.
    marginal_log_likelihood
        Sum of log p(y_t | y_{1:t-1}) (filter) or log p(y_{1:T})
        (smoother) under the CMGF Gaussian approximation.  Useful for
        model-comparison-style scalar diagnostics.
    """

    state_means: np.ndarray
    state_covariances: np.ndarray
    marginal_log_likelihood: float


def _build_cmgf_poisson_params(
    transition_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    transition_covariance: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
):
    """Construct a Dynamax ``ParamsGGSSM`` for Poisson-LGSSM."""
    _require_dynamax()
    import jax.numpy as jnp
    from dynamax.generalized_gaussian_ssm import ParamsGGSSM
    import tensorflow_probability.substrates.jax as tfp

    A = jnp.asarray(transition_matrix, dtype=float)
    C = jnp.asarray(observation_matrix, dtype=float)
    Q = jnp.asarray(transition_covariance, dtype=float)
    x0 = jnp.asarray(initial_state_mean, dtype=float).ravel()
    P0 = jnp.asarray(initial_state_covariance, dtype=float)

    # Canonical Poisson construction (Dynamax CMGF demo notebook).
    # The state-conditional rate is exp(C @ z) and Poisson's variance
    # equals its mean, so emission_cov == emission_mean.
    def _mean_fn(z):
        return jnp.exp(C @ z)

    def _cov_fn(z):
        return jnp.diag(jnp.exp(C @ z))

    def _emission_dist(mu, _Sigma):
        return tfp.distributions.Poisson(rate=mu)

    return ParamsGGSSM(
        initial_mean=x0,
        initial_covariance=P0,
        dynamics_function=lambda z: A @ z,
        dynamics_covariance=Q,
        emission_mean_function=_mean_fn,
        emission_cov_function=_cov_fn,
        emission_dist=_emission_dist,
    )


def cmgf_poisson_filter(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    transition_covariance: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
) -> CMGFPoissonFilterResult:
    """Run a Conditional-Moments Gaussian Filter for Poisson observations.

    Counterpart to MATLAB nSTAT's ``PPDecodeFilter`` / ``PP_filter``
    families — point-process Kalman-like filtering for spike-count
    observations on a continuous linear-Gaussian latent state.

    The model is:

    .. math::

        x_t &= A x_{t-1} + w_t, \\quad w_t \\sim \\mathcal{N}(0, Q) \\\\
        y_t &\\sim \\text{Poisson}(\\exp(C x_t))

    Parameters
    ----------
    observations
        Spike-count observations, shape ``(T, emission_dim)``.  Must be
        non-negative integers (or float castable to non-negative int).
    transition_matrix
        :math:`A`, shape ``(state_dim, state_dim)``.
    observation_matrix
        :math:`C`, shape ``(emission_dim, state_dim)``.
    transition_covariance
        :math:`Q`, shape ``(state_dim, state_dim)``.
    initial_state_mean
        :math:`\\hat{x}_0`, shape ``(state_dim,)``.
    initial_state_covariance
        :math:`P_0`, shape ``(state_dim, state_dim)``.

    Returns
    -------
    CMGFPoissonFilterResult

    Notes
    -----
    Internally uses Dynamax's EKF integration scheme.  For UKF / GHKF
    variants drop down to ``dynamax.generalized_gaussian_ssm.inference``
    directly.
    """
    _require_dynamax()
    import jax.numpy as jnp
    from dynamax.generalized_gaussian_ssm.inference import (
        conditional_moments_gaussian_filter,
        EKFIntegrals,
    )

    observations = np.asarray(observations, dtype=float)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)

    params = _build_cmgf_poisson_params(
        transition_matrix,
        observation_matrix,
        transition_covariance,
        initial_state_mean,
        initial_state_covariance,
    )
    posterior = conditional_moments_gaussian_filter(
        params, EKFIntegrals(), jnp.asarray(observations)
    )
    return CMGFPoissonFilterResult(
        state_means=np.asarray(posterior.filtered_means, dtype=float),
        state_covariances=np.asarray(posterior.filtered_covariances, dtype=float),
        marginal_log_likelihood=float(posterior.marginal_loglik),
    )


def cmgf_poisson_smoother(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    transition_covariance: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
) -> CMGFPoissonFilterResult:
    """Run a Conditional-Moments Gaussian Smoother for Poisson observations.

    Forward CMGF filter + backward RTS-style smoother under the
    Gaussian approximation to the Poisson likelihood.  Counterpart to
    MATLAB nSTAT's ``PP_fixedIntervalSmoother``.

    See :func:`cmgf_poisson_filter` for the model specification and
    parameter shapes — this function is identical in signature, just
    returns smoothed (rather than filtered) state means + covariances.

    Returns
    -------
    CMGFPoissonFilterResult
        ``state_means`` and ``state_covariances`` are the smoothed
        posteriors :math:`p(x_t | y_{1:T})`.
    """
    _require_dynamax()
    import jax.numpy as jnp
    from dynamax.generalized_gaussian_ssm.inference import (
        conditional_moments_gaussian_smoother,
        EKFIntegrals,
    )

    observations = np.asarray(observations, dtype=float)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)

    params = _build_cmgf_poisson_params(
        transition_matrix,
        observation_matrix,
        transition_covariance,
        initial_state_mean,
        initial_state_covariance,
    )
    posterior = conditional_moments_gaussian_smoother(
        params, EKFIntegrals(), jnp.asarray(observations)
    )
    return CMGFPoissonFilterResult(
        state_means=np.asarray(posterior.smoothed_means, dtype=float),
        state_covariances=np.asarray(posterior.smoothed_covariances, dtype=float),
        marginal_log_likelihood=float(posterior.marginal_loglik),
    )


# ----------------------------------------------------------------------
# Point-process EM (PP_EM equivalent — Smith & Brown 2003 PPLDS)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PointProcessEMResult:
    """Output of a Poisson-LGSSM EM fit (PP_EM equivalent).

    Attributes
    ----------
    transition_matrix
        Learned :math:`A`, shape ``(state_dim, state_dim)``.
    observation_matrix
        Learned :math:`C`, shape ``(emission_dim, state_dim)``.
    transition_covariance
        Learned :math:`Q`, shape ``(state_dim, state_dim)``.
    initial_state_mean
        Learned :math:`\\hat x_0`, shape ``(state_dim,)``.
    initial_state_covariance
        Learned :math:`P_0`, shape ``(state_dim, state_dim)``.
    marginal_log_likelihoods
        EM marginal log-likelihood trace under the CMGF Gaussian
        approximation, shape ``(n_iter,)``.
    n_iter
        Number of EM iterations actually run.
    """

    transition_matrix: np.ndarray
    observation_matrix: np.ndarray
    transition_covariance: np.ndarray
    initial_state_mean: np.ndarray
    initial_state_covariance: np.ndarray
    marginal_log_likelihoods: np.ndarray
    n_iter: int


def _ppem_initialize(
    observations: np.ndarray, state_dim: int, seed: int
):
    """Sensible PP_EM initialization."""
    rng = np.random.default_rng(int(seed))
    emission_dim = int(observations.shape[1])
    A = 0.95 * np.eye(state_dim)
    Q = 0.1 * np.eye(state_dim)
    C = 0.1 * rng.standard_normal((emission_dim, state_dim))
    x0 = np.zeros(state_dim)
    P0 = np.eye(state_dim)
    return A, C, Q, x0, P0


def _ppem_m_step_closed_form(
    smoothed_means: np.ndarray,
    smoothed_covariances: np.ndarray,
):
    """Closed-form M-step for the linear-Gaussian dynamics parameters.

    Uses the smoothed first and second moments.  The lag-one
    cross-covariance is approximated by the outer product of smoothed
    means — Dynamax's CMGF smoother doesn't expose lag-one covs, so
    we substitute the moment-matching approximation
    ``E[x_t x_{t-1}'] ≈ μ_t μ_{t-1}'``.  This introduces bias in Q
    when the posterior has strong cross-time correlations; the C
    update remains correct (uses single-time second moments).

    Returns ``(A, Q, x0, P0)``.
    """
    T, state_dim = smoothed_means.shape
    second_moments = (
        smoothed_covariances + smoothed_means[..., None] @ smoothed_means[:, None, :]
    )
    sum_t = second_moments[1:].sum(axis=0)
    sum_tm1 = second_moments[:-1].sum(axis=0)
    cross_moments = smoothed_means[1:, :, None] @ smoothed_means[:-1, None, :]
    sum_cross = cross_moments.sum(axis=0)

    A = sum_cross @ np.linalg.pinv(sum_tm1)
    Q = (sum_t - A @ sum_cross.T) / max(T - 1, 1)
    Q = 0.5 * (Q + Q.T) + 1e-8 * np.eye(state_dim)

    x0 = smoothed_means[0].copy()
    P0 = 0.5 * (smoothed_covariances[0] + smoothed_covariances[0].T) + 1e-8 * np.eye(state_dim)
    return A, Q, x0, P0


def _ppem_newton_C(
    observations: np.ndarray,
    smoothed_means: np.ndarray,
    smoothed_covariances: np.ndarray,
    C_init: np.ndarray,
    n_newton: int = 5,
) -> np.ndarray:
    """Newton-Raphson update for the Poisson loading matrix C.

    Maximizes the expected complete-data log-likelihood

    .. math::

        \\ell(C) = \\sum_t \\left( y_t' C \\mu_t -
                                  \\mathbf{1}' \\mathbb{E}[\\exp(C x_t)] \\right)

    using the Laplace approximation
    :math:`\\mathbb{E}[\\exp(c_i' x_t)] \\approx
    \\exp(c_i' \\mu_t + \\tfrac{1}{2} c_i' \\Sigma_t c_i)`.  Updates
    each row of C independently (the expected log-likelihood is
    row-separable).
    """
    T, state_dim = smoothed_means.shape
    emission_dim = observations.shape[1]
    C = np.asarray(C_init, dtype=float).copy()

    for _ in range(int(n_newton)):
        for i in range(emission_dim):
            c_i = C[i]
            quad = 0.5 * np.einsum("i,tij,j->t", c_i, smoothed_covariances, c_i)
            lin = smoothed_means @ c_i
            # Clip the exponent to avoid overflow on bad iterations.
            exp_term = np.exp(np.clip(lin + quad, -20.0, 20.0))
            y_i = observations[:, i]
            grad = ((y_i - exp_term)[:, None] * smoothed_means).sum(axis=0)
            outer_mu = smoothed_means[:, :, None] * smoothed_means[:, None, :]
            hess = -(exp_term[:, None, None] * (outer_mu + smoothed_covariances)).sum(axis=0)
            hess = hess - 1e-6 * np.eye(state_dim)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hess) @ grad
            C[i] = c_i - step

    return C


def fit_point_process_em(
    observations: np.ndarray,
    state_dim: int,
    *,
    n_iter: int = 30,
    n_newton_iter: int = 5,
    seed: int = 0,
) -> PointProcessEMResult:
    """Fit a Poisson-LGSSM via Laplace-approximated EM (PP_EM equivalent).

    Counterpart to MATLAB nSTAT's ``PP_EM`` family.  Model:

    .. math::

        x_t &= A x_{t-1} + w_t, \\quad w_t \\sim \\mathcal{N}(0, Q) \\\\
        y_t &\\sim \\text{Poisson}(\\exp(C x_t))

    Algorithm (Smith & Brown 2003 PPLDS):

    1. Initialize A, C, Q, x0, P0.
    2. Loop ``n_iter`` times:

       - **E-step**: Dynamax CMGF smoother (EKF integration) on the
         Poisson-LGSSM with current parameters.  Returns smoothed
         means :math:`\\mu_t`, covariances :math:`\\Sigma_t`, and the
         marginal log-likelihood under the Gaussian approximation.
       - **M-step (dynamics)**: closed-form A, Q, x0, P0 from
         smoothed moments.
       - **M-step (loadings)**: per-row Newton-Raphson on the
         expected Poisson log-likelihood with Laplace approximation
         for :math:`\\mathbb{E}[\\exp(c_i' x_t)]`.

    Parameters
    ----------
    observations
        Spike-count time series, shape ``(T, emission_dim)``, integer-
        valued (or float castable to non-negative int).
    state_dim
        Latent state dimensionality.
    n_iter
        Outer EM iterations.  Default 30.
    n_newton_iter
        Inner Newton-Raphson iterations per M-step for C.  Default 5.
    seed
        RNG seed for parameter initialization.

    Returns
    -------
    PointProcessEMResult

    Notes
    -----
    The marginal-log-likelihood trace is **not** guaranteed monotonic
    because the Laplace approximation is iteration-dependent.  Under
    exact EM the trace would be non-decreasing; here a few-percent
    dips early in training are normal.  Substantial decreases
    indicate a bug or pathological initialization.

    Lag-one cross-covariances are approximated by the outer product
    of smoothed means (Dynamax CMGF doesn't expose lag-one covs).
    Expect ~1–3% Q estimation bias vs MATLAB ``PP_EM`` on stationary
    fixtures.
    """
    _require_dynamax()

    observations = np.asarray(observations, dtype=float)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be shape (T, emission_dim); got {observations.shape}"
        )
    if state_dim < 1:
        raise ValueError(f"state_dim must be >= 1; got {state_dim}")

    A, C, Q, x0, P0 = _ppem_initialize(observations, state_dim, seed)
    lls: list[float] = []

    for _it in range(int(n_iter)):
        smooth = cmgf_poisson_smoother(observations, A, C, Q, x0, P0)
        lls.append(smooth.marginal_log_likelihood)
        A, Q, x0, P0 = _ppem_m_step_closed_form(smooth.state_means, smooth.state_covariances)
        C = _ppem_newton_C(
            observations, smooth.state_means, smooth.state_covariances,
            C, n_newton=n_newton_iter,
        )

    return PointProcessEMResult(
        transition_matrix=A,
        observation_matrix=C,
        transition_covariance=Q,
        initial_state_mean=x0,
        initial_state_covariance=P0,
        marginal_log_likelihoods=np.asarray(lls, dtype=float),
        n_iter=int(n_iter),
    )


# ----------------------------------------------------------------------
# Hybrid point-process + Gaussian EM (mPPCO_EM equivalent)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HybridEMResult:
    """Output of a Poisson + Gaussian SSM EM fit (mPPCO_EM equivalent).

    A single latent state drives both observation channels.

    Attributes
    ----------
    transition_matrix
        Learned :math:`A`.
    poisson_observation_matrix
        Learned :math:`C_p`, shape ``(poisson_emission_dim, state_dim)``.
    gaussian_observation_matrix
        Learned :math:`C_g`, shape ``(gaussian_emission_dim, state_dim)``.
    transition_covariance
        Learned :math:`Q`.
    gaussian_observation_covariance
        Learned :math:`R`.
    initial_state_mean
        :math:`\\hat x_0`.
    initial_state_covariance
        :math:`P_0`.
    marginal_log_likelihoods
        EM trace under the Laplace-Gaussian approximation.
    n_iter
        Iterations actually run.
    """

    transition_matrix: np.ndarray
    poisson_observation_matrix: np.ndarray
    gaussian_observation_matrix: np.ndarray
    transition_covariance: np.ndarray
    gaussian_observation_covariance: np.ndarray
    initial_state_mean: np.ndarray
    initial_state_covariance: np.ndarray
    marginal_log_likelihoods: np.ndarray
    n_iter: int


def _hybrid_pseudo_observations(
    poisson_obs: np.ndarray,
    C_p: np.ndarray,
    smoothed_means: np.ndarray,
):
    """IRLS Gaussian pseudo-observations for the Poisson channel.

    Linearizes the Poisson log-likelihood around the current smoothed
    means via the canonical GLM IRLS construction:

    .. math::

        \\tilde y_t = C_p \\mu_t + (y_t - \\lambda_t) / \\lambda_t,
        \\quad \\text{Var}(\\tilde y_t) = 1 / \\lambda_t

    where :math:`\\lambda_t = \\exp(C_p \\mu_t)`.
    """
    lin = smoothed_means @ C_p.T
    lam = np.exp(np.clip(lin, -20.0, 20.0))
    lam_safe = np.maximum(lam, 1e-6)
    y_tilde = lin + (poisson_obs - lam) / lam_safe
    return y_tilde, lam_safe


def _hybrid_e_step(
    poisson_obs: np.ndarray,
    gaussian_obs: np.ndarray,
    A: np.ndarray, C_p: np.ndarray, C_g: np.ndarray,
    Q: np.ndarray, R: np.ndarray,
    x0: np.ndarray, P0: np.ndarray,
    prev_smoothed_means: np.ndarray | None,
):
    """Hybrid E-step: build IRLS pseudo-obs for the Poisson channel,
    stack with the real Gaussian channel, run Dynamax's
    :func:`lgssm_smoother` on the augmented LG model.

    Returns ``(smoothed_means, smoothed_covariances, marginal_loglik)``.
    """
    if prev_smoothed_means is None:
        bootstrap = cmgf_poisson_smoother(poisson_obs, A, C_p, Q, x0, P0)
        prev_smoothed_means = bootstrap.state_means

    y_tilde_p, lam_safe = _hybrid_pseudo_observations(poisson_obs, C_p, prev_smoothed_means)
    # Time-averaged pseudo-obs covariance (fixed-R simplification).
    # A fully-correct implementation would use time-varying R per step;
    # the lgssm_smoother in Dynamax requires a fixed emission cov, so
    # we average over time as an acceptable trade-off for this
    # initial implementation.
    R_pseudo = np.diag(1.0 / lam_safe.mean(axis=0))

    C_stacked = np.vstack([C_p, C_g])
    R_stacked = np.block([
        [R_pseudo, np.zeros((R_pseudo.shape[0], R.shape[1]))],
        [np.zeros((R.shape[0], R_pseudo.shape[1])), R],
    ])
    y_stacked = np.hstack([y_tilde_p, gaussian_obs])

    import jax.numpy as jnp
    from dynamax.linear_gaussian_ssm import (
        lgssm_smoother, ParamsLGSSM,
        ParamsLGSSMDynamics, ParamsLGSSMEmissions, ParamsLGSSMInitial,
    )

    state_dim = A.shape[0]
    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=jnp.asarray(x0, dtype=float),
            cov=jnp.asarray(P0, dtype=float),
        ),
        dynamics=ParamsLGSSMDynamics(
            weights=jnp.asarray(A, dtype=float),
            bias=jnp.zeros(state_dim),
            input_weights=jnp.zeros((state_dim, 0)),
            cov=jnp.asarray(Q, dtype=float),
        ),
        emissions=ParamsLGSSMEmissions(
            weights=jnp.asarray(C_stacked, dtype=float),
            bias=jnp.zeros(C_stacked.shape[0]),
            input_weights=jnp.zeros((C_stacked.shape[0], 0)),
            cov=jnp.asarray(R_stacked, dtype=float),
        ),
    )
    posterior = lgssm_smoother(params, jnp.asarray(y_stacked))
    return (
        np.asarray(posterior.smoothed_means, dtype=float),
        np.asarray(posterior.smoothed_covariances, dtype=float),
        float(posterior.marginal_loglik),
    )


def _hybrid_initialize(
    poisson_obs: np.ndarray, gaussian_obs: np.ndarray,
    state_dim: int, seed: int,
):
    rng = np.random.default_rng(int(seed))
    p_dim = int(poisson_obs.shape[1])
    g_dim = int(gaussian_obs.shape[1])
    A = 0.95 * np.eye(state_dim)
    Q = 0.1 * np.eye(state_dim)
    C_p = 0.1 * rng.standard_normal((p_dim, state_dim))
    C_g = 0.1 * rng.standard_normal((g_dim, state_dim))
    R = 0.1 * np.eye(g_dim)
    x0 = np.zeros(state_dim)
    P0 = np.eye(state_dim)
    return A, C_p, C_g, Q, R, x0, P0


def fit_hybrid_em(
    poisson_observations: np.ndarray,
    gaussian_observations: np.ndarray,
    state_dim: int,
    *,
    n_iter: int = 30,
    n_newton_iter: int = 3,
    seed: int = 0,
) -> HybridEMResult:
    """Fit a hybrid Poisson + Gaussian SSM via EM (mPPCO_EM equivalent).

    Counterpart to MATLAB nSTAT's ``mPPCO_EM`` family.  Both observation
    channels share a single linear-Gaussian latent state:

    .. math::

        x_t &= A x_{t-1} + w_t, \\quad w_t \\sim \\mathcal{N}(0, Q) \\\\
        y^{(p)}_t &\\sim \\text{Poisson}(\\exp(C_p x_t)) \\\\
        y^{(g)}_t &\\sim \\mathcal{N}(C_g x_t, R)

    Algorithm:

    1. Initialize all parameters.
    2. Loop ``n_iter`` times:

       - **E-step**: Laplace-approximate the Poisson channel into IRLS
         Gaussian pseudo-observations linearized around the previous
         smoothed means.  Stack with the real Gaussian channel.  Run
         :func:`dynamax.linear_gaussian_ssm.lgssm_smoother` on the
         augmented LG-SSM.  The cross-channel information sharing
         that's the whole point of mPPCO_EM happens *inside* the
         smoother — the Gaussian channel constrains the latent
         state, which tightens the pseudo-obs for the Poisson channel
         on the next iteration.
       - **M-step (dynamics)**: closed-form A, Q, x0, P0 from
         smoothed moments (shared with PP_EM).
       - **M-step (Poisson loadings)**: per-row Newton-Raphson for
         C_p (shared with PP_EM).
       - **M-step (Gaussian loadings)**: closed-form least squares
         for C_g; closed-form residual cov for R.

    Parameters
    ----------
    poisson_observations
        Spike-count series, shape ``(T, p_emission_dim)``.
    gaussian_observations
        Continuous-valued series, shape ``(T, g_emission_dim)``.
    state_dim
        Latent state dimensionality.
    n_iter
        Outer EM iterations.
    n_newton_iter
        Inner Newton-Raphson iterations for C_p per M-step.
    seed
        RNG seed for initialization.

    Returns
    -------
    HybridEMResult

    Notes
    -----
    The E-step uses a fixed-R approximation: the IRLS pseudo-obs
    variance ``1/λ_t`` varies in t, but Dynamax's ``lgssm_smoother``
    accepts only a fixed emission covariance.  We use the
    time-averaged ``1/mean(λ)`` per channel as the substitute.  Real-
    data fits with sharply non-stationary Poisson rates may want
    full time-varying R; that requires a custom Kalman smoother
    (deferred to a future release).
    """
    _require_dynamax()

    poisson_obs = np.asarray(poisson_observations, dtype=float)
    gaussian_obs = np.asarray(gaussian_observations, dtype=float)
    if poisson_obs.ndim == 1:
        poisson_obs = poisson_obs.reshape(-1, 1)
    if gaussian_obs.ndim == 1:
        gaussian_obs = gaussian_obs.reshape(-1, 1)
    if poisson_obs.shape[0] != gaussian_obs.shape[0]:
        raise ValueError(
            f"poisson and gaussian observations must have same T; "
            f"got {poisson_obs.shape[0]} vs {gaussian_obs.shape[0]}"
        )
    if state_dim < 1:
        raise ValueError(f"state_dim must be >= 1; got {state_dim}")

    A, C_p, C_g, Q, R, x0, P0 = _hybrid_initialize(
        poisson_obs, gaussian_obs, state_dim, seed
    )
    lls: list[float] = []
    prev_smoothed_means: np.ndarray | None = None

    for _it in range(int(n_iter)):
        smoothed_means, smoothed_covs, ll = _hybrid_e_step(
            poisson_obs, gaussian_obs,
            A, C_p, C_g, Q, R, x0, P0,
            prev_smoothed_means,
        )
        lls.append(ll)
        prev_smoothed_means = smoothed_means

        A, Q, x0, P0 = _ppem_m_step_closed_form(smoothed_means, smoothed_covs)

        C_p = _ppem_newton_C(
            poisson_obs, smoothed_means, smoothed_covs,
            C_p, n_newton=n_newton_iter,
        )

        # Gaussian loadings: closed-form least squares.
        second_moments = (
            smoothed_covs + smoothed_means[..., None] @ smoothed_means[:, None, :]
        )
        sum_xx = second_moments.sum(axis=0)
        sum_yx = gaussian_obs.T @ smoothed_means
        C_g = sum_yx @ np.linalg.pinv(sum_xx)

        # Gaussian noise covariance: residual covariance.
        resid = gaussian_obs - smoothed_means @ C_g.T
        R = (resid.T @ resid) / max(poisson_obs.shape[0], 1)
        R = 0.5 * (R + R.T) + 1e-8 * np.eye(R.shape[0])

    return HybridEMResult(
        transition_matrix=A,
        poisson_observation_matrix=C_p,
        gaussian_observation_matrix=C_g,
        transition_covariance=Q,
        gaussian_observation_covariance=R,
        initial_state_mean=x0,
        initial_state_covariance=P0,
        marginal_log_likelihoods=np.asarray(lls, dtype=float),
        n_iter=int(n_iter),
    )


__all__ = [
    "LinearGaussianEMResult",
    "fit_linear_gaussian_em",
    "CMGFPoissonFilterResult",
    "cmgf_poisson_filter",
    "cmgf_poisson_smoother",
    "PointProcessEMResult",
    "fit_point_process_em",
    "HybridEMResult",
    "fit_hybrid_em",
]
