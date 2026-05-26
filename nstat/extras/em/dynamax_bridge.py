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


__all__ = [
    "LinearGaussianEMResult",
    "fit_linear_gaussian_em",
    "CMGFPoissonFilterResult",
    "cmgf_poisson_filter",
    "cmgf_poisson_smoother",
]
