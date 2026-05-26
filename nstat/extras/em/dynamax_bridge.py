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


__all__ = ["LinearGaussianEMResult", "fit_linear_gaussian_em"]
