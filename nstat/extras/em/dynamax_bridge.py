"""EM-trained linear-Gaussian state-space models via Dynamax.

This module wraps :mod:`dynamax.linear_gaussian_ssm.LinearGaussianSSM`
behind a thin nstat-style API so users can fit EM-trained state-space
models without nstat owning the EM code itself.

Scope
-----
- :func:`fit_linear_gaussian_em` — KF_EM equivalent (linear-Gaussian EM).
- :func:`cmgf_poisson_filter` / :func:`cmgf_poisson_smoother` —
  PPDecodeFilter / PP_fixedIntervalSmoother (point-process inference).
- :func:`fit_point_process_em` — PP_EM equivalent (Poisson-LGSSM EM).
- :func:`fit_hybrid_em` — mPPCO_EM equivalent (Poisson + Gaussian EM).
- :func:`point_process_predictive_ll` / :func:`hybrid_predictive_ll` —
  true one-step-ahead held-out predictive log-likelihood, a valid
  convergence / model-comparison diagnostic (pure NumPy; does **not**
  require dynamax).

All fit/inference routines return plain NumPy arrays (NOT pytrees) so
callers stay decoupled from JAX.

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
    smoothed_cross_covariances: np.ndarray | None = None,
):
    """Closed-form M-step for the linear-Gaussian dynamics parameters.

    Uses the EM sufficient statistics

    .. math::

        S_{11} &= \\sum_{t} \\mathbb{E}[x_t x_t'] = \\sum_t (\\Sigma_t + \\mu_t \\mu_t') \\\\
        S_{10} &= \\sum_{t} \\mathbb{E}[x_t x_{t-1}'] = \\sum_t (P_{t,t-1} + \\mu_t \\mu_{t-1}')

    where :math:`P_{t,t-1} = \\mathrm{Cov}[x_t, x_{t-1} \\mid y_{1:T}]` is
    the **lag-one smoothed cross-covariance**.

    The lag-one term is essential: dropping it (the moment-matching
    approximation ``E[x_t x_{t-1}'] ≈ μ_t μ_{t-1}'``) systematically
    biases A toward zero, which in a Poisson observation model triggers
    a degenerate feedback loop — the smoothed states shrink toward the
    prior, and the C Newton step then inflates the loadings without
    bound to compensate.  When ``smoothed_cross_covariances`` is
    supplied (from ``lgssm_smoother``), the exact lag-one term is used
    and A/Q recover correctly.  ``None`` falls back to moment-matching
    (kept only for the CMGF bootstrap, which has no cross-covs).

    Parameters
    ----------
    smoothed_means
        ``(T, state_dim)`` smoothed means μ_t.
    smoothed_covariances
        ``(T, state_dim, state_dim)`` smoothed covariances Σ_t.
    smoothed_cross_covariances
        ``(T-1, state_dim, state_dim)`` lag-one cross-covariances where
        entry ``[t]`` is ``Cov[x_{t+1}, x_t | y_{1:T}]`` (Dynamax
        convention).  If ``None``, the moment-matching approximation is
        used (biased — see above).

    Returns ``(A, Q, x0, P0)``.
    """
    T, state_dim = smoothed_means.shape
    second_moments = (
        smoothed_covariances + smoothed_means[..., None] @ smoothed_means[:, None, :]
    )
    sum_t = second_moments[1:].sum(axis=0)
    sum_tm1 = second_moments[:-1].sum(axis=0)

    # E[x_t x_{t-1}'] = Cov[x_t, x_{t-1}] + μ_t μ_{t-1}', summed t=1..T-1.
    mean_cross = (smoothed_means[1:, :, None] @ smoothed_means[:-1, None, :]).sum(axis=0)
    if smoothed_cross_covariances is not None:
        sum_cross = np.asarray(smoothed_cross_covariances, dtype=float).sum(axis=0) + mean_cross
    else:
        sum_cross = mean_cross  # biased fallback (moment-matching)

    A = sum_cross @ np.linalg.pinv(sum_tm1)
    Q = (sum_t - A @ sum_cross.T) / max(T - 1, 1)
    Q = 0.5 * (Q + Q.T) + 1e-8 * np.eye(state_dim)

    x0 = smoothed_means[0].copy()
    P0 = 0.5 * (smoothed_covariances[0] + smoothed_covariances[0].T) + 1e-8 * np.eye(state_dim)
    return A, Q, x0, P0


def _canonical_scale(smoothed_means, smoothed_covariances):
    """Per-dimension latent RMS — a *lightweight* in-loop scale pin.

    Used INSIDE the EM loop only to keep ``|C|`` finite (stop the scale
    drift of the PLDS gauge ridge).  It is intentionally cheap (diagonal,
    no rotation): a full gauge transform applied every iteration changes
    the optimization landscape each step and destabilizes the EM.  The
    full canonical form is applied once after convergence by
    :func:`_canonicalize_gauge`.
    """
    T = smoothed_means.shape[0]
    e_x2 = (np.einsum("tjj->j", smoothed_covariances) + (smoothed_means ** 2).sum(axis=0)) / T
    return np.sqrt(np.maximum(e_x2, 1e-8))


def _apply_canonical_scale(A, C, Q, x0, P0, s):
    """Apply the diagonal scale transform ``T = diag(1/s)`` (loop stabilizer)."""
    Tinv = np.diag(1.0 / s)
    Tfwd = np.diag(s)
    return (Tinv @ A @ Tfwd, C @ Tfwd, Tinv @ Q @ Tinv.T, Tinv @ x0, Tinv @ P0 @ Tinv.T)


def _canonicalize_gauge(A, C_list, Q, x0, P0, smoothed_means, smoothed_covariances):
    """Pin the **full** PLDS identifiability gauge to a canonical form.

    Apply ONCE after EM convergence (not per-iteration — see
    :func:`_canonical_scale`).

    PLDS/PPLDS models are invariant under the state reparameterization
    ``(A, C, x) → (T A T⁻¹, C T⁻¹, T x)`` for any invertible ``T`` (the
    observable log-rate ``C x``, hence the likelihood, is unchanged).
    The gauge group is the full ``GL(d)`` — ``d²`` degrees of freedom —
    so a diagonal scale normalization (which pins only ``d`` of them)
    leaves a residual rotation that lets ``A``/``C`` drift across seeds.

    This pins the entire gauge with the standard LDS canonical form
    (cf. Macke et al. 2011; Buesing et al. 2012):

    1. **Whiten** the latent: choose ``T`` so the empirical state
       second moment ``M = (1/T) Σ_t E[x_t x_t']`` becomes the identity
       (``T = M^{-1/2}``).  Removes the symmetric part of the gauge.
    2. **SVD-rotate**: of the remaining orthogonal freedom, pick the
       rotation that makes the *stacked* emission matrix have orthogonal
       columns ordered by descending singular value (``C_canon = U S``).
       Removes the residual ``O(d)``.
    3. **Sign-fix**: flip each latent axis so the largest-magnitude
       entry of each canonical emission column is positive — a
       deterministic representative (removes the remaining ``2^d`` sign
       flips).

    The result is a unique, seed-stable representative of the parameter
    equivalence class; ``C x`` is exactly preserved for every emission
    matrix in ``C_list``.

    Parameters
    ----------
    A, Q, x0, P0
        Current dynamics parameters.
    C_list
        List of emission matrices sharing the latent (``[C]`` for
        PP_EM; ``[C_p, C_g]`` for the hybrid).  The canonical rotation
        is computed from their vertical stack.
    smoothed_means, smoothed_covariances
        Current E-step posteriors, used to form ``M``.

    Returns
    -------
    ``(A, C_list, Q, x0, P0)`` in canonical coordinates.
    """
    T, d = smoothed_means.shape
    M = (smoothed_covariances.sum(axis=0) + smoothed_means.T @ smoothed_means) / T
    M = 0.5 * (M + M.T)

    # Symmetric inverse-sqrt (and sqrt) of M via eigendecomposition.
    evals, evecs = np.linalg.eigh(M)
    evals = np.maximum(evals, 1e-8)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T     # M^{-1/2}; x' = W x  ⇒ E[x'x']=I
    Winv = evecs @ np.diag(evals ** 0.5) @ evecs.T   # M^{1/2}

    # Whitened stacked emission matrix, then SVD to pin the rotation.
    C_stack = np.vstack([np.atleast_2d(C) for C in C_list])
    C_white = C_stack @ Winv
    _, _, Vt = np.linalg.svd(C_white, full_matrices=False)
    # Canonical stacked emission after rotation: C_white @ Vt.T = U @ diag(S).
    C_canon = C_white @ Vt.T

    # Sign convention: make the max-magnitude entry of each column positive.
    lead = np.argmax(np.abs(C_canon), axis=0)
    signs = np.sign(C_canon[lead, np.arange(d)])
    signs[signs == 0] = 1.0
    sign_d = np.diag(signs)

    # Total forward transform x' = Ttot x, and its inverse.
    Ttot = sign_d @ Vt @ W
    Tinv = Winv @ Vt.T @ sign_d   # = Ttot^{-1} (sign_d, Vt orthogonal, W·Winv=I)

    A_new = Ttot @ A @ Tinv
    Q_new = Ttot @ Q @ Ttot.T
    Q_new = 0.5 * (Q_new + Q_new.T)
    x0_new = Ttot @ x0
    P0_new = Ttot @ P0 @ Ttot.T
    P0_new = 0.5 * (P0_new + P0_new.T)
    C_list_new = [np.atleast_2d(C) @ Tinv for C in C_list]
    return A_new, C_list_new, Q_new, x0_new, P0_new


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
            # Trust-region cap: bound the per-iteration step norm so a
            # single bad Newton step can't send the loadings to
            # infinity along the (unconstrained) PLDS gauge ridge.
            step_norm = float(np.linalg.norm(step))
            max_step = 2.0
            if step_norm > max_step:
                step = step * (max_step / step_norm)
            C[i] = c_i - step

    return C


def _kalman_rts_smoother_tv(
    y: np.ndarray,
    A: np.ndarray, C: np.ndarray, Q: np.ndarray,
    R_t: np.ndarray,
    x0: np.ndarray, P0: np.ndarray,
):
    """Pure-NumPy Kalman filter + RTS smoother with **time-varying**
    observation noise.

    Unlike Dynamax's batched ``lgssm_smoother`` (fixed emission cov),
    this accepts per-timestep ``R_t`` — essential for the Poisson IRLS
    E-step, where the working-response variance is ``1/λ_t`` and varies
    across time.  A fixed-R substitution breaks the GLM weight
    cancellation and is numerically unstable for low rates.

    Returns ``(xs, Ps, cross, ll)`` where
    ``xs`` is ``(T, d)`` smoothed means,
    ``Ps`` is ``(T, d, d)`` smoothed covariances,
    ``cross[t] = Cov[x_{t+1}, x_t | y_{1:T}]`` is ``(T-1, d, d)``
    (Dynamax orientation), and ``ll`` is the Gaussian marginal
    log-likelihood of the (pseudo-)observations.
    """
    y = np.asarray(y, dtype=float)
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R_t = np.asarray(R_t, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    P0 = np.asarray(P0, dtype=float)

    T, m = y.shape
    d = A.shape[0]

    xp = np.zeros((T, d))       # one-step predicted means
    Pp = np.zeros((T, d, d))    # one-step predicted covs
    xf = np.zeros((T, d))       # filtered means
    Pf = np.zeros((T, d, d))    # filtered covs
    ll = 0.0
    log2pi = float(np.log(2.0 * np.pi))

    x_prev, P_prev = x0, P0
    for t in range(T):
        if t == 0:
            xp[t], Pp[t] = x0, P0
        else:
            xp[t] = A @ x_prev
            Pp[t] = A @ P_prev @ A.T + Q
        S = C @ Pp[t] @ C.T + R_t[t]
        S = 0.5 * (S + S.T) + 1e-9 * np.eye(m)
        S_inv = np.linalg.inv(S)
        K = Pp[t] @ C.T @ S_inv
        innov = y[t] - C @ xp[t]
        xf[t] = xp[t] + K @ innov
        Pf[t] = Pp[t] - K @ C @ Pp[t]
        Pf[t] = 0.5 * (Pf[t] + Pf[t].T)
        sign, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (m * log2pi + logdet + float(innov @ S_inv @ innov))
        x_prev, P_prev = xf[t], Pf[t]

    xs = xf.copy()
    Ps = Pf.copy()
    cross = np.zeros((max(T - 1, 0), d, d))
    for t in range(T - 2, -1, -1):
        Pp_next_inv = np.linalg.inv(
            0.5 * (Pp[t + 1] + Pp[t + 1].T) + 1e-9 * np.eye(d)
        )
        J = Pf[t] @ A.T @ Pp_next_inv          # RTS gain at time t
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
        Ps[t] = 0.5 * (Ps[t] + Ps[t].T)
        # Lag-one smoothed cross-covariance Cov[x_{t+1}, x_t | y_{1:T}].
        cross[t] = Ps[t + 1] @ J.T

    return xs, Ps, cross, ll


def _ppem_e_step_lgssm(
    poisson_obs: np.ndarray,
    A: np.ndarray, C: np.ndarray, Q: np.ndarray,
    x0: np.ndarray, P0: np.ndarray,
    prev_smoothed_means: np.ndarray | None,
):
    """Poisson E-step: IRLS pseudo-observations + a time-varying-R RTS
    smoother.

    The Poisson likelihood is linearized around the previous smoothed
    means (Fisher-scoring working response ``z_t = C μ_t + (y_t-λ_t)/λ_t``
    with per-timestep variance ``1/λ_t``).  Crucially the smoother uses
    the **time-varying** noise ``R_t = diag(1/λ_t)`` — substituting a
    fixed R (as a batched smoother forces) breaks the IRLS
    weight-cancellation and is numerically unstable at low rates.  The
    smoother also returns lag-one cross-covariances, required for an
    unbiased A/Q M-step (without them A collapses to zero and C
    diverges).

    First iteration bootstraps the linearization point with the CMGF
    smoother (which needs no prior estimate).

    Returns ``(smoothed_means, smoothed_covariances,
    smoothed_cross_covariances, marginal_loglik)``.
    """
    if prev_smoothed_means is None:
        prev_smoothed_means = cmgf_poisson_smoother(poisson_obs, A, C, Q, x0, P0).state_means

    y_tilde, lam_safe = _hybrid_pseudo_observations(poisson_obs, C, prev_smoothed_means)
    # Per-timestep IRLS working-response variance: Var(z_t) = 1/λ_t.
    T = y_tilde.shape[0]
    em_dim = C.shape[0]
    R_t = np.zeros((T, em_dim, em_dim))
    inv_lam = 1.0 / lam_safe
    for t in range(T):
        R_t[t] = np.diag(inv_lam[t])

    means, covs, cross, ll = _kalman_rts_smoother_tv(
        y_tilde, A, C, Q, R_t, x0, P0
    )
    return means, covs, cross, ll


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

    Warnings
    --------
    **Experimental.**  A Poisson LDS has a gauge freedom
    :math:`(A, C, x) \\to (T A T^{-1}, C T^{-1}, T x)` that leaves the
    observable log-rate :math:`C x` (hence the likelihood) invariant
    for any invertible :math:`T` (the full :math:`GL(d)` group).  This
    implementation pins that gauge to a canonical form — a cheap
    diagonal scale pin during the loop, then a single whiten + SVD-rotate
    + sign-fix after convergence (see :func:`_canonicalize_gauge`) — so
    the returned ``C`` satisfies :math:`C^\\top C = \\mathrm{diag}(S^2)`
    and is a unique, seed-stable representative.  What remains is
    *local-optima* multiplicity: EM may converge to genuinely different
    likelihoods across seeds, so prefer the *predictions* (the fitted
    log-rate / firing rate ``exp(C x)``) over a single fit's raw
    ``A`` / ``C``.  Bit-exact parity with MATLAB ``PP_EM`` would
    additionally require multi-restart model selection.

    Notes
    -----
    The marginal-log-likelihood trace is the time-varying-R Gaussian
    smoother surrogate, **not** the true Poisson marginal likelihood;
    it is not guaranteed monotonic.  The E-step uses an IRLS
    pseudo-observation linearization + a time-varying-R RTS smoother
    that exposes the lag-one cross-covariances the A/Q M-step needs.
    """
    import warnings

    _require_dynamax()
    warnings.warn(
        "fit_point_process_em is experimental: A/C are returned in a "
        "canonical PLDS gauge (the scale/rotation freedom is pinned), but "
        "EM may still reach different local optima across seeds — prefer "
        "the fitted rates exp(C x) for interpretation.  See the function "
        "docstring and docs/extras/em_dynamax.md.",
        UserWarning,
        stacklevel=2,
    )

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
    prev_means: np.ndarray | None = None

    for _it in range(int(n_iter)):
        # E-step via IRLS pseudo-obs + lgssm_smoother → exposes the
        # lag-one cross-covariances the A/Q M-step needs (the CMGF
        # smoother does not).  Without them A collapses toward zero and
        # the C Newton step diverges (see _ppem_m_step_closed_form).
        means, covs, cross_covs, ll = _ppem_e_step_lgssm(
            observations, A, C, Q, x0, P0, prev_means
        )
        prev_means = means
        lls.append(ll)
        A, Q, x0, P0 = _ppem_m_step_closed_form(means, covs, cross_covs)
        C = _ppem_newton_C(observations, means, covs, C, n_newton=n_newton_iter)

        # In-loop: cheap diagonal scale-pin only.  A full GL(d) gauge
        # transform every iteration reshapes the optimization landscape
        # each step and fights the Newton trust-region (empirically: NaN /
        # |ΔC|~460 across seeds).  Pinning just the d scale DOF keeps |C|
        # finite without disturbing the rotation the optimizer is settling.
        s = _canonical_scale(means, covs)
        A, C, Q, x0, P0 = _apply_canonical_scale(A, C, Q, x0, P0, s)

    # Post-convergence: pin the FULL PLDS gauge once (whiten + SVD-rotate +
    # sign-fix) using fresh posteriors under the final parameters, so the
    # reported A/C are a unique, seed-stable canonical representative.
    means, covs, _, _ = _ppem_e_step_lgssm(
        observations, A, C, Q, x0, P0, prev_means
    )
    A, (C,), Q, x0, P0 = _canonicalize_gauge(A, [C], Q, x0, P0, means, covs)

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
    stack with the real Gaussian channel, and run the time-varying-R
    RTS smoother on the augmented LG model.

    The Poisson channel's pseudo-observation variance is ``1/λ_t``
    (time-varying); the Gaussian channel's is the fixed ``R``.  Using a
    per-timestep stacked ``R_t`` preserves the IRLS weight cancellation
    (a fixed-R substitution is numerically unstable at low rates).

    Returns ``(smoothed_means, smoothed_covariances,
    smoothed_cross_covariances, marginal_loglik)``.  The lag-one
    cross-covariances are needed for an unbiased A/Q M-step.
    """
    if prev_smoothed_means is None:
        bootstrap = cmgf_poisson_smoother(poisson_obs, A, C_p, Q, x0, P0)
        prev_smoothed_means = bootstrap.state_means

    y_tilde_p, lam_safe = _hybrid_pseudo_observations(poisson_obs, C_p, prev_smoothed_means)

    C_stacked = np.vstack([C_p, C_g])
    y_stacked = np.hstack([y_tilde_p, gaussian_obs])

    # Per-timestep stacked observation noise: Poisson channel 1/λ_t
    # (time-varying), Gaussian channel fixed R.
    T = y_stacked.shape[0]
    p_dim = C_p.shape[0]
    g_dim = C_g.shape[0]
    m = p_dim + g_dim
    R_t = np.zeros((T, m, m))
    inv_lam = 1.0 / lam_safe
    for t in range(T):
        R_t[t, :p_dim, :p_dim] = np.diag(inv_lam[t])
        R_t[t, p_dim:, p_dim:] = R

    means, covs, cross, ll = _kalman_rts_smoother_tv(
        y_stacked, A, C_stacked, Q, R_t, x0, P0
    )
    return means, covs, cross, ll


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
         for C_g; closed-form residual cov for R (with the
         latent-uncertainty trace correction).
       - **Gauge pin (in-loop)**: a cheap diagonal scale normalization
         keeps ``|C_p|`` / ``|C_g|`` finite without reshaping the
         optimization landscape each step.
    3. **After convergence**: pin the full PLDS gauge once (whiten +
       SVD-rotate + sign-fix on the stacked ``[C_p; C_g]``) so both
       emission channels share a unique, seed-stable latent frame
       (see :func:`_canonicalize_gauge`).

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

    Warnings
    --------
    **Experimental.**  Like :func:`fit_point_process_em`, the shared
    latent is gauge-free up to :math:`GL(d)`; this fit pins it to a
    canonical form (computed from the stacked ``[C_p; C_g]``), so the
    returned loadings are seed-stable up to *local-optima* multiplicity.
    The identifiable, observation-space outputs — the Gaussian noise
    ``R`` and the fitted Poisson rates — are the most reliable; treat a
    single fit's raw ``C_p`` / ``C_g`` with the same care.

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
    import warnings

    _require_dynamax()
    warnings.warn(
        "fit_hybrid_em is experimental: the shared-latent loadings "
        "(C_p, C_g) are returned in a canonical PLDS gauge (computed from "
        "the stacked [C_p; C_g]), but EM may still reach different local "
        "optima across seeds.  The Gaussian noise R and the fitted rates "
        "are the most reliable outputs.  See the docstring and "
        "docs/extras/em_dynamax.md.",
        UserWarning,
        stacklevel=2,
    )

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
        smoothed_means, smoothed_covs, smoothed_cross_covs, ll = _hybrid_e_step(
            poisson_obs, gaussian_obs,
            A, C_p, C_g, Q, R, x0, P0,
            prev_smoothed_means,
        )
        lls.append(ll)
        prev_smoothed_means = smoothed_means

        # Pass the lag-one cross-covariances so A/Q don't collapse.
        A, Q, x0, P0 = _ppem_m_step_closed_form(
            smoothed_means, smoothed_covs, smoothed_cross_covs
        )

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

        # Gaussian noise covariance.  The EM M-step for R must include
        # the latent-uncertainty trace term, NOT just the mean residual:
        #   R = (1/T) Σ_t E[(y_t - C_g x_t)(y_t - C_g x_t)']
        #     = (1/T) Σ_t [ (y_t - C_g μ_t)(y_t - C_g μ_t)' + C_g Σ_t C_g' ]
        # Omitting the C_g Σ_t C_g' term systematically underestimates R
        # (it would converge toward zero as EM proceeds, making the
        # smoother over-trust the Gaussian channel and distort the shared
        # latent trajectory).
        resid = gaussian_obs - smoothed_means @ C_g.T
        n_samples = max(poisson_obs.shape[0], 1)
        mean_resid_cov = (resid.T @ resid) / n_samples
        # Σ_t C_g Σ_t C_g'  (summed latent-uncertainty contribution)
        trace_correction = np.einsum(
            "ij,tjk,lk->il", C_g, smoothed_covs, C_g
        ) / n_samples
        R = mean_resid_cov + trace_correction
        R = 0.5 * (R + R.T) + 1e-8 * np.eye(R.shape[0])

        # In-loop: cheap diagonal scale-pin only (see fit_point_process_em
        # — a full per-iteration GL(d) transform destabilizes the EM).
        s = _canonical_scale(smoothed_means, smoothed_covs)
        A, C_p, Q, x0, P0 = _apply_canonical_scale(A, C_p, Q, x0, P0, s)
        C_g = C_g @ np.diag(s)

    # Post-convergence: pin the FULL PLDS gauge once using fresh posteriors
    # under the final parameters.  The canonical rotation is computed from
    # the stacked [C_p; C_g] so both emission channels share a consistent,
    # seed-stable latent frame.
    smoothed_means, smoothed_covs, _, _ = _hybrid_e_step(
        poisson_obs, gaussian_obs, A, C_p, C_g, Q, R, x0, P0, prev_smoothed_means
    )
    A, (C_p, C_g), Q, x0, P0 = _canonicalize_gauge(
        A, [C_p, C_g], Q, x0, P0, smoothed_means, smoothed_covs
    )

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


# ----------------------------------------------------------------------
# Held-out predictive log-likelihood (a true convergence / quality metric)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PredictiveLogLik:
    """True one-step-ahead predictive log-likelihood of a state-space fit.

    Unlike the ``marginal_log_likelihoods`` trace reported by the EM
    trainers — a Gaussian-smoother *surrogate* that re-linearizes each
    iteration and is **not** a valid objective — this is the genuine
    predictive log-likelihood of the *observations* under the fitted
    parameters.  At each step the latent state is predicted causally
    from the past only (one-step-ahead), and the true Poisson (and, for
    the hybrid, Gaussian) likelihood of the actual observation is scored
    under that predictive distribution.

    Use it to (a) confirm EM actually improved the fit, (b) compare
    models / EM restarts on equal footing, and (c) score held-out data
    (pass a test segment together with the train-fitted parameters).

    Attributes
    ----------
    total
        Summed predictive log-likelihood over all time steps and
        channels (nats).  Higher is better.
    per_timestep
        Per-time-step predictive log-likelihood, shape ``(T,)`` (summed
        across neurons / channels).
    poisson
        Poisson-channel contribution to ``total``.
    gaussian
        Gaussian-channel contribution (``None`` for the point-process
        case).
    """

    total: float
    per_timestep: np.ndarray
    poisson: float
    gaussian: float | None = None


def _gauss_hermite(n: int):
    """Probabilists' Gauss-Hermite rule for ``E_{N(0,1)}[f]``.

    ``np.polynomial.hermite_e.hermegauss`` integrates against the weight
    :math:`e^{-x^2/2}` with :math:`\\sum_i w_i = \\sqrt{2\\pi}`, so
    :math:`E_{\\mathcal N(0,1)}[f] = (2\\pi)^{-1/2} \\sum_i w_i f(x_i)`.
    Returns the nodes and the *log* of the normalized weights.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n))
    log_w = np.log(weights) - 0.5 * np.log(2.0 * np.pi)
    return nodes, log_w


def _poisson_marginal_logpmf(y, m, v, nodes, log_w):
    """Per-neuron marginal Poisson log-likelihood under a Gaussian log-rate.

    For each ``(t, i)`` the log-rate :math:`\\eta = C_i x_t` is Gaussian
    with mean ``m[t,i]`` and variance ``v[t,i]`` under the predictive
    state; this returns
    :math:`\\log E_\\eta[\\mathrm{Poisson}(y \\mid e^\\eta)]` evaluated by
    Gauss-Hermite quadrature.  Neurons are treated as conditionally
    independent given their own marginal log-rate (the standard
    mean-field / per-neuron predictive likelihood; the exact joint would
    require a ``state_dim``-dimensional integral).
    """
    from scipy.special import gammaln, logsumexp

    sd = np.sqrt(np.maximum(v, 0.0))
    eta = m[..., None] + sd[..., None] * nodes          # (T, P, K)
    eta = np.clip(eta, -20.0, 20.0)
    log_terms = log_w + y[..., None] * eta - np.exp(eta)   # (T, P, K)
    return logsumexp(log_terms, axis=-1) - gammaln(y + 1.0)


def _gaussian_predictive_logpdf(yg, pred_means, pred_covs, C_g, R):
    """Exact multivariate-normal predictive log-density of the Gaussian channel.

    ``y^{(g)}_t | y_{1:t-1} ~ N(C_g μ⁻_t, C_g P⁻_t C_g' + R)`` where
    ``(μ⁻_t, P⁻_t)`` is the one-step-ahead predictive state.
    """
    T = yg.shape[0]
    out = np.empty(T)
    mean = pred_means @ C_g.T
    for t in range(T):
        S = C_g @ pred_covs[t] @ C_g.T + R
        diff = yg[t] - mean[t]
        _sign, logdet = np.linalg.slogdet(2.0 * np.pi * S)
        out[t] = -0.5 * (diff @ np.linalg.solve(S, diff) + logdet)
    return out


def _predictive_forward_filter(
    A, Q, x0, P0, C_p, poisson_obs,
    C_g=None, R=None, gaussian_obs=None, n_inner: int = 5,
):
    """Causal forward filter returning one-step-ahead predictive state moments.

    Pure-NumPy iterated-EKF filter (no JAX): each step predicts the state
    from the past, records the *predictive* ``(μ⁻, P⁻)`` (the moments the
    predictive likelihood is scored against), then updates with the
    current observation to propagate forward.  The Gaussian channel uses
    an exact Kalman update; the Poisson channel an iterated-EKF
    (Gauss-Newton / IRLS) update.
    """
    T, d = poisson_obs.shape[0], A.shape[0]
    pred_means = np.empty((T, d))
    pred_covs = np.empty((T, d, d))
    eye = np.eye(d)
    mu = P = None  # filtered posterior at t-1
    for t in range(T):
        if t == 0:
            mu_pred = x0.copy()
            P_pred = P0.copy()
        else:
            mu_pred = A @ mu
            P_pred = A @ P @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        pred_means[t] = mu_pred
        pred_covs[t] = P_pred

        # ---- causal update with y_t (so t+1's prediction sees it) ----
        mu_u, P_u = mu_pred.copy(), P_pred.copy()
        if C_g is not None:                       # exact Gaussian Kalman update
            S = C_g @ P_u @ C_g.T + R
            K = P_u @ C_g.T @ np.linalg.inv(S)
            mu_u = mu_u + K @ (gaussian_obs[t] - C_g @ mu_u)
            P_u = (eye - K @ C_g) @ P_u
            P_u = 0.5 * (P_u + P_u.T)
        # iterated-EKF Poisson update from the (post-Gaussian) prior (mu_u, P_u)
        mu_i = mu_u.copy()
        P_post = P_u
        for _ in range(int(n_inner)):
            lam = np.maximum(np.exp(np.clip(C_p @ mu_i, -20.0, 20.0)), 1e-6)
            z = C_p @ mu_i + (poisson_obs[t] - lam) / lam
            S = C_p @ P_u @ C_p.T + np.diag(1.0 / lam)
            K = P_u @ C_p.T @ np.linalg.inv(S)
            mu_i = mu_u + K @ (z - C_p @ mu_u)
            P_post = (eye - K @ C_p) @ P_u
        mu, P = mu_i, 0.5 * (P_post + P_post.T)
    return pred_means, pred_covs


def point_process_predictive_ll(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    transition_covariance: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
    *,
    n_quad: int = 15,
) -> PredictiveLogLik:
    """True one-step-ahead predictive log-likelihood of a Poisson-LGSSM.

    The honest convergence / quality diagnostic for
    :func:`fit_point_process_em` — replaces the surrogate
    Gaussian-smoother trace.  Runs a causal forward filter and scores the
    actual spike counts under the one-step-ahead predictive state via
    Gauss-Hermite quadrature of the Poisson likelihood (integrating over
    the latent uncertainty, not just plugging in the mean rate).

    Pure NumPy — does **not** require dynamax.  Pass a held-out segment
    plus the train-fitted parameters for a proper held-out score.

    Parameters
    ----------
    observations
        Spike counts, shape ``(T, emission_dim)`` (1-D is reshaped).
    transition_matrix, observation_matrix, transition_covariance
        Fitted :math:`A`, :math:`C`, :math:`Q`.
    initial_state_mean, initial_state_covariance
        Fitted :math:`\\hat x_0`, :math:`P_0`.
    n_quad
        Gauss-Hermite nodes for the Poisson marginal (default 15;
        increase for sharper accuracy at high firing rates).

    Returns
    -------
    PredictiveLogLik
    """
    obs = np.asarray(observations, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    A = np.asarray(transition_matrix, dtype=float)
    C = np.atleast_2d(np.asarray(observation_matrix, dtype=float))
    Q = np.asarray(transition_covariance, dtype=float)
    x0 = np.asarray(initial_state_mean, dtype=float).ravel()
    P0 = np.asarray(initial_state_covariance, dtype=float)

    pred_means, pred_covs = _predictive_forward_filter(A, Q, x0, P0, C, obs)
    m = pred_means @ C.T
    v = np.einsum("ij,tjk,ik->ti", C, pred_covs, C)
    nodes, log_w = _gauss_hermite(n_quad)
    logp = _poisson_marginal_logpmf(obs, m, v, nodes, log_w)
    per_t = logp.sum(axis=1)
    total = float(per_t.sum())
    return PredictiveLogLik(total=total, per_timestep=per_t, poisson=total, gaussian=None)


def hybrid_predictive_ll(
    poisson_observations: np.ndarray,
    gaussian_observations: np.ndarray,
    transition_matrix: np.ndarray,
    poisson_observation_matrix: np.ndarray,
    gaussian_observation_matrix: np.ndarray,
    transition_covariance: np.ndarray,
    gaussian_observation_covariance: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
    *,
    n_quad: int = 15,
) -> PredictiveLogLik:
    """True one-step-ahead predictive log-likelihood of a hybrid SSM.

    The :func:`fit_hybrid_em` counterpart of
    :func:`point_process_predictive_ll`.  Both channels share the causal
    forward filter; the Poisson channel is scored by Gauss-Hermite
    quadrature, the Gaussian channel by its exact multivariate-normal
    predictive density.  ``total = poisson + gaussian``.

    Pure NumPy — does **not** require dynamax.

    Parameters
    ----------
    poisson_observations, gaussian_observations
        Spike counts ``(T, p_dim)`` and continuous signal ``(T, g_dim)``.
    transition_matrix, poisson_observation_matrix,
    gaussian_observation_matrix, transition_covariance,
    gaussian_observation_covariance, initial_state_mean,
    initial_state_covariance
        Fitted :math:`A`, :math:`C_p`, :math:`C_g`, :math:`Q`, :math:`R`,
        :math:`\\hat x_0`, :math:`P_0`.
    n_quad
        Gauss-Hermite nodes for the Poisson marginal (default 15).

    Returns
    -------
    PredictiveLogLik
    """
    yp = np.asarray(poisson_observations, dtype=float)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    yg = np.asarray(gaussian_observations, dtype=float)
    if yg.ndim == 1:
        yg = yg.reshape(-1, 1)
    A = np.asarray(transition_matrix, dtype=float)
    C_p = np.atleast_2d(np.asarray(poisson_observation_matrix, dtype=float))
    C_g = np.atleast_2d(np.asarray(gaussian_observation_matrix, dtype=float))
    Q = np.asarray(transition_covariance, dtype=float)
    R = np.atleast_2d(np.asarray(gaussian_observation_covariance, dtype=float))
    x0 = np.asarray(initial_state_mean, dtype=float).ravel()
    P0 = np.asarray(initial_state_covariance, dtype=float)

    pred_means, pred_covs = _predictive_forward_filter(
        A, Q, x0, P0, C_p, yp, C_g=C_g, R=R, gaussian_obs=yg
    )
    m = pred_means @ C_p.T
    v = np.einsum("ij,tjk,ik->ti", C_p, pred_covs, C_p)
    nodes, log_w = _gauss_hermite(n_quad)
    logp_p = _poisson_marginal_logpmf(yp, m, v, nodes, log_w).sum(axis=1)
    logp_g = _gaussian_predictive_logpdf(yg, pred_means, pred_covs, C_g, R)
    per_t = logp_p + logp_g
    total = float(per_t.sum())
    return PredictiveLogLik(
        total=total,
        per_timestep=per_t,
        poisson=float(logp_p.sum()),
        gaussian=float(logp_g.sum()),
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
    "PredictiveLogLik",
    "point_process_predictive_ll",
    "hybrid_predictive_ll",
]
