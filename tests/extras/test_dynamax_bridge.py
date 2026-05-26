"""Tests for ``nstat.extras.em.dynamax_bridge``.

Dynamax is the JAX-backed state-space modeling library from probml.
This bridge is the foundation for closing the ``KF_EM`` / ``PP_EM`` /
``mPPCO_EM`` gap in AUDIT_REPORT.md §3.2 without porting 7,500 LOC of
MATLAB EM code.

The functional test is gated on Dynamax being installed.  The CI
``extras-functional`` job (added in PR-D) runs it for real.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_dynamax_bridge_emits_install_hint_when_missing() -> None:
    """When Dynamax is absent, the bridge raises a clear ImportError
    naming the pip-install hint."""
    try:
        import dynamax  # noqa: F401
        pytest.skip("dynamax is installed; import-error path unreachable")
    except ImportError:
        pass

    from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em

    n_required = sum(
        1
        for p in inspect.signature(fit_linear_gaussian_em).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    with pytest.raises(ImportError) as excinfo:
        fit_linear_gaussian_em(*[None] * n_required)
    assert "pip install nstat-toolbox[" in str(excinfo.value)


def test_fit_linear_gaussian_em_runs_on_synthetic_data() -> None:
    """Smoke test: simulate a linear-Gaussian process, fit via EM,
    verify the result has the expected shapes and log-likelihood
    monotonically (or near-monotonically) improves."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em

    rng = np.random.default_rng(0)
    T, state_dim, emission_dim = 200, 2, 2
    A_true = np.eye(state_dim) * 0.9
    C_true = np.eye(emission_dim)
    Q_true = np.eye(state_dim) * 0.02
    R_true = np.eye(emission_dim) * 0.1

    x = np.zeros((T, state_dim))
    y = np.zeros((T, emission_dim))
    x[0] = rng.multivariate_normal(np.zeros(state_dim), np.eye(state_dim))
    y[0] = C_true @ x[0] + rng.multivariate_normal(np.zeros(emission_dim), R_true)
    for t in range(1, T):
        x[t] = A_true @ x[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q_true)
        y[t] = C_true @ x[t] + rng.multivariate_normal(np.zeros(emission_dim), R_true)

    result = fit_linear_gaussian_em(y, state_dim=state_dim, n_iter=20, seed=0)

    # Shape contract
    assert result.transition_matrix.shape == (state_dim, state_dim)
    assert result.observation_matrix.shape == (emission_dim, state_dim)
    assert result.transition_covariance.shape == (state_dim, state_dim)
    assert result.observation_covariance.shape == (emission_dim, emission_dim)
    assert result.initial_state_mean.shape == (state_dim,)
    assert result.initial_state_covariance.shape == (state_dim, state_dim)
    assert result.log_likelihoods.shape == (result.n_iter,)
    assert result.n_iter == 20

    # EM log-likelihood is theoretically non-decreasing.  Allow a small
    # tolerance for floating-point noise.
    diffs = np.diff(result.log_likelihoods)
    assert np.all(diffs >= -1e-6), (
        f"EM log-likelihood not monotonic: min increment = {diffs.min():.3e}"
    )

    # Final LL should be finite and better than the initial.
    assert np.isfinite(result.log_likelihoods[-1])
    assert result.log_likelihoods[-1] > result.log_likelihoods[0]


def test_fit_linear_gaussian_em_rejects_invalid_state_dim() -> None:
    """Input validation: state_dim must be >= 1."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em

    with pytest.raises(ValueError, match="state_dim must be >= 1"):
        fit_linear_gaussian_em(np.zeros((10, 2)), state_dim=0, n_iter=5)


def test_fit_linear_gaussian_em_handles_1d_observations() -> None:
    """1D observation arrays are reshaped to (T, 1) automatically."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em

    rng = np.random.default_rng(1)
    T = 100
    y = rng.standard_normal(T)  # shape (T,) not (T, 1)
    result = fit_linear_gaussian_em(y, state_dim=1, n_iter=5)
    assert result.observation_matrix.shape == (1, 1)


def test_fit_linear_gaussian_em_rejects_3d_observations() -> None:
    """Higher-rank observation arrays are rejected with a clear message."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em

    with pytest.raises(ValueError, match=r"shape \(T, emission_dim\)"):
        fit_linear_gaussian_em(np.zeros((10, 2, 3)), state_dim=2, n_iter=5)


# ----------------------------------------------------------------------
# CMGF Poisson filter / smoother (PP-state-space inference)
# ----------------------------------------------------------------------


def _simulate_poisson_lgssm(
    T: int = 100, state_dim: int = 2, emission_dim: int = 2, rng_seed: int = 0
):
    """Synthetic Poisson-LGSSM fixture: linear-Gaussian state, Poisson obs."""
    rng = np.random.default_rng(rng_seed)
    A = np.eye(state_dim) * 0.95
    C = np.eye(emission_dim, state_dim) * 0.3  # small loadings → moderate rates
    Q = np.eye(state_dim) * 0.05
    x0 = np.zeros(state_dim)
    P0 = np.eye(state_dim) * 0.1

    x = np.zeros((T, state_dim))
    y = np.zeros((T, emission_dim), dtype=int)
    x[0] = rng.multivariate_normal(x0, P0)
    y[0] = rng.poisson(np.exp(C @ x[0]))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q)
        y[t] = rng.poisson(np.exp(C @ x[t]))
    return y, A, C, Q, x0, P0, x


def test_cmgf_poisson_filter_smoke() -> None:
    """CMGF Poisson filter runs end-to-end with expected output shapes."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import cmgf_poisson_filter

    y, A, C, Q, x0, P0, _ = _simulate_poisson_lgssm()
    result = cmgf_poisson_filter(y, A, C, Q, x0, P0)
    assert result.state_means.shape == (100, 2)
    assert result.state_covariances.shape == (100, 2, 2)
    assert np.isfinite(result.marginal_log_likelihood)


def test_cmgf_poisson_smoother_reduces_posterior_variance() -> None:
    """Smoothed posterior variance must be <= filtered posterior variance
    (universal property of Gaussian smoothers — backward pass only adds
    information).
    """
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import (
        cmgf_poisson_filter, cmgf_poisson_smoother,
    )

    y, A, C, Q, x0, P0, _ = _simulate_poisson_lgssm()
    filt = cmgf_poisson_filter(y, A, C, Q, x0, P0)
    smooth = cmgf_poisson_smoother(y, A, C, Q, x0, P0)

    # Compare diagonal entries (per-component variance) at every t.
    filt_var = np.diagonal(filt.state_covariances, axis1=1, axis2=2)
    smooth_var = np.diagonal(smooth.state_covariances, axis1=1, axis2=2)

    # Smoothed variance should be <= filtered variance everywhere
    # (modulo numerical tolerance from the Gaussian approximation).
    assert np.all(smooth_var <= filt_var + 1e-8), (
        "Smoothed posterior variance exceeded filtered — algorithmic bug."
    )


def test_cmgf_poisson_filter_recovers_latent_state_qualitatively() -> None:
    """On a well-conditioned fixture, the filtered mean should track the
    true latent state with a much smaller squared error than the naive
    zero predictor.
    """
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import cmgf_poisson_filter

    y, A, C, Q, x0, P0, x_true = _simulate_poisson_lgssm(T=200)
    filt = cmgf_poisson_filter(y, A, C, Q, x0, P0)

    mse_filtered = float(np.mean((filt.state_means - x_true) ** 2))
    mse_naive = float(np.mean(x_true ** 2))

    # Filter MSE should be substantially below the naive (zero-prediction)
    # MSE.  Factor of 2 is loose enough to absorb the CMGF Gaussian
    # approximation error.
    assert mse_filtered < 0.5 * mse_naive, (
        f"CMGF Poisson filter not tracking latent state: "
        f"mse_filt={mse_filtered:.3e}, mse_naive={mse_naive:.3e}"
    )


def test_cmgf_poisson_filter_handles_1d_observations() -> None:
    """Single-channel observations reshape to (T, 1) automatically."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import cmgf_poisson_filter

    rng = np.random.default_rng(2)
    T = 80
    A = np.array([[0.95]])
    C = np.array([[0.5]])
    Q = np.array([[0.05]])
    x0 = np.zeros(1)
    P0 = np.array([[0.1]])

    x = np.zeros(T)
    y = np.zeros(T, dtype=int)
    x[0] = rng.normal()
    y[0] = rng.poisson(np.exp(C[0, 0] * x[0]))
    for t in range(1, T):
        x[t] = A[0, 0] * x[t - 1] + rng.normal(scale=Q[0, 0] ** 0.5)
        y[t] = rng.poisson(np.exp(C[0, 0] * x[t]))

    result = cmgf_poisson_filter(y, A, C, Q, x0, P0)  # y is 1D
    assert result.state_means.shape == (T, 1)


def test_cmgf_poisson_filter_emits_install_hint_when_dynamax_missing() -> None:
    """Same actionable-ImportError contract as the rest of the bridge."""
    try:
        import dynamax  # noqa: F401
        pytest.skip("dynamax is installed; import-error path unreachable")
    except ImportError:
        pass

    from nstat.extras.em.dynamax_bridge import cmgf_poisson_filter
    with pytest.raises(ImportError) as excinfo:
        cmgf_poisson_filter(None, None, None, None, None, None)
    assert "pip install nstat-toolbox[" in str(excinfo.value)


# ----------------------------------------------------------------------
# PP_EM (Poisson-LGSSM expectation-maximization)
# ----------------------------------------------------------------------


def test_fit_point_process_em_smoke_and_shape_contract() -> None:
    """PP_EM runs end-to-end on synthetic Poisson-LGSSM data; returned
    parameters have the expected shapes."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_point_process_em

    y, *_ = _simulate_poisson_lgssm(T=150, state_dim=2, emission_dim=2, rng_seed=0)
    result = fit_point_process_em(y, state_dim=2, n_iter=10, seed=0)

    assert result.transition_matrix.shape == (2, 2)
    assert result.observation_matrix.shape == (2, 2)
    assert result.transition_covariance.shape == (2, 2)
    assert result.initial_state_mean.shape == (2,)
    assert result.initial_state_covariance.shape == (2, 2)
    assert result.marginal_log_likelihoods.shape == (10,)
    assert result.n_iter == 10
    assert np.all(np.isfinite(result.marginal_log_likelihoods))


def test_fit_point_process_em_likelihood_mostly_increases() -> None:
    """Laplace-EM trace not guaranteed monotonic (the approximation
    changes each iter), but final must exceed initial and majority of
    steps non-decreasing.  Catches divergence / wild oscillation while
    tolerating Laplace noise."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_point_process_em

    y, *_ = _simulate_poisson_lgssm(T=200, state_dim=2, emission_dim=2, rng_seed=1)
    result = fit_point_process_em(y, state_dim=2, n_iter=20, seed=0)
    lls = result.marginal_log_likelihoods

    assert lls[-1] > lls[0], (
        f"PP_EM made no progress: ll[0]={lls[0]:.3f}, ll[-1]={lls[-1]:.3f}"
    )
    n_descent = int(np.sum(np.diff(lls) < 0))
    assert n_descent < len(lls) // 2, (
        f"PP_EM descended on {n_descent}/{len(lls)-1} iterations"
    )


def test_fit_point_process_em_rejects_invalid_state_dim() -> None:
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_point_process_em
    with pytest.raises(ValueError, match="state_dim must be >= 1"):
        fit_point_process_em(np.zeros((10, 2), dtype=int), state_dim=0, n_iter=3)


def test_fit_point_process_em_handles_1d_observations() -> None:
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_point_process_em
    rng = np.random.default_rng(42)
    y = rng.poisson(np.exp(0.3 * np.ones(80)))
    result = fit_point_process_em(y, state_dim=1, n_iter=5, seed=0)
    assert result.observation_matrix.shape == (1, 1)


def test_fit_point_process_em_emits_install_hint_when_dynamax_missing() -> None:
    try:
        import dynamax  # noqa: F401
        pytest.skip("dynamax is installed; import-error path unreachable")
    except ImportError:
        pass
    from nstat.extras.em.dynamax_bridge import fit_point_process_em
    with pytest.raises(ImportError) as excinfo:
        fit_point_process_em(np.zeros((10, 1), dtype=int), state_dim=1)
    assert "pip install nstat-toolbox[" in str(excinfo.value)


# ----------------------------------------------------------------------
# mPPCO_EM (hybrid Poisson + Gaussian)
# ----------------------------------------------------------------------


def _simulate_hybrid(
    T: int = 150, state_dim: int = 2,
    p_dim: int = 2, g_dim: int = 1, rng_seed: int = 0,
):
    """Synthetic Poisson + Gaussian hybrid fixture."""
    rng = np.random.default_rng(rng_seed)
    A = np.eye(state_dim) * 0.92
    Q = np.eye(state_dim) * 0.03
    C_p = 0.3 * np.eye(p_dim, state_dim)
    C_g = 0.5 * rng.standard_normal((g_dim, state_dim))
    R = 0.05 * np.eye(g_dim)
    x0 = np.zeros(state_dim)
    P0 = np.eye(state_dim) * 0.1

    x = np.zeros((T, state_dim))
    yp = np.zeros((T, p_dim), dtype=int)
    yg = np.zeros((T, g_dim))
    x[0] = rng.multivariate_normal(x0, P0)
    yp[0] = rng.poisson(np.exp(C_p @ x[0]))
    yg[0] = C_g @ x[0] + rng.multivariate_normal(np.zeros(g_dim), R)
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q)
        yp[t] = rng.poisson(np.exp(C_p @ x[t]))
        yg[t] = C_g @ x[t] + rng.multivariate_normal(np.zeros(g_dim), R)
    return yp, yg


def test_fit_hybrid_em_smoke_and_shape_contract() -> None:
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em

    yp, yg = _simulate_hybrid(T=120, state_dim=2, p_dim=2, g_dim=1, rng_seed=0)
    result = fit_hybrid_em(yp, yg, state_dim=2, n_iter=10, seed=0)

    assert result.transition_matrix.shape == (2, 2)
    assert result.poisson_observation_matrix.shape == (2, 2)
    assert result.gaussian_observation_matrix.shape == (1, 2)
    assert result.transition_covariance.shape == (2, 2)
    assert result.gaussian_observation_covariance.shape == (1, 1)
    assert result.initial_state_mean.shape == (2,)
    assert result.initial_state_covariance.shape == (2, 2)
    assert result.marginal_log_likelihoods.shape == (10,)
    assert np.all(np.isfinite(result.marginal_log_likelihoods))


def test_fit_hybrid_em_likelihood_progresses() -> None:
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em

    yp, yg = _simulate_hybrid(T=200, state_dim=2, p_dim=2, g_dim=1, rng_seed=2)
    result = fit_hybrid_em(yp, yg, state_dim=2, n_iter=15, seed=0)
    lls = result.marginal_log_likelihoods
    assert lls[-1] > lls[0], (
        f"hybrid EM made no progress: ll[0]={lls[0]:.3f}, ll[-1]={lls[-1]:.3f}"
    )


def test_fit_hybrid_em_rejects_mismatched_observation_lengths() -> None:
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em
    with pytest.raises(ValueError, match="same T"):
        fit_hybrid_em(np.zeros((100, 2), dtype=int), np.zeros((99, 1)), state_dim=2)


def test_fit_hybrid_em_rejects_invalid_state_dim() -> None:
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em
    with pytest.raises(ValueError, match="state_dim must be >= 1"):
        fit_hybrid_em(
            np.zeros((10, 2), dtype=int), np.zeros((10, 1)),
            state_dim=0, n_iter=3,
        )


def test_fit_hybrid_em_emits_install_hint_when_dynamax_missing() -> None:
    try:
        import dynamax  # noqa: F401
        pytest.skip("dynamax is installed; import-error path unreachable")
    except ImportError:
        pass
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em
    with pytest.raises(ImportError) as excinfo:
        fit_hybrid_em(
            np.zeros((10, 1), dtype=int), np.zeros((10, 1)), state_dim=1
        )
    assert "pip install nstat-toolbox[" in str(excinfo.value)
