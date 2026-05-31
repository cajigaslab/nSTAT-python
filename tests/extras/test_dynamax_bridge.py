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

    # Use *informative* observation loadings.  With log-link Poisson
    # emissions λ = exp(Cx), small C makes the firing rate nearly
    # constant in x, so spike counts barely constrain the latent state
    # (an observability limit, not an algorithm flaw).  A coupling sweep
    # shows mse_filt/mse_naive ≈ 0.83 at C=0.3 (uninformative) → 0.43 at
    # C=0.9 (clearly tracking).  Use C=0.9 so the test exercises real
    # tracking with margin below the 0.5 threshold.
    rng = np.random.default_rng(0)
    T, state_dim, emission_dim = 400, 2, 2
    A = np.eye(state_dim) * 0.95
    C = np.eye(emission_dim, state_dim) * 0.9
    Q = np.eye(state_dim) * 0.05
    x0 = np.zeros(state_dim)
    P0 = np.eye(state_dim) * 0.1
    x_true = np.zeros((T, state_dim))
    y = np.zeros((T, emission_dim), dtype=int)
    x_true[0] = rng.multivariate_normal(x0, P0)
    y[0] = rng.poisson(np.exp(C @ x_true[0]))
    for t in range(1, T):
        x_true[t] = A @ x_true[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q)
        y[t] = rng.poisson(np.exp(C @ x_true[t]))

    filt = cmgf_poisson_filter(y, A, C, Q, x0, P0)
    mse_filtered = float(np.mean((filt.state_means - x_true) ** 2))
    mse_naive = float(np.mean(x_true ** 2))

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


def test_fit_point_process_em_is_finite_and_bounded() -> None:
    """PP_EM is experimental: the Poisson-LDS latent parameters are not
    uniquely identified (a gauge freedom leaves ``C x`` invariant), so
    we do NOT assert parameter or surrogate-likelihood recovery — those
    are not well-defined targets.  What we *can* assert:

    1. The fit runs end-to-end and returns all-finite parameters
       (no NaN/Inf — the gauge-scale pin + Newton trust-region keep
       things bounded; a regression that removed them would blow up).
    2. The implied firing rate ``exp(C x)`` tracks the observed spike
       counts in aggregate — the *observable* the model actually fits.

    The raw |C| can still vary across seeds (the rotational gauge is
    not pinned); this is documented in the function's Warnings section.
    """
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import (
        fit_point_process_em, cmgf_poisson_smoother,
    )

    # Informative loadings so spikes constrain the latent (see the CMGF
    # observability discussion).
    rng = np.random.default_rng(0)
    T, sd = 300, 2
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    C = np.eye(2, sd) * 0.6
    Q = np.eye(sd) * 0.05
    x0 = np.zeros(sd)
    P0 = np.eye(sd) * 0.1
    x = np.zeros((T, sd))
    y = np.zeros((T, 2), dtype=int)
    x[0] = rng.multivariate_normal(x0, P0)
    y[0] = rng.poisson(np.exp(C @ x[0]))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(sd), Q)
        y[t] = rng.poisson(np.exp(C @ x[t]))

    result = fit_point_process_em(y, state_dim=2, n_iter=25, seed=0)

    # (1) all parameters finite + correctly shaped.
    for arr in (result.transition_matrix, result.observation_matrix,
                result.transition_covariance, result.initial_state_mean,
                result.initial_state_covariance, result.marginal_log_likelihoods):
        assert np.all(np.isfinite(arr)), "PP_EM produced non-finite output"

    # (2) implied firing rate tracks observed spike counts (the
    #     identifiable observable).  Re-smooth at the fitted params and
    #     compare aggregate rate to the empirical mean — loose tolerance
    #     because this is an experimental estimator.
    sm = cmgf_poisson_smoother(
        y, result.transition_matrix, result.observation_matrix,
        result.transition_covariance, result.initial_state_mean,
        result.initial_state_covariance,
    )
    implied_rate = np.exp(sm.state_means @ result.observation_matrix.T)
    rel_err = np.abs(implied_rate.mean(axis=0) - y.mean(axis=0)) / y.mean(axis=0)
    assert np.all(rel_err < 0.6), (
        f"PP_EM implied rate far from observed: rel_err={rel_err}"
    )


def test_fit_point_process_em_gauge_is_canonical() -> None:
    """Tier 0.1 identifiability: after EM, the PLDS gauge is pinned once
    to a canonical form (whiten + SVD-rotate + sign-fix), removing the
    full ``GL(d)`` reparameterization freedom.  We assert the *structural*
    invariants of that canonical form — these hold regardless of which
    local optimum EM lands in, and are the identifiable counterpart to
    the (ill-posed) "recover C_true" target:

    1. ``CᵀC`` is diagonal with non-increasing diagonal — the emission
       columns are orthogonal and ordered by descending norm (the
       SVD-rotation guarantee; off-diagonals vanish to machine eps).
    2. The largest-magnitude entry of each ``C`` column is positive (the
       deterministic sign convention).
    3. ``|C|`` stays bounded across init seeds.  A prior regression
       applied the full gauge transform *per iteration* instead of once
       after convergence; it fought the Newton trust-region and blew up
       to ``|C|~10²`` with NaNs.  This is the guard against that.
    """
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_point_process_em

    y, *_ = _simulate_poisson_lgssm(T=300, state_dim=2, emission_dim=2, rng_seed=1)

    c_norms = []
    for seed in range(3):
        C = fit_point_process_em(
            y, state_dim=2, n_iter=20, seed=seed
        ).observation_matrix
        assert np.all(np.isfinite(C)), "PP_EM produced non-finite C"
        c_norms.append(float(np.abs(C).max()))

        gram = C.T @ C
        offdiag = gram - np.diag(np.diag(gram))
        assert np.allclose(offdiag, 0.0, atol=1e-8), (
            f"canonical emission columns not orthogonal: off-diag max "
            f"{np.abs(offdiag).max():.2e}"
        )
        diag = np.diag(gram)
        assert np.all(np.diff(diag) <= 1e-8), (
            f"canonical singular values not descending: {diag}"
        )
        lead = np.argmax(np.abs(C), axis=0)
        assert np.all(C[lead, np.arange(C.shape[1])] >= 0.0), (
            "canonical sign convention violated (leading entry not positive)"
        )

    assert max(c_norms) < 50.0, (
        f"|C| unbounded across seeds: {c_norms} — the per-iteration "
        f"canonicalization regression is back."
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


def test_fit_hybrid_em_recovers_gaussian_noise() -> None:
    """The mPPCO_EM surrogate marginal-likelihood is NOT a valid
    convergence objective (the IRLS pseudo-observations are
    re-linearized each iteration, so the trace changes basis and is not
    expected to increase).  The *identifiable, observation-space*
    quantity is the Gaussian noise covariance R — which the
    trace-corrected M-step recovers well.  Assert that instead.
    """
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em

    # Fixture with a Gaussian channel whose true noise variance is 0.05.
    rng = np.random.default_rng(0)
    T, sd = 300, 2
    A = np.eye(sd) * 0.92
    Q = np.eye(sd) * 0.03
    C_p = 0.4 * np.eye(2, sd)
    C_g = np.array([[1.0, 0.3]])
    x = np.zeros((T, sd))
    x[0] = rng.multivariate_normal(np.zeros(sd), np.eye(sd))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(sd), Q)
    yp = rng.poisson(np.exp(x @ C_p.T))
    yg = x @ C_g.T + rng.normal(scale=np.sqrt(0.05), size=(T, 1))

    result = fit_hybrid_em(yp, yg, state_dim=2, n_iter=25, seed=0)
    R_hat = float(result.gaussian_observation_covariance.ravel()[0])
    # True R = 0.05.  Recovery within a factor of ~3 (loose — the latent
    # is shared with a noisy Poisson channel and only partially observed).
    assert 0.015 < R_hat < 0.15, (
        f"Gaussian R not recovered: R_hat={R_hat:.4f} (true 0.05); the "
        f"trace correction in the R M-step may be missing."
    )
    assert np.all(np.isfinite(result.poisson_observation_matrix))
    assert np.all(np.isfinite(result.gaussian_observation_matrix))


def test_fit_hybrid_em_gauge_is_canonical() -> None:
    """Tier 0.1 identifiability for the hybrid model.  The canonical
    rotation is computed from the *stacked* ``[C_p; C_g]`` so both
    emission channels share one consistent, seed-stable latent frame.
    Assert the canonical invariants on that stack (cf.
    :func:`test_fit_point_process_em_gauge_is_canonical`):

    1. ``Sᵀ S`` diagonal & descending for ``S = [C_p; C_g]`` (orthogonal,
       ordered columns).
    2. Leading entry of each stacked column positive (sign convention).
    3. Both emission matrices bounded across init seeds (regression guard
       for the per-iteration-canonicalization blow-up).
    """
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em

    yp, yg = _simulate_hybrid(T=300, state_dim=2, p_dim=2, g_dim=1, rng_seed=1)

    norms = []
    for seed in range(3):
        result = fit_hybrid_em(yp, yg, state_dim=2, n_iter=20, seed=seed)
        C_p = result.poisson_observation_matrix
        C_g = result.gaussian_observation_matrix
        assert np.all(np.isfinite(C_p)) and np.all(np.isfinite(C_g))
        norms.append(float(max(np.abs(C_p).max(), np.abs(C_g).max())))

        S = np.vstack([C_p, C_g])
        gram = S.T @ S
        offdiag = gram - np.diag(np.diag(gram))
        assert np.allclose(offdiag, 0.0, atol=1e-8), (
            f"stacked canonical columns not orthogonal: off-diag max "
            f"{np.abs(offdiag).max():.2e}"
        )
        diag = np.diag(gram)
        assert np.all(np.diff(diag) <= 1e-8), (
            f"stacked canonical singular values not descending: {diag}"
        )
        lead = np.argmax(np.abs(S), axis=0)
        assert np.all(S[lead, np.arange(S.shape[1])] >= 0.0), (
            "stacked canonical sign convention violated"
        )

    assert max(norms) < 50.0, (
        f"emission matrices unbounded across seeds: {norms}"
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


# ----------------------------------------------------------------------
# Held-out predictive log-likelihood (Tier 0.2 — a true quality metric)
#
# These exercise the pure-NumPy diagnostic, which does NOT require
# dynamax — so they run in the base unit suite too, no importorskip.
# ----------------------------------------------------------------------


def _sim_pp_with_state(T=400, sd=2, ed=4, c_scale=0.8, seed=3):
    """Poisson-LGSSM fixture returning observations AND the true params."""
    rng = np.random.default_rng(seed)
    A = np.eye(sd) * 0.92
    C = np.zeros((ed, sd))
    for i in range(ed):
        C[i, i % sd] = c_scale
    Q = np.eye(sd) * 0.05
    x0 = np.zeros(sd)
    P0 = np.eye(sd) * 0.1
    x = np.zeros((T, sd))
    y = np.zeros((T, ed), dtype=int)
    x[0] = rng.multivariate_normal(x0, P0)
    y[0] = rng.poisson(np.exp(C @ x[0]))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(sd), Q)
        y[t] = rng.poisson(np.exp(C @ x[t]))
    return y, A, C, Q, x0, P0, x


def test_predictive_ll_finite_and_additive() -> None:
    """The point-process predictive LL is finite, and its per-timestep
    array sums to the reported total (and to the Poisson channel, which
    is the only channel here)."""
    from nstat.extras.em.dynamax_bridge import point_process_predictive_ll

    y, A, C, Q, x0, P0, _ = _sim_pp_with_state()
    r = point_process_predictive_ll(y, A, C, Q, x0, P0)
    assert np.isfinite(r.total)
    assert r.per_timestep.shape == (y.shape[0],)
    assert np.isclose(r.total, r.per_timestep.sum())
    assert np.isclose(r.total, r.poisson)
    assert r.gaussian is None


def test_predictive_ll_ranks_true_above_misspecified() -> None:
    """Core validity: the true generating parameters achieve a higher
    held-out predictive log-likelihood than (a) a near-flat-rate model,
    (b) a homogeneous Poisson at the empirical mean, and (c) a degenerate
    collapsed-dynamics fit (A→0, inflated C/Q — the failure mode the
    diagnostic is built to catch).  This is what makes it a usable
    model-selection / convergence metric."""
    from scipy.special import gammaln

    from nstat.extras.em.dynamax_bridge import point_process_predictive_ll

    y, A, C, Q, x0, P0, _ = _sim_pp_with_state()
    y_tr, y_te = y[:300], y[300:]

    def pll(seg, params):
        return point_process_predictive_ll(seg, *params).total

    true = pll(y_te, (A, C, Q, x0, P0))
    flat = pll(y_te, (A, C * 0.0, Q, x0, P0))
    collapsed = pll(y_te, (A * 0.05, C * 3.0, Q * 20.0, x0, P0))

    lam_bar = np.maximum(y_tr.mean(axis=0), 1e-6)
    homogeneous = float(
        (y_te * np.log(lam_bar) - lam_bar - gammaln(y_te + 1.0)).sum()
    )

    assert true > flat, f"true {true:.1f} !> flat {flat:.1f}"
    assert true > homogeneous, f"true {true:.1f} !> homogeneous {homogeneous:.1f}"
    assert true > collapsed, f"true {true:.1f} !> collapsed {collapsed:.1f}"


def test_predictive_ll_gauss_hermite_converges() -> None:
    """The Gauss-Hermite quadrature of the Poisson marginal is stable:
    a coarse rule (5 nodes) already agrees with a fine one (30 nodes)."""
    from nstat.extras.em.dynamax_bridge import point_process_predictive_ll

    y, A, C, Q, x0, P0, _ = _sim_pp_with_state()
    coarse = point_process_predictive_ll(y, A, C, Q, x0, P0, n_quad=5).total
    fine = point_process_predictive_ll(y, A, C, Q, x0, P0, n_quad=30).total
    assert abs(coarse - fine) < 1.0, f"GH not converged: {coarse:.4f} vs {fine:.4f}"


def test_hybrid_predictive_ll_splits_and_ranks() -> None:
    """The hybrid predictive LL decomposes additively into Poisson +
    Gaussian channels, its per-timestep array sums to the total, and the
    true parameters outscore a misspecified set."""
    from nstat.extras.em.dynamax_bridge import hybrid_predictive_ll

    y, A, C, Q, x0, P0, x = _sim_pp_with_state(ed=2)
    rng = np.random.default_rng(7)
    C_g = np.array([[1.0, 0.0]])
    R = np.array([[0.09]])
    yg = x @ C_g.T + rng.normal(scale=0.3, size=(x.shape[0], 1))

    r = hybrid_predictive_ll(y, yg, A, C, C_g, Q, R, x0, P0)
    assert np.isfinite(r.total)
    assert np.isclose(r.total, r.poisson + r.gaussian)
    assert np.isclose(r.total, r.per_timestep.sum())

    wrong = hybrid_predictive_ll(y, yg, A, C * 0.0, C_g * 0.0, Q, R * 10.0, x0, P0)
    assert r.total > wrong.total


def test_hybrid_gaussian_channel_matches_exact_kalman() -> None:
    """When the Poisson loadings are zero, the Poisson update contributes
    nothing (Kalman gain is 0) and the filter reduces to an exact linear-
    Gaussian Kalman filter.  The Gaussian-channel predictive LL must then
    equal an independent textbook Kalman one-step-ahead predictive
    log-likelihood — a strong correctness check on the forward filter and
    the Gaussian predictive density."""
    from nstat.extras.em.dynamax_bridge import hybrid_predictive_ll

    rng = np.random.default_rng(0)
    T, sd, gd = 200, 2, 1
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    C_g = np.array([[1.0, 0.5]])
    Q = np.eye(sd) * 0.05
    R = np.array([[0.2]])
    x0 = np.zeros(sd)
    P0 = np.eye(sd) * 0.1
    x = np.zeros((T, sd))
    yg = np.zeros((T, gd))
    x[0] = rng.multivariate_normal(x0, P0)
    yg[0] = C_g @ x[0] + rng.multivariate_normal(np.zeros(gd), R)
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(sd), Q)
        yg[t] = C_g @ x[t] + rng.multivariate_normal(np.zeros(gd), R)

    # Zero Poisson channel: counts are arbitrary (gain is 0 regardless).
    C_p = np.zeros((1, sd))
    yp = np.zeros((T, 1), dtype=int)
    r = hybrid_predictive_ll(yp, yg, A, C_p, C_g, Q, R, x0, P0)

    # Independent exact Kalman predictive log-likelihood.
    mu, P = x0.copy(), P0.copy()
    ref = 0.0
    for t in range(T):
        if t > 0:
            mu = A @ mu
            P = A @ P @ A.T + Q
        S = C_g @ P @ C_g.T + R
        diff = yg[t] - C_g @ mu
        _s, logdet = np.linalg.slogdet(2.0 * np.pi * S)
        ref += -0.5 * (diff @ np.linalg.solve(S, diff) + logdet)
        K = P @ C_g.T @ np.linalg.inv(S)
        mu = mu + K @ diff
        P = (np.eye(sd) - K @ C_g) @ P

    assert np.isclose(r.gaussian, ref, rtol=1e-6, atol=1e-6), (
        f"Gaussian predictive LL {r.gaussian:.6f} != exact Kalman {ref:.6f}"
    )


def test_predictive_ll_runs_on_em_output() -> None:
    """Integration smoke: the diagnostic scores real EM output and returns
    finite per-timestep values.  (No improvement assertion — held-out
    gains are observability-dependent; see the docs caveat.)"""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import (
        fit_point_process_em, point_process_predictive_ll,
    )

    y, *_ = _sim_pp_with_state(T=300, ed=4, c_scale=0.9)
    fit = fit_point_process_em(y[:250], state_dim=2, n_iter=15, seed=0)
    r = point_process_predictive_ll(
        y[250:], fit.transition_matrix, fit.observation_matrix,
        fit.transition_covariance, fit.initial_state_mean,
        fit.initial_state_covariance,
    )
    assert np.isfinite(r.total)
    assert r.per_timestep.shape == (50,)
    assert np.all(np.isfinite(r.per_timestep))


# ----------------------------------------------------------------------
# Multi-restart selection (Tier 0.3 — harden PP_EM weak-observability)
# ----------------------------------------------------------------------


def test_fit_point_process_em_best_of_smoke_and_shapes() -> None:
    """Multi-restart end-to-end: runs n_restarts seeds, returns a
    MultiRestartResult whose ``best_*`` fields correspond to the
    argmax of ``all_predictive_lls``."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import (
        MultiRestartResult,
        PointProcessEMResult,
        fit_point_process_em_best_of,
    )

    y, *_ = _sim_pp_with_state(T=300, ed=4, c_scale=0.9)
    result = fit_point_process_em_best_of(
        y, state_dim=2, n_restarts=4, holdout_fraction=0.25,
        n_iter=15, base_seed=0,
    )

    assert isinstance(result, MultiRestartResult)
    assert isinstance(result.best_result, PointProcessEMResult)
    assert result.all_seeds.shape == (4,)
    assert result.all_predictive_lls.shape == (4,)
    assert result.best_seed in result.all_seeds.tolist()
    # best_result + best_predictive_ll correspond to argmax of all LLs
    best_idx = int(np.nanargmax(result.all_predictive_lls))
    assert int(result.all_seeds[best_idx]) == result.best_seed
    assert np.isclose(result.best_predictive_ll, result.all_predictive_lls[best_idx])
    # best LL is >= median (trivially, by construction; guards refactors)
    finite = result.all_predictive_lls[np.isfinite(result.all_predictive_lls)]
    if finite.size >= 2:
        assert result.best_predictive_ll >= float(np.median(finite)) - 1e-6


def test_fit_point_process_em_best_of_input_validation() -> None:
    """``n_restarts`` and ``holdout_fraction`` are validated upfront."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_point_process_em_best_of

    y = np.zeros((100, 2), dtype=int)
    with pytest.raises(ValueError, match="n_restarts must be >= 1"):
        fit_point_process_em_best_of(y, state_dim=2, n_restarts=0)
    with pytest.raises(ValueError, match="holdout_fraction must be in"):
        fit_point_process_em_best_of(y, state_dim=2, holdout_fraction=0.0)
    with pytest.raises(ValueError, match="holdout_fraction must be in"):
        fit_point_process_em_best_of(y, state_dim=2, holdout_fraction=1.0)
    with pytest.raises(ValueError, match="training segment too short"):
        # 5 train + 5 test bins isn't enough to fit PP_EM.
        fit_point_process_em_best_of(
            np.zeros((10, 2), dtype=int), state_dim=2, holdout_fraction=0.5
        )


def test_fit_hybrid_em_best_of_smoke_and_shapes() -> None:
    """Hybrid multi-restart smoke + shape contract."""
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import (
        HybridEMResult,
        MultiRestartResult,
        fit_hybrid_em_best_of,
    )

    yp, yg = _simulate_hybrid(T=240, state_dim=2, p_dim=2, g_dim=1, rng_seed=2)
    result = fit_hybrid_em_best_of(
        yp, yg, state_dim=2, n_restarts=3, holdout_fraction=0.25, n_iter=12,
    )

    assert isinstance(result, MultiRestartResult)
    assert isinstance(result.best_result, HybridEMResult)
    assert result.all_seeds.shape == (3,)
    assert result.all_predictive_lls.shape == (3,)
    finite = result.all_predictive_lls[np.isfinite(result.all_predictive_lls)]
    assert finite.size >= 1, "all hybrid restarts failed to score"


def test_fit_hybrid_em_best_of_rejects_mismatched_lengths() -> None:
    pytest.importorskip("dynamax")
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em_best_of

    with pytest.raises(ValueError, match="same T"):
        fit_hybrid_em_best_of(
            np.zeros((100, 2), dtype=int),
            np.zeros((99, 1)),
            state_dim=2,
        )
