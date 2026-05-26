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
