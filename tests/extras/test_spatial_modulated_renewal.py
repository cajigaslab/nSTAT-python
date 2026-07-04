"""Tests for nstat.extras.spatial.modulated_renewal — modulated renewal /
conditional-ISI point-process model.

Synthetic data only (np.random.default_rng); no patient data.  No source
code from any external repository was consulted for this test file.

Contract checks (mirrors the four validation items in the build brief):

1. Poisson special-case collapse — at the exact gamma shape=1 Poisson
   limit, ``fit_modulated_renewal`` collapses onto a plain
   :func:`nstat.glm.fit_poisson_glm` fit of the same binned data.
2. Known-CV recovery — an inhomogeneous gamma / inverse-Gaussian renewal
   train simulated with a KNOWN shape (hence known CV) via
   :func:`simulate_modulated_renewal` is recovered by
   :func:`fit_modulated_renewal` within tolerance.
3. Time-rescaling goodness-of-fit tie — the ``rescaled_isis`` of a
   correctly-fit model, transformed through :func:`renewal_cdf`, are
   ~Uniform(0,1) (a plain KS test), *and* the same fitted CIF's per-bin
   spike probabilities pass the shipped
   :func:`nstat.extras.spatial.marked_gof.marked_time_rescaling`
   discrete-time-rescaling GoF check.
4. Degenerate inputs (<2 spikes, missing/empty covariates) raise clear
   ``ValueError``s.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from nstat.extras.spatial.basis import bspline_basis_1d
from nstat.extras.spatial.marked_gof import marked_time_rescaling
from nstat.extras.spatial.modulated_renewal import (
    ModulatedRenewalResult,
    fit_modulated_renewal,
    renewal_cdf,
    renewal_hazard,
    simulate_modulated_renewal,
)
from nstat.glm import fit_poisson_glm

# A mild sinusoidal covariate rate used across several tests: two harmonics
# on a period long relative to the mean ISI, evaluated on a uniform grid.
_PERIOD = 40.0
_TRUE_BETA = np.array([np.log(2.0), 0.4, -0.3])


def _rate_fn(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.exp(
        _TRUE_BETA[0]
        + _TRUE_BETA[1] * np.sin(2 * np.pi * t / _PERIOD)
        + _TRUE_BETA[2] * np.cos(2 * np.pi * t / _PERIOD)
    )


def _covariate_design(T: float, dt: float) -> np.ndarray:
    n_bins = int(round(T / dt))
    t_grid = (np.arange(n_bins) + 0.5) * dt
    return np.column_stack(
        [np.sin(2 * np.pi * t_grid / _PERIOD), np.cos(2 * np.pi * t_grid / _PERIOD)]
    )


# ----------------------------------------------------------------------
# 1. Poisson special-case collapse
# ----------------------------------------------------------------------


def test_gamma_shape_one_hazard_is_exactly_poisson():
    """Analytic sanity check: gamma shape=1 (Exponential) hazard is
    identically 1 for *any* elapsed time -- the exact Poisson limit."""
    tau = np.array([1e-6, 1e-3, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0])
    r = renewal_hazard(tau, shape_param=1.0, renewal="gamma")
    assert np.allclose(r, 1.0, atol=1e-10)


def test_poisson_collapse_regression_tie_to_fit_poisson_glm():
    """A modulated-renewal fit of a *true* inhomogeneous Poisson train
    (gamma shape=1) collapses onto plain fit_poisson_glm on the same
    binned data -- gamma shape=1 is an EXACT (not asymptotic) Poisson
    equivalent, so beta should tie tightly."""
    dt = 0.02
    T = 800.0
    rng = np.random.default_rng(0)
    spikes = simulate_modulated_renewal(
        _rate_fn, shape_param=1.0, T=T, renewal="gamma", rng=rng, dt=dt
    )
    assert len(spikes) > 500

    covariates = _covariate_design(T, dt)
    result = fit_modulated_renewal(
        spikes, covariates, renewal="gamma", dt=dt, tol=1e-4, n_inner=5, max_iter=50
    )
    assert result.converged

    n_bins = covariates.shape[0]
    bin_edges = np.arange(n_bins + 1) * dt
    y_counts, _ = np.histogram(spikes, bins=bin_edges)
    plain = fit_poisson_glm(
        covariates, y_counts, offset=np.full(n_bins, np.log(dt)), l2=1e-6
    )
    plain_beta = np.concatenate([[plain.intercept], plain.coefficients])

    assert np.allclose(result.beta, plain_beta, atol=5e-3)
    # theta itself should also land near the true (exact) Poisson value.
    assert abs(result.shape_param - 1.0) < 0.2


# ----------------------------------------------------------------------
# 2 & 3. Known-CV recovery + time-rescaling GoF tie (shared simulate+fit)
# ----------------------------------------------------------------------

_TRUE_SHAPE = 5.0  # inverse-Gaussian shape -> CV = 1/sqrt(5) ~= 0.4472
_FIT_DT = 0.01
_FIT_T = 1000.0


@pytest.fixture(scope="module")
def _ig_recovery_fit() -> tuple[np.ndarray, np.ndarray, ModulatedRenewalResult]:
    """Simulate a known inverse-Gaussian modulated-renewal train and fit
    it once; shared by the CV-recovery and GoF-tie tests below."""
    rng = np.random.default_rng(0)
    spikes = simulate_modulated_renewal(
        _rate_fn,
        shape_param=_TRUE_SHAPE,
        T=_FIT_T,
        renewal="inverse_gaussian",
        rng=rng,
        dt=_FIT_DT,
    )
    covariates = _covariate_design(_FIT_T, _FIT_DT)
    result = fit_modulated_renewal(
        spikes,
        covariates,
        renewal="inverse_gaussian",
        dt=_FIT_DT,
        tol=1e-6,
        n_inner=20,
        max_iter=100,
    )
    return spikes, covariates, result


def test_known_cv_recovery_inverse_gaussian(_ig_recovery_fit):
    _, _, result = _ig_recovery_fit
    assert result.converged
    assert result.renewal == "inverse_gaussian"

    true_cv = 1.0 / np.sqrt(_TRUE_SHAPE)
    assert abs(result.cv - true_cv) / true_cv < 0.2
    assert abs(result.shape_param - _TRUE_SHAPE) / _TRUE_SHAPE < 0.25

    # cv is exactly 1/sqrt(shape_param) by construction.
    assert result.cv == pytest.approx(1.0 / np.sqrt(result.shape_param))

    # beta recovery (covariate-driven rate) within a generous tolerance.
    assert np.allclose(result.beta, _TRUE_BETA, atol=0.2)


def test_known_cv_recovery_gamma_family():
    """Same recovery check for the gamma renewal family (constant
    baseline rate, to keep this second check fast and independent of the
    covariate/operational-time interaction exercised above)."""
    true_rate = 3.0
    true_shape = 4.0  # CV = 0.5
    dt = 0.02
    T = 800.0
    rng = np.random.default_rng(0)
    spikes = simulate_modulated_renewal(
        true_rate, true_shape, T=T, renewal="gamma", rng=rng, dt=dt
    )
    n_bins = int(round(T / dt))
    covariates = np.zeros((n_bins, 1))  # constant baseline design
    result = fit_modulated_renewal(
        spikes, covariates, renewal="gamma", dt=dt, tol=1e-5, n_inner=12, max_iter=80
    )
    assert result.converged

    true_cv = 1.0 / np.sqrt(true_shape)
    assert abs(result.cv - true_cv) / true_cv < 0.2
    fitted_rate = float(np.exp(result.beta[0]))
    assert abs(fitted_rate - true_rate) / true_rate < 0.15


def test_time_rescaling_gof_tie_renewal_cdf_is_uniform(_ig_recovery_fit):
    """The load-bearing correctness test: the operational-time ISIs of a
    correctly-fit model, transformed through the renewal CDF, are
    ~Uniform(0,1) (Brown et al. 2002's time-rescaling theorem, generalized
    to a renewal density -- see module docstring)."""
    _, _, result = _ig_recovery_fit
    pit = renewal_cdf(result.rescaled_isis, result.shape_param, renewal="inverse_gaussian")
    assert pit.min() >= 0.0 and pit.max() <= 1.0

    ks = stats.kstest(pit, "uniform")
    assert ks.pvalue > 0.05, f"KS rejects uniformity of the renewal-CDF PIT: {ks}"


def test_time_rescaling_gof_tie_exercises_marked_gof(_ig_recovery_fit):
    """Same correctness property, exercised through the shipped discrete-
    time-rescaling GoF machinery (Haslinger-Pipa-Brown 2010): per-bin spike
    probabilities from the fitted CIF (:meth:`ModulatedRenewalResult.rate_fn`)
    should pass both the naive and discrete-time-corrected KS checks."""
    spikes, covariates, result = _ig_recovery_fit
    n_bins = covariates.shape[0]
    bin_edges = np.arange(n_bins + 1) * _FIT_DT
    bin_centers = bin_edges[:-1] + 0.5 * _FIT_DT

    spike_bins = np.clip(
        np.searchsorted(bin_edges, spikes, side="right") - 1, 0, n_bins - 1
    )
    rate_fn = result.rate_fn()
    lam_at_centers = rate_fn(bin_centers)
    p_k = np.clip(lam_at_centers * _FIT_DT, 1e-9, 1.0 - 1e-9)
    assert 0.0 < p_k.mean() < 0.5  # sanity: a plausible per-bin spike probability

    gof = marked_time_rescaling(spike_bins, None, p_k, rng=np.random.default_rng(4))
    assert gof.inside_corrected, (
        f"discrete-time-corrected KS rejects the fitted model: "
        f"stat={gof.ks_corrected} band={gof.ks_band}"
    )


# ----------------------------------------------------------------------
# basis.py reuse
# ----------------------------------------------------------------------


def test_basis_kwarg_reuses_bspline_basis_1d():
    """``basis=`` accepts a pre-built B-spline design (bspline_basis_1d),
    exactly as basis.py's own docstring promises for fit_poisson_glm."""
    dt = 0.02
    T = 500.0
    rng = np.random.default_rng(7)

    def bump_rate(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return 1.0 + 2.0 * np.exp(-0.5 * ((t % 50.0 - 25.0) / 8.0) ** 2)

    spikes = simulate_modulated_renewal(
        bump_rate, shape_param=1.0, T=T, renewal="gamma", rng=rng, dt=dt
    )
    n_bins = int(round(T / dt))
    t_grid = (np.arange(n_bins) + 0.5) * dt
    basis = bspline_basis_1d(t_grid % 50.0, n_knots=8, degree=3)
    assert basis.shape == (n_bins, 8)

    result = fit_modulated_renewal(
        spikes,
        covariates=None,
        basis=basis,
        renewal="gamma",
        dt=dt,
        tol=1e-4,
        n_inner=5,
        max_iter=50,
    )
    assert result.converged
    assert result.beta.shape == (9,)  # intercept + 8 basis coefficients

    rate_fn = result.rate_fn()
    # The bump peaks at t=25 (mod 50); the fitted rate there should exceed
    # the fitted rate at a trough (t=0).
    assert rate_fn(25.0) > rate_fn(0.0)


# ----------------------------------------------------------------------
# 4. Degenerate inputs
# ----------------------------------------------------------------------


def test_fewer_than_two_spikes_raises():
    covariates = np.ones((10, 1))
    with pytest.raises(ValueError, match="at least 2 spikes"):
        fit_modulated_renewal(np.array([0.5]), covariates)
    with pytest.raises(ValueError, match="at least 2 spikes"):
        fit_modulated_renewal(np.array([]), covariates)


def test_missing_covariates_raises():
    spikes = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="covariates must be provided"):
        fit_modulated_renewal(spikes, None)


def test_empty_covariates_raises():
    spikes = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="non-empty"):
        fit_modulated_renewal(spikes, np.zeros((0, 2)))


def test_invalid_renewal_family_raises():
    spikes = np.array([0.1, 0.5, 0.9])
    covariates = np.ones((10, 1))
    with pytest.raises(ValueError, match="gamma.*inverse_gaussian|renewal must be"):
        fit_modulated_renewal(spikes, covariates, renewal="log-normal")
    with pytest.raises(ValueError, match="gamma.*inverse_gaussian|renewal must be"):
        simulate_modulated_renewal(
            1.0, 1.0, T=10.0, renewal="log-normal", rng=np.random.default_rng(0)
        )


def test_simulate_invalid_T_and_shape_raise():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="T must be positive"):
        simulate_modulated_renewal(1.0, 1.0, T=0.0, rng=rng)
    with pytest.raises(ValueError, match="shape_param must be positive"):
        simulate_modulated_renewal(1.0, -2.0, T=10.0, rng=rng)


def test_spike_times_outside_grid_raises():
    covariates = np.ones((10, 1))  # spans [0, 10*dt); default dt inferred
    spikes = np.array([0.1, 100.0])  # far beyond the inferred grid
    with pytest.raises(ValueError, match="spike_times must lie within"):
        fit_modulated_renewal(spikes, covariates, dt=0.01)


# ----------------------------------------------------------------------
# simulate_modulated_renewal basic sanity
# ----------------------------------------------------------------------


def test_simulate_scalar_rate_produces_sorted_spikes_in_range():
    rng = np.random.default_rng(1)
    spikes = simulate_modulated_renewal(
        2.0, shape_param=3.0, T=100.0, renewal="gamma", rng=rng, dt=0.01
    )
    assert spikes.ndim == 1
    assert np.all(np.diff(spikes) >= 0)
    assert spikes.size > 0
    assert spikes.min() >= 0.0
    assert spikes.max() <= 100.0


def test_simulate_zero_rate_returns_empty():
    rng = np.random.default_rng(1)
    spikes = simulate_modulated_renewal(
        0.0, shape_param=3.0, T=10.0, renewal="gamma", rng=rng, dt=0.01
    )
    assert spikes.shape == (0,)


def test_renewal_cdf_and_hazard_agree_with_survival_identity():
    """r(u) = f(u) / (1 - F(u)); a spot check tying renewal_hazard and
    renewal_cdf together (both public API) via finite differences."""
    for renewal in ("gamma", "inverse_gaussian"):
        theta = 2.5
        u = np.array([0.2, 0.5, 1.0, 2.0])
        cdf = renewal_cdf(u, theta, renewal=renewal)
        haz = renewal_hazard(u, theta, renewal=renewal)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)
        assert np.all(haz > 0.0)
        # finite-difference pdf estimate should be consistent with hazard * sf
        eps = 1e-5
        cdf_plus = renewal_cdf(u + eps, theta, renewal=renewal)
        pdf_fd = (cdf_plus - cdf) / eps
        sf = 1.0 - cdf
        assert np.allclose(pdf_fd, haz * sf, rtol=0.05, atol=1e-4)
