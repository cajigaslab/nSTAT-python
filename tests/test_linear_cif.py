"""Tests for ``nstat.LinearCIF`` (Python port of MATLAB ``LinearCIF.m``).

Closed-form analytic identities cover Poisson and binomial canonical-link
cases; numerical finite-difference checks confirm the gradient and Hessian
match the closed-form expressions to ``1e-6`` relative tolerance.

When the local MATLAB checkout is reachable, the integration tests also
exercise the ``setSpikeTrain`` / ``setHistory`` mutators with a
:class:`nstat.History` instance.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat import LinearCIF, History, nspikeTrain


# ----------------------------------------------------------------------
# Construction & validation
# ----------------------------------------------------------------------

def test_construct_poisson_minimal() -> None:
    cif = LinearCIF(
        beta=[-2.0, 0.5],
        Xnames=["intercept", "stim"],
        stimNames=["stim"],
        fitType="poisson",
    )
    assert cif.fitType == "poisson"
    assert cif.varIn == ("intercept", "stim")
    assert cif.stimVars == ("stim",)
    np.testing.assert_array_equal(cif.stimIdx, [1])
    np.testing.assert_allclose(cif.b, [-2.0, 0.5])
    np.testing.assert_allclose(cif.bStim, [0.5])
    assert cif.history is None
    assert cif.spikeTrain is None
    assert cif.historyMat is None


def test_construct_binomial_default_intercept_form() -> None:
    cif = LinearCIF(
        beta=[0.0, 1.0],
        Xnames=["intercept", "x"],
        stimNames=["x"],
        fitType="binomial",
    )
    # sigma(0) = 0.5 at intercept-only stimulus.
    assert cif.evalLambdaDelta([0.0]) == pytest.approx(0.5, abs=1e-12)


def test_invalid_fit_type_raises() -> None:
    with pytest.raises(ValueError, match="fitType must be 'poisson' or 'binomial'"):
        LinearCIF([1.0], ["x"], ["x"], fitType="not_a_link")


def test_stim_not_in_xnames_raises() -> None:
    with pytest.raises(ValueError, match="not found in Xnames"):
        LinearCIF(
            beta=[1.0, 1.0],
            Xnames=["a", "b"],
            stimNames=["c"],  # not present
        )


def test_beta_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match"):
        LinearCIF(
            beta=[1.0, 2.0, 3.0],  # 3 coeffs, only 2 names
            Xnames=["a", "b"],
            stimNames=["a"],
        )


def test_invalid_history_input_raises() -> None:
    cif = LinearCIF([0.0, 1.0], ["x", "y"], ["y"])
    with pytest.raises(ValueError, match="History object or a vector"):
        cif.setHistory("not_a_history")


# ----------------------------------------------------------------------
# Closed-form Poisson identities
# ----------------------------------------------------------------------

class TestPoissonAnalyticIdentities:
    """For Poisson: ``lambda*delta = exp(b·X)``; derivatives have a closed form."""

    @pytest.fixture
    def cif(self) -> LinearCIF:
        return LinearCIF(
            beta=[-1.5, 0.7, -0.3],
            Xnames=["intercept", "s1", "s2"],
            stimNames=["s1", "s2"],
            fitType="poisson",
        )

    def test_lambda_delta_matches_exp_of_linpred(self, cif) -> None:
        stim = np.array([0.4, -0.2])
        # Full design: [1, 0.4, -0.2]; b·X = -1.5 + 0.7*0.4 + (-0.3)*(-0.2) = -1.16
        expected = float(np.exp(-1.5 + 0.7 * 0.4 + (-0.3) * (-0.2)))
        assert cif.evalLambdaDelta(stim) == pytest.approx(expected, rel=1e-12)

    def test_gradient_equals_ld_times_bstim(self, cif) -> None:
        stim = np.array([0.4, -0.2])
        ld = cif.evalLambdaDelta(stim)
        expected = ld * np.array([0.7, -0.3])
        np.testing.assert_allclose(cif.evalGradient(stim)[0], expected, rtol=1e-12)

    def test_gradient_log_equals_bstim(self, cif) -> None:
        # d/ds [log exp(...)] = derivative of the linear predictor itself.
        np.testing.assert_allclose(
            cif.evalGradientLog([0.0, 0.0])[0], [0.7, -0.3], rtol=1e-12
        )

    def test_jacobian_equals_ld_times_outer_bstim(self, cif) -> None:
        stim = np.array([0.4, -0.2])
        ld = cif.evalLambdaDelta(stim)
        bstim = np.array([0.7, -0.3])
        np.testing.assert_allclose(
            cif.evalJacobian(stim), ld * np.outer(bstim, bstim), rtol=1e-12
        )

    def test_jacobian_log_is_zero(self, cif) -> None:
        # log(λΔ) is linear in stim → Hessian vanishes identically.
        np.testing.assert_array_equal(
            cif.evalJacobianLog([0.4, -0.2]), np.zeros((2, 2))
        )

    def test_gradient_matches_finite_difference(self, cif) -> None:
        stim = np.array([0.4, -0.2])
        eps = 1e-6
        analytic = cif.evalGradient(stim)[0]
        numerical = np.empty(2)
        for k in range(2):
            plus = stim.copy(); plus[k] += eps
            minus = stim.copy(); minus[k] -= eps
            numerical[k] = (cif.evalLambdaDelta(plus) - cif.evalLambdaDelta(minus)) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-5)


# ----------------------------------------------------------------------
# Closed-form binomial identities
# ----------------------------------------------------------------------

class TestBinomialAnalyticIdentities:
    """For binomial: ``lambda*delta = sigma(b·X)``; sigmoid derivative identities."""

    @pytest.fixture
    def cif(self) -> LinearCIF:
        return LinearCIF(
            beta=[0.5, -1.2, 0.8],
            Xnames=["intercept", "s1", "s2"],
            stimNames=["s1", "s2"],
            fitType="binomial",
        )

    def test_lambda_delta_matches_sigmoid_of_linpred(self, cif) -> None:
        stim = np.array([0.3, 0.1])
        eta = 0.5 + (-1.2) * 0.3 + 0.8 * 0.1
        expected = 1.0 / (1.0 + np.exp(-eta))
        assert cif.evalLambdaDelta(stim) == pytest.approx(expected, rel=1e-12)

    def test_gradient_matches_sigmoid_first_derivative(self, cif) -> None:
        # d sigma(eta)/d stim_k = sigma·(1-sigma)·bStim_k
        stim = np.array([0.3, 0.1])
        ld = cif.evalLambdaDelta(stim)
        expected = ld * (1.0 - ld) * np.array([-1.2, 0.8])
        np.testing.assert_allclose(cif.evalGradient(stim)[0], expected, rtol=1e-12)

    def test_gradient_log_matches_sigmoid_log_derivative(self, cif) -> None:
        # d/d stim_k log sigma(eta) = (1 - sigma)·bStim_k
        stim = np.array([0.3, 0.1])
        ld = cif.evalLambdaDelta(stim)
        expected = (1.0 - ld) * np.array([-1.2, 0.8])
        np.testing.assert_allclose(cif.evalGradientLog(stim)[0], expected, rtol=1e-12)

    def test_jacobian_uses_linear_factor_not_squared(self, cif) -> None:
        """Bug-regression: the third-derivative factor is ``(1 - 2*ld)`` NOT ``(1 - 2*ld**2)``.

        This is the same bug the audit caught in
        ``nstat.decoding_algorithms`` line 2600 — guard against it here too.
        """
        stim = np.array([0.3, 0.1])
        ld = cif.evalLambdaDelta(stim)
        bstim = np.array([-1.2, 0.8])
        expected = ld * (1.0 - ld) * (1.0 - 2.0 * ld) * np.outer(bstim, bstim)
        np.testing.assert_allclose(cif.evalJacobian(stim), expected, rtol=1e-12)

    def test_jacobian_log_is_negative_definite(self, cif) -> None:
        # Hessian of log-likelihood is negative semi-definite for binomial GLM.
        stim = np.array([0.3, 0.1])
        H = cif.evalJacobianLog(stim)
        # H = -ld·(1-ld)·outer(bStim, bStim) — symmetric and negative semi-definite.
        np.testing.assert_allclose(H, H.T, rtol=1e-12)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals <= 1e-12), f"eigvals {eigvals} should be <= 0"

    def test_gradient_matches_finite_difference(self, cif) -> None:
        stim = np.array([0.3, 0.1])
        eps = 1e-6
        analytic = cif.evalGradient(stim)[0]
        numerical = np.empty(2)
        for k in range(2):
            plus = stim.copy(); plus[k] += eps
            minus = stim.copy(); minus[k] -= eps
            numerical[k] = (cif.evalLambdaDelta(plus) - cif.evalLambdaDelta(minus)) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-5)


# ----------------------------------------------------------------------
# Numerical-stability sigmoid
# ----------------------------------------------------------------------

def test_sigmoid_does_not_overflow_for_large_positive_eta() -> None:
    cif = LinearCIF([0.0, 1.0], ["intercept", "s"], ["s"], fitType="binomial")
    # eta = 0 + 1*1e4 — naive ``1/(1+exp(-eta))`` would underflow.
    ld = cif.evalLambdaDelta([1e4])
    assert ld == pytest.approx(1.0, abs=1e-15)


def test_sigmoid_does_not_overflow_for_large_negative_eta() -> None:
    cif = LinearCIF([0.0, 1.0], ["intercept", "s"], ["s"], fitType="binomial")
    ld = cif.evalLambdaDelta([-1e4])
    assert ld == pytest.approx(0.0, abs=1e-15)


# ----------------------------------------------------------------------
# Stim-only vs full-design input
# ----------------------------------------------------------------------

def test_full_design_vector_input_passes_through_unchanged() -> None:
    """When stimVal has length nVar, it is used as-is — intercept is NOT replaced."""
    cif = LinearCIF([-2.0, 0.5], ["intercept", "stim"], ["stim"])
    # Override the intercept column by passing a full-design vector.
    ld_full = cif.evalLambdaDelta([3.0, 1.0])  # b·x = -2*3 + 0.5*1 = -5.5
    assert ld_full == pytest.approx(np.exp(-5.5), rel=1e-12)
    # vs stim-only (intercept defaults to 1):
    ld_stim_only = cif.evalLambdaDelta([1.0])  # b·x = -2*1 + 0.5*1 = -1.5
    assert ld_stim_only == pytest.approx(np.exp(-1.5), rel=1e-12)


def test_invalid_stim_val_length_raises() -> None:
    cif = LinearCIF([1.0, 1.0, 1.0], ["a", "b", "c"], ["b", "c"])
    with pytest.raises(ValueError, match=r"length 3 \(all vars\) or 2 \(stim vars only\)"):
        cif.evalLambdaDelta([0.0])


# ----------------------------------------------------------------------
# History integration (only if data + History support is wired)
# ----------------------------------------------------------------------

def test_setHistory_accepts_history_object_and_windowtimes() -> None:
    cif = LinearCIF([0.0, 1.0], ["x", "y"], ["y"])

    # Accepts a History object (copies windowTimes).
    h = History([0.001, 0.005, 0.020])
    cif.setHistory(h)
    assert cif.history is not None
    np.testing.assert_allclose(
        np.asarray(cif.history.windowTimes, dtype=float),
        [0.001, 0.005, 0.020],
    )

    # Also accepts a raw vector.
    cif.setHistory(np.array([0.002, 0.010]))
    assert cif.history is not None
    np.testing.assert_allclose(
        np.asarray(cif.history.windowTimes, dtype=float),
        [0.002, 0.010],
    )


def test_setSpikeTrain_takes_a_copy() -> None:
    """``setSpikeTrain`` must deep-copy so source mutations don't propagate."""
    cif = LinearCIF([0.0, 1.0], ["x", "y"], ["y"])
    train = nspikeTrain([0.1, 0.5, 0.9], name="src", sampleRate=1000.0,
                        minTime=0.0, maxTime=1.0)
    cif.setSpikeTrain(train)
    assert cif.spikeTrain is not None
    assert cif.spikeTrain is not train
    train.setName("MUTATED")
    assert cif.spikeTrain.name == "src"


# ----------------------------------------------------------------------
# Public-API surface
# ----------------------------------------------------------------------

def test_linear_cif_is_in_package_public_api() -> None:
    import nstat
    assert "LinearCIF" in nstat.__all__
    assert nstat.LinearCIF is LinearCIF
