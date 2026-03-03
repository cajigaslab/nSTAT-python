from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import Analysis as MatlabAnalysis


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "Analysis" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def test_analysis_fit_and_diagnostics_match_matlab_fixture() -> None:
    m = _mat()
    X = np.asarray(m["X"], dtype=float)
    y_p = _vec(m, "y_poisson")
    y_b = _vec(m, "y_binomial")
    dt = _scalar(m, "dt")

    fit_p = MatlabAnalysis.fitGLM(X, y_p, fitType="poisson", dt=dt)
    b_p = _vec(m, "b_poisson")
    # MATLAB glmfit estimates expected counts; Python fitGLM estimates rate with
    # Poisson log-likelihood on lambda*dt. Intercepts therefore differ by -log(dt).
    np.testing.assert_allclose(float(fit_p.intercept), b_p[0] - np.log(dt), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fit_p.coefficients, dtype=float), b_p[1:], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fit_p.predict(X), dtype=float), _vec(m, "mu_poisson") / dt, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(fit_p.log_likelihood), _scalar(m, "loglik_poisson"), rtol=1e-4, atol=1e-6)

    res_p = MatlabAnalysis.computeFitResidual(y_p, X, fit_p, dt=dt)
    np.testing.assert_allclose(np.asarray(res_p, dtype=float), _vec(m, "residual_poisson"), rtol=1e-4, atol=1e-6)

    inv_p = MatlabAnalysis.computeInvGausTrans(y_p, X, fit_p, dt=dt)
    np.testing.assert_allclose(np.asarray(inv_p, dtype=float), _vec(m, "invgaus_poisson"), rtol=1e-4, atol=1e-6)
    ks_p = MatlabAnalysis.computeKSStats(inv_p)
    assert np.isclose(float(ks_p["d_stat"]), _scalar(m, "ks_d_poisson"), rtol=1e-5, atol=1e-7)
    assert np.isclose(float(ks_p["n_events"]), _scalar(m, "ks_n_poisson"), rtol=0.0, atol=1e-12)

    fit_b = MatlabAnalysis.fitGLM(X, y_b, fitType="binomial", dt=dt)
    b_b = _vec(m, "b_binomial")
    np.testing.assert_allclose(float(fit_b.intercept), b_b[0], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fit_b.coefficients, dtype=float), b_b[1:], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fit_b.predict(X), dtype=float), _vec(m, "p_binomial"), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(fit_b.log_likelihood), _scalar(m, "loglik_binomial"), rtol=1e-4, atol=1e-6)

    res_b = MatlabAnalysis.computeFitResidual(y_b, X, fit_b, dt=dt)
    np.testing.assert_allclose(np.asarray(res_b, dtype=float), _vec(m, "residual_binomial"), rtol=1e-4, atol=1e-6)

    inv_b = MatlabAnalysis.computeInvGausTrans(y_b, X, fit_b, dt=dt)
    np.testing.assert_allclose(np.asarray(inv_b, dtype=float), _vec(m, "invgaus_binomial"), rtol=1e-4, atol=1e-6)
    ks_b = MatlabAnalysis.computeKSStats(inv_b)
    assert np.isclose(float(ks_b["d_stat"]), _scalar(m, "ks_d_binomial"), rtol=1e-5, atol=1e-7)
    assert np.isclose(float(ks_b["n_events"]), _scalar(m, "ks_n_binomial"), rtol=0.0, atol=1e-12)


def test_analysis_fdr_matches_matlab_fixture() -> None:
    m = _mat()
    pvals = _vec(m, "p_values")
    alpha = _scalar(m, "alpha")
    expected = np.asarray(m["fdr_mask"], dtype=float).reshape(-1) > 0.5
    observed = MatlabAnalysis.fdr_bh(pvals, alpha=alpha)
    np.testing.assert_array_equal(np.asarray(observed, dtype=bool), expected)
