"""Correctness of the PP_EM / PPLFP_EM routines, mirroring the repaired MATLAB.

The MATLAB EM routines (``PointProcessEM.m`` / ``PPLFP.m``) received a set of
correctness fixes (repaired MATLAB ``fix/pp-em`` @ ``a457b54``, pending upstream
merge).  This file pins the Python mirror of each one with a test that is
independent of the implementation (finite differences, invariances, exact
equalities), in the style of the MATLAB ``testPointProcessEMCorrectness`` /
``testPPLFPEMCorrectness`` suites.

Standard errors (``PP_ComputeParamStandardErrors`` /
``PPLFP_ComputeParamStandardErrors``)
---------------------------------------------------------------------
The harness puts every parameter at its complete-data MLE given a known state
path (``W_K = 1e-12 I``): A and Q (and C, alpha, R for PPLFP) in closed form,
(mu, beta, gamma) of each cell by Newton on that cell's complete-data
log-likelihood.  Every complete-data score is then ~0, so the Monte-Carlo
missing information vanishes and each SE block must equal
``sqrt(diag(inv(-H_fd)))``, where ``H_fd`` is a central finite difference of
the analytic score in that block.  This checks the information blocks *and*
their layout (cell c's SEs in column c) at once:

* binomial beta information sign (MATLAB C4 / A4): it was
  ``E[p xx'] + E[p^2 xx'] - 2E[p^3 xx']`` (negative definite);
* binomial mu information cubic coefficient -3 -> -2 (MATLAB A1);
* SE.beta / SE.gamma layout (MATLAB 2a / B4: ``reshape(v, C, dx)'`` scrambled
  the cell-by-cell vector; NumPy's row-major ``v.reshape(C, dx).T`` never did,
  and this pins it);
* single-cell history (C == 1, W > 1; MATLAB B6), including MATLAB's 2-D
  ``N x W`` storage of a one-cell history.

``Pvals.gamma`` of ``PP_ComputeParamStandardErrors`` paired the window-major
``gammahat.ravel()`` with the cell-major SE vector (scrambled for W > 1 and
C > 1); each p-value must be the z-test of its own (gamma, SE) pair.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from nstat.decoding.PPLFP import PPLFP
from nstat.decoding_algorithms import DecodingAlgorithms, _compute_history_terms

FD_STEP = 1e-6


def _fd_jac(f, x0):
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    f0 = np.atleast_1d(f(x0))
    J = np.zeros((f0.size, x0.size))
    for i in range(x0.size):
        e = np.zeros_like(x0)
        e[i] = FD_STEP
        J[:, i] = (np.atleast_1d(f(x0 + e)) - np.atleast_1d(f(x0 - e))) / (2 * FD_STEP)
    return J


def _link(eta, fit):
    return np.exp(eta) if fit == "poisson" else 1.0 / (1.0 + np.exp(-eta))


def _score_factor(dN, p, fit):
    """d/d eta of the toolbox log-likelihood (binomial: dN*log p - p, logistic p)."""
    return dN - p if fit == "poisson" else dN - (dN + 1) * p + p ** 2


def _hess_factor(dN, p, fit):
    return -p if fit == "poisson" else -(dN + 1) * p + (dN + 3) * p ** 2 - 2 * p ** 3


def _cell_mle(Z, d, fit, theta0):
    """Newton on one cell's complete-data log-likelihood; design Z is K x P."""
    theta = np.array(theta0, dtype=float)
    for _ in range(200):
        p = _link(Z @ theta, fit)
        g = Z.T @ _score_factor(d, p, fit)
        Hm = (Z * _hess_factor(d, p, fit)[:, None]).T @ Z
        step = np.linalg.solve(Hm, g)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-13:
            break
    return theta


def _mle_problem(fit, *, dx=2, C=2, nW=3, K=1500, seed=0):
    """State path, spikes and history, with (A, Q, mu, beta, gamma) at the complete-data MLE."""
    rng = np.random.default_rng(seed)
    A_true = np.array([[0.97, 0.03], [-0.02, 0.95]])[:dx, :dx]
    Q_true = np.diag([0.02, 0.03])[:dx, :dx]
    x0 = np.zeros(dx)
    x = np.zeros((dx, K))
    prev = x0
    for k in range(K):
        prev = A_true @ prev + rng.multivariate_normal(np.zeros(dx), Q_true)
        x[:, k] = prev
    mu0 = np.linspace(-2.0, -1.5, C) if fit == "poisson" else np.linspace(-1.4, -0.9, C)
    beta0 = 0.6 * rng.standard_normal((dx, C))
    p = _link(mu0[:, None] + beta0.T @ x, fit)
    dN = (rng.random((C, K)) < np.minimum(p, 0.9)).astype(float)
    wt = np.arange(nW + 1) * 0.001
    HkAll = _compute_history_terms(dN, 0.001, wt)  # N x nW x C
    mu = np.zeros(C)
    beta = np.zeros((dx, C))
    gamma = np.zeros((nW, C))
    for c in range(C):
        Z = np.column_stack([np.ones(K), x.T, HkAll[:, :, c]])
        th = _cell_mle(Z, dN[c], fit, np.concatenate([[mu0[c]], beta0[:, c], -0.2 * np.ones(nW)]))
        mu[c], beta[:, c], gamma[:, c] = th[0], th[1:1 + dx], th[1 + dx:]
    # A, Q at their MLE (full A, diagonal Q, x0 fixed): the SE routine's own
    # Monte-Carlo sums over the (here ~exact) draws, with x_0 = x0.
    xm1 = np.column_stack([x0, x[:, :-1]])
    Sxkm1xkm1 = xm1 @ xm1.T
    Sxkxkm1 = x @ xm1.T
    A = Sxkxkm1 @ np.linalg.inv(Sxkm1xkm1)
    resid = x - A @ xm1
    Q = np.diag(np.diag(resid @ resid.T)) / K
    WK = np.tile((1e-12 * np.eye(dx))[:, :, None], (1, 1, K))
    ES = dict(Sxkm1xkm1=Sxkm1xkm1, Sxkxkm1=Sxkxkm1, Sxkm1xk=Sxkxkm1.T, Sxkxk=x @ x.T,
              sumXkTerms=resid @ resid.T)
    return dict(x=x, x0=x0, dN=dN, wt=wt, HkAll=HkAll, mu=mu, beta=beta, gamma=gamma, A=A, Q=Q, WK=WK,
                ES=ES, K=K, dx=dx, C=C, nW=nW, fit=fit)


def _expected_se(P):
    """Per-block SE = sqrt(diag(inv(-H_fd))) of each cell's complete-data score."""
    x, dN, Hk, fit, K = P["x"], P["dN"], P["HkAll"], P["fit"], P["K"]
    se_mu, se_beta, se_gamma = [], [], []
    for c in range(P["C"]):
        mu, b, g = P["mu"][c], P["beta"][:, c], P["gamma"][:, c]
        H = Hk[:, :, c]

        def sc(m, bb, gg):
            return _score_factor(dN[c], _link(m + bb @ x + H @ gg, fit), fit)

        J_mu = _fd_jac(lambda m: np.sum(sc(m[0], b, g)), [mu])
        J_b = _fd_jac(lambda bb: x @ sc(mu, bb, g), b)
        J_g = _fd_jac(lambda gg: H.T @ sc(mu, b, gg), g)
        se_mu.append(np.sqrt(1.0 / -J_mu[0, 0]))
        se_beta.append(np.sqrt(np.diag(np.linalg.inv(-J_b))))
        se_gamma.append(np.sqrt(np.diag(np.linalg.inv(-J_g))))
    return np.array(se_mu), np.column_stack(se_beta), np.column_stack(se_gamma)


def _pp_se(P, HkAll=None):
    cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0, 0, 10)
    np.random.seed(0)
    return DecodingAlgorithms.PP_ComputeParamStandardErrors(
        P["dN"], P["x"], P["WK"], P["A"], P["Q"], P["x0"], 1e-9 * np.eye(P["dx"]), P["ES"], P["fit"],
        P["mu"], P["beta"], P["gamma"], P["wt"], P["HkAll"] if HkAll is None else HkAll, cons,
    )


def _pplfp_mle_extra(P, seed=1):
    """Gaussian channel y = C x + alpha + noise with (C, alpha, R) at their MLE."""
    rng = np.random.default_rng(seed)
    K, dx = P["K"], P["dx"]
    Ctrue = np.array([[1.0, -0.5]])[:, :dx]
    y = Ctrue @ P["x"] + 0.3 + np.sqrt(0.05) * rng.standard_normal((1, K))
    X1 = np.vstack([P["x"], np.ones((1, K))])
    coef = y @ X1.T @ np.linalg.inv(X1 @ X1.T)
    Chat, alpha = coef[:, :dx], coef[:, dx]
    r = y - Chat @ P["x"] - alpha[:, None]
    R = np.diag(np.diag(r @ r.T)) / K
    return y, Chat, alpha, R


def _pplfp_se(P, y, Chat, alpha, R, HkAll=None):
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 10, 0)
    ES = dict(P["ES"])
    return PPLFP.PPLFP_ComputeParamStandardErrors(
        y, P["dN"], P["x"], P["WK"], P["A"], P["Q"], Chat, R, alpha, P["x0"], 1e-9 * np.eye(P["dx"]), ES,
        P["fit"], P["mu"], P["beta"], P["gamma"], P["wt"], P["HkAll"] if HkAll is None else HkAll, cons,
    )


SE_RTOL = 1e-4  # measured agreement <= 3.4e-8 (posterior noise of the 1e-12 draws, FD error)


@pytest.mark.parametrize("fit", ["poisson", "binomial"])
@pytest.mark.parametrize("shape", [dict(C=2, nW=3), dict(C=1, nW=3)], ids=["C2W3", "C1W3"])
def test_pp_standard_errors_match_finite_difference(fit, shape) -> None:
    P = _mle_problem(fit, **shape)
    SE, Pvals, _ = _pp_se(P)
    se_mu, se_beta, se_gamma = _expected_se(P)
    np.testing.assert_allclose(SE["mu"], se_mu, rtol=SE_RTOL, err_msg="SE.mu")
    np.testing.assert_allclose(SE["beta"], se_beta, rtol=SE_RTOL, err_msg="SE.beta")
    np.testing.assert_allclose(SE["gamma"], se_gamma, rtol=SE_RTOL, err_msg="SE.gamma")


@pytest.mark.parametrize("fit", ["poisson", "binomial"])
@pytest.mark.parametrize("shape", [dict(C=2, nW=3), dict(C=1, nW=3)], ids=["C2W3", "C1W3"])
def test_pplfp_standard_errors_match_finite_difference(fit, shape) -> None:
    P = _mle_problem(fit, **shape)
    y, Chat, alpha, R = _pplfp_mle_extra(P)
    SE, Pvals, _ = _pplfp_se(P, y, Chat, alpha, R)
    se_mu, se_beta, se_gamma = _expected_se(P)
    np.testing.assert_allclose(np.ravel(SE["mu"]), se_mu, rtol=SE_RTOL, err_msg="SE.mu")
    np.testing.assert_allclose(SE["beta"], se_beta, rtol=SE_RTOL, err_msg="SE.beta")
    np.testing.assert_allclose(SE["gamma"], se_gamma, rtol=SE_RTOL, err_msg="SE.gamma")


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_standard_error_pvalues_pair_each_parameter_with_its_own_se(family) -> None:
    # W = 3 windows, C = 2 cells, dx = 2: any window/cell or state/cell mix-up
    # between the parameter and SE orderings changes some p-value.
    P = _mle_problem("poisson", C=2, nW=3)
    if family == "PP":
        SE, Pvals, _ = _pp_se(P)
    else:
        SE, Pvals, _ = _pplfp_se(P, *_pplfp_mle_extra(P))
    for key, param in (("beta", P["beta"]), ("gamma", P["gamma"])):
        expected = 2.0 * (1.0 - norm.cdf(np.abs(param / SE[key])))
        np.testing.assert_allclose(Pvals[key], expected, rtol=1e-12, atol=1e-300, err_msg=key)


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
@pytest.mark.parametrize("fit", ["poisson", "binomial"])
def test_single_cell_history_stored_2d_equals_3d(family, fit) -> None:
    # MATLAB stores a one-cell N x W x 1 history as N x W; both SE routines
    # must read it as the history (it used to be dropped silently in PP).
    P = _mle_problem(fit, C=1, nW=3, K=600)
    if family == "PP":
        out3 = _pp_se(P)
        out2 = _pp_se(P, HkAll=P["HkAll"][:, :, 0])
    else:
        extra = _pplfp_mle_extra(P)
        from nstat.extras.matlab_rng import seeded_global_rng

        with seeded_global_rng(4):
            out3 = _pplfp_se(P, *extra)
        with seeded_global_rng(4):
            out2 = _pplfp_se(P, *extra, HkAll=P["HkAll"][:, :, 0])
    assert out2[2] == out3[2]
    for part in (0, 1):
        assert sorted(out2[part]) == sorted(out3[part])
        for key in out3[part]:
            assert np.array_equal(np.asarray(out2[part][key]), np.asarray(out3[part][key])), (part, key)


# ---------------------------------------------------------------------------
# M-steps: binomial Newton-Raphson beta Hessian (MATLAB bug 6 / C5) and the
# all-zero gamma skip (MATLAB ``any(any(gammahat_new~=0))``)
# ---------------------------------------------------------------------------


def _mstep_problem(fit="binomial", *, dx=2, C=3, nW=2, K=400, seed=4):
    """Known state path x (W_K = 0.02 I) and spikes from the generating (mu, beta, gamma)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.95, 0.02], [-0.03, 0.9]])[:dx, :dx]
    x = np.zeros((dx, K))
    prev = np.zeros(dx)
    for k in range(K):
        prev = A @ prev + rng.multivariate_normal(np.zeros(dx), np.diag([0.02, 0.03])[:dx, :dx])
        x[:, k] = prev
    mu = np.linspace(-1.2, -0.6, C) if fit == "binomial" else np.linspace(-2.5, -2.0, C)
    beta = 0.8 * rng.standard_normal((dx, C))
    gamma = -0.3 - 0.5 * rng.random((nW, C))
    p = _link(mu[:, None] + beta.T @ x, fit)
    dN = (rng.random((C, K)) < np.minimum(p, 0.9)).astype(float)
    wt = np.arange(nW + 1) * 0.001
    HkAll = _compute_history_terms(dN, 0.001, wt)
    W_K = np.tile((0.02 * np.eye(dx))[:, :, None], (1, 1, K))
    ES = dict(Sxkm1xkm1=x @ x.T, Sxkxkm1=x[:, 1:] @ x[:, :-1].T, Sxkm1xk=x[:, :-1] @ x[:, 1:].T, Sxkxk=x @ x.T,
              sumXkTerms=0.02 * K * np.eye(dx), Sxkyk=x @ (np.array([[1.0, 0.5]])[:, :dx] @ x).T,
              sumYkTerms=0.1 * K * np.eye(1))
    return dict(x=x, dN=dN, wt=wt, HkAll=HkAll, mu=mu, beta=beta, gamma=gamma, W_K=W_K, ES=ES, dx=dx, C=C, K=K,
                y=np.array([[1.0, 0.5]])[:, :dx] @ x)


def _mc_draws(P, z_source, McExp=50):
    """The M-step's own draws x_K(:,k) + chol(W_K(:,:,k))' z, z from ``z_source`` in time order."""
    dx, K = P["dx"], P["K"]
    draws = np.zeros((dx, McExp, K))
    for k in range(K):
        z = z_source((dx, McExp))
        draws[:, :, k] = P["x"][:, k:k + 1] + np.linalg.cholesky(P["W_K"][:, :, k]).T @ z
    return draws


def _beta_gradient(P, b, c, draws, fit="binomial"):
    """GradTerm of the Newton beta step: d/d beta of the MC expected complete-data log-likelihood."""
    G = np.zeros(P["dx"])
    for k in range(P["K"]):
        xk = draws[:, :, k]
        p = _link(P["mu"][c] + b @ xk + P["gamma"][:, c] @ P["HkAll"][k, :, c], fit)
        d = P["dN"][c, k]
        if fit == "poisson":
            G += d * P["x"][:, k] - np.mean(p * xk, axis=1)
        else:
            G += d * P["x"][:, k] - (d + 1) * np.mean(p * xk, axis=1) + np.mean(p ** 2 * xk, axis=1)
    return G


def _run_pp_mstep(P, fit="binomial", gamma=None, seed=42):
    cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0)
    np.random.seed(seed)
    return DecodingAlgorithms.PP_MStep(
        P["dN"], P["x"], P["W_K"], np.zeros(P["dx"]), 1e-9 * np.eye(P["dx"]), P["ES"], fit, P["mu"], P["beta"],
        P["gamma"] if gamma is None else gamma, P["wt"], P["HkAll"], cons, "NewtonRaphson",
    )


def _run_pplfp_mstep(P, fit="binomial", gamma=None, seed=42):
    from nstat.extras.matlab_rng import seeded_global_rng

    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 50, 0)
    with seeded_global_rng(seed):
        return PPLFP.PPLFP_MStep(
            P["dN"], P["y"], P["x"], P["W_K"], np.zeros(P["dx"]), 1e-9 * np.eye(P["dx"]), P["ES"], fit, P["mu"],
            P["beta"], P["gamma"] if gamma is None else gamma, P["wt"], P["HkAll"], cons, "NewtonRaphson",
        )


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
@pytest.mark.parametrize("fit", ["poisson", "binomial"])
def test_newton_raphson_beta_step_reaches_a_stationary_point(family, fit) -> None:
    # The Newton beta step must end where the gradient of the Monte-Carlo
    # expected complete-data log-likelihood (with the step's own draws) is 0.
    # With the former positive-definite binomial Hessian the step moved
    # downhill: one M-step from the generating beta went to |beta| ~ 1e4-1e14.
    P = _mstep_problem(fit)
    if family == "PP":
        out = _run_pp_mstep(P, fit)
        beta_out = out[3]
        rs = np.random.RandomState()
        rs.seed(42)
        draws = _mc_draws(P, lambda shape: rs.randn(*shape))
    else:
        out = _run_pplfp_mstep(P, fit)
        beta_out = out[6]
        gen = np.random.default_rng(42)
        draws = _mc_draws(P, gen.standard_normal)
    assert np.all(np.isfinite(beta_out))
    assert np.max(np.abs(beta_out - P["beta"])) < 2.0
    for c in range(P["C"]):
        g_in = _beta_gradient(P, P["beta"][:, c], c, draws, fit)
        g_out = _beta_gradient(P, beta_out[:, c], c, draws, fit)
        assert np.max(np.abs(g_out)) < 1e-4 * max(1.0, np.max(np.abs(g_in))), (c, g_in, g_out)


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_all_zero_gamma_is_not_estimated(family) -> None:
    # MATLAB skips the gamma Newton step when ~any(any(gammahat_new~=0)): a
    # zero gamma means "no history coefficients", even with windowTimes set.
    P = _mstep_problem("poisson")
    zero = np.zeros_like(P["gamma"])
    out = _run_pp_mstep(P, "poisson", gamma=zero) if family == "PP" else _run_pplfp_mstep(P, "poisson", gamma=zero)
    gamma_out = out[4] if family == "PP" else out[7]
    assert np.array_equal(np.asarray(gamma_out), zero)
    # ... while a nonzero gamma is estimated.
    out = _run_pp_mstep(P, "poisson") if family == "PP" else _run_pplfp_mstep(P, "poisson")
    gamma_out = out[4] if family == "PP" else out[7]
    assert not np.array_equal(np.asarray(gamma_out), P["gamma"])
