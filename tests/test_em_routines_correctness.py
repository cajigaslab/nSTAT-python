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
        P["mu"], P["beta"], P.get("gamma_arg", P["gamma"]), P["wt"], P["HkAll"] if HkAll is None else HkAll, cons,
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


def _pplfp_se(P, y, Chat, alpha, R, HkAll=None, seed=0):
    from nstat.extras.matlab_rng import seeded_global_rng

    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 10, 0)
    ES = dict(P["ES"])
    with seeded_global_rng(seed):  # PPLFP draws through np.random.default_rng()
        return PPLFP.PPLFP_ComputeParamStandardErrors(
            y, P["dN"], P["x"], P["WK"], P["A"], P["Q"], Chat, R, alpha, P["x0"], 1e-9 * np.eye(P["dx"]), ES,
            P["fit"], P["mu"], P["beta"], P.get("gamma_arg", P["gamma"]), P["wt"],
            P["HkAll"] if HkAll is None else HkAll, cons,
        )


# Measured agreement over 20 Monte-Carlo seeds x every case below (640 SE
# computations): <= 6.2e-8 (posterior noise of the 1e-12 draws, FD error), so
# 1e-6 (MATLAB's own FD tolerance) leaves a >15x margin.
SE_RTOL = 1e-6

# C = 1 with W = 1 is a single history coefficient; it is also passed as a
# 0-d scalar ("C1W1s").  The MATLAB SE routines left the gamma parameter
# count unassigned there (fixed upstream after a457b54: one gamma parameter,
# as in PP_EM's IC count); a 1 x 1 gamma raised in the PP routines here.
SE_SHAPES = [dict(C=2, nW=3), dict(C=1, nW=3), dict(C=1, nW=1), dict(C=1, nW=1, scalar=True)]
SE_SHAPE_IDS = ["C2W3", "C1W3", "C1W1", "C1W1s"]


def _se_problem(fit, shape):
    shape = dict(shape)
    scalar = shape.pop("scalar", False)
    P = _mle_problem(fit, **shape)
    if scalar:
        P["gamma_arg"] = np.asarray(float(P["gamma"][0, 0]))
    return P


@pytest.mark.parametrize("fit", ["poisson", "binomial"])
@pytest.mark.parametrize("shape", SE_SHAPES, ids=SE_SHAPE_IDS)
def test_pp_standard_errors_match_finite_difference(fit, shape) -> None:
    P = _se_problem(fit, shape)
    SE, Pvals, nTerms = _pp_se(P)
    # A (full, dx^2) + Q (diagonal, dx) + mu + beta + gamma
    assert nTerms == P["dx"] ** 2 + P["dx"] + P["C"] * (1 + P["dx"] + P["nW"])
    se_mu, se_beta, se_gamma = _expected_se(P)
    np.testing.assert_allclose(SE["mu"], se_mu, rtol=SE_RTOL, err_msg="SE.mu")
    np.testing.assert_allclose(SE["beta"], se_beta, rtol=SE_RTOL, err_msg="SE.beta")
    se_g = np.ravel(SE["gamma"]) if "gamma_arg" in P else SE["gamma"]  # 0-d gamma -> 1-element SE
    np.testing.assert_allclose(se_g, se_gamma.reshape(np.shape(se_g)), rtol=SE_RTOL, err_msg="SE.gamma")


@pytest.mark.parametrize("fit", ["poisson", "binomial"])
@pytest.mark.parametrize("shape", SE_SHAPES, ids=SE_SHAPE_IDS)
def test_pplfp_standard_errors_match_finite_difference(fit, shape) -> None:
    P = _se_problem(fit, shape)
    y, Chat, alpha, R = _pplfp_mle_extra(P)
    SE, Pvals, nTerms = _pplfp_se(P, y, Chat, alpha, R)
    # A + Q + C (1 x dx) + R + alpha (1 channel) + mu + beta + gamma
    assert nTerms == P["dx"] ** 2 + P["dx"] + P["dx"] + 1 + 1 + P["C"] * (1 + P["dx"] + P["nW"])
    se_mu, se_beta, se_gamma = _expected_se(P)
    np.testing.assert_allclose(np.ravel(SE["mu"]), se_mu, rtol=SE_RTOL, err_msg="SE.mu")
    np.testing.assert_allclose(SE["beta"], se_beta, rtol=SE_RTOL, err_msg="SE.beta")
    se_g = np.ravel(SE["gamma"]) if "gamma_arg" in P else SE["gamma"]
    np.testing.assert_allclose(se_g, se_gamma.reshape(np.shape(se_g)), rtol=SE_RTOL, err_msg="SE.gamma")


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
        expected = 2.0 * norm.cdf(-np.abs(param / SE[key]))  # MATLAB ztest: 2*normcdf(-|z|)
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


def _mstep_problem(fit="binomial", *, dx=2, C=3, nW=2, K=400, seed=4, W=None):
    """Known state path x (W_K = W, default 0.02 I) and spikes from the generating (mu, beta, gamma)."""
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
    W_K = np.tile((0.02 * np.eye(dx) if W is None else np.asarray(W, dtype=float))[:, :, None], (1, 1, K))
    ES = dict(Sxkm1xkm1=x @ x.T, Sxkxkm1=x[:, 1:] @ x[:, :-1].T, Sxkm1xk=x[:, :-1] @ x[:, 1:].T, Sxkxk=x @ x.T,
              sumXkTerms=0.02 * K * np.eye(dx), Sxkyk=x @ (np.array([[1.0, 0.5]])[:, :dx] @ x).T,
              sumYkTerms=0.1 * K * np.eye(1))
    return dict(x=x, dN=dN, wt=wt, HkAll=HkAll, mu=mu, beta=beta, gamma=gamma, W_K=W_K, ES=ES, dx=dx, C=C, K=K,
                y=np.array([[1.0, 0.5]])[:, :dx] @ x)


def _mc_draws(P, z_source, McExp=50):
    """The M-step's own draws x_K(:,k) + R' z, R = chol(W_K(:,:,k)) (MATLAB mcStateDraws, F9).

    ``R' = np.linalg.cholesky(W)``, the lower factor; z from ``z_source`` in
    time order.  (The legacy upper-factor draw R z has covariance R R' != W
    for a non-diagonal W.)
    """
    dx, K = P["dx"], P["K"]
    draws = np.zeros((dx, McExp, K))
    for k in range(K):
        z = z_source((dx, McExp))
        draws[:, :, k] = P["x"][:, k:k + 1] + np.linalg.cholesky(P["W_K"][:, :, k]) @ z
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


@pytest.mark.parametrize("W", [None, [[0.02, 0.012], [0.012, 0.03]]], ids=["Wdiag", "Wfull"])
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
@pytest.mark.parametrize("fit", ["poisson", "binomial"])
def test_newton_raphson_beta_step_reaches_a_stationary_point(family, fit, W) -> None:
    # The Newton beta step must end where the gradient of the Monte-Carlo
    # expected complete-data log-likelihood (with the step's own draws) is 0.
    # With the former positive-definite binomial Hessian the step moved
    # downhill: one M-step from the generating beta went to |beta| ~ 1e4-1e14.
    # Wfull: the draws are x_K + chol(W)' z, covariance W (MATLAB F9); the
    # former chol(W) z draws (covariance R R') end elsewhere for a
    # non-diagonal W_K.
    P = _mstep_problem(fit, W=W)
    if family == "PP":
        out = _run_pp_mstep(P, fit)
        beta_out = out[3]
    else:
        out = _run_pplfp_mstep(P, fit)
        beta_out = out[6]
    # Both M-steps draw from NumPy's global stream, which seeded_global_rng(42)
    # seeds (PPLFP drew from an unseeded default_rng(), patched to
    # default_rng(42), before the EM final pass made its draws reproducible).
    rs = np.random.RandomState()
    rs.seed(42)
    draws = _mc_draws(P, lambda shape: rs.randn(*shape))
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


# ---------------------------------------------------------------------------
# Monte Carlo state draws (MATLAB F9, fix/pp-em @ 8dbd0e4: mcStateDraws).
# MATLAB drew m + chol(W)*z; chol is the UPPER factor R (R'R = W), so the
# draws had covariance R R' -- W only for a diagonal W.  The draw is m + R'z.
# ---------------------------------------------------------------------------

_W3 = np.array([[1.0, 0.8, 0.3], [0.8, 2.0, -0.6], [0.3, -0.6, 1.5]])


def _cov_se(W, n):
    """Monte-Carlo SE of each sample-covariance entry of n N(., W) draws: sqrt((W_ii W_jj + W_ij^2) / n)."""
    d = np.diag(W)
    return np.sqrt((np.outer(d, d) + W ** 2) / n)


def _assert_draw_covariance(X, W, m=None):
    """Sample mean / covariance within 5 MC SEs of (m, W); the legacy R R' more than 50 SEs away."""
    n = X.shape[1]
    if m is not None:
        assert np.all(np.abs(X.mean(axis=1) - m) < 5 * np.sqrt(np.diag(W) / n))
    S = np.cov(X)
    assert np.all(np.abs(S - W) < 5 * _cov_se(W, n)), S
    R = np.linalg.cholesky(W).T  # MATLAB chol(W): upper, R'R = W
    legacy = R @ R.T
    assert np.max(np.abs(S - legacy) / _cov_se(legacy, n)) > 50


def test_mc_state_draws_have_the_requested_covariance() -> None:
    from nstat.decoding_algorithms import _mc_state_draws

    n = 100_000
    m = np.array([0.5, -1.0, 2.0])
    gen = np.random.default_rng(11)
    X = _mc_state_draws(m, _W3, n, lambda d, k: gen.standard_normal((d, k)))
    assert X.shape == (3, n)
    _assert_draw_covariance(X, _W3, m)


def test_mc_state_draws_diagonal_w_is_the_legacy_draw() -> None:
    # For a diagonal W, R' = R: the draws are bit-identical to the former
    # chol(W)*z (MATLAB's own check), so no diagonal-W result moves.
    from nstat.decoding_algorithms import _mc_state_draws

    W = np.diag([0.3, 1.2, 0.05])
    m = np.array([0.1, 0.2, -0.3])
    z = np.random.default_rng(5).standard_normal((3, 200))
    X = _mc_state_draws(m, W, 200, lambda d, k: z)
    assert np.array_equal(X, m[:, None] + np.linalg.cholesky(W).T @ z)


def _draw_site_problem(K=400, C=2):
    rng = np.random.default_rng(8)
    dx = 3
    x = np.zeros((dx, K))  # draws centred at 0: pooled over k they are N(0, W)
    W_K = np.tile(_W3[:, :, None], (1, 1, K))
    dN = (rng.random((C, K)) < 0.05).astype(float)
    mu = np.full(C, -3.0)
    beta = 0.1 * rng.standard_normal((dx, C))
    y = rng.standard_normal((1, K))
    ES = dict(Sxkm1xkm1=K * _W3, Sxkxkm1=0.9 * K * _W3, Sxkm1xk=0.9 * K * _W3, Sxkxk=K * _W3,
              sumXkTerms=0.2 * K * _W3, Sxkyk=np.zeros((dx, 1)), sumYkTerms=K * np.eye(1))
    H = np.zeros((K, 1, C))
    return dict(dx=dx, K=K, C=C, x=x, W_K=W_K, dN=dN, mu=mu, beta=beta, y=y, ES=ES, H=H)


@pytest.mark.parametrize("site", ["PP_ComputeParamStandardErrors", "PP_MStep", "PPLFP_ComputeParamStandardErrors",
                                  "PPLFP_MStep"])
def test_every_monte_carlo_state_draw_uses_the_covariance(site, monkeypatch) -> None:
    # Every Monte Carlo draw of the four routines goes through
    # _mc_state_draws (the x_k draws for the expectations and for the missing
    # information, and the x_0 draw), and the draws the routine actually uses
    # have covariance W_K (pooled over k with x_K = 0).
    import sys

    import nstat.decoding_algorithms as da
    from nstat.extras.matlab_rng import seeded_global_rng

    lfp_mod = sys.modules[PPLFP.__module__]  # the module (nstat.decoding re-exports the class under its name)
    real = da._mc_state_draws
    seen = []

    def spy(m, W, M, normal, **kw):
        out = real(m, W, M, normal, **kw)
        seen.append((np.asarray(W, dtype=float).copy(), out))
        return out

    monkeypatch.setattr(da, "_mc_state_draws", spy)
    monkeypatch.setattr(lfp_mod, "_mc_state_draws", spy)
    P = _draw_site_problem()
    dx, K, C = P["dx"], P["K"], P["C"]
    x0, Px0 = np.zeros(dx), _W3.copy()
    with seeded_global_rng(3):
        if site == "PP_ComputeParamStandardErrors":
            cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 1, 1, 0, 50)
            DecodingAlgorithms.PP_ComputeParamStandardErrors(
                P["dN"], P["x"], P["W_K"], 0.9 * np.eye(dx), 0.2 * _W3, x0, Px0, P["ES"], "poisson", P["mu"],
                P["beta"], np.array(0.0), [], P["H"], cons)
            expected_calls = 2 * K + 1
        elif site == "PP_MStep":
            cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0)
            DecodingAlgorithms.PP_MStep(P["dN"], P["x"], P["W_K"], x0, Px0, P["ES"], "poisson", P["mu"], P["beta"],
                                        np.array(0.0), [], P["H"], cons, "NewtonRaphson")
            expected_calls = K
        elif site == "PPLFP_ComputeParamStandardErrors":
            cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 1, 1, 0, 50, 0)
            PPLFP.PPLFP_ComputeParamStandardErrors(
                P["y"], P["dN"], P["x"], P["W_K"], 0.9 * np.eye(dx), 0.2 * _W3, np.ones((1, dx)), np.eye(1),
                np.zeros(1), x0, Px0, P["ES"], "poisson", P["mu"], P["beta"], np.array(0.0), [], P["H"], cons)
            expected_calls = 2 * K + 1
        else:
            cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 50, 0)
            PPLFP.PPLFP_MStep(P["dN"], P["y"], P["x"], P["W_K"], x0, Px0, P["ES"], "poisson", P["mu"], P["beta"],
                              np.array(0.0), [], P["H"], cons, "NewtonRaphson")
            expected_calls = K
    assert len(seen) == expected_calls
    assert all(np.array_equal(W, _W3) for W, _ in seen)
    pooled = np.concatenate([out for _, out in seen], axis=1)
    _assert_draw_covariance(pooled, _W3, np.zeros(dx))


# ---------------------------------------------------------------------------
# PP_EStep log-likelihood with a square history (nW == C; MATLAB C3)
# ---------------------------------------------------------------------------


def _pp_estep_gold_case(case):
    from pathlib import Path

    from scipy.io import loadmat

    g = loadmat(Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "pp_estep.mat")
    f = lambda k: np.asarray(g[f"{case}_{k}"], dtype=float)  # noqa: E731
    fit = str(np.asarray(g[f"{case}_fitType"]).reshape(-1)[0])
    return f("A"), f("Q"), f("dN"), f("mu"), f("beta"), fit, f("gamma"), f("HkAll"), f("x0"), f("Px0")


@pytest.mark.parametrize("case", ["c2", "c5", "c6"])
def test_pp_estep_invariant_to_an_appended_zero_history_window(case) -> None:
    # Appending an all-zero history window with a zero gamma row changes no
    # model term, so x_K, W_K and logll must be unchanged.  c2 (nW = 2, C = 3)
    # becomes square; the square c5 / c6 (nW = C = 3) become non-square.  The
    # old rows-based orientation test transposed the square slice in the logll.
    A, Q, dN, mu, beta, fit, gamma, HkAll, x0, Px0 = _pp_estep_gold_case(case)
    N, nW, C = HkAll.shape
    H2 = np.concatenate([HkAll, np.zeros((N, 1, C))], axis=1)
    g2 = np.vstack([gamma, np.zeros((1, C))])
    assert {nW, nW + 1} & {C}
    a = DecodingAlgorithms.PP_EStep(A, Q, dN, mu, beta, fit, gamma, HkAll, x0, Px0)
    b = DecodingAlgorithms.PP_EStep(A, Q, dN, mu, beta, fit, g2, H2, x0, Px0)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


# ---------------------------------------------------------------------------
# EM drivers: non-finite E-step log-likelihood (MATLAB bug 8 / B5) and the
# PP_EM standard-error exception swallow
# ---------------------------------------------------------------------------


def _pp_em_gold_args():
    A, Q, dN, mu, beta, fit, gamma, HkAll, x0, Px0 = _pp_estep_gold_case("c2")
    cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0, 0, 30)
    return (dN, A, Q, mu.reshape(-1), beta, fit, 0.001, gamma, [0.0, 0.001, 0.003], x0.reshape(-1), Px0, cons,
            "NewtonRaphson")


def _pplfp_em_gold_args():
    from pathlib import Path

    from scipy.io import loadmat

    fx = loadmat(Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "pplfp_EM.mat",
                 squeeze_me=True, struct_as_record=False)
    f = lambda k: np.asarray(fx[k], dtype=float)  # noqa: E731
    v = lambda k: np.asarray(fx[k], dtype=float).reshape(-1)  # noqa: E731
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, int(fx["mcIter"]), 0)
    return (f("y"), f("dN"), f("Ahat0"), f("Qhat0"), f("Chat0"), f("Rhat0"), v("alphahat0"), v("mu"), f("beta"),
            "poisson", 0.001, None, None, v("x0"), f("Px0"), cons, "NewtonRaphson")


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_em_stops_on_a_non_finite_estep_loglikelihood(family, bad, monkeypatch) -> None:
    # The second E-step reports a non-finite logll: EM must stop before the
    # M-step and return the best FINITE iterate (here the first: the initial
    # parameters).  It used to keep iterating (NaN) and np.argmax picked the
    # NaN / +Inf iterate (IC['llcomp'] = nan / inf).
    from nstat.extras.matlab_rng import seeded_global_rng

    owner = DecodingAlgorithms if family == "PP" else PPLFP
    estep_name = "PP_EStep" if family == "PP" else "PPLFP_EStep"
    mstep_name = "PP_MStep" if family == "PP" else "PPLFP_MStep"
    real_estep, real_mstep = getattr(owner, estep_name), getattr(owner, mstep_name)
    calls = {"E": 0, "M": 0, "ll": []}

    def estep(*a, **k):
        out = list(real_estep(*a, **k))
        calls["E"] += 1
        if calls["E"] == 2:
            out[2] = bad
        calls["ll"].append(out[2])
        return tuple(out)

    def mstep(*a, **k):
        calls["M"] += 1
        return real_mstep(*a, **k)

    monkeypatch.setattr(owner, estep_name, staticmethod(estep))
    monkeypatch.setattr(owner, mstep_name, staticmethod(mstep))
    with seeded_global_rng(3):
        if family == "PP":
            args = _pp_em_gold_args()
            out = DecodingAlgorithms.PP_EM(*args)
            IC, mu_out, mu_in, n_iter = out[9], out[4], args[3], out[12]
            assert n_iter == 1
        else:
            args = _pplfp_em_gold_args()
            out = PPLFP.PPLFP_EM(*args)
            IC, mu_out, mu_in = out[12], out[7], args[7]
    assert calls["E"] == 2 and calls["M"] == 1
    # IC.llcomp is the first (best finite) E-step's logll on the original
    # scale: the scaled-system logll plus the Jacobian of x_s = Tq x
    # ((K+1) log|det Tq|) and, for PPLFP, of y_s = Tr y (K log|det Tr|)
    # (MATLAB F10, which redesigned its own version of this test the same way).
    K = np.asarray(args[1 if family == "PPLFP" else 0]).shape[1]
    Q0 = np.asarray(args[3 if family == "PPLFP" else 2], dtype=float)
    jac = (K + 1) * np.log(abs(np.linalg.det(np.linalg.inv(np.linalg.cholesky(Q0)))))
    if family == "PPLFP":
        jac += K * np.log(abs(np.linalg.det(np.linalg.inv(np.linalg.cholesky(np.asarray(args[5], dtype=float))))))
    assert np.isfinite(IC["llcomp"])
    np.testing.assert_allclose(IC["llcomp"], calls["ll"][0] + jac, rtol=1e-12)
    np.testing.assert_array_equal(np.ravel(mu_out), np.ravel(mu_in))


def test_pp_em_does_not_swallow_standard_error_failures(monkeypatch) -> None:
    # MATLAB computes the SEs without a guard; the port used to wrap them in
    # `except Exception: pass` and return SE = Pvals = {} silently.
    class _SEFailure(RuntimeError):
        pass

    def boom(*_a, **_k):
        raise _SEFailure("SE failure must propagate")

    monkeypatch.setattr(DecodingAlgorithms, "PP_ComputeParamStandardErrors", staticmethod(boom))
    np.random.seed(0)
    args = list(_pp_em_gold_args())
    with pytest.raises(_SEFailure):
        DecodingAlgorithms.PP_EM(*args)


# ---------------------------------------------------------------------------
# History windows of the EM drivers: default rule and shared gamma (MATLAB
# B9, a457b54) and the delta time base of PP_EM (MATLAB C6)
# ---------------------------------------------------------------------------


def _em_problem(C=4, N=120, seed=21, delta=0.001):
    """Small EM problem, 150-200 Hz cells (enough spikes for finite history coefficients)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02], [-0.03, 0.96]])
    Q = np.diag([0.01, 0.02])
    x = np.zeros((2, N))
    prev = np.zeros(2)
    for k in range(N):
        prev = A @ prev + rng.multivariate_normal(np.zeros(2), Q)
        x[:, k] = prev
    mu = np.log(np.linspace(150, 200, C) * 0.001)
    beta = 0.7 * rng.standard_normal((2, C))
    dN = (rng.random((C, N)) < np.minimum(np.exp(mu[:, None] + beta.T @ x), 1)).astype(float)
    y = np.array([[1.0, 0.5]]) @ x + 0.1 * rng.standard_normal((1, N))
    return dict(A=A, Q=Q, mu=mu, beta=beta, dN=dN, y=y, delta=delta, C=C)


def _run_em(family, P, gamma, windowTimes, seed=5, method="NewtonRaphson"):
    from nstat.extras.matlab_rng import seeded_global_rng

    with seeded_global_rng(seed):
        if family == "PP":
            cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0, 0, 20)
            return DecodingAlgorithms.PP_EM(P["dN"], P["A"], P["Q"], P["mu"], P["beta"], "poisson", P["delta"],
                                            gamma, windowTimes, None, None, cons, method)
        cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 20, 0)
        return PPLFP.PPLFP_EM(P["y"], P["dN"], P["A"], P["Q"], np.array([[1.0, 0.5]]), 0.01 * np.eye(1),
                              np.zeros(1), P["mu"], P["beta"], "poisson", P["delta"], gamma, windowTimes, None, None,
                              cons, method)


def _assert_outputs_identical(a, b):
    assert len(a) == len(b)
    for i, (u, v) in enumerate(zip(a, b)):
        if isinstance(u, dict):
            assert sorted(u) == sorted(v), i
            for key in u:
                np.testing.assert_array_equal(np.asarray(u[key]), np.asarray(v[key]), err_msg=f"{i}.{key}")
        else:
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v), err_msg=str(i))


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
@pytest.mark.parametrize("gamma", [-0.05, np.array([[-0.6, -0.3, -0.5, -0.2], [-0.2, -0.1, -0.3, -0.05]])],
                         ids=["scalar", "2x4"])
def test_default_history_windows_equal_the_explicit_call(family, gamma) -> None:
    # windowTimes omitted: one window per history coefficient,
    # 0:delta:numWindows*delta, and a shared column (a scalar is one shared
    # window) expanded to every cell -- exactly the explicit call.  The former
    # rule built length(gamma)+1 windows (2 for a scalar, 5 for 2 x 4), whose
    # history no longer matched gamma.
    from nstat.core import _matlab_colon_exact

    P = _em_problem()
    g = np.asarray(gamma, dtype=float)
    W = 1 if g.ndim == 0 else g.shape[0]
    explicit_gamma = np.full((1, P["C"]), float(g)) if g.ndim == 0 else g
    a = _run_em(family, P, gamma, None)
    b = _run_em(family, P, explicit_gamma, _matlab_colon_exact(0.0, P["delta"], W * P["delta"]))
    _assert_outputs_identical(a, b)
    gammahat = a[6] if family == "PP" else a[9]
    assert np.shape(gammahat) == (W, P["C"])


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_zero_gamma_with_explicit_windows_is_not_expanded(family) -> None:
    # MATLAB a457b54: an all-zero gamma means "no history coefficients" (the
    # M-step skips it, the IC parameter count and the SE gamma block test
    # gamma == 0), so it is not expanded to the cells.
    P = _em_problem()
    wt = [0.0, (P["dN"].shape[1] - 1) * P["delta"]]
    out = _run_em(family, P, 0.0, wt)
    gammahat = out[6] if family == "PP" else out[9]
    assert np.asarray(gammahat).size == 1 and float(np.asarray(gammahat).reshape(-1)[0]) == 0.0


@pytest.mark.parametrize("method", ["NewtonRaphson", "GLM"])
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_time_base_equivalence(family, method) -> None:
    # The same spike matrix at delta = 2 ms with windows [0 4 10 20] ms is the
    # same per-bin model as at 1 ms with [0 2 5 10] ms (window w covers the
    # same bins), so every EM output must agree (MATLAB testTimeBaseEquivalence
    # PP / PPLFP, testGLMTimeBaseEquivalencePPLFP).  PP_EM's history used to
    # be built on a 1 kHz grid regardless of delta (at 2 ms it raised "HkAll
    # must align ..."); PPLFP_EM's was already on the delta grid (MATLAB R4b;
    # a pin); the GLM M-steps built their Trial on a hard-coded 1 ms grid
    # (MATLAB C6 / R4c) -- PP_MStep had no GLM branch, PPLFP_MStep's crashed.
    P1 = _em_problem(C=3, delta=0.001)
    P2 = dict(P1, delta=0.002)
    wt1, wt2 = [0.0, 0.002, 0.005, 0.010], [0.0, 0.004, 0.010, 0.020]
    assert np.array_equal(_compute_history_terms(P1["dN"], 0.001, wt1), _compute_history_terms(P1["dN"], 0.002, wt2))
    gamma = np.array([[-0.8, -0.6, -0.7], [-0.4, -0.3, -0.5], [-0.2, -0.1, -0.15]])
    a = _run_em(family, P1, gamma, wt1, method=method)
    b = _run_em(family, P2, gamma, wt2, method=method)
    if method == "NewtonRaphson":
        _assert_outputs_identical(a, b)
    else:  # the GLM fits of the two Trials agree to round-off
        for i, (u, v) in enumerate(zip(a, b)):
            if isinstance(u, dict):
                assert sorted(u) == sorted(v), i
                for key in u:
                    np.testing.assert_allclose(np.asarray(u[key]), np.asarray(v[key]), rtol=1e-8, atol=1e-12,
                                               err_msg=f"{i}.{key}")
            else:
                np.testing.assert_allclose(np.asarray(u, dtype=float), np.asarray(v, dtype=float), rtol=1e-8,
                                           atol=1e-12, err_msg=str(i))


# ---------------------------------------------------------------------------
# Decoders: a shared numWindows x 1 gamma column (MATLAB #20 / B2 / B3)
# ---------------------------------------------------------------------------


_SHARED_CALLERS = [
    "PPDecodeFilterLinear", "PP_fixedIntervalSmoother", "PPDecode_updateLinear", "PPHybridFilterLinear",
    "PP_EStep", "PPLFP_Decode_update", "PPLFP_DecodeLinear", "PPLFP_fixedIntervalSmoother", "PPLFP_EStep",
]


@pytest.mark.parametrize("W", [2, 3], ids=["W2", "W3_square"])
@pytest.mark.parametrize("caller", _SHARED_CALLERS)
def test_shared_gamma_column_equals_the_replicated_gamma(caller, W) -> None:
    # MATLAB's drivers expand a shared numWindows x 1 gamma with
    # if(size(gamma,2)==1 && C>1) gamma = repmat(gamma,1,C) (PPAF #20, PPHF B2,
    # PPLFP B3).  The port accepted a 1-D shared gamma but raised on MATLAB's
    # column (except PPLFP_DecodeLinear / _fixedIntervalSmoother).  Every
    # caller must now give exactly the replicated (W, C) result -- including
    # PP_EStep's log-likelihood, which reads gamma itself.
    P = _em_problem(C=3)
    dN, mu, beta, A, Q = P["dN"], P["mu"], P["beta"], P["A"], P["Q"]
    N = dN.shape[1]
    wt = list(np.arange(W + 1) * 0.001)
    H = _compute_history_terms(dN, 0.001, wt)
    Cm, R, y, Pi0 = np.array([[1.0, 0.5]]), 0.01 * np.eye(1), P["y"], 1e-3 * np.eye(2)
    run = {
        "PPDecodeFilterLinear": lambda g: DecodingAlgorithms.PPDecodeFilterLinear(
            A, Q, dN, mu, beta, "poisson", 0.001, g, wt, np.zeros(2), Pi0)[:4],
        "PP_fixedIntervalSmoother": lambda g: DecodingAlgorithms.PP_fixedIntervalSmoother(
            A, Q, dN, 2, mu, beta, "poisson", 0.001, g, wt, np.zeros(2), Pi0),
        "PPDecode_updateLinear": lambda g: DecodingAlgorithms.PPDecode_updateLinear(
            np.zeros(2), 0.1 * np.eye(2), dN, mu, beta, "poisson", g, H, N // 2),
        "PPHybridFilterLinear": lambda g: DecodingAlgorithms.PPHybridFilterLinear(
            [A, A], [Q, Q], np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([0.5, 0.5]), dN, mu, beta, "poisson",
            0.001, g, wt, [np.zeros(2)] * 2, [Pi0] * 2)[1:4],
        "PP_EStep": lambda g: DecodingAlgorithms.PP_EStep(A, Q, dN, mu, beta, "poisson", g, H, np.zeros(2), Pi0)[:3],
        "PPLFP_Decode_update": lambda g: PPLFP.PPLFP_Decode_update(
            np.zeros(2), 0.1 * np.eye(2), Cm, R, y[:, N // 2], np.zeros(1), dN, mu, beta, "poisson", g, H, N // 2),
        "PPLFP_DecodeLinear": lambda g: PPLFP.PPLFP_DecodeLinear(
            A, Q, Cm, R, y, np.zeros(1), dN, mu, beta, "poisson", 0.001, g, wt, np.zeros(2), Pi0, H),
        "PPLFP_fixedIntervalSmoother": lambda g: PPLFP.PPLFP_fixedIntervalSmoother(
            A, Q, Cm, R, y, np.zeros(1), dN, 2, mu, beta, "poisson", 0.001, g, wt, np.zeros(2), Pi0),
        "PPLFP_EStep": lambda g: PPLFP.PPLFP_EStep(
            A, Q, Cm, R, y, np.zeros(1), dN, mu, beta, "poisson", 0.001, g, H, np.zeros(2), Pi0)[:3],
    }[caller]
    col = -0.3 * np.arange(1, W + 1, dtype=float).reshape(W, 1)
    shared, replicated = run(col), run(np.tile(col, (1, 3)))
    for u, v in zip(shared, replicated):
        np.testing.assert_array_equal(np.asarray(u, dtype=float), np.asarray(v, dtype=float))


@pytest.mark.parametrize("which", ["DecodeLinear", "fixedIntervalSmoother"])
def test_pplfp_shared_gamma_column_reaches_every_cell(which) -> None:
    # MATLAB B3: PPLFP_DecodeLinear / PPLFP_fixedIntervalSmoother set only the
    # last cell's column of a shared gamma (post-loop c); the repaired MATLAB
    # repmats it, which is what the port does.  Shared == replicated, exactly.
    P = _em_problem(C=3)
    wt = [0.0, 0.002, 0.005, 0.010]
    col = np.array([[-0.8], [-0.4], [-0.2]])
    Cm, R = np.array([[1.0, 0.5]]), 0.01 * np.eye(1)
    HkAll = np.zeros((120, 3, 3))  # non-empty: PPLFP_DecodeLinear then rebuilds it from windowTimes

    def run(g):
        if which == "DecodeLinear":
            return PPLFP.PPLFP_DecodeLinear(P["A"], P["Q"], Cm, R, P["y"], np.zeros(1), P["dN"], P["mu"], P["beta"],
                                            "poisson", 0.001, g, wt, np.zeros(2), 1e-3 * np.eye(2), HkAll)
        return PPLFP.PPLFP_fixedIntervalSmoother(P["A"], P["Q"], Cm, R, P["y"], np.zeros(1), P["dN"], 2, P["mu"],
                                                 P["beta"], "poisson", 0.001, g, wt, np.zeros(2), 1e-3 * np.eye(2))

    shared, replicated = run(col), run(np.tile(col, (1, 3)))
    zero_history = run(np.zeros((3, 3)))
    for u, v in zip(shared, replicated):
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(shared[2], zero_history[2])  # the history does act


# ---------------------------------------------------------------------------
# Defaults (repaired MATLAB round 2 / B8): NewtonRaphson M-step, x0 / Px0 not
# estimated -- the Px0 estimator collapses Px0 and drives the logll to +Inf
# ---------------------------------------------------------------------------


def test_new_constraint_defaults() -> None:
    import warnings

    pp = DecodingAlgorithms.PP_EMCreateConstraints()
    assert [pp[k] for k in ("EstimateA", "AhatDiag", "QhatDiag", "QhatIsotropic", "Estimatex0", "EstimatePx0",
                            "Px0Isotropic", "mcIter", "EnableIkeda")] == [1, 0, 1, 0, 0, 0, 0, 1000, 0]
    lfp = PPLFP.PPLFP_EMCreateConstraints()
    assert [lfp[k] for k in ("EstimateA", "AhatDiag", "QhatDiag", "QhatIsotropic", "RhatDiag", "RhatIsotropic",
                             "Estimatex0", "EstimatePx0", "Px0Isotropic", "mcIter", "EnableIkeda")] == \
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 1000, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert DecodingAlgorithms.mPPCO_EMCreateConstraints() == lfp


def test_mstep_defaults_equal_the_explicit_defaults() -> None:
    # Omitted constraints / MstepMethod resolve to PP_/PPLFP_EMCreateConstraints()
    # and 'NewtonRaphson' (PPLFP_MStep used to default to the GLM branch).
    P = _mstep_problem("poisson")
    args_pp = (P["dN"], P["x"], P["W_K"], np.zeros(2), 1e-9 * np.eye(2), P["ES"], "poisson", P["mu"], P["beta"],
               P["gamma"], P["wt"], P["HkAll"])
    np.random.seed(9)
    a = DecodingAlgorithms.PP_MStep(*args_pp)
    np.random.seed(9)
    b = DecodingAlgorithms.PP_MStep(*args_pp, DecodingAlgorithms.PP_EMCreateConstraints(), "NewtonRaphson")
    _assert_outputs_identical(a, b)
    from nstat.extras.matlab_rng import seeded_global_rng

    args_lfp = (P["dN"], P["y"], P["x"], P["W_K"], np.zeros(2), 1e-9 * np.eye(2), P["ES"], "poisson", P["mu"],
                P["beta"], P["gamma"], P["wt"], P["HkAll"])
    with seeded_global_rng(9):
        a = PPLFP.PPLFP_MStep(*args_lfp)
    with seeded_global_rng(9):
        b = PPLFP.PPLFP_MStep(*args_lfp, PPLFP.PPLFP_EMCreateConstraints(), "NewtonRaphson")
    _assert_outputs_identical(a, b)


def _bare_problem():
    """MATLAB testPPLFPEMCorrectness.makeProblem('poisson', false, 600) analogue."""
    rng = np.random.default_rng(22)
    N = 600
    A, Q = 0.98 * np.eye(2), 0.01 * np.eye(2)
    x = np.zeros((2, N))
    prev = np.zeros(2)
    for k in range(N):
        prev = A @ prev + rng.multivariate_normal(np.zeros(2), Q)
        x[:, k] = prev
    mu = np.log(40 * 0.001) * np.ones(4)
    beta = np.array([[1.0, -0.5], [0.3, 0.8], [-0.7, 0.4], [0.6, 0.6]]).T
    dN = (rng.random((4, N)) < np.minimum(np.exp(mu[:, None] + beta.T @ x), 1)).astype(float)
    Cm, alpha, R = np.array([[1.0, 0.5], [-0.3, 1.0]]), np.array([0.1, -0.1]), 0.05 * np.eye(2)
    y = Cm @ x + alpha[:, None] + rng.multivariate_normal(np.zeros(2), R, size=N).T
    return dict(A=A, Q=Q, mu=mu, beta=beta, dN=dN, y=y, Cm=Cm, alpha=alpha, R=R)


def _record_ll(monkeypatch, owner, name):
    real = getattr(owner, name)
    trace: list = []

    def wrapped(*a, **k):
        out = real(*a, **k)
        trace.append(out[2])
        return out

    monkeypatch.setattr(owner, name, staticmethod(wrapped))
    return trace


def _assert_converging(trace):
    ll = np.asarray(trace, dtype=float)
    assert ll.size > 2 and np.all(np.isfinite(ll))
    assert ll[1] > ll[0]
    assert np.all(np.diff(ll[:-1]) >= 0)


def test_bare_default_pp_em_converges(monkeypatch) -> None:
    # PP_EM(dN, A, Q, mu, beta, 'poisson', delta) with every other argument
    # defaulted.  (The SE pass, default mcIter = 1000, is stubbed for speed --
    # MATLAB's own test requests 10 outputs, which skips it.)
    from nstat.extras.matlab_rng import seeded_global_rng

    P = _bare_problem()
    trace = _record_ll(monkeypatch, DecodingAlgorithms, "PP_EStep")
    monkeypatch.setattr(DecodingAlgorithms, "PP_ComputeParamStandardErrors", staticmethod(lambda *a, **k: ({}, {}, 0)))
    with seeded_global_rng(42):
        out = DecodingAlgorithms.PP_EM(P["dN"], P["A"], P["Q"], P["mu"], P["beta"], "poisson", 0.001)
    _assert_converging(trace)
    assert out[12] == len(trace) and out[12] > 2
    assert np.max(np.abs(out[4] - P["mu"])) < 0.5
    assert np.max(np.abs(out[5] - P["beta"])) < 0.8
    # x0 / Px0 are not estimated by default (they stay at 0 and 1e-9 I).
    assert np.array_equal(out[7], np.zeros(2))
    np.testing.assert_allclose(out[8], 1e-9 * np.eye(2), rtol=1e-10, atol=1e-24)


def test_bare_default_pplfp_em_and_mppco_em_converge(monkeypatch) -> None:
    # PPLFP_EM(y, dN, A, Q, C, R, alpha, mu, beta): the default M-step used to
    # be the (broken) GLM branch; the mPPCO_EM alias forwards the bare call
    # unchanged.
    import warnings

    from nstat.extras.matlab_rng import seeded_global_rng

    P = _bare_problem()
    args = (P["y"], P["dN"], P["A"], P["Q"], P["Cm"], P["R"], P["alpha"], P["mu"], P["beta"])
    monkeypatch.setattr(PPLFP, "PPLFP_ComputeParamStandardErrors", staticmethod(lambda *a, **k: ({}, {}, 0)))
    with seeded_global_rng(42):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            alias = DecodingAlgorithms.mPPCO_EM(*args)
    trace = _record_ll(monkeypatch, PPLFP, "PPLFP_EStep")
    with seeded_global_rng(42):
        out = PPLFP.PPLFP_EM(*args)
    _assert_outputs_identical(alias, out)
    _assert_converging(trace)
    assert np.max(np.abs(out[7] - P["mu"])) < 0.5
    assert np.max(np.abs(out[8] - P["beta"])) < 0.8
    assert np.max(np.abs(out[4] - P["Cm"])) < 0.1
    assert np.max(np.abs(np.ravel(out[6]) - P["alpha"])) < 0.1
    assert np.array_equal(np.ravel(out[10]), np.zeros(2))
    np.testing.assert_allclose(out[11], 1e-9 * np.eye(2), rtol=1e-10, atol=1e-24)


@pytest.mark.parametrize("gamma", [-0.3, np.array([[-0.3]])], ids=["scalar", "1x1"])
def test_single_cell_single_window_em_runs_with_one_gamma_parameter(gamma) -> None:
    # C == 1, W == 1: one history coefficient.  Both EMs compute their SEs with
    # it (the MATLAB SE routines left that parameter count unassigned; a 1 x 1
    # gamma raised in PP_MStep / the SE routine and a scalar in PPLFP_MStep,
    # and PPLFP_EM's SE pass never ran).
    P = _em_problem(C=1)
    for wt in (None, [0.0, 0.001]):
        for family, ig, ise, ip in (("PP", 6, 10, 11), ("PPLFP", 9, 13, 14)):
            out = _run_em(family, P, gamma, wt)
            assert np.size(out[ig]) == 1 and np.shape(out[ig]) == np.shape(gamma)
            assert np.size(out[ise]["gamma"]) == 1 and np.all(np.isfinite(out[ise]["gamma"])), family
            assert np.size(out[ip]["gamma"]) == 1 and np.all(np.isfinite(out[ip]["gamma"])), family


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
@pytest.mark.parametrize("gamma", [-0.3, np.array([[-0.3]])], ids=["scalar", "1x1"])
def test_single_gamma_mstep_keeps_the_callers_shape(family, gamma) -> None:
    # The M-step itself (not just the EM's selected iterate) returns one
    # estimated coefficient in the shape it was given (PP_MStep returned a
    # scalar as shape (1,)).
    P = _mstep_problem("poisson", C=1, nW=1)
    out = _run_pp_mstep(P, "poisson", gamma=gamma) if family == "PP" else _run_pplfp_mstep(P, "poisson", gamma=gamma)
    g_out = np.asarray(out[4] if family == "PP" else out[7])
    assert g_out.shape == np.shape(gamma)
    assert np.isfinite(g_out).all() and float(g_out.reshape(-1)[0]) != -0.3  # estimated


def test_pplfp_em_returns_standard_errors_from_the_original_observations(monkeypatch) -> None:
    # PPLFP_EM computes its SEs (it unpacked the routine's three outputs into
    # two inside `except Exception: pass`, so SE = Pvals = {} always).  The SE
    # routine receives the ORIGINAL observations y (MATLAB F8, fix/pp-em
    # bac99f9): it used to get the whitened y = Tr*y (Tr = inv(chol(R0)))
    # together with the unscaled C / alpha / R.
    from nstat.extras.matlab_rng import seeded_global_rng

    args = _pplfp_em_gold_args()
    seen = {}
    real = PPLFP.PPLFP_ComputeParamStandardErrors

    def spy(*a, **k):
        seen["y"] = np.asarray(a[0], dtype=float)
        return real(*a, **k)

    monkeypatch.setattr(PPLFP, "PPLFP_ComputeParamStandardErrors", staticmethod(spy))
    with seeded_global_rng(42):
        out = PPLFP.PPLFP_EM(*args)
    SE, Pvals = out[13], out[14]
    assert sorted(SE) == sorted(Pvals) == ["A", "C", "Q", "R", "alpha", "beta", "mu"]  # the MATLAB gold's fields
    for d in (SE, Pvals):
        for key, value in d.items():
            assert np.all(np.isfinite(np.asarray(value, dtype=float))), key
    np.testing.assert_array_equal(seen["y"], args[0])


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_default_windows_read_a_2d_row_as_the_shared_column(family) -> None:
    # MATLAB (default-window branch): if(isrow(gamma) && numel(gamma)~=C)
    # gamma = gamma(:).  A (1, 3) row -- what scipy.io.loadmat returns for a
    # MATLAB row vector -- with C = 4 is 3 shared windows, expanded to the
    # cells: exactly the explicit 3-window call with the replicated gamma.
    # It raised in both drivers.
    from nstat.core import _matlab_colon_exact

    P = _em_problem(C=4)
    row = np.array([[-0.6, -0.3, -0.15]])
    a = _run_em(family, P, row, None)
    b = _run_em(family, P, np.tile(row.reshape(3, 1), (1, 4)), _matlab_colon_exact(0.0, 0.001, 3 * 0.001))
    _assert_outputs_identical(a, b)
    assert np.shape(a[6] if family == "PP" else a[9]) == (3, 4)


# ---------------------------------------------------------------------------
# EM whitening with a non-diagonal Q0 / R0 (MATLAB G1, repaired after
# fix/pp-em @ 8dbd0e4: Tq = inv(chol(Q0, 'lower')), Tr = inv(chol(R0, 'lower')))
# ---------------------------------------------------------------------------


def _nondiag_problem(N=400, C=3):
    """The reviewer's probe4 problem: dx = 2, non-diagonal state and observation noise."""
    rng = np.random.default_rng(7)
    delta, dx = 0.001, 2
    A = np.array([[0.95, 0.10], [-0.05, 0.90]])
    Q = np.array([[0.010, 0.006], [0.006, 0.020]])
    x = np.zeros((dx, N))
    prev = np.zeros(dx)
    for k in range(N):
        prev = A @ prev + np.linalg.cholesky(Q) @ rng.standard_normal(dx)
        x[:, k] = prev
    mu = np.log(40 * delta) * np.ones(C)
    beta = np.array([[1.0, -0.6, 0.8], [0.4, 0.9, -0.7]])[:, :C]
    dN = (rng.random((C, N)) < np.minimum(np.exp(mu[:, None] + beta.T @ x), 1)).astype(float)
    Cm = np.array([[1.0, 0.5], [-0.3, 1.0]])
    R = np.array([[0.05, 0.02], [0.02, 0.08]])
    alpha = np.array([0.1, -0.1])
    y = Cm @ x + alpha[:, None] + np.linalg.cholesky(R) @ rng.standard_normal((2, N))
    x0, Px0 = np.array([0.05, -0.02]), np.array([[2e-3, 5e-4], [5e-4, 1e-3]])
    return dict(A=A, Q=Q, mu=mu, beta=beta, dN=dN, Cm=Cm, R=R, alpha=alpha, y=y, x0=x0, Px0=Px0, delta=delta)


def _run_nondiag_em(family, P, monkeypatch, mcIter=20, **scale):
    """PP_EM / PPLFP_EM with the default diagonal-Q (R) constraints; returns (outputs, E-step calls)."""
    from nstat.extras.matlab_rng import seeded_global_rng

    owner = DecodingAlgorithms if family == "PP" else PPLFP
    name = "PP_EStep" if family == "PP" else "PPLFP_EStep"
    real = getattr(owner, name)
    calls = []

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append((a, out))
        return out

    monkeypatch.setattr(owner, name, staticmethod(spy))
    with seeded_global_rng(42):
        if family == "PP":
            cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0, 0, mcIter)
            out = DecodingAlgorithms.PP_EM(P["dN"], P["A"], P["Q"], P["mu"], P["beta"], "poisson", P["delta"], None,
                                           None, P["x0"], P["Px0"], cons)
        else:
            cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, mcIter, 0)
            out = PPLFP.PPLFP_EM(P["y"], P["dN"], P["A"], P["Q"], P["Cm"], P["R"], P["alpha"], P["mu"], P["beta"],
                                 "poisson", P["delta"], None, None, P["x0"], P["Px0"], cons)
    monkeypatch.setattr(owner, name, staticmethod(real))
    return out, calls


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_whitening_maps_a_nondiagonal_noise_covariance_to_the_identity(family, monkeypatch) -> None:
    # The scaled system's initial Q (and R) must be the identity.  With the
    # former Tq = inv(chol(Q0)) -- the upper factor -- Tq*Q0*Tq' != I for a
    # non-diagonal Q0, the default diagonal-Q constraint acted on a mixed
    # parameterisation, the first M-step lowered the log-likelihood and EM
    # returned the initial parameters (MATLAB reviewer probe4: 2 iterations,
    # |Ahat - A0| ~ 1e-17).
    P = _nondiag_problem()
    out, calls = _run_nondiag_em(family, P, monkeypatch)
    first = calls[0][0]
    np.testing.assert_allclose(first[1], np.eye(2), atol=1e-12)  # Q
    if family == "PPLFP":
        np.testing.assert_allclose(first[3], np.eye(2), atol=1e-12)  # R
    ll = [c[1][2] for c in calls]
    assert len(ll) > 2 and ll[1] > ll[0], ll
    Ahat = out[2]
    assert np.max(np.abs(Ahat - P["A"])) > 1e-4
    # With a non-diagonal Q0, QhatDiag = 1 means "diagonal in the Q0-whitened
    # frame" (MATLAB G1, a recorded open design question): Tq Qhat Tq' is
    # diagonal for the returned Qhat (and Tr Rhat Tr' for RhatDiag = 1).
    Tq = np.linalg.inv(np.linalg.cholesky(P["Q"]))
    Qw = Tq @ out[3] @ Tq.T
    assert abs(Qw[0, 1]) <= 1e-12 * np.max(np.abs(Qw)) and abs(Qw[1, 0]) <= 1e-12 * np.max(np.abs(Qw))
    assert abs(out[3][0, 1]) > 1e-6  # ... and not diagonal in the original frame
    if family == "PPLFP":
        Tr = np.linalg.inv(np.linalg.cholesky(P["R"]))
        Rw = Tr @ out[5] @ Tr.T
        assert abs(Rw[0, 1]) <= 1e-12 * np.max(np.abs(Rw)) and abs(Rw[1, 0]) <= 1e-12 * np.max(np.abs(Rw))


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_maps_covariances_back_with_the_transposed_factor(family, monkeypatch) -> None:
    # The estimates are mapped back as MATLAB (T\S)/T' = T^-1 S T^-T.
    # PPLFP_EM computed T^-1 S T^-1 (solve(T.T, X.T).T), the same only for a
    # symmetric T (diagonal Q0 / R0): with a non-diagonal Q0 / R0 it returned
    # non-symmetric Qhat, Rhat and WKFinal.  (PP_EM was already right: pin.)
    P = _nondiag_problem()
    out, _ = _run_nondiag_em(family, P, monkeypatch)
    WK = out[1]
    mats = {"Qhat": out[3], "WKFinal": WK}
    mats.update({"Px0hat": out[8]} if family == "PP" else {"Rhat": out[5], "Px0hat": out[11]})
    for key, M in mats.items():
        M = np.asarray(M, dtype=float)
        MT = M.transpose(1, 0, 2) if M.ndim == 3 else M.T
        np.testing.assert_allclose(M, MT, rtol=0, atol=1e-12 * np.max(np.abs(M)), err_msg=key)
    # Px0 is not estimated: it comes back exactly as passed (up to round-off).
    np.testing.assert_allclose(mats["Px0hat"], P["Px0"], rtol=1e-12)


# ---------------------------------------------------------------------------
# Standard errors of the EM drivers on one scale (MATLAB F8, fix/pp-em
# bac99f9), with the dx = 2 non-diagonal Q0 / R0 cases that pin the
# (Tq\S)/Tq' orientation (MATLAB G3; at dx = 1 a transposition passes)
# ---------------------------------------------------------------------------


def _f8_problem():
    """MATLAB's F8 problem shape: dx = 1, Q0 = 0.01 (Tq = 10), R0 = diag(0.05, 0.08), 3 poisson cells."""
    rng = np.random.default_rng(3)
    N, C = 400, 3
    x = np.zeros((1, N))
    prev = 0.0
    for k in range(N):
        prev = 0.97 * prev + 0.1 * rng.standard_normal()
        x[0, k] = prev
    mu = np.log(40e-3) * np.ones(C)
    beta = np.array([[1.0, -0.8, 0.6]])
    dN = (rng.random((C, N)) < np.minimum(np.exp(mu[:, None] + beta.T @ x), 1)).astype(float)
    Cm, R, alpha = np.array([[1.0], [-0.5]]), np.diag([0.05, 0.08]), np.array([0.1, -0.1])
    y = Cm @ x + alpha[:, None] + np.sqrt(np.diag(R))[:, None] * rng.standard_normal((2, N))
    return dict(A=np.array([[0.97]]), Q=np.array([[0.01]]), mu=mu, beta=beta, dN=dN, Cm=Cm, R=R, alpha=alpha, y=y,
                x0=np.zeros(1), Px0=1e-3 * np.eye(1), delta=0.001)


_SCALE_PROBLEMS = {"dx1": _f8_problem, "dx2_nondiag": _nondiag_problem}


def _run_scaled_em(family, P, t=1.0, s=1.0, mcIter=50):
    """EM on the problem with the state rescaled by t (x -> t x) and the observations by s (y -> s y)."""
    from nstat.extras.matlab_rng import seeded_global_rng

    with seeded_global_rng(42):
        if family == "PP":
            cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0, 0, mcIter)
            return DecodingAlgorithms.PP_EM(P["dN"], P["A"], t * t * P["Q"], P["mu"], P["beta"] / t, "poisson",
                                            P["delta"], None, None, t * P["x0"], t * t * P["Px0"], cons)
        cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, mcIter, 0)
        return PPLFP.PPLFP_EM(s * P["y"], P["dN"], P["A"], t * t * P["Q"], s * P["Cm"] / t, s * s * P["R"],
                              s * P["alpha"], P["mu"], P["beta"] / t, "poisson", P["delta"], None, None, t * P["x0"],
                              t * t * P["Px0"], cons)


@pytest.mark.parametrize("problem", sorted(_SCALE_PROBLEMS))
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_standard_errors_receive_the_original_scale_inputs(family, problem, monkeypatch) -> None:
    # The SE routine reads ES.Sxkm1xkm1 (A information), ES.Sxkxk (C
    # information, PPLFP) and y (PPLFP alpha / C / R scores).  With every
    # estimate on the original scale they must be the original-scale values:
    # exactly what the E-step returns when run in the original coordinates at
    # the returned estimates (the best iterate's parameters), and the input y.
    # They used to be the scaled system's sums (and the scaled y).
    P = _SCALE_PROBLEMS[problem]()
    owner = DecodingAlgorithms if family == "PP" else PPLFP
    name = "PP_ComputeParamStandardErrors" if family == "PP" else "PPLFP_ComputeParamStandardErrors"
    real = getattr(owner, name)
    seen = {}

    def spy(*a, **k):
        seen["args"] = a
        return real(*a, **k)

    monkeypatch.setattr(owner, name, staticmethod(spy))
    out = _run_scaled_em(family, P)
    a = seen["args"]
    if family == "PP":
        ES, HkAll, gamma = a[7], a[13], a[11]
        Ahat, Qhat, mu, beta, x0, Px0 = out[2], out[3], out[4], out[5], out[7], out[8]
        ref = DecodingAlgorithms.PP_EStep(Ahat, Qhat, P["dN"], mu, beta, "poisson", gamma, HkAll, x0, Px0)[3]
        keys = ["Sxkm1xkm1"]
    else:
        np.testing.assert_array_equal(a[0], P["y"])
        ES, HkAll, gamma = a[11], a[17], a[15]
        ref = PPLFP.PPLFP_EStep(out[2], out[3], out[4], out[5], P["y"], out[6], P["dN"], out[7], out[8], "poisson",
                                P["delta"], gamma, HkAll, out[10], out[11])[3]
        keys = ["Sxkm1xkm1", "Sxkxk"]
    for key in keys:
        np.testing.assert_allclose(ES[key], ref[key], rtol=1e-9, err_msg=key)


@pytest.mark.parametrize("problem", sorted(_SCALE_PROBLEMS))
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_standard_errors_are_equivariant_to_rescaling(family, problem, monkeypatch) -> None:
    # MATLAB testStandardErrorsInvariantToStateScaling /
    # ...ToObservationScaling.  x -> t x (t = 3: Q0 -> t^2 Q0, C -> C/t,
    # beta -> beta/t, x0 -> t x0, Px0 -> t^2 Px0) leaves the scaled problem,
    # the EM path and the draws unchanged, so SE.A, SE.mu, SE.alpha, SE.R are
    # unchanged, SE.Q scales by t^2 and SE.C, SE.beta by 1/t.  y -> s y
    # (s = 2: C, alpha -> s, R -> s^2) scales SE.C, SE.alpha by s and SE.R by
    # s^2.  The nearest-SPD projection of the inverse observed information
    # is not equivariant (a recorded MATLAB property, F8 / final defect 5):
    # it is disabled here, which makes the relations exact (measured
    # <= 2e-12; before F8 SE.A was off by up to a factor 3).
    import sys

    monkeypatch.setattr(DecodingAlgorithms, "_nearestSPD", staticmethod(lambda A: A))
    monkeypatch.setattr(sys.modules[PPLFP.__module__], "_nearest_spd", lambda A: A)
    P = _SCALE_PROBLEMS[problem]()
    ise = 10 if family == "PP" else 13
    base = _run_scaled_em(family, P)[ise]
    t = 3.0
    expect = {"A": 1.0, "Q": t * t, "C": 1 / t, "beta": 1 / t, "mu": 1.0, "alpha": 1.0, "R": 1.0}
    scaled = _run_scaled_em(family, P, t=t)[ise]
    assert sorted(scaled) == sorted(base)
    for key in base:
        np.testing.assert_allclose(scaled[key], expect[key] * np.asarray(base[key]), rtol=1e-9, atol=0,
                                   err_msg=f"state x{t}: SE.{key}")
    if family == "PPLFP":
        s = 2.0
        expect = {"A": 1.0, "Q": 1.0, "C": s, "beta": 1.0, "mu": 1.0, "alpha": s, "R": s * s}
        scaled = _run_scaled_em(family, P, s=s)[ise]
        for key in base:
            np.testing.assert_allclose(scaled[key], expect[key] * np.asarray(base[key]), rtol=1e-9, atol=0,
                                       err_msg=f"observation x{s}: SE.{key}")


# ---------------------------------------------------------------------------
# Information criteria on the original scale (MATLAB F10, fix/pp-em 8843a94),
# including the dx = 2 non-diagonal Q0 / R0 cases (MATLAB G3)
# ---------------------------------------------------------------------------


def _run_em_with_estep_spy(family, P, monkeypatch, **scale):
    owner = DecodingAlgorithms if family == "PP" else PPLFP
    name = "PP_EStep" if family == "PP" else "PPLFP_EStep"
    real = getattr(owner, name)
    first = {}

    def spy(*a, **k):
        first.setdefault("args", a)
        return real(*a, **k)

    monkeypatch.setattr(owner, name, staticmethod(spy))
    out = _run_scaled_em(family, P, **scale)
    monkeypatch.setattr(owner, name, staticmethod(real))
    return out, first["args"]


@pytest.mark.parametrize("problem", sorted(_SCALE_PROBLEMS))
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_information_criteria_match_the_estep_at_the_estimates(family, problem, monkeypatch) -> None:
    # MATLAB testInformationCriteriaMatchEStepAtEstimates: IC.llcomp is the
    # expected complete-data log-likelihood that the E-step returns in the
    # ORIGINAL coordinates at the returned estimates (the best iterate's
    # parameters), and IC.llobs its observation term -- sumPPll (PP), or
    # sumPPll + E[log p(y | x)] (PPLFP).  Both were scaled-system values.
    P = _SCALE_PROBLEMS[problem]()
    out, a = _run_em_with_estep_spy(family, P, monkeypatch)
    if family == "PP":
        IC = out[9]
        _, _, ll, ES = DecodingAlgorithms.PP_EStep(out[2], out[3], P["dN"], out[4], out[5], "poisson", out[6], a[7],
                                                   out[7], out[8])
        obs = ES["sumPPll"]
    else:
        IC = out[12]
        _, _, ll, ES = PPLFP.PPLFP_EStep(out[2], out[3], out[4], out[5], P["y"], out[6], P["dN"], out[7], out[8],
                                         "poisson", P["delta"], out[9], a[12], out[10], out[11])
        R, (dy, K) = out[5], P["y"].shape
        obs = (ES["sumPPll"] - dy * K / 2 * np.log(2 * np.pi) - K / 2 * np.log(np.linalg.det(R))
               - 0.5 * np.trace(np.linalg.solve(R, ES["sumYkTerms"])))
    np.testing.assert_allclose(IC["llcomp"], ll, rtol=1e-9)
    np.testing.assert_allclose(IC["llobs"], obs, rtol=1e-9)


@pytest.mark.parametrize("problem", sorted(_SCALE_PROBLEMS))
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_information_criteria_are_invariant_to_rescaling(family, problem) -> None:
    # MATLAB testInformationCriteriaInvariantToStateScaling: under x -> t x
    # llobs, AIC, AICc and BIC are unchanged and llcomp shifts by
    # -(K+1) dx log t (the state densities' Jacobian); under y -> s y (PPLFP)
    # llobs and llcomp shift by -K dy log s.  They used to depend on the units
    # (MATLAB: PP llobs 18680 -> 1342 at t = 3).
    P = _SCALE_PROBLEMS[problem]()
    iic = 9 if family == "PP" else 12
    base = _run_scaled_em(family, P)[iic]
    K, dx = P["dN"].shape[1], P["A"].shape[0]
    t = 3.0
    st = _run_scaled_em(family, P, t=t)[iic]
    for key in ("llobs", "AIC", "AICc", "BIC"):
        np.testing.assert_allclose(st[key], base[key], rtol=1e-8, err_msg=key)
    np.testing.assert_allclose(st["llcomp"], base["llcomp"] - (K + 1) * dx * np.log(t), rtol=1e-8)
    if family == "PPLFP":
        s, dy = 2.0, P["y"].shape[0]
        ob = _run_scaled_em(family, P, s=s)[iic]
        shift = -K * dy * np.log(s)
        np.testing.assert_allclose(ob["llobs"], base["llobs"] + shift, rtol=1e-8)
        np.testing.assert_allclose(ob["llcomp"], base["llcomp"] + shift, rtol=1e-8)


@pytest.mark.parametrize(("QhatDiag", "RhatDiag", "RhatIsotropic", "R_count"),
                         [(1, 0, 0, "full"), (0, 1, 0, "diag"), (1, 1, 0, "diag"), (1, 1, 1, "iso")])
def test_pplfp_em_counts_r_parameters_with_r_flags(QhatDiag, RhatDiag, RhatIsotropic, R_count, monkeypatch) -> None:
    # MATLAB F11 (testInformationCriteriaCountRWithRFlags): R's parameter
    # count follows R's own flags.  It tested QhatDiag / QhatIsotropic, so
    # (1, 0, 0) counted dy instead of dy^2 and (0, 1, 0) dy^2 instead of dy.
    # The count is recovered from IC: nTerms = (AIC + 2 llobs) / 2 and
    # (BIC + 2 llobs) / log K.
    from nstat.extras.matlab_rng import seeded_global_rng

    monkeypatch.setattr(PPLFP, "PPLFP_ComputeParamStandardErrors", staticmethod(lambda *a, **k: ({}, {}, 0)))
    P = _f8_problem()  # dx = 1, dy = 2, C = 3
    dx, (dy, K), C = 1, P["y"].shape, P["dN"].shape[0]
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, QhatDiag, 0, RhatDiag, RhatIsotropic, 0, 0, 0, 10, 0)
    with seeded_global_rng(42):
        IC = PPLFP.PPLFP_EM(P["y"], P["dN"], P["A"], P["Q"], P["Cm"], P["R"], P["alpha"], P["mu"], P["beta"],
                            "poisson", P["delta"], None, None, P["x0"], P["Px0"], cons)[12]
    n_R = {"full": dy * dy, "diag": dy, "iso": 1}[R_count]
    expected = dx * dx + (dx if QhatDiag else dx * dx) + dy * dx + n_R + dy + C + dx * C
    assert round((IC["AIC"] + 2 * IC["llobs"]) / 2) == expected
    np.testing.assert_allclose((IC["BIC"] + 2 * IC["llobs"]) / np.log(K), expected, rtol=1e-9)


@pytest.mark.parametrize("nW", [1, 2], ids=["scalar", "column"])
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_shared_gamma_column_standard_errors(family, nW) -> None:
    # MATLAB F12 (testSharedGammaColumnStandardErrors): with 4 cells, a shared
    # gamma -- a scalar for one window, a numWindows x 1 column for two -- is
    # expanded per cell at the SE routine's entry, so SE, Pvals and nTerms are
    # exactly those of the expanded numWindows x 4 gamma.  The column raised
    # IndexError (gammahat[:, c]); the scalar returned a 1-D SE.gamma.
    P = _mle_problem("poisson", C=4, nW=nW, K=600)
    col = P["gamma"][:, :1]  # (nW, 1)
    shared = np.asarray(float(col[0, 0])) if nW == 1 else col
    expanded = np.tile(col, (1, 4))
    if family == "PP":
        got = _pp_se(dict(P, gamma_arg=shared))
        want = _pp_se(dict(P, gamma_arg=expanded))
    else:
        extra = _pplfp_mle_extra(P)
        got = _pplfp_se(dict(P, gamma_arg=shared), *extra)
        want = _pplfp_se(dict(P, gamma_arg=expanded), *extra)
    assert got[2] == want[2]
    for part in (0, 1):
        assert sorted(got[part]) == sorted(want[part])
        for key in want[part]:
            np.testing.assert_array_equal(np.asarray(got[part][key]), np.asarray(want[part][key]), err_msg=key)
    assert np.shape(got[0]["gamma"]) == (nW, 4)


@pytest.mark.parametrize("cfg", ["AhatDiag", "EstimateA0"])
def test_pp_em_standard_errors_honour_the_constraints(cfg, monkeypatch) -> None:
    # MATLAB G2 (reviewer probe5): PointProcessEM.PP_ComputeParamStandardErrors
    # tested nargin < 19 in a 15-input function, so it always replaced the
    # caller's constraints with PP_EMCreateConstraints() -- SE.A full with
    # AhatDiag = 1, SE.A reported with EstimateA = 0, mcIter always 1000.  The
    # port always used the constraints it is given (a pin): SE.A is diagonal
    # with AhatDiag = 1, absent with EstimateA = 0, and every Monte Carlo
    # draw uses the caller's mcIter.
    import nstat.decoding_algorithms as da
    from nstat.extras.matlab_rng import seeded_global_rng

    rng = np.random.default_rng(7)
    N, C = 400, 3
    A, Q = np.diag([0.95, 0.90]), np.diag([0.010, 0.020])
    x = np.zeros((2, N))
    prev = np.zeros(2)
    for k in range(N):
        prev = A @ prev + np.sqrt(np.diag(Q)) * rng.standard_normal(2)
        x[:, k] = prev
    mu = np.log(40e-3) * np.ones(C)
    beta = np.array([[1.0, -0.6, 0.8], [0.4, 0.9, -0.7]])
    dN = (rng.random((C, N)) < np.minimum(np.exp(mu[:, None] + beta.T @ x), 1)).astype(float)
    flags = (1, 1) if cfg == "AhatDiag" else (0, 0)
    cons = DecodingAlgorithms.PP_EMCreateConstraints(flags[0], flags[1], 1, 0, 0, 0, 0, 30)
    draws = []
    real = da._mc_state_draws

    def spy(m, W, M, normal, **kw):
        draws.append(M)
        return real(m, W, M, normal, **kw)

    monkeypatch.setattr(da, "_mc_state_draws", spy)
    with seeded_global_rng(42):
        out = DecodingAlgorithms.PP_EM(dN, A, Q, mu, beta, "poisson", 0.001, None, None, None, None, cons)
    Ahat, SE, Pvals = out[2], out[10], out[11]
    assert draws and set(draws) == {50, 30}  # the M-step's fixed McExp = 50; the SE pass uses mcIter
    if cfg == "AhatDiag":
        assert np.count_nonzero(Ahat - np.diag(np.diag(Ahat))) == 0
        assert SE["A"].shape == (2, 2) and np.count_nonzero(SE["A"] - np.diag(np.diag(SE["A"]))) == 0
        assert np.all(np.diag(SE["A"]) > 0) and "A" in Pvals
    else:
        assert "A" not in SE and "A" not in Pvals
        np.testing.assert_allclose(Ahat, A, rtol=1e-14, atol=0)  # A is held (scale / unscale round-off only)


# ---------------------------------------------------------------------------
# GLM M-step (MstepMethod = 'GLM'; MATLAB PP_MStep / PPLFP_MStep, repaired
# through fix/pp-em @ 8dbd0e4: written to the returned variables, mu / beta /
# gamma mapped BY LABEL (F3, R4a), keep-previous for unestimable coefficients
# (FitResSummary se >= 100) and for an empty label list (F1), delta time base
# (C6 / R4c)).  PP_MStep had no GLM branch (it ran Newton-Raphson for any
# MstepMethod); PPLFP_MStep's crashed on the getCoeffs() tuple.
# ---------------------------------------------------------------------------


def _glm_problem(C=3, dx=2, K=1500, seed=11, refractory=False, delta=0.001, wt=(0.0, 0.002, 0.005, 0.010)):
    """Smoothed means x_K and spikes from (mu, beta) (+ an optional hard 1-bin refractory period)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((dx, K))
    prev = np.zeros(dx)
    for k in range(K):
        prev = 0.98 * prev + 0.1 * rng.standard_normal(dx)
        x[:, k] = prev
    mu = np.log(np.linspace(40, 60, C) * delta)
    beta = rng.uniform(-1.0, 1.0, (dx, C))
    dN = np.zeros((C, K))
    for c in range(C):
        p = np.minimum(np.exp(mu[c] + beta[:, c] @ x), 1.0)
        u = rng.random(K)
        for k in range(K):
            if u[k] < p[k] and not (refractory and k > 0 and dN[c, k - 1] == 1):
                dN[c, k] = 1.0
    wt = np.asarray(wt, dtype=float)
    HkAll = _compute_history_terms(dN, delta, wt)
    W_K = np.tile((0.01 * np.eye(dx))[:, :, None], (1, 1, K))
    ES = dict(Sxkm1xkm1=x @ x.T, Sxkxkm1=x[:, 1:] @ x[:, :-1].T, Sxkm1xk=x[:, :-1] @ x[:, 1:].T, Sxkxk=x @ x.T,
              sumXkTerms=0.01 * K * np.eye(dx), Sxkyk=x @ x[:1].T, sumYkTerms=0.1 * K * np.eye(1))
    return dict(x=x, dN=dN, mu=mu, beta=beta, wt=wt, HkAll=HkAll, W_K=W_K, ES=ES, dx=dx, C=C, K=K, y=x[:1].copy(),
                delta=delta)


def _glm_mstep(family, P, mu, beta, gamma, wt="P", fit="poisson", delta=None):
    """One GLM M-step; returns (mu, beta, gamma)."""
    wt = P["wt"] if isinstance(wt, str) else wt
    delta = P["delta"] if delta is None else delta
    dx = P["dx"]
    if family == "PP":
        cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0)
        out = DecodingAlgorithms.PP_MStep(P["dN"], P["x"], P["W_K"], np.zeros(dx), 1e-9 * np.eye(dx), P["ES"], fit,
                                          mu, beta, gamma, wt, P["HkAll"], cons, "GLM", delta)
        return out[2], out[3], out[4]
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 50, 0)
    out = PPLFP.PPLFP_MStep(P["dN"], P["y"], P["x"], P["W_K"], np.zeros(dx), 1e-9 * np.eye(dx), P["ES"], fit, mu,
                            beta, gamma, wt, P["HkAll"], cons, "GLM", delta)
    return out[5], out[6], out[7]


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_mstep_rejects_an_unknown_mstep_method(family) -> None:
    # MATLAB runs Newton-Raphson for any MstepMethod other than 'GLM'; the
    # port used to ignore it in PP_MStep (no GLM branch at all).  Unknown
    # values now raise, naming the two allowed methods.
    P = _glm_problem(K=50)
    for bad in ("NoSuchMethod", "glm"):
        with pytest.raises(ValueError, match="'GLM' or 'NewtonRaphson'"):
            if family == "PP":
                DecodingAlgorithms.PP_MStep(P["dN"], P["x"], P["W_K"], np.zeros(2), 1e-9 * np.eye(2), P["ES"],
                                            "poisson", P["mu"], P["beta"], np.array(0.0), None, P["HkAll"], None, bad)
            else:
                PPLFP.PPLFP_MStep(P["dN"], P["y"], P["x"], P["W_K"], np.zeros(2), 1e-9 * np.eye(2), P["ES"],
                                  "poisson", P["mu"], P["beta"], np.array(0.0), None, P["HkAll"], None, bad)


def _f3_gold_case(name):
    """One f3 case of em_glm_mstep.mat: MATLAB's own by-label test inputs and outputs."""
    from pathlib import Path

    from scipy.io import loadmat

    g = loadmat(Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "em_glm_mstep.mat")
    return {k[len(name) + 1:]: g[k] for k in g if k.startswith(name + "_")}


@pytest.mark.parametrize("case", ["dx10_C2", "dx2_C1", "dx2_C3_drop2"])
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_glm_mstep_maps_coefficients_by_label(family, case) -> None:
    # MATLAB F3 (testGLMMStepMapsCoefficientsByLabel), on MATLAB's own
    # construction, captured from fix/pp-em @ aa88a2b: E-step output at
    # A = 0.95 I, Q = 0.01 I (rng(19)); one GLM M-step (no history) must equal
    # glmfit(x_K', dN(c,:)', 'poisson') per cell, mu from 'constant' and
    # beta(i, :) from 'v<i>' -- for dx = 10 ('v10' must not land in row 2),
    # for a single cell, and with x_K row 2 ~ 1e-5 noise (v2 not identifiable,
    # se >= 100: beta row 2 keeps its previous value).  With the isotropic
    # state model the PP smoothed means stay in span(beta), so [1 x_K'] is
    # rank-deficient for dx > C (also PPLFP dx = 10): glmfit drops the dependent
    # columns (b = 0, se = 0) and so must the port.  It returned garbage there
    # (PP dx = 2, C = 1: beta [33.0, -620.7] vs MATLAB [10.66, 0]) -- the former
    # version of this test simulated full-rank states and never reached it.
    g = _f3_gold_case(("f3pp_" if family == "PP" else "f3lfp_") + case)
    f = lambda k: np.asarray(g[k], dtype=float)  # noqa: E731
    dN = np.atleast_2d(f("dN"))
    C, N = dN.shape
    x_K = f("x_K")
    dx = x_K.shape[0]
    ES = {k[3:]: np.asarray(v, dtype=float) for k, v in g.items() if k.startswith("ES_")}
    mu_in, beta_in = f("mu").reshape(C), f("beta").reshape(dx, C)
    W_K = np.full((dx, dx, N), np.nan)  # the GLM branch never reads W_K
    H0 = np.zeros((N, 1, C))
    if family == "PP":
        out = DecodingAlgorithms.PP_MStep(dN, x_K, W_K, np.zeros(dx), 1e-9 * np.eye(dx), ES, "poisson", mu_in,
                                          beta_in, np.array(0.0), None, H0, None, "GLM")
        mu, beta = out[2], out[3]
    else:
        out = PPLFP.PPLFP_MStep(dN, f("y"), x_K, W_K, np.zeros(dx), 1e-9 * np.eye(dx), ES, "poisson", mu_in,
                                beta_in, np.array(0.0), None, H0, None, "GLM")
        mu, beta = out[5], out[6]
    B = f("glmfit_b")
    np.testing.assert_allclose(mu, f("muhat_new").reshape(C), rtol=1e-10, atol=1e-12, err_msg="mu vs MATLAB")
    np.testing.assert_allclose(beta, f("betahat_new").reshape(dx, C), rtol=1e-10, atol=1e-12,
                               err_msg="beta vs MATLAB")
    np.testing.assert_allclose(mu, B[0], rtol=1e-10, atol=1e-12, err_msg="mu vs glmfit")
    for i in range(dx):
        if case.endswith("drop2") and i == 1:
            np.testing.assert_array_equal(beta[i], beta_in[i])  # not identifiable: previous value kept
        else:
            np.testing.assert_allclose(beta[i], B[i + 1], rtol=1e-10, atol=1e-12, err_msg=f"beta row {i} vs glmfit")


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_glm_history_unestimable_window_keeps_previous(family) -> None:
    # MATLAB R4a (testGLMHistoryUnestimableWindowKeepsPrevious): 4 cells, a
    # hard 1-bin refractory period and windows [0 1 5 20] ms -- the (0, 1] ms
    # window is separated for every cell (se >= 100), so its row is returned
    # unchanged and the other rows are fitted (they used to be reshaped from a
    # shorter label list, which raised).
    P = _glm_problem(C=4, K=1500, refractory=True, wt=(0.0, 0.001, 0.005, 0.020))
    g0 = np.full((3, 4), -0.25)
    mu, beta, gamma = _glm_mstep(family, P, P["mu"], P["beta"], g0)
    assert gamma.shape == (3, 4)
    np.testing.assert_array_equal(gamma[0], g0[0])
    assert np.all(gamma[1:] != g0[1:]) and np.all(np.isfinite(gamma))


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_glm_history_all_windows_unestimable(family) -> None:
    # MATLAB F1 (testGLMHistoryAllWindowsUnestimable): one refractory window
    # [0 1] ms -- no history label is estimable for any cell, so gamma comes
    # back unchanged (MATLAB indexed an empty label list); mu / beta are fitted.
    P = _glm_problem(C=3, K=1500, refractory=True, wt=(0.0, 0.001))
    g0 = np.full((1, 3), -0.4)
    mu, beta, gamma = _glm_mstep(family, P, P["mu"], P["beta"], g0)
    np.testing.assert_array_equal(gamma, g0)
    assert np.all(mu != P["mu"]) and np.all(np.isfinite(beta))


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_glm_mstep_time_base_equivalence(family) -> None:
    # MATLAB R4c (testGLMTimeBaseEquivalence): the same spikes and smoothed
    # means at delta = 2 ms with windows [0 4 10 20] ms are the same per-bin
    # model as at 1 ms with [0 2 5 10] ms, so the GLM M-step outputs agree.
    # The Trial used to be built on a hard-coded 1 ms grid.
    P1 = _glm_problem(C=3, K=1500, wt=(0.0, 0.002, 0.005, 0.010))
    P2 = dict(P1, delta=0.002, wt=np.array([0.0, 0.004, 0.010, 0.020]))
    g0 = np.full((3, 3), -0.2)
    a = _glm_mstep(family, P1, P1["mu"], P1["beta"], g0)
    b = _glm_mstep(family, P2, P2["mu"], P2["beta"], g0)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_glm_em_returns_the_initial_gamma_when_no_window_is_estimable(family) -> None:
    # MATLAB F1: PP_EM(..., 'GLM') with one refractory window [0 1] ms returns
    # the initial gamma (no history label is ever estimable).
    P = _glm_problem(C=3, K=600, refractory=True, wt=(0.0, 0.001))
    E = dict(A=np.diag([0.98, 0.98]), Q=0.01 * np.eye(2), mu=P["mu"], beta=P["beta"], dN=P["dN"], y=P["y"],
             delta=0.001, C=3)
    g0 = np.full((1, 3), -0.4)
    out = _run_em(family, E, g0, [0.0, 0.001], method="GLM")
    np.testing.assert_array_equal(out[6] if family == "PP" else out[9], g0)


def _nc_problem(seed=6):
    """N == C == 6 (six time bins, six cells), two history windows."""
    rng = np.random.default_rng(seed)
    N, C, dx = 6, 6, 2
    x = 0.3 * rng.standard_normal((dx, N))
    dN = (rng.random((C, N)) < 0.4).astype(float)
    wt = np.array([0.0, 0.001, 0.002])
    HkAll = _compute_history_terms(dN, 0.001, wt)
    W_K = np.tile((0.02 * np.eye(dx))[:, :, None], (1, 1, N))
    ES = dict(Sxkm1xkm1=x @ x.T, Sxkxkm1=x[:, 1:] @ x[:, :-1].T, Sxkm1xk=x[:, :-1] @ x[:, 1:].T, Sxkxk=x @ x.T,
              sumXkTerms=0.02 * N * np.eye(dx), Sxkyk=x @ x[:1].T, sumYkTerms=0.1 * N * np.eye(1))
    return dict(x=x, dN=dN, wt=wt, HkAll=HkAll, W_K=W_K, ES=ES, mu=np.full(C, -0.5), beta=0.4 * rng.standard_normal((dx, C)),
                gamma=-0.3 * np.ones((2, C)), y=x[:1].copy())


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_newton_raphson_mstep_with_as_many_bins_as_cells(family) -> None:
    # MATLAB R4d (testMStepNumBinsEqualsNumCells): with N == numCells the
    # M-step's per-cell history slice HkAll(:, :, c) (N x numWindows) was
    # transposed ("if size(Hk,1)==numCells") and Hk(k, :) then read the wrong
    # entries.  Adding a 7th dummy cell (same E-step output, same Monte Carlo
    # draws) must leave cells 1..6 bit-identical.  PPLFP_MStep had the guard;
    # PP_MStep did not (a pin).
    from nstat.extras.matlab_rng import seeded_global_rng

    P = _nc_problem()
    Q = dict(P)
    Q["dN"] = np.vstack([P["dN"], np.zeros((1, 6))])
    Q["dN"][6, ::2] = 1.0
    Q["HkAll"] = _compute_history_terms(Q["dN"], 0.001, P["wt"])
    Q["mu"] = np.append(P["mu"], -0.5)
    Q["beta"] = np.column_stack([P["beta"], np.array([0.1, -0.1])])
    Q["gamma"] = np.column_stack([P["gamma"], -0.3 * np.ones(2)])
    outs = []
    for R in (P, Q):
        with seeded_global_rng(4):
            if family == "PP":
                cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0)
                o = DecodingAlgorithms.PP_MStep(R["dN"], R["x"], R["W_K"], np.zeros(2), 1e-9 * np.eye(2), R["ES"],
                                                "poisson", R["mu"], R["beta"], R["gamma"], R["wt"], R["HkAll"], cons,
                                                "NewtonRaphson")
                outs.append((o[2], o[3], o[4]))
            else:
                cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 50, 0)
                o = PPLFP.PPLFP_MStep(R["dN"], R["y"], R["x"], R["W_K"], np.zeros(2), 1e-9 * np.eye(2), R["ES"],
                                      "poisson", R["mu"], R["beta"], R["gamma"], R["wt"], R["HkAll"], cons,
                                      "NewtonRaphson")
                outs.append((o[5], o[6], o[7]))
    (mu6, b6, g6), (mu7, b7, g7) = outs
    np.testing.assert_array_equal(mu7[:6], mu6)
    np.testing.assert_array_equal(b7[:, :6], b6)
    np.testing.assert_array_equal(g7[:, :6], g6)


# (EstimateA, AhatDiag, QhatDiag, QhatIsotropic, Estimatex0, EstimatePx0, Px0Isotropic): MATLAB G2's eight
# constraint sets (testStandardErrorsHonourConstraints).
_G2_CONSTRAINTS = [(1, 0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (1, 1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0),
                   (1, 0, 1, 1, 0, 0, 0), (1, 0, 1, 0, 1, 0, 0), (1, 0, 1, 0, 0, 1, 0), (1, 0, 1, 0, 0, 1, 1)]


def _g2_expected(v, dx, C, extra=0):
    """MATLAB G2's count A + Q + Px0 + x0 + mu + beta (+ PPLFP's C, R, alpha) and SE fields."""
    EstimateA, AhatDiag, QhatDiag, QhatIso, Ex0, EPx0, Px0Iso = v
    n = (dx if AhatDiag else dx * dx) * EstimateA
    n += (1 if QhatIso else dx) if QhatDiag else dx * dx
    n += (1 if Px0Iso else dx) * EPx0 + dx * Ex0 + C + dx * C + extra
    fields = {"Q", "mu", "beta"} | ({"A"} if EstimateA else set()) | ({"x0"} if Ex0 else set())
    return n, fields | ({"Px0"} if EPx0 else set())


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_standard_errors_honour_every_constraint_set(family) -> None:
    # MATLAB G2 (testStandardErrorsHonourConstraints): once MATLAB's PP SE
    # routine honoured its constraints, EstimateA = 0, QhatIsotropic = 1 and
    # Px0Isotropic = 1 crashed on undefined N / dx (fixed by defining them up
    # front, as the PPLFP routine does).  Over the same eight constraint sets
    # nTerms and the SE fields follow the constraints, SE.A is diagonal under
    # AhatDiag = 1, and mcIter sets the Monte Carlo size.  Both Python
    # routines already ran every set (a pin; PPLFP for completeness, with
    # its default diagonal R).
    from nstat.extras.matlab_rng import seeded_global_rng

    P = _em_problem(C=4, N=300)
    A, Q, dN, mu, beta, y = P["A"], P["Q"], P["dN"], P["mu"], P["beta"], P["y"]
    C, dx = 4, 2
    H = np.zeros((300, 1, C))
    Cm, R = np.array([[1.0, 0.5]]), 0.01 * np.eye(1)
    x0, Px0 = np.zeros(dx), 1e-6 * np.eye(dx)
    if family == "PP":
        xK, WK, _, ES = DecodingAlgorithms.PP_EStep(A, Q, dN, mu, beta, "poisson", np.array(0.0), H, x0, Px0)
    else:
        xK, WK, _, ES = PPLFP.PPLFP_EStep(A, Q, Cm, R, y, np.zeros(1), dN, mu, beta, "poisson", 0.001, np.array(0.0),
                                          H, x0, Px0)

    def se(v, mcIter):
        with seeded_global_rng(1):
            if family == "PP":
                cons = DecodingAlgorithms.PP_EMCreateConstraints(*v, mcIter)
                return DecodingAlgorithms.PP_ComputeParamStandardErrors(dN, xK, WK, A, Q, x0, Px0, ES, "poisson", mu,
                                                                       beta, np.array(0.0), [], H, cons)
            cons = PPLFP.PPLFP_EMCreateConstraints(v[0], v[1], v[2], v[3], 1, 0, v[4], v[5], v[6], mcIter, 0)
            return PPLFP.PPLFP_ComputeParamStandardErrors(y, dN, xK, WK, A, Q, Cm, R, np.zeros(1), x0, Px0, ES,
                                                          "poisson", mu, beta, np.array(0.0), [], H, cons)

    extra, extra_fields = (0, set()) if family == "PP" else (dx + 1 + 1, {"C", "R", "alpha"})
    for v in _G2_CONSTRAINTS:
        SE, Pvals, nTerms = se(v, 20)
        n, fields = _g2_expected(v, dx, C, extra)
        assert nTerms == n, v
        assert set(SE) == set(Pvals) == fields | extra_fields, v
        if v[0] and v[1]:
            assert np.count_nonzero(SE["A"] - np.diag(np.diag(SE["A"]))) == 0, v
    assert not np.array_equal(se(_G2_CONSTRAINTS[0], 5)[0]["mu"], se(_G2_CONSTRAINTS[0], 7)[0]["mu"])


# ---------------------------------------------------------------------------
# Covariance information blocks of the SE routines (MATLAB H1: the MATLAB
# expression N/2*(Q)\e*e'/(Q) evaluates left to right as (N/2 Q)^-1 e e' Q^-1;
# the intended block is (N/2) Q^-1 e e' Q^-1 -- likewise R and 1/2 Px0).
# The port computes the intended form; this pins every reachable block.
# ---------------------------------------------------------------------------


def _vanishing_problem(K=1500, seed=5, correlated=False):
    """Known states (W_K = 1e-12 I) and every parameter at its complete-data MLE (missing information ~ 0)."""
    rng = np.random.default_rng(seed)
    dx, C = 2, 2
    L = np.linalg.cholesky(np.array([[0.02, 0.012], [0.012, 0.03]])) if correlated else np.sqrt(0.02) * np.eye(2)
    x = np.zeros((dx, K))
    prev = np.zeros(dx)
    for k in range(K):
        prev = 0.99 * prev + L @ rng.standard_normal(dx)
        x[:, k] = prev
    muT, betaT = np.log(0.05) * np.ones(C), np.array([[1.0, -0.8], [0.5, 0.7]])
    dN = (rng.random((C, K)) < np.minimum(np.exp(muT[:, None] + betaT.T @ x), 1)).astype(float)
    mu, beta = np.zeros(C), np.zeros((dx, C))
    for c in range(C):
        th = _cell_mle(np.column_stack([np.ones(K), x.T]), dN[c], "poisson", np.concatenate([[muT[c]], betaT[:, c]]))
        mu[c], beta[:, c] = th[0], th[1:]
    x0 = np.zeros(dx)
    xm1 = np.column_stack([x0, x[:, :-1]])
    Sx1, Sx10 = xm1 @ xm1.T, x @ xm1.T
    A = Sx10 @ np.linalg.inv(Sx1)
    sumX = x @ x.T - A @ Sx10.T - Sx10 @ A.T + A @ Sx1 @ A.T
    Cm, alphaT, Rt = np.array([[1.0, 0.5], [-0.3, 1.0]]), np.array([0.1, -0.1]), np.diag([0.05, 0.08])
    y = Cm @ x + alphaT[:, None] + np.sqrt(np.diag(Rt))[:, None] * rng.standard_normal((2, K))
    Z = np.vstack([x, np.ones((1, K))])
    CA = (y @ Z.T) @ np.linalg.inv(Z @ Z.T)
    Chat, alphahat = CA[:, :dx], CA[:, dx]
    res = y - Chat @ x - alphahat[:, None]
    ES = dict(Sxkm1xkm1=Sx1, Sxkxk=x @ x.T)
    return dict(x=x, dN=dN, mu=mu, beta=beta, A=A, Sx1=Sx1, sumX=sumX, y=y, Chat=Chat, alphahat=alphahat,
                resres=res @ res.T, ES=ES, K=K, dx=dx, WK=np.tile((1e-12 * np.eye(dx))[:, :, None], (1, 1, K)))


_COV_CASES = {
    # name: (EstimateA, QhatDiag, QhatIsotropic, RhatIsotropic, EstimatePx0, Px0Isotropic)
    "Qdiag": (1, 1, 0, 0, 0, 0),
    "Qiso": (1, 1, 1, 1, 0, 0),
    "EstimateA0": (0, 1, 0, 0, 0, 0),
    "Px0diag": (0, 1, 0, 0, 1, 0),
    "Px0iso": (0, 1, 0, 0, 1, 1),
}


@pytest.mark.parametrize("case", sorted(_COV_CASES))
@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_covariance_information_blocks_match_the_complete_information(family, case, monkeypatch) -> None:
    # Every parameter at its complete-data MLE given known states, so the
    # missing information vanishes and each SE is 1/sqrt(complete
    # information): SE.Q = q sqrt(2/K) (diagonal) or q sqrt(2/(K dx))
    # (isotropic), SE.R likewise with dy, SE.A(i, j) = sqrt(q_i inv(Sx1)_jj),
    # SE.C(i, j) = sqrt(r_i inv(Sxkxk)_jj), SE.alpha = sqrt(r / K).  For Px0
    # (not otherwise vanishing) the x0 draws are replaced by x0 +- sqrt(p),
    # where the Px0 score is exactly 0: SE.Px0 = p sqrt(2) (diagonal) or
    # p sqrt(2/dx) (isotropic).  MATLAB's block evaluated to (2/K) Q^-1 e e'
    # Q^-1 (SE.Q, SE.R about K/2 too large) and (2) P^-1 e e' P^-1 (SE.Px0
    # half) -- MATLAB H1; the port already had the intended form (a pin).
    # The A / C checks found two Python-only layout defects, fixed with this
    # test: PP's A information block was filled column-major against
    # row-major parameters (a row permutation of Q^-1 (x) Sxkm1xkm1, not
    # symmetric), and both routines unpacked SE.A / SE.C (and a full SE.Q)
    # column-major, i.e. transposed.
    import sys

    import nstat.decoding_algorithms as da
    from nstat.extras.matlab_rng import seeded_global_rng

    EstA, QDiag, QIso, RIso, EPx0, Px0Iso = _COV_CASES[case]
    P = _vanishing_problem()
    K, dx, sumX = P["K"], P["dx"], P["sumX"]
    Q = (np.trace(sumX) / (K * dx)) * np.eye(dx) if QIso else np.diag(np.diag(sumX)) / K
    R = (np.trace(P["resres"]) / (K * 2)) * np.eye(2) if RIso else np.diag(np.diag(P["resres"])) / K
    x0 = np.zeros(dx)
    Px0 = np.diag([0.3, 0.3]) if Px0Iso else (np.diag([0.3, 0.2]) if EPx0 else 1e-6 * np.eye(dx))
    real = da._mc_state_draws

    def draws(m, W, M, normal, **kw):
        out = real(m, W, M, normal, **kw)
        if EPx0 and np.array_equal(np.asarray(W), Px0) and np.array_equal(np.asarray(m).reshape(-1), x0):
            signs = np.where(np.arange(M) % 2 == 0, 1.0, -1.0)
            out = x0[:, None] + np.sqrt(np.diag(Px0))[:, None] * np.vstack([signs, signs[::-1]])
        return out

    monkeypatch.setattr(da, "_mc_state_draws", draws)
    monkeypatch.setattr(sys.modules[PPLFP.__module__], "_mc_state_draws", draws)
    with seeded_global_rng(1):
        if family == "PP":
            cons = DecodingAlgorithms.PP_EMCreateConstraints(EstA, 0, QDiag, QIso, 0, EPx0, Px0Iso, 20)
            SE = DecodingAlgorithms.PP_ComputeParamStandardErrors(
                P["dN"], P["x"], P["WK"], P["A"], Q, x0, Px0, P["ES"], "poisson", P["mu"], P["beta"],
                np.array(0.0), [], np.zeros((K, 1, 2)), cons)[0]
        else:
            cons = PPLFP.PPLFP_EMCreateConstraints(EstA, 0, QDiag, QIso, 1, RIso, 0, EPx0, Px0Iso, 20, 0)
            SE = PPLFP.PPLFP_ComputeParamStandardErrors(
                P["y"], P["dN"], P["x"], P["WK"], P["A"], Q, P["Chat"], R, P["alphahat"], x0, Px0, P["ES"],
                "poisson", P["mu"], P["beta"], np.array(0.0), [], np.zeros((K, 1, 2)), cons)[0]
    rtol = 1e-6
    q = np.diag(Q)
    se_q = q[:1] * np.sqrt(2 / (K * dx)) if QIso else q * np.sqrt(2 / K)
    if not EPx0:  # the x0 draws below move the Q score (x_1 - A x_0), so Q is checked without them
        np.testing.assert_allclose(np.diag(np.atleast_2d(SE["Q"]))[:se_q.size], se_q, rtol=rtol, err_msg="SE.Q")
    if EstA:
        np.testing.assert_allclose(SE["A"], np.sqrt(np.outer(q, np.diag(np.linalg.inv(P["Sx1"])))), rtol=rtol,
                                   err_msg="SE.A")
    if EPx0:
        p = np.diag(Px0)
        se_p = p[:1] * np.sqrt(2 / dx) if Px0Iso else p * np.sqrt(2)
        np.testing.assert_allclose(np.diag(np.atleast_2d(SE["Px0"]))[:se_p.size], se_p, rtol=rtol, err_msg="SE.Px0")
    if family == "PPLFP":
        r = np.diag(R)
        se_r = r[:1] * np.sqrt(2 / (K * 2)) if RIso else r * np.sqrt(2 / K)
        np.testing.assert_allclose(np.diag(np.atleast_2d(SE["R"]))[:se_r.size], se_r, rtol=rtol, err_msg="SE.R")
        Sxx_inv = np.linalg.inv(P["ES"]["Sxkxk"])
        np.testing.assert_allclose(SE["C"], np.sqrt(np.outer(r, np.diag(Sxx_inv))), rtol=rtol, err_msg="SE.C")
        np.testing.assert_allclose(np.ravel(SE["alpha"]), np.sqrt(r / K), rtol=rtol, err_msg="SE.alpha")


def test_pp_diagonal_a_information_with_a_full_q(monkeypatch) -> None:
    # AhatDiag = 1 with a non-diagonal Q (QhatDiag = 0): the diagonal-A
    # information is I(i, l) = (Q^-1)_il Sxkm1xkm1_il, MATLAB's
    # (Q^-1 e_l e_l' S) .* I evaluated left to right.  The port computed
    # Q^-1 e_l e_l' (S .* I), which keeps only i == l.  (A, Q) are at their
    # joint complete-data MLE (A diagonal, Q full), so SE.A = sqrt(diag(inv(I))).
    # A full Q is parameterised by all dx^2 entries (MATLAB too), so its
    # information block is singular; the nearest-SPD projection would spread
    # that over every SE, so it is disabled here (the A block of the
    # block-diagonal inverse is exact).
    from nstat.extras.matlab_rng import seeded_global_rng

    monkeypatch.setattr(DecodingAlgorithms, "_nearestSPD", staticmethod(lambda A: A))

    P = _vanishing_problem(correlated=True)
    K, dx, x = P["K"], P["dx"], P["x"]
    xm1 = np.column_stack([np.zeros(dx), x[:, :-1]])
    Sx1, Sx10 = xm1 @ xm1.T, x @ xm1.T
    a = np.diag(P["A"]).copy()
    for _ in range(500):  # the diagonal-A / full-Q MLE, by alternating the two closed forms
        r = x - np.diag(a) @ xm1
        Qi = np.linalg.inv(r @ r.T / K)
        a_new = np.linalg.solve(Qi * Sx1, np.diag(Qi @ Sx10))  # diag(Q^-1 (Sx10 - A Sx1)) = 0
        if np.max(np.abs(a_new - a)) < 1e-15:
            break
        a = a_new
    A = np.diag(a)
    r = x - A @ xm1
    Q = r @ r.T / K
    assert abs(Q[0, 1]) > 0.2 * np.sqrt(Q[0, 0] * Q[1, 1])
    with seeded_global_rng(1):
        cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 1, 0, 0, 0, 0, 0, 20)
        SE = DecodingAlgorithms.PP_ComputeParamStandardErrors(
            P["dN"], x, P["WK"], A, Q, np.zeros(dx), 1e-6 * np.eye(dx), dict(Sxkm1xkm1=Sx1), "poisson", P["mu"],
            P["beta"], np.array(0.0), [], np.zeros((K, 1, 2)), cons)[0]
    expected = np.sqrt(np.diag(np.linalg.inv(np.linalg.inv(Q) * Sx1)))
    np.testing.assert_allclose(np.diag(SE["A"]), expected, rtol=1e-8)
    assert np.count_nonzero(SE["A"] - np.diag(np.diag(SE["A"]))) == 0


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_covariance_information_forms_agree_for_one_state(family) -> None:
    # MATLAB H1 (testCovarianceInformationFormsAgree): with one state (and one
    # LFP channel) the diagonal, full and isotropic forms of Q (and R), and the
    # diagonal and isotropic forms of Px0, each describe a single parameter,
    # so every SE must agree.  MATLAB's mis-parenthesised blocks made the Px0
    # forms 2.2x apart; the port's forms agree (a pin).
    from nstat.extras.matlab_rng import seeded_global_rng

    rng = np.random.default_rng(12)
    K, C = 200, 2
    x = np.zeros((1, K))
    prev = 0.0
    for k in range(K):
        prev = 0.97 * prev + 0.15 * rng.standard_normal()
        x[0, k] = prev
    mu, beta = np.log(0.05) * np.ones(C), np.array([[0.8, -0.6]])
    dN = (rng.random((C, K)) < np.exp(mu[:, None] + beta.T @ x)).astype(float)
    y = 0.9 * x + 0.1 + 0.2 * rng.standard_normal((1, K))
    WK = np.full((1, 1, K), 1e-3)
    xm1 = np.column_stack([[0.0], x[:, :-1]])
    ES = dict(Sxkm1xkm1=xm1 @ xm1.T, Sxkxk=x @ x.T)
    A, Q, Cm, R, alpha = np.array([[0.97]]), np.array([[0.02]]), np.array([[0.9]]), np.array([[0.04]]), np.array([0.1])
    x0, Px0 = np.zeros(1), np.array([[0.05]])
    H = np.zeros((K, 1, C))

    def se(QDiag, QIso, RDiag, RIso, Px0Iso):
        with seeded_global_rng(5):
            if family == "PP":
                cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, QDiag, QIso, 1, 1, Px0Iso, 30)
                SE = DecodingAlgorithms.PP_ComputeParamStandardErrors(dN, x, WK, A, Q, x0, Px0, ES, "poisson", mu,
                                                                      beta, np.array(0.0), [], H, cons)[0]
            else:
                cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, QDiag, QIso, RDiag, RIso, 1, 1, Px0Iso, 30, 0)
                SE = PPLFP.PPLFP_ComputeParamStandardErrors(y, dN, x, WK, A, Q, Cm, R, alpha, x0, Px0, ES, "poisson",
                                                            mu, beta, np.array(0.0), [], H, cons)[0]
        return {k: np.asarray(v, dtype=float).reshape(-1) for k, v in SE.items()}

    ref = se(1, 0, 1, 0, 0)
    forms = {"Q full": se(0, 0, 1, 0, 0), "Q isotropic": se(1, 1, 1, 0, 0), "Px0 isotropic": se(1, 0, 1, 0, 1)}
    if family == "PPLFP":
        forms.update({"R full": se(1, 0, 0, 0, 0), "R isotropic": se(1, 0, 1, 1, 0)})
    for name, out in forms.items():
        assert sorted(out) == sorted(ref), name
        for key in ref:
            np.testing.assert_allclose(out[key], ref[key], rtol=1e-10, atol=0, err_msg=f"{name}: SE.{key}")



@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_glm_mstep_history_coefficients_need_windows(family) -> None:
    # MATLAB's GLM M-step with a nonzero gamma and windowTimes = [] errors
    # (History([], ...).computeHistory(...).getCov(1): MATLAB:nonLogicalConditional,
    # for a scalar, row, column or matrix gamma; checked at fix/pp-em aa88a2b).
    # The port returned gamma unchanged; it now raises too.  gamma = 0 (no
    # history) runs.
    P = _glm_problem(C=2, K=300)
    for g in (np.array(-0.2), np.array([[-0.2, -0.3]]), np.array([[-0.2], [-0.3]]), np.full((2, 2), -0.2)):
        with pytest.raises(ValueError, match="need history windows"):
            _glm_mstep(family, P, P["mu"], P["beta"], g, wt=None)
    mu, beta, gamma = _glm_mstep(family, P, P["mu"], P["beta"], np.array(0.0), wt=None)
    assert float(gamma) == 0.0 and np.all(np.isfinite(mu))
