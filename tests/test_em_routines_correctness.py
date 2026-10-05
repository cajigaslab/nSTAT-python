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
    assert np.isfinite(IC["llcomp"]) and IC["llcomp"] == calls["ll"][0]
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


def _run_em(family, P, gamma, windowTimes, seed=5):
    from nstat.extras.matlab_rng import seeded_global_rng

    with seeded_global_rng(seed):
        if family == "PP":
            cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0, 0, 20)
            return DecodingAlgorithms.PP_EM(P["dN"], P["A"], P["Q"], P["mu"], P["beta"], "poisson", P["delta"],
                                            gamma, windowTimes, None, None, cons, "NewtonRaphson")
        cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 20, 0)
        return PPLFP.PPLFP_EM(P["y"], P["dN"], P["A"], P["Q"], np.array([[1.0, 0.5]]), 0.01 * np.eye(1),
                              np.zeros(1), P["mu"], P["beta"], "poisson", P["delta"], gamma, windowTimes, None, None,
                              cons, "NewtonRaphson")


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


def test_pp_em_time_base_equivalence() -> None:
    # The same spike matrix at delta = 2 ms with windows [0 4 10 20] ms is the
    # same per-bin model as at 1 ms with [0 2 5 10] ms (window w covers the
    # same bins), so every PP_EM output must agree (MATLAB testTimeBaseEquivalence).
    # The history used to be built on a 1 kHz grid regardless of delta: at
    # 2 ms PP_EM raised "HkAll must align ..." in its first E-step.
    P1 = _em_problem(C=3, delta=0.001)
    P2 = dict(P1, delta=0.002)
    wt1, wt2 = [0.0, 0.002, 0.005, 0.010], [0.0, 0.004, 0.010, 0.020]
    assert np.array_equal(_compute_history_terms(P1["dN"], 0.001, wt1), _compute_history_terms(P1["dN"], 0.002, wt2))
    gamma = np.array([[-0.8, -0.6, -0.7], [-0.4, -0.3, -0.5], [-0.2, -0.1, -0.15]])
    _assert_outputs_identical(_run_em("PP", P1, gamma, wt1), _run_em("PP", P2, gamma, wt2))


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


def test_pplfp_em_returns_standard_errors_from_the_scaled_observations(monkeypatch) -> None:
    # PPLFP_EM computes its SEs (it unpacked the routine's three outputs into
    # two inside `except Exception: pass`, so SE = Pvals = {} always).  It
    # mirrors the pinned MATLAB, which passes its whitened observations
    # y = Tr*y (Tr = inv(chol(R0))) -- an open parity question upstream.
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
    Tr = np.linalg.inv(np.linalg.cholesky(args[5]).T)
    np.testing.assert_array_equal(seen["y"], Tr @ args[0])


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
