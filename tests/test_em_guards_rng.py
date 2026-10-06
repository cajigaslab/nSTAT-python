"""EM numerics that mirror the repaired MATLAB exactly (nSTAT PR #135, ``fix/pp-em`` @ ``aa88a2b``).

* Newton-Raphson loop length: MATLAB's M-step loops are ``iter = 1; while
  (~converged && iter < maxIter)`` with ``maxIter = 100`` -- at most 99 Newton
  steps.  ``PP_MStep`` ran ``range(100)``.
* No Python-only numerical guards where MATLAB returns a value: the E-step /
  M-step / SE terms are MATLAB's unclipped ``exp(terms)`` (this port clipped
  them to +-30, the decoder's to +-20), the log-determinants are unfloored, the
  closed-form updates have no ridges or eigenvalue floors, and a Newton step
  is MATLAB's ``H\\g``: elementwise ``g/H`` for one coefficient, MATLAB's LU
  with division by the pivots otherwise (``_matlab_mldivide``).  The expected
  values below were produced by MATLAB R2025b at ``aa88a2b`` on the same
  inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nstat.decoding.PPLFP import PPLFP
from nstat.decoding_algorithms import DecodingAlgorithms


def _nonconverging_mstep_problem(K=200, x=0.5):
    """One cell that never spikes, on a known state (W_K = 1e-30): each Newton step
    moves mu and gamma by exactly -1 and beta by -1/x, so none converges."""
    dN = np.zeros((1, K))
    x_K = np.full((1, K), x)
    W_K = np.full((1, 1, K), 1e-30)
    HkAll = np.ones((K, 1, 1))  # one history window, count 1 in every bin
    ES = {"Sxkm1xkm1": np.array([[K * x * x]]), "Sxkxkm1": np.array([[K * x * x]]),
          "sumXkTerms": np.array([[1e-3]])}
    return dN, x_K, W_K, HkAll, ES


def test_newton_raphson_runs_at_most_99_steps() -> None:
    # MATLAB (and this port now): 99 steps of -1 from mu = -3, gamma = -0.5 and
    # of -1/x = -2 from beta = 0.2.  The port ran 100.
    dN, x_K, W_K, HkAll, ES = _nonconverging_mstep_problem()
    np.random.seed(0)
    _, _, mu, beta, gamma, _, _ = DecodingAlgorithms.PP_MStep(
        dN, x_K, W_K, np.zeros(1), np.eye(1), ES, "poisson", np.array([-3.0]), np.array([[0.2]]),
        np.array([[-0.5]]), np.array([0.0, 0.001]), HkAll, DecodingAlgorithms.PP_EMCreateConstraints(),
        "NewtonRaphson", 0.001)
    assert float(mu[0]) == -3.0 - 99
    assert float(np.ravel(gamma)[0]) == -0.5 - 99
    np.testing.assert_allclose(float(np.ravel(beta)[0]), 0.2 - 99 / 0.5, rtol=1e-12)


def _walk_sums(K=200, x=0.5):
    dN, x_K, W_K, HkAll, ES = _nonconverging_mstep_problem(K, x)
    ES.update({"Sxkm1xk": np.array([[K * x * x]]), "Sxkxk": np.array([[K * x * x]]), "Sxkyk": np.array([[0.0]]),
               "Sykyk": np.array([[1.0]]), "sumYkTerms": np.array([[1.0]]), "Sx0": np.zeros(1),
               "Sx0x0": np.eye(1)})
    return dN, x_K, W_K, HkAll, ES


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_newton_steps_walk_to_the_exp_underflow_as_matlab(family) -> None:
    # A coefficient with no data support walks by -1 per Newton step (as in the
    # step-count test) until exp(terms) underflows to exactly 0; then H = g = 0,
    # MATLAB's 0\0 is NaN and the previous value is kept.  MATLAB R2025b
    # (aa88a2b): mu from -700 stops at -745 and gamma from -700 at -743, in both
    # M-steps.  The +-30 clip kept the walk going (-799 after 99 steps), the
    # |H| < 1e-30 / H == 0 guards stopped mu at once (-700), and LAPACK's
    # reciprocal pivot turned the denormal 1 x 1 gamma Hessian into a -Inf step.
    dN, x_K, W_K, HkAll, ES = _walk_sums()
    wt = np.array([0.0, 0.001])
    out = []
    for mu0, beta0, gamma0 in ((-700.0, 0.0, -0.5), (-3.0, 0.2, -700.0)):
        np.random.seed(0)
        if family == "PP":
            o = DecodingAlgorithms.PP_MStep(dN, x_K, W_K, np.zeros(1), np.eye(1), ES, "poisson", np.array([mu0]),
                                            np.array([[beta0]]), np.array([[gamma0]]), wt, HkAll,
                                            DecodingAlgorithms.PP_EMCreateConstraints(), "NewtonRaphson", 0.001)
            out.append((float(np.ravel(o[2])[0]), float(np.ravel(o[4])[0])))
        else:
            o = PPLFP.PPLFP_MStep(dN, np.zeros((1, x_K.shape[1])), x_K, W_K, np.zeros(1), np.eye(1), ES, "poisson",
                                  np.array([mu0]), np.array([[beta0]]), np.array([[gamma0]]), wt, HkAll,
                                  PPLFP.PPLFP_EMCreateConstraints(), "NewtonRaphson", 0.001)
            out.append((float(np.ravel(o[5])[0]), float(np.ravel(o[7])[0])))
    assert out[0][0] == -745.0
    assert out[1][1] == -743.0


_EM_GOLD = Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "em_drivers.mat"


@pytest.fixture(scope="module")
def em_gold() -> dict:
    from scipy.io import loadmat

    return loadmat(_EM_GOLD, squeeze_me=True, struct_as_record=False)


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_newton_walk_on_an_n_by_n_hessian_matches_matlab(em_gold, family) -> None:
    # MATLAB R2025b (aa88a2b; em_drivers.mat walk_*, capture_em_drivers.m): one
    # cell on a known state, three history windows, window 1 separated.  Its
    # coefficient walks from -700 by exactly -1 per Newton step to the exp()
    # underflow (-743) while windows 2 and 3 converge (1.83, 0.377); on the way
    # the 3 x 3 Hessian has a denormal pivot.  np.linalg.solve (LAPACK's
    # reciprocal pivot) made that step -Inf: gamma [-inf, 1.83, 3.79], and in
    # PP_EM the next E-step was NaN, which stopped every pp_sep run early.
    f = lambda key: em_gold[f"walk_{key}"]  # noqa: E731
    K = np.size(f("dN"))
    dN = np.asarray(f("dN"), dtype=float).reshape(1, K)
    x_K = np.asarray(f("x_K"), dtype=float).reshape(1, K)
    W_K = np.asarray(f("W_K"), dtype=float).reshape(1, 1, K)
    HkAll = np.asarray(f("HkAll"), dtype=float).reshape(K, -1, 1)
    gamma0 = np.asarray(f("gamma0"), dtype=float).reshape(-1, 1)
    wt = np.asarray(f("windowTimes"), dtype=float)
    ES = {k[len("walk_ES_"):]: np.atleast_2d(np.asarray(v, dtype=float))
          for k, v in em_gold.items() if k.startswith("walk_ES_")}
    ES["Sx0"] = ES["Sx0"].reshape(1)
    mu0, beta0 = np.array([float(f("mu0"))]), np.array([[float(f("beta0"))]])
    np.random.seed(0)
    if family == "PP":
        o = DecodingAlgorithms.PP_MStep(dN, x_K, W_K, np.zeros(1), np.eye(1), ES, "poisson", mu0, beta0, gamma0,
                                        wt, HkAll, DecodingAlgorithms.PP_EMCreateConstraints(), "NewtonRaphson", 0.001)
        mu, beta, gamma, pre = o[2], o[3], o[4], "walk_pp_"
    else:
        o = PPLFP.PPLFP_MStep(dN, np.zeros((1, K)), x_K, W_K, np.zeros(1), np.eye(1), ES, "poisson", mu0, beta0,
                              gamma0, wt, HkAll, PPLFP.PPLFP_EMCreateConstraints(), "NewtonRaphson", 0.001)
        mu, beta, gamma, pre = o[5], o[6], o[7], "walk_lfp_"
    ref_gamma = np.asarray(em_gold[pre + "gamma"], dtype=float)
    gamma = np.ravel(gamma)
    assert ref_gamma[0] == -743.0 and gamma[0] == ref_gamma[0]  # exact: 43 steps of exactly -1
    # The converged coefficients: MATLAB's randn vs NumPy's perturb the known
    # state by ~1e-15 (W_K = 1e-30), hence not bit for bit.
    np.testing.assert_allclose(gamma[1:], ref_gamma[1:], rtol=1e-12)
    np.testing.assert_allclose(float(np.ravel(mu)[0]), float(em_gold[pre + "mu"]), rtol=1e-12)
    np.testing.assert_allclose(float(np.ravel(beta)[0]), float(em_gold[pre + "beta"]), rtol=1e-12)


def test_newton_solve_is_matlabs_mldivide(em_gold) -> None:
    # MATLAB R2025b H\g on 20 square systems (em_drivers.mat mldivide_*), bit for
    # bit: denormal pivots (pp_sep_hess is a pp_sep history Hessian: MATLAB's
    # step is [1, 1, 5.4e-17], LAPACK's reciprocal pivot gives [Inf, 1, 5.4e-17]),
    # exactly singular systems (MATLAB returns +-Inf / NaN; np.linalg.solve
    # raised), ill-conditioned ones (-hilb(5): np.linalg.solve differs by
    # 3e-12), triangular, Hessenberg, permuted triangular and symmetric definite
    # ones of either sign (MATLAB solves them by LU too, not Cholesky).
    from nstat.decoding_algorithms import _matlab_mldivide

    names = [str(n) for n in np.atleast_1d(em_gold["mldivide_names"])]
    assert len(names) == 20 and "pp_sep_hess" in names and "sing_sym" in names
    for name in names:
        H = np.atleast_2d(np.asarray(em_gold[f"mldivide_{name}_H"], dtype=float))
        g = np.atleast_1d(np.asarray(em_gold[f"mldivide_{name}_g"], dtype=float))
        ref = np.atleast_1d(np.asarray(em_gold[f"mldivide_{name}_x"], dtype=float))
        got = _matlab_mldivide(H, g)
        assert np.array_equal(got, ref, equal_nan=True), name
        finite = np.isfinite(ref)
        assert np.array_equal(np.signbit(got[finite]), np.signbit(ref[finite])), name
        # A column right-hand side (PPLFP's GradTerm) gives the same column.
        np.testing.assert_array_equal(_matlab_mldivide(H, g.reshape(-1, 1)), ref.reshape(-1, 1), err_msg=name)


def test_estep_log_likelihood_is_not_floored() -> None:
    # MATLAB: -1/2*log(det(Px0)) with Px0 = 0 is +Inf (PP_EStep logll = Inf at
    # aa88a2b), which stops EM.  The determinant floors made it finite.
    K = 50
    ll = DecodingAlgorithms.PP_EStep(0.9, 0.01, np.zeros((1, K)), np.array([-3.0]), np.array([[0.5]]), "poisson",
                                     0.0, np.zeros((K, 1, 1)), np.zeros(1), np.zeros((1, 1)))[2]
    assert ll == np.inf
    ll = PPLFP.PPLFP_EStep(np.array([[0.9]]), np.array([[0.01]]), np.array([[1.0]]), np.array([[0.1]]),
                           np.zeros((1, K)), np.zeros(1), np.zeros((1, K)), np.array([-3.0]), np.array([[0.5]]),
                           "poisson", 0.001, np.array(0.0), np.zeros((K, 1, 1)), np.zeros(1), np.zeros((1, 1)))[2]
    assert ll == np.inf


@pytest.mark.parametrize("fit", ["poisson", "binomial"])
def test_decoder_intensity_follows_matlab_rule(fit, monkeypatch) -> None:
    # MATLAB PPDecode_updateLinear: lambdaDelta = exp(linTerm) (binomial
    # exp./(1+exp)), unclipped, then NaN / Inf -> 1.  The port clipped linTerm to
    # [-20, 20].  The PPDecodeFilterLinear fast path (pure Python and numba) must
    # agree with PPDecode_updateLinear.
    from nstat.extras import _numba_kernels

    for lin in (25.0, -25.0, 800.0):
        lam = DecodingAlgorithms.PPDecode_updateLinear(np.zeros(1), np.eye(1), np.zeros((1, 1)), np.array([lin]),
                                                       np.array([[1.0]]), fit)[2].item()
        if lin == 800.0:
            expected = 1.0
        else:
            expected = np.exp(lin) if fit == "poisson" else np.exp(lin) / (1.0 + np.exp(lin))
        assert lam == expected, lin
    A, Q = np.array([[1.0]]), np.array([[0.01]])
    dN = np.array([[1.0, 0.0, 1.0, 0.0]])
    mu = np.array([21.0])
    beta = np.array([[0.5]])
    ref_x = []
    x_p, W_p = np.zeros(1), A @ np.array([[0.1]]) @ A.T + Q
    for k in range(dN.shape[1]):
        x_u, W_u, _ = DecodingAlgorithms.PPDecode_updateLinear(x_p, W_p, dN, mu, beta, fit, None, None, k)
        ref_x.append(x_u[0])
        x_p, W_p = A @ x_u, A @ W_u @ A.T + Q
    for numba_on in (False, True):
        monkeypatch.setattr(_numba_kernels, "_NUMBA_AVAILABLE", numba_on and _numba_kernels._NUMBA_AVAILABLE)
        x_u = DecodingAlgorithms.PPDecodeFilterLinear(A, Q, dN, mu, beta, fit, 0.001, None, None, np.zeros(1),
                                                      np.array([[0.1]]))[2]
        np.testing.assert_allclose(np.ravel(x_u), ref_x, rtol=1e-12, atol=1e-14)


def _closed_form_sums(seed=3, dx=2, K=40):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dx, K + 1))
    S = X[:, :-1] @ X[:, :-1].T
    Sx = X[:, 1:] @ X[:, :-1].T
    A = np.linalg.solve(S.T, Sx.T).T
    R = X[:, 1:] - A @ X[:, :-1]
    return {"Sxkm1xkm1": S, "Sxkxkm1": Sx, "Sxkm1xk": Sx.T, "Sxkxk": X[:, 1:] @ X[:, 1:].T,
            "sumXkTerms": R @ R.T}, X[:, 1:]


@pytest.mark.parametrize("cons", [(1, 0, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 1, 0), (1, 0, 1, 1, 1, 1, 1)])
def test_pp_mstep_closed_form_updates_are_matlabs(cons) -> None:
    # MATLAB: Ahat = Sxkxkm1/Sxkm1xkm1 (or the .*I form), Q from sumXkTerms,
    # x0hat = (inv(Px0)+Ahat'/Qhat*Ahat)\(Ahat'/Qhat*x_K(:,1)+Px0\x0) and
    # Px0hat = (x0hat*x0hat' - x0*x0hat' - x0hat*x0' + x0*x0').*I -- no 1e-12
    # ridges, no 1e-10 eigenvalue floors (the eigh rebuild also moved Qhat by
    # round-off when no floor was active).
    ES, x_K = _closed_form_sums()
    dx, K = x_K.shape
    I = np.eye(dx)
    x0, Px0 = np.array([0.3, -0.2]), np.diag([0.5, 0.2])
    c = DecodingAlgorithms.PP_EMCreateConstraints(*cons)
    A, Q, _, _, _, x0hat, Px0hat = DecodingAlgorithms.PP_MStep(
        np.zeros((1, K)), x_K, np.zeros((dx, dx, K)), x0, Px0, ES, "poisson", np.array([-3.0]),
        np.zeros((dx, 1)), np.array(0.0), None, np.zeros((K, 0, 1)), c, "GLM")
    S, Sx, sx = ES["Sxkm1xkm1"], ES["Sxkxkm1"], ES["sumXkTerms"]
    A_m = np.linalg.solve((S * I).T, (Sx * I).T).T if c["AhatDiag"] else np.linalg.solve(S.T, Sx.T).T
    if c["QhatDiag"]:  # MATLAB 1/(dx*K)*trace(S)*eye, or 1/K*(S.*I) then (Q + Q')/2
        Q_m = (1.0 / (dx * K)) * np.trace(sx) * I if c["QhatIsotropic"] else (1.0 / K) * (sx * I)
    else:
        Q_m = (1.0 / K) * sx
    if not c["QhatIsotropic"]:
        Q_m = (Q_m + Q_m.T) / 2
    AtQ = np.linalg.solve(Q_m.T, A_m).T
    x0_m = np.linalg.solve(np.linalg.inv(Px0) + AtQ @ A_m, AtQ @ x_K[:, 0] + np.linalg.solve(Px0, x0))
    a, b = x0_m.reshape(dx, 1), x0.reshape(dx, 1)
    outer = a @ a.T - b @ a.T - a @ b.T + b @ b.T
    Px0_m = np.trace(outer) / (dx * K) * I if c["Px0Isotropic"] else ((outer * I) + (outer * I).T) / 2
    for got, ref in ((A, A_m), (Q, Q_m), (x0hat, x0_m), (Px0hat, Px0_m)):
        np.testing.assert_array_equal(got, ref)


def test_px0_estimate_collapses_as_in_matlab() -> None:
    # With x0 fixed (Estimatex0 = 0) and Px0 estimated, MATLAB's single-sample
    # estimate is exactly 0 (the collapse that stops EM, MATLAB C2); the port
    # floored its eigenvalues at 1e-10.
    ES, x_K = _closed_form_sums()
    dx, K = x_K.shape
    c = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 1)
    Px0hat = DecodingAlgorithms.PP_MStep(np.zeros((1, K)), x_K, np.zeros((dx, dx, K)), np.zeros(dx), np.eye(dx), ES,
                                         "poisson", np.array([-3.0]), np.zeros((dx, 1)), np.array(0.0), None,
                                         np.zeros((K, 0, 1)), c, "GLM")[6]
    np.testing.assert_array_equal(Px0hat, np.zeros((dx, dx)))


def test_pplfp_mstep_divisions_are_matlabs_mrdivide() -> None:
    # MATLAB Ahat = Sxkxkm1/Sxkm1xkm1 and Chat = Sxkyk'/Sxkxk (LU solves); the
    # port used least squares.
    ES, x_K = _closed_form_sums()
    dx, K = x_K.shape
    y = np.vstack([x_K[0] + 0.1, x_K[1] - 0.2])
    ES.update({"Sxkyk": x_K @ y.T, "Sykyk": y @ y.T, "sumYkTerms": np.eye(2) * K * 0.1, "Sx0": np.zeros(dx),
               "Sx0x0": np.eye(dx)})
    out = PPLFP.PPLFP_MStep(np.zeros((1, K)), y, x_K, np.zeros((dx, dx, K)), np.zeros(dx), np.eye(dx), ES, "poisson",
                            np.array([-3.0]), np.zeros((dx, 1)), np.array(0.0), None, np.zeros((K, 1, 1)),
                            PPLFP.PPLFP_EMCreateConstraints(), "GLM")
    np.testing.assert_array_equal(out[0], np.linalg.solve(ES["Sxkm1xkm1"].T, ES["Sxkxkm1"].T).T)
    np.testing.assert_array_equal(out[2], np.linalg.solve(ES["Sxkxk"].T, ES["Sxkyk"]).T)


def test_pp_em_raises_for_a_non_positive_definite_qhat0() -> None:
    # MATLAB's chol(Qhat0, 'lower') errors; the port whitened with Tq = I.
    with pytest.raises(np.linalg.LinAlgError):
        DecodingAlgorithms.PP_EM(np.zeros((1, 20)), np.eye(2), np.diag([0.01, -0.01]), np.array([-3.0]),
                                 np.zeros((2, 1)))


# ---------------------------------------------------------------------------
# Reproducible Monte Carlo (brief item 3).  MATLAB's EM draws (normrnd, mvnrnd)
# come from its global stream, which rng(seed) controls.  The PP routines drew
# from NumPy's global legacy stream (np.random.randn); the PPLFP routines from an
# unseeded np.random.default_rng() per call, so np.random.seed did not reproduce
# them, and under nstat.extras.matlab_rng.seeded_global_rng (which patches
# default_rng to a fresh generator with the same seed on every call) every
# M-step and both SE blocks drew the same normals.  All EM Monte Carlo now draws
# from NumPy's global stream.
# ---------------------------------------------------------------------------
def _em_problem(seed=11, K=150, C=3):
    rng = np.random.default_rng(seed)  # data only
    A = np.array([[0.95, 0.02], [-0.02, 0.93]])
    Q = np.diag([0.01, 0.02])
    x = np.zeros((2, K))
    for k in range(1, K):
        x[:, k] = A @ x[:, k - 1] + np.sqrt(np.diag(Q)) * rng.standard_normal(2)
    beta = np.array([[0.8, -0.6, 0.4], [0.3, 0.7, -0.5]])[:, :C]
    mu = np.log(np.full(C, 0.05))
    dN = (rng.random((C, K)) < np.exp(mu[:, None] + beta.T @ x)).astype(float)
    Cm, R, alpha = np.array([[1.0, 0.4], [-0.3, 1.0]]), np.diag([0.05, 0.08]), np.array([0.1, -0.1])
    y = Cm @ x + alpha[:, None] + np.sqrt(np.diag(R))[:, None] * rng.standard_normal((2, K))
    return dict(A=A, Q=Q, beta=beta, mu=mu, dN=dN, Cm=Cm, R=R, alpha=alpha, y=y)


def _run_em(family, P):
    if family == "PP":
        out = DecodingAlgorithms.PP_EM(P["dN"], P["A"], P["Q"], P["mu"] + 0.3, 0.5 * P["beta"], "poisson", 0.001,
                                       None, None, None, None, DecodingAlgorithms.PP_EMCreateConstraints(mcIter=20))
        arrays = list(out[:9]) + [out[9][k] for k in sorted(out[9])] + [out[10][k] for k in sorted(out[10])]
    else:
        out = PPLFP.PPLFP_EM(P["y"], P["dN"], P["A"], P["Q"], P["Cm"], P["R"], P["alpha"], P["mu"] + 0.3,
                             0.5 * P["beta"], PPLFP_EM_Constraints=PPLFP.PPLFP_EMCreateConstraints(mcIter=20))
        arrays = list(out[:12]) + [out[12][k] for k in sorted(out[12])] + [out[13][k] for k in sorted(out[13])]
    return [np.asarray(a, dtype=float) for a in arrays]


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_em_monte_carlo_follows_the_global_seed(family) -> None:
    from nstat.extras.matlab_rng import seeded_global_rng

    P = _em_problem()
    runs = []
    for seed in (5, 5, 6):
        np.random.seed(seed)
        runs.append(_run_em(family, P))
    with seeded_global_rng(5):
        runs.append(_run_em(family, P))
    with seeded_global_rng(5):
        runs.append(_run_em(family, P))
    for a, b in zip(runs[0], runs[1]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(runs[3], runs[4]):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(runs[0], runs[2]))


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_successive_monte_carlo_blocks_draw_fresh_normals(family, monkeypatch) -> None:
    import sys

    from nstat.extras.matlab_rng import seeded_global_rng

    module = sys.modules["nstat.decoding_algorithms" if family == "PP" else "nstat.decoding.PPLFP"]
    real = module._mc_state_draws
    first_z = []  # the first standard-normal block of every Monte Carlo draw loop

    def spy(m, W, M, normal, **kw):
        def recording_normal(d, n):
            z = normal(d, n)
            first_z.append(np.array(z, dtype=float))
            return z
        return real(m, W, M, recording_normal, **kw)

    monkeypatch.setattr(module, "_mc_state_draws", spy)
    P = _em_problem()
    with seeded_global_rng(3):
        _run_em(family, P)
    by_shape = {}
    for z in first_z:
        by_shape.setdefault(z.shape, []).append(z)
    blocks = [zs for zs in by_shape.values() if len(zs) > 1]
    assert blocks
    for zs in blocks:  # every normal block of a run is distinct
        assert len({z.tobytes() for z in zs}) == len(zs)


def test_pp_em_rejects_ikeda_acceleration() -> None:
    # MATLAB's PP_EM runs an Ikeda acceleration step when EnableIkeda = 1 (a
    # model without history; it errors with history).  The step is not ported;
    # the port ignored the option silently (fence P7) and now raises.
    cons = DecodingAlgorithms.PP_EMCreateConstraints(EnableIkeda=1, mcIter=5)
    with pytest.raises(NotImplementedError, match="EnableIkeda"):
        DecodingAlgorithms.PP_EM(np.zeros((1, 20)), np.eye(1), 0.01 * np.eye(1), np.array([-3.0]), np.zeros((1, 1)),
                                 PPEM_Constraints=cons)


@pytest.mark.parametrize("family", ["PP", "PPLFP"])
def test_singular_observed_information_flags_the_nonidentifiable_parameters(em_gold, family) -> None:
    # At MATLAB's pp_sep estimates five history coefficients sit at the exp()
    # underflow (separated windows): their information and scores are exactly
    # 0, so IObs is singular.  MATLAB's SE pass never returns there (eye/IObs is
    # Inf / NaN and nearestSPD loops); this port uses the pseudo-inverse.  It
    # used to report SE ~1e-8 and p = 0 for those coefficients; now SE = p =
    # NaN for every parameter in the null space, a RuntimeWarning names them,
    # and every other SE stays finite and positive.
    from nstat.decoding_algorithms import _compute_history_terms
    from nstat.extras.matlab_rng import seeded_global_rng

    f = lambda key: em_gold[f"pp_sep_{key}"]  # noqa: E731
    dN = np.atleast_2d(f("dN")).astype(float)
    wt = np.asarray(f("windowTimes"), dtype=float)
    H = _compute_history_terms(dN, float(f("delta")), wt)
    gamma = np.asarray(f("gammahat"), dtype=float)
    sep = gamma < -100
    assert sep.sum() == 5
    x0, Px0 = np.ravel(f("x0hat")), np.atleast_2d(f("Px0hat"))
    expected = ", ".join(f"gamma[{w}, {c}]" for c, w in sorted((c, w) for w, c in zip(*np.nonzero(sep))))
    with seeded_global_rng(1), pytest.warns(RuntimeWarning, match="singular") as record:
        if family == "PP":
            ES = {k[len("pp_sep_es_ES_"):]: np.asarray(v, dtype=float)
                  for k, v in em_gold.items() if k.startswith("pp_sep_es_ES_")}
            for key in ("Sxkm1xkm1", "Sxkxkm1", "Sxkm1xk", "Sxkxk", "sumXkTerms"):
                ES[key] = ES[key].reshape(2, 2)
            SE, P, _ = DecodingAlgorithms.PP_ComputeParamStandardErrors(
                dN, f("xKFinal"), f("WKFinal"), f("Ahat"), f("Qhat"), x0, Px0, ES, "poisson", np.ravel(f("muhat")),
                f("betahat"), gamma, wt, H, DecodingAlgorithms.PP_EMCreateConstraints(mcIter=100))
        else:
            # The same cells plus one Gaussian channel y = C x + alpha + noise.
            rng = np.random.default_rng(0)
            C, R, alpha = np.array([[1.0, 0.5]]), np.array([[0.05]]), np.array([0.1])
            y = C @ np.asarray(f("x"), dtype=float) + alpha[:, None] + np.sqrt(R[0, 0]) * rng.standard_normal((1, dN.shape[1]))
            x_K, W_K, _, ES = PPLFP.PPLFP_EStep(f("Ahat"), f("Qhat"), C, R, y, alpha, dN, np.ravel(f("muhat")),
                                                f("betahat"), "poisson", float(f("delta")), gamma, H, x0, Px0)
            SE, P, _ = PPLFP.PPLFP_ComputeParamStandardErrors(
                y, dN, x_K, W_K, f("Ahat"), f("Qhat"), C, R, alpha, x0, Px0, ES, "poisson", np.ravel(f("muhat")),
                f("betahat"), gamma, wt, H, PPLFP.PPLFP_EMCreateConstraints(mcIter=100))
    # Count only this warning: any other RuntimeWarning a NumPy / SciPy build
    # may emit is not what is tested here.
    singular = [str(w.message) for w in record if "observed information matrix is singular" in str(w.message)]
    assert len(singular) == 1 and expected in singular[0]
    se_g, p_g = np.asarray(SE["gamma"], dtype=float), np.asarray(P["gamma"], dtype=float)
    assert np.all(np.isnan(se_g[sep])) and np.all(np.isnan(p_g[sep]))
    assert np.all(np.isfinite(se_g[~sep]) & (se_g[~sep] > 0)) and np.all(np.isfinite(p_g[~sep]))
    for key, value in SE.items():
        if key != "gamma":
            assert np.all(np.isfinite(np.asarray(value, dtype=float))), key
