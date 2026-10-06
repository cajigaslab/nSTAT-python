"""EM numerics that mirror the repaired MATLAB exactly (nSTAT PR #135, ``fix/pp-em`` @ ``aa88a2b``).

* Newton-Raphson loop length: MATLAB's M-step loops are ``iter = 1; while
  (~converged && iter < maxIter)`` with ``maxIter = 100`` -- at most 99 Newton
  steps.  ``PP_MStep`` ran ``range(100)``.
* No Python-only numerical guards where MATLAB returns a value: the E-step /
  M-step / SE terms are MATLAB's unclipped ``exp(terms)`` (this port clipped
  them to +-30, the decoder's to +-20), the log-determinants are unfloored, the
  closed-form updates have no ridges or eigenvalue floors, and a scalar Newton
  step is MATLAB's elementwise ``g/H``.  The expected values below were
  produced by MATLAB R2025b at ``aa88a2b`` on the same inputs.
"""
from __future__ import annotations

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
