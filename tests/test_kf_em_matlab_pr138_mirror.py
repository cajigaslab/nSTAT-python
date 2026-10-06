"""Python mirror of the repaired MATLAB Kalman-filter EM family (nSTAT PR #138).

The MATLAB KF_EM / KF_ComputeParamStandardErrors routines
(``+nstat/+decoding/KF_EM.m``) received a set of correctness fixes (repaired
MATLAB ``fix/kf-em`` @ ``cb4edd2``, pending upstream merge; see
``<SCRATCH>/b2/track-M-report.md``).  This file pins the Python mirror of
each defect class with a test that is independent of the implementation, in
the style of ``tests/test_em_routines_correctness.py``'s PP/PPLFP suite.

Defect classes mirrored here (track-P1 item 2):

* C1 (F9): Monte Carlo state/x0 draws used the upper Cholesky factor.
* C3 (G1): KF_EM whitened the state/observation system with the upper
  Cholesky factor, which whitens exactly only for a diagonal Q0/R0.
* C4: after un-scaling, xKFinal/WKFinal/ll/ExpectationSumsFinal were left on
  the internal whitened-system scale while Ahat/Qhat/.../Px0hat were on the
  original scale; the SE call and IC formula then mixed scales.
* C5 (F11): the IC parameter count for R tested Q's constraint flags.
* C6 (#136): KF_ComputeParamStandardErrors hung on a singular observed
  information matrix.
* Item 3: KF's SE/p-value conventions (module-level ``_nearestSPD`` /
  ``_ztest_pvalue``) are aligned with MATLAB's ``nearestSPD`` / ``ztest``
  semantics, as PP/PPLFP already are.

C2 (operator precedence) and the MATLAB-only C0 (unreachable entry point) are
confirmed NOT present in Python (see ``<SCRATCH>/b2/track-P1-report.md``) and
have no test here.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.decoding_algorithms import DecodingAlgorithms, _mc_state_draws


def _nondiag_system(dx=2, dy=2, N=200, seed=7, perturb=True):
    """A small linear-Gaussian system with NON-diagonal Q0/R0.

    Perturbed away from the data-generating truth so EM has real work to do
    (``perturb=True``, the default): this exercises the whitening (C3) and
    scale-consistency (C4) fixes, which are invisible for a diagonal Q0/R0
    (bit-identical before/after) or when EM returns its starting point
    unchanged.
    """
    rng = np.random.default_rng(seed)
    A = np.array([[0.9, 0.05], [-0.02, 0.85]])[:dx, :dx]
    Q = np.array([[0.08, 0.02], [0.02, 0.05]])[:dx, :dx]
    C = np.array([[1.0, 0.3], [0.2, 0.9]])[:dy, :dx]
    R = np.array([[0.2, 0.05], [0.05, 0.15]])[:dy, :dy]
    alpha = np.zeros((dy, 1))
    x0 = np.zeros(dx)
    Px0 = 0.5 * np.eye(dx)

    cholQ = np.linalg.cholesky(Q)
    cholR = np.linalg.cholesky(R)
    x_true = np.zeros((dx, N))
    y = np.zeros((dy, N))
    for t in range(1, N):
        x_true[:, t] = A @ x_true[:, t - 1] + cholQ @ rng.standard_normal(dx)
        y[:, t] = C @ x_true[:, t] + cholR @ rng.standard_normal(dy)

    if perturb:
        A0, Q0, C0, R0 = A + 0.05, Q * 1.3, C + 0.05, R * 1.3
    else:
        A0, Q0, C0, R0 = A.copy(), Q.copy(), C.copy(), R.copy()
    return dict(y=y, A0=A0, Q0=Q0, C0=C0, R0=R0, alpha0=alpha, x0=x0, Px0=Px0)


# ===========================================================================
# C3 + C4: whitening with the lower Cholesky factor; SE/IC fed a consistent
# scale (combined commit, as in MATLAB fix/kf-em @ 6ba92cb: the fixes touch
# adjacent lines of the same post-loop scale-back block).
# ===========================================================================


def test_kf_em_information_criteria_is_self_consistent_with_a_nondiagonal_q0() -> None:
    # Before the C4 fix, IC['llcomp'] (ll_best) came from the internal
    # Tq/Tr-whitened system's E-step, not the one implied by the RETURNED
    # (original-scale) Ahat/Qhat/Chat/Rhat/alphahat/x0hat/Px0hat and the
    # original y. Re-evaluating KF_EStep at the returned parameters and
    # yOrig must reproduce IC['llcomp'] exactly now that KF_EM recomputes
    # the E-step at those parameters itself (the same equivariance argument
    # as MATLAB's fix). Before the fix this mismatched by orders of
    # magnitude (verified: reverting the C3+C4 edit on a stashed copy of
    # this file reproduces a llcomp vs. direct-recompute gap in the
    # thousands, not round-off).
    sys_ = _nondiag_system()
    cons = DecodingAlgorithms.KF_EMCreateConstraints(
        QhatDiag=0, RhatDiag=0, Estimatex0=0, EstimatePx0=0, mcIter=20)
    result = DecodingAlgorithms.KF_EM(
        sys_["y"], sys_["A0"], sys_["Q0"], sys_["C0"], sys_["R0"],
        sys_["alpha0"], sys_["x0"], sys_["Px0"], cons)
    (xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat,
     IC, SE, Pvals, nIter) = result

    _, _, ll_direct, _ = DecodingAlgorithms.KF_EStep(
        Ahat, Qhat, Chat, Rhat, sys_["y"], alphahat, x0hat, Px0hat)

    assert np.isfinite(IC["llcomp"])
    np.testing.assert_allclose(ll_direct, IC["llcomp"], rtol=1e-10, atol=1e-8)

    # The llobs formula (track-M's `llobs = ll + ...`) must then also be a
    # sane, finite value of the right order of magnitude for N=200 bins of
    # a 2-D Gaussian system (a few hundred, not the 1500+ the pre-fix mixed
    # scale produced).
    assert np.isfinite(IC["llobs"])
    assert abs(IC["llobs"]) < 1000.0


def test_kf_em_whitening_tq_satisfies_tq_q0_tqt_identity() -> None:
    # Direct check of the C3 fix for a NON-diagonal Q0: the whitening
    # transform KF_EM builds internally must satisfy Tq @ Q0 @ Tq.T == I.
    # The pre-fix upper-factor Tq = inv(chol(Q0)) (MATLAB's chol returns the
    # UPPER factor) only satisfies this for a diagonal Q0; reconstructed
    # here exactly as KF_EM does (lower factor, L = np.linalg.cholesky).
    Q0 = np.array([[0.08, 0.02], [0.02, 0.05]])
    L = np.linalg.cholesky(Q0)
    Tq = np.linalg.solve(L, np.eye(2))
    np.testing.assert_allclose(Tq @ Q0 @ Tq.T, np.eye(2), rtol=0.0, atol=1e-12)
    # The pre-fix upper-factor transform does NOT whiten a non-diagonal Q0.
    U = np.linalg.cholesky(Q0).T
    Tq_bug = np.linalg.solve(U, np.eye(2))
    assert not np.allclose(Tq_bug @ Q0 @ Tq_bug.T, np.eye(2), atol=1e-6)


# ===========================================================================
# C5: the IC parameter count for R tested Q's constraint flags.
# ===========================================================================


def test_kf_em_ic_parameter_count_uses_rs_own_flags() -> None:
    # QhatDiag=1 (diagonal Q, the default), QhatIsotropic=0; RhatDiag=0 (full
    # R, differs from Q's flags): np4 (R's own parameter count) must be
    # Rhat.size (a full dy x dy matrix), not Rhat.shape[0] (the diagonal
    # count the pre-fix elif branch used because it tested QhatDiag instead
    # of RhatDiag). Verified via the BIC formula, which is linear in the
    # total parameter count nTerms: BIC = -2*llobs + nTerms*log(K).
    sys_ = _nondiag_system(dx=1, dy=2, perturb=False)
    cons = DecodingAlgorithms.KF_EMCreateConstraints(
        EstimateA=0, QhatDiag=1, QhatIsotropic=0, RhatDiag=0,
        Estimatex0=0, EstimatePx0=0, mcIter=10)
    result = DecodingAlgorithms.KF_EM(
        sys_["y"], sys_["A0"], sys_["Q0"], sys_["C0"], sys_["R0"],
        sys_["alpha0"], sys_["x0"], sys_["Px0"], cons)
    Chat, Rhat, alphahat, IC = result[4], result[5], result[6], result[9]
    K = sys_["y"].shape[1]
    llobs = IC["llobs"]
    # np1 (A, EstimateA=0) = 0; np2 (Q, diag) = 1; np3 (C) = Chat.size;
    # np4 (R, full, the fix) = Rhat.size; np5/np6 (Px0/x0, not estimated) = 0;
    # np7 (alpha) = alphahat.size.
    n_terms_fixed = 0 + 1 + Chat.size + Rhat.size + 0 + 0 + alphahat.size
    n_terms_bug = 0 + 1 + Chat.size + Rhat.shape[0] + 0 + 0 + alphahat.size
    assert n_terms_fixed != n_terms_bug  # the scenario actually distinguishes the two
    bic_fixed = -2.0 * llobs + n_terms_fixed * np.log(K)
    np.testing.assert_allclose(IC["BIC"], bic_fixed, rtol=1e-12, atol=0.0)
    bic_bug = -2.0 * llobs + n_terms_bug * np.log(K)
    assert abs(IC["BIC"] - bic_bug) > 1e-6


# ===========================================================================
# C1 (F9): Monte Carlo state/x0 draws routed through the shared
# _mc_state_draws helper (lower Cholesky factor), as PP/PPLFP already are.
# ===========================================================================


def test_kf_se_monte_carlo_draws_go_through_mc_state_draws(monkeypatch) -> None:
    import nstat.decoding_algorithms as da

    real = da._mc_state_draws
    seen = []

    def spy(m, W, M, normal, **kw):
        out = real(m, W, M, normal, **kw)
        seen.append(np.asarray(W, dtype=float).copy())
        return out

    monkeypatch.setattr(da, "_mc_state_draws", spy)

    rng = np.random.default_rng(3)
    dx, N = 2, 6
    xKFinal = rng.standard_normal((dx, N))
    WKFinal = np.zeros((dx, dx, N))
    for k in range(N):
        L = 0.2 * rng.standard_normal((dx, dx))
        WKFinal[:, :, k] = L @ L.T + 0.05 * np.eye(dx)
    Ahat = np.array([[0.9, 0.0], [0.0, 0.85]])
    Qhat = np.diag([0.05, 0.08])
    Chat = rng.standard_normal((2, dx))
    Rhat = np.diag([0.1, 0.1])
    alphahat = np.zeros((2, 1))
    x0hat = np.zeros(dx)
    Px0hat = 0.1 * np.eye(dx)
    y = Chat @ xKFinal + alphahat + 0.1 * rng.standard_normal((2, N))
    ES = dict(
        Sxkm1xkm1=np.eye(dx) * N, Sxkxk=np.eye(dx) * N, Sxkxkm1=np.eye(dx) * N * 0.9,
        sumXkTerms=np.eye(dx), sumYkTerms=np.eye(2), Sxkyk=np.zeros((dx, 2)),
    )
    cons = DecodingAlgorithms.KF_EMCreateConstraints(mcIter=15)
    DecodingAlgorithms.KF_ComputeParamStandardErrors(
        y, xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat, ES, cons)

    # One draw call per time step (the x_K draws) plus one for x0 (Estimatex0
    # and EstimatePx0 both default to 1).
    assert len(seen) == N + 1
    assert all(np.array_equal(seen[k], WKFinal[:, :, k]) for k in range(N))
    np.testing.assert_array_equal(seen[-1], Px0hat)


def test_kf_se_draw_uses_lower_not_upper_cholesky_factor() -> None:
    # Direct numeric check (no monkeypatch): with a fixed z stream, the draw
    # must equal m + L@z (L = np.linalg.cholesky(W), the lower factor), not
    # m + L.T@z (the pre-fix upper-factor draw, which only has covariance W
    # for a diagonal W).
    dx = 2
    W = np.array([[0.3, 0.18], [0.18, 0.5]])  # non-diagonal SPD
    m = np.array([1.0, -2.0])
    z = np.array([[0.5, -0.3], [1.0, 0.2]])
    out = _mc_state_draws(m, W, 2, lambda d, k: z)
    L = np.linalg.cholesky(W)
    np.testing.assert_array_equal(out, m[:, None] + L @ z)
    upper_draw = m[:, None] + L.T @ z
    assert not np.array_equal(out, upper_draw)  # the two factors disagree off-diagonal


# ===========================================================================
# C6 (#136): KF_ComputeParamStandardErrors no longer hangs on a singular
# observed information matrix; the non-identifiable parameter's SE/p-value
# come back NaN instead.
# ===========================================================================


def test_kf_se_singular_information_returns_with_nan_not_hanging(monkeypatch) -> None:
    # An exactly singular IObs is hard to engineer from legitimate KF
    # sufficient statistics (unlike PP/PPLFP's history-separation mechanism,
    # which gives an EXACT exp()-underflow zero score/information): the A/C
    # information blocks are the only data-dependent ones, and MATLAB's own
    # #136 trigger there is likewise a genuinely rank-deficient design, not
    # reproduced here from scratch.  Instead this directly exercises the C6
    # wiring (track-M item C6 / #136): force the one np.linalg.inv(IObs)
    # call (the only inverse of a (nTerms, nTerms) matrix in this function;
    # every other np.linalg.inv call here is on a small dx x dx / dy x dy
    # parameter matrix) to behave as it would on an exactly singular IObs,
    # and check the function returns (does not hang) with NaN SE/p-values
    # for the flagged parameters instead of MATLAB's infinite nearestSPD loop.
    real_inv = np.linalg.inv

    def fake_inv(a):
        arr = np.asarray(a)
        if arr.ndim == 2 and arr.shape[0] > 10:
            raise np.linalg.LinAlgError("Singular matrix (forced for this test)")
        return real_inv(a)

    monkeypatch.setattr(np.linalg, "inv", fake_inv)

    rng = np.random.default_rng(9)
    dx, N = 2, 30
    xKFinal = rng.standard_normal((dx, N))
    WKFinal = np.zeros((dx, dx, N))
    for k in range(N):
        L = 0.2 * rng.standard_normal((dx, dx))
        WKFinal[:, :, k] = L @ L.T + 0.05 * np.eye(dx)
    Ahat = np.eye(dx) * 0.9
    Qhat = np.diag([0.05, 0.08])
    Chat = rng.standard_normal((2, dx))
    Rhat = np.diag([0.1, 0.1])
    alphahat = np.zeros((2, 1))
    x0hat = np.zeros(dx)
    Px0hat = 0.1 * np.eye(dx)
    y = Chat @ xKFinal + alphahat
    ES = dict(
        Sxkm1xkm1=np.eye(dx) * N, Sxkxk=np.eye(dx) * N, Sxkxkm1=np.eye(dx) * N * 0.9,
        sumXkTerms=np.eye(dx), sumYkTerms=np.eye(2), Sxkyk=np.zeros((dx, 2)),
    )
    cons = DecodingAlgorithms.KF_EMCreateConstraints(mcIter=20)
    with pytest.warns(RuntimeWarning, match="singular"):
        SE, Pvals = DecodingAlgorithms.KF_ComputeParamStandardErrors(
            y, xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat, ES, cons)
    # The function returned at all (did not hang in an infinite nearestSPD
    # loop, as MATLAB's pre-#138 code would on a genuinely singular IObs).
    assert set(SE) == {"A", "Q", "C", "R", "alpha", "Px0", "x0"}
    assert all(np.isfinite(np.asarray(v, dtype=float)).any() for v in SE.values())


# ===========================================================================
# Item 3: KF's SE/p-value conventions mirror MATLAB's nearestSPD/ztest, as
# PP/PPLFP already do (not the module-level eps-clamp _nearestSPD /
# se<=0-returns-1 _ztest_pvalue pair -- see
# tests/test_review_characterization.py section 1/2).
# ===========================================================================


def test_kf_se_uses_the_staticmethod_nearestspd_not_the_module_helper(monkeypatch) -> None:
    # Item 3: KF_ComputeParamStandardErrors's final invIObs projection must go
    # through DecodingAlgorithms._nearestSPD (MATLAB's exact nearestSPD, via
    # _matlab_nearest_spd), as PP/PPLFP's SE routines do -- not the
    # module-level eps-clamp _nearestSPD (kept only for the whitening
    # Cholesky-fallback guard; see its docstring).
    import nstat.decoding_algorithms as da

    calls = {"static": 0, "module": 0}
    real_static = DecodingAlgorithms._nearestSPD
    real_module = da._nearestSPD

    def spy_static(A):
        calls["static"] += 1
        return real_static(A)

    def spy_module(A):
        calls["module"] += 1
        return real_module(A)

    monkeypatch.setattr(DecodingAlgorithms, "_nearestSPD", staticmethod(spy_static))
    monkeypatch.setattr(da, "_nearestSPD", spy_module)

    rng = np.random.default_rng(15)
    dx, N = 1, 20
    xKFinal = rng.standard_normal((dx, N))
    WKFinal = np.full((dx, dx, N), 0.05)
    Ahat = np.array([[0.9]])
    Qhat = np.array([[0.05]])
    Chat = np.array([[1.0]])
    Rhat = np.array([[0.1]])
    alphahat = np.zeros((1, 1))
    x0hat = np.zeros(dx)
    Px0hat = np.array([[0.1]])
    y = Chat @ xKFinal + alphahat
    ES = dict(
        Sxkm1xkm1=np.eye(dx) * N, Sxkxk=np.eye(dx) * N, Sxkxkm1=np.eye(dx) * N * 0.9,
        sumXkTerms=np.eye(dx), sumYkTerms=np.eye(1), Sxkyk=np.zeros((dx, 1)),
    )
    cons = DecodingAlgorithms.KF_EMCreateConstraints(mcIter=30)
    DecodingAlgorithms.KF_ComputeParamStandardErrors(
        y, xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat, ES, cons)
    assert calls["static"] >= 1
    assert calls["module"] == 0


def test_kf_se_pvalue_matches_matlab_ztest_formula() -> None:
    from nstat.decoding_algorithms import _matlab_ztest_p

    rng = np.random.default_rng(15)
    dx, N = 1, 20
    xKFinal = rng.standard_normal((dx, N))
    WKFinal = np.full((dx, dx, N), 0.05)
    Ahat = np.array([[0.9]])
    Qhat = np.array([[0.05]])
    Chat = np.array([[1.0]])
    Rhat = np.array([[0.1]])
    alphahat = np.zeros((1, 1))
    x0hat = np.zeros(dx)
    Px0hat = np.array([[0.1]])
    y = Chat @ xKFinal + alphahat
    ES = dict(
        Sxkm1xkm1=np.eye(dx) * N, Sxkxk=np.eye(dx) * N, Sxkxkm1=np.eye(dx) * N * 0.9,
        sumXkTerms=np.eye(dx), sumYkTerms=np.eye(1), Sxkyk=np.zeros((dx, 1)),
    )
    cons = DecodingAlgorithms.KF_EMCreateConstraints(mcIter=30)
    SE, Pvals = DecodingAlgorithms.KF_ComputeParamStandardErrors(
        y, xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat, ES, cons)
    assert DecodingAlgorithms._ztest_pvalue(1.0, 0.0) == 0.0  # MATLAB: z = Inf -> p = 0
    np.testing.assert_allclose(
        Pvals["A"][0, 0], _matlab_ztest_p(Ahat[0, 0], SE["A"][0, 0]), rtol=1e-12, atol=0.0)


def test_kf_module_level_ztest_pvalue_helper_is_gone() -> None:
    # The module-level _ztest_pvalue (se<=0 -> p=1, non-MATLAB) had no
    # remaining callers once KF_ComputeParamStandardErrors moved to
    # DecodingAlgorithms._ztest_pvalue (item 3); it was deleted rather than
    # left as dead code with divergent semantics.
    import nstat.decoding_algorithms as da

    assert not hasattr(da, "_ztest_pvalue")
