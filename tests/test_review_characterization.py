"""Characterization tests for previously-untested production helpers.

Codebase review 2026-10-04, item B8.  These tests pin the CURRENT behavior
(``main @ dcf0de6``) of code that had zero test references, so that later
behavior-preserving refactors (lazy ``scipy.stats`` imports, shared helpers,
loop hoists) cannot silently change it:

* both ``_nearestSPD`` definitions and both ``_ztest_pvalue`` definitions in
  ``nstat/decoding_algorithms.py`` (review finding C2: they differ
  numerically -- the module-level pair serves the KF family, the
  ``DecodingAlgorithms`` staticmethod pair serves the PP family);
* ``KF_ComputeParamStandardErrors`` and ``PP_ComputeParamStandardErrors``;
  ``mPPCO_ComputeParamStandardErrors`` is pinned as an exact alias of
  ``PPLFP_ComputeParamStandardErrors`` (MATLAB's deprecation shim);
* every ``scipy.stats`` use site in ``decoding_algorithms.py`` and
  ``analysis.py`` (module ``_ztest_pvalue`` -> ``norm.sf``; staticmethod
  ``_ztest_pvalue`` -> ``norm.cdf``; ``ComputeStimulusCIs`` -> ``norm.ppf``;
  ``Analysis.computeInvGausTrans`` -> ``norm.ppf``;
  ``Analysis.computeGrangerCausalityMatrix`` -> ``chi2.sf``);
* the time-rescaling helper ``fit._time_rescaled_uniforms`` (review finding
  C4), including its ``sum(counts) <= 1`` guard.  (Its near-duplicate
  ``analysis._time_rescaled_z`` had no callers and was deleted.)

These are characterization tests, not correctness tests: where the current
behavior is a known defect it is pinned and labelled as such.  A deliberate
behavior change must update the pin in the same commit.  (The former
``KNOWN DEFECT`` pins -- ``mPPCO_ComputeParamStandardErrors`` raising
``NameError`` on an undefined ``nearestSPD`` -- were replaced when the
``mPPCO_*`` family became MATLAB's deprecated aliases of ``PPLFP_*``: the
stale implementation they pinned was deleted.)

Tolerance policy
----------------
Expected values were captured on macOS arm64 (NumPy 2.5.3 with Accelerate,
SciPy 1.18.1).  CI runs Linux / OpenBLAS, whose BLAS/LAPACK reductions and
libm transcendental functions may differ from Accelerate in the last few
ULPs, so:

* ``assert_array_equal`` (exact) is used wherever the result is exact by
  construction on any IEEE-754 platform: diagonal inputs to ``_nearestSPD``
  (SVD/eigh of a diagonal matrix involve no rounding), the ``se <= 0`` /
  non-finite / ``z == 0`` p-value branches, p-values that underflow to exactly
  ``0.0``, empty guard outputs, shapes, keys and integer outputs.
* ``rtol=1e-12`` (``atol=0``) for direct, unamplified evaluations of libm /
  ``scipy.special`` functions (``exp``, ``expm1``, ``ndtr``, ``ndtri``).
* The ``*_ComputeParamStandardErrors`` outputs pass through BLAS products, a
  matrix (pseudo-)inverse of the observed-information matrix and a nearest-SPD
  projection, which amplify ULP-level differences.  Each pinned array carries
  its own ``rtol`` = ``max(1e-12, 10**ceil(log10(100 * s)))`` where ``s`` is the
  largest relative change of that output measured when every float input is
  perturbed by ``1e-15`` relative noise (8 trials) -- i.e. a 100x margin over
  the observed float-order sensitivity.  ``atol=0`` throughout, so exact zeros
  (structural off-diagonals, p-values that underflow) must stay exact zeros.
  The ``computeInvGausTrans`` autocorrelation uses the same measured rule.
* General (non-diagonal) ``_nearestSPD`` inputs: ``rtol=1e-12`` plus
  ``atol=1e-13 * max|A|`` (SVD/eigh are backward stable, errors are
  O(eps * ||A||); measured sensitivity ~3e-15 * max|A|).
* ``computeGrangerCausalityMatrix`` Gamma / deviance: ``rtol=1e-9`` (iterative
  IRLS GLM fits, converged to their 1e-8 tolerance).

Bit-exact same-machine before/after checks for the refactors that follow
this commit are done separately with the out-of-tree snapshot script, not here.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import nstat.decoding_algorithms as da
import nstat.fit as fit_mod
from nstat.analysis import Analysis
from nstat.decoding_algorithms import DecodingAlgorithms

_EPS = np.finfo(float).eps


@contextlib.contextmanager
def _legacy_global_rng(seed: int) -> Iterator[None]:
    """Seed NumPy's legacy global RNG for the block, then restore it.

    The ``*_ComputeParamStandardErrors`` functions draw their Monte-Carlo
    samples with ``np.random.randn`` (global state), so they can only be
    pinned by seeding that state.  The previous state is restored so other
    tests are unaffected.
    """
    saved = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(saved)


def _assert_pinned(actual, shape, values, rtol: float) -> None:
    arr = np.asarray(actual, dtype=float)
    assert arr.shape == tuple(shape)
    expected = np.asarray(values, dtype=float).reshape(shape)
    np.testing.assert_allclose(arr, expected, rtol=rtol, atol=0.0)


# ===========================================================================
# 1. _nearestSPD -- module-level (KF family) vs staticmethod (PP family)
# ===========================================================================


@pytest.mark.parametrize(
    ("diag", "module_expected", "static_expected"),
    [
        # Indefinite diagonal: the Higham projection is diag(max(d, 0)) exactly,
        # so Cholesky deterministically fails on the exact zero pivot and both
        # Cholesky-fallback branches run.  The module-level fallback clamps the
        # eigenvalues to eps; the staticmethod fallback adds
        # spacing(norm(A)) * I (k = 1 loop pass) to EVERY diagonal entry.
        (
            [2.0, -1.0, 0.5],
            [2.0, _EPS, 0.5],
            [2.0000000000000004, 4.440892098500626e-16, 0.5000000000000004],
        ),
        (
            [4.0, 0.0, -3.0, 1e-3],
            [4.0, _EPS, _EPS, 1e-3],
            [4.000000000000001, 8.881784197001252e-16, 8.881784197001252e-16, 0.0010000000000008882],
        ),
        # Positive-definite diagonal: Cholesky succeeds; both return the input.
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ],
)
def test_nearestSPD_both_definitions_on_diagonal_inputs_exact(diag, module_expected, static_expected) -> None:
    A = np.diag(np.asarray(diag, dtype=float))
    out_module = da._nearestSPD(A.copy())
    out_static = DecodingAlgorithms._nearestSPD(A.copy())
    np.testing.assert_array_equal(out_module, np.diag(module_expected))
    np.testing.assert_array_equal(out_static, np.diag(static_expected))


def test_nearestSPD_definitions_agree_when_cholesky_succeeds() -> None:
    rng = np.random.default_rng(101)
    L = rng.standard_normal((4, 4))
    M = L @ L.T + 0.5 * np.eye(4)
    out_module = da._nearestSPD(M.copy())
    out_static = DecodingAlgorithms._nearestSPD(M.copy())
    # Identical code path up to the successful Cholesky -> bit-identical.
    np.testing.assert_array_equal(out_module, out_static)
    np.testing.assert_array_equal(out_module, out_module.T)
    # The Higham projection of an SPD matrix is the matrix itself (up to
    # SVD round-off at the eps * ||M|| scale).
    np.testing.assert_allclose(out_module, M, rtol=0.0, atol=1e-13 * np.abs(M).max())
    _assert_pinned(out_module, (4, 4), _NEAREST_SPD_SPD_EXPECTED, rtol=1e-12)


def test_nearestSPD_both_definitions_on_general_indefinite_input() -> None:
    # Non-symmetric, indefinite input: exercises the symmetrization and the
    # SVD-based Higham step.  Its result has eigenvalues at round-off level in
    # the negative directions, so which Cholesky branch runs is decided by
    # round-off (on the capture machine both fallbacks run); the fallbacks only
    # move the result by O(eps * ||A||), below the pinning tolerance.
    rng = np.random.default_rng(202)
    A = rng.standard_normal((4, 4))
    out_module = da._nearestSPD(A.copy())
    out_static = DecodingAlgorithms._nearestSPD(A.copy())
    for out in (out_module, out_static):
        np.testing.assert_array_equal(out, out.T)
        np.testing.assert_allclose(
            out,
            np.asarray(_NEAREST_SPD_INDEF_EXPECTED).reshape(4, 4),
            rtol=1e-12,
            atol=1e-13 * np.abs(A).max(),
        )


# ===========================================================================
# 2. _ztest_pvalue -- module-level norm.sf (scalar) vs staticmethod 1 - norm.cdf
# ===========================================================================

# (param, se, module-level p, staticmethod p)
_ZTEST_CASES = [
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, 0.31731050786291415, 0.31731050786291415),
    (-1.96, 1.0, 0.04999579029644087, 0.04999579029644097),
    (5.0, 1.0, 5.733031437583869e-07, 5.733031438470704e-07),
    (8.0, 1.0, 1.244192114854348e-15, 1.3322676295501878e-15),
    # Large |z|: 1 - norm.cdf(z) cancels to exactly 0.0 for z >~ 8.3 while
    # norm.sf keeps the tail (review C2 divergence).
    (8.3, 1.0, 1.0411139489780493e-16, 0.0),
    (9.0, 1.0, 2.2571768119076647e-19, 0.0),
    (-20.0, 1.0, 5.50724823721231e-89, 0.0),
    (37.0, 1.0, 1.1451142445047847e-299, 0.0),
    (40.0, 1.0, 0.0, 0.0),
    # Guard branches.
    (1.0, 0.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0),
    (1.0, np.nan, 1.0, 1.0),
    (1.0, np.inf, 1.0, 1.0),
    (np.nan, 1.0, np.nan, np.nan),
]


def test_module_ztest_pvalue_scalar_values() -> None:
    for param, se, expected, _ in _ZTEST_CASES:
        p = da._ztest_pvalue(param, se)
        assert type(p) is float
        if expected == 0.0 or expected == 1.0 or np.isnan(expected):
            np.testing.assert_array_equal(p, expected)
        else:
            np.testing.assert_allclose(p, expected, rtol=1e-12, atol=0.0)


def test_staticmethod_ztest_pvalue_vectorized_values() -> None:
    params = np.array([c[0] for c in _ZTEST_CASES])
    ses = np.array([c[1] for c in _ZTEST_CASES])
    expected = np.array([c[3] for c in _ZTEST_CASES])
    with np.errstate(invalid="ignore"):
        p = DecodingAlgorithms._ztest_pvalue(params, ses)
    assert isinstance(p, np.ndarray) and p.shape == params.shape
    exact = (expected == 0.0) | (expected == 1.0) | np.isnan(expected)
    np.testing.assert_array_equal(p[exact], expected[exact])
    np.testing.assert_allclose(p[~exact], expected[~exact], rtol=1e-12, atol=0.0)
    # Scalar inputs come back as a 0-d ndarray, not a float.
    p0 = DecodingAlgorithms._ztest_pvalue(1.0, 1.0)
    assert isinstance(p0, np.ndarray) and p0.shape == ()


def test_ztest_definitions_diverge_in_the_large_z_tail() -> None:
    assert da._ztest_pvalue(9.0, 1.0) > 0.0
    assert float(DecodingAlgorithms._ztest_pvalue(9.0, 1.0)) == 0.0


# ===========================================================================
# 3. *_ComputeParamStandardErrors
# ===========================================================================


def _ar_path(rng, A, Q, x0, N):
    cholQ = np.linalg.cholesky(Q)
    x = np.zeros((A.shape[0], N))
    prev = np.asarray(x0, dtype=float)
    for k in range(N):
        prev = A @ prev + cholQ @ rng.standard_normal(A.shape[0])
        x[:, k] = prev
    return x


def _spd_slices(rng, dx, N, scale=0.05, floor=0.01):
    W = np.zeros((dx, dx, N))
    for k in range(N):
        L = scale * rng.standard_normal((dx, dx))
        W[:, :, k] = L @ L.T + floor * np.eye(dx)
    return W


def _expectation_sums(xK, WK, x0, Px0):
    # Only the two keys the SE functions read.
    dx, N = xK.shape
    Sxkxk = np.zeros((dx, dx))
    Sxkm1xkm1 = Px0 + np.outer(x0, x0)
    for k in range(N):
        Sxkxk += WK[:, :, k] + np.outer(xK[:, k], xK[:, k])
        if k < N - 1:
            Sxkm1xkm1 += WK[:, :, k] + np.outer(xK[:, k], xK[:, k])
    return {"Sxkm1xkm1": Sxkm1xkm1, "Sxkxk": Sxkxk}


def _history(dN, nW):
    numCells, N = dN.shape
    HkAll = np.zeros((N, nW, numCells))
    for c in range(numCells):
        for w in range(nW):
            HkAll[w + 1 :, w, c] = dN[c, : N - w - 1]
    return HkAll


def _kf_inputs(seed, N=40, nonpd_slice=None, px0=(0.1, 0.2)):
    rng = np.random.default_rng(seed)
    dx, dy = 2, 3
    A = np.array([[0.9, 0.1], [-0.05, 0.85]])
    Q = np.diag([0.05, 0.08])
    C = rng.standard_normal((dy, dx))
    R = np.diag(rng.uniform(0.05, 0.2, size=dy))
    alpha = rng.standard_normal((dy, 1))
    x0 = 0.1 * rng.standard_normal(dx)
    Px0 = np.diag(np.asarray(px0, dtype=float))
    xK = _ar_path(rng, A, Q, x0, N)
    y = C @ xK + alpha + np.sqrt(np.diag(R))[:, None] * rng.standard_normal((dy, N))
    WK = _spd_slices(rng, dx, N)
    if nonpd_slice is not None:
        # Exactly non-PD slice -> Cholesky fails -> _nearestSPD fallback.
        WK[:, :, nonpd_slice] = np.diag([0.02, -0.01])
    ES = _expectation_sums(xK, WK, x0, np.abs(Px0))
    return dict(y=y, xKFinal=xK, WKFinal=WK, Ahat=A, Qhat=Q, Chat=C, Rhat=R,
                alphahat=alpha, x0hat=x0, Px0hat=Px0, ExpectationSumsFinal=ES)


def _pp_inputs(seed, N=60, fit="poisson", history=True, nonpd_slice=None, px0=(0.1, 0.2)):
    rng = np.random.default_rng(seed)
    dx, numCells, nW = 2, 2, 2
    A = np.array([[0.9, 0.0], [0.05, 0.85]])
    Q = np.diag([0.02, 0.03])
    x0 = np.array([0.5, -0.3])
    Px0 = np.diag(np.asarray(px0, dtype=float))
    xK = _ar_path(rng, A, Q, x0, N)
    mu = np.array([-2.0, -1.5])
    beta = rng.standard_normal((dx, numCells))
    gamma = -0.5 * rng.uniform(0.2, 1.0, size=(nW, numCells)) if history else np.array(0.0)
    eta = mu[:, None] + beta.T @ xK
    p = 1.0 / (1.0 + np.exp(-eta)) if fit == "binomial" else np.minimum(np.exp(eta), 0.9)
    dN = (rng.uniform(size=p.shape) < p).astype(float)
    if history:
        HkAll = _history(dN, nW)
        windowTimes = np.arange(nW + 1, dtype=float) * 0.001
    else:
        HkAll = np.zeros((N, 0, numCells))
        windowTimes = None
    WK = _spd_slices(rng, dx, N)
    if nonpd_slice is not None:
        WK[:, :, nonpd_slice] = np.diag([0.02, -0.01])
    ES = _expectation_sums(xK, WK, x0, np.abs(Px0))
    return dict(dN=dN, xKFinal=xK, WKFinal=WK, Ahat=A, Qhat=Q, x0hat=x0, Px0hat=Px0,
                ExpectationSumsFinal=ES, fitType=fit, muhat=mu, betahat=beta,
                gammahat=gamma, windowTimes=windowTimes, HkAll=HkAll)


def _mppco_inputs(seed, N=40, nonpd_slice=None):
    kf = _kf_inputs(seed, N=N)
    rng = np.random.default_rng(seed + 100)
    dx, numCells, nW = 2, 2, 2
    mu = np.array([-2.0, -1.5])
    beta = rng.standard_normal((dx, numCells))
    gamma = -0.5 * rng.uniform(0.2, 1.0, size=(nW, numCells))
    eta = mu[:, None] + beta.T @ kf["xKFinal"]
    dN = (rng.uniform(size=eta.shape) < np.minimum(np.exp(eta), 0.9)).astype(float)
    WK = kf["WKFinal"].copy()
    if nonpd_slice is not None:
        WK[:, :, nonpd_slice] = np.diag([0.02, -0.01])
    kf.update(dN=dN, WKFinal=WK, fitType="poisson", muhat=mu, betahat=beta, gammahat=gamma,
              windowTimes=np.arange(nW + 1, dtype=float) * 0.001, HkAll=_history(dN, nW))
    return kf


def _call_kf(I, constraints):
    return DecodingAlgorithms.KF_ComputeParamStandardErrors(
        I["y"], I["xKFinal"], I["WKFinal"], I["Ahat"], I["Qhat"], I["Chat"], I["Rhat"],
        I["alphahat"], I["x0hat"], I["Px0hat"], I["ExpectationSumsFinal"], constraints,
    )


def _call_pp(I, constraints):
    return DecodingAlgorithms.PP_ComputeParamStandardErrors(
        I["dN"], I["xKFinal"], I["WKFinal"], I["Ahat"], I["Qhat"], I["x0hat"], I["Px0hat"],
        I["ExpectationSumsFinal"], I["fitType"], I["muhat"], I["betahat"], I["gammahat"],
        I["windowTimes"], I["HkAll"], constraints,
    )


def _call_mppco(I, constraints):
    return DecodingAlgorithms.mPPCO_ComputeParamStandardErrors(
        I["y"], I["dN"], I["xKFinal"], I["WKFinal"], I["Ahat"], I["Qhat"], I["Chat"], I["Rhat"],
        I["alphahat"], I["x0hat"], I["Px0hat"], I["ExpectationSumsFinal"], I["fitType"],
        I["muhat"], I["betahat"], I["gammahat"], I["windowTimes"], I["HkAll"], constraints,
    )


# name -> (caller, input builder, constraint factory kwargs, legacy RNG seed)
_SE_CASES = {
    # Default constraints (full A, diagonal Q/R, x0 + Px0 estimated), linear-
    # Gaussian model; WKFinal[:, :, 5] is non-PD -> module _nearestSPD fallback
    # in the Monte-Carlo draw loop.
    "KF-A": (_call_kf, lambda: _kf_inputs(11, nonpd_slice=5),
             ("KF", dict(mcIter=50)), 5),
    # Diagonal A, isotropic Q/R/Px0; Px0hat non-PD -> module _nearestSPD
    # fallback on the x0 draw.
    "KF-B": (_call_kf, lambda: _kf_inputs(21, px0=(0.2, -0.1)),
             ("KF", dict(AhatDiag=1, QhatIsotropic=1, RhatIsotropic=1, Px0Isotropic=1, mcIter=50)), 5),
    # Poisson with 2-window history (gamma block present); WKFinal[:, :, 7]
    # non-PD -> inline eigh-clip fallback; staticmethod _nearestSPD on the
    # (indefinite) inverse observed information.
    "PP-A": (_call_pp, lambda: _pp_inputs(12, nonpd_slice=7),
             ("PP", dict(mcIter=50)), 6),
    # Binomial without history; diagonal A, isotropic Q/Px0; Px0hat non-PD ->
    # inline eigh-clip fallback on the x0 draw.
    "PP-B": (_call_pp, lambda: _pp_inputs(22, fit="binomial", history=False, px0=(0.2, -0.1)),
             ("PP", dict(AhatDiag=1, QhatIsotropic=1, Px0Isotropic=1, mcIter=50)), 6),
}


def _constraints(spec):
    family, kwargs = spec
    factory = {
        "KF": DecodingAlgorithms.KF_EMCreateConstraints,
        "PP": DecodingAlgorithms.PP_EMCreateConstraints,
        "mPPCO": DecodingAlgorithms.mPPCO_EMCreateConstraints,
    }[family]
    return factory(**kwargs)


def _run_se_case(name):
    caller, build, spec, seed = _SE_CASES[name]
    with _legacy_global_rng(seed):
        return caller(build(), _constraints(spec))


def _check_se_against_expected(out, expected) -> None:
    SE, Pvals = out[0], out[1]
    if "nTerms" in expected:
        assert len(out) == 3 and out[2] == expected["nTerms"]
    else:
        assert len(out) == 2
    assert sorted(SE) == sorted(expected["SE"])
    assert sorted(Pvals) == sorted(expected["P"])
    for part, got in (("SE", SE), ("P", Pvals)):
        for key, (shape, values, rtol) in expected[part].items():
            _assert_pinned(got[key], shape, values, rtol)


@pytest.mark.parametrize("name", sorted(_SE_CASES))
def test_compute_param_standard_errors_pinned(name) -> None:
    _check_se_against_expected(_run_se_case(name), _SE_EXPECTED[name])


def test_kf_se_pvalues_reach_module_ztest_large_z_tail() -> None:
    # KF uses the module-level norm.sf p-value: tail p-values stay > 0 where
    # the PP-family 1 - norm.cdf form would return exactly 0.0.
    _, Pvals = _run_se_case("KF-A")
    p = np.concatenate([np.asarray(v, dtype=float).ravel() for v in Pvals.values()])
    assert np.any((p > 0.0) & (p < 1e-20))


def test_pp_se_large_z_pvalues_underflow_to_exact_zero() -> None:
    # PP uses the staticmethod 1 - norm.cdf p-value: well-determined
    # parameters (9 < |z| < 35) get p == 0.0 exactly, whereas the module-level
    # helper used by KF would report a positive tail probability.
    caller, build, spec, seed = _SE_CASES["PP-A"]
    inputs = build()
    with _legacy_global_rng(seed):
        SE, Pvals, _ = caller(inputs, _constraints(spec))
    params = {"A": inputs["Ahat"], "Q": inputs["Qhat"], "x0": inputs["x0hat"],
              "mu": inputs["muhat"], "beta": inputs["betahat"]}
    hits = 0
    for key, theta in params.items():
        theta = np.asarray(theta, dtype=float).ravel()
        se = np.asarray(SE[key], dtype=float).ravel()
        p = np.asarray(Pvals[key], dtype=float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs(theta / se)
        for zi, thi, sei, pi in zip(z, theta, se, p):
            if 9.0 < zi < 35.0:
                assert pi == 0.0
                assert da._ztest_pvalue(thi, sei) > 0.0
                hits += 1
    assert hits >= 1


def _pplfp_se_gold_inputs(nonpd_slice=None):
    """``PPLFP_ComputeParamStandardErrors`` inputs from the MATLAB gold fixture.

    Mirrors ``tools/parity/matlab/export_pplfp_gold_fixtures.m``: the
    ExpectationSums are rebuilt by ``PPLFP_EStep`` at the EM-converged
    parameters; the fixture's gamma is all-zero (scalar 0 + zero HkAll).
    """
    from pathlib import Path

    from scipy.io import loadmat

    from nstat.decoding.PPLFP import PPLFP

    fx = loadmat(
        Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "pplfp_SE.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    f = lambda k: np.asarray(fx[k], dtype=float)  # noqa: E731
    v = lambda k: np.asarray(fx[k], dtype=float).reshape(-1)  # noqa: E731
    dN = f("dN")
    HkAll = np.zeros((dN.shape[1], 1, dN.shape[0]))
    fit, delta = str(fx["fitType"]), float(fx["delta"])
    _, _, _, es = PPLFP.PPLFP_EStep(
        f("Ahat"), f("Qhat"), f("Chat"), f("Rhat"), f("y"), v("alphahat"), dN,
        v("muhat_new"), f("betahat_new"), fit, delta, np.array(0.0), HkAll, v("x0hat"), f("Px0hat"),
    )
    WK = f("WKFinal").copy()
    if nonpd_slice is not None:
        WK[:, :, nonpd_slice] = np.diag([0.02, -0.01])
    return dict(
        y=f("y"), dN=dN, xKFinal=f("xKFinal"), WKFinal=WK, Ahat=f("Ahat"), Qhat=f("Qhat"),
        Chat=f("Chat"), Rhat=f("Rhat"), alphahat=v("alphahat"), x0hat=v("x0hat"),
        Px0hat=f("Px0hat"), ExpectationSumsFinal=es, fitType=fit, muhat=v("muhat_new"),
        betahat=f("betahat_new"), gammahat=np.array(0.0), windowTimes=None, HkAll=HkAll,
    )


def _assert_mppco_se_is_pplfp_se(inputs, mc_iter=50) -> None:
    """mPPCO_ComputeParamStandardErrors(...) is bit-identical to PPLFP's."""
    from nstat.decoding.PPLFP import PPLFP
    from nstat.extras.matlab_rng import seeded_global_rng

    # PPLFP draws its Monte-Carlo samples via np.random.default_rng(); a
    # fresh seeded_global_rng block per call gives both calls one stream.
    with pytest.warns(DeprecationWarning, match="mPPCO_ComputeParamStandardErrors is deprecated"):
        with seeded_global_rng(42):
            got = _call_mppco(inputs, _constraints(("mPPCO", dict(mcIter=mc_iter))))
    with seeded_global_rng(42):
        expected = PPLFP.PPLFP_ComputeParamStandardErrors(
            inputs["y"], inputs["dN"], inputs["xKFinal"], inputs["WKFinal"], inputs["Ahat"],
            inputs["Qhat"], inputs["Chat"], inputs["Rhat"], inputs["alphahat"], inputs["x0hat"],
            inputs["Px0hat"], inputs["ExpectationSumsFinal"], inputs["fitType"], inputs["muhat"],
            inputs["betahat"], inputs["gammahat"], inputs["windowTimes"], inputs["HkAll"],
            PPLFP.PPLFP_EMCreateConstraints(mcIter=mc_iter),
        )
    assert len(got) == 3 and got[2] == expected[2]
    for part in (0, 1):
        assert sorted(got[part]) == sorted(expected[part]) and len(got[part]) > 0
        for key in got[part]:
            a = np.asarray(got[part][key])
            b = np.asarray(expected[part][key])
            assert a.shape == b.shape and np.array_equal(a, b), (part, key)
            assert np.all(np.isfinite(a.astype(float))), (part, key)


def test_mppco_se_final_projection_forwards_to_pplfp() -> None:
    # Replaces the KNOWN DEFECT pin ``..._raises_nameerror_on_final_projection``:
    # the stale body called an undefined ``nearestSPD(invIObs)``.  MATLAB's
    # mPPCO_ComputeParamStandardErrors forwards to PPLFP_ComputeParamStandardErrors;
    # so does the Python alias now.  All WKFinal slices are PD here, so the
    # path through the end-of-function SPD projection is the one exercised.
    _assert_mppco_se_is_pplfp_se(_pplfp_se_gold_inputs())


def test_mppco_se_draw_fallback_forwards_to_pplfp() -> None:
    # Replaces the KNOWN DEFECT pin ``..._raises_nameerror_on_draw_fallback``:
    # a non-PD WKFinal slice sends the Monte-Carlo draw loop into its
    # Cholesky fallback (formerly the undefined ``nearestSPD(WuTemp)``).
    _assert_mppco_se_is_pplfp_se(_pplfp_se_gold_inputs(nonpd_slice=3))


def test_mppco_se_synthetic_history_case_forwards_to_pplfp() -> None:
    # Replaces ``..._downstream_numerics_with_nearestSPD_bound_to_module_helper``,
    # which pinned the deleted stale implementation under a monkeypatched
    # ``nearestSPD`` (its ``mPPCO-patched`` expected values were removed with
    # it).  Same synthetic Poisson + 2-window-history input; the pinned
    # behavior is now "identical to PPLFP".
    _assert_mppco_se_is_pplfp_se(_mppco_inputs(13))


# ===========================================================================
# 4. Remaining scipy.stats use sites
# ===========================================================================


def _stimulus_ci_inputs():
    rng = np.random.default_rng(303)
    N, Dx = 6, 2
    xK = rng.normal(-1.0, 1.0, size=(N, Dx))
    Wku = np.zeros((N, Dx, Dx))
    for k in range(N):
        L = 0.3 * rng.standard_normal((Dx, Dx))
        Wku[k] = L @ L.T + 0.05 * np.eye(Dx)
    return xK, Wku


@pytest.mark.parametrize("fitType", ["poisson", "binomial", "identity"])
@pytest.mark.parametrize("alphaVal", [0.05, 0.1])
def test_compute_stimulus_cis_zscore_fallback_pinned(fitType, alphaVal) -> None:
    # 3-D covariance -> z-score fallback, the norm.ppf use site.
    xK, Wku = _stimulus_ci_inputs()
    ci, stim = DecodingAlgorithms.ComputeStimulusCIs(fitType, xK, Wku, 0.001, alphaVal=alphaVal)
    expected_ci, expected_stim = _STIMULUS_CI_EXPECTED[(fitType, alphaVal)]
    _assert_pinned(ci, (6, 2, 2), expected_ci, rtol=1e-12)
    _assert_pinned(stim, (6, 2), expected_stim, rtol=1e-12)
    # State-major orientation (Dx, N) is transposed back on output.
    ci_t, stim_t = DecodingAlgorithms.ComputeStimulusCIs(fitType, xK.T, Wku, 0.001, alphaVal=alphaVal)
    np.testing.assert_array_equal(ci_t, np.transpose(ci, (1, 0, 2)))
    np.testing.assert_array_equal(stim_t, stim.T)


def _inv_gaus_z():
    rng = np.random.default_rng(404)
    # Includes values that hit both clip bounds of U = 1 - exp(-Z).
    return np.concatenate([rng.exponential(size=10), [1e-9, 50.0]])


def test_analysis_compute_inv_gaus_trans_pinned() -> None:
    X, rhoSig, confSig = Analysis.computeInvGausTrans(_inv_gaus_z())
    assert X.shape == (12, 1)
    # The last two entries sit on the U clip bounds [1e-6, 1 - 1e-6].
    _assert_pinned(X, (12, 1), _INV_GAUS_EXPECTED["X"], rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(rhoSig.time, dtype=float), np.arange(1.0, 12.0))
    _assert_pinned(np.asarray(rhoSig.data, dtype=float), (11, 1), _INV_GAUS_EXPECTED["rho"],
                   rtol=_INV_GAUS_EXPECTED["rho_rtol"])
    conf = np.asarray(confSig.data, dtype=float)
    np.testing.assert_array_equal(conf[:, 0], np.full(11, 1.96 / np.sqrt(12.0)))
    np.testing.assert_array_equal(conf[:, 1], -conf[:, 0])
    # A single value has no lags.
    X1, rho1, conf1 = Analysis.computeInvGausTrans(np.array([0.7]))
    assert X1.shape == (1, 1) and np.asarray(rho1.data).size == 0 and np.asarray(conf1.data).size == 0


def _granger_trial():
    from nstat.CovColl import CovColl
    from nstat.Covariate import Covariate
    from nstat.Events import Events
    from nstat.History import History
    from nstat.nspikeTrain import nspikeTrain
    from nstat.nstColl import nstColl
    from nstat.Trial import Trial

    rng = np.random.default_rng(4)
    T, n = 10.0, 40
    t = np.arange(0.0, T, 0.05)
    stim = Covariate(t, np.sin(np.pi * t), "Stimulus", "time", "s", "", ["stim"])
    vel = Covariate(t, np.cos(np.pi * t), "Velocity", "time", "s", "", ["vel"])
    s1 = np.sort(np.round(rng.uniform(0.05, T - 0.1, n), 2))
    s2 = np.sort(np.round(np.clip(s1 + 0.05 + 0.01 * rng.integers(0, 3, n), 0.0, T - 0.1), 2))
    spikes = nstColl([
        nspikeTrain(np.unique(s1), "1", 20.0, 0.0, T - 0.05, makePlots=-1),
        nspikeTrain(np.unique(s2), "2", 20.0, 0.0, T - 0.05, makePlots=-1),
    ])
    trial = Trial(spikes, CovColl([stim, vel]), Events([0.2], ["cue"]), History([0.0, 0.05, 0.10]))
    trial.setEnsCovHist([0.0, 0.05, 0.10])
    return trial


def test_analysis_compute_granger_causality_matrix_pinned() -> None:
    from nstat.analysis import computeNeighbors

    trial = _granger_trial()
    # Same preparation as tests/test_workflow_fidelity.py: on a freshly built
    # Trial the Granger configs fail with "Covariate selector index out of
    # bounds"; computeNeighbors leaves covMask in the state Granger expects.
    computeNeighbors(trial, 0, trial.sampleRate, [0.0, 0.05, 0.10], 0)
    _, gammaMat, phiMat, devianceMat, sigMat = Analysis.computeGrangerCausalityMatrix(trial, "GLM", 0.95, 0)
    # GLM fits are IRLS solutions (converged, tol 1e-8); deviance feeds chi2.sf.
    _assert_pinned(gammaMat, (2, 2), _GRANGER_EXPECTED["gamma"], rtol=1e-9)
    _assert_pinned(devianceMat, (2, 2), _GRANGER_EXPECTED["deviance"], rtol=1e-9)
    np.testing.assert_array_equal(phiMat, np.zeros((2, 2)))
    np.testing.assert_array_equal(sigMat, np.array([[0, 0], [1, 0]]))


# ===========================================================================
# 5. Time-rescaling helper (fit._time_rescaled_uniforms)
# ===========================================================================
# analysis._time_rescaled_z (the C4 near-duplicate) had no callers and was
# deleted in the review follow-up; only fit.py's helper remains to pin.

_TR_COUNTS = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.99, 1.5, 0.0, 2.5, 1.0, 0.0, 1.0, 1.0])
_TR_LAM = np.array([0.1, 0.2, -0.5, 0.0, 0.3, 0.25, 0.125, 0.5, 1e-15, 0.75, 0.0625, np.nan, 0.5, 1e-13])


def test_time_rescaled_uniforms_values() -> None:
    # Accumulated rate per inter-spike interval: pure float addition with a
    # 1e-12 floor; round() is banker's rounding (1.5 -> 2 repeats, 2.5 -> 2
    # repeats); 0.99 is not a spike; a NaN rate poisons only its own interval.
    a = 0.1 + 0.2
    b = 1e-12 + 1e-12 + 0.3
    c = 0.25 + 0.125 + 0.5
    d = 1e-12 + 0.75
    # Last interval: lam floors to 1e-12, so z = 1e-12 exactly; -expm1(-z) =
    # 9.999999999995e-13 whereas 1-exp(-z) = 9.99978e-13 (pins the expm1 form).
    z = np.array([a, b, 0.0, c, 0.0, d, 0.0, 0.0625, np.nan, 1e-12])
    u = fit_mod._time_rescaled_uniforms(_TR_COUNTS, _TR_LAM)
    assert u.shape == z.shape
    finite = np.isfinite(z)
    np.testing.assert_allclose(u[finite], -np.expm1(-z[finite]), rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(np.isnan(u), np.isnan(z))
    # Zero accumulated rate between coincident spikes maps to exactly 0.
    np.testing.assert_array_equal(u[z == 0.0], 0.0)


@pytest.mark.parametrize(
    ("counts", "lam"),
    [
        # sum(counts) <= 1: fit.py's guard returns an empty float array.
        ([0.0, 0.0, 1.0, 0.0], [0.5, 0.25, 0.125, 1.0]),
        ([1.0], [0.3]),
        ([0.5, 0.5], [0.2, 0.2]),
        ([0.0, 0.0], [0.2, 0.2]),
        ([], []),
    ],
)
def test_time_rescaling_guard(counts, lam) -> None:
    u = fit_mod._time_rescaled_uniforms(np.asarray(counts, dtype=float), np.asarray(lam, dtype=float))
    assert u.shape == (0,) and u.dtype == float


def test_time_rescaling_shape_mismatch_error() -> None:
    with pytest.raises(ValueError, match="y and lam_per_bin must have the same shape"):
        fit_mod._time_rescaled_uniforms(np.ones(3), np.ones(4))


# ===========================================================================
# Pinned values (captured from main @ dcf0de6; see module docstring)
# ===========================================================================

# BEGIN GENERATED EXPECTED VALUES
_NEAREST_SPD_SPD_EXPECTED = (
    6.181989169569013, 1.318768267351434, -1.761934545667593, 0.42787232375714696,
    1.318768267351434, 4.78157159484162, -0.4209353906429621, -2.1930310454321074,
    -1.761934545667593, -0.4209353906429621, 3.1488292112839043, -2.683600778567927,
    0.42787232375714696, -2.1930310454321074, -2.683600778567927, 6.506717446549172,
)
_NEAREST_SPD_INDEF_EXPECTED = (
    1.819444946374128, -0.9328349344408816, 0.05973533020676621, -0.03018563026076329,
    -0.9328349344408816, 1.51648510200518, 0.033402475041821325, -0.09226728198265197,
    0.05973533020676621, 0.033402475041821325, 0.04297802583098373, 0.22769792847531034,
    -0.03018563026076329, -0.09226728198265197, 0.22769792847531034, 1.5057451661555874,
)
_SE_EXPECTED = {
    'KF-A': {
        'SE': {
            'A': ((2, 2), (
                0.09226265688446741, 0.084213100609868, 0.10149807504787269, 0.13507786389381707,
            ), 1e-11),
            'Q': ((2, 2), (0.019264845378021332, 0.0, 0.0, 0.028302012417214348), 1e-11),
            'C': ((3, 2), (
                0.18923285070223092, 0.12364474543878076, 0.06523695506573107, 0.07892238170049223,
                0.11603172968727139, 0.2122415822555749,
            ), 1e-11),
            'R': ((3, 3), (
                0.026095598281321505, 0.0, 0.0, 0.0, 0.05227552477025547, 0.0, 0.0, 0.0,
                0.046810981814159915,
            ), 1e-11),
            'alpha': ((3, 1), (0.11251098298512734, 0.08053624223651673, 0.07835042305232702), 1e-11),
            'Px0': ((2, 2), (0.19665103868764106, 0.0, 0.0, 0.2887962286776014), 1e-10),
            'x0': ((2,), (0.08119854366152648, 0.12212541422480616), 1e-10),
        },
        'P': {
            'A': ((2, 2), (
                1.760179217673524e-22, 0.235044766928039, 0.6222809864805882, 3.120575566143552e-10,
            ), 1e-09),
            'C': ((3, 2), (
                0.8566097326000915, 3.941272735015034e-28, 1.2458827289850973e-78,
                1.0067361789561677e-10, 0.010228708162944692, 0.01296134134121472,
            ), 1e-09),
            'R': ((3, 3), (
                0.02029676749240268, 0.0, 0.0, 0.0, 0.18389896912757164, 0.0, 0.0, 0.0,
                4.009585147372811e-05,
            ), 1e-10),
            'Q': ((2, 2), (0.009448060210326598, 0.0, 0.0, 0.0047037103211173175), 1e-10),
            'Px0': ((2, 2), (0.6110922328278972, 0.0, 0.0, 0.4886046663456981), 1e-11),
            'alpha': ((3,), (1.3970992005557263e-60, 2.8316242243496e-84, 0.21840503130170086), 1e-08),
            'x0': ((2,), (0.40207590080904565, 0.9109624312195514), 1e-11),
        },
    },
    'KF-B': {
        'SE': {
            'A': ((2, 2), (0.09281690223106626, 0.0, 0.0, 0.10055611042234533), 1e-11),
            'Q': ((1, 1), (0.014198791278429702,), 1e-11),
            'C': ((3, 2), (
                0.22704323424120207, 0.32756063856283424, 0.1114173561374298, 0.06943202614920814,
                0.39991685695975693, 0.1996801286799998,
            ), 1e-11),
            'R': ((1, 1), (0.026347698850421108,), 1e-12),
            'alpha': ((3, 1), (0.06787476420965521, 0.07708928754878842, 0.07767240385165608), 1e-11),
            'Px0': ((1, 1), (0.15351587987939724,), 1e-11),
            'x0': ((2,), (0.16220934104515716, 0.1862809038987016), 1e-11),
        },
        'P': {
            'A': ((2, 2), (3.1198875918574794e-22, 0.0, 0.0, 2.839310649645488e-17), 1e-09),
            'C': ((3, 2), (
                0.11406138512826575, 3.990026106243555e-06, 7.541745082836943e-58,
                2.406232321171041e-130, 0.9058159938620606, 6.167719240200466e-05,
            ), 1e-08),
            'R': ((1, 1), (1.6459926542167364e-05,), 1e-10),
            'Q': ((1, 1), (0.0004292316265384852,), 1e-10),
            'Px0': ((1, 1), (0.19264413700104588,), 1e-10),
            'alpha': ((3,), (1.0820000177965435e-34, 3.5511722226718174e-14, 2.074985220111852e-16), 1e-09),
            'x0': ((2,), (0.296102534678345, 0.39904409790319395), 1e-10),
        },
    },
    'PP-A': {
        'nTerms': 20,
        'SE': {
            'A': ((2, 2), (
                0.06013266657213476, 0.06350251258338986, 0.06809217476544484, 0.0650667578445929,
            ), 1e-11),
            'Q': ((2, 2), (0.0024831666432183945, 0.0, 0.0, 0.004592330650127187), 1e-09),
            'Px0': ((2, 2), (0.31684378243445477, 0.0, 0.0, 0.3393741272010992), 1e-12),
            'x0': ((2,), (0.03412819385742489, 0.04980907226161009), 1e-11),
            'mu': ((2,), (0.329948147384861, 0.2729481261784977), 1e-12),
            'beta': ((2, 2), (
                1.1714608531082702, 0.8865662045352912, 1.233966459267492, 0.9365929454817485,
            ), 1e-12),
            'gamma': ((2, 2), (
                0.7836513685452229, 0.5171729924589396, 0.8154577903881022, 0.5649207539407073,
            ), 1e-12),
        },
        'P': {
            'A': ((2, 2), (0.0, 1.0, 0.4627666689738217, 0.0), 1e-11),
            'Q': ((2, 2), (8.881784197001252e-16, 0.0, 0.0, 6.462430590659096e-11), 1e-12),
            'Px0': ((2, 2), (0.7522963091431536, 0.0, 0.0, 0.5556465531050687), 1e-12),
            'x0': ((2,), (0.0, 1.7121433160127708e-09), 1e-12),
            'mu': ((2,), (1.3480883076510963e-09, 3.8949516945052665e-08), 1e-12),
            'beta': ((2, 2), (
                0.44940637758315494, 0.395355041021082, 0.48227775141545326, 0.11579933609187498,
            ), 1e-11),
            'gamma': ((2, 2), (
                0.8758981431049202, 0.8984254943129535, 0.7723754465777457, 0.40688448904635566,
            ), 1e-12),
        },
    },
    'PP-B': {
        'nTerms': 12,
        'SE': {
            'A': ((2, 2), (0.0935003345877258, 0.0, 0.0, 0.05529039187926433), 1e-11),
            'Q': ((1, 1), (0.0009130221878309711,), 1e-07),
            'Px0': ((1, 1), (0.40115888383317905,), 1e-12),
            'x0': ((2,), (0.04655323846366523, 0.6543298619699379), 1e-11),
            'mu': ((2,), (0.4849659972895409, 0.3718077183438585), 1e-12),
            'beta': ((2, 2), (
                0.10344969863385285, 0.14180593306432998, 0.18542978991097495, 0.06703763312987136,
            ), 1e-11),
        },
        'P': {
            'A': ((2, 2), (0.0, 0.0, 0.0, 0.0), 1e-12),
            'Q': ((1, 1), (0.0,), 1e-12),
            'Px0': ((1, 1), (0.6180925048535364,), 1e-12),
            'x0': ((2,), (0.0, 0.6466045051468337), 1e-12),
            'mu': ((2,), (3.723481348050228e-05, 5.475520170428183e-05), 1e-12),
            'beta': ((2, 2), (0.0, 2.220446049250313e-16, 0.016607936804470924, 0.0), 1e-10),
        },
    },
}
_STIMULUS_CI_EXPECTED = {
    ('poisson', 0.05): (
        (
            0.0788284447979953, 0.3694694496507913, 0.06439825205504865, 0.8606719690862922,
            0.02255831050525213, 1.0436660950957732, 0.2612621964435571, 0.8301098787981532,
            0.0822745647993191, 0.4245918454367515, 0.6488282306228291, 5.371521911001097,
            0.03843544187936469, 0.43247586744465244, 0.08598431732655111, 0.2088155076369478,
            0.029396911215224468, 0.33029948137035925, 0.2702932815306859, 1.1181534043853114,
            0.03812063169443668, 0.3457844333955482, 0.6332864905140123, 1.8054147622971999,
        ),
        (
            0.17065960891887427, 0.23542678352713844, 0.15343840404857723, 0.46569982845637864,
            0.18690401092712935, 1.8668677128459314, 0.1289278909600121, 0.13399574198966244,
            0.09853823891403345, 0.549753902147146, 0.11481080537624133, 1.0692730141256646,
        ),
    ),
    ('poisson', 0.1): (
        (
            0.08925120501528648, 0.3263227887103013, 0.07932072181026699, 0.6987552449977832,
            0.030702309498887255, 0.7668264772663999, 0.2867051386530658, 0.7564438197486818,
            0.0938766823907166, 0.3721169987159985, 0.7689926546937661, 4.532156498496724,
            0.04669139667780905, 0.35600565093605085, 0.09234125396388214, 0.1944402756148725,
            0.035707261289103794, 0.27192745054469114, 0.30297517956052883, 0.9975383243089522,
            0.04551369632200974, 0.28961657910361327, 0.6889287951356701, 1.659597895762545,
        ),
        (
            0.17065960891887427, 0.23542678352713844, 0.15343840404857723, 0.46569982845637864,
            0.18690401092712935, 1.8668677128459314, 0.1289278909600121, 0.13399574198966244,
            0.09853823891403345, 0.549753902147146, 0.11481080537624133, 1.0692730141256646,
        ),
    ),
    ('binomial', 0.05): (
        (
            0.07306856356828402, 0.2697902094457123, 0.06050202725409784, 0.46255975442513747,
            0.022060659302750113, 0.5106832753160019, 0.2071434450190063, 0.4535847210131954,
            0.07602004840109577, 0.29804455697033705, 0.3935086860914192, 0.8430516266022101,
            0.03701283712909882, 0.3019079603875856, 0.07917638952487374, 0.17274390204105725,
            0.02855741152411348, 0.24828956636900534, 0.2127802181280422, 0.5278906627208145,
            0.03672081117607266, 0.2569389456549885, 0.3877375427961266, 0.643546468265835,
        ),
        (
            0.14578072705223133, 0.19056312091194585, 0.13302695966252495, 0.31773206178705543,
            0.1574718841678971, 0.6511872537685728, 0.11420383178802948, 0.11816247365669909,
            0.08969941639122551, 0.3547362593412254, 0.10298680710893669, 0.5167384906807317,
        ),
    ),
    ('binomial', 0.1): (
        (
            0.08193812832553528, 0.24603572485368608, 0.07349133599253801, 0.41133367920738634,
            0.029787756577177246, 0.4340134626309309, 0.2228211655027594, 0.4306678137060578,
            0.08582016958762198, 0.27119917548155054, 0.43470652783851615, 0.8192386639330008,
            0.044608560676057146, 0.2625399464156363, 0.08453517033142774, 0.16278777565063166,
            0.03447621024174377, 0.213791636015278, 0.23252567225625673, 0.4993838226628519,
            0.043532376937883636, 0.2245757256819077, 0.40790872718842425, 0.6240033120821501,
        ),
        (
            0.14578072705223133, 0.19056312091194585, 0.13302695966252495, 0.31773206178705543,
            0.1574718841678971, 0.6511872537685728, 0.11420383178802948, 0.11816247365669909,
            0.08969941639122551, 0.3547362593412254, 0.10298680710893669, 0.5167384906807317,
        ),
    ),
    ('identity', 0.05): (
        (
            -2.5404813726601088, -0.9956872222535109, -2.74266878824253, -0.15004183543425698,
            -3.7916517441081865, 0.042739606019451815, -1.34223079184141, -0.18619720285933383,
            -2.497693273752635, -0.8566269351806094, -0.43258726507017453, 1.681111278237633,
            -3.258775279450858, -0.8382287519827916, -2.4535903560256385, -1.5663041554396593,
            -3.5268656708634145, -1.1077555167641637, -1.3082476816109867, 0.11167857854482044,
            -3.266999629125759, -1.0619397229534127, -0.45683236757736045, 0.5907903505944723,
        ),
        (
            -1.7680842974568098, -1.4463553118383934, -1.8744560690443675, -0.7642139973503719,
            -1.6771601044666222, 0.6242620065837292, -2.0485020157168248, -2.0099472557326488,
            -2.3173105938137892, -0.5982845515330831, -2.1644696760395856, 0.06697899150855591,
        ),
    ),
    ('identity', 0.1): (
        (
            -2.4163003568416968, -1.119868238071923, -2.5342558753937947, -0.35845474828299184,
            -3.4834173992416844, -0.26549473884705055, -1.2493009828036077, -0.279127011897136,
            -2.365773247452925, -0.9885469614803193, -0.26267386128574466, 1.511197874453203,
            -3.0641953565934346, -1.0328086748402148, -2.382264282139726, -1.6376302293255716,
            -3.3324012134251135, -1.302219974202465, -1.1941043924698564, -0.002464710596309816,
            -3.0897419802559325, -1.239197371823239, -0.37261735854021294, 0.5065753415573248,
        ),
        (
            -1.7680842974568098, -1.4463553118383934, -1.8744560690443675, -0.7642139973503719,
            -1.6771601044666222, 0.6242620065837292, -2.0485020157168248, -2.0099472557326488,
            -2.3173105938137892, -0.5982845515330831, -2.1644696760395856, 0.06697899150855591,
        ),
    ),
}
_INV_GAUS_EXPECTED = {
    'X': (
        -1.5900192788667709, -0.10009821582892103, -0.2155903296791741, -0.034292202701647456,
        0.03678353395119202, -0.7971714826592825, -1.3742817046709028, -0.9006454393929771,
        -1.0555906625400677, -0.4591982739459349, -4.753424308822899, 4.753424308817087,
    ),
    'rho': (
        -0.4496348700623757, 0.0521414520293957, -0.043653748409572694, 0.0023593078698239105,
        -0.07688207295885187, -0.06541490151983162, 0.022510835476420996, 0.038581645475065235,
        -0.004503783598624861, 0.13802358899290268, -0.1135274532943518,
    ),
    'rho_rtol': 1e-11,
}
_GRANGER_EXPECTED = {
    'gamma': (0.0, 0.0, -1074.616504558921, 0.0),
    'deviance': (0.0, 0.0, 2149.233009117842, 0.0),
}
# END GENERATED EXPECTED VALUES
