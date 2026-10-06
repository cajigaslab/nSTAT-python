"""End-to-end ``PP_EM`` / ``PPLFP_EM`` against the MATLAB gold fixture.

Gold: ``tests/parity/fixtures/matlab_gold/em_drivers.mat``, captured by
``tools/parity/matlab/capture_em_drivers.m`` from the repaired MATLAB
(``fix/pp-em`` @ ``aa88a2b``, pending upstream merge as nSTAT PR #135): the
real drivers on synthetic data (dx = 2, C = 4 cells, delta = 1 ms):

* ``pp_pois`` / ``pp_binom`` -- PP_EM, explicit windows [0 2 5 10] ms (nW = 3
  != C), N = 800, SEs requested (13 outputs, ``mcIter = 100``);
* ``pp_defwin`` -- PP_EM with ``windowTimes = []`` and a nonzero 3 x 1 shared
  gamma: MATLAB's default-window rule (``0:delta:size(gamma,1)*delta``) and
  shared-column expansion, through the driver itself;
* ``lfp_def`` -- the bare default call ``PPLFP_EM(y, dN, A0, Q0, C0, R0,
  alpha0, mu0, beta0)`` (no history, NewtonRaphson, x0 / Px0 fixed,
  ``mcIter = 1000``);
* ``pp_sep`` -- PP_EM with a separated history window (a refractory cell
  with no spike after a spike in the (0, 1] ms window).  The Newton step walks
  that coefficient by -1 per step (at most 99 per EM iteration) until exp()
  underflows (MATLAB keeps the previous value on the 0/0 there): MATLAB's run
  takes 11 iterations and ends at -741.5 .. -743.5.  MATLAB returns without
  SEs only (10 outputs): with SEs requested its observed information is
  singular and nearestSPD never returns.

The Monte Carlo draws (Newton M-step, SE pass) use MATLAB's Ziggurat ``randn``,
which Python does not reproduce, so the comparison is split:

* **Deterministic, tight** (``rtol 1e-10``): the returned estimates are the
  inputs of the selected iterate's E-step, so ``PP_EStep`` / ``PPLFP_EStep`` at
  MATLAB's returned estimates, in the original coordinates and with the
  history Python builds from the same windows, must reproduce ``xKFinal`` /
  ``WKFinal``, MATLAB's own E-step there (``es_*``), ``IC.llcomp`` (= its
  logll, F10) and, for PP, ``IC.llobs`` (= its ``sumPPll``); the closed-form
  M-step updates (A, Q, [C, R, alpha,] x0, Px0) on those sums, for three
  constraint sets each (``ms1..3``), and the parameter count read from
  ``IC.AIC``.  Measured: <= 2.2e-13 relative.
* **Monte Carlo dependent, regression pin** (one Python seed,
  ``seeded_global_rng(1)``): every estimate, ``xKFinal``, ``IC.llcomp`` and
  the checkable SEs, each within twice the largest deviation from the gold
  measured over Python seeds 1..8 (rounded up to one significant digit; table
  in ``_MC_ATOL``).  Seed 1 is one of those seeds, so this pins the seed-1 run
  near MATLAB but cannot see a bias smaller than the tolerance; the bias check
  below does that.  For calibration, MATLAB's
  own spread over ``rng(1..8)`` on pp_pois / pp_sep is of the same size
  (max |beta - gold| 0.078 / 0.185, Python 0.070 / 0.175).  The iteration at
  which EM stops (stop on the first likelihood decrease or a change below
  1e-3) is itself Monte Carlo dependent.  On pp_sep it ranges over 6..11 on
  both sides (MATLAB ``rng(1..8)``: 11, 6, 8, 10, 9, 8, 10, 8; Python seeds
  1..8 and 101..120: 6 (3 seeds), 7 (2), 8 (4), 9 (11), 10 (7), 11 (1)), and
  it dominates the estimates: EM is still moving there, the separated
  coefficient walks by up to -99 per iteration (it ends near -397.5, -496.5,
  -595.5, -694.5 or at the underflow), and e.g. ``Ahat`` is off MATLAB's
  11-iteration value by about 1.4e-3 per iteration short of 11, against a
  spread of < 1e-3 among runs that stop at the same iteration.  So pp_sep is
  compared at seed 1, which stops at MATLAB's 11 (asserted), with ``_MC_ATOL``
  = 3 x the largest deviation from the mean of the runs that stop at the same
  iteration (28 Python seeds), and the separated coefficients to 1e-2.
  (Before nstat-python's Newton step divided by its pivots, a denormal pivot
  made that step -Inf and every pp_sep run stopped by iteration 8, at about
  -694.5.)
* **Not checkable at mcIter = 100** (``_MC_UNCHECKABLE``): in the PP cases
  ``SE.beta`` and the off-diagonal ``SE.A`` vary from seed to seed by 10..71 %
  of their value (coefficient of variation over held-out seeds 101..120), as
  the 100-draw missing information is subtracted from a complete information
  of similar size; they are only checked to be finite and positive (``SE.A``
  is compared on its diagonal, CV <= 2 %).  At lfp_def's mcIter = 1000 every
  SE block varies by <= 3 % and is compared.
* **Bias check** (held-out seeds 101..106, ``test_monte_carlo_estimates_show_no_bias``):
  for the cases whose stopping iteration is stable across seeds (pp_pois,
  pp_binom, pp_defwin, lfp_def), MATLAB's value of every key estimate and
  checkable SE against the Python seed mean: z = (MATLAB - mean) /
  (sd sqrt(1 + 1/n)), the prediction-interval statistic of one more draw from
  the same distribution.  Required: |z| <= 6 for each entry, and over a case
  RMS(z) <= 2 and |mean(z)| <= 1, so a shift of the Python distribution by
  about 2 Monte Carlo sd on average is caught.  Measured (51-53 entries per
  case): seeds 101..106 max |z| 4.2, RMS 0.5..1.4; seeds 101..120 max |z| 2.8,
  RMS 0.4..1.1, |mean z| <= 0.14.  pp_sep is not included: its estimates are set by
  the stopping iteration (6..11 above), and of 28 Python seeds only seed 1
  stops at MATLAB's 11.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from nstat.decoding.PPLFP import PPLFP
from nstat.decoding_algorithms import DecodingAlgorithms, _compute_history_terms, _em_history_windows

FIXTURE = Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "em_drivers.mat"
CASES = ["pp_pois", "pp_binom", "pp_defwin", "lfp_def", "pp_sep"]
RTOL, ATOL = 1e-10, 1e-12

# 2 x the largest |Python - MATLAB| over Python seeds 1..8, one significant digit
# (measured on macOS arm64 / Accelerate against MATLAB R2025b).  "SE.A" in the
# PP cases is its diagonal (the off-diagonal is in _MC_UNCHECKABLE).
_MC_ATOL = {
    "pp_pois": {"muhat": 5e-02, "betahat": 2e-01, "gammahat": 9e-02, "Ahat": 4e-03, "Qhat": 2e-05, "xKFinal": 3e-01,
                "IC.llcomp": 2e+00, "SE.A": 7e-04, "SE.Q": 1e-05, "SE.mu": 5e-03, "SE.gamma": 4e-02},
    "pp_binom": {"muhat": 4e-02, "betahat": 2e-01, "gammahat": 2e-01, "Ahat": 2e-03, "Qhat": 3e-05, "xKFinal": 1e-01,
                 "IC.llcomp": 5e-01, "SE.A": 3e-04, "SE.Q": 3e-06, "SE.mu": 2e-03, "SE.gamma": 4e-02},
    "pp_defwin": {"muhat": 2e-02, "betahat": 8e-02, "gammahat": 5e-02, "Ahat": 2e-03, "Qhat": 2e-05,
                  "xKFinal": 6e-02, "IC.llcomp": 2e+00, "SE.A": 4e-04, "SE.Q": 4e-06, "SE.mu": 2e-03,
                  "SE.gamma": 4e-02},
    "lfp_def": {"muhat": 2e-02, "betahat": 3e-02, "Ahat": 5e-06, "Qhat": 7e-07, "Chat": 9e-05, "Rhat": 3e-06,
                "alphahat": 2e-05, "xKFinal": 9e-04, "IC.llcomp": 7e-02, "SE.A": 2e-04, "SE.Q": 2e-05,
                "SE.C": 4e-03, "SE.R": 2e-03, "SE.alpha": 2e-03, "SE.mu": 3e-03, "SE.beta": 8e-03},
    # pp_sep: 3 x the largest deviation from the mean of the runs that stop at
    # the same iteration, over Python seeds 1..8 and 101..120 (see the module
    # docstring); seed 1 stops at MATLAB's 11.
    "pp_sep": {"muhat": 2e-02, "betahat": 3e-01, "gammahat": 8e-02, "Ahat": 3e-03, "Qhat": 5e-05, "xKFinal": 5e-01,
               "IC.llcomp": 7e-01},
}

# Not determined at the PP cases' mcIter = 100: the seed-to-seed coefficient of
# variation of each entry over held-out seeds 101..120 (20 runs).
_MC_UNCHECKABLE = {
    "pp_pois": {"SE.beta": "CV 0.27..0.64", "SE.A offdiag": "CV 0.46, 0.71"},
    "pp_binom": {"SE.beta": "CV 0.10..0.58", "SE.A offdiag": "CV 0.40, 0.47"},
    "pp_defwin": {"SE.beta": "CV 0.10..0.57", "SE.A offdiag": "CV 0.29, 0.49"},
}
_BIAS_SEEDS = range(101, 107)
_BIAS_CASES = ["pp_pois", "pp_binom", "pp_defwin", "lfp_def"]


@pytest.fixture(scope="module")
def gold() -> dict:
    return loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)


def _get(gold, case):
    return lambda key: gold[f"{case}_{key}"]


def _history(gold, case):
    """The windows and history tensor Python's driver builds for this case."""
    f = _get(gold, case)
    dN = np.atleast_2d(f("dN")).astype(float)
    wt = np.asarray(f("windowTimes"), dtype=float)
    if f("family") != "PP":
        return None, np.zeros((dN.shape[1], 1, dN.shape[0])), np.array(0.0)
    g0 = np.asarray(f("gamma0"), dtype=float)
    if wt.size == 0 and g0.ndim == 1:
        g0 = g0.reshape(-1, 1)  # the MATLAB 3 x 1 column (squeeze_me reads it 1-D)
    gamma, wt = _em_history_windows(g0, None if wt.size == 0 else wt, float(f("delta")), dN.shape[0])
    return wt, _compute_history_terms(dN, float(f("delta")), wt), gamma


def test_fixture_provenance(gold) -> None:
    assert list(gold["case_names"]) == CASES
    assert "aa88a2b" in str(gold["matlab_source_note"])


@pytest.mark.parametrize("case", CASES)
def test_driver_history_matches_matlab(gold, case) -> None:
    # pp_defwin: windowTimes = [] and a 3 x 1 gamma -> MATLAB's driver used
    # 0:delta:3*delta and expanded gamma to 3 x C (its returned gammahat).
    f = _get(gold, case)
    wt, H, gamma0 = _history(gold, case)
    if f("family") == "PP":
        np.testing.assert_array_equal(wt, np.asarray(f("es_windowTimes"), dtype=float))
        np.testing.assert_array_equal(H, np.asarray(f("es_HkAll"), dtype=float).reshape(H.shape))
        assert np.shape(f("gammahat")) == np.shape(gamma0) == (3, 4)


@pytest.mark.parametrize("case", CASES)
def test_estep_at_the_returned_estimates_reproduces_the_driver(gold, case) -> None:
    f = _get(gold, case)
    dN = np.atleast_2d(f("dN")).astype(float)
    _, H, _ = _history(gold, case)
    IC = f("IC")
    if f("family") == "PP":
        x, W, ll, ES = DecodingAlgorithms.PP_EStep(
            f("Ahat"), f("Qhat"), dN, np.ravel(f("muhat")), f("betahat"), str(f("fitType")),
            np.asarray(f("gammahat"), dtype=float), H, np.ravel(f("x0hat")), np.atleast_2d(f("Px0hat")))
        # IC.llobs is the E-step's observation term (F10).
        np.testing.assert_allclose(ES["sumPPll"], IC.llobs, rtol=RTOL)
    else:
        x, W, ll, ES = PPLFP.PPLFP_EStep(
            f("Ahat"), f("Qhat"), f("Chat"), f("Rhat"), np.atleast_2d(f("y")), np.ravel(f("alphahat")), dN,
            np.ravel(f("muhat")), f("betahat"), str(f("fitType")), float(f("delta")), np.array(0.0), H,
            np.ravel(f("x0hat")), np.atleast_2d(f("Px0hat")))
    np.testing.assert_allclose(x, f("xKFinal"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(W, f("WKFinal"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(x, f("es_x_K"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(W, f("es_W_K"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ll, f("es_logll"), rtol=RTOL)
    np.testing.assert_allclose(ll, IC.llcomp, rtol=RTOL)  # IC.llcomp identity (F10)
    for key, value in ES.items():
        np.testing.assert_allclose(np.asarray(value, dtype=float).ravel(),
                                   np.atleast_1d(np.asarray(f(f"es_ES_{key}"), dtype=float)).ravel(),
                                   rtol=RTOL, atol=1e-9, err_msg=key)


@pytest.mark.parametrize("j", [1, 2, 3])
@pytest.mark.parametrize("case", [c for c in CASES if c != "pp_sep"])
def test_closed_form_mstep_updates_match_matlab(gold, case, j) -> None:
    # A, Q (C, R, alpha), x0 and Px0 given the E-step sums at the returned
    # estimates, for the default constraints, (AhatDiag, full Q / R, x0 and
    # Px0 estimated) and (isotropic Q / R / Px0).  No Monte Carlo is involved.
    f = _get(gold, case)
    q = lambda key: gold[f"{case}_ms{j}_{key}"]  # noqa: E731
    dN = np.atleast_2d(f("dN")).astype(float)
    wt, H, _ = _history(gold, case)
    dx = 2
    ES = {k[len(case) + 7:]: np.asarray(v, dtype=float) for k, v in gold.items() if k.startswith(f"{case}_es_ES_")}
    for key in ("Sxkm1xkm1", "Sxkxkm1", "Sxkm1xk", "Sxkxk", "sumXkTerms"):
        ES[key] = ES[key].reshape(dx, dx)
    cons = [int(v) for v in np.ravel(q("cons"))]
    np.random.seed(7)
    if f("family") == "PP":
        out = DecodingAlgorithms.PP_MStep(
            dN, f("es_x_K"), f("es_W_K"), np.ravel(f("x0hat")), np.atleast_2d(f("Px0hat")), ES, str(f("fitType")),
            np.ravel(f("muhat")), f("betahat"), np.asarray(f("gammahat"), dtype=float), wt, H,
            DecodingAlgorithms.PP_EMCreateConstraints(*cons), "NewtonRaphson", float(f("delta")))
        pairs = [(out[0], "Ahat"), (out[1], "Qhat"), (out[5], "x0hat"), (out[6], "Px0hat")]
    else:
        y = np.atleast_2d(f("y"))
        ES["Sxkyk"] = ES["Sxkyk"].reshape(dx, y.shape[0])
        for key in ("Sykyk", "sumYkTerms"):
            ES[key] = ES[key].reshape(y.shape[0], y.shape[0])
        out = PPLFP.PPLFP_MStep(
            dN, y, f("es_x_K"), f("es_W_K"), np.ravel(f("x0hat")), np.atleast_2d(f("Px0hat")), ES,
            str(f("fitType")), np.ravel(f("muhat")), f("betahat"), np.array(0.0), None, H,
            PPLFP.PPLFP_EMCreateConstraints(*cons), "NewtonRaphson", float(f("delta")))
        pairs = [(out[0], "Ahat"), (out[1], "Qhat"), (out[2], "Chat"), (out[3], "Rhat"), (out[4], "alphahat"),
                 (out[8], "x0hat"), (out[9], "Px0hat")]
    for got, key in pairs:
        np.testing.assert_allclose(np.ravel(got), np.ravel(q(key)), rtol=RTOL, atol=ATOL, err_msg=key)


def _n_terms(IC) -> int:
    return int(round((IC.AIC + 2 * IC.llobs) / 2))


def _run_driver(gold, case, seed) -> dict:
    """Python's PP_EM / PPLFP_EM on the case's inputs, under seeded_global_rng(seed)."""
    import warnings

    from nstat.extras.matlab_rng import seeded_global_rng

    f = _get(gold, case)
    dN = np.atleast_2d(f("dN")).astype(float)
    with seeded_global_rng(seed), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        if f("family") == "PP":
            wt = np.asarray(f("windowTimes"), dtype=float)
            g0 = np.asarray(f("gamma0"), dtype=float)
            if wt.size == 0 and g0.ndim == 1:
                g0 = g0.reshape(-1, 1)
            o = DecodingAlgorithms.PP_EM(
                dN, f("A0"), f("Q0"), np.ravel(f("mu0")), f("beta0"), str(f("fitType")), float(f("delta")), g0,
                None if wt.size == 0 else wt, None, None,
                DecodingAlgorithms.PP_EMCreateConstraints(*[int(v) for v in np.ravel(f("cons"))]))
            run = dict(xKFinal=o[0], Ahat=o[2], Qhat=o[3], muhat=o[4], betahat=o[5], gammahat=o[6],
                       IC=o[9], SE=o[10], Pvals=o[11], nIter=o[12])
        else:
            o = PPLFP.PPLFP_EM(np.atleast_2d(f("y")), dN, f("A0"), f("Q0"), f("C0"), f("R0"),
                               np.ravel(f("alpha0")), np.ravel(f("mu0")), f("beta0"))
            run = dict(xKFinal=o[0], Ahat=o[2], Qhat=o[3], Chat=o[4], Rhat=o[5], alphahat=o[6],
                       muhat=o[7], betahat=o[8], IC=o[12], SE=o[13])
    run["warnings"] = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    return run


@pytest.fixture(scope="module")
def python_runs(gold) -> dict:
    return {case: _run_driver(gold, case, 1) for case in CASES}


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES)
def test_monte_carlo_estimates_within_the_measured_spread(gold, python_runs, case) -> None:
    f = _get(gold, case)
    py = python_runs[case]
    IC = f("IC")
    # The parameter count (deterministic given the constraints) and IC on the
    # original scale.
    assert _n_terms(IC) == int(round((py["IC"]["AIC"] + 2 * py["IC"]["llobs"]) / 2))
    for key, atol in _MC_ATOL[case].items():
        if key == "IC.llcomp":
            got, ref = py["IC"]["llcomp"], IC.llcomp
        elif key.startswith("SE."):
            got, ref = py["SE"][key[3:]], getattr(f("SE"), key[3:])
        else:
            got, ref = py[key], f(key)
        got = np.asarray(got, dtype=float)
        ref = np.asarray(ref, dtype=float).reshape(got.shape)
        if key == "SE.A" and case in _MC_UNCHECKABLE:
            got, ref = np.diag(got), np.diag(ref)
        if case == "pp_sep" and key == "gammahat":
            # The separated coefficients walk to the exp() underflow on both
            # sides (11 iterations each at seed 1; MATLAB keeps the previous
            # value on the 0/0 there), to -741.5 .. -743.5; Python's agree with
            # MATLAB's to 2.4e-3 (the spread of the runs that stop at 10 or 11
            # is 1.7e-3).  This port's former +-30
            # clip walked them past it (-892), and its reciprocal-pivot Newton
            # solve stopped EM at iteration 8 (-694.5).
            sep = ref < -100
            assert sep.any() and np.array_equal(got < -100, sep)
            assert np.all(ref[sep] > -745.2) and np.all(ref[sep] < -741)
            np.testing.assert_allclose(got[sep], ref[sep], rtol=0, atol=1e-2, err_msg="separated gamma")
            # Their information is then exactly 0: the SE pass (where MATLAB
            # never returns) reports SE = p = NaN for them, and says so.
            se, pv = np.asarray(py["SE"]["gamma"], dtype=float), np.asarray(py["Pvals"]["gamma"], dtype=float)
            assert np.all(np.isnan(se[sep])) and np.all(np.isnan(pv[sep]))
            assert np.all(np.isfinite(se[~sep])) and np.all(np.isfinite(pv[~sep]))
            assert len(py["warnings"]) == 1 and "Not identifiable" in py["warnings"][0]
            got, ref = got[~sep], ref[~sep]
        np.testing.assert_allclose(got, ref, rtol=0, atol=atol, err_msg=key)
    if case in _MC_UNCHECKABLE:
        # Not determined at mcIter = 100 (see _MC_UNCHECKABLE): only sane.
        for got in (np.asarray(py["SE"]["beta"], dtype=float),
                    np.asarray(py["SE"]["A"], dtype=float)[~np.eye(2, dtype=bool)]):
            assert np.all(np.isfinite(got)) and np.all(got > 0)
    if case != "pp_sep":
        assert not [w for w in py["warnings"] if "singular" in w]
    if f("family") == "PP":
        # The stopping iteration is Monte Carlo dependent (pp_pois stops after
        # 8 instead of 6 iterations for Python seed 2; pp_sep after 6..11), but
        # seed 1 is fixed, so the draws -- and this count -- are reproducible; a
        # failure here on another platform means a likelihood change landed
        # within round-off of a stopping threshold, not a parity regression.
        nIter = f("nIter") if f"{case}_nIter" in gold else np.size(f("ll_trace"))
        assert py["nIter"] == nIter


def _bias_entries(src, case, matlab: bool) -> np.ndarray:
    """The key estimates and checkable SEs of one run, flattened in a fixed order."""
    get = (lambda key: src(key)) if matlab else (lambda key: src[key])
    se = (lambda key: getattr(src("SE"), key)) if matlab else (lambda key: src["SE"][key])
    ll = src("IC").llcomp if matlab else src["IC"]["llcomp"]
    out = [np.ravel(get("muhat")), np.ravel(get("betahat")), np.ravel(get("Ahat")), np.diag(np.atleast_2d(get("Qhat"))),
           np.atleast_1d(ll), np.ravel(se("mu")), np.diag(np.atleast_2d(se("Q")))]
    if case == "lfp_def":
        out += [np.ravel(get("Chat")), np.diag(np.atleast_2d(get("Rhat"))), np.ravel(get("alphahat"))]
        out += [np.ravel(se(k)) for k in ("A", "C", "alpha", "beta")] + [np.diag(np.atleast_2d(se("R")))]
    else:
        out += [np.ravel(get("gammahat")), np.ravel(se("gamma")), np.diag(np.atleast_2d(se("A")))]
    return np.concatenate([np.asarray(v, dtype=float).ravel() for v in out])


@pytest.fixture(scope="module")
def held_out_runs(gold) -> dict:
    return {case: [_run_driver(gold, case, seed) for seed in _BIAS_SEEDS] for case in _BIAS_CASES}


@pytest.mark.slow
@pytest.mark.parametrize("case", _BIAS_CASES)
def test_monte_carlo_estimates_show_no_bias(gold, held_out_runs, case) -> None:
    # MATLAB's value against the distribution of Python runs over seeds that
    # were not used to set _MC_ATOL (see the module docstring).
    f = _get(gold, case)
    ref = _bias_entries(f, case, True)
    P = np.array([_bias_entries(r, case, False) for r in held_out_runs[case]])
    n = P.shape[0]
    sd = P.std(axis=0, ddof=1)
    assert np.all(sd > 0)
    z = (ref - P.mean(axis=0)) / (sd * np.sqrt(1.0 + 1.0 / n))
    assert np.max(np.abs(z)) <= 6.0, np.round(z, 2)
    assert np.sqrt(np.mean(z ** 2)) <= 2.0, np.round(z, 2)
    assert abs(np.mean(z)) <= 1.0, np.round(z, 2)
