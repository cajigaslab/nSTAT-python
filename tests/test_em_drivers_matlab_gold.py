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
  that coefficient by about -1 per step until exp() underflows (MATLAB keeps
  the previous value on the 0/0 there), so it ends near -743; MATLAB returns
  without SEs only (10 outputs): with SEs requested its observed information is
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
* **Monte Carlo dependent** (one Python seed, ``seeded_global_rng(1)``): every
  estimate, ``xKFinal``, ``IC.llcomp`` and the SEs, each within twice the
  largest deviation from the gold measured over Python seeds 1..8 (rounded up
  to one significant digit; table in ``_MC_ATOL``).  For calibration, MATLAB's
  own spread over ``rng(1..8)`` on pp_pois / pp_sep is of the same size
  (max |beta - gold| 0.078 / 0.185, Python 0.070 / 0.175).  The iteration at
  which EM stops (stop on the first likelihood decrease or a change below
  1e-3) is itself Monte Carlo dependent: on pp_sep MATLAB stopped after 6..11
  iterations over ``rng(1..8)`` and Python after 6..8, so the separated
  coefficient ends at one of -397.5, -496.5, -595.5, -694.5 or the underflow
  limit near -743 on both sides; the test checks the walk, not the stop.
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
# (measured on macOS arm64 / Accelerate against MATLAB R2025b).
_MC_ATOL = {
    "pp_pois": {"muhat": 5e-02, "betahat": 2e-01, "gammahat": 9e-02, "Ahat": 4e-03, "Qhat": 2e-05, "xKFinal": 3e-01,
                "IC.llcomp": 2e+00, "SE.A": 3e-03, "SE.Q": 1e-05, "SE.mu": 5e-03, "SE.beta": 2e+00, "SE.gamma": 4e-02},
    "pp_binom": {"muhat": 4e-02, "betahat": 2e-01, "gammahat": 2e-01, "Ahat": 2e-03, "Qhat": 3e-05, "xKFinal": 1e-01,
                 "IC.llcomp": 5e-01, "SE.A": 2e-03, "SE.Q": 3e-06, "SE.mu": 2e-03, "SE.beta": 8e-01,
                 "SE.gamma": 4e-02},
    "pp_defwin": {"muhat": 2e-02, "betahat": 8e-02, "gammahat": 5e-02, "Ahat": 2e-03, "Qhat": 2e-05,
                  "xKFinal": 6e-02, "IC.llcomp": 2e+00, "SE.A": 9e-04, "SE.Q": 4e-06, "SE.mu": 2e-03,
                  "SE.beta": 5e-01, "SE.gamma": 4e-02},
    "lfp_def": {"muhat": 2e-02, "betahat": 3e-02, "Ahat": 5e-06, "Qhat": 7e-07, "Chat": 9e-05, "Rhat": 3e-06,
                "alphahat": 2e-05, "xKFinal": 9e-04, "IC.llcomp": 7e-02, "SE.A": 2e-04, "SE.Q": 2e-05,
                "SE.C": 4e-03, "SE.R": 2e-03, "SE.alpha": 2e-03, "SE.mu": 3e-03, "SE.beta": 8e-03},
    "pp_sep": {"muhat": 8e-02, "betahat": 4e-01, "gammahat": 3e-01, "Ahat": 3e-02, "Qhat": 2e-04, "xKFinal": 2e+00,
               "IC.llcomp": 7e-01},
}


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


@pytest.fixture(scope="module")
def python_runs(gold) -> dict:
    from nstat.extras.matlab_rng import seeded_global_rng

    runs = {}
    for case in CASES:
        f = _get(gold, case)
        dN = np.atleast_2d(f("dN")).astype(float)
        with seeded_global_rng(1):
            if f("family") == "PP":
                wt = np.asarray(f("windowTimes"), dtype=float)
                g0 = np.asarray(f("gamma0"), dtype=float)
                if wt.size == 0 and g0.ndim == 1:
                    g0 = g0.reshape(-1, 1)
                o = DecodingAlgorithms.PP_EM(
                    dN, f("A0"), f("Q0"), np.ravel(f("mu0")), f("beta0"), str(f("fitType")), float(f("delta")), g0,
                    None if wt.size == 0 else wt, None, None,
                    DecodingAlgorithms.PP_EMCreateConstraints(*[int(v) for v in np.ravel(f("cons"))]))
                runs[case] = dict(xKFinal=o[0], Ahat=o[2], Qhat=o[3], muhat=o[4], betahat=o[5], gammahat=o[6],
                                  IC=o[9], SE=o[10], nIter=o[12])
            else:
                o = PPLFP.PPLFP_EM(np.atleast_2d(f("y")), dN, f("A0"), f("Q0"), f("C0"), f("R0"),
                                   np.ravel(f("alpha0")), np.ravel(f("mu0")), f("beta0"))
                runs[case] = dict(xKFinal=o[0], Ahat=o[2], Qhat=o[3], Chat=o[4], Rhat=o[5], alphahat=o[6],
                                  muhat=o[7], betahat=o[8], IC=o[12], SE=o[13])
    return runs


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
        if case == "pp_sep" and key == "gammahat":
            # The separated coefficients walk to large negative values on both
            # sides and stop at most at the exp() underflow (MATLAB keeps the
            # previous value on 0/0); this port's former +-30 clip let them
            # walk past it (-892 here).
            sep = ref < -100
            assert sep.any() and np.array_equal(got < -100, sep)
            assert np.all(got[sep] > -745.2) and np.all(got[sep] < -300) and np.all(ref[sep] > -745.2)
            got, ref = got[~sep], ref[~sep]
        np.testing.assert_allclose(got, ref, rtol=0, atol=atol, err_msg=key)
    if f("family") == "PP" and case != "pp_sep":
        assert py["nIter"] == f("nIter")
