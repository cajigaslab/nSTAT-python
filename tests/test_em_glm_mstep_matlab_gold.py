"""GLM M-step of ``PP_MStep`` / ``PPLFP_MStep`` against the MATLAB gold fixture.

Gold: ``tests/parity/fixtures/matlab_gold/em_glm_mstep.mat``, captured by
``tools/parity/matlab/capture_em_glm_mstep.m`` from the repaired MATLAB
(``fix/pp-em`` @ ``8dbd0e4``, pending upstream merge): one M-step call with
``MstepMethod = 'GLM'`` per case, on the output of one E-step at the generating
parameters (rng(42) synthetic data, dx = 2, N = 1500, previous gamma -0.2):

* ``pp_pois`` / ``pp_binom`` -- PP_MStep, poisson / binomial, C = 3, windows
  [0 2 5 10] ms;
* ``lfp_pois`` / ``lfp_binom`` -- PPLFP_MStep, the same (dy = 2);
* ``pp_unest`` / ``lfp_unest`` -- poisson, C = 4, a hard 1-bin refractory
  period and windows [0 1 5 20] ms: the (0, 1] ms window is separated for every
  cell (FitResSummary drops it, se >= 100), so its gamma row keeps the previous
  -0.2 (MATLAB R4a);
* ``pp_c1`` / ``lfp_c1`` -- a single cell (poisson / binomial; MATLAB's
  getCoeffs returns a 1 x nLabels row there, F3);
* ``pp_2ms`` -- PP_MStep at delta = 2 ms, windows [0 4 10 20] ms (the delta
  time base, MATLAB C6 / R4c).

Every output is compared: A, Q (C, R, alpha), x0, Px0 (closed form) and mu,
beta, gamma (the GLM fit, mapped by label).  ``W_K`` is not in the fixture:
the GLM branch never reads it, which the test checks by passing an all-NaN
``W_K``.

Tolerance, from the measured agreement (macOS arm64 / Accelerate vs MATLAB
R2025b):

* closed-form outputs and every poisson GLM coefficient: <= 5.0e-14 absolute
  (MATLAB ``glmfit`` and the Python IRLS converge to the same MLE), so
  ``rtol = 1e-10``, ``atol = 1e-12``;
* binomial GLM coefficients: <= 1.8e-4 absolute (pp_binom 8.4e-5, lfp_binom
  1.8e-4, lfp_c1 9.8e-6).  MATLAB's ``'BNLRCG'`` is Demba Ba's truncated
  conjugate-gradient logistic regression (``bnlrCG``), which stops short of the
  MLE that the Python ``fit_binomial_glm`` reaches; ``atol = 1e-3`` (5x the
  worst case; the coefficients are O(1)).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from nstat.decoding.PPLFP import PPLFP
from nstat.decoding_algorithms import DecodingAlgorithms

FIXTURE = Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "em_glm_mstep.mat"
CASES = ["pp_pois", "pp_binom", "lfp_pois", "lfp_binom", "pp_unest", "lfp_unest", "pp_c1", "lfp_c1", "pp_2ms"]
RTOL, ATOL = 1e-10, 1e-12
BINOMIAL_GLM_ATOL = 1e-3
GLM_KEYS = ("muhat_new", "betahat_new", "gammahat_new")


@pytest.fixture(scope="module")
def gold() -> dict:
    return loadmat(FIXTURE)


def _f(g, case, key):
    return np.asarray(g[f"{case}_{key}"], dtype=float)


def _s(g, case, key):
    return str(np.asarray(g[f"{case}_{key}"]).reshape(-1)[0])


def run_case(g, case, W_K=None):
    """Run the case's GLM M-step; returns {output name: value}."""
    family, fit = _s(g, case, "family"), _s(g, case, "fitType")
    dN = _f(g, case, "dN")
    C, N = dN.shape
    H = _f(g, case, "HkAll")
    if H.ndim == 2:  # MATLAB stores a one-cell N x W x 1 history as N x W
        H = H.reshape(N, -1, 1)
    ES = {k[len(case) + 4:]: np.asarray(g[k], dtype=float) for k in g if k.startswith(f"{case}_ES_")}
    x_K = _f(g, case, "x_K")
    W_K = np.full((x_K.shape[0], x_K.shape[0], N), np.nan) if W_K is None else W_K
    args = dict(mu=_f(g, case, "mu").reshape(-1), beta=_f(g, case, "beta"), gamma=_f(g, case, "gamma"),
                wt=_f(g, case, "windowTimes").reshape(-1), x0=_f(g, case, "x0").reshape(-1), Px0=_f(g, case, "Px0"),
                delta=float(_f(g, case, "delta").reshape(-1)[0]))
    if family == "PP":
        out = DecodingAlgorithms.PP_MStep(dN, x_K, W_K, args["x0"], args["Px0"], ES, fit, args["mu"], args["beta"],
                                          args["gamma"], args["wt"], H, DecodingAlgorithms.PP_EMCreateConstraints(),
                                          "GLM", args["delta"])
        keys = ["Ahat", "Qhat", "muhat_new", "betahat_new", "gammahat_new", "x0hat", "Px0hat"]
    else:
        out = PPLFP.PPLFP_MStep(dN, _f(g, case, "y"), x_K, W_K, args["x0"], args["Px0"], ES, fit, args["mu"],
                                args["beta"], args["gamma"], args["wt"], H, PPLFP.PPLFP_EMCreateConstraints(), "GLM",
                                args["delta"])
        keys = ["Ahat", "Qhat", "Chat", "Rhat", "alphahat", "muhat_new", "betahat_new", "gammahat_new", "x0hat",
                "Px0hat"]
    return dict(zip(keys, out))


@pytest.mark.parametrize("case", CASES)
def test_glm_mstep_matches_matlab(gold, case) -> None:
    out = run_case(gold, case)
    binomial = _s(gold, case, "fitType") == "binomial"
    for key, value in out.items():
        expected = _f(gold, case, key)
        actual = np.asarray(value, dtype=float)
        assert actual.size == expected.size, key
        if binomial and key in GLM_KEYS:
            np.testing.assert_allclose(actual.reshape(-1), expected.reshape(-1), rtol=0, atol=BINOMIAL_GLM_ATOL,
                                       err_msg=f"{case}: {key}")
        else:
            np.testing.assert_allclose(actual.reshape(expected.shape), expected, rtol=RTOL, atol=ATOL,
                                       err_msg=f"{case}: {key}")


@pytest.mark.parametrize("case", ["pp_unest", "lfp_unest"])
def test_glm_mstep_unestimable_window_keeps_previous_gamma(gold, case) -> None:
    # The separated (0, 1] ms window keeps the previous gamma exactly, in the
    # gold and here; the other windows are fitted.
    gamma = np.asarray(run_case(gold, case)["gammahat_new"], dtype=float)
    previous = _f(gold, case, "gamma")
    np.testing.assert_array_equal(_f(gold, case, "gammahat_new")[0], previous[0])
    np.testing.assert_array_equal(gamma[0], previous[0])
    assert np.all(gamma[1:] != previous[1:])
