"""GLM M-step plug-in warning (track-P1 item 4; user decision; mirrors nSTAT PR #138).

MATLAB's ``PP_EM`` / ``PPLFP_EM`` / ``PP_MStep`` / ``PPLFP_MStep`` warn once
with id ``nSTAT:EM:glmPlugIn`` when ``MstepMethod='GLM'``: it is a plug-in fit
on the smoothed means (ignores ``W_K``), which inflates beta and can drift;
``'NewtonRaphson'`` (the default) is preferred.  ``PP_EM`` / ``PPLFP_EM`` warn
once before their loop and suppress the warning for the loop's duration (so
their internal M-step calls do not re-warn every iteration); a direct
``PP_MStep`` / ``PPLFP_MStep`` call still warns exactly once.  No numerical
change: this is a warning only.  Mirrors MATLAB's
``tests/unit/testGLMPlugInWarning.m``.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from nstat.decoding.PPLFP import PPLFP
from nstat.decoding_algorithms import DecodingAlgorithms, GLMPlugInWarning
import nstat.decoding_algorithms as da


def _pp_em_gold_args():
    from test_em_routines_correctness import _pp_estep_gold_case

    A, Q, dN, mu, beta, fit, gamma, HkAll, x0, Px0 = _pp_estep_gold_case("c2")
    cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0, 0, 30)
    return [dN, A, Q, mu.reshape(-1), beta, fit, 0.001, gamma, [0.0, 0.001, 0.003],
            x0.reshape(-1), Px0, cons, "NewtonRaphson"]


def _pplfp_em_gold_args():
    from pathlib import Path

    from scipy.io import loadmat

    fx = loadmat(Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "pplfp_EM.mat",
                 squeeze_me=True, struct_as_record=False)
    f = lambda k: np.asarray(fx[k], dtype=float)  # noqa: E731
    v = lambda k: np.asarray(fx[k], dtype=float).reshape(-1)  # noqa: E731
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, int(fx["mcIter"]), 0)
    return [f("y"), f("dN"), f("Ahat0"), f("Qhat0"), f("Chat0"), f("Rhat0"), v("alphahat0"), v("mu"), f("beta"),
            "poisson", 0.001, None, None, v("x0"), f("Px0"), cons, "NewtonRaphson"]


def _glm_warnings(record):
    return [w for w in record if issubclass(w.category, GLMPlugInWarning)]


@pytest.fixture
def few_iterations(monkeypatch):
    # The GLM M-step runs a real GLM fit per iteration; cap PP_EM/PPLFP_EM's
    # loop at 2 iterations so the warning-counting tests stay fast. The
    # warning fires before the loop starts, so this does not affect the count.
    #
    # Review fix: nstat.decoding.PPLFP does `from nstat.decoding_algorithms
    # import (..., _EM_MAX_ITER, ...)`, which binds its OWN
    # nstat.decoding.PPLFP._EM_MAX_ITER name at import time; patching
    # da._EM_MAX_ITER alone leaves that binding untouched, so PPLFP_EM's
    # `maxIter = _EM_MAX_ITER` read the real 100, not this fixture's 2 (the
    # PPLFP_EM tests happened to still be fast only because the gold-fixture
    # EM-converged inputs stop on their own well under 100 iterations, not
    # because this fixture capped them). Patch both module bindings.
    monkeypatch.setattr(da, "_EM_MAX_ITER", 2)
    monkeypatch.setattr(sys.modules[PPLFP.__module__], "_EM_MAX_ITER", 2)


def test_pp_em_glm_warns_exactly_once(few_iterations) -> None:
    args = _pp_em_gold_args()
    args[-1] = "GLM"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        DecodingAlgorithms.PP_EM(*args)
    assert len(_glm_warnings(rec)) == 1
    assert "PP_EM" in str(_glm_warnings(rec)[0].message)


def test_pp_em_newton_raphson_default_does_not_warn(few_iterations) -> None:
    args = _pp_em_gold_args()  # MstepMethod="NewtonRaphson"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        DecodingAlgorithms.PP_EM(*args)
    assert len(_glm_warnings(rec)) == 0


def test_pplfp_em_glm_warns_exactly_once(few_iterations) -> None:
    args = _pplfp_em_gold_args()
    args[-1] = "GLM"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        PPLFP.PPLFP_EM(*args)
    assert len(_glm_warnings(rec)) == 1
    assert "PPLFP_EM" in str(_glm_warnings(rec)[0].message)


def test_pplfp_em_newton_raphson_default_does_not_warn(few_iterations) -> None:
    args = _pplfp_em_gold_args()  # MstepMethod="NewtonRaphson"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        PPLFP.PPLFP_EM(*args)
    assert len(_glm_warnings(rec)) == 0


def _direct_mstep_problem(K=60, C=1):
    rng = np.random.default_rng(8)
    dx = 2
    x = 0.1 * rng.standard_normal((dx, K))
    W_K = np.tile((0.01 * np.eye(dx))[:, :, None], (1, 1, K))
    dN = (rng.random((C, K)) < 0.05).astype(float)
    mu = np.full(C, -3.0)
    beta = 0.1 * rng.standard_normal((dx, C))
    ES = dict(Sxkm1xkm1=K * 0.01 * np.eye(dx), Sxkxkm1=0.9 * K * 0.01 * np.eye(dx),
              Sxkm1xk=0.9 * K * 0.01 * np.eye(dx), Sxkxk=K * 0.01 * np.eye(dx),
              sumXkTerms=0.2 * K * 0.01 * np.eye(dx),
              Sxkyk=np.zeros((dx, 1)), sumYkTerms=K * np.eye(1), Sykyk=K * np.eye(1))
    x0, Px0 = np.zeros(dx), 0.01 * np.eye(dx)
    return dict(dN=dN, x=x, W_K=W_K, mu=mu, beta=beta, x0=x0, Px0=Px0, ES=ES, dx=dx, K=K, C=C)


def test_pp_mstep_direct_call_with_glm_warns_exactly_once() -> None:
    P = _direct_mstep_problem()
    cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        DecodingAlgorithms.PP_MStep(
            P["dN"], P["x"], P["W_K"], P["x0"], P["Px0"], P["ES"], "poisson",
            P["mu"], P["beta"], np.array(0.0), [], np.zeros((P["K"], 0, P["C"])), cons, "GLM")
    assert len(_glm_warnings(rec)) == 1
    assert "PP_MStep" in str(_glm_warnings(rec)[0].message)


def test_pp_mstep_direct_call_with_newton_raphson_does_not_warn() -> None:
    P = _direct_mstep_problem()
    cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        DecodingAlgorithms.PP_MStep(
            P["dN"], P["x"], P["W_K"], P["x0"], P["Px0"], P["ES"], "poisson",
            P["mu"], P["beta"], np.array(0.0), [], np.zeros((P["K"], 0, P["C"])), cons, "NewtonRaphson")
    assert len(_glm_warnings(rec)) == 0


def test_pplfp_mstep_direct_call_with_glm_warns_exactly_once() -> None:
    P = _direct_mstep_problem()
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 10, 0)
    y = np.ones((1, P["K"]))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        PPLFP.PPLFP_MStep(
            P["dN"], y, P["x"], P["W_K"], P["x0"], P["Px0"], P["ES"], "poisson",
            P["mu"], P["beta"], np.array(0.0), [], np.zeros((P["K"], 0, P["C"])), cons, "GLM")
    assert len(_glm_warnings(rec)) == 1
    assert "PPLFP_MStep" in str(_glm_warnings(rec)[0].message)


def test_pplfp_mstep_direct_call_with_newton_raphson_does_not_warn() -> None:
    # A single all-zero history window (rather than PP_MStep's no-history
    # 0-window case, which trips an unrelated, pre-existing PPLFP_MStep
    # Newton-Raphson shape quirk with a scalar zero gamma -- out of scope
    # for the track-P1 GLM-warning item): nW=1, gammahat all zero.
    P = _direct_mstep_problem()
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 10, 0)
    y = np.ones((1, P["K"]))
    gamma = np.zeros((1, P["C"]))
    HkAll = np.zeros((P["K"], 1, P["C"]))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        PPLFP.PPLFP_MStep(
            P["dN"], y, P["x"], P["W_K"], P["x0"], P["Px0"], P["ES"], "poisson",
            P["mu"], P["beta"], gamma, [0.0, 0.001], HkAll, cons, "NewtonRaphson")
    assert len(_glm_warnings(rec)) == 0
