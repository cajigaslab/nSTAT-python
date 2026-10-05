"""``DecodingAlgorithms.mPPCO_*`` are deprecated aliases of ``PPLFP_*``.

MATLAB's ``DecodingAlgorithms.m`` defines ``mPPCO_fixedIntervalSmoother``,
``mPPCODecodeLinear``, ``mPPCODecode_predict``, ``mPPCO_EMCreateConstraints``,
``mPPCO_ComputeParamStandardErrors``, ``mPPCO_EM``, ``mPPCO_EStep`` and
``mPPCO_MStep`` as deprecation shims: each warns ``nSTAT:deprecated:mPPCO`` and
forwards ``varargin{:}`` to the matching ``DecodingAlgorithms.PPLFP_*``.  The
Python mirrors do the same (a ``DeprecationWarning`` with MATLAB's text,
positional forwarding), replacing stale standalone implementations (the EM
ones raised ``NameError``).  ``mPPCODecode_update`` is not a forwarder: its
Python body takes MATLAB's permuted ``(nW, C, N)`` history, which the Python
``PPLFP_Decode_update`` does not accept, so forwarding it would change its
contract (left as is pending a maintainer decision).

Contract pinned here, per alias:

* the public signature is unchanged from main @ 98d468d (frozen API);
* it emits a ``DeprecationWarning`` whose text is MATLAB's message;
* it returns exactly (``np.array_equal``, bit-for-bit) what the ``PPLFP_*``
  target returns on the same inputs -- the inputs of the MATLAB-gold-validated
  ``pplfp_*.mat`` fixtures.  Monte-Carlo paths draw through
  ``np.random.default_rng()``; each call runs in its own fresh
  ``seeded_global_rng(42)`` block so both see the same stream.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import nstat.decoding_algorithms as da
from nstat.decoding.PPLFP import PPLFP
from nstat.decoding_algorithms import DecodingAlgorithms
from nstat.extras.matlab_rng import seeded_global_rng

FIXTURE_ROOT = Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold"

# (alias, PPLFP target, signature string on main @ 98d468d -- except the three
# defaults the repaired MATLAB changed (B8): mPPCO_EMCreateConstraints'
# Estimatex0 / EstimatePx0 (1 -> 0) and mPPCO_EM / mPPCO_MStep's MstepMethod
# ('GLM' -> 'NewtonRaphson'), and mPPCO_MStep's trailing delta=0.001 (R4c:
# PPLFP_MStep's new optional 16th input).  MATLAB's aliases forward varargin,
# so they inherit PPLFP_*'s inputs and defaults; the Python aliases spell
# them out and follow.)
ALIASES = [
    (
        "mPPCO_fixedIntervalSmoother",
        "PPLFP_fixedIntervalSmoother",
        "(A, Q, C, R, y, alpha, dN, lags, mu, beta, fitType, delta=0.001, gamma=None, "
        "windowTimes=None, x0=None, Px0=None, HkAll=None)",
    ),
    (
        "mPPCODecodeLinear",
        "PPLFP_DecodeLinear",
        "(A, Q, C, R, y, alpha, dN, mu, beta, fitType='poisson', delta=0.001, gamma=None, "
        "windowTimes=None, x0=None, Px0=None, HkAll=None)",
    ),
    (
        "mPPCODecode_predict",
        "PPLFP_Decode_predict",
        "(x_u, W_u, A, Q)",
    ),
    (
        "mPPCO_EMCreateConstraints",
        "PPLFP_EMCreateConstraints",
        "(EstimateA=1, AhatDiag=0, QhatDiag=1, QhatIsotropic=0, RhatDiag=1, "
        "RhatIsotropic=0, Estimatex0=0, EstimatePx0=0, Px0Isotropic=0, mcIter=1000, "
        "EnableIkeda=0)",
    ),
    (
        "mPPCO_ComputeParamStandardErrors",
        "PPLFP_ComputeParamStandardErrors",
        "(y, dN, xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat, "
        "ExpectationSumsFinal, fitType, muhat, betahat, gammahat, windowTimes, HkAll, "
        "mPPCOEM_Constraints=None)",
    ),
    (
        "mPPCO_EM",
        "PPLFP_EM",
        "(y, dN, Ahat0, Qhat0, Chat0, Rhat0, alphahat0, mu, beta, fitType='poisson', "
        "delta=0.001, gamma=None, windowTimes=None, x0=None, Px0=None, "
        "mPPCOEM_Constraints=None, MstepMethod='NewtonRaphson')",
    ),
    (
        "mPPCO_EStep",
        "PPLFP_EStep",
        "(A, Q, C, R, y, alpha, dN, mu, beta, fitType='poisson', delta=0.001, "
        "gamma=None, HkAll=None, x0=None, Px0=None)",
    ),
    (
        "mPPCO_MStep",
        "PPLFP_MStep",
        "(dN, y, x_K, W_K, x0, Px0, ExpectationSums, fitType='poisson', muhat=None, "
        "betahat=None, gammahat=None, windowTimes=None, HkAll=None, "
        "mPPCOEM_Constraints=None, MstepMethod='NewtonRaphson', delta=0.001)",
    ),
]
_ALIAS_IDS = [a[0] for a in ALIASES]


def _message(alias: str, target: str) -> str:
    return (
        f"DecodingAlgorithms.{alias} is deprecated; use DecodingAlgorithms.{target} "
        "instead. See §4.B.7 for the PPLFP derivation."
    )


def _assert_identical(a, b, path: str = "out") -> None:
    """Recursive bit-for-bit equality over tuples / dicts / arrays / scalars."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), path
        for key in a:
            _assert_identical(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_identical(x, y, f"{path}[{i}]")
    elif a is None or isinstance(a, str):
        assert a == b, path
    else:
        x, y = np.asarray(a), np.asarray(b)
        assert x.shape == y.shape and x.dtype == y.dtype, path
        assert np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"), path


# ---------------------------------------------------------------------------
# Gold-fixture inputs (same reconstruction as tools/parity/numerical_drift.py's
# pplfp_* recipes and tools/parity/matlab/export_pplfp_gold_fixtures.m)
# ---------------------------------------------------------------------------


def _load(name: str) -> dict:
    return loadmat(FIXTURE_ROOT / name, squeeze_me=True, struct_as_record=False)


def _f(fx: dict, key: str) -> np.ndarray:
    return np.asarray(fx[key], dtype=float)


def _v(fx: dict, key: str) -> np.ndarray:
    return np.asarray(fx[key], dtype=float).reshape(-1)


def _estep_args(fx: dict) -> tuple:
    """Positional PPLFP_EStep inputs; all-zero gamma -> scalar 0 + zero HkAll."""
    dN = _f(fx, "dN")
    num_cells, K = dN.shape
    gamma = _v(fx, "gamma")
    assert float(np.max(np.abs(gamma))) == 0.0  # true for every pplfp_*.mat
    return (
        _f(fx, "A"), _f(fx, "Q"), _f(fx, "C"), _f(fx, "R"), _f(fx, "y"), _v(fx, "alpha"),
        dN, _v(fx, "mu"), _f(fx, "beta"), str(fx["fitType"]), float(fx["delta"]),
        np.array(0.0), np.zeros((K, 1, num_cells)), _v(fx, "x0"), _f(fx, "Px0"),
    )


def _se_args(fx: dict) -> tuple:
    """Positional PPLFP_ComputeParamStandardErrors inputs from pplfp_SE.mat.

    ExpectationSumsFinal is rebuilt with PPLFP_EStep at the EM-converged
    parameters, as the MATLAB capture recipe does.
    """
    dN = _f(fx, "dN")
    num_cells, K = dN.shape
    HkAll = np.zeros((K, 1, num_cells))
    y, fit, delta = _f(fx, "y"), str(fx["fitType"]), float(fx["delta"])
    Ahat, Qhat, Chat, Rhat = _f(fx, "Ahat"), _f(fx, "Qhat"), _f(fx, "Chat"), _f(fx, "Rhat")
    alphahat, x0hat, Px0hat = _v(fx, "alphahat"), _v(fx, "x0hat"), _f(fx, "Px0hat")
    muhat, betahat = _v(fx, "muhat_new"), _f(fx, "betahat_new")
    _, _, _, es = PPLFP.PPLFP_EStep(
        Ahat, Qhat, Chat, Rhat, y, alphahat, dN, muhat, betahat, fit, delta,
        np.array(0.0), HkAll, x0hat, Px0hat,
    )
    return (
        y, dN, _f(fx, "xKFinal"), _f(fx, "WKFinal"), Ahat, Qhat, Chat, Rhat, alphahat,
        x0hat, Px0hat, es, fit, muhat, betahat, np.array(0.0), None, HkAll,
    )


def _calls(alias: str):
    """Return (args, kwargs_alias, kwargs_target) for one alias on gold inputs."""
    if alias == "mPPCO_EMCreateConstraints":
        args = (1, 1, 1, 1, 0, 0, 0, 1, 1, 37, 0)
        return args, {}, {}
    if alias == "mPPCO_EStep":
        return _estep_args(_load("pplfp_EStep.mat")), {}, {}
    if alias == "mPPCO_fixedIntervalSmoother":
        A, Q, C, R, y, alpha, dN, mu, beta, fit, delta, gamma, HkAll, x0, Px0 = _estep_args(
            _load("pplfp_EStep.mat")
        )
        return (A, Q, C, R, y, alpha, dN, 2, mu, beta, fit, delta, gamma, None, x0, Px0, HkAll), {}, {}
    if alias == "mPPCODecodeLinear":
        A, Q, C, R, y, alpha, dN, mu, beta, fit, delta, gamma, HkAll, x0, Px0 = _estep_args(
            _load("pplfp_EStep.mat")
        )
        return (A, Q, C, R, y, alpha, dN, mu, beta, fit, delta, gamma, None, x0, Px0, HkAll), {}, {}
    if alias == "mPPCODecode_predict":
        fx = _load("pplfp_EStep.mat")
        return (_v(fx, "x0"), _f(fx, "Px0"), _f(fx, "A"), _f(fx, "Q")), {}, {}
    if alias == "mPPCO_MStep":
        fx = _load("pplfp_MStep.mat")
        e = _estep_args(fx)
        _, _, _, es = PPLFP.PPLFP_EStep(*e)
        args = (
            e[6], e[4], _f(fx, "x_K"), _f(fx, "W_K"), e[13], e[14], es, e[9], e[7], e[8],
            e[11], None, e[12],
        )
        # NewtonRaphson: the GLM branch fails on this 2-cell fixture
        # (matlab_defects.yml pplfp-mstep-fixture-missing), see the drift recipe.
        return args, {"MstepMethod": "NewtonRaphson"}, {"MstepMethod": "NewtonRaphson"}
    if alias == "mPPCO_EM":
        fx = _load("pplfp_EM.mat")
        constraints = PPLFP.PPLFP_EMCreateConstraints(
            EstimateA=1, AhatDiag=0, QhatDiag=1, RhatDiag=1, Estimatex0=0, EstimatePx0=0,
            mcIter=int(fx["mcIter"]),
        )
        args = (
            _f(fx, "y"), _f(fx, "dN"), _f(fx, "Ahat0"), _f(fx, "Qhat0"), _f(fx, "Chat0"),
            _f(fx, "Rhat0"), _v(fx, "alphahat0"), _v(fx, "mu"), _f(fx, "beta"),
            str(fx["fitType"]), float(fx["delta"]), None, None, _v(fx, "x0"), _f(fx, "Px0"),
        )
        return (
            args,
            {"mPPCOEM_Constraints": constraints, "MstepMethod": "NewtonRaphson"},
            {"PPLFP_EM_Constraints": constraints, "MstepMethod": "NewtonRaphson"},
        )
    if alias == "mPPCO_ComputeParamStandardErrors":
        constraints = PPLFP.PPLFP_EMCreateConstraints(mcIter=100)
        return (
            _se_args(_load("pplfp_SE.mat")),
            {"mPPCOEM_Constraints": constraints},
            {"PPLFP_EM_Constraints": constraints},
        )
    raise AssertionError(alias)


def _run(fn, args, kwargs):
    with seeded_global_rng(42):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("alias", "target", "signature"), ALIASES, ids=_ALIAS_IDS)
def test_alias_signature_unchanged(alias, target, signature) -> None:
    fn = getattr(DecodingAlgorithms, alias)
    assert str(inspect.signature(fn)) == signature
    # Module-level alias still bound to the class staticmethod.
    assert getattr(da, alias) is fn
    assert "DEPRECATED" in (fn.__doc__ or "") and target in fn.__doc__


@pytest.mark.parametrize(("alias", "target", "signature"), ALIASES, ids=_ALIAS_IDS)
def test_alias_warns_and_returns_exactly_the_pplfp_result(alias, target, signature) -> None:
    args, kw_alias, kw_target = _calls(alias)
    with pytest.warns(DeprecationWarning, match=re.escape(_message(alias, target))) as record:
        got = _run(getattr(DecodingAlgorithms, alias), args, kw_alias)
    # Exactly one deprecation warning, attributed to this test file (stacklevel).
    deps = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deps) == 1 and deps[0].filename == __file__
    expected = _run(getattr(PPLFP, target), args, kw_target)
    _assert_identical(got, expected)


def test_mppco_se_returns_finite_se_and_pvals() -> None:
    # Before the forwarding fix every call raised NameError (undefined
    # ``nearestSPD``) and mPPCO_EM returned empty SE / Pvals.
    args, kw_alias, _ = _calls("mPPCO_ComputeParamStandardErrors")
    with pytest.warns(DeprecationWarning):
        SE, Pvals, nTerms = _run(DecodingAlgorithms.mPPCO_ComputeParamStandardErrors, args, kw_alias)
    assert sorted(SE) == sorted(Pvals) and len(SE) > 0 and int(nTerms) > 0
    for key in SE:
        assert np.all(np.isfinite(np.asarray(SE[key], dtype=float))), key
        p = np.asarray(Pvals[key], dtype=float)
        assert np.all((p >= 0.0) & (p <= 1.0)), key


def test_mppco_estep_forwarder_no_longer_raises_nameerror() -> None:
    # Regression: the stale mPPCO_EStep referenced an undefined ``N``.
    args, _, _ = _calls("mPPCO_EStep")
    with pytest.warns(DeprecationWarning):
        x_K, W_K, logll, sums = DecodingAlgorithms.mPPCO_EStep(*args)
    assert x_K.shape == (2, 10) and W_K.shape == (2, 2, 10) and np.isfinite(logll)


@pytest.mark.parametrize("case", ["pdfl_pois_sq", "pdfl_pois_ctrl"])
def test_mppco_decode_linear_with_history_returns_exactly_the_pplfp_result(case) -> None:
    # Nonzero history, square (nW == C) and not: the alias is PPLFP_DecodeLinear
    # for every history layout, not just the zero-history gold inputs above.
    fx = loadmat(FIXTURE_ROOT / "pp_square_history.mat", squeeze_me=False)
    f = lambda key: np.asarray(fx[f"{case}_{key}"], dtype=float)  # noqa: E731
    fit = str(np.asarray(fx[f"{case}_fitType"]).reshape(-1)[0])
    dN = f("dN")
    rng = np.random.default_rng(3)
    C = np.array([[1.0, 0.3], [-0.2, 0.8]])
    R = np.diag([0.05, 0.08])
    y = 0.2 * rng.standard_normal((2, dN.shape[1]))
    args = (
        f("A"), f("Q"), C, R, y, np.array([0.1, -0.2]), dN, f("mu"), f("beta"), fit, 0.001,
        f("gamma"), None, f("x0"), f("Pi0"), f("HkAll"),
    )
    with pytest.warns(DeprecationWarning, match=re.escape(_message("mPPCODecodeLinear", "PPLFP_DecodeLinear"))):
        got = DecodingAlgorithms.mPPCODecodeLinear(*args)
    _assert_identical(got, PPLFP.PPLFP_DecodeLinear(*args))
