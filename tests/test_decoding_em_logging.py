"""EM progress output goes through ``logging``, not ``print`` (review C3 / B4).

``KF_EM`` / ``KF_EStep`` / ``PP_EM`` used to print iteration banners to
stdout unconditionally (MATLAB's DecodingAlgorithms.m prints nothing there).
They now log at INFO on ``nstat.decoding_algorithms``: silent by default,
visible when the caller configures logging.  ``PPLFP_EM`` / ``PPLFP_MStep``
(and so the deprecated alias ``mPPCO_EM``) do the same on
``nstat.decoding.PPLFP``.
"""
from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.DecodingAlgorithms import DecodingAlgorithms
from nstat.decoding.PPLFP import PPLFP
from nstat.extras.matlab_rng import seeded_global_rng

_PPLFP_EM_FIXTURE = Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "pplfp_EM.mat"


def _run_kf_em():
    rng = np.random.default_rng(5)
    T = 100
    y = np.cumsum(rng.standard_normal((1, T)) * 0.3, axis=1) + rng.standard_normal((1, T)) * 0.5
    return DecodingAlgorithms.KF_EM(
        y,
        np.eye(1) * 0.95,
        np.eye(1) * 0.1,
        np.eye(1),
        np.eye(1) * 0.5,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.eye(1),
        DecodingAlgorithms.KF_EMCreateConstraints(),
    )


def test_kf_em_is_silent_by_default_and_logs_progress_at_info(capsys, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nstat.decoding_algorithms"):
        _run_kf_em()
    captured = capsys.readouterr()
    assert captured.out == ""
    records = [r for r in caplog.records if r.name == "nstat.decoding_algorithms"]
    assert records, "expected EM progress records on the module logger"
    assert {r.levelno for r in records} == {logging.INFO}
    messages = [r.getMessage() for r in records]
    assert "Kalman Filter/Gaussian Observation EM Algorithm" in messages[0]
    assert "Iteration #1" in messages
    assert any(m.startswith("logll: ") for m in messages)
    assert any(m.startswith("Max Parameter Change: ") for m in messages)


def _run_pplfp_em():
    """PPLFP_EM on the pplfp_EM.mat gold inputs, as the numerical-drift recipe runs it."""
    fx = loadmat(_PPLFP_EM_FIXTURE, squeeze_me=True, struct_as_record=False)
    f = lambda key: np.asarray(fx[key], dtype=float)  # noqa: E731
    constraints = PPLFP.PPLFP_EMCreateConstraints(
        EstimateA=1, AhatDiag=0, QhatDiag=1, RhatDiag=1, Estimatex0=0, EstimatePx0=0,
        mcIter=int(fx["mcIter"]),
    )
    with seeded_global_rng(42):
        out = PPLFP.PPLFP_EM(
            f("y"), f("dN"), f("Ahat0"), f("Qhat0"), f("Chat0"), f("Rhat0"),
            f("alphahat0").reshape(-1), f("mu").reshape(-1), f("beta"),
            fitType=str(fx["fitType"]), delta=float(fx["delta"]), x0=f("x0").reshape(-1),
            Px0=f("Px0"), PPLFP_EM_Constraints=constraints, MstepMethod="NewtonRaphson",
        )
    return out


def _assert_identical(a, b, path: str = "out") -> None:
    """Bit-for-bit equality over nested tuples / dicts / arrays / scalars."""
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


def test_pplfp_em_is_silent_by_default_and_logs_progress_at_info(capsys, caplog, monkeypatch) -> None:
    with caplog.at_level(logging.INFO, logger="nstat.decoding.PPLFP"):
        out = _run_pplfp_em()
    assert capsys.readouterr().out == ""
    records = [r for r in caplog.records if r.name == "nstat.decoding.PPLFP"]
    assert records, "expected EM progress records on the module logger"
    assert {r.levelno for r in records} == {logging.INFO}
    messages = [r.getMessage() for r in records]
    assert "Joint Point-Process/Gaussian Observation EM Algorithm" in messages[0]
    assert "Iteration #1" in messages
    assert "****M-step for beta****" in messages  # PPLFP_MStep (NewtonRaphson)
    assert "neuron:1 iter: 1,2,3,4,5" in messages  # one record per neuron
    assert any(m.startswith("Max Parameter Change: ") for m in messages)
    # Logging cannot change the numerics: the same seeded run with the logger
    # disabled returns every output bit for bit.  (MATLAB parity of PPLFP_EM
    # is the PPLFP_EM numerical-drift entry.)
    monkeypatch.setattr(logging.getLogger("nstat.decoding.PPLFP"), "disabled", True)
    caplog.clear()
    silent = _run_pplfp_em()
    assert not [r for r in caplog.records if r.name == "nstat.decoding.PPLFP"]
    _assert_identical(out, silent)


def test_pplfp_module_has_no_print_calls() -> None:
    tree = ast.parse(Path(sys.modules["nstat.decoding.PPLFP"].__file__).read_text())
    calls = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]
    assert calls == []
