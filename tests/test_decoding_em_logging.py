"""EM progress output goes through ``logging``, not ``print`` (review C3 / B4).

``KF_EM`` / ``KF_EStep`` / ``PP_EM`` used to print iteration banners to
stdout unconditionally (MATLAB's DecodingAlgorithms.m prints nothing there).
They now log at INFO on ``nstat.decoding_algorithms``: silent by default,
visible when the caller configures logging.  (``mPPCO_EM`` is now a
deprecated alias of ``PPLFP_EM``, whose progress output lives in
``nstat/decoding/PPLFP.py``.)
"""
from __future__ import annotations

import logging

import numpy as np

from nstat.DecodingAlgorithms import DecodingAlgorithms


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
