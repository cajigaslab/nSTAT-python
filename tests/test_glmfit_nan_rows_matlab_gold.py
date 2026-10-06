"""``Analysis.GLMFit``'s poisson ('GLM') path on a design with a NaN row,
against MATLAB gold.

Gold: ``tests/parity/fixtures/matlab_gold/glmfit_nan_rows.mat``, captured by
``tools/parity/matlab/capture_glmfit_nan_rows.m`` (rng(42); a 30 x 3 poisson
design with one NaN injected into row 7 (1-based), column 2).

MATLAB ``glmfit`` (toolbox/stats/stats/glmfit.m) calls
``statremovenan(y, x, ...)`` before fitting: the NaN row is dropped from the
regression (``b``, ``dev``, ``stats.se`` / ``covb``), not from the downstream
evaluation.  ``Analysis.GLMFit`` (Analysis.m:565-634) then evaluates
``data = exp(X*b)`` on the *original* (full) ``X``, so the NaN row still
gives a NaN ``data``/``lambda`` row; its ``eps``-floor uses MATLAB's
NaN-ignoring ``max`` (``max(NaN, eps) == eps``, unlike ``np.maximum``), so
that row's contribution to ``logLL`` collapses to exactly
``log(eps) * (y + (1-y)) == log(eps)``, not NaN.

Before the fix in this commit, ``_glmfit_independent_columns`` returned
``None`` for any non-finite ``X`` (NaN treated the same as Inf) and the
unmodified solver then ran on the NaN-containing ``X`` directly, returning
an all-NaN fit -- not MATLAB's NaN-row-dropped fit.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nstat.analysis import Analysis


@pytest.fixture(scope="module")
def gold() -> dict:
    from scipy.io import loadmat

    path = "tests/parity/fixtures/matlab_gold/glmfit_nan_rows.mat"
    return loadmat(path)


class _StubTrial:
    """Duck-typed stand-in exposing exactly what ``Analysis.GLMFit`` reads
    off ``tObj`` (``getDesignMatrix``, ``getSpikeVector``, ``getCov(0).time``,
    ``sampleRate``, ``nspikeColl.getNST(i).isSigRepBinary()``) -- avoids
    building a full ``Trial``/``nspikeTrain`` pipeline, which cannot carry a
    NaN design entry or non-binary per-bin counts > 1 (both present in the
    MATLAB-captured ``y``) through its spike-binning machinery.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sample_rate: float) -> None:
        self._X = X
        self._y = y
        self.sampleRate = sample_rate
        self.nspikeColl = SimpleNamespace(getNST=lambda i: SimpleNamespace(isSigRepBinary=lambda: False))

    def getDesignMatrix(self, _index: int) -> np.ndarray:
        return self._X

    def getSpikeVector(self, _index: int) -> np.ndarray:
        return self._y

    def getCov(self, _index: int) -> SimpleNamespace:
        n = self._X.shape[0]
        return SimpleNamespace(time=np.arange(n, dtype=float) / self.sampleRate)


def test_glmfit_nan_row_matches_matlab_statremovenan(gold) -> None:
    X = np.asarray(gold["X"], dtype=float)
    y = np.asarray(gold["y"], dtype=float).reshape(-1)
    sample_rate = float(np.asarray(gold["sampleRate"]).reshape(-1)[0])
    nan_row0 = int(np.asarray(gold["nanRow"]).reshape(-1)[0]) - 1  # MATLAB 1-based -> 0-based

    assert np.isnan(X[nan_row0]).any()
    assert np.isfinite(y).all()

    trial = _StubTrial(X, y, sample_rate)
    result = Analysis.GLMFit(trial, 0, 0, "GLM")

    want_b = np.asarray(gold["b"], dtype=float).reshape(-1)
    want_se = np.asarray(gold["se"], dtype=float).reshape(-1)
    want_dev = float(np.asarray(gold["dev"]).reshape(-1)[0])
    want_AIC = float(np.asarray(gold["AIC"]).reshape(-1)[0])
    want_BIC = float(np.asarray(gold["BIC"]).reshape(-1)[0])
    want_logLL = float(np.asarray(gold["logLL"]).reshape(-1)[0])

    got_b = np.asarray(result.b, dtype=float).reshape(-1)
    got_se = np.asarray(result.stats["se"], dtype=float).reshape(-1)

    # Tolerances are the measured diffs (b ~2.2e-16, se ~2.6e-10, dev/AIC/BIC
    # exactly 0, logLL ~7.1e-15), not placeholders: this fixture converges to
    # round-off, no eps-floor cancellation sensitivity in practice.
    np.testing.assert_allclose(got_b, want_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_se, want_se, rtol=0, atol=1e-8)
    assert np.isclose(result.dev, want_dev, atol=1e-9), (result.dev, want_dev)
    assert np.isclose(result.AIC, want_AIC, atol=1e-9), (result.AIC, want_AIC)
    assert np.isclose(result.BIC, want_BIC, atol=1e-9), (result.BIC, want_BIC)
    assert np.isclose(result.logLL, want_logLL, atol=1e-10), (result.logLL, want_logLL)

    # The NaN row of lambda (rate_hz / sampleRate) must itself still be NaN
    # (MATLAB's `data = exp(X*b)` is NaN there too); it is the *floored*
    # eps/eps-complement terms feeding logLL that are finite, not this row.
    lambda_sig_data = np.asarray(result.lambda_signal.data, dtype=float).reshape(-1)
    assert np.isnan(lambda_sig_data[nan_row0])


def test_matlab_max_scalar_ignores_nan() -> None:
    from nstat.analysis import _matlab_max_scalar

    a = np.array([np.nan, 1.0, -5.0, np.nan])
    out = _matlab_max_scalar(a, 2.22e-16)
    assert np.array_equal(out, np.array([2.22e-16, 1.0, 2.22e-16, 2.22e-16]))
