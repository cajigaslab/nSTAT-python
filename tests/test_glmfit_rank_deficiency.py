"""``Analysis.GLMFit`` on rank-deficient designs mirrors MATLAB ``glmfit``.

MATLAB ``Analysis.GLMFit`` calls ``glmfit(X, y, 'poisson', 'constant', 'off')``.
``glmfit`` (R2025b) takes a column-pivoted QR of the design,
``rankx = sum(abs(diag(R)) > abs(R(1)) * max(n, p) * eps)``, fits on the
``rankx`` pivot columns and returns ``b = 0`` and ``se = 0`` for the dependent
columns.  The port fitted the singular design directly: its standard errors
came from ``inv(X'WX)`` of a singular matrix, clipped at 0, so arbitrary
coefficients passed the EM GLM M-step's ``se < 100`` filter (MATLAB's own F3
test construction: beta [33.0, -620.7] instead of [10.66, 0]; see
``tests/test_em_glm_mstep_matlab_gold.py``).  A full-rank design keeps the
unchanged solver, bit for bit, and so does a ridge fit (``l2 > 0``, Python-only:
MATLAB's glmfit is unpenalized).  (The binomial ``'BNLRCG'`` fit mirrors MATLAB's
``bnlrCG``, which has no rank handling, and is unchanged.)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from nstat.analysis import Analysis, _glmfit_independent_columns
from nstat.glm import fit_poisson_glm


def _trial(x, dN, delta=0.001):
    """A Trial with covariates v1..vd (columns of x) and a constant, as the EM GLM M-step builds it."""
    from nstat._spike_train_impl import nspikeTrain
    from nstat._trial_config_impl import ConfigCollection, TrialConfig
    from nstat.core import Covariate
    from nstat.trial import CovariateCollection, SpikeTrainCollection, Trial

    d, K = x.shape
    time = np.arange(K) * delta
    labels = [f"v{i + 1}" for i in range(d)]
    vel = Covariate(time, x.T, "vel", "time", "s", "m/s", labels)
    base = Covariate(time, np.ones((K, 1)), "Baseline", "time", "s", "", ["constant"])
    nst = nspikeTrain(time[np.flatnonzero(dN == 1)], "", binwidth=delta)
    trial = Trial(SpikeTrainCollection([nst]), CovariateCollection([vel, base]))
    cfg = TrialConfig([["Baseline", "constant"], ["vel", *labels]], 1.0 / delta, None, None)
    return trial, ConfigCollection([cfg])


def _fit(x, dN, **kwargs):
    trial, configs = _trial(x, dN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        configs.setConfig(trial, 0)
        return Analysis.GLMFit(trial, 0, 0, "GLM", **kwargs), np.asarray(trial.getDesignMatrix(0), dtype=float)


def _data(K=1500, seed=3):
    rng = np.random.default_rng(seed)
    x = np.cumsum(0.05 * rng.standard_normal((2, K)), axis=1)
    dN = (rng.random(K) < np.exp(np.log(0.04) + np.array([0.8, -0.5]) @ x)).astype(float)
    return x, dN


def test_independent_columns_follow_glmfit_rule() -> None:
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 3))])
    assert _glmfit_independent_columns(X) is None
    Xd = np.column_stack([X, 2.5 * X[:, 1]])  # an exactly dependent column
    kept = _glmfit_independent_columns(Xd)
    assert kept is not None and kept.size == 4 and np.linalg.matrix_rank(Xd[:, kept]) == 4
    # Below glmfit's tolerance |R11| * max(n, p) * eps, a nearly dependent column is dropped too;
    # well above it (1e-5 noise, MATLAB's F3 "dropped" row), it is kept.
    assert _glmfit_independent_columns(np.column_stack([X, X[:, 1] + 1e-16 * rng.standard_normal(200)])) is not None
    assert _glmfit_independent_columns(np.column_stack([X, 1e-5 * rng.standard_normal(200)])) is None


def test_glmfit_drops_dependent_columns_with_zero_coefficient_and_se() -> None:
    x, dN = _data()
    full, X_full = _fit(x, dN)
    dup, X_dup = _fit(np.vstack([x, 2.0 * x[1]]), dN)  # v3 = 2 v2: rank 3 of 4
    b, se = np.asarray(dup.b, dtype=float), np.asarray(dup.stats["se"], dtype=float)
    dropped = np.flatnonzero(b == 0.0)
    assert dropped.size == 1 and se[dropped[0]] == 0.0
    # The kept columns carry the reduced fit: the same linear predictor as the
    # full-rank fit without the duplicate.
    np.testing.assert_allclose(X_dup @ b, X_full @ np.asarray(full.b, dtype=float), rtol=1e-9, atol=1e-12)
    assert np.all(se[np.flatnonzero(b != 0.0)] > 0)


def test_glmfit_full_rank_design_runs_the_unchanged_solver() -> None:
    # Bit for bit what the solver returns on the design: the rank check adds
    # nothing on a full-rank design.
    x, dN = _data()
    fit, X = _fit(x, dN)
    y = np.asarray(dN, dtype=float)[: X.shape[0]]
    ref = fit_poisson_glm(X, y, include_intercept=False, l2=0.0, max_iter=120)
    np.testing.assert_array_equal(np.asarray(fit.b, dtype=float), ref.coefficients)


@pytest.mark.parametrize("scale", [1.0, -3.0])
def test_glmfit_rank_deficiency_is_not_triggered_by_scaling(scale) -> None:
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(500), scale * rng.standard_normal((500, 2))])
    assert _glmfit_independent_columns(X) is None


def test_nan_rows_are_dropped_from_the_fit_like_matlab_statremovenan() -> None:
    # Before P2a, a design with a NaN sample went straight to the solver,
    # which returned an all-NaN fit (scipy's qr, called with
    # check_finite=True, used to raise "array must not contain infs or
    # NaNs" before that was guarded).  MATLAB glmfit instead drops the NaN
    # rows (statremovenan) and fits the rest; GLMFit now mirrors that
    # (parity/matlab_defects.yml: glmfit-statremovenan-nan-rows). A NaN
    # *column entry* (not a whole non-finite design) still does not trigger
    # the rank-deficiency QR path (_glmfit_independent_columns still returns
    # None for it, per glmfit-rank-handling-notes): the fit runs on the
    # NaN-row-dropped X/y directly.
    from nstat.decoding_algorithms import _em_glm_mstep

    x, dN = _data()
    x[0, 100] = np.nan
    fit, X = _fit(x, dN)
    assert _glmfit_independent_columns(X) is None
    y = np.asarray(dN, dtype=float)[: X.shape[0]]
    nan_rows = np.isnan(X).any(axis=1)
    assert nan_rows.sum() == 1
    ref = fit_poisson_glm(X[~nan_rows], y[~nan_rows], include_intercept=False, l2=0.0, max_iter=120)
    np.testing.assert_array_equal(np.asarray(fit.b, dtype=float), ref.coefficients)
    assert np.all(np.isfinite(np.asarray(fit.b, dtype=float)))
    assert np.all(np.isfinite(np.asarray(fit.stats["se"], dtype=float)))
    assert np.isfinite(float(fit.dev))

    trial, configs = _trial(x, dN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Analysis.RunAnalysisForNeuron(trial, 0, configs, 0)
    assert np.all(np.isfinite(np.asarray(result.b[0], dtype=float)))

    # The GLM M-step picks up the now-finite, NaN-row-dropped fit (previously
    # it kept every previous coefficient, since a NaN se there failed the
    # se < 100 identifiability filter).
    rng = np.random.default_rng(1)
    xs = np.vstack([np.cumsum(0.05 * rng.standard_normal(800)), np.sin(np.arange(800) / 40.0)])
    spikes = (rng.random((2, 800)) < 0.04).astype(float)
    xs[0, 100] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, beta, _ = _em_glm_mstep(spikes, xs, "poisson", np.full(2, -3.0), np.zeros((2, 2)), np.array(0.0), None, 0.001)
    assert np.all(np.isfinite(mu)) and not np.array_equal(mu, [-3.0, -3.0])
    assert np.all(np.isfinite(beta)) and not np.array_equal(beta, np.zeros((2, 2)))


def test_ridge_fit_keeps_every_column() -> None:
    # MATLAB glmfit is unpenalized; with the Python-only ridge (l2 > 0)
    # X'WX + l2 I is invertible, so no column is dropped and every SE is finite.
    x, dN = _data()
    fit, _ = _fit(np.vstack([x, 2.0 * x[1]]), dN, l2=1e-3)
    b, se = np.asarray(fit.b, dtype=float), np.asarray(fit.stats["se"], dtype=float)
    assert np.all(b != 0.0) and np.all(np.isfinite(se)) and np.all(se > 0)
