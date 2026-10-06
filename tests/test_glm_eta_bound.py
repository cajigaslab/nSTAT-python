"""``nstat.glm``'s linear-predictor clip must match MATLAB ``glmfit``'s actual
link-specific bound, not an arbitrary +-20.

MATLAB's ``stattestlink.m`` (R2026a,
``toolbox/stats/stats/private/stattestlink.m``) constrains the argument to
each canonical inverse link to a link-specific bound before applying the
elementary function:

* ``log`` (Poisson canonical): ``tiny = realmin(class)^.25``,
  ``bound = -log(tiny)`` -> ``+-177.0991046330660...`` for double.
* ``logit`` (binomial canonical): ``bound = -log(eps(class))`` ->
  ``+-36.04365338911715...`` for double.

``nstat/glm.py`` used a flat ``np.clip(eta, -20.0, 20.0)`` at all six eta
sites (``fit_poisson_glm`` / ``fit_binomial_glm`` fit loops and
``PoissonGLMResult.predict_rate`` / ``BinomialGLMResult.predict_probability``),
tighter than MATLAB's real bound in both cases. Since ``fit_poisson_glm`` /
``fit_binomial_glm`` back ``Analysis.GLMFit``'s GLM / BNLRCG paths
(``nstat/analysis.py``), that mismatch could diverge from MATLAB on any
design whose fitted linear predictor exceeds +-20 in magnitude.
"""
from __future__ import annotations

import numpy as np

from nstat.glm import (
    BinomialGLMResult,
    PoissonGLMResult,
    _BINOMIAL_ETA_BOUND,
    _POISSON_ETA_BOUND,
)


def test_poisson_eta_bound_matches_matlab_log_link_formula() -> None:
    # stattestlink.m 'log' case: tiny = realmin(class)^.25; bound = -log(tiny).
    tiny = np.finfo(float).tiny ** 0.25
    expected = -np.log(tiny)
    assert expected == 177.0991046330660 or abs(expected - 177.0991046330660) < 1e-9
    assert _POISSON_ETA_BOUND == expected
    assert _POISSON_ETA_BOUND > 20.0, "must be wider than the old (wrong) +-20 clip"


def test_binomial_eta_bound_matches_matlab_logit_link_formula() -> None:
    # stattestlink.m 'logit' case: bound = -log(eps(class)).
    expected = -np.log(np.finfo(float).eps)
    assert abs(expected - 36.04365338911715) < 1e-9
    assert _BINOMIAL_ETA_BOUND == expected
    assert _BINOMIAL_ETA_BOUND > 20.0, "must be wider than the old (wrong) +-20 clip"


def test_poisson_predict_rate_not_clipped_at_old_plus20_bound() -> None:
    # eta = 25 lies strictly between the old (wrong) +-20 clip and the
    # correct MATLAB log-link bound (+-177.1); predict_rate must return
    # exp(25) exactly, not exp(20) (what the pre-fix clip would have given).
    result = PoissonGLMResult(
        intercept=25.0, coefficients=np.zeros(1), n_iter=1, converged=True,
        log_likelihood=0.0,
    )
    rate = result.predict_rate(np.zeros((1, 1)))
    assert np.isclose(rate[0], np.exp(25.0)), rate
    assert not np.isclose(rate[0], np.exp(20.0))


def test_binomial_predict_probability_not_clipped_at_old_plus20_bound() -> None:
    # eta = 25 lies strictly between the old (wrong) +-20 clip and the
    # correct MATLAB logit-link bound (+-36.04).
    result = BinomialGLMResult(
        intercept=25.0, coefficients=np.zeros(1), n_iter=1, converged=True,
        log_likelihood=0.0,
    )
    p = result.predict_probability(np.zeros((1, 1)))
    expected = 1.0 / (1.0 + np.exp(-25.0))
    old_clip_value = 1.0 / (1.0 + np.exp(-20.0))
    assert np.isclose(p[0], expected), p
    # Both values saturate near 1.0, so compare in logit space rather than
    # with isclose (which would call 1-1e-11 and 1-2e-9 "close").
    assert p[0] > old_clip_value, (p[0], old_clip_value)
    logit_p = np.log(p[0] / (1.0 - p[0]))
    assert np.isclose(logit_p, 25.0), logit_p
