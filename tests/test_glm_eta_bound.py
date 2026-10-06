"""``Analysis.GLMFit``'s poisson path must use MATLAB ``glmfit``'s actual
link bound for its Newton iterations, and MATLAB's unclipped ``exp(X*b)``
for its downstream prediction -- not an arbitrary ``+-20`` clip on both.

MATLAB's ``stattestlink.m`` (R2026a,
``toolbox/stats/stats/private/stattestlink.m``) constrains the argument to
the ``'log'`` (poisson canonical) inverse link to a specific bound before
applying ``exp``: ``tiny = realmin(class)^.25``, ``bound = -log(tiny)`` ->
``+-177.0991046330660...`` for double. This bound governs ``glmfit``'s own
*fitting iterations* only. ``Analysis.GLMFit`` (``Analysis.m:565-634``) then
evaluates its own ``data = exp(X*b)`` with **no clip at all** -- a value
that can legitimately overflow to ``Inf``, exactly as MATLAB's unclipped
``exp`` would.

``nstat/glm.py`` used a flat ``np.clip(eta, -20.0, 20.0)`` at every eta site
in ``fit_poisson_glm`` / ``fit_binomial_glm`` (and their result classes'
``predict_rate`` / ``predict_probability``) -- including inside
``Analysis.GLMFit``'s poisson path, where the correct value is MATLAB's
``+-177.1`` for the fitting iterations and *no clip* for the final
prediction. The binomial (``'BNLRCG'``) path has no MATLAB bound to adopt
at all: MATLAB's ``Algorithm == 'BNLRCG'`` calls ``Analysis.m``'s own nested
``bnlrCG`` (not ``glmfit``), which computes ``u = exp(n)./(1+exp(n))`` with
no constrain. Every caller with no MATLAB counterpart (paper examples,
extras, tutorials, docs figures) keeps the original Python-only ``+-20``
default -- widening it was shown to destabilize at least one such caller (a
near-collinear tensor-product B-spline design with no Newton damping to
compensate; see ``tests/extras/test_spatial_basis.py``).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nstat.analysis import Analysis
from nstat.glm import (
    _DEFAULT_ETA_BOUND,
    _MATLAB_GLMFIT_POISSON_ETA_BOUND,
    PoissonGLMResult,
    fit_poisson_glm,
)


class _StubTrial:
    """Minimal duck-typed ``tObj`` exposing exactly what ``Analysis.GLMFit``
    reads (see ``tests/test_glmfit_nan_rows_matlab_gold.py`` for the same
    pattern, used there for the same reason: a real ``Trial`` cannot easily
    carry an engineered large-eta design through its spike-binning
    machinery)."""

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


def test_matlab_glmfit_poisson_eta_bound_matches_stattestlink_formula() -> None:
    # stattestlink.m 'log' case: tiny = realmin(class)^.25; bound = -log(tiny).
    tiny = np.finfo(float).tiny ** 0.25
    expected = -np.log(tiny)
    assert abs(expected - 177.0991046330660) < 1e-9
    assert _MATLAB_GLMFIT_POISSON_ETA_BOUND == expected
    assert _MATLAB_GLMFIT_POISSON_ETA_BOUND > _DEFAULT_ETA_BOUND


def test_default_eta_bound_is_unchanged_python_only_value() -> None:
    # No caller without a MATLAB counterpart should see any behavior change
    # from the stattestlink.m correction; the default stays at its original
    # value.
    assert _DEFAULT_ETA_BOUND == 20.0


def test_fit_poisson_glm_eta_bound_is_keyword_configurable() -> None:
    # Analysis.GLMFit passes eta_bound=_MATLAB_GLMFIT_POISSON_ETA_BOUND;
    # every other caller relies on the default (20.0) being unchanged.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    y = rng.poisson(1.0, size=200).astype(float)

    default_result = fit_poisson_glm(x, y, l2=1e-3)
    wide_result = fit_poisson_glm(x, y, l2=1e-3, eta_bound=_MATLAB_GLMFIT_POISSON_ETA_BOUND)
    # On a well-posed, well-converged fit (|eta| never near either bound),
    # both bounds give the identical answer -- the bound only matters when
    # the iteration path would otherwise cross it.
    np.testing.assert_allclose(default_result.coefficients, wide_result.coefficients, atol=1e-8)


def test_poisson_predict_rate_still_uses_the_python_only_default_bound() -> None:
    # PoissonGLMResult.predict_rate has no MATLAB counterpart (it is not
    # used by Analysis.GLMFit, which computes `data = exp(X*b)` directly,
    # unclipped); its own clip stays at the original +-20 default so every
    # direct caller (paper examples, extras, tutorials) is unaffected by
    # the stattestlink.m correction.
    result = PoissonGLMResult(
        intercept=25.0, coefficients=np.zeros(1), n_iter=1, converged=True,
        log_likelihood=0.0,
    )
    rate = result.predict_rate(np.zeros((1, 1)))
    assert np.isclose(rate[0], np.exp(_DEFAULT_ETA_BOUND))
    assert not np.isclose(rate[0], np.exp(25.0))


def test_glmfit_poisson_path_passes_matlab_eta_bound_to_the_fit(monkeypatch) -> None:
    # Structural test, not an end-to-end large-eta fit: a real Newton-IRLS
    # fit from beta=0 on a huge-count design (e.g. y ~ 1e11) does not
    # converge to the MLE either way (it diverges from its first step,
    # unlike MATLAB glmfit, which starts from startingVals(y) --
    # Analysis.GLMFit does not mirror that initialization; see the module
    # docstring's "glmfit starting-value gap" note), so asserting against
    # its *output* there would pass vacuously once both sides overflow to
    # the same inf. Instead, intercept the fit_poisson_glm call itself and
    # check Analysis.GLMFit passes MATLAB's bound.
    import nstat.analysis as analysis_mod

    captured_kwargs = {}
    real_fit_poisson_glm = analysis_mod.fit_poisson_glm

    def _spy(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return real_fit_poisson_glm(*args, **kwargs)

    monkeypatch.setattr(analysis_mod, "fit_poisson_glm", _spy)

    n = 20
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.poisson(2.0, size=n).astype(float)
    trial = _StubTrial(X, y, 1.0)
    Analysis.GLMFit(trial, 0, 0, "GLM", l2=0.0)

    assert captured_kwargs.get("eta_bound") == _MATLAB_GLMFIT_POISSON_ETA_BOUND


def test_glmfit_poisson_path_uses_unclipped_exp_not_predict_rate(monkeypatch) -> None:
    # Fake a converged fit with a coefficient well past +-20 in magnitude
    # (no real IRLS run needed -- beta=0 Newton genuinely diverges on a
    # design extreme enough to need this, per the test above) and confirm
    # Analysis.GLMFit's lambda_signal is the raw, unclipped exp(eta) that
    # MATLAB's `data = exp(X*b)` computes, not exp(clip(eta, -20, 20))
    # (what calling PoissonGLMResult.predict_rate would give -- its own
    # default clip is unrelated to Analysis.GLMFit and intentionally
    # unchanged; see the module docstring). This is the real fails-before:
    # on base @ 2566769 (before any eta-bound fix), predict_rate clips at
    # +-20 and this assertion fails (exp(20), not exp(25)).
    import nstat.analysis as analysis_mod

    # Analysis.GLMFit calls fit_poisson_glm with include_intercept=False (X
    # already carries its own constant column), so only `coefficients` (not
    # `intercept`, which Analysis.GLMFit never reads) determines eta = X@b.
    fake_result = PoissonGLMResult(
        intercept=0.0, coefficients=np.array([25.0]), n_iter=1, converged=True,
        log_likelihood=0.0,
    )
    monkeypatch.setattr(analysis_mod, "fit_poisson_glm", lambda *a, **k: fake_result)

    n = 5
    X = np.ones((n, 1), dtype=float)  # one constant column; full rank, kept stays None
    y = np.zeros(n, dtype=float)
    trial = _StubTrial(X, y, 1.0)
    result = Analysis.GLMFit(trial, 0, 0, "GLM", l2=0.0)

    lambda_got = np.asarray(result.lambda_signal.data, dtype=float).reshape(-1)[0]
    np.testing.assert_allclose(lambda_got, np.exp(25.0), rtol=1e-9)
    assert not np.isclose(lambda_got, np.exp(_DEFAULT_ETA_BOUND))
