"""Cross-validate :func:`nstat.fit_poisson_glm` against statsmodels.

`statsmodels.genmod.GLM` is the third independent Poisson-GLM
implementation triangulated by ``nstat.extras.validation``:

- :mod:`nstat.extras.validation.nemos_bridge` — Flatiron NeMoS (JAX).
- :mod:`nstat.extras.validation.statsmodels_bridge` — statsmodels (this
  module).
- :mod:`nstat.fit_poisson_glm` itself (IRLS with a tiny ridge).

Triangulating against two **unrelated** reference implementations is
much stronger than against one — statsmodels uses IRLS like nstat but
with a different stopping criterion and no ridge, while NeMoS uses an
optax-driven first-order optimizer entirely. Coefficients that agree
across all three to high precision are far less likely to be the
result of a shared implementation bug.

Install:
    pip install nstat-toolbox[test-parity]   # or just: pip install statsmodels
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nstat import fit_poisson_glm


_IMPORT_ERROR_MSG = (
    "nstat.extras.validation.statsmodels_bridge requires the 'statsmodels' "
    "package, which is not installed.  "
    "Install with: pip install nstat-toolbox[test-parity]"
)


def _require_statsmodels():
    try:
        import statsmodels.api  # noqa: F401
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e


@dataclass(frozen=True)
class StatsmodelsGLMComparison:
    """Side-by-side Poisson-GLM fit comparison: nstat vs statsmodels.

    Attributes
    ----------
    nstat_coef
        Coefficient vector from :func:`nstat.fit_poisson_glm`.  Includes
        the intercept as the first element if the fit included one.
    statsmodels_coef
        Coefficient vector from
        :func:`statsmodels.api.GLM(...).fit().params`.  Layout matches
        ``nstat_coef`` (intercept-first if present).
    coef_inf_norm
        :math:`\\|\\beta_{nstat} - \\beta_{statsmodels}\\|_\\infty`.
    coef_rel_inf_norm
        Same but normalized by ``|nstat_coef|_inf`` for unitless reports.
    """

    nstat_coef: np.ndarray
    statsmodels_coef: np.ndarray
    coef_inf_norm: float
    coef_rel_inf_norm: float

    def assert_agree(self, atol: float = 1e-3, rtol: float = 1e-3) -> None:
        """Assert nstat and statsmodels coefficients agree within tolerance.

        Default ``atol=1e-3`` reflects the empirical baseline: both
        libraries use IRLS but with different ridge / stopping
        conventions, and on a 1000-sample fixture the disagreement
        runs ~1e-4 to ~1e-3.  Tighten only on a per-fixture basis.

        Raises
        ------
        AssertionError
            If neither the absolute (``atol``) nor relative (``rtol``)
            tolerance is met.
        """
        if not (self.coef_inf_norm <= atol or self.coef_rel_inf_norm <= rtol):
            raise AssertionError(
                f"nstat vs statsmodels Poisson GLM coefficients disagree: "
                f"|Δβ|_∞ = {self.coef_inf_norm:.3e}, "
                f"relative = {self.coef_rel_inf_norm:.3e} "
                f"(atol={atol}, rtol={rtol})"
            )


def cross_validate_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    include_intercept: bool = True,
) -> StatsmodelsGLMComparison:
    """Fit a Poisson GLM with both nstat and statsmodels; return comparison.

    Both fits use canonical log-link, no regularization on the
    statsmodels side, and nstat's default tiny ridge (``l2=1e-6``).

    Parameters
    ----------
    X
        Design matrix, shape ``(n_samples, n_features)``.
    y
        Spike-count vector, shape ``(n_samples,)``.
    include_intercept
        If ``True`` (default), both fits include an intercept and the
        returned coefficient vectors are intercept-first.

    Returns
    -------
    StatsmodelsGLMComparison

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((1000, 3))
    >>> beta_true = np.array([0.2, -0.4, 0.1])
    >>> y = rng.poisson(np.exp(0.5 + X @ beta_true))
    >>> cmp = cross_validate_poisson_glm(X, y)  # doctest: +SKIP
    >>> cmp.assert_agree(atol=1e-2)             # doctest: +SKIP
    """
    _require_statsmodels()
    import statsmodels.api as sm_api

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # --- nstat fit ------------------------------------------------------
    nstat_result = fit_poisson_glm(X, y, include_intercept=include_intercept)
    if include_intercept:
        nstat_coef = np.concatenate(
            ([nstat_result.intercept], nstat_result.coefficients)
        )
    else:
        nstat_coef = np.asarray(nstat_result.coefficients, dtype=float)

    # --- statsmodels fit ------------------------------------------------
    X_sm = sm_api.add_constant(X) if include_intercept else X
    glm = sm_api.GLM(y, X_sm, family=sm_api.families.Poisson())
    sm_fit = glm.fit()
    sm_coef = np.asarray(sm_fit.params, dtype=float)

    # --- Diff -----------------------------------------------------------
    diff = nstat_coef - sm_coef
    inf_norm = float(np.max(np.abs(diff)))
    denom = max(float(np.max(np.abs(nstat_coef))), 1e-12)
    rel_inf_norm = inf_norm / denom

    return StatsmodelsGLMComparison(
        nstat_coef=nstat_coef,
        statsmodels_coef=sm_coef,
        coef_inf_norm=inf_norm,
        coef_rel_inf_norm=rel_inf_norm,
    )


__all__ = ["StatsmodelsGLMComparison", "cross_validate_poisson_glm"]
