"""Cross-validate :func:`nstat.fit_poisson_glm` against NeMoS.

NeMoS (https://github.com/flatironinstitute/nemos) is a JAX-backed
Poisson/Gamma GLM toolbox from the Flatiron Institute Center for
Computational Neuroscience.  It uses the *same* raised-cosine / B-spline
basis families that nstat does (and that the 2012 paper assumes), with
an MIT license that is GPL-2 compatible.

This bridge fits the same Poisson GLM in both libraries on a
user-supplied design matrix and response vector, then returns a
diff-comparison object so callers can assert agreement to a chosen
tolerance.  It is the strongest available Python-side cross-check on
:func:`nstat.fit_poisson_glm` short of running MATLAB-engine itself.

Install:
    pip install nstat-toolbox[test-parity]   # or just: pip install nemos
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nstat import fit_poisson_glm
from nstat.extras._lazy import require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def _require_nemos() -> object:
    return require_optional("nemos", install_key="test-parity")


@dataclass(frozen=True)
class GLMComparison:
    """Side-by-side Poisson-GLM fit comparison.

    Attributes
    ----------
    nstat_coef
        Coefficient vector from :func:`nstat.fit_poisson_glm`.  Includes
        the intercept as the first element if the nstat fit included one.
    nemos_coef
        Coefficient vector from ``nemos.glm.GLM(...).fit(...)``.
        Layout matches ``nstat_coef`` (intercept-first if present).
    coef_inf_norm
        :math:`\\|\\beta_{nstat} - \\beta_{nemos}\\|_\\infty`.  Use this
        as the assertion target in parity tests.
    coef_rel_inf_norm
        Same but normalized by ``|nstat_coef|_inf`` for unitless reports.
    """

    nstat_coef: np.ndarray
    nemos_coef: np.ndarray
    coef_inf_norm: float
    coef_rel_inf_norm: float

    def assert_agree(self, atol: float = 1e-3, rtol: float = 1e-3) -> None:
        """Assert nstat and NeMoS coefficients agree within tolerance.

        Raises
        ------
        AssertionError
            If neither the absolute (``atol``) nor relative (``rtol``)
            tolerance is met.
        """
        if not (self.coef_inf_norm <= atol or self.coef_rel_inf_norm <= rtol):
            raise AssertionError(
                f"nstat vs NeMoS Poisson GLM coefficients disagree: "
                f"|Δβ|_∞ = {self.coef_inf_norm:.3e}, "
                f"relative = {self.coef_rel_inf_norm:.3e} "
                f"(atol={atol}, rtol={rtol})"
            )


def cross_validate_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    include_intercept: bool = True,
) -> GLMComparison:
    """Fit a Poisson GLM with both nstat and NeMoS; return the comparison.

    Both fits use canonical log-link, no L2 regularization on the NeMoS
    side, and a tiny ridge on the nstat side (its default ``l2=1e-6``);
    the agreement target should be loose enough to absorb that ridge.

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
    GLMComparison

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 3))
    >>> beta_true = np.array([0.2, -0.4, 0.1])
    >>> rates = np.exp(0.5 + X @ beta_true)
    >>> y = rng.poisson(rates)
    >>> cmp = cross_validate_poisson_glm(X, y)  # doctest: +SKIP
    >>> cmp.assert_agree(atol=1e-2)             # doctest: +SKIP
    """
    nemos = _require_nemos()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # nstat fit
    nstat_result = fit_poisson_glm(X, y, include_intercept=include_intercept)
    if include_intercept:
        nstat_coef = np.concatenate(([nstat_result.intercept], nstat_result.coefficients))
    else:
        nstat_coef = np.asarray(nstat_result.coefficients, dtype=float)

    # NeMoS fit
    glm_cls = nemos.glm.GLM
    model = glm_cls()
    model.fit(X, y)

    coef = np.asarray(model.coef_, dtype=float).ravel()
    if include_intercept:
        nemos_coef = np.concatenate(([float(model.intercept_)], coef))
    else:
        nemos_coef = coef

    diff = nstat_coef - nemos_coef
    inf_norm = float(np.max(np.abs(diff)))
    denom = max(float(np.max(np.abs(nstat_coef))), 1e-12)
    rel_inf_norm = inf_norm / denom

    return GLMComparison(
        nstat_coef=nstat_coef,
        nemos_coef=nemos_coef,
        coef_inf_norm=inf_norm,
        coef_rel_inf_norm=rel_inf_norm,
    )


__all__ = ["GLMComparison", "cross_validate_poisson_glm"]
