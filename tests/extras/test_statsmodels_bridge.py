"""Tests for ``nstat.extras.validation.statsmodels_bridge``.

statsmodels is the third independent Poisson-GLM reference triangulated
by ``nstat.extras.validation`` (alongside NeMoS and nstat's own IRLS).
Because both nstat and statsmodels use IRLS, they should agree to
near-machine precision — this is the tightest cross-validation oracle
available without writing a fresh implementation.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Import-error pathway
# ----------------------------------------------------------------------


def test_statsmodels_bridge_emits_install_hint_when_missing() -> None:
    """When statsmodels is absent, the bridge raises a clear ImportError
    naming the pip-install hint."""
    try:
        import statsmodels  # noqa: F401
        pytest.skip("statsmodels is installed; import-error path unreachable")
    except ImportError:
        pass

    from nstat.extras.validation.statsmodels_bridge import (
        cross_validate_poisson_glm,
    )

    n_required = sum(
        1
        for p in inspect.signature(cross_validate_poisson_glm).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    with pytest.raises(ImportError) as excinfo:
        cross_validate_poisson_glm(*[None] * n_required)
    assert "pip install nstat-toolbox[" in str(excinfo.value)


# ----------------------------------------------------------------------
# Functional cross-validation
# ----------------------------------------------------------------------


def test_statsmodels_glm_agrees_with_nstat_to_machine_precision() -> None:
    """nstat IRLS and statsmodels IRLS should agree to ~1e-9 or better
    on a well-conditioned Poisson GLM — same algorithm, near-identical
    stopping criteria.

    This is the **tightest** cross-validation oracle in
    ``nstat.extras.validation`` (NeMoS agrees to ~5e-3; pykalman filter
    to ~1e-2).  A regression that loosens this beyond ~1e-6 likely
    indicates a real bug in nstat's IRLS path.
    """
    pytest.importorskip("statsmodels")
    from nstat.extras.validation.statsmodels_bridge import (
        cross_validate_poisson_glm,
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 3))
    beta_true = np.array([0.2, -0.4, 0.1])
    y = rng.poisson(np.exp(0.5 + X @ beta_true))

    cmp = cross_validate_poisson_glm(X, y)
    # Both implementations recover the true beta well, and agree with
    # each other to near machine precision.
    cmp.assert_agree(atol=1e-6, rtol=1e-6)


def test_statsmodels_comparison_assertion_fires_when_tolerance_violated() -> None:
    """Failure-path test — guards against the assert_agree branch being
    silently bypassed (e.g., inverted comparison)."""
    from nstat.extras.validation.statsmodels_bridge import (
        StatsmodelsGLMComparison,
    )

    cmp = StatsmodelsGLMComparison(
        nstat_coef=np.zeros(3),
        statsmodels_coef=np.zeros(3),
        coef_inf_norm=0.5,         # deliberately > any reasonable atol
        coef_rel_inf_norm=0.5,
    )
    with pytest.raises(AssertionError, match="coefficients disagree"):
        cmp.assert_agree(atol=1e-3, rtol=1e-3)


def test_statsmodels_bridge_handles_no_intercept() -> None:
    """When ``include_intercept=False``, both fits omit the intercept
    and return shape-(p,) coef vectors."""
    pytest.importorskip("statsmodels")
    from nstat.extras.validation.statsmodels_bridge import (
        cross_validate_poisson_glm,
    )

    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 2))
    y = rng.poisson(np.exp(X @ np.array([0.3, -0.2])))

    cmp = cross_validate_poisson_glm(X, y, include_intercept=False)
    assert cmp.nstat_coef.shape == (2,)
    assert cmp.statsmodels_coef.shape == (2,)
    cmp.assert_agree(atol=1e-4, rtol=1e-4)
