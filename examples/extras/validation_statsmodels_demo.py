"""Demo: cross-validate nstat's Poisson GLM against statsmodels.

statsmodels uses IRLS like nstat (different from NeMoS's optax-driven
first-order optimizer), so the two should agree to **near machine
precision** — this is the tightest cross-validation oracle available
in ``nstat.extras.validation``.

Demonstrates :mod:`nstat.extras.validation.statsmodels_bridge`:

- :func:`cross_validate_poisson_glm` returns a
  :class:`StatsmodelsGLMComparison`.
- Default ``atol=1e-3`` is loose enough for typical real data; the
  agreement on synthetic well-conditioned fixtures is ~1e-9.

Run::

    pip install nstat-toolbox[test-parity]   # also installs nemos, pykalman, nitime
    python examples/extras/validation_statsmodels_demo.py
"""
from __future__ import annotations

import numpy as np


def main() -> int:
    try:
        from nstat.extras.validation.statsmodels_bridge import (
            cross_validate_poisson_glm,
        )
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 3))
    beta_true = np.array([0.2, -0.4, 0.1])
    intercept_true = 0.5
    rates = np.exp(intercept_true + X @ beta_true)
    y = rng.poisson(rates)
    print(f"Fixture       : {len(y)} samples, {X.shape[1]} features, "
          f"E[y]={y.mean():.2f}")
    print(f"True β        : intercept={intercept_true:.3f}, "
          f"coef={beta_true.tolist()}")

    try:
        cmp = cross_validate_poisson_glm(X, y)
    except ImportError as exc:
        print(f"statsmodels missing: {exc}")
        return 1

    print(f"nstat fit     : {cmp.nstat_coef.tolist()}")
    print(f"statsmodels   : {cmp.statsmodels_coef.tolist()}")
    print(f"|Δβ|_∞        : {cmp.coef_inf_norm:.3e}  "
          f"(typically <1e-9 — both use IRLS)")
    print(f"relative      : {cmp.coef_rel_inf_norm:.3e}")

    try:
        cmp.assert_agree(atol=1e-6, rtol=1e-6)
        print("PARITY OK     : nstat ↔ statsmodels coefficients agree to "
              "near machine precision.")
        return 0
    except AssertionError as exc:
        print(f"PARITY MISS   : {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
