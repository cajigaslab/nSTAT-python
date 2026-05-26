"""Demo: cross-validate nstat's Poisson GLM against NeMoS.

Generates a synthetic Poisson spike-count fixture with known
coefficients, fits the same model in :func:`nstat.fit_poisson_glm` and
:class:`nemos.glm.GLM`, then reports the coefficient agreement.

Demonstrates :mod:`nstat.extras.validation.nemos_bridge`:

- :func:`cross_validate_poisson_glm` returns a :class:`GLMComparison`.
- :meth:`GLMComparison.assert_agree` is the regression-guard hook for
  parity tests.

Run::

    pip install nstat-toolbox[nemos]   # ~200 MB JAX install
    python examples/extras/validation_nemos_demo.py
"""
from __future__ import annotations

import numpy as np


def main() -> int:
    try:
        from nstat.extras.validation.nemos_bridge import cross_validate_poisson_glm
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    # --- Synthetic Poisson GLM fixture (n=1000, p=3) -------------------
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

    # --- Fit + cross-validate -------------------------------------------
    try:
        cmp = cross_validate_poisson_glm(X, y)
    except ImportError as exc:
        print(f"NeMoS missing: {exc}")
        return 1

    print(f"nstat fit     : {cmp.nstat_coef.tolist()}")
    print(f"NeMoS fit     : {cmp.nemos_coef.tolist()}")
    print(f"|Δβ|_∞        : {cmp.coef_inf_norm:.3e}")
    print(f"relative      : {cmp.coef_rel_inf_norm:.3e}")

    try:
        cmp.assert_agree(atol=5e-2, rtol=5e-2)
        print("PARITY OK     : nstat ↔ NeMoS coefficients within tolerance.")
        return 0
    except AssertionError as exc:
        print(f"PARITY MISS   : {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
