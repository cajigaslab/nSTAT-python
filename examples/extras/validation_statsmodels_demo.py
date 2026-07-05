"""Demo: regression guard for a clinical spike-encoding GLM -- nstat vs. statsmodels.

Before a clinical encoding-model paper's tuning-coefficient claims can
be trusted, any silent change to nstat's own Poisson-GLM fitter must
be caught immediately. This demo is that regression guard: it refits
the identical spike-encoding design matrix with
``statsmodels.genmod.GLM`` -- an independent, general-purpose
statistical-modeling package (Seabold & Perktold 2010) that, like
nstat, fits the Poisson GLM by IRLS (unlike NeMoS's optax-driven
first-order optimizer; see
:mod:`nstat.extras.validation.nemos_bridge`). Because both fitters
solve the same IRLS optimum, the recovered coefficients agree to
**near machine precision** -- the tightest cross-validation oracle in
``nstat.extras.validation``, and the one best suited to catching a
silent regression in nstat's own IRLS path.

Generates the same synthetic Poisson spike-encoding-GLM fixture as the
NeMoS demo (``validation_nemos_demo.py``) -- a synthetic unit's known
encoding coefficients driving spike counts through a log-linear
Poisson GLM -- and cross-checks it against
:mod:`nstat.extras.validation.statsmodels_bridge`:

- :func:`cross_validate_poisson_glm` returns a
  :class:`StatsmodelsGLMComparison`.
- Default ``atol=1e-3`` is loose enough for typical real data; the
  agreement on synthetic well-conditioned fixtures is ~1e-9.

References:
    Seabold S, Perktold J (2010). statsmodels: Econometric and
    Statistical Modeling with Python. Proceedings of the 9th Python in
    Science Conference (SciPy 2010), pp. 92-96. This is the
    statsmodels software/proceedings paper (not a neuroscience paper);
    it is cited here because statsmodels is the independent,
    general-purpose reference GLM implementation this demo
    cross-checks against.

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

    # --- Synthetic spike-encoding GLM fixture (n=1000 bins, p=3 --------
    # --- encoding covariates) -------------------------------------------
    # A synthetic unit's known intercept/coefficients standing in for a
    # clinical encoding model's tuning parameters, driving spike counts
    # through a log-linear Poisson GLM. Identical to the fixture used
    # in validation_nemos_demo.py, so the two regression-guard demos
    # share the same ground truth.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 3))
    beta_true = np.array([0.2, -0.4, 0.1])
    intercept_true = 0.5
    rates = np.exp(intercept_true + X @ beta_true)
    y = rng.poisson(rates)
    print(f"Fixture       : {len(y)} bins, {X.shape[1]} encoding covariates, "
          f"E[spike count]={y.mean():.2f}")
    print(f"True β        : intercept={intercept_true:.3f}, "
          f"encoding coef={beta_true.tolist()}")

    # --- Refit the identical design matrix in statsmodels: the ----------
    # --- regression guard ------------------------------------------------
    # Both nstat and statsmodels solve the Poisson GLM by IRLS, so any
    # regression in nstat's own IRLS path shows up here immediately as
    # a coefficient mismatch against this independent implementation.
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
        print("PARITY OK     : nstat ↔ statsmodels encoding coefficients "
              "agree to near machine precision -- no regression detected "
              "in nstat's IRLS path.")
        return 0
    except AssertionError as exc:
        print(f"PARITY MISS   : {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
