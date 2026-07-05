"""Demo: reviewer-grade cross-check of a reach-tuned encoding GLM.

A single-unit motor-cortex *encoding model* -- firing rate driven by
reach-kinematic covariates (e.g. hand-velocity components and speed)
through a log-linear Poisson GLM -- is an instance of the point-process
Poisson-GLM encoding family; Weber & Pillow (2017) show GLMs of this
form reproduce a broad repertoire of single-neuron response dynamics.
Before a clinical encoding-model paper's tuning-coefficient claims can
be trusted, reviewers expect the identical design matrix to be refit in
a second, independently-maintained implementation and shown to recover
the same coefficients: the numerical cross-check this demo performs.

Generates a synthetic Poisson spike-count fixture with known
coefficients -- standing in for a reach-tuned unit's encoding
coefficients -- fits the same GLM in :func:`nstat.fit_poisson_glm` and
in NeMoS's :class:`nemos.glm.GLM` (NeMoS: a JAX-backed point-process
GLM toolbox from the Flatiron Institute Center for Computational
Neuroscience, https://github.com/flatironinstitute/nemos), then reports
the coefficient agreement between the two independently-maintained
fits.

Demonstrates :mod:`nstat.extras.validation.nemos_bridge`:

- :func:`cross_validate_poisson_glm` returns a :class:`GLMComparison`.
- :meth:`GLMComparison.assert_agree` is the regression-guard hook for
  parity tests.

References:
    Weber AI, Pillow JW (2017). Capturing the Dynamical Repertoire of
    Single Neurons with Generalized Linear Models. Neural Computation
    29(12):3260-3289. arXiv:1602.07389.

    NeMoS (Flatiron Institute Center for Computational Neuroscience).
    https://github.com/flatironinstitute/nemos

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

    # --- Synthetic reach-tuned encoding-GLM fixture (n=1000 bins, ------
    # --- p=3 reach-kinematic covariates) --------------------------------
    # X's 3 columns stand in for reach-kinematic regressors (e.g. hand
    # velocity components + speed); beta_true/intercept_true are the
    # single reach-tuned unit's known encoding coefficients that the
    # fixture is built to recover -- fully synthetic, no real kinematic
    # recording involved.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 3))
    beta_true = np.array([0.2, -0.4, 0.1])
    intercept_true = 0.5
    rates = np.exp(intercept_true + X @ beta_true)
    y = rng.poisson(rates)
    print(f"Fixture       : {len(y)} bins, {X.shape[1]} reach-kinematic "
          f"covariates, E[spike count]={y.mean():.2f}")
    print(f"True β        : intercept={intercept_true:.3f}, "
          f"reach-tuning coef={beta_true.tolist()}")

    # --- Refit the identical design matrix in NeMoS and cross-check -----
    # This is the reviewer-grade rigor check: an independently-maintained
    # implementation (NeMoS, optax-driven first-order optimization) must
    # recover the same encoding coefficients as nstat's IRLS fit on the
    # exact same (X, y) before the tuning-coefficient claim is trustworthy.
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
        print("PARITY OK     : nstat ↔ NeMoS reach-tuning coefficients "
              "agree within tolerance.")
        return 0
    except AssertionError as exc:
        print(f"PARITY MISS   : {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
