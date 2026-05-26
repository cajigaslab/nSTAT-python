"""Demo: fit a linear-Gaussian state-space model via Dynamax EM.

Generates a synthetic 2-state linear-Gaussian process with known
parameters, then fits the same model via Dynamax's EM (wrapped by
``nstat.extras.em.dynamax_bridge``) and prints the fitted parameters
plus the EM log-likelihood trace.

This is the foundation for closing the ``KF_EM`` / ``PP_EM`` /
``mPPCO_EM`` gap in AUDIT_REPORT.md §3.2 without re-porting ~7,500 LOC
of MATLAB EM code.

Run::

    pip install nstat-toolbox[dynamax]   # pulls JAX (~200 MB)
    python examples/extras/em_dynamax_demo.py
"""
from __future__ import annotations

import numpy as np


def main() -> int:
    try:
        from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    rng = np.random.default_rng(0)
    T, state_dim, emission_dim = 300, 2, 2
    A_true = np.array([[0.95, 0.05], [-0.05, 0.95]])
    C_true = np.eye(emission_dim)
    Q_true = np.eye(state_dim) * 0.02
    R_true = np.eye(emission_dim) * 0.1
    print(f"Fixture       : T={T}, state_dim={state_dim}, "
          f"emission_dim={emission_dim} linear-Gaussian")

    x = np.zeros((T, state_dim))
    y = np.zeros((T, emission_dim))
    x[0] = rng.multivariate_normal(np.zeros(state_dim), np.eye(state_dim))
    y[0] = C_true @ x[0] + rng.multivariate_normal(np.zeros(emission_dim), R_true)
    for t in range(1, T):
        x[t] = A_true @ x[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q_true)
        y[t] = C_true @ x[t] + rng.multivariate_normal(np.zeros(emission_dim), R_true)

    try:
        result = fit_linear_gaussian_em(y, state_dim=state_dim, n_iter=30, seed=0)
    except ImportError as exc:
        print(f"dynamax missing: {exc}")
        return 1

    print(f"\nLearned parameters after {result.n_iter} EM iterations:")
    print(f"  Â =\n{result.transition_matrix}")
    print(f"  Ĉ =\n{result.observation_matrix}")
    print(f"  Q̂ =\n{result.transition_covariance}")
    print(f"  R̂ =\n{result.observation_covariance}")

    print(f"\nEM log-likelihood trace (first → last 3):")
    lls = result.log_likelihoods
    print(f"  {lls[0]:.2f}, {lls[1]:.2f}, {lls[2]:.2f}  →  "
          f"{lls[-3]:.2f}, {lls[-2]:.2f}, {lls[-1]:.2f}")

    diffs = np.diff(lls)
    if np.all(diffs >= -1e-6):
        print(f"\nEM monotonicity OK  : Δll_min = {diffs.min():.3e} (theory says >= 0)")
    else:
        print(f"\nEM monotonicity FAIL: Δll_min = {diffs.min():.3e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
