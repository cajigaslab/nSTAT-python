"""Demo: cross-validate nstat's Kalman filter against pykalman.

Simulates a 2-state linear-Gaussian process, fits the Kalman filter
(and lag-(T-1) augmented-state smoother) in both nstat and pykalman,
then prints the agreement metric.

Documents the empirical baselines this bridge ships with:

- Filtered means: ~2.6e-3 disagreement (t=0 initialization convention).
- Smoothed means: ~0.4 disagreement (AUDIT D3: nstat's augmented-state
  smoother with finite lag vs pykalman's full-window RTS smoother).

Demonstrates :mod:`nstat.extras.validation.pykalman_bridge`:

- :func:`cross_validate_kalman` returns a :class:`KalmanComparison`.
- :meth:`KalmanComparison.assert_filtered_agree` and
  :meth:`assert_smoothed_agree` are the regression-guard hooks.

Run::

    pip install nstat-toolbox[test-parity]   # also pulls statsmodels, nitime
    python examples/extras/validation_pykalman_demo.py
"""
from __future__ import annotations

import numpy as np


def main() -> int:
    try:
        from nstat.extras.validation.pykalman_bridge import cross_validate_kalman
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    # --- 2-state linear-Gaussian model ----------------------------------
    rng = np.random.default_rng(0)
    T, Dx, Dy = 100, 2, 2
    A = np.eye(Dx) * 0.95
    C = np.eye(Dy)
    Q = np.eye(Dx) * 0.01
    R = np.eye(Dy) * 0.1
    x0 = np.zeros(Dx)
    P0 = np.eye(Dx)

    # --- Simulate -------------------------------------------------------
    x = np.zeros((T, Dx))
    y = np.zeros((T, Dy))
    x[0] = rng.multivariate_normal(x0, P0)
    y[0] = C @ x[0] + rng.multivariate_normal(np.zeros(Dy), R)
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(Dx), Q)
        y[t] = C @ x[t] + rng.multivariate_normal(np.zeros(Dy), R)
    print(f"Fixture       : T={T}, Dx={Dx}, Dy={Dy} linear-Gaussian")

    # --- Cross-validate -------------------------------------------------
    try:
        cmp = cross_validate_kalman(y, A, C, Q, R, x0, P0)
    except ImportError as exc:
        print(f"pykalman missing: {exc}")
        return 1

    print(f"filtered_inf_norm : {cmp.filtered_inf_norm:.3e}  "
          f"(empirical baseline ~2.6e-3, t=0 init convention)")
    print(f"smoothed_inf_norm : {cmp.smoothed_inf_norm:.3e}  "
          f"(AUDIT D3: augmented-state lag ≠ full RTS)")

    cmp.assert_filtered_agree(atol=1e-2)
    print("FILTER PARITY OK   : nstat ↔ pykalman filtered means within tolerance.")
    print("Smoother gap is documented & known (AUDIT D3) — see "
          "AUDIT_REPORT.md §3.1 and parity/integration_opportunities.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
