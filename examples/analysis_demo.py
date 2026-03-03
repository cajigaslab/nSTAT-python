"""Analysis demo aligned to MATLAB GLM workflow with deterministic arrays."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import Analysis


def main() -> None:
    X = np.array(
        [
            [-1.00, 0.20],
            [-0.50, -0.10],
            [0.00, 0.00],
            [0.30, 0.80],
            [0.70, -0.60],
            [1.10, 0.40],
            [1.60, -1.20],
            [2.00, 0.90],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 2, 1, 3, 2, 4], dtype=float)

    fit = Analysis.fitGLM(X, y, fitType="poisson", dt=0.1)
    resid = Analysis.computeFitResidual(y, X, fit, dt=0.1)
    transformed = Analysis.computeInvGausTrans(y, X, fit, dt=0.1)
    ks = Analysis.computeKSStats(transformed)

    print("Fit intercept:", float(fit.intercept))
    print("Fit coefficients:", np.asarray(fit.coefficients, dtype=float).tolist())
    print("Log-likelihood:", float(fit.log_likelihood))
    print("Residual shape:", np.asarray(resid).shape)
    print("Transformed events:", np.asarray(transformed).shape)
    print("KS stats:", ks)


if __name__ == "__main__":
    main()
