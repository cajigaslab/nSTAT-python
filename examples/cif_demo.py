"""CIF demo aligned to MATLAB CIFExamples core derivatives workflow."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import CIF


def main() -> None:
    beta = np.array([0.4, -0.25], dtype=float)
    stim = np.array(
        [
            [-1.00, 0.20],
            [-0.25, -0.50],
            [0.00, 0.00],
            [0.50, 0.70],
            [1.20, -1.00],
        ],
        dtype=float,
    )

    cif = CIF(coefficients=beta, intercept=0.0, link="poisson")
    lam_delta = cif.evalLambdaDelta(stim)
    grad = cif.evalGradient(stim)
    jac = cif.evalJacobian(stim)

    print("CIF link:", cif.link)
    print("Stimulus shape:", stim.shape)
    print("Lambda*delta shape:", np.asarray(lam_delta).shape)
    print("Gradient shape:", np.asarray(grad).shape)
    print("Jacobian shape:", np.asarray(jac).shape)


if __name__ == "__main__":
    main()
