"""FitResult demo aligned to MATLAB FitResultExamples core methods."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import FitResult


def main() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=float,
    )
    fit = FitResult(
        coefficients=np.array([0.4, -0.2], dtype=float),
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=-1.6,
        n_samples=X.shape[0],
        n_parameters=2,
        parameter_labels=["stim1", "stim2"],
        xval_data=[X],
        xval_time=[np.arange(X.shape[0], dtype=float)],
    )

    lam = fit.evalLambda(1, X)
    coeff_idx, epoch_id, num_epochs = fit.getCoeffIndex(1, False)
    coeff_mat, coeff_labels, coeff_se = fit.getCoeffs(1)

    print("Lambda:", np.asarray(lam, dtype=float).round(6).tolist())
    print("Coeff index:", np.asarray(coeff_idx, dtype=int).tolist())
    print("Epoch id:", np.asarray(epoch_id, dtype=int).tolist(), "num_epochs:", int(num_epochs))
    print("Coeff matrix:", np.asarray(coeff_mat, dtype=float).round(6).tolist())
    print("Coeff labels:", coeff_labels)
    print("Coeff SE:", np.asarray(coeff_se, dtype=float).round(6).tolist())
    print("Validation present:", bool(fit.isValDataPresent()))


if __name__ == "__main__":
    main()
