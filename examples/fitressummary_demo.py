"""FitResSummary demo aligned to MATLAB FitResSummaryExamples diff metrics."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import FitResSummary
from nstat.compat.matlab import FitResult


def main() -> None:
    fit1 = FitResult(
        coefficients=np.array([0.4, -0.2], dtype=float),
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=-1.6,
        n_samples=5,
        n_parameters=0,
        parameter_labels=["stim1", "stim2"],
    )
    fit2 = FitResult(
        coefficients=np.array([0.1, 0.3], dtype=float),
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=-1.4,
        n_samples=5,
        n_parameters=0,
        parameter_labels=["stim1", "stim2"],
    )

    summary = FitResSummary([fit1, fit2])
    print("Delta AIC vs fit #1:", np.asarray(summary.getDiffAIC(1, False), dtype=float).tolist())
    print("Delta BIC vs fit #1:", np.asarray(summary.getDiffBIC(1, False), dtype=float).tolist())
    print("Delta logLL vs fit #1:", np.asarray(summary.getDifflogLL(1, False), dtype=float).tolist())


if __name__ == "__main__":
    main()
