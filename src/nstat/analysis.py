"""Model fitting and analysis entry points."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .fit import FitResult
from .trial import Trial, TrialConfig


class Analysis:
    """Static analysis methods for point-process GLM fitting.

    This class intentionally mirrors MATLAB's class-centric access pattern,
    while returning plain typed Python result objects.
    """

    @staticmethod
    def fit_trial(trial: Trial, config: TrialConfig, unit_index: int = 0) -> FitResult:
        """Fit Poisson/binomial GLM for a single unit within a trial."""

        dt = 1.0 / config.sample_rate_hz
        _, y, X = trial.aligned_binned_observation(bin_size_s=dt, unit_index=unit_index)

        n_features = X.shape[1]
        theta0 = np.zeros(n_features + 1, dtype=float)

        def unpack(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return float(theta[0]), theta[1:]

        if config.fit_type == "poisson":

            def objective(theta: np.ndarray) -> float:
                b0, b = unpack(theta)
                eta = b0 + X @ b
                lam = np.exp(eta)
                # Negative log-likelihood for independent Poisson bins
                return float(np.sum(lam - y * eta))

        else:

            def objective(theta: np.ndarray) -> float:
                b0, b = unpack(theta)
                eta = b0 + X @ b
                p = 1.0 / (1.0 + np.exp(-eta))
                p = np.clip(p, 1e-9, 1.0 - 1e-9)
                return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        opt = minimize(objective, theta0, method="L-BFGS-B")
        if not opt.success:
            raise RuntimeError(f"GLM optimization failed: {opt.message}")

        intercept = float(opt.x[0])
        coeffs = opt.x[1:].astype(float)
        nll = float(opt.fun)

        return FitResult(
            coefficients=coeffs,
            intercept=intercept,
            fit_type=config.fit_type,
            log_likelihood=-nll,
            n_samples=int(y.size),
            n_parameters=int(opt.x.size),
        )
