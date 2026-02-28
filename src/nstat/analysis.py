"""Model fitting and analysis entry points."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from .fit import FitResult
from .trial import Trial, TrialConfig


class Analysis:
    """Static analysis methods for point-process GLM fitting.

    This class intentionally mirrors MATLAB's class-centric access pattern,
    while returning plain typed Python result objects.
    """

    @staticmethod
    def fit_glm(
        X: np.ndarray,
        y: np.ndarray,
        fit_type: str = "poisson",
        dt: float = 1.0,
        l2_penalty: float = 0.0,
    ) -> FitResult:
        """Fit independent-bin GLM with analytical gradients.

        Parameters
        ----------
        X:
            Design matrix with shape ``(n_samples, n_features)``.
        y:
            Observation vector with shape ``(n_samples,)``.
        fit_type:
            ``"poisson"`` or ``"binomial"``.
        dt:
            Bin width in seconds; used for Poisson expected counts.
        l2_penalty:
            Ridge penalty applied to coefficients (not intercept).
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != y.size:
            raise ValueError("X and y must have matching sample count")
        if fit_type not in {"poisson", "binomial"}:
            raise ValueError("fit_type must be 'poisson' or 'binomial'")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if l2_penalty < 0.0:
            raise ValueError("l2_penalty must be non-negative")

        n_features = X.shape[1]
        theta0 = np.zeros(n_features + 1, dtype=float)

        def unpack(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return float(theta[0]), theta[1:]

        if fit_type == "poisson":

            def objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
                b0, b = unpack(theta)
                eta = np.clip(b0 + X @ b, -50.0, 50.0)
                rate = np.exp(eta)
                mu = np.clip(rate * dt, 1e-12, None)
                nll = float(np.sum(mu - y * np.log(mu) + gammaln(y + 1.0)))
                d_eta = mu - y
                grad = np.zeros_like(theta)
                grad[0] = np.sum(d_eta)
                grad[1:] = X.T @ d_eta
                if l2_penalty > 0.0:
                    nll += 0.5 * l2_penalty * float(np.sum(b * b))
                    grad[1:] += l2_penalty * b
                return nll, grad

        else:

            def objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
                b0, b = unpack(theta)
                eta = b0 + X @ b
                p = np.clip(expit(eta), 1e-9, 1.0 - 1e-9)
                nll = float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
                d_eta = p - y
                grad = np.zeros_like(theta)
                grad[0] = np.sum(d_eta)
                grad[1:] = X.T @ d_eta
                if l2_penalty > 0.0:
                    nll += 0.5 * l2_penalty * float(np.sum(b * b))
                    grad[1:] += l2_penalty * b
                return nll, grad

        def objective(theta: np.ndarray) -> float:
            nll, _ = objective_and_grad(theta)
            return nll

        def gradient(theta: np.ndarray) -> np.ndarray:
            _, grad = objective_and_grad(theta)
            return grad

        opt = minimize(objective, theta0, method="L-BFGS-B", jac=gradient)
        if not opt.success:
            raise RuntimeError(f"GLM optimization failed: {opt.message}")

        intercept = float(opt.x[0])
        coeffs = opt.x[1:].astype(float)
        nll = float(opt.fun)
        return FitResult(
            coefficients=coeffs,
            intercept=intercept,
            fit_type=fit_type,
            log_likelihood=-nll,
            n_samples=int(y.size),
            n_parameters=int(opt.x.size),
        )

    @staticmethod
    def fit_trial(trial: Trial, config: TrialConfig, unit_index: int = 0) -> FitResult:
        """Fit Poisson/binomial GLM for a single unit within a trial."""

        dt = 1.0 / config.sample_rate_hz
        mode: Literal["binary", "count"] = "count" if config.fit_type == "poisson" else "binary"
        _, y, X = trial.aligned_binned_observation(bin_size_s=dt, unit_index=unit_index, mode=mode)
        return Analysis.fit_glm(
            X=X,
            y=y,
            fit_type=config.fit_type,
            dt=dt,
        )

    @staticmethod
    def glm_fit(
        X: np.ndarray,
        y: np.ndarray,
        fit_type: str = "poisson",
        dt: float = 1.0,
        l2_penalty: float = 0.0,
    ) -> FitResult:
        """MATLAB-style alias for :meth:`fit_glm`."""

        return Analysis.fit_glm(X=X, y=y, fit_type=fit_type, dt=dt, l2_penalty=l2_penalty)

    @staticmethod
    def run_analysis_for_neuron(trial: Trial, config: TrialConfig, unit_index: int = 0) -> FitResult:
        """High-level wrapper matching MATLAB naming for per-neuron analysis."""

        return Analysis.fit_trial(trial=trial, config=config, unit_index=unit_index)

    @staticmethod
    def run_analysis_for_all_neurons(trial: Trial, config: TrialConfig) -> list[FitResult]:
        """Fit one model per unit in a trial using the same configuration."""

        return [
            Analysis.fit_trial(trial=trial, config=config, unit_index=i)
            for i in range(trial.spikes.n_units)
        ]

    @staticmethod
    def compute_fit_residual(
        y: np.ndarray, X: np.ndarray, fit_result: FitResult, dt: float = 1.0
    ) -> np.ndarray:
        """Return Pearson-like residuals for fitted GLM observations."""

        y = np.asarray(y, dtype=float)
        mu = np.asarray(fit_result.predict(X), dtype=float)
        if fit_result.fit_type == "poisson":
            mu = np.clip(mu * dt, 1e-12, None)
            var = mu
        else:
            mu = np.clip(mu, 1e-9, 1.0 - 1e-9)
            var = np.clip(mu * (1.0 - mu), 1e-12, None)
        return (y - mu) / np.sqrt(var)

    @staticmethod
    def compute_inv_gaus_trans(
        y: np.ndarray, X: np.ndarray, fit_result: FitResult, dt: float = 1.0
    ) -> np.ndarray:
        """Compute integrated-intensity transform surrogate used in GOF diagnostics."""

        y = np.asarray(y, dtype=float)
        mu = np.asarray(fit_result.predict(X), dtype=float)
        if fit_result.fit_type == "poisson":
            increments = np.clip(mu * dt, 1e-12, None)
        else:
            increments = np.clip(mu, 1e-9, 1.0 - 1e-9)
        # Return cumulative expected intensity sampled at spike/event bins.
        cumsum = np.cumsum(increments)
        spike_bins = np.where(y > 0.0)[0]
        if spike_bins.size == 0:
            return np.array([], dtype=float)
        return cumsum[spike_bins]

    @staticmethod
    def compute_ks_stats(transformed_events: np.ndarray) -> dict[str, float]:
        """Compute one-sample KS statistic against Uniform(0, 1) after rescaling."""

        z = np.asarray(transformed_events, dtype=float)
        if z.size == 0:
            return {"d_stat": 0.0, "n_events": 0.0}
        z = np.sort(z / max(float(np.max(z)), 1e-12))
        n = z.size
        ecdf = np.arange(1, n + 1, dtype=float) / float(n)
        d_plus = np.max(ecdf - z)
        d_minus = np.max(z - np.arange(0, n, dtype=float) / float(n))
        return {"d_stat": float(max(d_plus, d_minus)), "n_events": float(n)}

    @staticmethod
    def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Benjamini-Hochberg FDR control for one-dimensional p-value arrays."""

        p = np.asarray(p_values, dtype=float).ravel()
        if p.size == 0:
            return np.array([], dtype=bool)
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("p_values must be in [0, 1]")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        order = np.argsort(p)
        ranked = p[order]
        threshold = alpha * (np.arange(1, p.size + 1) / p.size)
        passing = np.where(ranked <= threshold)[0]
        mask = np.zeros_like(p, dtype=bool)
        if passing.size > 0:
            cutoff = ranked[int(np.max(passing))]
            mask = p <= cutoff
        return mask
