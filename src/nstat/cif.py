"""Conditional intensity function models and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CIFModel:
    """Generalized linear conditional intensity model.

    Parameters
    ----------
    coefficients:
        Model coefficients, one per feature.
    intercept:
        Scalar intercept term.
    link:
        `poisson` (log link) or `binomial` (logit link).
    """

    coefficients: np.ndarray
    intercept: float = 0.0
    link: str = "poisson"

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if self.coefficients.ndim != 1:
            raise ValueError("coefficients must be 1D")
        if self.link not in {"poisson", "binomial"}:
            raise ValueError("link must be 'poisson' or 'binomial'")

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate point-wise intensity or probability."""

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.coefficients.size:
            raise ValueError("X feature dimension mismatch")

        eta = self.intercept + X @ self.coefficients
        if self.link == "poisson":
            return np.exp(eta)
        return 1.0 / (1.0 + np.exp(-eta))

    def simulate_by_thinning(self, time: np.ndarray, X: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate spike times with per-bin Bernoulli approximation.

        Notes
        -----
        For computational simplicity and deterministic notebook execution,
        this implementation uses one Bernoulli draw per time step based on
        intensity*dt (Poisson) or direct probability (binomial).
        """

        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time must be a 1D grid with >=2 samples")

        rng = rng or np.random.default_rng()
        values = self.evaluate(X)
        dt = float(np.median(np.diff(time)))
        if self.link == "poisson":
            p = np.clip(values * dt, 0.0, 1.0)
        else:
            p = np.clip(values, 0.0, 1.0)
        draws = rng.random(time.size) < p
        return time[draws]
