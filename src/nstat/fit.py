"""Fit result and summary objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FitResult:
    """Result container for one fitted model."""

    coefficients: np.ndarray
    intercept: float
    fit_type: str
    log_likelihood: float
    n_samples: int
    n_parameters: int

    def aic(self) -> float:
        """Akaike information criterion."""

        return float(2 * self.n_parameters - 2 * self.log_likelihood)

    def bic(self) -> float:
        """Bayesian information criterion."""

        return float(np.log(max(self.n_samples, 1)) * self.n_parameters - 2 * self.log_likelihood)


@dataclass(slots=True)
class FitSummary:
    """Summary statistics across multiple fitted models."""

    results: list[FitResult]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("FitSummary requires at least one FitResult")

    def best_by_aic(self) -> FitResult:
        return min(self.results, key=lambda r: r.aic())

    def best_by_bic(self) -> FitResult:
        return min(self.results, key=lambda r: r.bic())
