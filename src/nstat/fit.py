"""Fit result and summary objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cif import CIFModel


@dataclass(slots=True)
class FitResult:
    """Result container for one fitted model."""

    coefficients: np.ndarray
    intercept: float
    fit_type: str
    log_likelihood: float
    n_samples: int
    n_parameters: int
    parameter_labels: list[str] = field(default_factory=list)

    def as_cif_model(self) -> CIFModel:
        """Return a :class:`nstat.cif.CIFModel` view of this fitted model."""

        return CIFModel(coefficients=self.coefficients.copy(), intercept=self.intercept, link=self.fit_type)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean response from feature matrix ``X``."""

        return self.as_cif_model().evaluate(X)

    def aic(self) -> float:
        """Akaike information criterion."""

        return float(2 * self.n_parameters - 2 * self.log_likelihood)

    def bic(self) -> float:
        """Bayesian information criterion."""

        return float(np.log(max(self.n_samples, 1)) * self.n_parameters - 2 * self.log_likelihood)

    def compute_val_lambda(self, X: np.ndarray) -> np.ndarray:
        """MATLAB-style alias for evaluating predicted intensity/probability."""

        return self.predict(X)

    def get_coeffs(self) -> np.ndarray:
        """Return non-intercept coefficients."""

        return self.coefficients.copy()

    def get_coeff_index(self, label: str) -> int:
        """Return index of a parameter label."""

        if not self.parameter_labels:
            raise ValueError("parameter_labels are not populated")
        try:
            return self.parameter_labels.index(label)
        except ValueError as exc:
            raise KeyError(f"label '{label}' not present") from exc

    def get_param(self, key: str) -> float | np.ndarray | str | int:
        """Access a parameter/statistic by canonical key."""

        lookup: dict[str, float | np.ndarray | str | int] = {
            "intercept": self.intercept,
            "coefficients": self.coefficients.copy(),
            "fit_type": self.fit_type,
            "log_likelihood": self.log_likelihood,
            "n_samples": self.n_samples,
            "n_parameters": self.n_parameters,
            "aic": self.aic(),
            "bic": self.bic(),
        }
        if key not in lookup:
            raise KeyError(f"unknown key: {key}")
        return lookup[key]

    def get_unique_labels(self) -> list[str]:
        """Return unique parameter labels preserving order."""

        seen: set[str] = set()
        ordered: list[str] = []
        for label in self.parameter_labels:
            if label in seen:
                continue
            seen.add(label)
            ordered.append(label)
        return ordered


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

    def get_diff_aic(self) -> np.ndarray:
        """Return AIC differences relative to best model."""

        aic = np.array([result.aic() for result in self.results], dtype=float)
        return aic - np.min(aic)

    def get_diff_bic(self) -> np.ndarray:
        """Return BIC differences relative to best model."""

        bic = np.array([result.bic() for result in self.results], dtype=float)
        return bic - np.min(bic)

    def get_diff_log_likelihood(self) -> np.ndarray:
        """Return log-likelihood differences relative to best model."""

        ll = np.array([result.log_likelihood for result in self.results], dtype=float)
        return np.max(ll) - ll

    def compute_diff_mat(self, metric: str = "aic") -> np.ndarray:
        """Compute pairwise absolute difference matrix for selected metric."""

        metric_map = {
            "aic": np.array([result.aic() for result in self.results], dtype=float),
            "bic": np.array([result.bic() for result in self.results], dtype=float),
            "log_likelihood": np.array([result.log_likelihood for result in self.results], dtype=float),
        }
        if metric not in metric_map:
            raise ValueError("metric must be one of {'aic', 'bic', 'log_likelihood'}")
        values = metric_map[metric]
        return np.abs(values[:, None] - values[None, :])
