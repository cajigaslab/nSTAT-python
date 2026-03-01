"""Fit result and summary objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

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

    def to_structure(self) -> dict[str, Any]:
        """Serialize this fit result to a MATLAB-like plain structure."""

        return {
            "coefficients": self.coefficients.copy(),
            "intercept": float(self.intercept),
            "fit_type": str(self.fit_type),
            "log_likelihood": float(self.log_likelihood),
            "n_samples": int(self.n_samples),
            "n_parameters": int(self.n_parameters),
            "parameter_labels": list(self.parameter_labels),
        }

    @classmethod
    def from_structure(cls, payload: dict[str, Any]) -> "FitResult":
        """Build a :class:`FitResult` from a serialized structure."""

        return cls(
            coefficients=np.asarray(payload["coefficients"], dtype=float).reshape(-1),
            intercept=float(payload["intercept"]),
            fit_type=str(payload["fit_type"]),
            log_likelihood=float(payload["log_likelihood"]),
            n_samples=int(payload["n_samples"]),
            n_parameters=int(payload["n_parameters"]),
            parameter_labels=[str(v) for v in payload.get("parameter_labels", [])],
        )

    @staticmethod
    def cell_array_to_structure(results: list["FitResult"]) -> list[dict[str, Any]]:
        """Serialize a list of fit results into structure list form."""

        return [result.to_structure() for result in results]


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

    def get_unique_labels(self) -> list[str]:
        """Return union of parameter labels across all fit results."""

        labels: list[str] = []
        seen: set[str] = set()
        for result in self.results:
            source = result.parameter_labels or [f"coef_{i+1}" for i in range(result.coefficients.size)]
            for label in source:
                if label in seen:
                    continue
                seen.add(label)
                labels.append(label)
        return labels

    def get_coeffs(self, fit_num: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return coefficient and SE matrices aligned by unique labels."""

        _ = fit_num  # Compatibility placeholder: fit index semantics are MATLAB-specific.
        labels = self.get_unique_labels()
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        coeff_mat = np.full((len(labels), len(self.results)), np.nan, dtype=float)
        se_mat = np.full_like(coeff_mat, np.nan)

        for col, result in enumerate(self.results):
            source_labels = result.parameter_labels or [f"coef_{i+1}" for i in range(result.coefficients.size)]
            for coeff, label in zip(result.coefficients, source_labels):
                row = label_to_idx[label]
                coeff_mat[row, col] = float(coeff)
                # Standard errors are not stored in the simplified Python FitResult.
                se_mat[row, col] = 0.0

        return coeff_mat, labels, se_mat

    def get_coeff_index(self, fit_num: int = 1, sort_by_epoch: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
        """Return coefficient row indices and epoch metadata for a fit index."""

        _ = sort_by_epoch  # Compatibility placeholder.
        coeff_mat, _labels, _se_mat = self.get_coeffs(fit_num=fit_num)
        col = max(0, min(len(self.results) - 1, int(fit_num) - 1))
        idx = np.where(np.isfinite(coeff_mat[:, col]))[0].astype(int)
        epoch_id = np.ones(idx.shape[0], dtype=int)
        num_epochs = 1
        return idx, epoch_id, num_epochs

    def bin_coeffs(
        self,
        min_val: float,
        max_val: float,
        bin_size: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Histogram coefficients and estimate percent-significant per bin."""

        if bin_size <= 0:
            raise ValueError("bin_size must be > 0")

        coeff_mat, _labels, _se = self.get_coeffs()
        values = coeff_mat[np.isfinite(coeff_mat)]
        if values.size == 0:
            edges = np.arange(min_val, max_val + bin_size, bin_size, dtype=float)
            return np.zeros(edges.size - 1, dtype=int), edges, np.zeros(edges.size - 1, dtype=float)

        edges = np.arange(min_val, max_val + bin_size, bin_size, dtype=float)
        counts, edges = np.histogram(values, bins=edges)
        percent_sig = np.zeros(counts.size, dtype=float)
        for i in range(counts.size):
            in_bin = (values >= edges[i]) & (values < edges[i + 1])
            if i == counts.size - 1:
                in_bin = (values >= edges[i]) & (values <= edges[i + 1])
            if not np.any(in_bin):
                percent_sig[i] = 0.0
            else:
                percent_sig[i] = float(np.mean(np.abs(values[in_bin]) > 0.0))

        return counts.astype(int), edges.astype(float), percent_sig

    def box_plot(
        self,
        X: np.ndarray | None = None,
        diff_index: int = 1,
    ) -> dict[str, np.ndarray]:
        """Return box-plot summary statistics for compatibility workflows."""

        if X is None:
            data = self.get_diff_aic()
        else:
            data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data[:, None]

        _ = diff_index  # MATLAB API compatibility placeholder.
        q1 = np.nanpercentile(data, 25, axis=0)
        median = np.nanpercentile(data, 50, axis=0)
        q3 = np.nanpercentile(data, 75, axis=0)
        return {
            "q1": q1.astype(float),
            "median": median.astype(float),
            "q3": q3.astype(float),
        }

    def to_structure(self) -> dict[str, Any]:
        """Serialize summary and nested fit results into plain structures."""

        return {"results": [result.to_structure() for result in self.results]}

    @classmethod
    def from_structure(cls, payload: dict[str, Any]) -> "FitSummary":
        """Deserialize a summary from structure form."""

        rows = payload.get("results", payload.get("fitResCell", []))
        return cls(results=[FitResult.from_structure(cast(dict[str, Any], row)) for row in rows])
