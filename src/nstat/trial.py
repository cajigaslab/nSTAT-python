"""Trial-level composition objects.

The trial module ties spike observations and covariates together for fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .signal import Covariate
from .spikes import SpikeTrainCollection


@dataclass(slots=True)
class CovariateCollection:
    """Grouped covariates with aligned time grids."""

    covariates: list[Covariate]

    def __post_init__(self) -> None:
        if not self.covariates:
            raise ValueError("CovariateCollection requires at least one covariate")

        # Enforce shared time axis because downstream design matrix assembly
        # assumes row-wise temporal alignment.
        ref = self.covariates[0].time
        for cov in self.covariates[1:]:
            if cov.time.shape != ref.shape or not np.allclose(cov.time, ref):
                raise ValueError("all covariates must share the same time grid")

    @property
    def time(self) -> np.ndarray:
        return self.covariates[0].time

    def design_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return `(X, labels)` where `X` has shape `(n_time, n_features)`."""

        blocks: list[np.ndarray] = []
        labels: list[str] = []
        for cov in self.covariates:
            if cov.data.ndim == 1:
                blocks.append(cov.data[:, None])
            else:
                blocks.append(cov.data)
            labels.extend(cov.labels)
        return np.hstack(blocks), labels

    def copy(self) -> "CovariateCollection":
        """Return a deep copy preserving covariate order."""

        copied = [
            Covariate(
                time=cov.time.copy(),
                data=cov.data.copy(),
                name=cov.name,
                units=cov.units,
                labels=list(cov.labels),
            )
            for cov in self.covariates
        ]
        return CovariateCollection(copied)

    def add_to_coll(self, covariate: Covariate) -> "CovariateCollection":
        """Append one covariate after validating time-grid alignment."""

        ref = self.time
        if covariate.time.shape != ref.shape or not np.allclose(covariate.time, ref):
            raise ValueError("added covariate must share the existing time grid")
        self.covariates.append(covariate)
        return self

    def get_cov(self, selector: int | str) -> Covariate:
        """Return covariate by index or by name."""

        if isinstance(selector, int):
            if selector < 0 or selector >= len(self.covariates):
                raise IndexError("covariate index out of range")
            return self.covariates[selector]
        for cov in self.covariates:
            if cov.name == selector:
                return cov
        raise KeyError(f"covariate '{selector}' not found")

    def get_cov_indices_from_names(self, names: list[str]) -> list[int]:
        """Return indices for a list of covariate names."""

        name_to_index = {cov.name: i for i, cov in enumerate(self.covariates)}
        return [name_to_index[name] for name in names if name in name_to_index]

    def get_cov_ind_from_name(self, name: str) -> int:
        """Return first index matching covariate name."""

        indices = self.get_cov_indices_from_names([name])
        if not indices:
            raise KeyError(f"covariate '{name}' not found")
        return indices[0]

    def is_cov_present(self, name: str) -> bool:
        """Return whether a covariate name exists in the collection."""

        return any(cov.name == name for cov in self.covariates)

    def get_cov_dimension(self) -> int:
        """Total number of covariate columns across all entries."""

        return int(sum(cov.n_channels for cov in self.covariates))

    def get_all_cov_labels(self) -> list[str]:
        """Flatten all covariate labels into one list."""

        labels: list[str] = []
        for cov in self.covariates:
            labels.extend(cov.labels)
        return labels

    def n_act_covar(self) -> int:
        """MATLAB-style active covariate count helper."""

        return len(self.covariates)

    def num_act_cov(self) -> int:
        """Alias for `n_act_covar`."""

        return self.n_act_covar()

    def sum_dimensions(self) -> int:
        """Alias for total feature dimension."""

        return self.get_cov_dimension()

    def data_to_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Alias for design-matrix export."""

        return self.design_matrix()

    def data_to_matrix_from_names(self, names: list[str]) -> tuple[np.ndarray, list[str]]:
        """Build design matrix using selected covariate names."""

        indices = self.get_cov_indices_from_names(names)
        if not indices:
            raise ValueError("no covariates matched the provided names")
        selected = CovariateCollection([self.covariates[i] for i in indices])
        return selected.design_matrix()

    def data_to_matrix_from_sel(self, selectors: list[int]) -> tuple[np.ndarray, list[str]]:
        """Build design matrix using selected covariate indices."""

        if not selectors:
            raise ValueError("selectors cannot be empty")
        selected = CovariateCollection([self.covariates[i] for i in selectors])
        return selected.design_matrix()

    def find_min_time(self) -> float:
        return float(self.time[0])

    def find_max_time(self) -> float:
        return float(self.time[-1])

    def find_min_sample_rate(self) -> float:
        return float(min(cov.sample_rate_hz for cov in self.covariates))

    def find_max_sample_rate(self) -> float:
        return float(max(cov.sample_rate_hz for cov in self.covariates))


@dataclass(slots=True)
class TrialConfig:
    """Model specification for one fit configuration."""

    covariate_labels: list[str] = field(default_factory=list)
    sample_rate_hz: float = 1000.0
    fit_type: str = "poisson"
    name: str = "config"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.fit_type not in {"poisson", "binomial"}:
            raise ValueError("fit_type must be 'poisson' or 'binomial'")


@dataclass(slots=True)
class ConfigCollection:
    """Collection of trial configurations."""

    configs: list[TrialConfig]

    def __post_init__(self) -> None:
        if not self.configs:
            raise ValueError("ConfigCollection requires at least one TrialConfig")


@dataclass(slots=True)
class Trial:
    """Container binding spike observations to covariates."""

    spikes: SpikeTrainCollection
    covariates: CovariateCollection

    def aligned_binned_observation(
        self, bin_size_s: float, unit_index: int = 0, mode: Literal["binary", "count"] = "binary"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned `(time, y, X)` for one unit.

        `X` is covariate matrix sampled to nearest available covariate time
        index. ``mode`` controls whether `y` contains binary indicators or
        integer spike counts per bin.
        """

        t_bins, mat = self.spikes.to_binned_matrix(bin_size_s=bin_size_s, mode=mode)
        if unit_index < 0 or unit_index >= mat.shape[0]:
            raise IndexError("unit_index out of range")
        y = mat[unit_index]

        X_full, _ = self.covariates.design_matrix()
        t_cov = self.covariates.time
        idx = np.searchsorted(t_cov, t_bins, side="left")
        idx = np.clip(idx, 0, t_cov.size - 1)
        X = X_full[idx]
        return t_bins, y, X

    def get_design_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return full-trial design matrix and labels."""

        return self.covariates.design_matrix()

    def get_spike_vector(
        self, bin_size_s: float, unit_index: int = 0, mode: Literal["binary", "count"] = "binary"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return binned spike vector `(time, y)` for one unit."""

        t, y, _ = self.aligned_binned_observation(
            bin_size_s=bin_size_s, unit_index=unit_index, mode=mode
        )
        return t, y

    def get_cov(self, selector: int | str) -> Covariate:
        """Return trial covariate by index or name."""

        return self.covariates.get_cov(selector)

    def get_neuron(self, unit_index: int = 0) -> SpikeTrainCollection:
        """Return a single-neuron collection preserving collection API."""

        if unit_index < 0 or unit_index >= self.spikes.n_units:
            raise IndexError("unit_index out of range")
        return SpikeTrainCollection([self.spikes.get_nst(unit_index).copy()])

    def get_all_cov_labels(self) -> list[str]:
        return self.covariates.get_all_cov_labels()

    def find_min_time(self) -> float:
        return float(min(self.covariates.find_min_time(), min(train.t_start for train in self.spikes.trains)))

    def find_max_time(self) -> float:
        spike_max = max(train.t_end if train.t_end is not None else train.t_start for train in self.spikes.trains)
        return float(max(self.covariates.find_max_time(), spike_max))

    def find_min_sample_rate(self) -> float:
        return self.covariates.find_min_sample_rate()

    def find_max_sample_rate(self) -> float:
        return self.covariates.find_max_sample_rate()

    def is_sample_rate_consistent(self, rtol: float = 1.0e-6) -> bool:
        """Check whether covariates share effectively identical sample rates."""

        rates = np.array([cov.sample_rate_hz for cov in self.covariates.covariates], dtype=float)
        return bool(np.allclose(rates, rates[0], rtol=rtol))
