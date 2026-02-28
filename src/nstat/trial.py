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
