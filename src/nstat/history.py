"""Spike-history basis construction.

The basis matrix approximates the effect of past spikes on current intensity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class HistoryBasis:
    """Piecewise-constant history basis.

    Parameters
    ----------
    bin_edges_s:
        Increasing edges (seconds) defining history windows.
        Example: [0.0, 0.01, 0.05, 0.1].
    """

    bin_edges_s: np.ndarray
    min_time_s: float | None = None
    max_time_s: float | None = None

    def __post_init__(self) -> None:
        self.bin_edges_s = np.sort(np.asarray(self.bin_edges_s, dtype=float).reshape(-1))
        if self.bin_edges_s.size < 2:
            raise ValueError("bin_edges_s must be 1D with at least two elements")
        if not np.all(np.isfinite(self.bin_edges_s)):
            raise ValueError("bin_edges_s must be finite")
        if self.min_time_s is not None:
            self.min_time_s = float(self.min_time_s)
        if self.max_time_s is not None:
            self.max_time_s = float(self.max_time_s)

    @property
    def n_bins(self) -> int:
        return int(self.bin_edges_s.size - 1)

    @property
    def windowTimes(self) -> np.ndarray:
        """MATLAB-style alias for history window edges."""

        return self.bin_edges_s

    @windowTimes.setter
    def windowTimes(self, values: np.ndarray) -> None:
        edges = np.sort(np.asarray(values, dtype=float).reshape(-1))
        if edges.size < 2:
            raise ValueError("windowTimes must contain at least two entries")
        self.bin_edges_s = edges

    @property
    def minTime(self) -> float | None:
        """MATLAB-style alias for minimum retained time."""

        return self.min_time_s

    @minTime.setter
    def minTime(self, value: float | None) -> None:
        self.min_time_s = None if value is None else float(value)

    @property
    def maxTime(self) -> float | None:
        """MATLAB-style alias for maximum retained time."""

        return self.max_time_s

    @maxTime.setter
    def maxTime(self, value: float | None) -> None:
        self.max_time_s = None if value is None else float(value)

    def design_matrix(self, spike_times_s: np.ndarray, time_grid_s: np.ndarray) -> np.ndarray:
        """Build history design matrix for a binned point-process model.

        Notes
        -----
        For each time point and basis window, the entry counts spikes in
        the lag interval `(t - edge_hi, t - edge_lo]`. This mirrors common
        GLM history encoding while remaining explicit and testable.
        """

        spike_times_s = np.asarray(spike_times_s, dtype=float)
        time_grid_s = np.asarray(time_grid_s, dtype=float)

        mat = np.zeros((time_grid_s.size, self.n_bins), dtype=float)
        for i, t_now in enumerate(time_grid_s):
            lags = t_now - spike_times_s
            for j in range(self.n_bins):
                lo = self.bin_edges_s[j]
                hi = self.bin_edges_s[j + 1]
                mat[i, j] = float(np.sum((lags > lo) & (lags <= hi)))
        return mat

    def to_structure(self) -> dict[str, Any]:
        """Serialize using MATLAB field conventions."""

        return {
            "windowTimes": self.bin_edges_s.copy(),
            "minTime": self.min_time_s,
            "maxTime": self.max_time_s,
        }

    @staticmethod
    def from_structure(payload: dict[str, Any]) -> "HistoryBasis":
        """Deserialize from MATLAB-style structure payload."""

        if "windowTimes" in payload:
            return HistoryBasis(
                bin_edges_s=np.asarray(payload["windowTimes"], dtype=float),
                min_time_s=payload.get("minTime"),
                max_time_s=payload.get("maxTime"),
            )
        # Backward-compatible path used by early clean-room snapshots.
        return HistoryBasis(
            bin_edges_s=np.asarray(payload["bin_edges_s"], dtype=float),
            min_time_s=payload.get("min_time_s"),
            max_time_s=payload.get("max_time_s"),
        )
