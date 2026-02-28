"""Spike-history basis construction.

The basis matrix approximates the effect of past spikes on current intensity.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        self.bin_edges_s = np.asarray(self.bin_edges_s, dtype=float)
        if self.bin_edges_s.ndim != 1 or self.bin_edges_s.size < 2:
            raise ValueError("bin_edges_s must be 1D with at least two elements")
        if np.any(np.diff(self.bin_edges_s) <= 0.0):
            raise ValueError("bin_edges_s must be strictly increasing")

    @property
    def n_bins(self) -> int:
        return int(self.bin_edges_s.size - 1)

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
