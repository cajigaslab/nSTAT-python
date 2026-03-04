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
        if spike_times_s.ndim != 1:
            spike_times_s = spike_times_s.reshape(-1)
        if time_grid_s.ndim != 1:
            time_grid_s = time_grid_s.reshape(-1)
        spike_times_s = np.sort(spike_times_s)

        mat = np.zeros((time_grid_s.size, self.n_bins), dtype=float)
        if spike_times_s.size == 0 or time_grid_s.size == 0:
            return mat

        # Equivalent to counting lags in (lo, hi], i.e., spikes in [t-hi, t-lo).
        for j in range(self.n_bins):
            lo = float(self.bin_edges_s[j])
            hi = float(self.bin_edges_s[j + 1])
            lower = time_grid_s - hi
            upper = time_grid_s - lo
            lo_idx = np.searchsorted(spike_times_s, lower, side="left")
            hi_idx = np.searchsorted(spike_times_s, upper, side="left")
            mat[:, j] = (hi_idx - lo_idx).astype(float)
        return mat
