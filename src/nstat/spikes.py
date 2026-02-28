"""Spike train classes.

These classes mirror MATLAB `nspikeTrain`/`nstColl` responsibilities while
providing explicit NumPy-based interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(slots=True)
class SpikeTrain:
    """Single-neuron spike-time sequence."""

    spike_times: np.ndarray
    t_start: float = 0.0
    t_end: float | None = None
    name: str = "unit"

    def __post_init__(self) -> None:
        self.spike_times = np.asarray(self.spike_times, dtype=float)
        if self.spike_times.ndim != 1:
            raise ValueError("spike_times must be 1D")
        if np.any(np.diff(self.spike_times) < 0.0):
            raise ValueError("spike_times must be sorted")

        if self.t_end is None:
            self.t_end = float(self.spike_times[-1]) if self.spike_times.size else self.t_start
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")

        in_bounds = (self.spike_times >= self.t_start) & (self.spike_times <= self.t_end)
        if not np.all(in_bounds):
            raise ValueError("all spike times must be inside [t_start, t_end]")

    def duration_s(self) -> float:
        if self.t_end is None:
            raise RuntimeError("SpikeTrain internal state invalid: t_end is None")
        return float(self.t_end - self.t_start)

    def firing_rate_hz(self) -> float:
        dur = self.duration_s()
        if dur <= 0.0:
            return 0.0
        return float(self.spike_times.size / dur)

    def bin_counts(self, bin_size_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return bin centers and integer spike-count vector."""

        if bin_size_s <= 0.0:
            raise ValueError("bin_size_s must be positive")
        if self.t_end is None:
            raise RuntimeError("SpikeTrain internal state invalid: t_end is None")
        edges = np.arange(self.t_start, self.t_end + bin_size_s, bin_size_s)
        counts, _ = np.histogram(self.spike_times, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts.astype(float)

    def binarize(self, bin_size_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return bin centers and binary spike-presence vector."""

        centers, counts = self.bin_counts(bin_size_s=bin_size_s)
        return centers, (counts > 0.0).astype(float)


@dataclass(slots=True)
class SpikeTrainCollection:
    """Collection of spike trains."""

    trains: list[SpikeTrain] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trains:
            raise ValueError("SpikeTrainCollection requires at least one train")

    @property
    def n_units(self) -> int:
        return len(self.trains)

    def to_binned_matrix(
        self, bin_size_s: float, mode: Literal["binary", "count"] = "binary"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert collection to `(n_units, n_bins)` matrix.

        Parameters
        ----------
        bin_size_s:
            Width of each time bin in seconds.
        mode:
            ``"binary"`` for per-bin event indicators or ``"count"`` for
            integer spike counts per bin.
        """

        if bin_size_s <= 0.0:
            raise ValueError("bin_size_s must be positive")
        if mode not in {"binary", "count"}:
            raise ValueError("mode must be 'binary' or 'count'")

        ref_t_start = min(train.t_start for train in self.trains)
        ref_t_end = max(train.t_end if train.t_end is not None else train.t_start for train in self.trains)
        edges = np.arange(ref_t_start, ref_t_end + bin_size_s, bin_size_s)
        centers = 0.5 * (edges[:-1] + edges[1:])

        mat = np.zeros((self.n_units, centers.size), dtype=float)
        for i, train in enumerate(self.trains):
            counts, _ = np.histogram(train.spike_times, bins=edges)
            if mode == "binary":
                mat[i, :] = (counts > 0).astype(float)
            else:
                mat[i, :] = counts.astype(float)
        return centers, mat
