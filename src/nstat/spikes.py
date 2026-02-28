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

    def copy(self) -> "SpikeTrain":
        """Return a deep copy of the spike train."""

        return SpikeTrain(
            spike_times=self.spike_times.copy(),
            t_start=float(self.t_start),
            t_end=float(self.t_end) if self.t_end is not None else None,
            name=self.name,
        )

    def firing_rate_hz(self) -> float:
        dur = self.duration_s()
        if dur <= 0.0:
            return 0.0
        return float(self.spike_times.size / dur)

    def get_spike_times(self) -> np.ndarray:
        """Return spike times (MATLAB-style helper)."""

        return self.spike_times.copy()

    def shift_time(self, offset_s: float) -> "SpikeTrain":
        """Shift spike times and support interval by a constant offset."""

        self.spike_times = self.spike_times + float(offset_s)
        self.t_start = float(self.t_start + offset_s)
        if self.t_end is not None:
            self.t_end = float(self.t_end + offset_s)
        return self

    def set_min_time(self, t_min: float) -> "SpikeTrain":
        """Set lower support bound and drop spikes before that bound."""

        self.t_start = float(t_min)
        self.spike_times = self.spike_times[self.spike_times >= self.t_start]
        if self.t_end is not None and self.t_end < self.t_start:
            self.t_end = self.t_start
        return self

    def set_max_time(self, t_max: float) -> "SpikeTrain":
        """Set upper support bound and drop spikes after that bound."""

        if t_max < self.t_start:
            raise ValueError("t_max must be >= t_start")
        self.t_end = float(t_max)
        self.spike_times = self.spike_times[self.spike_times <= self.t_end]
        return self

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

    def copy(self) -> "SpikeTrainCollection":
        """Return a deep copy of the collection."""

        return SpikeTrainCollection([train.copy() for train in self.trains])

    def merge(self, other: "SpikeTrainCollection") -> "SpikeTrainCollection":
        """Return a new collection containing trains from both inputs."""

        return SpikeTrainCollection(self.trains + [train.copy() for train in other.trains])

    def add_to_coll(self, train: SpikeTrain) -> "SpikeTrainCollection":
        """Append a spike train in-place and return self."""

        self.trains.append(train)
        return self

    def add_single_spike_to_coll(
        self, unit_index: int, spike_time_s: float, sort_times: bool = True
    ) -> "SpikeTrainCollection":
        """Add one spike event to a selected unit."""

        if unit_index < 0 or unit_index >= self.n_units:
            raise IndexError("unit_index out of range")
        train = self.trains[unit_index]
        train.spike_times = np.append(train.spike_times, float(spike_time_s))
        if sort_times:
            train.spike_times = np.sort(train.spike_times)
        if train.t_end is not None:
            train.t_end = max(train.t_end, float(spike_time_s))
        return self

    def get_first_spike_time(self) -> float:
        """Minimum spike time across all units."""

        first = [train.spike_times[0] for train in self.trains if train.spike_times.size > 0]
        if not first:
            return float(min(train.t_start for train in self.trains))
        return float(min(first))

    def get_last_spike_time(self) -> float:
        """Maximum spike time across all units."""

        last = [train.spike_times[-1] for train in self.trains if train.spike_times.size > 0]
        if not last:
            return float(max(train.t_end if train.t_end is not None else train.t_start for train in self.trains))
        return float(max(last))

    def get_spike_times(self) -> list[np.ndarray]:
        """Return all spike-time vectors as copies."""

        return [train.spike_times.copy() for train in self.trains]

    def get_nst(self, index: int) -> SpikeTrain:
        """Return train by zero-based index."""

        if index < 0 or index >= self.n_units:
            raise IndexError("index out of range")
        return self.trains[index]

    def get_nst_names(self) -> list[str]:
        """Return all train names."""

        return [train.name for train in self.trains]

    def get_unique_nst_names(self) -> list[str]:
        """Return unique train names preserving first occurrence order."""

        seen: set[str] = set()
        ordered: list[str] = []
        for name in self.get_nst_names():
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    def get_nst_indices_from_name(self, name: str) -> list[int]:
        """Return all indices matching a train name."""

        return [i for i, train in enumerate(self.trains) if train.name == name]

    def get_nst_name_from_ind(self, index: int) -> str:
        """Return train name from index."""

        return self.get_nst(index).name

    def get_nst_from_name(self, name: str, first_only: bool = True) -> SpikeTrain | list[SpikeTrain]:
        """Return train(s) matching name."""

        matches = [train for train in self.trains if train.name == name]
        if not matches:
            raise KeyError(f"no spike train named '{name}'")
        if first_only:
            return matches[0]
        return matches

    def shift_time(self, offset_s: float) -> "SpikeTrainCollection":
        """Shift all trains by a constant temporal offset."""

        for train in self.trains:
            train.shift_time(offset_s)
        return self

    def set_min_time(self, t_min: float) -> "SpikeTrainCollection":
        """Apply lower bound to all trains."""

        for train in self.trains:
            train.set_min_time(t_min)
        return self

    def set_max_time(self, t_max: float) -> "SpikeTrainCollection":
        """Apply upper bound to all trains."""

        for train in self.trains:
            train.set_max_time(t_max)
        return self

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

    def data_to_matrix(self, bin_size_s: float, mode: Literal["binary", "count"] = "binary") -> np.ndarray:
        """Return only the binned matrix."""

        _, mat = self.to_binned_matrix(bin_size_s=bin_size_s, mode=mode)
        return mat

    def to_spike_train(self, name: str = "merged") -> SpikeTrain:
        """Merge all spikes into one spike train."""

        merged = np.concatenate([train.spike_times for train in self.trains])
        merged.sort()
        t_start = min(train.t_start for train in self.trains)
        t_end = max(train.t_end if train.t_end is not None else train.t_start for train in self.trains)
        return SpikeTrain(spike_times=merged, t_start=float(t_start), t_end=float(t_end), name=name)
