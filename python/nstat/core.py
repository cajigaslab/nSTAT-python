from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_1d_float(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return array


@dataclass(frozen=True)
class Covariate:
    """Time-aligned covariate signal."""

    time: np.ndarray
    values: np.ndarray
    name: str = "covariate"
    units: str = ""

    def __post_init__(self) -> None:
        time = _as_1d_float(self.time, "time")
        values = _as_1d_float(self.values, "values")
        if time.shape[0] != values.shape[0]:
            raise ValueError("time and values must have identical length.")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time must be strictly increasing.")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SpikeTrain:
    """Container for spike times in seconds."""

    times: np.ndarray
    t_start: float = 0.0
    t_stop: float | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if times.size:
            times = np.sort(times)
            if np.any(np.diff(times) < 0):
                raise ValueError("spike times must be monotonic.")
            t_start = float(min(self.t_start, times[0]))
            t_stop = float(times[-1] if self.t_stop is None else self.t_stop)
            if t_stop < t_start:
                raise ValueError("t_stop must be >= t_start.")
        else:
            t_start = float(self.t_start)
            t_stop = float(self.t_start if self.t_stop is None else self.t_stop)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "t_start", t_start)
        object.__setattr__(self, "t_stop", t_stop)

    @property
    def n_spikes(self) -> int:
        return int(self.times.shape[0])

    @property
    def duration(self) -> float:
        return float(self.t_stop - self.t_start)

    @property
    def firing_rate_hz(self) -> float:
        duration = self.duration
        if duration <= 0:
            return 0.0
        return float(self.n_spikes / duration)

    def interspike_intervals(self) -> np.ndarray:
        if self.n_spikes < 2:
            return np.array([], dtype=float)
        return np.diff(self.times)

    def to_binned_counts(self, bin_edges: Sequence[float]) -> np.ndarray:
        edges = np.asarray(bin_edges, dtype=float).reshape(-1)
        if edges.size < 2:
            raise ValueError("bin_edges must contain at least 2 values.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing.")
        counts, _ = np.histogram(self.times, bins=edges)
        return counts.astype(float)

