"""Signal and covariate data containers.

These classes provide explicit, typed wrappers around time-indexed arrays.
The implementation intentionally keeps validation strict because many
nSTAT algorithms rely on aligned, monotonic time grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ArrayLike = np.ndarray


@dataclass(slots=True)
class Signal:
    """Continuous signal sampled on a 1D time grid.

    Parameters
    ----------
    time:
        Strictly increasing time samples (seconds by convention).
    data:
        Signal values. Can be 1D (`n_time`) or 2D (`n_time`, `n_channels`).
    name:
        Human-readable signal name.
    units:
        Optional unit label.

    Notes
    -----
    MATLAB `SignalObj` supports broad plotting and metadata utilities.
    This base class keeps the core numerical contract minimal and explicit,
    then downstream classes layer workflow-specific behavior.
    """

    time: ArrayLike
    data: ArrayLike
    name: str = "signal"
    units: str | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.data = np.asarray(self.data, dtype=float)

        if self.time.ndim != 1:
            raise ValueError("time must be a 1D array")
        if self.time.size < 2:
            raise ValueError("time must contain at least two samples")

        dtime = np.diff(self.time)
        if np.any(dtime <= 0.0):
            raise ValueError("time must be strictly increasing")

        if self.data.ndim == 1:
            if self.data.shape[0] != self.time.size:
                raise ValueError("1D data length must match time length")
        elif self.data.ndim == 2:
            if self.data.shape[0] != self.time.size:
                raise ValueError("2D data first dimension must match time length")
        else:
            raise ValueError("data must be 1D or 2D")

    @property
    def n_samples(self) -> int:
        """Number of time samples."""

        return int(self.time.size)

    @property
    def n_channels(self) -> int:
        """Number of channels (1 for a vector signal)."""

        if self.data.ndim == 1:
            return 1
        return int(self.data.shape[1])

    @property
    def sample_rate_hz(self) -> float:
        """Estimated sample rate in Hertz using median time delta."""

        dt = float(np.median(np.diff(self.time)))
        return 1.0 / dt

    @property
    def duration_s(self) -> float:
        """Signal duration in seconds."""

        return float(self.time[-1] - self.time[0])

    def copy(self) -> "Signal":
        """Return a deep copy."""

        return Signal(time=self.time.copy(), data=self.data.copy(), name=self.name, units=self.units)


@dataclass(slots=True)
class Covariate(Signal):
    """Named design covariate with optional per-column labels.

    Parameters
    ----------
    labels:
        Optional labels for each covariate column. For vector covariates,
        one label is expected.

    Notes
    -----
    This class is intentionally lightweight; higher-level composition and
    selection live in :mod:`nstat.trial`.
    """

    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        Signal.__post_init__(self)

        if not self.labels:
            if self.n_channels == 1:
                self.labels = [self.name]
            else:
                self.labels = [f"{self.name}_{i}" for i in range(self.n_channels)]

        if len(self.labels) != self.n_channels:
            raise ValueError("labels length must match number of channels")
