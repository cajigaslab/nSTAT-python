"""Event marker utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Events:
    """Discrete event times with labels.

    Parameters
    ----------
    times:
        Event times in seconds.
    labels:
        Optional event labels; defaults to empty strings.
    """

    times: np.ndarray
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        if self.times.ndim != 1:
            raise ValueError("times must be 1D")
        if np.any(np.diff(self.times) < 0.0):
            raise ValueError("times must be non-decreasing")

        if not self.labels:
            self.labels = ["" for _ in range(self.times.size)]
        if len(self.labels) != self.times.size:
            raise ValueError("labels length must equal number of events")

    def subset(self, start_s: float, end_s: float) -> "Events":
        """Return events within inclusive time interval."""

        mask = (self.times >= start_s) & (self.times <= end_s)
        return Events(times=self.times[mask], labels=[self.labels[i] for i in np.where(mask)[0]])
