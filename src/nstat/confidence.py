"""Confidence interval container for time-varying quantities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ConfidenceInterval:
    """Lower/upper confidence envelope over time.

    Parameters
    ----------
    time:
        Monotonic time grid.
    lower:
        Lower confidence bound values.
    upper:
        Upper confidence bound values.
    level:
        Confidence level in (0,1), defaults to 0.95.
    """

    time: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float = 0.95

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)

        if self.time.ndim != 1:
            raise ValueError("time must be 1D")
        if self.lower.shape != self.time.shape or self.upper.shape != self.time.shape:
            raise ValueError("lower and upper must match time shape")
        if np.any(self.lower > self.upper):
            raise ValueError("lower bound cannot exceed upper bound")
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must be in (0, 1)")

    def width(self) -> np.ndarray:
        """Return point-wise interval width."""

        return self.upper - self.lower

    def contains(self, values: np.ndarray) -> np.ndarray:
        """Return boolean mask indicating whether values are inside the interval."""

        values = np.asarray(values, dtype=float)
        if values.shape != self.time.shape:
            raise ValueError("values shape must match time shape")
        return (values >= self.lower) & (values <= self.upper)
