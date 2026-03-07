from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidenceInterval:
    time: np.ndarray
    bounds: np.ndarray
    color: str = "b"

    def __init__(self, time, bounds, color: str = "b") -> None:
        t = np.asarray(time, dtype=float).reshape(-1)
        b = np.asarray(bounds, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must have shape (n, 2)")
        if b.shape[0] != t.shape[0]:
            raise ValueError("bounds rows must match time length")
        self.time = t
        self.bounds = b
        self.color = color

    @property
    def lower(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def upper(self) -> np.ndarray:
        return self.bounds[:, 1]

    def setColor(self, color: str) -> None:
        self.color = str(color)

    def _coerce_signal_values(self, other) -> np.ndarray:
        if hasattr(other, "time") and hasattr(other, "data"):
            other_time = np.asarray(other.time, dtype=float).reshape(-1)
            if other_time.shape != self.time.shape or np.max(np.abs(other_time - self.time)) > 1e-9:
                raise ValueError("ConfidenceInterval operations require matching time grids")
            values = np.asarray(other.data, dtype=float)
            if values.ndim == 2:
                if values.shape[1] != 1:
                    raise ValueError("ConfidenceInterval arithmetic expects a scalar signal per operation")
                values = values[:, 0]
            return values.reshape(-1)
        values = np.asarray(other, dtype=float)
        if values.ndim == 0:
            return np.full(self.time.shape, float(values), dtype=float)
        return values.reshape(-1)

    def __add__(self, other):
        if isinstance(other, ConfidenceInterval):
            if other.time.shape != self.time.shape or np.max(np.abs(other.time - self.time)) > 1e-9:
                raise ValueError("ConfidenceInterval operations require matching time grids")
            bounds = np.column_stack([self.lower + other.lower, self.upper + other.upper])
            return ConfidenceInterval(self.time, bounds, self.color)
        offset = self._coerce_signal_values(other)
        return ConfidenceInterval(self.time, self.bounds + offset[:, None], self.color)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ConfidenceInterval):
            if other.time.shape != self.time.shape or np.max(np.abs(other.time - self.time)) > 1e-9:
                raise ValueError("ConfidenceInterval operations require matching time grids")
            bounds = np.column_stack([self.lower - other.upper, self.upper - other.lower])
            return ConfidenceInterval(self.time, bounds, self.color)
        offset = self._coerce_signal_values(other)
        return ConfidenceInterval(self.time, self.bounds - offset[:, None], self.color)

    def __rsub__(self, other):
        offset = self._coerce_signal_values(other)
        bounds = np.column_stack([offset - self.upper, offset - self.lower])
        return ConfidenceInterval(self.time, bounds, self.color)

    def __neg__(self):
        return ConfidenceInterval(self.time, np.column_stack([-self.upper, -self.lower]), self.color)

    def plot(self, color: str | None = None, ax=None):
        import matplotlib.pyplot as plt

        axis = plt.gca() if ax is None else ax
        return axis.fill_between(self.time, self.lower, self.upper, color=color or self.color, alpha=0.2)
