"""Confidence interval container for time-varying quantities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    color:
        MATLAB-style plotting color token.
    value:
        MATLAB-style confidence value metadata (defaults to ``level``).
    """

    time: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float = 0.95
    color: str = "b"
    value: float | np.ndarray | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.lower = np.asarray(self.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(self.upper, dtype=float).reshape(-1)

        if self.lower.shape != self.time.shape or self.upper.shape != self.time.shape:
            raise ValueError("lower and upper must match time shape")
        if np.any(self.lower > self.upper):
            raise ValueError("lower bound cannot exceed upper bound")
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must be in (0, 1)")
        self.color = str(self.color)
        if self.value is None:
            self.value = float(self.level)

    def set_color(self, color: str) -> "ConfidenceInterval":
        self.color = str(color)
        return self

    def set_value(self, value: float | np.ndarray) -> "ConfidenceInterval":
        # MATLAB ConfidenceInterval.setValue stores metadata; it does not
        # reshape or overwrite lower/upper bounds.
        if np.asarray(value).ndim == 0:
            self.value = float(np.asarray(value, dtype=float))
        else:
            self.value = np.asarray(value, dtype=float).copy()
        return self

    def width(self) -> np.ndarray:
        """Return point-wise interval width."""

        return self.upper - self.lower

    def contains(self, values: np.ndarray) -> np.ndarray:
        """Return boolean mask indicating whether values are inside the interval."""

        values = np.asarray(values, dtype=float)
        if values.shape != self.time.shape:
            raise ValueError("values shape must match time shape")
        return (values >= self.lower) & (values <= self.upper)

    def to_structure(self) -> dict[str, Any]:
        """Serialize with MATLAB-compatible and native fields."""

        values = np.column_stack([self.lower, self.upper])
        return {
            "time": self.time.copy(),
            "signals": {
                "values": values,
                "dimensions": np.array([values.shape[0], values.shape[1]], dtype=float),
            },
            "name": "ConfidenceInterval",
            "dimension": 2,
            "minTime": float(self.time.min()) if self.time.size else 0.0,
            "maxTime": float(self.time.max()) if self.time.size else 0.0,
            "xlabelval": "time",
            "xunits": "s",
            "yunits": "",
            "dataLabels": ["lower", "upper"],
            "dataMask": [],
            "sampleRate": float((self.time.size - 1) / (self.time[-1] - self.time[0])) if self.time.size > 1 and self.time[-1] != self.time[0] else 1.0,
            "plotProps": [],
            # Native convenience fields
            "lower": self.lower.copy(),
            "upper": self.upper.copy(),
            "level": float(self.level),
            "color": self.color,
            "value": self.value,
        }

    @staticmethod
    def from_structure(payload: dict[str, Any]) -> "ConfidenceInterval":
        """Deserialize from MATLAB-style or native payload."""

        if "signals" in payload:
            sig = payload["signals"]
            if isinstance(sig, dict):
                sig_values = np.asarray(sig["values"], dtype=float)
            elif hasattr(sig, "values"):
                sig_values = np.asarray(getattr(sig, "values"), dtype=float)
            else:
                arr = np.asarray(sig, dtype=object)
                if arr.size != 1:
                    raise ValueError("signals payload must be scalar struct-like")
                s0 = arr.reshape(-1)[0]
                if hasattr(s0, "values"):
                    sig_values = np.asarray(getattr(s0, "values"), dtype=float)
                elif isinstance(s0, dict):
                    sig_values = np.asarray(s0["values"], dtype=float)
                else:
                    raise ValueError("Unsupported signals payload")
            if sig_values.ndim != 2 or sig_values.shape[1] < 2:
                raise ValueError("signals.values must be a [N,2] array")
            lower = sig_values[:, 0]
            upper = sig_values[:, 1]
            return ConfidenceInterval(
                time=np.asarray(payload["time"], dtype=float),
                lower=lower,
                upper=upper,
                level=float(payload.get("level", 0.95)),
                color=str(payload.get("color", "b")),
                value=payload.get("value", payload.get("level", 0.95)),
            )
        return ConfidenceInterval(
            time=np.asarray(payload["time"], dtype=float),
            lower=np.asarray(payload["lower"], dtype=float),
            upper=np.asarray(payload["upper"], dtype=float),
            level=float(payload.get("level", 0.95)),
            color=str(payload.get("color", "b")),
            value=payload.get("value", payload.get("level", 0.95)),
        )
