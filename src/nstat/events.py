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
    color:
        Plot color token (MATLAB-compatible default: ``"r"``).
    """

    times: np.ndarray
    labels: list[str] = field(default_factory=list)
    color: str = "r"

    def __post_init__(self) -> None:
        # MATLAB accepts row/column vectors and preserves ordering.
        self.times = np.asarray(self.times, dtype=float).reshape(-1)

        if not self.labels:
            self.labels = ["" for _ in range(self.times.size)]
        if len(self.labels) != self.times.size:
            raise ValueError("Number of eventTimes must equal number of eventLabels")
        self.labels = [str(label) for label in self.labels]
        self.color = str(self.color)

    @property
    def eventTimes(self) -> np.ndarray:
        """MATLAB-style alias for event times."""

        return self.times

    @eventTimes.setter
    def eventTimes(self, values: np.ndarray) -> None:
        self.times = np.asarray(values, dtype=float).reshape(-1)

    @property
    def eventLabels(self) -> list[str]:
        """MATLAB-style alias for event labels."""

        return self.labels

    @eventLabels.setter
    def eventLabels(self, values: list[str]) -> None:
        self.labels = [str(label) for label in values]

    @property
    def eventColor(self) -> str:
        """MATLAB-style alias for plot color."""

        return self.color

    @eventColor.setter
    def eventColor(self, value: str) -> None:
        self.color = str(value)

    def subset(self, start_s: float, end_s: float) -> "Events":
        """Return events within inclusive time interval."""

        mask = (self.times >= start_s) & (self.times <= end_s)
        indices = np.where(mask)[0]
        return Events(
            times=self.times[mask],
            labels=[self.labels[i] for i in indices],
            color=self.color,
        )

    def to_structure(self) -> dict[str, object]:
        """Serialize with MATLAB field names."""

        return {
            "eventTimes": self.times.copy(),
            "eventLabels": list(self.labels),
            "eventColor": self.color,
        }

    @staticmethod
    def from_structure(payload: dict[str, object]) -> "Events":
        """Deserialize from MATLAB-style structure payload."""

        if not payload:
            raise ValueError("payload must be non-empty")
        return Events(
            times=np.asarray(payload["eventTimes"], dtype=float),
            labels=[str(label) for label in payload["eventLabels"]],
            color=str(payload["eventColor"]),
        )
