from __future__ import annotations

from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class Events:
    """MATLAB-style event container."""

    def __init__(self, eventTimes, eventLabels: Sequence[str] | None = None, eventColor: str = "r") -> None:
        times = np.asarray(eventTimes, dtype=float).reshape(-1)
        labels = [""] * int(times.size) if eventLabels is None else list(eventLabels)
        if len(labels) != int(times.size):
            raise ValueError("Number of eventTimes must match number of eventLabels")

        self.eventTimes = times
        self.eventLabels = labels
        self.eventColor = str(eventColor)

        # Legacy Python-side aliases kept for compatibility.
        self.event_times = self.eventTimes
        self.labels = self.eventLabels

    def toStructure(self) -> dict[str, Any]:
        return {
            "eventTimes": self.eventTimes.tolist(),
            "eventLabels": list(self.eventLabels),
            "eventColor": self.eventColor,
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any] | None) -> "Events" | None:
        if structure is None:
            return None
        event_times = structure.get("eventTimes", structure.get("event_times", []))
        event_labels = structure.get("eventLabels", structure.get("labels"))
        event_color = structure.get("eventColor", "r")
        return Events(event_times, event_labels, event_color)

    def plot(self, *_, handle=None, **__):
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 2.2))[1]
        ax.clear()
        if self.eventTimes.size:
            ax.vlines(self.eventTimes, 0.0, 1.0, color=self.eventColor, linewidth=1.5)
            for x, label in zip(self.eventTimes, self.eventLabels, strict=False):
                if label:
                    ax.text(float(x), 1.02, label, rotation=45, ha="left", va="bottom", fontsize=8)
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel("time [s]")
        ax.set_yticks([])
        ax.set_title("Events")
        return ax


__all__ = ["Events"]
