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

    def plot(self, *_, handle=None, colorString: str | None = None, **__):
        """Plot event markers on one or more axes.

        Parameters
        ----------
        handle : Axes or list[Axes], optional
            Axes to plot into (default: current axes).
        colorString : str, optional
            Override line colour for event lines (default: ``'r'``).
            Matches Matlab ``Events.plot`` ``colorString`` parameter.
        """
        color = colorString if colorString is not None else "r"
        if handle is None:
            handles = [plt.gca()]
        elif isinstance(handle, Sequence) and not hasattr(handle, "plot"):
            handles = list(handle)
        else:
            handles = [handle]

        last_ax = None
        for ax in handles:
            last_ax = ax
            v = ax.axis()
            if self.eventTimes.size:
                times = np.vstack([self.eventTimes, self.eventTimes])
                y = np.vstack(
                    [
                        np.full(self.eventTimes.shape, float(v[2]), dtype=float),
                        np.full(self.eventTimes.shape, float(v[3]), dtype=float),
                    ]
                )
                ax.plot(times, y, color, linewidth=4)
                for event_time, label in zip(self.eventTimes, self.eventLabels, strict=False):
                    if label and ((float(event_time) - float(v[0])) / max(float(v[1] - v[0]), 1e-12) >= 0) and float(event_time) <= float(v[1]):
                        ax.text(
                            (float(event_time) - float(v[0])) / max(float(v[1] - v[0]), 1e-12) - 0.02,
                            1.03,
                            label,
                            rotation=0,
                            fontsize=10,
                            color=[0, 0, 0],
                            transform=ax.transAxes,
                        )
        return last_ax


__all__ = ["Events"]
