"""Discrete event-marker container for raster and signal overlays.

This module mirrors MATLAB ``Events.m``.  An :class:`Events` instance
holds a vector of event times (in **seconds**) together with optional
per-event text labels and a colour code, and knows how to overlay itself
onto Matplotlib axes.  It is the third optional component (alongside a
:class:`~nstat.trial.SpikeTrainCollection` and a
:class:`~nstat.trial.CovariateCollection`) of a
:class:`~nstat.trial.Trial`.

Exported symbol
---------------
- :class:`Events` — event-marker container (Matlab ``Events``).
"""
from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


# MATLAB's single-letter colour codes use saturated primaries; matplotlib's
# defaults are darker.  Translate so rendered colours match MATLAB exactly
# (e.g. ``'g'`` → ``[0,1,0]`` rather than matplotlib's ``[0,0.5,0]``).
_MATLAB_COLOR_LOOKUP: dict[str, tuple[float, float, float]] = {
    "r": (1.0, 0.0, 0.0),
    "g": (0.0, 1.0, 0.0),
    "b": (0.0, 0.0, 1.0),
    "c": (0.0, 1.0, 1.0),
    "m": (1.0, 0.0, 1.0),
    "y": (1.0, 1.0, 0.0),
    "k": (0.0, 0.0, 0.0),
    "w": (1.0, 1.0, 1.0),
}


class Events:
    """Experimental event markers for highlighting epochs in figures.

    Events represent times of importance during an experiment (e.g.
    stimulus onset, trial boundaries) that are overlaid on raster or
    signal plots.

    Parameters
    ----------
    eventTimes : array_like
        Vector of event times (seconds).
    eventLabels : sequence of str or None
        Labels for each event.  Must match the length of *eventTimes*
        when provided.
    eventColor : str, default ``'r'``
        Colour string for the event lines (Matlab-style colour codes).
    """

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

    def plot(self, *_, handle=None, colorString: str | None = None, **__):
        """Plot event markers on one or more axes.

        Parameters
        ----------
        handle : Axes or list[Axes], optional
            Axes to plot into (default: current axes).
        colorString : str, optional
            Override line colour for event lines.  When ``None`` (the
            default), falls back to ``self.eventColor`` set at
            construction.  Matches MATLAB ``Events.m:87`` after the
            upstream fix from hardcoded ``'r'`` to ``EObj.eventColor``
            (see AUDIT_REPORT M17).  Gold fixture
            ``events_exactness.mat`` was regenerated 2026-06-11 against
            this corrected MATLAB behavior.

        Notes
        -----
        MATLAB single-letter colour codes (``'g'``, ``'b'``, ...) differ
        from matplotlib's defaults: MATLAB's ``'g'`` is RGB ``[0,1,0]``
        while matplotlib's ``'g'`` is the darker ``[0,0.5,0]``.  Single-
        letter codes are translated to the MATLAB convention so the
        rendered colour matches MATLAB exactly.
        """
        color = colorString if colorString is not None else self.eventColor
        color = _MATLAB_COLOR_LOOKUP.get(color, color)
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
                ax.plot(times, y, color=color, linewidth=4)
                for event_time, label in zip(self.eventTimes, self.eventLabels, strict=False):
                    if label and ((float(event_time) - float(v[0])) / max(float(v[1] - v[0]), 1e-12) >= 0) and float(event_time) <= float(v[1]):
                        # Match MATLAB Events.m:97 intent — label sits in data-x
                        # above its event line. Using get_xaxis_transform keeps
                        # x in data coordinates (tracks the line even with tight
                        # xlim) and y in axes coordinates (just above the axis),
                        # ha='center' replaces MATLAB's -0.02 normalized nudge.
                        ax.text(
                            float(event_time),
                            1.02,
                            label,
                            rotation=0,
                            fontsize=10,
                            color=[0, 0, 0],
                            ha="center",
                            va="bottom",
                            transform=ax.get_xaxis_transform(),
                        )
        return last_ax

    def toStructure(self) -> dict[str, Any]:
        """Serialise the Events to a plain dictionary."""
        return {
            "eventTimes": self.eventTimes.tolist(),
            "eventLabels": list(self.eventLabels),
            "eventColor": self.eventColor,
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any] | None) -> "Events" | None:
        """Reconstruct Events from a dictionary (inverse of :meth:`toStructure`)."""
        if structure is None:
            return None
        event_times = structure.get("eventTimes", structure.get("event_times", []))
        event_labels = structure.get("eventLabels", structure.get("labels"))
        event_color = structure.get("eventColor", "r")
        return Events(event_times, event_labels, event_color)


__all__ = ["Events"]
