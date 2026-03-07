from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .core import Covariate, nspikeTrain


class History:
    """MATLAB-style spike-history basis described by window boundaries."""

    def __init__(self, windowTimes, minTime: float | None = None, maxTime: float | None = None, name: str = "History") -> None:
        times = np.asarray(windowTimes, dtype=float).reshape(-1)
        if times.size <= 1:
            raise ValueError("At least two times points must be specified to determine a window")
        if np.any(np.diff(times) <= 0):
            raise ValueError("windowTimes must be strictly increasing")

        self.windowTimes = times
        self.minTime = float(times[0] if minTime is None else minTime)
        self.maxTime = float(times[-1] if maxTime is None else maxTime)
        self.name = str(name)

    @property
    def lags(self) -> np.ndarray:
        return np.asarray(self.windowTimes[1:], dtype=float).copy()

    @property
    def numWindows(self) -> int:
        return int(self.windowTimes.size - 1)

    def setWindow(self, windowTimes) -> None:
        replacement = History(windowTimes, self.minTime, self.maxTime, self.name)
        self.windowTimes = replacement.windowTimes
        self.minTime = replacement.minTime
        self.maxTime = replacement.maxTime

    def _compute_single_history(self, train: nspikeTrain, historyIndex: int | None = None) -> Covariate:
        sigrep = train.getSigRep()
        time = np.asarray(sigrep.time, dtype=float).reshape(-1)
        spikes = np.asarray(train.getSpikeTimes(), dtype=float).reshape(-1)
        history = np.zeros((time.size, self.numWindows), dtype=float)

        for col, (window_start, window_stop) in enumerate(zip(self.windowTimes[:-1], self.windowTimes[1:])):
            for row, tval in enumerate(time):
                left = float(tval - window_stop)
                right = float(tval - window_start)
                history[row, col] = float(np.sum((spikes >= left) & (spikes < right)))

        label_prefix = train.name or f"neuron_{historyIndex or 1}"
        labels = [
            f"{label_prefix}_hist_{col + 1}"
            for col in range(self.numWindows)
        ]
        return Covariate(time, history, self.name, "time", "s", "count", labels)

    def compute_history(self, trains, historyIndex: int | None = None):
        from .trial import CovariateCollection

        if isinstance(trains, nspikeTrain):
            return CovariateCollection([self._compute_single_history(trains, historyIndex)])
        if hasattr(trains, "getNST") and hasattr(trains, "numSpikeTrains"):
            covariates = [self._compute_single_history(trains.getNST(index), index) for index in range(1, int(trains.numSpikeTrains) + 1)]
            return CovariateCollection(covariates)
        if isinstance(trains, Sequence) and not isinstance(trains, (str, bytes, np.ndarray)):
            covariates = [self._compute_single_history(train, index) for index, train in enumerate(trains, start=1)]
            return CovariateCollection(covariates)
        raise TypeError("History can only be computed from nspikeTrain, nstColl, or sequences of nspikeTrain")

    def computeHistory(self, trains, historyIndex: int | None = None):
        return self.compute_history(trains, historyIndex)

    def toStructure(self) -> dict[str, Any]:
        return {
            "windowTimes": self.windowTimes.tolist(),
            "minTime": self.minTime,
            "maxTime": self.maxTime,
            "name": self.name,
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any] | None) -> "History" | None:
        if structure is None:
            return None
        if "windowTimes" in structure:
            windowTimes = structure["windowTimes"]
        elif "lags" in structure:
            lags = np.asarray(structure["lags"], dtype=float).reshape(-1)
            windowTimes = np.concatenate([[0.0], lags])
        else:
            windowTimes = [0.0, 1.0]
        return History(
            windowTimes,
            minTime=structure.get("minTime"),
            maxTime=structure.get("maxTime"),
            name=structure.get("name", "History"),
        )

    def plot(self, *_, **__) -> None:
        return None


HistoryBasis = History


__all__ = ["History", "HistoryBasis"]
