from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import Covariate, SignalObj, nspikeTrain


@dataclass(frozen=True)
class HistoryFilter:
    """Discrete-time MATLAB-style transfer function numerator/denominator pair."""

    numerator: np.ndarray
    denominator: np.ndarray
    delta: float
    variable: str = "z^-1"


@dataclass(frozen=True)
class HistoryFilterBank:
    """Matrix-like collection of discrete history-window filters."""

    numerators: tuple[np.ndarray, ...]
    denominators: tuple[np.ndarray, ...]
    delta: float
    variable: str = "z^-1"

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.numerators), 1)

    @property
    def numFilters(self) -> int:
        return len(self.numerators)

    def __len__(self) -> int:
        return self.numFilters

    def __getitem__(self, index: int) -> HistoryFilter:
        return HistoryFilter(
            numerator=np.asarray(self.numerators[index], dtype=float).copy(),
            denominator=np.asarray(self.denominators[index], dtype=float).copy(),
            delta=float(self.delta),
            variable=self.variable,
        )

    def combine(self, coefficients) -> HistoryFilter:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if coeffs.size != self.numFilters:
            raise ValueError("Number of coefficients must match the number of history filters.")
        max_len = max(len(numerator) for numerator in self.numerators)
        padded = np.zeros((self.numFilters, max_len), dtype=float)
        for idx, numerator in enumerate(self.numerators):
            arr = np.asarray(numerator, dtype=float).reshape(-1)
            padded[idx, : arr.size] = arr
        numerator = coeffs @ padded
        denominator = np.zeros(max_len, dtype=float)
        denominator[0] = 1.0
        return HistoryFilter(numerator=np.asarray(numerator, dtype=float), denominator=denominator, delta=float(self.delta), variable=self.variable)

    def __rmatmul__(self, coefficients) -> HistoryFilter:
        return self.combine(coefficients)


class History:
    """MATLAB-style spike-history basis described by window boundaries."""

    def __init__(self, windowTimes, minTime: float | None = None, maxTime: float | None = None, name: str = "History") -> None:
        times = np.asarray(windowTimes, dtype=float).reshape(-1)
        if times.size <= 1:
            raise ValueError("At least two times points must be specified to determine a window")
        if np.any(np.diff(times) <= 0):
            raise ValueError("windowTimes must be strictly increasing")

        self.windowTimes = times
        self.minTime = None if minTime is None else float(minTime)
        self.maxTime = None if maxTime is None else float(maxTime)
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

    def toFilter(self, delta: float) -> HistoryFilterBank:
        delta = float(delta)
        if delta <= 0:
            raise ValueError("delta must be positive")
        tmin = np.asarray(self.windowTimes[:-1], dtype=float)
        tmax = np.asarray(self.windowTimes[1:], dtype=float)
        numerators: list[np.ndarray] = []
        denominators: list[np.ndarray] = []
        for row, (window_start, window_stop) in enumerate(zip(tmin, tmax)):
            num_samples = int(np.ceil(float(window_stop) / delta))
            start_sample = int(np.ceil(float(window_start) / delta)) + 1
            del row
            numerator = np.zeros(num_samples + 1, dtype=float)
            denominator = np.zeros(num_samples + 1, dtype=float)
            denominator[0] = 1.0
            numerator[start_sample : num_samples + 1] = 1.0
            numerators.append(numerator)
            denominators.append(denominator)
        return HistoryFilterBank(numerators=tuple(numerators), denominators=tuple(denominators), delta=delta)

    def _compute_single_history(self, train: nspikeTrain, historyIndex: int | None = None, time_grid=None) -> Covariate:
        sigrep = train.getSigRep() if time_grid is None else train.getSigRep(None, float(np.min(time_grid)), float(np.max(time_grid)))
        tmin = np.asarray(self.windowTimes[:-1], dtype=float)
        tmax = np.asarray(self.windowTimes[1:], dtype=float)
        data_columns: list[np.ndarray] = []
        data_labels: list[str] = []

        for window_start, window_stop in zip(tmin, tmax, strict=False):
            num_samples = int(np.ceil(float(window_stop) * float(train.sampleRate)))
            numerator = np.zeros(max(num_samples, 0), dtype=float)
            start_sample = int(np.ceil(float(window_start) * float(train.sampleRate))) + 1
            if num_samples > 0 and start_sample <= num_samples:
                numerator[max(start_sample - 1, 0) : num_samples] = 1.0
            filtered = sigrep.filter(numerator if numerator.size else [0.0], [1.0])
            delayed = filtered.filter([0.0, 1.0], [1.0])
            data_columns.append(np.asarray(delayed.dataToMatrix(), dtype=float))
            if historyIndex is None:
                data_labels.append(f"[{window_start:.3g},{window_stop:.3g}]")
            else:
                data_labels.append(f"[{window_start:.3g},{window_stop:.3g}]_{historyIndex}")

        data = np.hstack(data_columns) if data_columns else np.zeros((sigrep.time.size, 0), dtype=float)
        name = "History" if not getattr(train, "name", "") else f"History {train.name}"
        cov = Covariate(sigrep.time, data, name, sigrep.xlabelval, sigrep.xunits, sigrep.yunits, data_labels)

        if time_grid is not None:
            return cov

        if data.size == 0:
            return Covariate([], data, name, sigrep.xlabelval, sigrep.xunits, sigrep.yunits, data_labels)

        if (self.minTime is not None or self.maxTime is not None) and round(float(cov.sampleRate), 9) != round(float(train.sampleRate), 9):
            cov.resampleMe(float(train.sampleRate))
        min_time = float(cov.minTime) if self.minTime is None else float(self.minTime)
        max_time = float(cov.maxTime) if self.maxTime is None else float(self.maxTime)
        windowed = cov.getSigInTimeWindow(min_time, max_time)
        windowed.setMinTime(float(train.minTime))
        windowed.setMaxTime(float(train.maxTime))
        windowed.minTime = float(train.minTime)
        windowed.maxTime = float(train.maxTime)
        return windowed

    def compute_history(self, trains, historyIndex: int | None = None, time_grid=None):
        from .trial import CovariateCollection

        if isinstance(trains, nspikeTrain):
            cov = self._compute_single_history(trains, historyIndex, time_grid=time_grid)
            if historyIndex is not None:
                cov.name = f"History #{historyIndex} for {trains.name}"
            return CovariateCollection([cov])
        if hasattr(trains, "getNST") and hasattr(trains, "numSpikeTrains"):
            covariates = [
                self._compute_single_history(trains.getNST(index), historyIndex, time_grid=time_grid)
                for index in range(1, int(trains.numSpikeTrains) + 1)
            ]
            return CovariateCollection(covariates)
        if isinstance(trains, Sequence) and not isinstance(trains, (str, bytes, np.ndarray)):
            covariates = [self._compute_single_history(train, historyIndex, time_grid=time_grid) for train in trains]
            return CovariateCollection(covariates)
        raise TypeError("History can only be computed from nspikeTrain, nstColl, or sequences of nspikeTrain")

    def computeHistory(self, trains, historyIndex: int | None = None, time_grid=None):
        return self.compute_history(trains, historyIndex, time_grid=time_grid)

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

    def plot(self, *_, handle=None, **__):
        tmin = np.asarray(self.windowTimes[:-1], dtype=float)
        tmax = np.asarray(self.windowTimes[1:], dtype=float)
        sampleRate = 1000.0
        num_samples = max(1, int(round((float(np.max(tmax)) - float(np.min(tmin))) * sampleRate)))
        data = np.zeros((num_samples, tmax.size), dtype=float)
        dataLabels: list[str] = []
        for index, (start, stop) in enumerate(zip(tmin, tmax)):
            indMin = max(1, int(round((float(start) - float(np.min(tmin))) * sampleRate)))
            indMax = int(round((float(stop) - float(np.min(tmin))) * sampleRate))
            if indMax >= indMin:
                data[indMin - 1 : indMax, index] = 1.0
            dataLabels.append(f"[{start:.3g},{stop:.3g}]")
        time = np.linspace(float(np.min(tmin)), float(np.max(tmax)), num_samples)
        signal = SignalObj(time, data, "History", "time", "s", "", dataLabels)
        created_ax = handle is None
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 2.2))[1]
        plot_handles = signal.plot(handle=ax)
        return ax if created_ax else plot_handles


HistoryBasis = History


__all__ = ["History", "HistoryBasis", "HistoryFilter", "HistoryFilterBank"]
