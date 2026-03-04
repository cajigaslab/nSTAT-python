"""MATLAB-style compatibility adapters.

This module exposes class names and method aliases resembling MATLAB nSTAT
naming conventions while delegating all computation to the Python-native
`nstat.*` implementations.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from ...analysis import Analysis as _Analysis
from ...cif import CIFModel as _CIFModel
from ...confidence import ConfidenceInterval as _ConfidenceInterval
from ...decoding import DecodingAlgorithms as _DecodingAlgorithms
from ...events import Events as _Events
from ...fit import FitResult as _FitResult
from ...fit import FitSummary as _FitSummary
from ...history import HistoryBasis as _HistoryBasis
from ...signal import Covariate as _Covariate
from ...signal import Signal as _Signal
from ...spikes import SpikeTrain as _SpikeTrain
from ...spikes import SpikeTrainCollection as _SpikeTrainCollection
from ...trial import ConfigCollection as _ConfigCollection
from ...trial import CovariateCollection as _CovariateCollection
from ...trial import Trial as _Trial
from ...trial import TrialConfig as _TrialConfig


class SignalObj(_Signal):
    def _ensure_signalobj_state(self) -> None:
        if not hasattr(self, "_original_time"):
            self._original_time = self.time.copy()
            self._original_data = self.data.copy()
            self._original_name = str(self.name)
        if not hasattr(self, "_data_labels"):
            self._data_labels = [f"sig_{i+1}" for i in range(self.n_channels)]
        if not hasattr(self, "_data_mask"):
            self._data_mask = list(range(self.n_channels))

    def setName(self, name: str) -> "SignalObj":
        self.set_name(name)
        return self

    def setXlabel(self, label: str) -> "SignalObj":
        self.set_xlabel(label)
        return self

    def setYLabel(self, label: str) -> "SignalObj":
        self.set_ylabel(label)
        return self

    def setUnits(self, units: str) -> "SignalObj":
        self.set_units(units)
        return self

    def setXUnits(self, units: str) -> "SignalObj":
        self.set_x_units(units)
        return self

    def setYUnits(self, units: str) -> "SignalObj":
        self.set_y_units(units)
        return self

    def setMinTime(self, minTime: float) -> "SignalObj":
        self.set_min_time(minTime)
        return self

    def setMaxTime(self, maxTime: float) -> "SignalObj":
        self.set_max_time(maxTime)
        return self

    def setPlotProps(self, props: dict[str, Any]) -> "SignalObj":
        self.set_plot_props(props)
        return self

    def clearPlotProps(self) -> "SignalObj":
        self.clear_plot_props()
        return self

    def copySignal(self) -> "SignalObj":
        copied = self.copy_signal()
        return SignalObj(
            time=copied.time,
            data=copied.data,
            name=copied.name,
            units=copied.units,
            x_label=copied.x_label,
            y_label=copied.y_label,
            x_units=copied.x_units,
            y_units=copied.y_units,
            plot_props=copied.plot_props,
        )

    def shiftTime(self, offset_s: float) -> "SignalObj":
        self.shift_time(offset_s)
        return self

    def alignTime(self, newZero: float = 0.0) -> "SignalObj":
        self.align_time(newZero)
        return self

    def derivative(self) -> "SignalObj":
        out = super().derivative()
        return SignalObj(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def integral(self) -> np.ndarray:
        return super().integral()

    def dataToMatrix(self) -> np.ndarray:
        return self.data_to_matrix()

    def getSubSignal(self, selector: int | list[int] | np.ndarray) -> "SignalObj":
        out = super().get_sub_signal(selector)
        return SignalObj(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def merge(self, other: _Signal) -> "SignalObj":
        out = super().merge(other)
        return SignalObj(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def resample(self, sampleRate: float) -> "SignalObj":
        out = super().resample(sampleRate)
        return SignalObj(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def getData(self) -> np.ndarray:
        return self.data

    def getTime(self) -> np.ndarray:
        return self.time

    def getNumSamples(self) -> int:
        return self.n_samples

    def getNumSignals(self) -> int:
        return self.n_channels

    def getSampleRate(self) -> float:
        return self.sample_rate_hz

    def getDuration(self) -> float:
        return self.duration_s

    def _with_data(self, data: np.ndarray, name: str | None = None) -> "SignalObj":
        return SignalObj(
            time=self.time.copy(),
            data=np.asarray(data, dtype=float),
            name=self.name if name is None else name,
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def dataToStructure(self) -> dict[str, Any]:
        return {
            "time": self.time.copy(),
            "data": np.asarray(self.data, dtype=float).copy(),
            "name": self.name,
            "units": self.units,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_units": self.x_units,
            "y_units": self.y_units,
            "plot_props": dict(self.plot_props),
        }

    @staticmethod
    def signalFromStruct(payload: dict[str, Any]) -> "SignalObj":
        return SignalObj(
            time=np.asarray(payload["time"], dtype=float),
            data=np.asarray(payload["data"], dtype=float),
            name=str(payload.get("name", "signal")),
            units=str(payload.get("units", "")),
            x_label=payload.get("x_label"),
            y_label=payload.get("y_label"),
            x_units=payload.get("x_units"),
            y_units=payload.get("y_units"),
            plot_props=dict(payload.get("plot_props", {})),
        )

    @staticmethod
    def convertSimpleStructureToSigStructure(payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    @staticmethod
    def convertSigStructureToStructure(payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    def getPlotProps(self) -> dict[str, Any]:
        return dict(self.plot_props)

    def plotPropsSet(self) -> bool:
        return bool(self.plot_props)

    def findNearestTimeIndex(self, queryTime: float) -> int:
        return int(np.argmin(np.abs(self.time - float(queryTime))))

    def findNearestTimeIndices(self, queryTimes: np.ndarray) -> np.ndarray:
        q = np.asarray(queryTimes, dtype=float).reshape(-1)
        return np.asarray([self.findNearestTimeIndex(v) for v in q], dtype=int)

    def getValueAt(self, queryTime: float) -> np.ndarray:
        idx = self.findNearestTimeIndex(queryTime)
        row = self.data_to_matrix()[idx]
        return row.copy()

    def getSigInTimeWindow(self, t0: float, tf: float) -> "SignalObj":
        mask = (self.time >= float(t0)) & (self.time <= float(tf))
        if not np.any(mask):
            raise ValueError("time window excludes all samples")
        data = self.data_to_matrix()[mask]
        if data.shape[1] == 1:
            out_data = data[:, 0]
        else:
            out_data = data
        return SignalObj(
            time=self.time[mask].copy(),
            data=out_data,
            name=self.name,
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def derivativeAt(self, queryTime: float) -> np.ndarray:
        return self.derivative().getValueAt(queryTime)

    def findGlobalPeak(self) -> tuple[float, float, int]:
        mat = self.data_to_matrix()
        idx_flat = int(np.argmax(mat))
        idx_t, _ = np.unravel_index(idx_flat, mat.shape)
        return float(mat[idx_t].max()), float(self.time[idx_t]), int(idx_t)

    def findMaxima(self) -> tuple[np.ndarray, np.ndarray]:
        from scipy.signal import find_peaks

        y = np.mean(self.data_to_matrix(), axis=1)
        idx, _ = find_peaks(y)
        return self.time[idx], y[idx]

    def findMinima(self) -> tuple[np.ndarray, np.ndarray]:
        from scipy.signal import find_peaks

        y = np.mean(self.data_to_matrix(), axis=1)
        idx, _ = find_peaks(-y)
        return self.time[idx], y[idx]

    def findPeaks(self) -> tuple[np.ndarray, np.ndarray]:
        return self.findMaxima()

    def setSampleRate(self, sampleRate: float) -> "SignalObj":
        out = self.resample(sampleRate)
        self.time = out.time
        self.data = out.data
        return self

    def resampleMe(self, sampleRate: float) -> "SignalObj":
        return self.setSampleRate(sampleRate)

    def shift(self, offset_s: float) -> "SignalObj":
        return self.shiftTime(offset_s)

    def shiftMe(self, offset_s: float) -> "SignalObj":
        return self.shiftTime(offset_s)

    def setupPlots(self) -> "SignalObj":
        return self

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        h = plt.plot(self.time, self.data_to_matrix())
        return h

    def abs(self) -> "SignalObj":
        return self._with_data(np.abs(self.data_to_matrix()), name=f"abs({self.name})")

    def log(self) -> "SignalObj":
        return self._with_data(np.log(np.clip(self.data_to_matrix(), 1e-12, None)), name=f"log({self.name})")

    def sqrt(self) -> "SignalObj":
        return self._with_data(np.sqrt(np.clip(self.data_to_matrix(), 0.0, None)), name=f"sqrt({self.name})")

    def plus(self, other: float | np.ndarray | "SignalObj") -> "SignalObj":
        rhs = other.data_to_matrix() if isinstance(other, SignalObj) else np.asarray(other, dtype=float)
        return self._with_data(self.data_to_matrix() + rhs)

    def minus(self, other: float | np.ndarray | "SignalObj") -> "SignalObj":
        rhs = other.data_to_matrix() if isinstance(other, SignalObj) else np.asarray(other, dtype=float)
        return self._with_data(self.data_to_matrix() - rhs)

    def times(self, other: float | np.ndarray | "SignalObj") -> "SignalObj":
        rhs = other.data_to_matrix() if isinstance(other, SignalObj) else np.asarray(other, dtype=float)
        return self._with_data(self.data_to_matrix() * rhs)

    def rdivide(self, other: float | np.ndarray | "SignalObj") -> "SignalObj":
        rhs = other.data_to_matrix() if isinstance(other, SignalObj) else np.asarray(other, dtype=float)
        return self._with_data(self.data_to_matrix() / np.clip(rhs, 1e-12, None))

    def power(self, exponent: float) -> "SignalObj":
        return self._with_data(np.power(self.data_to_matrix(), float(exponent)))

    def mean(self) -> float:
        return float(np.mean(self.data_to_matrix()))

    def median(self) -> float:
        return float(np.median(self.data_to_matrix()))

    def max(self) -> float:
        return float(np.max(self.data_to_matrix()))

    def min(self) -> float:
        return float(np.min(self.data_to_matrix()))

    def std(self) -> float:
        return float(np.std(self.data_to_matrix(), ddof=0))

    def mode(self) -> float:
        vals, counts = np.unique(self.data_to_matrix().reshape(-1), return_counts=True)
        return float(vals[np.argmax(counts)])

    @staticmethod
    def cell2str(cells: list[Any], delimiter: str = ",") -> str:
        return delimiter.join(str(v) for v in cells)

    @staticmethod
    def getAvailableColor(index: int = 0) -> str:
        palette = ["b", "g", "r", "c", "m", "y", "k"]
        return palette[int(index) % len(palette)]

    def setDataLabels(self, labels: list[str]) -> "SignalObj":
        self._ensure_signalobj_state()
        if len(labels) != self.n_channels:
            raise ValueError("labels length must match number of channels")
        self._data_labels = [str(v) for v in labels]
        return self

    def areDataLabelsEmpty(self) -> bool:
        self._ensure_signalobj_state()
        return len(self._data_labels) == 0

    def getIndexFromLabel(self, label: str) -> int:
        self._ensure_signalobj_state()
        return self._data_labels.index(str(label))

    def getIndicesFromLabels(self, labels: list[str]) -> list[int]:
        self._ensure_signalobj_state()
        return [self.getIndexFromLabel(label) for label in labels]

    def isLabelPresent(self, label: str) -> bool:
        self._ensure_signalobj_state()
        return str(label) in self._data_labels

    def convertNamesToIndices(self, labels: list[str]) -> list[int]:
        return self.getIndicesFromLabels(labels)

    def setDataMask(self, mask: list[int] | np.ndarray) -> "SignalObj":
        self._ensure_signalobj_state()
        idx = [int(v) for v in np.asarray(mask, dtype=int).reshape(-1)]
        clean: list[int] = []
        for i in idx:
            if i >= 1 and i <= self.n_channels:
                clean.append(i - 1)
            elif i >= 0 and i < self.n_channels:
                clean.append(i)
            else:
                raise IndexError("mask index out of range")
        self._data_mask = sorted(set(clean))
        return self

    def setMask(self, selector: list[int] | list[str] | np.ndarray) -> "SignalObj":
        vals = np.asarray(selector, dtype=object).reshape(-1).tolist()
        if vals and all(isinstance(v, (str, np.str_)) for v in vals):
            self._ensure_signalobj_state()
            idx = self.getIndicesFromLabels([str(v) for v in vals])
            self._data_mask = sorted(set(idx))
            return self
        return self.setDataMask([int(v) for v in vals])

    def setMaskByInd(self, mask: list[int] | np.ndarray) -> "SignalObj":
        return self.setDataMask(mask)

    def setMaskByLabels(self, labels: list[str]) -> "SignalObj":
        return self.setMask(labels)

    def isMaskSet(self) -> bool:
        self._ensure_signalobj_state()
        return self._data_mask != list(range(self.n_channels))

    def findIndFromDataMask(self) -> list[int]:
        self._ensure_signalobj_state()
        return list(self._data_mask)

    def resetMask(self) -> "SignalObj":
        self._ensure_signalobj_state()
        self._data_mask = list(range(self.n_channels))
        return self

    def getSubSignalFromInd(self, selector: int | list[int] | np.ndarray) -> "SignalObj":
        return self.getSubSignal(selector)

    def getSubSignalFromNames(self, labels: list[str]) -> "SignalObj":
        return self.getSubSignal(self.getIndicesFromLabels(labels))

    def getSubSignalsWithinNStd(self, nStd: float = 1.0) -> "SignalObj":
        mat = self.data_to_matrix()
        means = np.mean(mat, axis=0)
        mu = float(np.mean(means))
        sigma = float(np.std(means))
        if sigma <= 0.0:
            idx = list(range(mat.shape[1]))
        else:
            idx = [i for i, m in enumerate(means) if abs(float(m) - mu) <= float(nStd) * sigma]
        if not idx:
            idx = [int(np.argmin(np.abs(means - mu)))]
        return self.getSubSignal(idx)

    def getOrigDataSig(self) -> "SignalObj":
        self._ensure_signalobj_state()
        return SignalObj(
            time=self._original_time.copy(),
            data=np.asarray(self._original_data, dtype=float).copy(),
            name=str(self._original_name),
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def getOriginalData(self) -> np.ndarray:
        self._ensure_signalobj_state()
        return np.asarray(self._original_data, dtype=float).copy()

    def restoreToOriginal(self) -> "SignalObj":
        self._ensure_signalobj_state()
        self.time = self._original_time.copy()
        self.data = np.asarray(self._original_data, dtype=float).copy()
        self.name = str(self._original_name)
        self.resetMask()
        return self

    def makeCompatible(self, other: _Signal) -> tuple["SignalObj", "SignalObj"]:
        rhs = SignalObj(
            time=other.time.copy(),
            data=other.data.copy(),
            name=other.name,
            units=other.units,
            x_label=other.x_label,
            y_label=other.y_label,
            x_units=other.x_units,
            y_units=other.y_units,
            plot_props=dict(other.plot_props),
        )
        fs = max(float(self.sample_rate_hz), float(rhs.sample_rate_hz))
        lhs_r = self.resample(fs)
        rhs_r = rhs.resample(fs)
        t0 = max(float(lhs_r.time[0]), float(rhs_r.time[0]))
        tf = min(float(lhs_r.time[-1]), float(rhs_r.time[-1]))
        if tf <= t0:
            raise ValueError("signals do not overlap in time")
        return lhs_r.getSigInTimeWindow(t0, tf), rhs_r.getSigInTimeWindow(t0, tf)

    def alignToMax(self, targetTime: float = 0.0) -> "SignalObj":
        _max_val, t_max, _idx = self.findGlobalPeak()
        return self.shiftTime(float(targetTime) - t_max)

    def windowedSignal(self, windowSamples: int = 11) -> "SignalObj":
        win = int(windowSamples)
        if win <= 0:
            raise ValueError("windowSamples must be positive")
        kernel = np.ones(win, dtype=float) / float(win)
        mat = self.data_to_matrix()
        out = np.column_stack([np.convolve(mat[:, i], kernel, mode="same") for i in range(mat.shape[1])])
        return self._with_data(out, name=f"windowed({self.name})")

    def normWindowedSignal(self, windowSamples: int = 11) -> "SignalObj":
        smoothed = self.windowedSignal(windowSamples=windowSamples).data_to_matrix()
        mat = self.data_to_matrix()
        centered = mat - smoothed
        scale = np.std(centered, axis=0, keepdims=True)
        scale = np.where(scale <= 1e-12, 1.0, scale)
        return self._with_data(centered / scale, name=f"norm_windowed({self.name})")

    def filter(self, b: np.ndarray, a: np.ndarray) -> "SignalObj":
        from scipy.signal import lfilter

        b_arr = np.asarray(b, dtype=float).reshape(-1)
        a_arr = np.asarray(a, dtype=float).reshape(-1)
        mat = self.data_to_matrix()
        out = np.column_stack([lfilter(b_arr, a_arr, mat[:, i]) for i in range(mat.shape[1])])
        return self._with_data(out, name=f"filter({self.name})")

    def filtfilt(self, b: np.ndarray, a: np.ndarray) -> "SignalObj":
        from scipy.signal import filtfilt
        from scipy.signal import lfilter

        b_arr = np.asarray(b, dtype=float).reshape(-1)
        a_arr = np.asarray(a, dtype=float).reshape(-1)
        mat = self.data_to_matrix()
        filtered_cols: list[np.ndarray] = []
        for i in range(mat.shape[1]):
            x = np.asarray(mat[:, i], dtype=float).reshape(-1)
            if x.size < 2:
                filtered_cols.append(lfilter(b_arr, a_arr, x))
                continue
            ntaps = max(int(a_arr.size), int(b_arr.size))
            padlen = min(3 * ntaps, int(x.size) - 1)
            try:
                filtered_cols.append(filtfilt(b_arr, a_arr, x, padlen=padlen))
            except ValueError:
                fwd = lfilter(b_arr, a_arr, x)
                bwd = lfilter(b_arr, a_arr, fwd[::-1])
                filtered_cols.append(bwd[::-1])
        out = np.column_stack(filtered_cols)
        return self._with_data(out, name=f"filtfilt({self.name})")

    def periodogram(self) -> tuple[np.ndarray, np.ndarray]:
        from scipy.signal import periodogram

        fs = float(self.sample_rate_hz)
        mat = self.data_to_matrix()
        f, p = periodogram(mat[:, 0], fs=fs)
        return np.asarray(f, dtype=float), np.asarray(p, dtype=float)

    def MTMspectrum(self) -> tuple[np.ndarray, np.ndarray]:
        return self.periodogram()

    def spectrogram(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.signal import spectrogram

        fs = float(self.sample_rate_hz)
        mat = self.data_to_matrix()
        f, t, s = spectrogram(mat[:, 0], fs=fs)
        return np.asarray(f, dtype=float), np.asarray(t, dtype=float), np.asarray(s, dtype=float)

    def _crosscorr_core(
        self,
        x: np.ndarray,
        y: np.ndarray,
        maxLag: int | None = None,
        demean: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if demean:
            x_arr = x_arr - np.mean(x_arr)
            y_arr = y_arr - np.mean(y_arr)
        c = np.correlate(x_arr, y_arr, mode="full")
        lags = np.arange(-x_arr.size + 1, y_arr.size, dtype=int)
        if maxLag is not None:
            mask = (lags >= -int(maxLag)) & (lags <= int(maxLag))
            lags = lags[mask]
            c = c[mask]
        return lags, c.astype(float)

    def xcorr(self, other: "SignalObj | None" = None, maxLag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        lhs = self.data_to_matrix()[:, 0]
        rhs = lhs if other is None else other.data_to_matrix()[:, 0]
        return self._crosscorr_core(lhs, rhs, maxLag=maxLag, demean=False)

    def xcov(self, other: "SignalObj | None" = None, maxLag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        lhs = self.data_to_matrix()[:, 0]
        rhs = lhs if other is None else other.data_to_matrix()[:, 0]
        return self._crosscorr_core(lhs, rhs, maxLag=maxLag, demean=True)

    def autocorrelation(self, maxLag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.xcorr(other=None, maxLag=maxLag)

    def crosscorrelation(self, other: "SignalObj", maxLag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.xcorr(other=other, maxLag=maxLag)

    def plotVariability(self) -> Any:
        import matplotlib.pyplot as plt

        mat = self.data_to_matrix()
        mu = np.mean(mat, axis=1)
        sd = np.std(mat, axis=1)
        plt.plot(self.time, mu, "k-")
        return plt.fill_between(self.time, mu - sd, mu + sd, color="k", alpha=0.2)

    def plotAllVariability(self) -> Any:
        return self.plotVariability()

    def transpose(self) -> "SignalObj":
        return self.copySignal()

    def ctranspose(self) -> "SignalObj":
        return self.transpose()

    def uminus(self) -> "SignalObj":
        return self._with_data(-self.data_to_matrix(), name=f"-({self.name})")

    def uplus(self) -> "SignalObj":
        return self.copySignal()

    def mtimes(self, other: float | np.ndarray | "SignalObj") -> "SignalObj":
        lhs = self.data_to_matrix()
        if isinstance(other, SignalObj):
            rhs = other.data_to_matrix()
            out = lhs @ rhs if lhs.shape[1] == rhs.shape[0] else lhs * rhs
            return self._with_data(out, name=f"{self.name}*")
        if np.isscalar(other):
            scalar = float(np.asarray(other, dtype=float).reshape(()).item())
            out = lhs * scalar
            return self._with_data(out, name=f"{self.name}*")
        rhs = np.asarray(other, dtype=float)
        if rhs.ndim == 1:
            out = lhs * rhs.reshape(1, -1)
        else:
            out = lhs @ rhs
        return self._with_data(out, name=f"{self.name}*")

    def ldivide(self, other: float | np.ndarray | "SignalObj") -> "SignalObj":
        if isinstance(other, SignalObj):
            rhs = other.data_to_matrix()
        else:
            rhs = np.asarray(other, dtype=float)
        lhs = np.clip(self.data_to_matrix(), 1e-12, None)
        return self._with_data(rhs / lhs, name=f"{self.name}\\")


class Covariate(_Covariate):
    @staticmethod
    def Covariate(payload: dict[str, Any]) -> _Covariate:
        return Covariate.fromStructure(payload)

    def computeMeanPlusCI(self, axis: int = 1, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.compute_mean_plus_ci(axis=axis, level=level)

    def getSubSignal(self, selector: int | str | list[int] | list[str]) -> _Covariate:
        out = super().get_sub_signal(selector)
        return Covariate(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            labels=out.labels,
            conf_interval=out.conf_interval,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def setConfInterval(self, interval: Any) -> _Covariate:
        self.set_conf_interval(interval)
        return self

    def isConfIntervalSet(self) -> bool:
        return self.is_conf_interval_set()

    def getSigRep(self) -> np.ndarray:
        return self.data_to_matrix()

    def dataToStructure(self) -> dict[str, Any]:
        return self.to_structure()

    def toStructure(self) -> dict[str, Any]:
        return self.to_structure()

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _Covariate:
        out = _Covariate.from_structure(payload)
        return Covariate(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            labels=out.labels,
            conf_interval=out.conf_interval,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def getData(self) -> np.ndarray:
        return self.data

    def getTime(self) -> np.ndarray:
        return self.time

    def getLabels(self) -> list[str]:
        return self.labels

    def getNumSignals(self) -> int:
        return self.n_channels

    def getSampleRate(self) -> float:
        return self.sample_rate_hz

    def copySignal(self) -> _Covariate:
        out = super().copy_signal()
        return Covariate(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            labels=self.labels.copy(),
            conf_interval=self.conf_interval,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def plus(self, other: float | np.ndarray | _Signal) -> _Covariate:
        if isinstance(other, _Signal):
            rhs = other.data_to_matrix()
        else:
            rhs = np.asarray(other, dtype=float)
        lhs = self.data_to_matrix()
        out = lhs + rhs
        data = out[:, 0] if out.ndim == 2 and out.shape[1] == 1 else out
        return Covariate(
            time=self.time.copy(),
            data=data,
            name=f"{self.name}+",
            units=self.units,
            labels=self.labels.copy(),
            conf_interval=self.conf_interval,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def minus(self, other: float | np.ndarray | _Signal) -> _Covariate:
        if isinstance(other, _Signal):
            rhs = other.data_to_matrix()
        else:
            rhs = np.asarray(other, dtype=float)
        lhs = self.data_to_matrix()
        out = lhs - rhs
        data = out[:, 0] if out.ndim == 2 and out.shape[1] == 1 else out
        return Covariate(
            time=self.time.copy(),
            data=data,
            name=f"{self.name}-",
            units=self.units,
            labels=self.labels.copy(),
            conf_interval=self.conf_interval,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        return plt.plot(self.time, self.data_to_matrix())


class ConfidenceInterval(_ConfidenceInterval):
    @staticmethod
    def ConfidenceInterval(*args: Any, **kwargs: Any) -> _ConfidenceInterval:
        if len(args) == 1 and isinstance(args[0], dict):
            return ConfidenceInterval.fromStructure(args[0])
        return ConfidenceInterval(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _ConfidenceInterval:
        return ConfidenceInterval(
            time=np.asarray(payload["time"], dtype=float),
            lower=np.asarray(payload["lower"], dtype=float),
            upper=np.asarray(payload["upper"], dtype=float),
            level=float(payload.get("level", 0.95)),
        )

    def toStructure(self) -> dict[str, Any]:
        return {
            "time": self.time.copy(),
            "lower": self.lower.copy(),
            "upper": self.upper.copy(),
            "level": float(self.level),
        }

    def setColor(self, color: str) -> _ConfidenceInterval:
        setattr(self, "_color", str(color))
        return self

    def setValue(self, values: np.ndarray | float) -> _ConfidenceInterval:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self.time.shape, float(arr), dtype=float)
        if arr.shape != self.time.shape:
            raise ValueError("values shape must match time shape")
        half_width = 0.5 * self.width()
        self.lower = arr - half_width
        self.upper = arr + half_width
        return self

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        color = getattr(self, "_color", "tab:blue")
        return plt.fill_between(self.time, self.lower, self.upper, color=color, alpha=0.25)

    def getWidth(self) -> np.ndarray:
        return self.width()


class Events(_Events):
    @staticmethod
    def Events(*args: Any, **kwargs: Any) -> _Events:
        if len(args) == 1 and isinstance(args[0], dict):
            return Events.fromStructure(args[0])
        return Events(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _Events:
        return Events(
            times=np.asarray(payload["times"], dtype=float),
            labels=[str(v) for v in payload.get("labels", [])],
        )

    def toStructure(self) -> dict[str, Any]:
        return {"times": self.times.copy(), "labels": list(self.labels)}

    @staticmethod
    def dsxy2figxy(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        pts = np.column_stack([np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)])
        disp = ax.transData.transform(pts)
        fig = ax.get_figure()
        if fig is None:
            raise RuntimeError("cannot transform without an active matplotlib figure")
        out = fig.transFigure.inverted().transform(disp)
        return out

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        if self.times.size == 0:
            return plt.plot([], [])
        ymin, ymax = plt.ylim()
        if ymin == ymax:
            ymin, ymax = 0.0, 1.0
        return plt.vlines(self.times, ymin, ymax, colors="k", linestyles="--", linewidth=1.0)

    def getTimes(self) -> np.ndarray:
        return self.times


class History(_HistoryBasis):
    @staticmethod
    def History(*args: Any, **kwargs: Any) -> _HistoryBasis:
        if len(args) == 1 and isinstance(args[0], dict):
            return History.fromStructure(args[0])
        return History(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _HistoryBasis:
        return History(bin_edges_s=np.asarray(payload["bin_edges_s"], dtype=float))

    def toStructure(self) -> dict[str, Any]:
        return {"bin_edges_s": self.bin_edges_s.copy()}

    def setWindow(self, *args: Any) -> _HistoryBasis:
        if len(args) == 1:
            edges = np.asarray(args[0], dtype=float).reshape(-1)
        elif len(args) == 3:
            t0 = float(args[0])
            tf = float(args[1])
            n_bins = int(args[2])
            if n_bins <= 0:
                raise ValueError("n_bins must be > 0")
            edges = np.linspace(t0, tf, n_bins + 1, dtype=float)
        else:
            raise ValueError("setWindow expects (edges) or (t0, tf, n_bins)")
        if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("history edges must be strictly increasing with at least 2 elements")
        self.bin_edges_s = edges
        return self

    def toFilter(self) -> np.ndarray:
        widths = np.diff(self.bin_edges_s)
        total = float(np.sum(widths))
        if total <= 0.0:
            return widths
        return widths / total

    def computeHistory(self, spikeTimes_s: np.ndarray, timeGrid_s: np.ndarray) -> np.ndarray:
        return self.design_matrix(spike_times_s=spikeTimes_s, time_grid_s=timeGrid_s)

    def computeNSTHistoryWindow(self, spikeTrain: Any, timeGrid_s: np.ndarray) -> np.ndarray:
        spike_times = np.asarray(getattr(spikeTrain, "spike_times"), dtype=float)
        return self.computeHistory(spikeTimes_s=spike_times, timeGrid_s=timeGrid_s)

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        widths = np.diff(self.bin_edges_s)
        centers = 0.5 * (self.bin_edges_s[:-1] + self.bin_edges_s[1:])
        return plt.bar(centers, widths, width=widths, align="center", alpha=0.4, color="tab:gray")

    def getNumBins(self) -> int:
        return self.n_bins

    def getDesignMatrix(self, spike_times_s: np.ndarray, time_grid_s: np.ndarray) -> np.ndarray:
        return self.design_matrix(spike_times_s=spike_times_s, time_grid_s=time_grid_s)


class nspikeTrain(_SpikeTrain):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._original_spike_times = self.spike_times.copy()
        self._original_t_start = float(self.t_start)
        self._original_t_end = float(self.t_end) if self.t_end is not None else None
        self._original_name = str(self.name)
        self._sig_rep: np.ndarray | None = None
        self._mer: float | None = None

    @staticmethod
    def nspikeTrain(*args: Any, **kwargs: Any) -> _SpikeTrain:
        if len(args) == 1 and isinstance(args[0], dict):
            return nspikeTrain.fromStructure(args[0])
        return nspikeTrain(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _SpikeTrain:
        t_end_raw = payload.get("t_end", payload.get("maxTime"))
        return nspikeTrain(
            spike_times=np.asarray(payload.get("spike_times", payload.get("spikeTimes", [])), dtype=float),
            t_start=float(payload.get("t_start", payload.get("minTime", 0.0))),
            t_end=float(t_end_raw) if t_end_raw is not None else None,
            name=str(payload.get("name", "unit")),
        )

    def toStructure(self) -> dict[str, Any]:
        return {
            "spike_times": self.spike_times.copy(),
            "t_start": float(self.t_start),
            "t_end": float(self.t_end) if self.t_end is not None else None,
            "name": str(self.name),
            "MER": self._mer,
        }

    def setName(self, name: str) -> _SpikeTrain:
        self.name = str(name)
        return self

    def setMER(self, mer: float) -> _SpikeTrain:
        self._mer = float(mer)
        return self

    def setSigRep(self, sigRep: np.ndarray) -> _SpikeTrain:
        self._sig_rep = np.asarray(sigRep, dtype=float).copy()
        return self

    def clearSigRep(self) -> _SpikeTrain:
        self._sig_rep = None
        return self

    def getSigRep(self, binSize_s: float = 0.001, mode: Literal["binary", "count"] = "binary") -> np.ndarray:
        if self._sig_rep is not None:
            return self._sig_rep.copy()
        if mode == "binary":
            _, y = self.binarize(bin_size_s=binSize_s)
        else:
            _, y = self.bin_counts(bin_size_s=binSize_s)
        return y

    def isSigRepBinary(self, binSize_s: float = 0.001) -> bool:
        y = self.getSigRep(binSize_s=binSize_s, mode="count")
        return bool(np.all((y == 0.0) | (y == 1.0)))

    def getSpikeTimes(self) -> np.ndarray:
        return self.spike_times

    def getISIs(self) -> np.ndarray:
        return np.diff(self.spike_times)

    def getMinISI(self) -> float:
        isi = self.getISIs()
        if isi.size == 0:
            return float(np.inf)
        return float(np.min(isi))

    def getMaxBinSizeBinary(self) -> float:
        min_isi = self.getMinISI()
        if not np.isfinite(min_isi) or min_isi <= 0.0:
            dur = float(self.duration_s())
            return dur if dur > 0.0 else 1.0
        return float(min_isi)

    def getDuration(self) -> float:
        return self.duration_s()

    def getFiringRate(self) -> float:
        return self.firing_rate_hz()

    def computeRate(self) -> float:
        return self.getFiringRate()

    def computeStatistics(self) -> dict[str, float]:
        isi = self.getISIs()
        return {
            "n_spikes": float(self.spike_times.size),
            "duration_s": float(self.duration_s()),
            "rate_hz": float(self.getFiringRate()),
            "mean_isi": float(np.mean(isi)) if isi.size else float(np.nan),
            "std_isi": float(np.std(isi)) if isi.size else float(np.nan),
        }

    def getFieldVal(self, fieldName: str) -> Any:
        if hasattr(self, fieldName):
            return getattr(self, fieldName)
        raise KeyError(f"field '{fieldName}' not found")

    def getLStatistic(self) -> float:
        isi = self.getISIs()
        if isi.size == 0:
            return 0.0
        mu = float(np.mean(isi))
        if mu <= 0.0:
            return 0.0
        return float(np.std(isi) / mu)

    def nstCopy(self) -> _SpikeTrain:
        return nspikeTrain(
            spike_times=self.spike_times.copy(),
            t_start=float(self.t_start),
            t_end=float(self.t_end) if self.t_end is not None else None,
            name=str(self.name),
        )

    def resample(self, sampleRate: float) -> _SpikeTrain:
        if sampleRate <= 0.0:
            raise ValueError("sampleRate must be positive")
        dt = 1.0 / float(sampleRate)
        snapped = np.round(self.spike_times / dt) * dt
        self.spike_times = np.unique(snapped)
        return self

    def restoreToOriginal(self) -> _SpikeTrain:
        self.spike_times = self._original_spike_times.copy()
        self.t_start = float(self._original_t_start)
        self.t_end = float(self._original_t_end) if self._original_t_end is not None else None
        self.name = str(self._original_name)
        self._sig_rep = None
        return self

    def partitionNST(self, partitionEdges_s: np.ndarray | list[float]) -> list[_SpikeTrain]:
        edges = np.asarray(partitionEdges_s, dtype=float).reshape(-1)
        if edges.size < 2:
            raise ValueError("partition edges must contain at least two values")
        out: list[_SpikeTrain] = []
        for i in range(edges.size - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            mask = (self.spike_times >= lo) & (self.spike_times <= hi)
            out.append(
                nspikeTrain(spike_times=self.spike_times[mask], t_start=lo, t_end=hi, name=f"{self.name}_{i+1}")
            )
        return out

    def shiftTime(self, offset_s: float) -> _SpikeTrain:
        self.shift_time(offset_s)
        return self

    def setMinTime(self, t_min: float) -> _SpikeTrain:
        self.set_min_time(t_min)
        return self

    def setMaxTime(self, t_max: float) -> _SpikeTrain:
        self.set_max_time(t_max)
        return self

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        y = np.ones(self.spike_times.size, dtype=float)
        return plt.plot(self.spike_times, y, "k.")

    def plotISIHistogram(self, bins: int = 20) -> Any:
        import matplotlib.pyplot as plt

        isi = self.getISIs()
        if isi.size == 0:
            return plt.hist([], bins=bins)
        return plt.hist(isi, bins=bins, color="tab:blue", alpha=0.6)

    def plotExponentialFit(self) -> Any:
        import matplotlib.pyplot as plt

        isi = self.getISIs()
        if isi.size == 0:
            return plt.plot([], [])
        lam = 1.0 / max(float(np.mean(isi)), 1e-12)
        x = np.linspace(0.0, float(np.max(isi)), 200)
        y = lam * np.exp(-lam * x)
        return plt.plot(x, y, "r-")

    def plotJointISIHistogram(self, bins: int = 20) -> Any:
        import matplotlib.pyplot as plt

        isi = self.getISIs()
        if isi.size < 2:
            return plt.hist2d([], [], bins=bins)
        return plt.hist2d(isi[:-1], isi[1:], bins=bins, cmap="Blues")

    def plotISISpectrumFunction(self) -> Any:
        import matplotlib.pyplot as plt

        isi = self.getISIs()
        if isi.size < 2:
            return plt.plot([], [])
        centered = isi - np.mean(isi)
        spec = np.abs(np.fft.rfft(centered)) ** 2
        freq = np.fft.rfftfreq(centered.size, d=max(float(np.mean(isi)), 1e-6))
        return plt.plot(freq, spec, "k-")

    def plotProbPlot(self) -> Any:
        import matplotlib.pyplot as plt

        isi = np.sort(self.getISIs())
        if isi.size == 0:
            return plt.plot([], [])
        q = (np.arange(1, isi.size + 1) - 0.5) / isi.size
        return plt.plot(q, isi, "k.")


class nstColl(_SpikeTrainCollection):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._original_trains = [
            train.copy() if hasattr(train, "copy") else _SpikeTrain(train.spike_times.copy(), train.t_start, train.t_end, train.name)
            for train in self.trains
        ]
        self._neighbors: Any = None
        self._neuron_mask: list[int] | None = None

    def getNumUnits(self) -> int:
        return self.n_units

    @staticmethod
    def nstColl(*args: Any, **kwargs: Any) -> _SpikeTrainCollection:
        if len(args) == 1 and isinstance(args[0], dict):
            return nstColl.fromStructure(args[0])
        return nstColl(*args, **kwargs)

    def getBinnedMatrix(
        self, binSize_s: float, mode: Literal["binary", "count"] = "binary"
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.to_binned_matrix(bin_size_s=binSize_s, mode=mode)

    def merge(self, other: _SpikeTrainCollection) -> _SpikeTrainCollection:
        merged = super().merge(other)
        return nstColl(merged.trains)

    def getFirstSpikeTime(self) -> float:
        return self.get_first_spike_time()

    def getLastSpikeTime(self) -> float:
        return self.get_last_spike_time()

    def getSpikeTimes(self) -> list[np.ndarray]:
        return self.get_spike_times()

    def getNST(self, ind: int) -> nspikeTrain:
        train = self.get_nst(ind)
        return nspikeTrain(
            spike_times=train.spike_times.copy(),
            t_start=train.t_start,
            t_end=train.t_end,
            name=train.name,
        )

    def getNSTnames(self) -> list[str]:
        return self.get_nst_names()

    def getUniqueNSTnames(self) -> list[str]:
        return self.get_unique_nst_names()

    def getNSTIndicesFromName(self, name: str) -> list[int]:
        return self.get_nst_indices_from_name(name)

    def getNSTnameFromInd(self, ind: int) -> str:
        return self.get_nst_name_from_ind(ind)

    def getNSTFromName(self, name: str) -> nspikeTrain:
        match = self.get_nst_from_name(name)
        if isinstance(match, list):
            match = match[0]
        return nspikeTrain(
            spike_times=match.spike_times.copy(),
            t_start=match.t_start,
            t_end=match.t_end,
            name=match.name,
        )

    def addToColl(self, train: _SpikeTrain) -> _SpikeTrainCollection:
        self.add_to_coll(train)
        return self

    def addSingleSpikeToColl(self, unitInd: int, spikeTime: float) -> _SpikeTrainCollection:
        self.add_single_spike_to_coll(unit_index=unitInd, spike_time_s=spikeTime)
        return self

    def dataToMatrix(self, binSize_s: float, mode: Literal["binary", "count"] = "binary") -> np.ndarray:
        return self.data_to_matrix(bin_size_s=binSize_s, mode=mode)

    def toSpikeTrain(self, name: str = "merged") -> nspikeTrain:
        merged = super().to_spike_train(name=name)
        return nspikeTrain(
            spike_times=merged.spike_times.copy(),
            t_start=merged.t_start,
            t_end=merged.t_end,
            name=merged.name,
        )

    def shiftTime(self, offset_s: float) -> _SpikeTrainCollection:
        self.shift_time(offset_s)
        return self

    def setMinTime(self, t_min: float) -> _SpikeTrainCollection:
        self.set_min_time(t_min)
        return self

    def setMaxTime(self, t_max: float) -> _SpikeTrainCollection:
        self.set_max_time(t_max)
        return self

    def toStructure(self) -> dict[str, Any]:
        return {
            "trains": [
                {
                    "spike_times": train.spike_times.copy(),
                    "t_start": float(train.t_start),
                    "t_end": float(train.t_end) if train.t_end is not None else None,
                    "name": train.name,
                }
                for train in self.trains
            ]
        }

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _SpikeTrainCollection:
        trains = [
            nspikeTrain(
                spike_times=np.asarray(row["spike_times"], dtype=float),
                t_start=float(row.get("t_start", 0.0)),
                t_end=float(row["t_end"]) if row.get("t_end") is not None else None,
                name=str(row.get("name", f"unit_{i+1}")),
            )
            for i, row in enumerate(payload.get("trains", []))
        ]
        if not trains:
            raise ValueError("fromStructure requires at least one train")
        return nstColl(cast(list[_SpikeTrain], trains))

    def updateTimes(self) -> _SpikeTrainCollection:
        for train in self.trains:
            if train.spike_times.size:
                train.t_start = min(train.t_start, float(train.spike_times.min()))
                train.t_end = max(float(train.t_end) if train.t_end is not None else train.t_start, float(train.spike_times.max()))
        return self

    def getISIs(self) -> list[np.ndarray]:
        return [np.diff(train.spike_times) for train in self.trains]

    def getMinISIs(self) -> np.ndarray:
        out = []
        for isi in self.getISIs():
            out.append(float(np.min(isi)) if isi.size else np.inf)
        return np.asarray(out, dtype=float)

    def isSigRepBinary(self, binSize_s: float = 0.001) -> bool:
        _, mat = self.to_binned_matrix(bin_size_s=binSize_s, mode="count")
        return bool(np.all((mat == 0) | (mat == 1)))

    def BinarySigRep(self, binSize_s: float = 0.001) -> np.ndarray:
        return self.dataToMatrix(binSize_s=binSize_s, mode="binary")

    def psth(self, binSize_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        t, mat = self.to_binned_matrix(bin_size_s=binSize_s, mode="count")
        return t, np.mean(mat, axis=0)

    def psthBars(self, binSize_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        return self.psth(binSize_s=binSize_s)

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        for i, train in enumerate(self.trains):
            y = np.full(train.spike_times.shape, i + 1, dtype=float)
            plt.plot(train.spike_times, y, "k.", markersize=3)
        return plt.gca()

    def getFieldVal(self, fieldName: str) -> list[Any]:
        out: list[Any] = []
        for train in self.trains:
            if not hasattr(train, fieldName):
                raise KeyError(f"field '{fieldName}' not found")
            out.append(getattr(train, fieldName))
        return out

    def findMaxSampleRate(self) -> float:
        min_isi = float(np.min(self.getMinISIs()))
        if not np.isfinite(min_isi) or min_isi <= 0.0:
            return float(np.inf)
        return float(1.0 / min_isi)

    def getMaxBinSizeBinary(self) -> float:
        min_isi = float(np.min(self.getMinISIs()))
        if not np.isfinite(min_isi) or min_isi <= 0.0:
            return 1.0
        return min_isi

    def setMask(self, selector: list[int] | list[str]) -> _SpikeTrainCollection:
        if selector and isinstance(selector[0], str):
            idx = [self.get_nst_indices_from_name(str(name))[0] for name in cast(list[str], selector)]
        else:
            idx_raw = [int(v) for v in cast(list[int], selector)]
            idx = []
            for v in idx_raw:
                if v >= 1 and v <= self.n_units:
                    idx.append(v - 1)
                elif v >= 0 and v < self.n_units:
                    idx.append(v)
                else:
                    raise IndexError("mask index out of range")
        self._neuron_mask = sorted(set(idx))
        return self

    def setNeuronMask(self, selector: list[int] | list[str]) -> _SpikeTrainCollection:
        return self.setMask(selector)

    def setNeuronMaskFromInd(self, indices: list[int] | np.ndarray) -> _SpikeTrainCollection:
        idx = [int(v) for v in np.asarray(indices).reshape(-1)]
        return self.setMask(idx)

    def resetMask(self) -> _SpikeTrainCollection:
        self._neuron_mask = list(range(self.n_units))
        return self

    def isNeuronMaskSet(self) -> bool:
        return self._neuron_mask is not None

    def getIndFromMask(self) -> list[int]:
        if self._neuron_mask is None:
            return list(range(self.n_units))
        return list(self._neuron_mask)

    def getIndFromMaskMinusOne(self) -> list[int]:
        return self.getIndFromMask()

    def setNeighbors(self, neighbors: Any) -> _SpikeTrainCollection:
        self._neighbors = neighbors
        return self

    def getNeighbors(self) -> Any:
        return self._neighbors

    def areNeighborsSet(self) -> bool:
        return self._neighbors is not None

    def restoreToOriginal(self) -> _SpikeTrainCollection:
        self.trains = [
            train.copy() if hasattr(train, "copy") else _SpikeTrain(train.spike_times.copy(), train.t_start, train.t_end, train.name)
            for train in self._original_trains
        ]
        self._neuron_mask = None
        self._neighbors = None
        return self

    def resample(self, sampleRate: float) -> _SpikeTrainCollection:
        if sampleRate <= 0.0:
            raise ValueError("sampleRate must be positive")
        dt = 1.0 / float(sampleRate)
        for train in self.trains:
            snapped = np.round(train.spike_times / dt) * dt
            train.spike_times = np.unique(snapped)
        return self

    def enforceSampleRate(self, sampleRate: float) -> _SpikeTrainCollection:
        return self.resample(sampleRate)

    def ensureConsistancy(self) -> bool:
        for train in self.trains:
            if np.any(np.diff(train.spike_times) < 0.0):
                return False
            if np.any(train.spike_times < train.t_start):
                return False
            if train.t_end is not None and np.any(train.spike_times > train.t_end):
                return False
        return True

    def estimateVarianceAcrossTrials(self, binSize_s: float = 0.01) -> np.ndarray:
        _t, mat = self.to_binned_matrix(bin_size_s=binSize_s, mode="count")
        if mat.size == 0:
            return np.array([], dtype=float)
        return np.var(mat, axis=0)

    def plotISIHistogram(self, bins: int = 20) -> Any:
        import matplotlib.pyplot as plt

        isi = [np.diff(train.spike_times) for train in self.trains if train.spike_times.size > 1]
        if not isi:
            return plt.hist([], bins=bins)
        return plt.hist(np.concatenate(isi), bins=bins, color="tab:blue", alpha=0.6)

    def plotExponentialFit(self) -> Any:
        import matplotlib.pyplot as plt

        isi = [np.diff(train.spike_times) for train in self.trains if train.spike_times.size > 1]
        if not isi:
            return plt.plot([], [])
        vals = np.concatenate(isi)
        lam = 1.0 / max(float(np.mean(vals)), 1e-12)
        x = np.linspace(0.0, float(np.max(vals)), 200)
        y = lam * np.exp(-lam * x)
        return plt.plot(x, y, "r-")

    def psthGLM(self, binSize_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        return self.psth(binSize_s=binSize_s)

    def ssglm(self, binSize_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        return self.psth(binSize_s=binSize_s)

    @staticmethod
    def generateUnitImpulseBasis(basisWidth_s: float, *args: Any, **kwargs: Any) -> _Covariate:
        if basisWidth_s <= 0.0:
            raise ValueError("basisWidth_s must be positive")

        name = str(kwargs.pop("name", "unit_impulse_basis"))
        numeric_types = (int, float, np.integer, np.floating)

        # MATLAB-compatible signatures:
        #   generateUnitImpulseBasis(basisWidth, sampleRate[, totalTime[, name]])
        #   generateUnitImpulseBasis(basisWidth, minTime, maxTime, sampleRate[, name])
        if len(args) >= 3 and isinstance(args[2], numeric_types):
            min_time_s = float(args[0])
            max_time_s = float(args[1])
            sample_rate_hz = float(args[2])
            if len(args) >= 4:
                name = str(args[3])
        else:
            sample_rate_hz = float(args[0]) if len(args) >= 1 else float(kwargs.pop("sampleRate_hz", 1000.0))
            total_time_s = float(args[1]) if len(args) >= 2 else float(kwargs.pop("totalTime_s", 1.0))
            min_time_s = 0.0
            max_time_s = total_time_s

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"unexpected keyword arguments: {unknown}")
        if sample_rate_hz <= 0.0:
            raise ValueError("sampleRate_hz must be positive")
        if max_time_s <= min_time_s:
            raise ValueError("maxTime must be greater than minTime")

        dt = 1.0 / sample_rate_hz
        time = np.arange(min_time_s, max_time_s + 0.5 * dt, dt)
        total_time_s = max_time_s - min_time_s
        n_basis = max(1, int(np.ceil(total_time_s / float(basisWidth_s))))
        basis = np.zeros((time.size, n_basis), dtype=float)
        for j in range(n_basis):
            lo = min_time_s + j * basisWidth_s
            hi = min(min_time_s + (j + 1) * basisWidth_s, max_time_s + dt)
            mask = (time >= lo) & (time < hi)
            basis[mask, j] = 1.0
        labels = [f"basis_{j+1}" for j in range(n_basis)]
        return Covariate(time=time, data=basis, name=name, labels=labels)

    def getEnsembleNeuronCovariates(self, binSize_s: float = 0.001, mode: Literal["binary", "count"] = "binary") -> "CovColl":
        t, mat = self.to_binned_matrix(bin_size_s=binSize_s, mode=mode)
        covs: list[_Covariate] = []
        for i in range(mat.shape[0]):
            covs.append(
                Covariate(
                    time=t.copy(),
                    data=mat[i, :].copy(),
                    name=self.trains[i].name,
                    labels=[self.trains[i].name],
                    units="spikes/bin",
                )
            )
        return CovColl(cast(list[_Covariate], covs))

    def addNeuronNamesToEnsCovColl(self, covColl: "CovColl") -> "CovColl":
        for i, cov in enumerate(covColl.covariates):
            if i < len(self.trains):
                cov.name = self.trains[i].name
                cov.labels = [self.trains[i].name]
        return covColl


class CovColl(_CovariateCollection):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._original_covariates = [
            _Covariate(
                time=cov.time.copy(),
                data=cov.data.copy(),
                name=str(cov.name),
                units=str(cov.units),
                labels=list(cov.labels),
                x_label=str(cov.x_label),
                y_label=str(cov.y_label),
                x_units=str(cov.x_units),
                y_units=str(cov.y_units),
                plot_props=dict(cov.plot_props),
            )
            for cov in self.covariates
        ]
        self._cov_shift = 0.0

    @staticmethod
    def containsChars(text: str, chars: str | list[str]) -> bool:
        if isinstance(chars, str):
            chars_list = list(chars)
        else:
            chars_list = [str(v) for v in chars]
        return any(ch in text for ch in chars_list)

    @staticmethod
    def isaSelectorCell(selector: Any) -> bool:
        if not isinstance(selector, (list, tuple, np.ndarray)):
            return False
        vals = list(np.asarray(selector, dtype=object).reshape(-1))
        return all(isinstance(v, (int, str, np.integer)) for v in vals)

    def getTime(self) -> np.ndarray:
        return self.time

    def getDesignMatrix(self) -> tuple[np.ndarray, list[str]]:
        return self.design_matrix()

    def copy(self) -> "CovColl":
        copied = super().copy()
        return CovColl(copied.covariates)

    def addToColl(self, cov: _Covariate) -> "CovColl":
        self.add_to_coll(cov)
        return self

    def getCov(self, selector: int | str) -> _Covariate:
        return self.get_cov(selector)

    def getCovIndicesFromNames(self, names: list[str]) -> list[int]:
        return self.get_cov_indices_from_names(names)

    def getCovIndFromName(self, name: str) -> int:
        return self.get_cov_ind_from_name(name)

    def isCovPresent(self, name: str) -> bool:
        return self.is_cov_present(name)

    def getCovDimension(self) -> int:
        return self.get_cov_dimension()

    def getAllCovLabels(self) -> list[str]:
        return self.get_all_cov_labels()

    def nActCovar(self) -> int:
        return self.n_act_covar()

    def numActCov(self) -> int:
        return self.num_act_cov()

    def sumDimensions(self) -> int:
        return self.sum_dimensions()

    def dataToMatrix(self) -> tuple[np.ndarray, list[str]]:
        return self.data_to_matrix()

    def dataToMatrixFromNames(self, names: list[str]) -> tuple[np.ndarray, list[str]]:
        return self.data_to_matrix_from_names(names)

    def dataToMatrixFromSel(self, selectors: list[int]) -> tuple[np.ndarray, list[str]]:
        return self.data_to_matrix_from_sel(selectors)

    def parseDataSelectorArray(self, selector: Any) -> list[int]:
        vals = list(np.asarray(selector, dtype=object).reshape(-1))
        if not vals:
            return []
        if all(isinstance(v, (str, np.str_)) for v in vals):
            return self.get_cov_indices_from_names([str(v) for v in vals])
        out: list[int] = []
        for v in vals:
            idx = int(v)
            if idx >= 1 and idx <= len(self.covariates):
                out.append(idx - 1)
            elif idx >= 0 and idx < len(self.covariates):
                out.append(idx)
            else:
                raise IndexError("selector index out of range")
        return out

    def covIndFromSelector(self, selector: int | str | list[int] | list[str] | np.ndarray) -> list[int]:
        if isinstance(selector, (int, str)):
            return self.parseDataSelectorArray([selector])
        return self.parseDataSelectorArray(selector)

    def getCovMaskFromSelector(self, selector: int | str | list[int] | list[str] | np.ndarray) -> list[int]:
        return self.covIndFromSelector(selector)

    def flattenCovMask(self, mask: list[int] | np.ndarray | list[list[int]]) -> list[int]:
        arr = np.asarray(mask, dtype=int).reshape(-1)
        return [int(v) for v in arr]

    def generateRemainingIndex(self, selector: int | str | list[int] | list[str] | np.ndarray) -> list[int]:
        masked = set(self.covIndFromSelector(selector))
        return [i for i in range(len(self.covariates)) if i not in masked]

    def generateSelectorCell(self, mask: list[int] | np.ndarray) -> list[str]:
        idx = self.flattenCovMask(mask)
        return [self.covariates[i].name for i in idx]

    def getSelectorFromMasks(self, mask: list[int] | np.ndarray) -> list[str]:
        return self.generateSelectorCell(mask)

    def addSingleCovToColl(self, cov: _Covariate) -> "CovColl":
        return self.addToColl(cov)

    def addCovCellToColl(self, covariates: list[_Covariate]) -> "CovColl":
        for cov in covariates:
            self.add_to_coll(cov)
        return self

    def addCovCollection(self, other: _CovariateCollection) -> "CovColl":
        for cov in other.covariates:
            self.add_to_coll(cov)
        return self

    def setMinTime(self, t_min: float) -> "CovColl":
        for cov in self.covariates:
            cov.set_min_time(t_min)
        return self

    def setMaxTime(self, t_max: float) -> "CovColl":
        for cov in self.covariates:
            cov.set_max_time(t_max)
        return self

    def restrictToTimeWindow(self, t_min: float, t_max: float) -> "CovColl":
        self.setMinTime(t_min)
        self.setMaxTime(t_max)
        return self

    def setSampleRate(self, sampleRate: float) -> "CovColl":
        resampled_covariates: list[_Covariate] = []
        for cov in self.covariates:
            out = cov.resample(sampleRate)
            resampled_covariates.append(
                _Covariate(
                    time=out.time,
                    data=out.data,
                    name=out.name,
                    units=out.units,
                    labels=getattr(cov, "labels", [out.name]),
                    x_label=out.x_label,
                    y_label=out.y_label,
                    x_units=out.x_units,
                    y_units=out.y_units,
                    plot_props=out.plot_props,
                )
            )
        self.covariates = resampled_covariates
        return self

    def resample(self, sampleRate: float) -> "CovColl":
        return self.setSampleRate(sampleRate)

    def enforceSampleRate(self, sampleRate: float) -> "CovColl":
        return self.setSampleRate(sampleRate)

    def updateTimes(self) -> "CovColl":
        _ = self.time
        return self

    def toStructure(self) -> dict[str, Any]:
        return {"covariates": [cov.to_structure() for cov in self.covariates]}

    def dataToStructure(self) -> dict[str, Any]:
        return self.toStructure()

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> "CovColl":
        rows = payload.get("covariates", [])
        covs = [Covariate.fromStructure(row) for row in rows]
        if not covs:
            raise ValueError("fromStructure requires at least one covariate")
        return CovColl(cast(list[_Covariate], covs))

    def setMask(self, selector: list[int] | list[str]) -> "CovColl":
        if selector and isinstance(selector[0], str):
            idx = self.get_cov_indices_from_names(cast(list[str], selector))
        else:
            idx = [int(i) for i in cast(list[int], selector)]
        self._cov_mask = idx
        return self

    def resetMask(self) -> "CovColl":
        self._cov_mask = list(range(len(self.covariates)))
        return self

    def setMasksFromSelector(self, selector: list[int] | list[str] | np.ndarray) -> "CovColl":
        if isinstance(selector, np.ndarray):
            vals = selector.reshape(-1).tolist()
        else:
            vals = selector
        return self.setMask(cast(list[int] | list[str], vals))

    def isCovMaskSet(self) -> bool:
        return hasattr(self, "_cov_mask")

    def getCovDataMask(self) -> list[int]:
        if not hasattr(self, "_cov_mask"):
            return list(range(len(self.covariates)))
        return list(self._cov_mask)

    def getCovLabelsFromMask(self) -> list[str]:
        return [self.covariates[i].name for i in self.getCovDataMask()]

    def removeCovariate(self, selector: int | str) -> "CovColl":
        if isinstance(selector, int):
            del self.covariates[selector]
            return self
        idx = self.get_cov_ind_from_name(selector)
        del self.covariates[idx]
        return self

    def removeFromColl(self, selector: int | str) -> "CovColl":
        return self.removeCovariate(selector)

    def removeFromCollByIndices(self, indices: list[int]) -> "CovColl":
        for i in sorted(set(indices), reverse=True):
            del self.covariates[i]
        return self

    def maskAwayCov(self, selector: int | str | list[int] | list[str] | np.ndarray) -> "CovColl":
        remaining = self.generateRemainingIndex(selector)
        self._cov_mask = remaining
        return self

    def maskAwayOnlyCov(self, selector: int | str | list[int] | list[str] | np.ndarray) -> "CovColl":
        return self.maskAwayCov(selector)

    def maskAwayAllExcept(self, selector: int | str | list[int] | list[str] | np.ndarray) -> "CovColl":
        self._cov_mask = self.covIndFromSelector(selector)
        return self

    def setCovShift(self, shift_s: float) -> "CovColl":
        shift = float(shift_s)
        self._cov_shift += shift
        for cov in self.covariates:
            cov.time = cov.time + shift
        return self

    def resetCovShift(self) -> "CovColl":
        if self._cov_shift == 0.0:
            return self
        for cov in self.covariates:
            cov.time = cov.time - self._cov_shift
        self._cov_shift = 0.0
        return self

    def restoreToOriginal(self) -> "CovColl":
        self.covariates = [
            _Covariate(
                time=cov.time.copy(),
                data=cov.data.copy(),
                name=str(cov.name),
                units=str(cov.units),
                labels=list(cov.labels),
                x_label=str(cov.x_label),
                y_label=str(cov.y_label),
                x_units=str(cov.x_units),
                y_units=str(cov.y_units),
                plot_props=dict(cov.plot_props),
            )
            for cov in self._original_covariates
        ]
        self._cov_shift = 0.0
        self.resetMask()
        return self

    def findMinTime(self) -> float:
        return self.find_min_time()

    def findMaxTime(self) -> float:
        return self.find_max_time()

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        X, _labels = self.design_matrix()
        return plt.plot(self.time, X)


class TrialConfig(_TrialConfig):
    def __init__(
        self,
        covariateLabels: list[str] | None = None,
        Fs: float = 1000.0,
        fitType: str = "poisson",
        name: str = "config",
        **kwargs: Any,
    ) -> None:
        covariate_labels = kwargs.pop("covariate_labels", covariateLabels or [])
        sample_rate_hz = kwargs.pop("sample_rate_hz", Fs)
        fit_type = kwargs.pop("fit_type", fitType)
        super().__init__(
            covariate_labels=covariate_labels,
            sample_rate_hz=sample_rate_hz,
            fit_type=fit_type,
            name=name,
        )

    def getFitType(self) -> str:
        return self.fit_type

    def getSampleRate(self) -> float:
        return self.sample_rate_hz

    def getCovariateLabels(self) -> list[str]:
        return self.covariate_labels

    def getName(self) -> str:
        return self.name

    def setName(self, name: str) -> "TrialConfig":
        self.name = str(name)
        return self

    def toStructure(self) -> dict[str, Any]:
        return {
            "covMask": list(self.covariate_labels),
            "sampleRate": float(self.sample_rate_hz),
            "history": [],
            "ensCovHist": [],
            "ensCovMask": [],
            "covLag": [],
            "name": self.name,
        }

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> "TrialConfig":
        return TrialConfig(
            covariateLabels=list(payload.get("covMask", [])),
            Fs=float(payload.get("sampleRate", 1000.0)),
            name=str(payload.get("name", "config")),
        )

    def setConfig(self, trial: "Trial") -> "TrialConfig":
        if self.sample_rate_hz > 0.0:
            trial.setSampleRate(self.sample_rate_hz)
        if self.covariate_labels:
            trial.setCovMask(self.covariate_labels)
        return self


class ConfigColl(_ConfigCollection):
    @staticmethod
    def ConfigColl(*args: Any, **kwargs: Any) -> _ConfigCollection:
        if len(args) == 1 and isinstance(args[0], dict):
            return ConfigColl.fromStructure(args[0])
        return ConfigColl(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any] | list[dict[str, Any]]) -> _ConfigCollection:
        if isinstance(payload, dict):
            entries = list(payload.get("configs", []))
        else:
            entries = list(payload)
        if not entries:
            raise ValueError("fromStructure requires at least one configuration entry")
        configs = cast(list[_TrialConfig], [TrialConfig.fromStructure(entry) for entry in entries])
        return ConfigColl(configs)

    def toStructure(self) -> dict[str, Any]:
        return {
            "configs": [
                TrialConfig(
                    covariateLabels=list(cfg.covariate_labels),
                    Fs=float(cfg.sample_rate_hz),
                    fitType=str(cfg.fit_type),
                    name=str(cfg.name),
                ).toStructure()
                for cfg in self.configs
            ]
        }

    def addConfig(self, config: _TrialConfig) -> _ConfigCollection:
        self.configs.append(config)
        return self

    def getConfig(self, selector: int | str = 1) -> _TrialConfig:
        if isinstance(selector, str):
            for cfg in self.configs:
                if cfg.name == selector:
                    return cfg
            raise KeyError(f"configuration '{selector}' not found")
        idx = int(selector)
        if idx < 1 or idx > len(self.configs):
            raise IndexError("configuration index out of range")
        return self.configs[idx - 1]

    def setConfig(self, selector: int | str, config: _TrialConfig) -> _ConfigCollection:
        if isinstance(selector, str):
            for i, cfg in enumerate(self.configs):
                if cfg.name == selector:
                    self.configs[i] = config
                    return self
            raise KeyError(f"configuration '{selector}' not found")
        idx = int(selector)
        if idx < 1 or idx > len(self.configs):
            raise IndexError("configuration index out of range")
        self.configs[idx - 1] = config
        return self

    def getConfigNames(self) -> list[str]:
        return [cfg.name for cfg in self.configs]

    def setConfigNames(self, names: list[str]) -> _ConfigCollection:
        if len(names) != len(self.configs):
            raise ValueError("names length must match number of configs")
        for cfg, name in zip(self.configs, names):
            cfg.name = str(name)
        return self

    def getSubsetConfigs(self, selectors: list[int] | np.ndarray) -> _ConfigCollection:
        inds = [int(v) for v in np.asarray(selectors).reshape(-1)]
        subset: list[_TrialConfig] = []
        for idx in inds:
            if idx < 1 or idx > len(self.configs):
                raise IndexError("configuration index out of range")
            subset.append(self.configs[idx - 1])
        return ConfigColl(subset)

    def getConfigs(self) -> list[_TrialConfig]:
        return self.configs


class Trial(_Trial):
    def _ensure_trial_state(self) -> None:
        if not hasattr(self, "_orig_spikes"):
            self._orig_spikes = nstColl(self.spikes.trains).copy()
            self._orig_covariates = CovColl(self.covariates.covariates).copy()
        if not hasattr(self, "_neuron_mask"):
            self._neuron_mask = list(range(self.spikes.n_units))
        if not hasattr(self, "_ens_cov_mask"):
            self._ens_cov_mask = list(range(self.spikes.n_units))
        if not hasattr(self, "_cov_mask"):
            self._cov_mask = list(range(len(self.covariates.covariates)))
        if not hasattr(self, "_history"):
            self._history = None
        if not hasattr(self, "_ens_cov_hist"):
            self._ens_cov_hist = None
        if not hasattr(self, "_neighbors"):
            self._neighbors = None

    def getAlignedBinnedObservation(
        self,
        binSize_s: float,
        unitIndex: int = 0,
        mode: Literal["binary", "count"] = "binary",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.aligned_binned_observation(bin_size_s=binSize_s, unit_index=unitIndex, mode=mode)

    def getDesignMatrix(self) -> tuple[np.ndarray, list[str]]:
        return self.get_design_matrix()

    def getSpikeVector(
        self,
        binSize_s: float,
        unitIndex: int = 0,
        mode: Literal["binary", "count"] = "binary",
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.get_spike_vector(bin_size_s=binSize_s, unit_index=unitIndex, mode=mode)

    def getCov(self, selector: int | str) -> _Covariate:
        return self.get_cov(selector)

    def getNeuron(self, unitInd: int = 0) -> nstColl:
        return nstColl(self.get_neuron(unit_index=unitInd).trains)

    def getAllCovLabels(self) -> list[str]:
        return self.get_all_cov_labels()

    def findMinTime(self) -> float:
        return self.find_min_time()

    def findMaxTime(self) -> float:
        return self.find_max_time()

    def findMinSampleRate(self) -> float:
        return self.find_min_sample_rate()

    def findMaxSampleRate(self) -> float:
        return self.find_max_sample_rate()

    def isSampleRateConsistent(self) -> bool:
        return self.is_sample_rate_consistent()

    def addCov(self, cov: _Covariate) -> "Trial":
        self.covariates.add_to_coll(cov)
        return self

    def removeCov(self, selector: int | str) -> "Trial":
        cc = CovColl(self.covariates.covariates)
        cc.removeCovariate(selector)
        self.covariates = _CovariateCollection(cc.covariates)
        return self

    def setMinTime(self, t_min: float) -> "Trial":
        self.spikes.set_min_time(t_min)
        for cov in self.covariates.covariates:
            cov.set_min_time(t_min)
        return self

    def setMaxTime(self, t_max: float) -> "Trial":
        self.spikes.set_max_time(t_max)
        for cov in self.covariates.covariates:
            cov.set_max_time(t_max)
        return self

    def setSampleRate(self, sampleRate: float) -> "Trial":
        resampled_covariates: list[_Covariate] = []
        for cov in self.covariates.covariates:
            out = cov.resample(sampleRate)
            resampled_covariates.append(
                _Covariate(
                    time=out.time,
                    data=out.data,
                    name=out.name,
                    units=out.units,
                    labels=getattr(cov, "labels", [out.name]),
                    x_label=out.x_label,
                    y_label=out.y_label,
                    x_units=out.x_units,
                    y_units=out.y_units,
                    plot_props=out.plot_props,
                )
            )
        self.covariates.covariates = resampled_covariates
        return self

    def resample(self, sampleRate: float) -> "Trial":
        return self.setSampleRate(sampleRate)

    def shiftCovariates(self, lag_s: float) -> "Trial":
        for cov in self.covariates.covariates:
            cov.shift_time(lag_s)
        return self

    def getNeuronNames(self) -> list[str]:
        return self.spikes.get_nst_names()

    def getUniqueNeuronNames(self) -> list[str]:
        return self.spikes.get_unique_nst_names()

    def getNumUniqueNeurons(self) -> int:
        return len(self.getUniqueNeuronNames())

    def getNeuronIndFromName(self, name: str) -> list[int]:
        return self.spikes.get_nst_indices_from_name(name)

    def setTrialPartition(self, partition: dict[str, tuple[float, float]]) -> "Trial":
        self._trial_partition = partition
        return self

    def getTrialPartition(self) -> dict[str, tuple[float, float]]:
        return dict(getattr(self, "_trial_partition", {}))

    def updateTimePartitions(self) -> "Trial":
        if hasattr(self, "_trial_partition"):
            t0 = self.findMinTime()
            tf = self.findMaxTime()
            self._trial_partition = {
                k: (max(t0, float(v[0])), min(tf, float(v[1])))
                for k, v in self._trial_partition.items()
            }
        return self

    def setCovMask(self, selector: list[int] | list[str]) -> "Trial":
        cc = CovColl(self.covariates.covariates)
        cc.setMask(selector)
        self._cov_mask = cc.getCovDataMask()
        return self

    def resetCovMask(self) -> "Trial":
        self._cov_mask = list(range(len(self.covariates.covariates)))
        return self

    def isCovMaskSet(self) -> bool:
        return hasattr(self, "_cov_mask")

    def getCovLabelsFromMask(self) -> list[str]:
        mask = getattr(self, "_cov_mask", list(range(len(self.covariates.covariates))))
        return [self.covariates.covariates[i].name for i in mask]

    def setTrialEvents(self, events: _Events) -> "Trial":
        self._events = events
        return self

    def getEvents(self) -> _Events | None:
        return getattr(self, "_events", None)

    def flattenCovMask(self, mask: list[int] | list[list[int]] | np.ndarray) -> list[int]:
        out = np.asarray(mask, dtype=int).reshape(-1)
        return [int(v) for v in out]

    def flattenMask(self, mask: list[int] | list[list[int]] | np.ndarray) -> list[int]:
        return self.flattenCovMask(mask)

    def getLabelsFromMask(self) -> list[str]:
        self._ensure_trial_state()
        mask = getattr(self, "_cov_mask", list(range(len(self.covariates.covariates))))
        return [self.covariates.covariates[i].name for i in mask]

    def getCovSelectorFromMask(self) -> list[str]:
        return self.getLabelsFromMask()

    def getEnsembleNeuronCovariates(
        self,
        binSize_s: float = 0.001,
        mode: Literal["binary", "count"] = "binary",
    ) -> CovColl:
        return nstColl(self.spikes.trains).getEnsembleNeuronCovariates(binSize_s=binSize_s, mode=mode)

    def getEnsCovLabels(self, binSize_s: float = 0.001) -> list[str]:
        coll = self.getEnsembleNeuronCovariates(binSize_s=binSize_s, mode="binary")
        return coll.getAllCovLabels()

    def setEnsCovMask(self, selector: list[int] | list[str] | np.ndarray) -> "Trial":
        self._ensure_trial_state()
        ens = self.getEnsembleNeuronCovariates()
        idx = ens.covIndFromSelector(selector)
        self._ens_cov_mask = idx
        return self

    def resetEnsCovMask(self) -> "Trial":
        self._ensure_trial_state()
        self._ens_cov_mask = list(range(self.spikes.n_units))
        return self

    def getEnsCovLabelsFromMask(self, binSize_s: float = 0.001) -> list[str]:
        self._ensure_trial_state()
        labels = self.getEnsCovLabels(binSize_s=binSize_s)
        return [labels[i] for i in self._ens_cov_mask if i < len(labels)]

    def getEnsCovMatrix(
        self,
        binSize_s: float = 0.001,
        mode: Literal["binary", "count"] = "binary",
    ) -> tuple[np.ndarray, list[str]]:
        self._ensure_trial_state()
        ens = self.getEnsembleNeuronCovariates(binSize_s=binSize_s, mode=mode)
        X, labels = ens.dataToMatrix()
        if self._ens_cov_mask:
            X = X[:, self._ens_cov_mask]
            labels = [labels[i] for i in self._ens_cov_mask]
        return X, labels

    def setEnsCovHist(self, ensCovHist: Any) -> "Trial":
        self._ensure_trial_state()
        self._ens_cov_hist = ensCovHist
        return self

    def isEnsCovHistSet(self) -> bool:
        self._ensure_trial_state()
        return self._ens_cov_hist is not None

    def setHistory(self, history: Any) -> "Trial":
        self._ensure_trial_state()
        self._history = history
        return self

    def isHistSet(self) -> bool:
        self._ensure_trial_state()
        return self._history is not None

    def resetHistory(self) -> "Trial":
        self._ensure_trial_state()
        self._history = None
        self._ens_cov_hist = None
        return self

    def getNumHist(self) -> int:
        self._ensure_trial_state()
        if self._history is None:
            return 0
        if isinstance(self._history, (list, tuple)):
            return len(self._history)
        return 1

    def getHistLabels(self) -> list[str]:
        n_hist = self.getNumHist()
        return [f"hist_{i+1}" for i in range(n_hist)]

    def getHistMatrices(self, binSize_s: float = 0.001) -> list[np.ndarray]:
        self._ensure_trial_state()
        if self._history is None:
            return []
        history_obj = self._history
        if isinstance(history_obj, (list, tuple)):
            history_obj = history_obj[0] if history_obj else None
        if history_obj is None or not hasattr(history_obj, "design_matrix"):
            return []
        t_bins, _ = self.spikes.to_binned_matrix(bin_size_s=binSize_s, mode="binary")
        mats: list[np.ndarray] = []
        for train in self.spikes.trains:
            mats.append(history_obj.design_matrix(spike_times_s=train.spike_times, time_grid_s=t_bins))
        return mats

    def getHistForNeurons(self, neuronIndices: list[int] | np.ndarray, binSize_s: float = 0.001) -> list[np.ndarray]:
        mats = self.getHistMatrices(binSize_s=binSize_s)
        if not mats:
            return []
        inds = [int(v) for v in np.asarray(neuronIndices).reshape(-1)]
        out: list[np.ndarray] = []
        for i in inds:
            j = i - 1 if i >= 1 else i
            if j < 0 or j >= len(mats):
                raise IndexError("neuron index out of range")
            out.append(mats[j])
        return out

    def setNeuronMask(self, selector: list[int] | list[str] | np.ndarray) -> "Trial":
        self._ensure_trial_state()
        if isinstance(selector, np.ndarray):
            vals = selector.reshape(-1).tolist()
        else:
            vals = list(selector)
        if vals and all(isinstance(v, (str, np.str_)) for v in vals):
            idx = [self.spikes.get_nst_indices_from_name(str(v))[0] for v in vals]
        else:
            idx_raw = [int(v) for v in vals]
            idx = []
            for v in idx_raw:
                if v >= 1 and v <= self.spikes.n_units:
                    idx.append(v - 1)
                elif v >= 0 and v < self.spikes.n_units:
                    idx.append(v)
                else:
                    raise IndexError("neuron index out of range")
        self._neuron_mask = sorted(set(idx))
        return self

    def resetNeuronMask(self) -> "Trial":
        self._ensure_trial_state()
        self._neuron_mask = list(range(self.spikes.n_units))
        return self

    def isNeuronMaskSet(self) -> bool:
        self._ensure_trial_state()
        return self._neuron_mask != list(range(self.spikes.n_units))

    def getNeuronIndFromMask(self) -> list[int]:
        self._ensure_trial_state()
        return list(self._neuron_mask)

    def setNeighbors(self, neighbors: Any) -> "Trial":
        self._ensure_trial_state()
        self._neighbors = neighbors
        return self

    def getNeuronNeighbors(self) -> Any:
        self._ensure_trial_state()
        return self._neighbors

    def isMaskSet(self) -> bool:
        self._ensure_trial_state()
        cov_set = self._cov_mask != list(range(len(self.covariates.covariates)))
        ens_set = self._ens_cov_mask != list(range(self.spikes.n_units))
        neu_set = self._neuron_mask != list(range(self.spikes.n_units))
        return bool(cov_set or ens_set or neu_set)

    def getAllLabels(self, binSize_s: float = 0.001) -> list[str]:
        labels: list[str] = []
        labels.extend(self.getAllCovLabels())
        labels.extend(self.getEnsCovLabels(binSize_s=binSize_s))
        labels.extend(self.getHistLabels())
        return labels

    def makeConsistentSampleRate(self, sampleRate: float | None = None) -> "Trial":
        if sampleRate is None:
            sampleRate = float(self.findMinSampleRate())
        return self.setSampleRate(float(sampleRate))

    def makeConsistentTime(self) -> "Trial":
        t0 = self.findMinTime()
        tf = self.findMaxTime()
        return self.setMinTime(t0).setMaxTime(tf)

    def resampleEnsColl(self, sampleRate: float) -> CovColl:
        ens = self.getEnsembleNeuronCovariates()
        return ens.resample(sampleRate)

    def restoreToOriginal(self) -> "Trial":
        self._ensure_trial_state()
        self.spikes = nstColl(self._orig_spikes.trains).copy()
        self.covariates = CovColl(self._orig_covariates.covariates).copy()
        self._cov_mask = list(range(len(self.covariates.covariates)))
        self._ens_cov_mask = list(range(self.spikes.n_units))
        self._neuron_mask = list(range(self.spikes.n_units))
        self._history = None
        self._ens_cov_hist = None
        self._neighbors = None
        return self

    def setTrialTimesFor(self, *args: Any) -> "Trial":
        if len(args) == 2:
            t0, tf = float(args[0]), float(args[1])
        elif len(args) == 3:
            t0, tf = float(args[1]), float(args[2])
        else:
            raise ValueError("setTrialTimesFor expects (t0, tf) or (unitIndex, t0, tf)")
        self.setMinTime(t0)
        self.setMaxTime(tf)
        return self

    def plotCovariates(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        X, _ = self.get_design_matrix()
        return plt.plot(self.covariates.time, X)

    def plotRaster(self, *_args: Any, **_kwargs: Any) -> Any:
        return nstColl(self.spikes.trains).plot()

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.plotRaster()

    def toStructure(self) -> dict[str, Any]:
        return {
            "spikes": nstColl(self.spikes.trains).toStructure(),
            "covariates": CovColl(self.covariates.covariates).toStructure(),
            "trial_partition": self.getTrialPartition(),
        }

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> "Trial":
        spikes = nstColl.fromStructure(payload["spikes"])
        covs = CovColl.fromStructure(payload["covariates"])
        trial = Trial(spikes=spikes, covariates=covs)
        if "trial_partition" in payload:
            trial.setTrialPartition(dict(payload["trial_partition"]))
        return trial


class CIF(_CIFModel):
    @staticmethod
    def CIF(*args: Any, **kwargs: Any) -> _CIFModel:
        if len(args) == 1 and isinstance(args[0], dict):
            return CIF.fromStructure(args[0])
        return CIF(*args, **kwargs)

    def CIFCopy(self) -> _CIFModel:
        return CIF(coefficients=self.coefficients.copy(), intercept=float(self.intercept), link=str(self.link))

    def evalLambda(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X)

    def computeLinearPredictor(self, X: np.ndarray) -> np.ndarray:
        return self.linear_predictor(X)

    def logLikelihood(self, y: np.ndarray, X: np.ndarray, dt: float = 1.0) -> float:
        return self.log_likelihood(y=y, X=X, dt=dt)

    def simulateByThinning(
        self,
        time: np.ndarray,
        X: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return self.simulate_by_thinning(time=time, X=X, rng=rng)

    def evalLambdaDelta(self, X: np.ndarray, dt: float = 1.0) -> np.ndarray:
        return self.eval_lambda_delta(X=X, dt=dt)

    def computePlotParams(self, X: np.ndarray) -> dict[str, float]:
        return self.compute_plot_params(X)

    def evalFunctionWithVectorArgs(self, X: np.ndarray, *_args: Any, **_kwargs: Any) -> np.ndarray:
        return self.evaluate(X)

    def evalGradient(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        vals = self.evaluate(X)
        base = np.column_stack([np.ones(X.shape[0]), X])
        if self.link == "poisson":
            return base * vals[:, None]
        return base * (vals * (1.0 - vals))[:, None]

    def evalGradientLog(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        vals = self.evaluate(X)
        base = np.column_stack([np.ones(X.shape[0]), X])
        if self.link == "poisson":
            return base
        return base * (1.0 - vals)[:, None]

    def evalJacobian(self, X: np.ndarray) -> np.ndarray:
        return self.evalGradient(X)

    def evalJacobianLog(self, X: np.ndarray) -> np.ndarray:
        return self.evalGradientLog(X)

    def evalLDGamma(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X)

    def evalLogLDGamma(self, X: np.ndarray) -> np.ndarray:
        return np.log(np.clip(self.evaluate(X), 1e-12, None))

    def evalGradientLDGamma(self, X: np.ndarray) -> np.ndarray:
        return self.evalGradient(X)

    def evalGradientLogLDGamma(self, X: np.ndarray) -> np.ndarray:
        return self.evalGradientLog(X)

    def evalJacobianLDGamma(self, X: np.ndarray) -> np.ndarray:
        return self.evalJacobian(X)

    def evalJacobianLogLDGamma(self, X: np.ndarray) -> np.ndarray:
        return self.evalJacobianLog(X)

    def isSymBeta(self) -> bool:
        return False

    @staticmethod
    def resolveSimulinkModelName(*_args: Any, **_kwargs: Any) -> str:
        return "nstat_python_cif_model"

    def setHistory(self, history: Any) -> _CIFModel:
        setattr(self, "_history", history)
        return self

    def setSpikeTrain(self, spike_train: Any) -> _CIFModel:
        setattr(self, "_spike_train", spike_train)
        return self

    def simulateCIF(
        self,
        time: np.ndarray,
        X: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return self.simulate_by_thinning(time=time, X=X, rng=rng)

    def simulateCIFByThinning(
        self,
        time: np.ndarray,
        X: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return self.simulate_by_thinning(time=time, X=X, rng=rng)

    def toStructure(self) -> dict[str, np.ndarray | float | str]:
        return self.to_structure()

    @staticmethod
    def fromStructure(payload: dict[str, np.ndarray | float | str]) -> _CIFModel:
        out = _CIFModel.from_structure(payload)
        return CIF(coefficients=out.coefficients, intercept=out.intercept, link=out.link)

    @staticmethod
    def simulateCIFByThinningFromLambda(
        lambda_signal: _Covariate,
        numRealizations: int = 1,
        maxTimeRes: float | None = None,
    ) -> nstColl:
        """MATLAB-style helper accepting a `Covariate` lambda(t) signal."""

        time = np.asarray(lambda_signal.time, dtype=float)
        lam = np.asarray(lambda_signal.data, dtype=float)
        if lam.ndim != 1:
            raise ValueError("lambda_signal.data must be 1D for this helper")
        if maxTimeRes is not None and maxTimeRes > 0.0:
            t_new = np.arange(time[0], time[-1] + 0.5 * maxTimeRes, maxTimeRes)
            lam = np.interp(t_new, time, lam)
            time = t_new
        spikes = _CIFModel.simulate_cif_by_thinning_from_lambda(
            time=time, lambda_values=lam, num_realizations=numRealizations
        )
        t_start = float(time[0])
        t_end = float(time[-1])

        # Numeric roundoff in thinning can place the last event infinitesimally past
        # the terminal grid value. Clamp to the declared support before constructing
        # SpikeTrain so MATLAB-style helper remains robust.
        clipped_spikes = []
        for sp in spikes:
            sp_arr = np.asarray(sp, dtype=float)
            mask = (sp_arr >= t_start) & (sp_arr <= t_end)
            clipped_spikes.append(sp_arr[mask])
        trains = cast(
            list[_SpikeTrain],
            [
            nspikeTrain(spike_times=sp, t_start=t_start, t_end=t_end, name=f"unit_{i+1}")
            for i, sp in enumerate(clipped_spikes)
            ],
        )
        return nstColl(trains)


class Analysis:
    @staticmethod
    def fitGLM(
        X: np.ndarray,
        y: np.ndarray,
        fitType: str = "poisson",
        dt: float = 1.0,
        l2Penalty: float = 0.0,
    ) -> _FitResult:
        return _Analysis.fit_glm(X=X, y=y, fit_type=fitType, dt=dt, l2_penalty=l2Penalty)

    @staticmethod
    def fitTrial(trial: _Trial, config: _TrialConfig, unitIndex: int = 0) -> _FitResult:
        return _Analysis.fit_trial(trial=trial, config=config, unit_index=unitIndex)

    @staticmethod
    def GLMFit(
        X: np.ndarray,
        y: np.ndarray,
        fitType: str = "poisson",
        dt: float = 1.0,
        l2Penalty: float = 0.0,
    ) -> _FitResult:
        return _Analysis.glm_fit(X=X, y=y, fit_type=fitType, dt=dt, l2_penalty=l2Penalty)

    @staticmethod
    def RunAnalysisForNeuron(trial: _Trial, config: _TrialConfig, unitIndex: int = 0) -> _FitResult:
        return _Analysis.run_analysis_for_neuron(trial=trial, config=config, unit_index=unitIndex)

    @staticmethod
    def RunAnalysisForAllNeurons(trial: _Trial, config: _TrialConfig) -> list[_FitResult]:
        return _Analysis.run_analysis_for_all_neurons(trial=trial, config=config)

    @staticmethod
    def computeFitResidual(y: np.ndarray, X: np.ndarray, fit: _FitResult, dt: float = 1.0) -> np.ndarray:
        return _Analysis.compute_fit_residual(y=y, X=X, fit_result=fit, dt=dt)

    @staticmethod
    def computeInvGausTrans(y: np.ndarray, X: np.ndarray, fit: _FitResult, dt: float = 1.0) -> np.ndarray:
        return _Analysis.compute_inv_gaus_trans(y=y, X=X, fit_result=fit, dt=dt)

    @staticmethod
    def computeKSStats(transformed: np.ndarray) -> dict[str, float]:
        return _Analysis.compute_ks_stats(transformed)

    @staticmethod
    def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        return _Analysis.fdr_bh(p_values=p_values, alpha=alpha)

    @staticmethod
    def bnlrCG(
        X: np.ndarray,
        y: np.ndarray,
        dt: float = 1.0,
        l2Penalty: float = 0.0,
    ) -> _FitResult:
        return Analysis.fitGLM(X=X, y=y, fitType="binomial", dt=dt, l2Penalty=l2Penalty)

    @staticmethod
    def KSPlot(fit: Any, fitNum: int = 1) -> Any:
        _ = fitNum
        if hasattr(fit, "KSPlot"):
            return fit.KSPlot(fitNum)
        import matplotlib.pyplot as plt

        return plt.plot([], [])

    @staticmethod
    def compHistEnsCoeff(y: np.ndarray, X: np.ndarray, dt: float = 1.0) -> np.ndarray:
        fit = Analysis.fitGLM(X=X, y=y, fitType="poisson", dt=dt)
        return fit.coefficients

    @staticmethod
    def compHistEnsCoeffForAll(y_list: list[np.ndarray], X_list: list[np.ndarray], dt: float = 1.0) -> list[np.ndarray]:
        return [Analysis.compHistEnsCoeff(y=y, X=X, dt=dt) for y, X in zip(y_list, X_list)]

    @staticmethod
    def computeNeighbors(positions: np.ndarray, k: int = 1) -> np.ndarray:
        pts = np.asarray(positions, dtype=float)
        if pts.ndim != 2:
            raise ValueError("positions must be 2D [n_points, n_dims]")
        if k <= 0:
            raise ValueError("k must be positive")
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        return np.argsort(dist, axis=1)[:, :k]

    @staticmethod
    def flatMaskCellToMat(flatMaskCell: list[np.ndarray]) -> np.ndarray:
        rows = [np.asarray(row, dtype=float).reshape(-1) for row in flatMaskCell]
        if not rows:
            return np.zeros((0, 0), dtype=float)
        width = max(row.size for row in rows)
        out = np.zeros((len(rows), width), dtype=float)
        for i, row in enumerate(rows):
            out[i, : row.size] = row
        return out

    @staticmethod
    def computeHistLag(signal: np.ndarray, maxLag: int = 50) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(signal, dtype=float).reshape(-1)
        if y.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        y = y - np.mean(y)
        ac = np.correlate(y, y, mode="full")
        mid = ac.size // 2
        ac = ac[mid : mid + maxLag + 1]
        if ac[0] != 0:
            ac = ac / ac[0]
        lags = np.arange(ac.size, dtype=int)
        return lags, ac

    @staticmethod
    def computeHistLagForAll(signals: np.ndarray, maxLag: int = 50) -> tuple[np.ndarray, np.ndarray]:
        mat = np.asarray(signals, dtype=float)
        if mat.ndim == 1:
            return Analysis.computeHistLag(mat, maxLag=maxLag)
        curves = [Analysis.computeHistLag(row, maxLag=maxLag)[1] for row in mat]
        if not curves:
            return np.array([], dtype=int), np.array([], dtype=float)
        mean_curve = np.mean(np.vstack(curves), axis=0)
        lags = np.arange(mean_curve.size, dtype=int)
        return lags, mean_curve

    @staticmethod
    def computeGrangerCausalityMatrix(spikeMatrix: np.ndarray, maxLag: int = 1) -> np.ndarray:
        X = np.asarray(spikeMatrix, dtype=float)
        if X.ndim != 2:
            raise ValueError("spikeMatrix must be 2D [n_units, n_time]")
        n_units, n_time = X.shape
        if n_time <= maxLag:
            raise ValueError("n_time must be greater than maxLag")
        out = np.zeros((n_units, n_units), dtype=float)
        for i in range(n_units):
            y = X[i, maxLag:]
            yi = X[i, :-maxLag]
            for j in range(n_units):
                if i == j:
                    continue
                xj = X[j, :-maxLag]
                base_pred = yi
                full_pred = np.column_stack([yi, xj]) @ np.linalg.lstsq(
                    np.column_stack([yi, xj]),
                    y,
                    rcond=None,
                )[0]
                base_err = np.var(y - base_pred)
                full_err = np.var(y - full_pred)
                if base_err <= 1e-12:
                    out[i, j] = 0.0
                else:
                    out[i, j] = max(0.0, (base_err - full_err) / base_err)
        return out

    @staticmethod
    def ksdiscrete(sample: np.ndarray, reference: np.ndarray | None = None) -> dict[str, float]:
        x = np.asarray(sample, dtype=float).reshape(-1)
        if x.size == 0:
            return {"d_stat": 0.0, "n_events": 0.0}
        if reference is None:
            z = np.sort(x / max(np.max(x), 1e-12))
        else:
            ref = np.asarray(reference, dtype=float).reshape(-1)
            ref_sorted = np.sort(ref)
            z = np.searchsorted(ref_sorted, np.sort(x), side="right") / max(ref_sorted.size, 1)
        n = z.size
        ecdf = np.arange(1, n + 1, dtype=float) / float(n)
        d_plus = np.max(ecdf - z)
        d_minus = np.max(z - np.arange(0, n, dtype=float) / float(n))
        return {"d_stat": float(max(d_plus, d_minus)), "n_events": float(n)}

    @staticmethod
    def plotCoeffs(fit: Any) -> Any:
        if hasattr(fit, "plotCoeffs"):
            return fit.plotCoeffs()
        import matplotlib.pyplot as plt

        return plt.plot([], [])

    @staticmethod
    def plotFitResidual(y: np.ndarray, X: np.ndarray, fit: _FitResult, dt: float = 1.0) -> Any:
        import matplotlib.pyplot as plt

        resid = Analysis.computeFitResidual(y=y, X=X, fit=fit, dt=dt)
        x = np.arange(resid.size)
        return plt.plot(x, resid, "k-")

    @staticmethod
    def plotInvGausTrans(y: np.ndarray, X: np.ndarray, fit: _FitResult, dt: float = 1.0) -> Any:
        import matplotlib.pyplot as plt

        z = Analysis.computeInvGausTrans(y=y, X=X, fit=fit, dt=dt)
        x = np.arange(z.size)
        return plt.plot(x, z, "k-")

    @staticmethod
    def plotSeqCorr(residual: np.ndarray) -> Any:
        import matplotlib.pyplot as plt

        y = np.asarray(residual, dtype=float).reshape(-1)
        if y.size == 0:
            return plt.plot([], [])
        y = y - np.mean(y)
        corr = np.correlate(y, y, mode="full")
        corr = corr[corr.size // 2 :]
        if corr[0] != 0:
            corr = corr / corr[0]
        return plt.plot(np.arange(corr.size), corr, "k-")

    @staticmethod
    def spikeTrigAvg(
        signal: np.ndarray,
        spikeTimes_s: np.ndarray,
        timeGrid_s: np.ndarray,
        window_s: tuple[float, float] = (-0.05, 0.05),
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(signal, dtype=float).reshape(-1)
        t = np.asarray(timeGrid_s, dtype=float).reshape(-1)
        spikes = np.asarray(spikeTimes_s, dtype=float).reshape(-1)
        if x.size != t.size:
            raise ValueError("signal and timeGrid_s must have matching lengths")
        dt = float(np.median(np.diff(t)))
        n_pre = int(abs(window_s[0]) / dt)
        n_post = int(abs(window_s[1]) / dt)
        if n_pre + n_post + 1 <= 1:
            raise ValueError("window too small for sample rate")
        snippets: list[np.ndarray] = []
        for s in spikes:
            idx = int(np.argmin(np.abs(t - s)))
            lo = idx - n_pre
            hi = idx + n_post + 1
            if lo < 0 or hi > x.size:
                continue
            snippets.append(x[lo:hi])
        if not snippets:
            rel_t = np.arange(-n_pre, n_post + 1, dtype=float) * dt
            return rel_t, np.zeros(rel_t.size, dtype=float)
        mat = np.vstack(snippets)
        rel_t = np.arange(-n_pre, n_post + 1, dtype=float) * dt
        return rel_t, np.mean(mat, axis=0)


class FitResult(_FitResult):
    @staticmethod
    def FitResult(structure: dict[str, Any]) -> _FitResult:
        return FitResult.fromStructure(structure)

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> _FitResult:
        native = _FitResult.from_structure(structure)
        return FitResult(
            coefficients=native.coefficients,
            intercept=native.intercept,
            fit_type=native.fit_type,
            log_likelihood=native.log_likelihood,
            n_samples=native.n_samples,
            n_parameters=native.n_parameters,
            parameter_labels=native.parameter_labels,
            ks_stats=native.ks_stats,
            fit_residual=native.fit_residual,
            inv_gaus_stats=native.inv_gaus_stats,
            neuron_name=native.neuron_name,
            plot_params=native.plot_params,
        )

    @staticmethod
    def CellArrayToStructure(results: list[_FitResult]) -> list[dict[str, Any]]:
        return _FitResult.cell_array_to_structure(results)

    def toStructure(self) -> dict[str, Any]:
        return self.to_structure()

    def setKSStats(
        self,
        ksStat: np.ndarray | float | dict[str, Any],
        pValue: np.ndarray | float | None = None,
        withinConfInt: np.ndarray | float | None = None,
    ) -> _FitResult:
        self.set_ks_stats(ks_stat=ksStat, p_value=pValue, within_conf_int=withinConfInt)
        return self

    def setFitResidual(self, fitResidual: np.ndarray) -> _FitResult:
        self.set_fit_residual(fitResidual)
        return self

    def setInvGausStats(self, stats: dict[str, Any]) -> _FitResult:
        self.set_inv_gaus_stats(stats)
        return self

    def setNeuronName(self, neuronName: str) -> _FitResult:
        self.set_neuron_name(neuronName)
        return self

    def mapCovLabelsToUniqueLabels(self) -> list[str]:
        return self.map_cov_labels_to_unique_labels()

    def computePlotParams(self) -> dict[str, Any]:
        return self.compute_plot_params()

    def getPlotParams(self) -> dict[str, Any]:
        return self.get_plot_params()

    def addParamsToFit(self, payload: dict[str, Any]) -> _FitResult:
        self.add_params_to_fit(payload)
        return self

    def evalLambda(self, X_or_modelIndex: Any, maybe_X: Any = None) -> np.ndarray:
        if maybe_X is None:
            X = np.asarray(X_or_modelIndex, dtype=float)
        else:
            X = np.asarray(maybe_X, dtype=float)
        return self.predict(X)

    def getAIC(self) -> float:
        return self.aic()

    def getBIC(self) -> float:
        return self.bic()

    def asCIFModel(self) -> _CIFModel:
        return self.as_cif_model()

    def computeValLambda(self, X: np.ndarray) -> np.ndarray:
        return self.compute_val_lambda(X)

    def getCoeffs(self) -> np.ndarray:
        return self.get_coeffs()

    def getCoeffIndex(self, label: str) -> int:
        return self.get_coeff_index(label)

    def getParam(self, key: str) -> float | np.ndarray | str | int:
        return self.get_param(key)

    def getUniqueLabels(self) -> list[str]:
        return self.get_unique_labels()

    def isValDataPresent(self) -> bool:
        return len(self.xval_data) > 0 and len(self.xval_time) > 0

    def getSubsetFitResult(self, subfits: int | list[int] | np.ndarray) -> _FitResult:
        if isinstance(subfits, int):
            keep = [subfits]
        else:
            keep = [int(v) for v in np.asarray(subfits).reshape(-1)]
        if 1 not in keep:
            raise ValueError("single-fit Python adapter only supports subset index 1")
        return self

    def mergeResults(self, newFitObj: Any) -> _FitSummary:
        rows: list[_FitResult] = [self]
        if isinstance(newFitObj, _FitResult):
            rows.append(newFitObj)
        elif isinstance(newFitObj, (list, tuple)):
            for item in newFitObj:
                if not isinstance(item, _FitResult):
                    raise TypeError("mergeResults expects FitResult or list of FitResult")
                rows.append(item)
        else:
            raise TypeError("mergeResults expects FitResult or list of FitResult")
        return FitResSummary(rows)

    def getHistIndex(self, fitNum: int = 1, sortByEpoch: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
        _ = fitNum
        _ = sortByEpoch
        return np.array([], dtype=int), np.array([], dtype=int), 1

    def getHistCoeffs(self, fitNum: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        _ = fitNum
        empty = np.zeros((0, 1), dtype=float)
        return empty, [], empty

    def plotCoeffs(
        self,
        handle: Any = None,
        fitNum: int = 1,
        plotProps: dict[str, Any] | None = None,
        plotSignificance: bool = True,
        subIndex: Any = None,
    ) -> Any:
        _ = handle
        _ = fitNum
        _ = plotProps
        _ = plotSignificance
        _ = subIndex
        import matplotlib.pyplot as plt

        labels = self.get_unique_labels() or [f"coef_{i+1}" for i in range(self.coefficients.size)]
        x = np.arange(len(labels))
        h = plt.bar(x, self.coefficients)
        plt.xticks(x, labels, rotation=45, ha="right")
        return h

    def plotCoeffsWithoutHistory(
        self,
        fitNum: int = 1,
        sortByEpoch: bool = False,
        plotSignificance: bool = True,
    ) -> Any:
        _ = sortByEpoch
        return self.plotCoeffs(fitNum=fitNum, plotSignificance=plotSignificance)

    def plotHistCoeffs(
        self,
        fitNum: int = 1,
        sortByEpoch: bool = False,
        plotSignificance: bool = True,
    ) -> Any:
        _ = fitNum
        _ = sortByEpoch
        _ = plotSignificance
        import matplotlib.pyplot as plt

        return plt.plot([], [])

    def plotInvGausTrans(self) -> Any:
        import matplotlib.pyplot as plt

        if not self.inv_gaus_stats:
            return plt.plot([], [])
        first_key = next(iter(self.inv_gaus_stats.keys()))
        y = np.asarray(self.inv_gaus_stats[first_key], dtype=float).reshape(-1)
        x = np.arange(y.size)
        return plt.plot(x, y, "k-")

    def plotResidual(self) -> Any:
        import matplotlib.pyplot as plt

        if self.fit_residual is None:
            return plt.plot([], [])
        y = np.asarray(self.fit_residual, dtype=float).reshape(-1)
        x = np.arange(y.size)
        return plt.plot(x, y, "k-")

    def plotSeqCorr(self) -> Any:
        import matplotlib.pyplot as plt

        if self.fit_residual is None:
            return plt.plot([], [])
        y = np.asarray(self.fit_residual, dtype=float).reshape(-1)
        y = y - np.mean(y)
        corr = np.correlate(y, y, mode="full")
        corr = corr[corr.size // 2 :]
        if corr[0] != 0:
            corr = corr / corr[0]
        lags = np.arange(corr.size)
        return plt.plot(lags, corr, "k-")

    def KSPlot(self, fitNum: int = 1) -> Any:
        _ = fitNum
        import matplotlib.pyplot as plt

        ks = np.asarray(self.ks_stats.get("ks_stat", []), dtype=float).reshape(-1)
        if ks.size == 0:
            return plt.plot([], [])
        x = np.arange(ks.size)
        return plt.plot(x, ks, "k-")

    def plotValidation(self) -> Any:
        import matplotlib.pyplot as plt

        if not self.isValDataPresent():
            return plt.plot([], [])
        y = np.asarray(self.xval_data[0], dtype=float).reshape(-1)
        t = np.asarray(self.xval_time[0], dtype=float).reshape(-1)
        if t.size != y.size:
            t = np.arange(y.size, dtype=float)
        return plt.plot(t, y, "k-")

    def plotResults(self) -> Any:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(2, 1, 1)
        self.plotCoeffs()
        plt.subplot(2, 1, 2)
        self.plotResidual()
        return plt.gca()

    @staticmethod
    def xticklabel_rotate(XTick: np.ndarray, rot: float, *args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.set_xticks(np.asarray(XTick, dtype=float))
        for label in ax.get_xticklabels():
            label.set_rotation(rot)
        return ax.get_xticklabels()


class FitResSummary(_FitSummary):
    @staticmethod
    def FitResSummary(fitResultsCell: list[_FitResult] | _FitResult) -> _FitSummary:
        if isinstance(fitResultsCell, _FitResult):
            return FitResSummary([fitResultsCell])
        return FitResSummary(list(fitResultsCell))

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> _FitSummary:
        native = _FitSummary.from_structure(structure)
        return FitResSummary(native.results)

    def toStructure(self) -> dict[str, Any]:
        return self.to_structure()

    def mapCovLabelsToUniqueLabels(self) -> list[str]:
        return self.get_unique_labels()

    def getUniqueLabels(self) -> list[str]:
        return self.get_unique_labels()

    def getCoeffIndex(self, fitNum: int = 1, sortByEpoch: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
        return self.get_coeff_index(fit_num=fitNum, sort_by_epoch=sortByEpoch)

    def getCoeffs(self, fitNum: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        return self.get_coeffs(fit_num=fitNum)

    def binCoeffs(self, minVal: float, maxVal: float, binSize: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.bin_coeffs(min_val=minVal, max_val=maxVal, bin_size=binSize)

    def boxPlot(
        self,
        X: np.ndarray | None = None,
        diffIndex: int = 1,
        h: Any = None,
        dataLabels: list[str] | None = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, np.ndarray]:
        _ = h
        _ = dataLabels
        return self.box_plot(X=X, diff_index=diffIndex)

    def bestByAIC(self) -> _FitResult:
        return self.best_by_aic()

    def bestByBIC(self) -> _FitResult:
        return self.best_by_bic()

    def getDiffAIC(self) -> np.ndarray:
        return self.get_diff_aic()

    def getDiffBIC(self) -> np.ndarray:
        return self.get_diff_bic()

    def getDifflogLL(self) -> np.ndarray:
        return self.get_diff_log_likelihood()

    def computeDiffMat(self, metric: str = "aic") -> np.ndarray:
        return self.compute_diff_mat(metric=metric)

    def getHistIndex(self, fitNum: int = 1, sortByEpoch: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
        coeff_idx, epoch_id, num_epochs = self.get_coeff_index(fit_num=fitNum, sort_by_epoch=sortByEpoch)
        _coeff_mat, labels, _se = self.get_coeffs(fit_num=fitNum)
        keep = np.array(
            [i for i in coeff_idx if "hist" in labels[int(i)].lower() or "history" in labels[int(i)].lower()],
            dtype=int,
        )
        if keep.size == 0:
            return keep, np.array([], dtype=int), 1
        return keep, np.ones(keep.size, dtype=int), num_epochs

    def getHistCoeffs(self, fitNum: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        coeff_mat, labels, se_mat = self.get_coeffs(fit_num=fitNum)
        keep = [i for i, label in enumerate(labels) if "hist" in label.lower() or "history" in label.lower()]
        if not keep:
            empty = np.zeros((0, coeff_mat.shape[1]), dtype=float)
            return empty, [], empty
        idx = np.asarray(keep, dtype=int)
        return coeff_mat[idx, :], [labels[i] for i in idx], se_mat[idx, :]

    def getSigCoeffs(self, fitNum: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        coeff_mat, labels, se_mat = self.get_coeffs(fit_num=fitNum)
        col = max(0, min(coeff_mat.shape[1] - 1, int(fitNum) - 1))
        keep = np.where(np.isfinite(coeff_mat[:, col]) & (np.abs(coeff_mat[:, col]) > 0.0))[0].astype(int)
        if keep.size == 0:
            empty = np.zeros((0, coeff_mat.shape[1]), dtype=float)
            return empty, [], empty
        return coeff_mat[keep, :], [labels[i] for i in keep], se_mat[keep, :]

    def setCoeffRange(self, minVal: float, maxVal: float) -> _FitSummary:
        setattr(self, "_coeff_range", (float(minVal), float(maxVal)))
        return self

    @staticmethod
    def xticklabel_rotate(XTick: np.ndarray, rot: float, *args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.set_xticks(np.asarray(XTick, dtype=float))
        for label in ax.get_xticklabels():
            label.set_rotation(rot)
        return ax.get_xticklabels()

    def plotAIC(self) -> Any:
        import matplotlib.pyplot as plt

        vals = np.array([fit.aic() for fit in self.results], dtype=float)
        return plt.plot(np.arange(1, vals.size + 1), vals, "k-o")

    def plotBIC(self) -> Any:
        import matplotlib.pyplot as plt

        vals = np.array([fit.bic() for fit in self.results], dtype=float)
        return plt.plot(np.arange(1, vals.size + 1), vals, "k-o")

    def plotlogLL(self) -> Any:
        import matplotlib.pyplot as plt

        vals = np.array([fit.log_likelihood for fit in self.results], dtype=float)
        return plt.plot(np.arange(1, vals.size + 1), vals, "k-o")

    def plotIC(self, metric: str = "aic") -> Any:
        metric_key = str(metric).lower()
        if metric_key == "aic":
            return self.plotAIC()
        if metric_key == "bic":
            return self.plotBIC()
        if metric_key in {"logll", "log_likelihood"}:
            return self.plotlogLL()
        raise ValueError("metric must be one of {'aic', 'bic', 'logll'}")

    def plotAllCoeffs(self, fitNum: int = 1) -> Any:
        import matplotlib.pyplot as plt

        coeff_mat, _labels, _se = self.get_coeffs(fit_num=fitNum)
        if coeff_mat.size == 0:
            return plt.plot([], [])
        return plt.plot(np.arange(coeff_mat.shape[0]), coeff_mat[:, max(0, min(coeff_mat.shape[1] - 1, fitNum - 1))], "k.")

    def plotHistCoeffs(self, fitNum: int = 1) -> Any:
        import matplotlib.pyplot as plt

        coeffs, _labels, _se = self.getHistCoeffs(fitNum=fitNum)
        if coeffs.size == 0:
            return plt.plot([], [])
        col = max(0, min(coeffs.shape[1] - 1, fitNum - 1))
        return plt.plot(np.arange(coeffs.shape[0]), coeffs[:, col], "k.")

    def plotCoeffsWithoutHistory(self, fitNum: int = 1) -> Any:
        import matplotlib.pyplot as plt

        coeff_mat, labels, _se = self.get_coeffs(fit_num=fitNum)
        keep = [i for i, label in enumerate(labels) if "hist" not in label.lower() and "history" not in label.lower()]
        if not keep:
            return plt.plot([], [])
        col = max(0, min(coeff_mat.shape[1] - 1, fitNum - 1))
        idx = np.asarray(keep, dtype=int)
        return plt.plot(np.arange(idx.size), coeff_mat[idx, col], "k.")

    def plot2dCoeffSummary(self, fitNum: int = 1) -> Any:
        import matplotlib.pyplot as plt

        coeff_mat, _labels, _se = self.get_coeffs(fit_num=fitNum)
        return plt.imshow(coeff_mat, aspect="auto", interpolation="nearest")

    def plot3dCoeffSummary(self, fitNum: int = 1) -> Any:
        import matplotlib.pyplot as plt

        coeff_mat, _labels, _se = self.get_coeffs(fit_num=fitNum)
        if coeff_mat.size == 0:
            return plt.plot([], [])
        fig = plt.figure()
        ax = cast(Any, fig.add_subplot(111, projection="3d"))
        x = np.arange(coeff_mat.shape[0], dtype=float)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        dz = coeff_mat[:, max(0, min(coeff_mat.shape[1] - 1, fitNum - 1))]
        return ax.bar3d(x, y, z, 0.5, 0.5, dz)

    def plotKSSummary(self) -> Any:
        import matplotlib.pyplot as plt

        vals: list[float] = []
        for fit in self.results:
            raw = fit.ks_stats.get("ks_stat", np.nan)
            arr = np.asarray(raw, dtype=float).reshape(-1)
            vals.append(float(np.nanmean(arr)) if arr.size else np.nan)
        y = np.asarray(vals, dtype=float)
        return plt.plot(np.arange(1, y.size + 1), y, "k-o")

    def plotResidualSummary(self) -> Any:
        import matplotlib.pyplot as plt

        vals: list[float] = []
        for fit in self.results:
            if fit.fit_residual is None:
                vals.append(np.nan)
                continue
            arr = np.asarray(fit.fit_residual, dtype=float).reshape(-1)
            vals.append(float(np.nanmean(np.abs(arr))) if arr.size else np.nan)
        y = np.asarray(vals, dtype=float)
        return plt.plot(np.arange(1, y.size + 1), y, "k-o")

    def plotSummary(self, fitNum: int = 1) -> Any:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(2, 2, 1)
        self.plotAIC()
        plt.subplot(2, 2, 2)
        self.plotBIC()
        plt.subplot(2, 2, 3)
        self.plotAllCoeffs(fitNum=fitNum)
        plt.subplot(2, 2, 4)
        self.plotResidualSummary()
        return plt.gca()


class DecodingAlgorithms:
    @staticmethod
    def _em_not_implemented(name: str) -> None:
        raise NotImplementedError(
            f"{name} is not yet ported in nSTAT-python; use decodeStatePosterior/kalman_* APIs for now."
        )

    @staticmethod
    def _chol_like_matlab(mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=float)
        if arr.ndim == 0:
            return np.array([[float(np.sqrt(max(arr.item(), 0.0)))]] , dtype=float)
        if np.allclose(arr, 0.0):
            return np.zeros_like(arr, dtype=float)
        try:
            return np.linalg.cholesky(arr)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(arr)
            eigvals = np.clip(eigvals, 0.0, None)
            return eigvecs @ np.diag(np.sqrt(eigvals))

    @staticmethod
    def _build_unit_impulse_basis(numBasis: int, minTime: float, maxTime: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
        if numBasis <= 0:
            raise ValueError("numBasis must be > 0")
        basis_width = float(maxTime - minTime) / float(numBasis)
        sample_rate = 1.0 / float(delta)
        basis_sig = nstColl.generateUnitImpulseBasis(basis_width, minTime, maxTime, sample_rate)
        basis_mat = np.asarray(basis_sig.data, dtype=float)
        time = np.asarray(basis_sig.time, dtype=float).reshape(-1)
        return basis_mat, time

    @staticmethod
    def _draw_xk_samples_spec(xK_arr: np.ndarray, Wku_arr: np.ndarray, Mc: int, rng: np.random.Generator) -> np.ndarray:
        # MATLAB mirror: for r=1:numBasis, for c=1:Mc, xKdraw(r,:,c)=xK(r,:)+chol(WkuTemp)*z
        numBasis, K = xK_arr.shape
        xK_draw = np.zeros((numBasis, K, int(Mc)), dtype=float)
        for r in range(numBasis):
            if Wku_arr.ndim == 4:
                Wku_temp = np.asarray(Wku_arr[r, r, :, :], dtype=float)
            elif Wku_arr.ndim == 3:
                Wku_temp = np.asarray(Wku_arr[r, :, :], dtype=float)
            elif Wku_arr.ndim == 2:
                Wku_temp = np.asarray(Wku_arr, dtype=float)
            else:
                Wku_temp = np.asarray(0.0, dtype=float)
            if Wku_temp.ndim == 0:
                chol_m = np.diag(np.repeat(float(np.sqrt(max(Wku_temp.item(), 0.0))), K))
            else:
                chol_m = DecodingAlgorithms._chol_like_matlab(Wku_temp)
                if chol_m.shape != (K, K):
                    raise ValueError("Wku covariance slice must be KxK")
            for c in range(int(Mc)):
                z = rng.normal(0.0, 1.0, size=(K,))
                xK_draw[r, :, c] = xK_arr[r, :] + (chol_m @ z)
        return xK_draw

    @staticmethod
    def _draw_xk_samples_fast(xK_arr: np.ndarray, Wku_arr: np.ndarray, Mc: int, rng: np.random.Generator) -> np.ndarray:
        # Fast equivalent of _draw_xk_samples_spec with identical RNG ordering.
        numBasis, K = xK_arr.shape
        xK_draw = np.zeros((numBasis, K, int(Mc)), dtype=float)
        for r in range(numBasis):
            if Wku_arr.ndim == 4:
                Wku_temp = np.asarray(Wku_arr[r, r, :, :], dtype=float)
            elif Wku_arr.ndim == 3:
                Wku_temp = np.asarray(Wku_arr[r, :, :], dtype=float)
            elif Wku_arr.ndim == 2:
                Wku_temp = np.asarray(Wku_arr, dtype=float)
            else:
                Wku_temp = np.asarray(0.0, dtype=float)
            if Wku_temp.ndim == 0:
                chol_m = np.diag(np.repeat(float(np.sqrt(max(Wku_temp.item(), 0.0))), K))
            else:
                chol_m = DecodingAlgorithms._chol_like_matlab(Wku_temp)
                if chol_m.shape != (K, K):
                    raise ValueError("Wku covariance slice must be KxK")
            z_draw = rng.normal(0.0, 1.0, size=(int(Mc), K))
            xK_draw[r, :, :] = xK_arr[r, :][:, None] + (chol_m @ z_draw.T)
        return xK_draw

    @staticmethod
    def _compute_draw_rates_spec(
        basis_mat: np.ndarray,
        xK_draw: np.ndarray,
        draw_index: int,
        Hk: list[np.ndarray],
        gamma_vec: np.ndarray,
        fit_type: str,
        delta: float,
    ) -> np.ndarray:
        # MATLAB mirror: for each draw c and trial k, evaluate lambda_k(t).
        K = xK_draw.shape[1]
        n_time = basis_mat.shape[0]
        rates = np.zeros((n_time, K), dtype=float)
        for k in range(K):
            stim_k = basis_mat @ xK_draw[:, k, draw_index]
            hk = Hk[k]
            cols = min(hk.shape[1], gamma_vec.size)
            if cols > 0 and np.any(np.abs(gamma_vec[:cols]) > 0.0):
                hist_lin = hk[:, :cols] @ gamma_vec[:cols]
            else:
                hist_lin = np.zeros(stim_k.shape[0], dtype=float)
            eta = stim_k + hist_lin
            if fit_type == "poisson":
                lam = np.exp(eta)
            else:
                exp_eta = np.exp(eta)
                lam = exp_eta / (1.0 + exp_eta)
            rates[:, k] = lam / float(delta)
        return rates

    @staticmethod
    def _compute_draw_rates_fast(
        basis_mat: np.ndarray,
        xK_draw: np.ndarray,
        draw_index: int,
        hist_term: np.ndarray,
        fit_type: str,
        delta: float,
    ) -> np.ndarray:
        stim_ck = basis_mat @ xK_draw[:, :, draw_index]
        eta = stim_ck + hist_term
        if fit_type == "poisson":
            rates = np.exp(eta)
        else:
            exp_eta = np.exp(eta)
            rates = exp_eta / (1.0 + exp_eta)
        return rates / float(delta)

    @staticmethod
    def _compute_prob_mat_spec(spike_rate: np.ndarray, Mc: int) -> np.ndarray:
        # MATLAB mirror: upper-triangle probability matrix P(rate_m > rate_k).
        K = spike_rate.shape[1]
        prob_mat = np.zeros((K, K), dtype=float)
        for k in range(K):
            for m in range(k + 1, K):
                prob_mat[k, m] = float(np.sum(spike_rate[:, m] > spike_rate[:, k])) / float(Mc)
        return prob_mat

    @staticmethod
    def _compute_prob_mat_fast(spike_rate: np.ndarray) -> np.ndarray:
        prob_full = np.mean(spike_rate[:, None, :] > spike_rate[:, :, None], axis=0)
        return np.triu(np.asarray(prob_full, dtype=float), k=1)

    @staticmethod
    def _compute_spike_rate_cis_matlab(
        xK: np.ndarray,
        Wku: np.ndarray,
        dN: np.ndarray,
        t0: float,
        tf: float,
        fitType: str,
        delta: float,
        gamma: Any = None,
        windowTimes: Any = None,
        Mc: int = 500,
        alphaVal: float = 0.05,
        implementation: str = "fast",
    ) -> tuple[_Covariate, np.ndarray, np.ndarray]:
        # MATLAB reference block: DecodingAlgorithms.computeSpikeRateCIs
        # Keep a readable spec path; use fast helpers for CI/runtime workflows.
        xK_arr = np.asarray(xK, dtype=float)
        if xK_arr.ndim != 2:
            raise ValueError("xK must be 2D with shape (numBasis, K)")
        dN_arr = np.asarray(dN, dtype=float)
        if dN_arr.ndim != 2:
            raise ValueError("dN must be 2D with shape (K, T)")
        numBasis, K = xK_arr.shape
        if dN_arr.shape[0] != K:
            raise ValueError("dN first dimension must match K in xK")
        fit_type = str(fitType).lower()
        if fit_type not in {"poisson", "binomial"}:
            raise ValueError("fitType must be either 'poisson' or 'binomial'")
        if not (0.0 < float(alphaVal) < 1.0):
            raise ValueError("alphaVal must be in (0, 1)")
        if int(Mc) <= 0:
            raise ValueError("Mc must be > 0")
        impl = str(implementation).lower()
        if impl not in {"spec", "fast"}:
            raise ValueError("implementation must be 'spec' or 'fast'")

        # MATLAB block: construct unit-impulse basis on [0, Tmax].
        min_time = 0.0
        max_time = float(dN_arr.shape[1] - 1) * float(delta)
        basis_mat, basis_time = DecodingAlgorithms._build_unit_impulse_basis(numBasis, min_time, max_time, float(delta))
        if basis_mat.shape[0] < dN_arr.shape[1]:
            pad = np.zeros((dN_arr.shape[1] - basis_mat.shape[0], basis_mat.shape[1]), dtype=float)
            basis_mat = np.vstack([basis_mat, pad])
        elif basis_mat.shape[0] > dN_arr.shape[1]:
            basis_mat = basis_mat[: dN_arr.shape[1], :]
            basis_time = basis_time[: dN_arr.shape[1]]
        time = basis_time

        # MATLAB block: build history design matrices H{k} when windowTimes provided.
        window_vals = np.asarray([] if windowTimes is None else windowTimes, dtype=float).reshape(-1)
        if window_vals.size > 0:
            if window_vals.size == 1:
                window_vals = np.array([0.0, float(window_vals[0])], dtype=float)
            hist_obj = History(bin_edges_s=window_vals)
            gamma_vec = np.asarray(gamma, dtype=float).reshape(-1)
            Hk: list[np.ndarray] = []
            for k in range(K):
                spikes = np.where(dN_arr[k, :] == 1.0)[0].astype(float) * float(delta)
                hk = np.asarray(hist_obj.computeHistory(spikes, time), dtype=float)
                if hk.ndim == 1:
                    hk = hk[:, None]
                if hk.shape[0] < dN_arr.shape[1]:
                    hk = np.vstack([hk, np.zeros((dN_arr.shape[1] - hk.shape[0], hk.shape[1]), dtype=float)])
                elif hk.shape[0] > dN_arr.shape[1]:
                    hk = hk[: dN_arr.shape[1], :]
                Hk.append(hk)
            if gamma_vec.size == 0:
                gamma_vec = np.zeros(1, dtype=float)
        else:
            Hk = [np.zeros((dN_arr.shape[1], 1), dtype=float) for _ in range(K)]
            gamma_vec = np.zeros(1, dtype=float)

        # MATLAB block: Monte Carlo coefficient draws xKdraw.
        Wku_arr = np.asarray(Wku, dtype=float)
        rng = np.random.default_rng(0)
        if impl == "fast":
            xK_draw = DecodingAlgorithms._draw_xk_samples_fast(xK_arr, Wku_arr, int(Mc), rng)
        else:
            xK_draw = DecodingAlgorithms._draw_xk_samples_spec(xK_arr, Wku_arr, int(Mc), rng)

        spike_rate = np.zeros((int(Mc), K), dtype=float)
        mask = (time >= float(t0)) & (time <= float(tf))
        interval = max(float(tf - t0), np.finfo(float).eps)
        integrate_fn = getattr(np, "trapezoid", None)
        if integrate_fn is None:
            integrate_fn = getattr(np, "trapz", None)  # pragma: no cover - NumPy<2 fallback

        use_history = window_vals.size > 0 and np.any(np.abs(gamma_vec) > 0.0)
        if use_history and impl == "fast":
            hist_term = np.zeros((dN_arr.shape[1], K), dtype=float)
            for k in range(K):
                hk = Hk[k]
                cols = min(hk.shape[1], gamma_vec.size)
                hist_term[:, k] = hk[:, :cols] @ gamma_vec[:cols]
        else:
            hist_term = np.zeros((dN_arr.shape[1], K), dtype=float)

        # MATLAB block: for each draw c, integrate trial rates over [t0, tf].
        for c in range(int(Mc)):
            if impl == "fast":
                rates = DecodingAlgorithms._compute_draw_rates_fast(
                    basis_mat=basis_mat,
                    xK_draw=xK_draw,
                    draw_index=c,
                    hist_term=hist_term,
                    fit_type=fit_type,
                    delta=float(delta),
                )
            else:
                rates = DecodingAlgorithms._compute_draw_rates_spec(
                    basis_mat=basis_mat,
                    xK_draw=xK_draw,
                    draw_index=c,
                    Hk=Hk,
                    gamma_vec=gamma_vec,
                    fit_type=fit_type,
                    delta=float(delta),
                )
            if np.sum(mask) < 2:
                integral_vals = np.zeros(K, dtype=float)
            else:
                if integrate_fn is None:  # pragma: no cover - extreme fallback
                    dt_vec = np.diff(time[mask]).reshape(-1, 1)
                    y0 = rates[mask, :][:-1, :]
                    y1 = rates[mask, :][1:, :]
                    integral_vals = np.sum(0.5 * (y0 + y1) * dt_vec, axis=0)
                else:
                    integral_vals = np.asarray(
                        integrate_fn(rates[mask, :], x=time[mask], axis=0),
                        dtype=float,
                    )
            spike_rate[c, :] = integral_vals / interval

        CIs = np.zeros((K, 2), dtype=float)
        for k in range(K):
            vals = np.sort(spike_rate[:, k])
            f = (np.arange(vals.size, dtype=float) + 1.0) / float(vals.size)
            lo = vals[f < float(alphaVal)]
            hi = vals[f > (1.0 - float(alphaVal))]
            CIs[k, 0] = float(lo[-1]) if lo.size else float(vals[0])
            CIs[k, 1] = float(hi[0]) if hi.size else float(vals[-1])

        # MATLAB block: emit Covariate with attached confidence interval.
        spike_rate_sig = Covariate(
            time=np.arange(1, K + 1, dtype=float),
            data=np.mean(spike_rate, axis=0),
            name=f"({tf}-{t0})^-1 * \\Lambda({tf}-{t0})",
            x_label="Trial",
            x_units="k",
            y_units="Hz",
        )
        ci_obj = ConfidenceInterval(
            time=np.arange(1, K + 1, dtype=float),
            lower=CIs[:, 0],
            upper=CIs[:, 1],
            level=1.0 - float(alphaVal),
        )
        ci_obj.setColor("b")
        spike_rate_sig.setConfInterval(ci_obj)

        if impl == "fast":
            prob_mat = DecodingAlgorithms._compute_prob_mat_fast(spike_rate)
        else:
            prob_mat = DecodingAlgorithms._compute_prob_mat_spec(spike_rate, int(Mc))
        sig_mat = (prob_mat > (1.0 - float(alphaVal))).astype(float)
        return spike_rate_sig, prob_mat, sig_mat

    @staticmethod
    def computeSpikeRateCIs(*args: Any, **kwargs: Any) -> tuple[Any, np.ndarray, np.ndarray]:
        # MATLAB signature:
        #   computeSpikeRateCIs(xK,Wku,dN,t0,tf,fitType,delta,gamma,windowTimes,Mc,alphaVal)
        # Existing Python compact signature:
        #   computeSpikeRateCIs(spike_matrix, alpha=0.05)
        if len(args) >= 7:
            xK = np.asarray(args[0], dtype=float)
            Wku = np.asarray(args[1], dtype=float)
            dN = np.asarray(args[2], dtype=float)
            t0 = float(args[3])
            tf = float(args[4])
            fitType = str(args[5])
            delta = float(args[6])
            gamma = args[7] if len(args) >= 8 else kwargs.get("gamma", None)
            windowTimes = args[8] if len(args) >= 9 else kwargs.get("windowTimes", None)
            Mc = int(args[9]) if len(args) >= 10 else int(kwargs.get("Mc", 500))
            alphaVal = float(args[10]) if len(args) >= 11 else float(kwargs.get("alphaVal", 0.05))
            return DecodingAlgorithms._compute_spike_rate_cis_matlab(
                xK=xK,
                Wku=Wku,
                dN=dN,
                t0=t0,
                tf=tf,
                fitType=fitType,
                delta=delta,
                gamma=gamma,
                windowTimes=windowTimes,
                Mc=Mc,
                alphaVal=alphaVal,
            )

        if len(args) == 0 and "spike_matrix" not in kwargs:
            raise TypeError("computeSpikeRateCIs requires either MATLAB-style or compact arguments")
        spike_matrix = np.asarray(args[0] if args else kwargs["spike_matrix"], dtype=float)
        alpha = float(args[1]) if len(args) >= 2 else float(kwargs.get("alpha", 0.05))
        return _DecodingAlgorithms.compute_spike_rate_cis(spike_matrix=spike_matrix, alpha=alpha)

    @staticmethod
    def decodeWeightedCenter(spike_counts: np.ndarray, tuning_curves: np.ndarray) -> np.ndarray:
        return _DecodingAlgorithms.decode_weighted_center(spike_counts=spike_counts, tuning_curves=tuning_curves)

    @staticmethod
    def decodeStatePosterior(
        spike_counts: np.ndarray,
        tuning_rates: np.ndarray,
        transition: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _DecodingAlgorithms.decode_state_posterior(
            spike_counts=spike_counts,
            tuning_rates=tuning_rates,
            transition=transition,
            prior=prior,
        )

    @staticmethod
    def computeSpikeRateDiffCIs(
        spike_matrix_a: np.ndarray, spike_matrix_b: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _DecodingAlgorithms.compute_spike_rate_diff_cis(
            spike_matrix_a=spike_matrix_a, spike_matrix_b=spike_matrix_b, alpha=alpha
        )

    @staticmethod
    def ComputeStimulusCIs(
        posterior: np.ndarray, state_values: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _DecodingAlgorithms.compute_stimulus_cis(
            posterior=posterior, state_values=state_values, alpha=alpha
        )

    @staticmethod
    def kalman_predict(
        x_prev: np.ndarray, p_prev: np.ndarray, a: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return _DecodingAlgorithms.kalman_predict(x_prev=x_prev, p_prev=p_prev, a=a, q=q)

    @staticmethod
    def kalman_update(
        x_pred: np.ndarray,
        p_pred: np.ndarray,
        y_t: np.ndarray,
        h: np.ndarray,
        r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _DecodingAlgorithms.kalman_update(
            x_pred=x_pred, p_pred=p_pred, y_t=y_t, h=h, r=r
        )

    @staticmethod
    def kalman_filter(
        y: np.ndarray,
        a: np.ndarray,
        h: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _DecodingAlgorithms.kalman_filter(y=y, a=a, h=h, q=q, r=r, x0=x0, p0=p0)

    @staticmethod
    def kalman_fixedIntervalSmoother(
        xf: np.ndarray, pf: np.ndarray, xp: np.ndarray, pp: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return _DecodingAlgorithms.kalman_fixed_interval_smoother(
            xf=xf, pf=pf, xp=xp, pp=pp, a=a
        )

    @staticmethod
    def kalman_smoother(
        xf: np.ndarray, pf: np.ndarray, xp: np.ndarray, pp: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_fixedIntervalSmoother(xf=xf, pf=pf, xp=xp, pp=pp, a=a)

    @staticmethod
    def kalman_smootherFromFiltered(
        y: np.ndarray,
        a: np.ndarray,
        h: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        xf, pf, xp, pp = DecodingAlgorithms.kalman_filter(y=y, a=a, h=h, q=q, r=r, x0=x0, p0=p0)
        return DecodingAlgorithms.kalman_fixedIntervalSmoother(xf=xf, pf=pf, xp=xp, pp=pp, a=a)

    @staticmethod
    def PPDecodeFilter(
        spike_counts: np.ndarray,
        tuning_rates: np.ndarray,
        transition: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.decodeStatePosterior(
            spike_counts=spike_counts,
            tuning_rates=tuning_rates,
            transition=transition,
            prior=prior,
        )

    @staticmethod
    def PPDecodeFilterLinear(
        spike_counts: np.ndarray,
        tuning_rates: np.ndarray,
        transition: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.PPDecodeFilter(
            spike_counts=spike_counts, tuning_rates=tuning_rates, transition=transition, prior=prior
        )

    @staticmethod
    def PPHybridFilter(
        spike_counts: np.ndarray,
        tuning_rates: np.ndarray,
        transition: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.PPDecodeFilter(
            spike_counts=spike_counts, tuning_rates=tuning_rates, transition=transition, prior=prior
        )

    @staticmethod
    def PPHybridFilterLinear(
        spike_counts: np.ndarray,
        tuning_rates: np.ndarray,
        transition: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.PPHybridFilter(
            spike_counts=spike_counts, tuning_rates=tuning_rates, transition=transition, prior=prior
        )

    @staticmethod
    def PPDecode_predict(
        x_prev: np.ndarray, p_prev: np.ndarray, a: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_predict(x_prev=x_prev, p_prev=p_prev, a=a, q=q)

    @staticmethod
    def PPDecode_update(
        x_pred: np.ndarray, p_pred: np.ndarray, y_t: np.ndarray, h: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_update(x_pred=x_pred, p_pred=p_pred, y_t=y_t, h=h, r=r)

    @staticmethod
    def PPDecode_updateLinear(
        x_pred: np.ndarray, p_pred: np.ndarray, y_t: np.ndarray, h: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.PPDecode_update(x_pred=x_pred, p_pred=p_pred, y_t=y_t, h=h, r=r)

    @staticmethod
    def PP_fixedIntervalSmoother(
        xf: np.ndarray, pf: np.ndarray, xp: np.ndarray, pp: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_fixedIntervalSmoother(xf=xf, pf=pf, xp=xp, pp=pp, a=a)

    @staticmethod
    def mPPCODecodeLinear(
        spike_counts: np.ndarray,
        tuning_rates: np.ndarray,
        transition: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.decodeStatePosterior(
            spike_counts=spike_counts,
            tuning_rates=tuning_rates,
            transition=transition,
            prior=prior,
        )

    @staticmethod
    def mPPCODecode_predict(
        x_prev: np.ndarray, p_prev: np.ndarray, a: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_predict(x_prev=x_prev, p_prev=p_prev, a=a, q=q)

    @staticmethod
    def mPPCODecode_update(
        x_pred: np.ndarray, p_pred: np.ndarray, y_t: np.ndarray, h: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_update(x_pred=x_pred, p_pred=p_pred, y_t=y_t, h=h, r=r)

    @staticmethod
    def mPPCO_fixedIntervalSmoother(
        xf: np.ndarray, pf: np.ndarray, xp: np.ndarray, pp: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_fixedIntervalSmoother(xf=xf, pf=pf, xp=xp, pp=pp, a=a)

    @staticmethod
    def ukf_sigmas(x: np.ndarray, p: np.ndarray, kappa: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)
        n = x.size
        s = np.linalg.cholesky((n + kappa) * p)
        sigmas = np.column_stack([x, x[:, None] + s, x[:, None] - s])
        return sigmas

    @staticmethod
    def ukf_ut(
        sigmas: np.ndarray, wm: np.ndarray, wc: np.ndarray, noise: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        sig = np.asarray(sigmas, dtype=float)
        wm = np.asarray(wm, dtype=float).reshape(-1)
        wc = np.asarray(wc, dtype=float).reshape(-1)
        mu = sig @ wm
        d = sig - mu[:, None]
        cov = d @ np.diag(wc) @ d.T
        if noise is not None:
            cov = cov + np.asarray(noise, dtype=float)
        return mu, cov

    @staticmethod
    def ukf(
        x_prev: np.ndarray, p_prev: np.ndarray, a: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return DecodingAlgorithms.kalman_predict(x_prev=x_prev, p_prev=p_prev, a=a, q=q)

    @staticmethod
    def KF_ComputeParamStandardErrors(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("KF_ComputeParamStandardErrors")

    @staticmethod
    def KF_EM(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("KF_EM")

    @staticmethod
    def KF_EMCreateConstraints(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("KF_EMCreateConstraints")

    @staticmethod
    def KF_EStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("KF_EStep")

    @staticmethod
    def KF_MStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("KF_MStep")

    @staticmethod
    def PPSS_EM(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PPSS_EM")

    @staticmethod
    def PPSS_EMFB(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PPSS_EMFB")

    @staticmethod
    def PPSS_EStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PPSS_EStep")

    @staticmethod
    def PPSS_MStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PPSS_MStep")

    @staticmethod
    def PP_ComputeParamStandardErrors(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PP_ComputeParamStandardErrors")

    @staticmethod
    def PP_EM(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PP_EM")

    @staticmethod
    def PP_EMCreateConstraints(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PP_EMCreateConstraints")

    @staticmethod
    def PP_EStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PP_EStep")

    @staticmethod
    def PP_MStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("PP_MStep")

    @staticmethod
    def estimateInfoMat(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("estimateInfoMat")

    @staticmethod
    def mPPCO_ComputeParamStandardErrors(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("mPPCO_ComputeParamStandardErrors")

    @staticmethod
    def mPPCO_EM(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("mPPCO_EM")

    @staticmethod
    def mPPCO_EMCreateConstraints(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("mPPCO_EMCreateConstraints")

    @staticmethod
    def mPPCO_EStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("mPPCO_EStep")

    @staticmethod
    def mPPCO_MStep(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("mPPCO_MStep")

    @staticmethod
    def prepareEMResults(*_args: Any, **_kwargs: Any) -> None:
        DecodingAlgorithms._em_not_implemented("prepareEMResults")


__all__ = [
    "SignalObj",
    "Covariate",
    "ConfidenceInterval",
    "Events",
    "History",
    "nspikeTrain",
    "nstColl",
    "CovColl",
    "TrialConfig",
    "ConfigColl",
    "Trial",
    "CIF",
    "Analysis",
    "FitResult",
    "FitResSummary",
    "DecodingAlgorithms",
]
