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


def _is_empty_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    return False


def _to_python_cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_to_python_cell(v) for v in value.reshape(-1)]
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item() if hasattr(value.reshape(-1)[0], "item") else value.reshape(-1)[0]
        return value.tolist()
    if isinstance(value, list):
        return [_to_python_cell(v) for v in value]
    if isinstance(value, tuple):
        return [_to_python_cell(v) for v in value]
    return value


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

    def alignTime(self, timeMarker: float = 0.0, newTime: float | None = None) -> "SignalObj":
        # MATLAB signature: alignTime(sObj, timeMarker, newTime).
        # Backward-compatible fallback: alignTime(newZero) shifts first sample.
        if newTime is None:
            self.align_time(timeMarker)
            return self
        marker = float(timeMarker)
        target = float(newTime)
        if self.time[0] <= marker <= self.time[-1]:
            self.shiftTime(target - marker)
        return self

    def derivative(self) -> "SignalObj":
        # MATLAB implementation uses forward differences with a leading zero row.
        mat = self.data_to_matrix()
        diff_data = np.diff(mat, axis=0) * float(self.sample_rate_hz)
        deriv = np.vstack([np.zeros((1, mat.shape[1]), dtype=float), diff_data])
        if deriv.shape[1] == 1:
            deriv_out: np.ndarray = deriv[:, 0]
        else:
            deriv_out = deriv
        return SignalObj(
            time=self.time.copy(),
            data=deriv_out,
            name=self.name,
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
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

    def merge(self, *args: Any) -> "SignalObj":
        if not args:
            raise ValueError("merge expects at least one signal")
        signals: list[_Signal] = []
        for idx, arg in enumerate(args):
            if isinstance(arg, (int, float)) and idx == len(args) - 1:
                # MATLAB supports optional holdVals argument.
                continue
            if not isinstance(arg, _Signal):
                raise ValueError("merge expects SignalObj arguments")
            signals.append(arg)
        if not signals:
            raise ValueError("merge expects at least one signal")

        merged: SignalObj = self
        for other in signals:
            lhs_c, rhs_c = merged.makeCompatible(other)
            data = np.hstack([lhs_c.dataToMatrix(), rhs_c.dataToMatrix()])
            merged = SignalObj(
                time=lhs_c.time.copy(),
                data=data,
                name=lhs_c.name,
                units=lhs_c.units,
                x_label=lhs_c.x_label,
                y_label=lhs_c.y_label,
                x_units=lhs_c.x_units,
                y_units=lhs_c.y_units,
                plot_props=dict(lhs_c.plot_props),
            )
        return merged

    def resample(self, sampleRate: float) -> "SignalObj":
        from scipy.interpolate import CubicSpline

        sample_rate = float(sampleRate)
        if sample_rate <= 0.0:
            raise ValueError("sampleRate must be positive")
        if np.isclose(sample_rate, self.sample_rate_hz):
            return self.copySignal()
        dt = 1.0 / sample_rate
        t_new = np.arange(self.time[0], self.time[-1] + 0.5 * dt, dt, dtype=float)
        mat = self.data_to_matrix()
        y_new = np.zeros((t_new.size, mat.shape[1]), dtype=float)
        for idx in range(mat.shape[1]):
            spline = CubicSpline(self.time, mat[:, idx], extrapolate=False)
            vals = spline(t_new)
            vals = np.asarray(vals, dtype=float)
            vals[~np.isfinite(vals)] = 0.0
            y_new[:, idx] = vals
        if y_new.shape[1] == 1:
            out_data: np.ndarray = y_new[:, 0]
        else:
            out_data = y_new
        out = SignalObj(
            time=t_new,
            data=out_data,
            name=self.name,
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )
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
        def _text(value: Any, default: str = "") -> str:
            if value is None:
                return default
            arr = np.asarray(value, dtype=object)
            if arr.size == 1:
                return str(arr.reshape(-1)[0])
            return str(value)

        if hasattr(payload, "_fieldnames"):
            payload = {name: getattr(payload, name) for name in payload._fieldnames}
        if "signals" in payload:
            signals = payload["signals"]
            if hasattr(signals, "_fieldnames"):
                values = np.asarray(getattr(signals, "values"), dtype=float)
            else:
                arr = np.asarray(signals, dtype=object)
                if arr.size == 1 and hasattr(arr.reshape(-1)[0], "_fieldnames"):
                    values = np.asarray(getattr(arr.reshape(-1)[0], "values"), dtype=float)
                elif isinstance(signals, dict):
                    values = np.asarray(signals["values"], dtype=float)
                else:
                    raise ValueError("Unsupported signals structure payload")
            data_values = values
        else:
            data_values = np.asarray(payload["data"], dtype=float)
        plot_props_raw = payload.get("plot_props", payload.get("plotProps", {}))
        if isinstance(plot_props_raw, dict):
            plot_props = dict(plot_props_raw)
        else:
            plot_props = {}
        return SignalObj(
            time=np.asarray(payload["time"], dtype=float).reshape(-1),
            data=data_values,
            name=_text(payload.get("name", "signal"), "signal"),
            units=_text(payload.get("units", ""), ""),
            x_label=_text(payload.get("x_label", payload.get("xlabelval")), "time"),
            y_label=payload.get("y_label"),
            x_units=_text(payload.get("x_units", payload.get("xunits")), ""),
            y_units=_text(payload.get("y_units", payload.get("yunits")), ""),
            plot_props=plot_props,
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
        out = self.copySignal()
        out.shiftTime(offset_s)
        return out

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
        x = np.asarray(mat[:, 0], dtype=float).reshape(-1)
        nperseg = min(256, max(1, x.size))
        f, t, s = spectrogram(x, fs=fs, nperseg=nperseg)
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

    @staticmethod
    def _text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        arr = np.asarray(value, dtype=object)
        if arr.size == 1:
            return str(arr.reshape(-1)[0])
        return str(value)

    @staticmethod
    def _to_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "_fieldnames"):
            return {name: getattr(payload, name) for name in payload._fieldnames}
        arr = np.asarray(payload, dtype=object)
        if arr.size == 1 and hasattr(arr.reshape(-1)[0], "_fieldnames"):
            s0 = arr.reshape(-1)[0]
            return {name: getattr(s0, name) for name in s0._fieldnames}
        raise ValueError("Unsupported structure payload")

    @staticmethod
    def _normalize_labels(raw: Any, fallback_name: str, n_channels: int) -> list[str]:
        if raw is None:
            raw_labels: list[str] = []
        else:
            arr = np.asarray(raw, dtype=object).reshape(-1)
            raw_labels = [str(v) for v in arr if str(v) != ""]
        if raw_labels and len(raw_labels) == n_channels:
            return raw_labels
        if n_channels == 1:
            return [fallback_name]
        return [f"{fallback_name}_{i}" for i in range(n_channels)]

    @staticmethod
    def _selector_to_indices(selector: int | str | list[int] | list[str], n_channels: int, labels: list[str]) -> np.ndarray:
        if isinstance(selector, str):
            return np.asarray([labels.index(selector)], dtype=int)
        if isinstance(selector, list) and selector and isinstance(selector[0], str):
            return np.asarray([labels.index(str(item)) for item in selector], dtype=int)
        idx = np.asarray(np.atleast_1d(selector), dtype=int).reshape(-1)
        # MATLAB selectors are 1-based.
        if idx.size and np.all(idx >= 1) and np.max(idx) <= n_channels:
            idx = idx - 1
        return idx

    @staticmethod
    def _as_ci_list(interval: Any) -> list[_ConfidenceInterval]:
        if interval is None:
            return []
        if isinstance(interval, _ConfidenceInterval):
            return [interval]
        if isinstance(interval, list):
            return [item for item in interval if isinstance(item, _ConfidenceInterval)]
        if isinstance(interval, tuple):
            return [item for item in list(interval) if isinstance(item, _ConfidenceInterval)]
        return []

    @staticmethod
    def _ci_from_operand(operand: Any, ref_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(operand, _ConfidenceInterval):
            return np.asarray(operand.lower, dtype=float), np.asarray(operand.upper, dtype=float)
        if isinstance(operand, _Signal):
            vec = np.asarray(operand.data_to_matrix(), dtype=float)[:, 0]
            return vec, vec
        arr = np.asarray(operand, dtype=float).reshape(-1)
        if arr.size == 1:
            vec = np.full(ref_time.size, float(arr.item()), dtype=float)
        elif arr.size == ref_time.size:
            vec = arr
        else:
            raise ValueError("Operand size incompatible with confidence interval length")
        return vec, vec

    @staticmethod
    def _ci_add(ci: _ConfidenceInterval, operand: Any) -> _ConfidenceInterval:
        lo_rhs, hi_rhs = Covariate._ci_from_operand(operand, np.asarray(ci.time, dtype=float))
        return ConfidenceInterval(
            time=np.asarray(ci.time, dtype=float),
            lower=np.asarray(ci.lower, dtype=float) + lo_rhs,
            upper=np.asarray(ci.upper, dtype=float) + hi_rhs,
            level=float(getattr(ci, "level", 0.95)),
            color=str(getattr(ci, "color", "b")),
            value=getattr(ci, "value", getattr(ci, "level", 0.95)),
        )

    @staticmethod
    def _ci_sub(ci: _ConfidenceInterval, operand: Any) -> _ConfidenceInterval:
        lo_rhs, hi_rhs = Covariate._ci_from_operand(operand, np.asarray(ci.time, dtype=float))
        return ConfidenceInterval(
            time=np.asarray(ci.time, dtype=float),
            lower=np.asarray(ci.lower, dtype=float) - lo_rhs,
            upper=np.asarray(ci.upper, dtype=float) - hi_rhs,
            level=float(getattr(ci, "level", 0.95)),
            color=str(getattr(ci, "color", "b")),
            value=getattr(ci, "value", getattr(ci, "level", 0.95)),
        )

    @staticmethod
    def _ci_neg(ci: _ConfidenceInterval) -> _ConfidenceInterval:
        # Keep interval ordering valid in Python while matching MATLAB arithmetic intent.
        lower = -np.asarray(ci.upper, dtype=float)
        upper = -np.asarray(ci.lower, dtype=float)
        return ConfidenceInterval(
            time=np.asarray(ci.time, dtype=float),
            lower=lower,
            upper=upper,
            level=float(getattr(ci, "level", 0.95)),
            color=str(getattr(ci, "color", "b")),
            value=getattr(ci, "value", getattr(ci, "level", 0.95)),
        )

    @staticmethod
    def _ecdf_quantiles(row: np.ndarray, alpha_val: float) -> tuple[float, float]:
        alpha = float(alpha_val)
        if row.size == 0:
            return 0.0, 0.0
        try:
            lower = float(np.quantile(row, alpha / 2.0, method="lower"))
            upper = float(np.quantile(row, 1.0 - alpha / 2.0, method="higher"))
        except TypeError:
            lower = float(np.quantile(row, alpha / 2.0, interpolation="lower"))
            upper = float(np.quantile(row, 1.0 - alpha / 2.0, interpolation="higher"))
        return lower, upper

    def computeMeanPlusCI(self, alphaVal: float = 0.05) -> _Covariate:
        mat = self.data_to_matrix()
        cimat = np.zeros((mat.shape[0], 2), dtype=float)
        for k in range(mat.shape[0]):
            cimat[k, 0], cimat[k, 1] = self._ecdf_quantiles(mat[k, :], alphaVal)
        mean_data = np.mean(mat, axis=1)
        out = Covariate(
            time=self.time.copy(),
            data=mean_data,
            name=f"\\mu({self.name})",
            units=self.units,
            labels=[f"\\mu({self.name})"],
            conf_interval=None,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )
        out.setConfInterval(
            ConfidenceInterval(
                time=self.time.copy(),
                lower=cimat[:, 0],
                upper=cimat[:, 1],
                level=max(1.0e-12, 1.0 - float(alphaVal)),
                color="b",
                value=max(1.0e-12, 1.0 - float(alphaVal)),
            )
        )
        return out

    def getSubSignal(self, selector: int | str | list[int] | list[str]) -> _Covariate:
        idx = self._selector_to_indices(selector, self.n_channels, self.labels)
        idx_sel: int | list[int]
        if idx.size == 1:
            idx_sel = int(idx[0])
        else:
            idx_sel = [int(v) for v in idx.tolist()]
        out = super().get_sub_signal(idx_sel)
        ci_list = self._as_ci_list(self.conf_interval)
        sub_ci: list[_ConfidenceInterval] = []
        for ind in idx.tolist():
            if not ci_list:
                break
            if ind < len(ci_list):
                sub_ci.append(ci_list[ind])
            elif len(ci_list) == 1:
                sub_ci.append(ci_list[0])
        return Covariate(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            labels=out.labels,
            conf_interval=sub_ci if sub_ci else None,
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def setConfInterval(self, interval: Any) -> _Covariate:
        ci_list = self._as_ci_list(interval)
        if ci_list:
            self.set_conf_interval(ci_list)
        else:
            self.set_conf_interval(interval)
        return self

    def isConfIntervalSet(self) -> bool:
        ci_list = self._as_ci_list(self.conf_interval)
        return len(ci_list) > 0

    def getSigRep(self, repType: str = "standard") -> _Covariate:
        if repType == "standard":
            return self
        if repType == "zero-mean":
            mat = self.data_to_matrix()
            centered = mat - np.mean(mat, axis=0, keepdims=True)
            data = centered[:, 0] if centered.shape[1] == 1 else centered
            return Covariate(
                time=self.time.copy(),
                data=data,
                name=self.name,
                units=self.units,
                labels=self.labels.copy(),
                conf_interval=None,
                x_label=self.x_label,
                y_label=self.y_label,
                x_units=self.x_units,
                y_units=self.y_units,
                plot_props=dict(self.plot_props),
            )
        raise ValueError("repType must be either 'zero-mean' or 'standard'")

    def dataToStructure(self) -> dict[str, Any]:
        return self.toStructure()

    def toStructure(self) -> dict[str, Any]:
        mat = self.data_to_matrix()
        n_channels = int(mat.shape[1])
        out: dict[str, Any] = {
            "time": self.time.copy(),
            "signals": {
                "values": mat.copy(),
                "dimensions": np.array([mat.shape[0], n_channels], dtype=float),
            },
            "name": self.name,
            "dimension": n_channels,
            "minTime": float(self.time.min()) if self.time.size else 0.0,
            "maxTime": float(self.time.max()) if self.time.size else 0.0,
            "xlabelval": self.x_label,
            "xunits": self.x_units,
            "yunits": self.y_units,
            "dataLabels": list(self.labels),
            "dataMask": list(np.ones(n_channels, dtype=int)),
            "sampleRate": float(self.sample_rate_hz),
            "plotProps": [],
        }
        ci_list = self._as_ci_list(self.conf_interval)
        if ci_list:
            if len(ci_list) == 1:
                out["ci"] = ci_list[0].to_structure()
            else:
                out["ci"] = [ci.to_structure() for ci in ci_list]
        return out

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _Covariate:
        structure = Covariate._to_dict(payload)
        if "signals" in structure:
            sig = structure["signals"]
            if isinstance(sig, dict):
                values = np.asarray(sig["values"], dtype=float)
            elif hasattr(sig, "values"):
                values = np.asarray(getattr(sig, "values"), dtype=float)
            else:
                sig_arr = np.asarray(sig, dtype=object)
                if sig_arr.size != 1:
                    raise ValueError("signals payload must be scalar struct-like")
                s0 = sig_arr.reshape(-1)[0]
                if hasattr(s0, "values"):
                    values = np.asarray(getattr(s0, "values"), dtype=float)
                elif isinstance(s0, dict):
                    values = np.asarray(s0["values"], dtype=float)
                else:
                    raise ValueError("Unsupported signals payload")
        else:
            values = np.asarray(structure["data"], dtype=float)

        if values.ndim == 2 and values.shape[1] == 1:
            data: np.ndarray = values[:, 0]
            n_channels = 1
        else:
            data = values
            n_channels = int(np.asarray(values).shape[1]) if np.asarray(values).ndim == 2 else 1

        name = Covariate._text(structure.get("name", "covariate"), "covariate")
        labels = Covariate._normalize_labels(structure.get("dataLabels", structure.get("labels")), name, n_channels)
        units = Covariate._text(structure.get("units", structure.get("yunits", "")), "")
        x_label = Covariate._text(structure.get("x_label", structure.get("xlabelval", "time")), "time")
        x_units = Covariate._text(structure.get("x_units", structure.get("xunits", "")), "")
        y_units = Covariate._text(structure.get("y_units", structure.get("yunits", "")), "")

        ci_list: list[_ConfidenceInterval] = []
        if "ci" in structure and structure["ci"] is not None:
            raw_ci = structure["ci"]
            if isinstance(raw_ci, _ConfidenceInterval):
                ci_list = [raw_ci]
            elif isinstance(raw_ci, dict) or hasattr(raw_ci, "_fieldnames"):
                ci_list = [ConfidenceInterval.fromStructure(Covariate._to_dict(raw_ci))]
            else:
                ci_arr = np.asarray(raw_ci, dtype=object).reshape(-1)
                for entry in ci_arr:
                    if entry is None:
                        continue
                    if isinstance(entry, _ConfidenceInterval):
                        ci_list.append(entry)
                    else:
                        ci_list.append(ConfidenceInterval.fromStructure(Covariate._to_dict(entry)))

        return Covariate(
            time=np.asarray(structure["time"], dtype=float).reshape(-1),
            data=data,
            name=name,
            units=units,
            labels=labels,
            conf_interval=ci_list if ci_list else None,
            x_label=x_label,
            y_label=Covariate._text(structure.get("y_label"), ""),
            x_units=x_units,
            y_units=y_units,
            plot_props={},
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

    def dataToMatrix(self) -> np.ndarray:
        return self.data_to_matrix()

    def filtfilt(self, b: np.ndarray, a: np.ndarray) -> _Covariate:
        out = super().filtfilt(np.asarray(b, dtype=float), np.asarray(a, dtype=float))
        return Covariate(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            labels=out.labels,
            conf_interval=self._as_ci_list(self.conf_interval),
            x_label=out.x_label,
            y_label=out.y_label,
            x_units=out.x_units,
            y_units=out.y_units,
            plot_props=out.plot_props,
        )

    def copySignal(self) -> _Covariate:
        out = super().copy_signal()
        ci_list = self._as_ci_list(self.conf_interval)
        return Covariate(
            time=out.time,
            data=out.data,
            name=out.name,
            units=out.units,
            labels=self.labels.copy(),
            conf_interval=ci_list.copy() if ci_list else None,
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
        out_cov = Covariate(
            time=self.time.copy(),
            data=data,
            name=f"{self.name}+",
            units=self.units,
            labels=self.labels.copy(),
            conf_interval=self._as_ci_list(self.conf_interval),
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )
        if isinstance(other, Covariate):
            lhs_ci = self._as_ci_list(self.conf_interval)
            rhs_ci = self._as_ci_list(other.conf_interval)
            temp_ci: list[_ConfidenceInterval] = []
            if lhs_ci and not rhs_ci:
                for i in range(self.n_channels):
                    ci = lhs_ci[i] if i < len(lhs_ci) else lhs_ci[0]
                    temp_ci.append(self._ci_add(ci, other.getSubSignal(i + 1)))
            elif lhs_ci and rhs_ci:
                for i in range(self.n_channels):
                    lci = lhs_ci[i] if i < len(lhs_ci) else lhs_ci[0]
                    rci = rhs_ci[i] if i < len(rhs_ci) else rhs_ci[0]
                    temp_ci.append(self._ci_add(lci, rci))
            elif (not lhs_ci) and rhs_ci:
                for i in range(other.n_channels):
                    rci = rhs_ci[i] if i < len(rhs_ci) else rhs_ci[0]
                    temp_ci.append(self._ci_add(rci, self.getSubSignal(i + 1)))
            if temp_ci:
                out_cov.setConfInterval(temp_ci)
            else:
                out_cov.set_conf_interval(None)
        return out_cov

    def minus(self, other: float | np.ndarray | _Signal) -> _Covariate:
        if isinstance(other, _Signal):
            rhs = other.data_to_matrix()
        else:
            rhs = np.asarray(other, dtype=float)
        lhs = self.data_to_matrix()
        out = lhs - rhs
        data = out[:, 0] if out.ndim == 2 and out.shape[1] == 1 else out
        out_cov = Covariate(
            time=self.time.copy(),
            data=data,
            name=f"{self.name}-",
            units=self.units,
            labels=self.labels.copy(),
            conf_interval=self._as_ci_list(self.conf_interval),
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )
        if isinstance(other, Covariate):
            lhs_ci = self._as_ci_list(self.conf_interval)
            rhs_ci = self._as_ci_list(other.conf_interval)
            temp_ci: list[_ConfidenceInterval] = []
            if lhs_ci and not rhs_ci:
                for i in range(self.n_channels):
                    ci = lhs_ci[i] if i < len(lhs_ci) else lhs_ci[0]
                    temp_ci.append(self._ci_sub(ci, other.getSubSignal(i + 1)))
            elif lhs_ci and rhs_ci:
                for i in range(self.n_channels):
                    lci = lhs_ci[i] if i < len(lhs_ci) else lhs_ci[0]
                    rci = rhs_ci[i] if i < len(rhs_ci) else rhs_ci[0]
                    temp_ci.append(self._ci_sub(lci, rci))
            elif (not lhs_ci) and rhs_ci:
                for i in range(other.n_channels):
                    rci = rhs_ci[i] if i < len(rhs_ci) else rhs_ci[0]
                    temp_ci.append(self._ci_add(self._ci_neg(rci), self.getSubSignal(i + 1)))
            if temp_ci:
                out_cov.setConfInterval(temp_ci)
            else:
                out_cov.set_conf_interval(None)
        return out_cov

    def plot(self, *_args: Any, **_kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        handles = plt.plot(self.time, self.data_to_matrix())
        ci_list = self._as_ci_list(self.conf_interval)
        if ci_list:
            for i, ci in enumerate(ci_list):
                color = "k"
                if i < len(handles):
                    color = handles[i].get_color()
                ConfidenceInterval.plot(cast(ConfidenceInterval, ci), color=color)
        return handles


class ConfidenceInterval(_ConfidenceInterval):
    @staticmethod
    def ConfidenceInterval(*args: Any, **kwargs: Any) -> _ConfidenceInterval:
        if len(args) == 1 and isinstance(args[0], dict):
            return ConfidenceInterval.fromStructure(args[0])
        return ConfidenceInterval(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _ConfidenceInterval:
        if "signals" in payload:
            sig = payload["signals"]
            if isinstance(sig, dict):
                values = np.asarray(sig["values"], dtype=float)
            elif hasattr(sig, "values"):
                values = np.asarray(getattr(sig, "values"), dtype=float)
            else:
                arr = np.asarray(sig, dtype=object)
                if arr.size != 1:
                    raise ValueError("signals payload must be scalar struct-like")
                s0 = arr.reshape(-1)[0]
                if hasattr(s0, "values"):
                    values = np.asarray(getattr(s0, "values"), dtype=float)
                elif isinstance(s0, dict):
                    values = np.asarray(s0["values"], dtype=float)
                else:
                    raise ValueError("Unsupported signals payload")
            if values.ndim != 2 or values.shape[1] < 2:
                raise ValueError("signals.values must be [N,2] for ConfidenceInterval")
            return ConfidenceInterval(
                time=np.asarray(payload["time"], dtype=float),
                lower=values[:, 0],
                upper=values[:, 1],
                level=0.95,
                color="b",
                value=0.95,
            )
        return ConfidenceInterval(
            time=np.asarray(payload["time"], dtype=float),
            lower=np.asarray(payload["lower"], dtype=float),
            upper=np.asarray(payload["upper"], dtype=float),
            level=0.95,
            color="b",
            value=0.95,
        )

    def toStructure(self) -> dict[str, Any]:
        values = np.column_stack([self.lower, self.upper])
        return {
            "time": self.time.copy(),
            "lower": self.lower.copy(),
            "upper": self.upper.copy(),
            "level": float(self.level),
            "value": self.value,
            "color": str(self.color),
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
            "sampleRate": float((self.time.size - 1) / (self.time[-1] - self.time[0]))
            if self.time.size > 1 and self.time[-1] != self.time[0]
            else 1.0,
            "plotProps": [],
        }

    def setColor(self, color: str) -> _ConfidenceInterval:
        self.color = str(color)
        return self

    def setValue(self, values: np.ndarray | float) -> _ConfidenceInterval:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            self.value = float(arr)
            self.level = float(arr)
        else:
            self.value = arr.copy()
        return self

    def plot(self, color: Any = None, alphaVal: float = 0.2, drawPatches: int = 0) -> Any:
        import matplotlib.pyplot as plt

        color_val = self.color if color is None else color
        ci_data = np.column_stack([self.lower, self.upper])
        ci_high = ci_data[:, 1]
        ci_low = ci_data[:, 0]
        time = self.time

        if int(drawPatches) == 1:
            x_poly = np.concatenate([time, np.flip(time)])
            y_poly = np.concatenate([ci_low, np.flip(ci_high)])
            patch = plt.fill(x_poly, y_poly, color=color_val, alpha=float(alphaVal), edgecolor="none")
            return patch
        lines = plt.plot(time, ci_data)
        if not isinstance(color_val, str):
            for line in lines:
                line.set_color(color_val)
        for line in lines:
            line.set_alpha(float(alphaVal))
        return lines

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
        if not payload:
            raise ValueError("Missing field in structure. Cant creats Events object!")
        required = ("eventTimes", "eventLabels", "eventColor")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError("Missing field in structure. Cant creats Events object!")
        return Events(
            times=np.asarray(payload["eventTimes"], dtype=float),
            labels=[str(v) for v in payload["eventLabels"]],
            color=str(payload["eventColor"]),
        )

    def toStructure(self) -> dict[str, Any]:
        return {
            "eventTimes": self.times.copy(),
            "eventLabels": list(self.labels),
            "eventColor": str(self.color),
        }

    @staticmethod
    def dsxy2figxy(*args: Any) -> np.ndarray:
        import matplotlib.pyplot as plt

        if not args:
            raise ValueError("dsxy2figxy expects at least one coordinate argument")
        if hasattr(args[0], "transData"):
            ax = args[0]
            rem = args[1:]
        else:
            ax = plt.gca()
            rem = args
        fig = ax.get_figure()
        if fig is None:
            raise RuntimeError("cannot transform without an active matplotlib figure")
        if len(rem) == 1:
            pos = np.asarray(rem[0], dtype=float).reshape(-1)
            if pos.size != 4:
                raise ValueError("single argument form expects [x, y, width, height]")
            x0, y0, w, h = pos.tolist()
            corners = np.array([[x0, y0], [x0 + w, y0 + h]], dtype=float)
            disp = ax.transData.transform(corners)
            fig_xy = fig.transFigure.inverted().transform(disp)
            out = np.array(
                [
                    fig_xy[0, 0],
                    fig_xy[0, 1],
                    fig_xy[1, 0] - fig_xy[0, 0],
                    fig_xy[1, 1] - fig_xy[0, 1],
                ],
                dtype=float,
            )
            return out
        if len(rem) != 2:
            raise ValueError("dsxy2figxy expects either (x,y) or ([x,y,w,h])")
        x = np.asarray(rem[0], dtype=float).reshape(-1)
        y = np.asarray(rem[1], dtype=float).reshape(-1)
        pts = np.column_stack([x, y])
        disp = ax.transData.transform(pts)
        fig = ax.get_figure()
        out = fig.transFigure.inverted().transform(disp)
        return out

    def plot(self, handle: Any = None, colorString: str | None = None) -> Any:
        import matplotlib.pyplot as plt

        if colorString is None or colorString == "":
            colorString = self.color
        _ = colorString  # MATLAB code computes this but plots fixed red lines.

        if handle is None:
            handles = [plt.gca()]
        elif isinstance(handle, (list, tuple, np.ndarray)):
            handles = list(handle)
        else:
            handles = [handle]

        h: Any = []
        for ax in handles:
            if ax is None:
                continue
            plt.sca(ax)
            v = ax.axis()
            times = np.vstack([self.times, self.times])
            y = np.vstack(
                [
                    np.full(self.times.size, float(v[2]), dtype=float),
                    np.full(self.times.size, float(v[3]), dtype=float),
                ]
            )
            if self.times.size:
                h = ax.plot(times, y, "r", linewidth=4)
            v = ax.axis()
            denom = float(v[1] - v[0])
            if denom == 0.0:
                continue
            for i, event_time in enumerate(self.times):
                if ((event_time - v[0]) / denom >= 0.0) and (event_time <= v[1]):
                    ax.text(
                        (event_time - v[0]) / denom - 0.02,
                        1.03,
                        self.labels[i],
                        rotation=0,
                        fontsize=10,
                        color=(0.0, 0.0, 0.0),
                        transform=ax.transAxes,
                    )
        return h

    @property
    def eventTimes(self) -> np.ndarray:
        return self.times

    @property
    def eventLabels(self) -> list[str]:
        return self.labels

    @property
    def eventColor(self) -> str:
        return self.color

    def getTimes(self) -> np.ndarray:
        return self.times.copy()

    def subset(self, start_s: float, end_s: float) -> "Events":
        out = super().subset(start_s, end_s)
        return Events(times=out.times, labels=out.labels, color=out.color)


class History(_HistoryBasis):
    @staticmethod
    def History(*args: Any, **kwargs: Any) -> _HistoryBasis:
        if len(args) == 1 and isinstance(args[0], dict):
            return History.fromStructure(args[0])
        return History(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _HistoryBasis:
        if "windowTimes" in payload:
            return History(
                bin_edges_s=np.asarray(payload["windowTimes"], dtype=float),
                min_time_s=payload.get("minTime"),
                max_time_s=payload.get("maxTime"),
            )
        return History(
            bin_edges_s=np.asarray(payload["bin_edges_s"], dtype=float),
            min_time_s=payload.get("min_time_s"),
            max_time_s=payload.get("max_time_s"),
        )

    def toStructure(self) -> dict[str, Any]:
        return {
            "windowTimes": self.bin_edges_s.copy(),
            "minTime": self.min_time_s,
            "maxTime": self.max_time_s,
            "bin_edges_s": self.bin_edges_s.copy(),
            "min_time_s": self.min_time_s,
            "max_time_s": self.max_time_s,
        }

    def setWindow(self, *args: Any) -> _HistoryBasis:
        if len(args) == 1:
            edges = np.sort(np.asarray(args[0], dtype=float).reshape(-1))
        elif len(args) == 3:
            t0 = float(args[0])
            tf = float(args[1])
            n_bins = int(args[2])
            if n_bins <= 0:
                raise ValueError("n_bins must be > 0")
            edges = np.linspace(t0, tf, n_bins + 1, dtype=float)
        else:
            raise ValueError("setWindow expects (edges) or (t0, tf, n_bins)")
        if edges.size < 2:
            raise ValueError("history edges must contain at least 2 entries")
        self.bin_edges_s = edges
        return self

    def toFilter(self, delta: float | None = None) -> np.ndarray:
        if delta is not None:
            delta_f = float(delta)
            if delta_f <= 0.0:
                raise ValueError("delta must be positive")
            tmin = self.bin_edges_s[:-1]
            tmax = self.bin_edges_s[1:]
            time_vec = np.arange(float(np.min(tmin)), float(np.max(tmax)) + delta_f / 2.0, delta_f)
            filt = np.zeros((tmax.size, time_vec.size), dtype=float)
            for i, (lo, hi) in enumerate(zip(tmin, tmax)):
                num_samples = int(np.ceil(hi / delta_f))
                start_sample = int(np.ceil(lo / delta_f)) + 1
                # MATLAB uses 1-based indices:
                #   idx1 = (start_sample:num_samples) + 1
                # Convert to 0-based Python by subtracting 1, yielding
                #   idx0 = start_sample:num_samples
                idx = np.arange(start_sample, num_samples + 1, dtype=int)
                idx = idx[(idx >= 0) & (idx < time_vec.size)]
                filt[i, idx] = 1.0
            return filt
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
        self._sig_rep_min_time: float | None = None
        self._sig_rep_max_time: float | None = None
        self._sig_rep_sample_rate_hz: float | None = None
        self._sig_rep_manual: bool = False
        self._mer: float | None = None
        self._sample_rate_hz: float = 1000.0

    @staticmethod
    def _to_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "_fieldnames"):
            return {name: getattr(payload, name) for name in payload._fieldnames}
        arr = np.asarray(payload, dtype=object)
        if arr.size == 1 and hasattr(arr.reshape(-1)[0], "_fieldnames"):
            s0 = arr.reshape(-1)[0]
            return {name: getattr(s0, name) for name in s0._fieldnames}
        raise ValueError("Unsupported structure payload")

    @staticmethod
    def _round_with_precision(values: np.ndarray, precision: int) -> np.ndarray:
        if precision < 0:
            return np.asarray(values, dtype=float)
        return np.round(np.asarray(values, dtype=float), int(precision))

    @staticmethod
    def _matlab_count_sigrep(
        spike_times: np.ndarray,
        bin_size_s: float,
        min_time_s: float,
        max_time_s: float,
    ) -> np.ndarray:
        if not np.isfinite(bin_size_s) or bin_size_s <= 0.0:
            raise ValueError("binSize_s must be positive")
        duration = float(max_time_s - min_time_s)
        if not np.isfinite(duration) or duration < 0.0:
            return np.array([], dtype=float)
        num_bins = int(np.floor(duration / float(bin_size_s) + 1.0))
        if num_bins < 1:
            num_bins = 1
        max_bins = int(1e6)
        if num_bins > max_bins:
            num_bins = max_bins

        time_vec = np.linspace(float(min_time_s), float(max_time_s), num_bins, dtype=float)
        if time_vec.size > 1:
            bin_width = float(np.mean(np.diff(time_vec)))
        else:
            bin_width = float(bin_size_s)
        window_times = np.concatenate(
            [
                np.array([float(min_time_s) - 0.5 * bin_width], dtype=float),
                time_vec + 0.5 * bin_width,
            ]
        )

        precision = int(max(0.0, 2.0 * np.ceil(np.log10(max(1.0 / float(bin_width), 1.0)))))
        spike_r = nspikeTrain._round_with_precision(spike_times, precision)
        window_r = nspikeTrain._round_with_precision(window_times, precision + 1)

        data = np.zeros(time_vec.size, dtype=float)
        lwindow = int(window_r.size)
        for j in range(time_vec.size):
            if j == (lwindow - 2):
                temp = spike_r[spike_r >= window_r[j]]
                data[j] = float(np.sum(temp <= window_r[j + 1]))
            elif (j + 1) > int(np.floor(lwindow / 2.0)):
                temp = spike_r[spike_r >= window_r[j]]
                data[j] = float(np.sum(temp < window_r[j + 1]))
            else:
                temp = spike_r[spike_r < window_r[j + 1]]
                data[j] = float(np.sum(temp >= window_r[j]))
        return data

    @staticmethod
    def nspikeTrain(*args: Any, **kwargs: Any) -> _SpikeTrain:
        if len(args) == 1 and isinstance(args[0], dict):
            return nspikeTrain.fromStructure(args[0])
        return nspikeTrain(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _SpikeTrain:
        structure = nspikeTrain._to_dict(payload)
        t_end_raw = structure.get("t_end", structure.get("maxTime"))
        spike_raw = structure.get("spike_times", structure.get("spikeTimes", []))
        spike_arr = np.asarray(spike_raw, dtype=float).reshape(-1)
        name_raw = structure.get("name", "unit")
        name_arr = np.asarray(name_raw, dtype=object).reshape(-1)
        unit_name = str(name_arr[0]) if name_arr.size else "unit"
        t_start_raw = structure.get("t_start", structure.get("minTime", 0.0))
        t_start_arr = np.asarray(t_start_raw, dtype=float).reshape(-1)
        t_start = float(t_start_arr[0]) if t_start_arr.size else 0.0
        out = nspikeTrain(
            spike_times=spike_arr,
            t_start=t_start,
            t_end=float(np.asarray(t_end_raw, dtype=float).reshape(-1)[0]) if t_end_raw is not None else None,
            name=unit_name,
        )
        sample_rate_raw = structure.get("sampleRate", structure.get("sample_rate_hz"))
        if sample_rate_raw is not None:
            sample_rate_arr = np.asarray(sample_rate_raw, dtype=float).reshape(-1)
            if sample_rate_arr.size:
                out._sample_rate_hz = float(sample_rate_arr[0])
        mer_raw = structure.get("MER", structure.get("mer"))
        if mer_raw is not None:
            mer_arr = np.asarray(mer_raw, dtype=float).reshape(-1)
            if mer_arr.size:
                out._mer = float(mer_arr[0])
        return out

    def toStructure(self) -> dict[str, Any]:
        sample_rate = float(self._sample_rate_hz)
        binwidth = 1.0 / sample_rate if sample_rate > 0.0 else np.inf
        return {
            "spike_times": self.spike_times.copy(),
            "t_start": float(self.t_start),
            "t_end": float(self.t_end) if self.t_end is not None else None,
            "name": str(self.name),
            "MER": self._mer,
            # MATLAB-compatible aliases
            "spikeTimes": self.spike_times.copy(),
            "sampleRate": sample_rate,
            "minTime": float(self.t_start),
            "maxTime": float(self.t_end) if self.t_end is not None else float(self.t_start),
            "xlabelval": "time",
            "xunits": "s",
            "yunits": "",
            "dataLabels": "",
            "binwidth": binwidth,
        }

    def setName(self, name: str) -> _SpikeTrain:
        self.name = str(name)
        return self

    def setMER(self, mer: float) -> _SpikeTrain:
        self._mer = float(mer)
        return self

    def setSigRep(self, sigRep: np.ndarray) -> _SpikeTrain:
        self._sig_rep = np.asarray(sigRep, dtype=float).copy()
        self._sig_rep_min_time = None
        self._sig_rep_max_time = None
        self._sig_rep_sample_rate_hz = None
        self._sig_rep_manual = True
        return self

    def clearSigRep(self) -> _SpikeTrain:
        self._sig_rep = None
        self._sig_rep_min_time = None
        self._sig_rep_max_time = None
        self._sig_rep_sample_rate_hz = None
        self._sig_rep_manual = False
        return self

    def getSigRep(
        self,
        binSize_s: float | None = None,
        mode: Literal["binary", "count"] = "binary",
        minTime_s: float | None = None,
        maxTime_s: float | None = None,
    ) -> np.ndarray:
        if binSize_s is None:
            if self._sample_rate_hz <= 0.0:
                binSize_s = 0.001
            else:
                binSize_s = 1.0 / float(self._sample_rate_hz)
        min_time = float(self.t_start) if minTime_s is None else float(minTime_s)
        max_time = float(self.t_end) if self.t_end is not None else float(self.t_start)
        if maxTime_s is not None:
            max_time = float(maxTime_s)
        if self._sig_rep is not None:
            if self._sig_rep_manual:
                cached = self._sig_rep.copy()
                return (cached > 0.0).astype(float) if mode == "binary" else cached
            same_rate = (
                self._sig_rep_sample_rate_hz is not None
                and np.isclose(float(self._sig_rep_sample_rate_hz), float(self._sample_rate_hz))
            )
            same_min = self._sig_rep_min_time is not None and np.isclose(float(self._sig_rep_min_time), min_time)
            same_max = self._sig_rep_max_time is not None and np.isclose(float(self._sig_rep_max_time), max_time)
            if same_rate and same_min and same_max:
                cached = self._sig_rep.copy()
                return (cached > 0.0).astype(float) if mode == "binary" else cached
        counts = self._matlab_count_sigrep(
            spike_times=np.asarray(self.spike_times, dtype=float).reshape(-1),
            bin_size_s=float(binSize_s),
            min_time_s=min_time,
            max_time_s=max_time,
        )
        self._sig_rep = counts.copy()
        self._sig_rep_min_time = float(min_time)
        self._sig_rep_max_time = float(max_time)
        self._sig_rep_sample_rate_hz = float(self._sample_rate_hz)
        self._sig_rep_manual = False
        return (counts > 0.0).astype(float) if mode == "binary" else counts

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
        return []

    def getLStatistic(self) -> float:
        isi = self.getISIs()
        if isi.size == 0:
            return float(np.nan)
        mu = float(np.mean(isi))
        if not np.isfinite(mu) or mu <= 0.0:
            return float(np.nan)
        duration = float((self.t_end if self.t_end is not None else self.t_start) - self.t_start)
        if not np.isfinite(duration) or duration <= 0.0:
            return float(np.nan)
        max_bins = float(1e6)
        est_bins = duration / mu + 1.0
        if np.isfinite(est_bins) and est_bins > max_bins:
            mu = duration / (max_bins - 1.0)
        pt = self.getSigRep(binSize_s=mu, mode="count")
        return float(np.unique(pt).size)

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
        self._sample_rate_hz = float(sampleRate)
        self.clearSigRep()
        return self

    def restoreToOriginal(self) -> _SpikeTrain:
        self.spike_times = self._original_spike_times.copy()
        if self.spike_times.size:
            self.t_start = float(np.min(self.spike_times))
            self.t_end = float(np.max(self.spike_times))
        else:
            self.t_start = float(self._original_t_start)
            self.t_end = float(self._original_t_end) if self._original_t_end is not None else None
        self.name = str(self._original_name)
        self.clearSigRep()
        return self

    def partitionNST(self, partitionEdges_s: np.ndarray | list[float]) -> list[_SpikeTrain]:
        edges = np.asarray(partitionEdges_s, dtype=float).reshape(-1)
        if edges.size < 2:
            raise ValueError("partition edges must contain at least two values")
        out: list[_SpikeTrain] = []
        for i in range(edges.size - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if i == edges.size - 2:
                mask = (self.spike_times >= lo) & (self.spike_times <= hi)
            else:
                mask = (self.spike_times >= lo) & (self.spike_times < hi)
            subset = self.spike_times[mask] - lo
            out.append(
                nspikeTrain(spike_times=subset, t_start=0.0, t_end=hi - lo, name=f"{self.name}_{i+1}")
            )
        return out

    def shiftTime(self, offset_s: float) -> _SpikeTrain:
        self.shift_time(offset_s)
        return self

    def setMinTime(self, t_min: float) -> _SpikeTrain:
        self.t_start = float(t_min)
        return self

    def setMaxTime(self, t_max: float) -> _SpikeTrain:
        self.t_end = float(t_max)
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

    def _selected_indices(self) -> list[int]:
        if self._neuron_mask is None:
            return list(range(self.n_units))
        return list(self._neuron_mask)

    def getBinnedMatrix(
        self, binSize_s: float, mode: Literal["binary", "count"] = "binary"
    ) -> tuple[np.ndarray, np.ndarray]:
        if binSize_s <= 0.0:
            raise ValueError("binSize_s must be positive")
        min_time = float(min(train.t_start for train in self.trains))
        max_time = float(max(train.t_end if train.t_end is not None else train.t_start for train in self.trains))
        selected = self._selected_indices()
        out_rows: list[np.ndarray] = []
        time_vec: np.ndarray | None = None
        for idx in selected:
            train = self.getNST(idx)
            counts = train.getSigRep(
                binSize_s=float(binSize_s),
                mode="count",
                minTime_s=min_time,
                maxTime_s=max_time,
            )
            if mode == "binary":
                counts = (counts > 0.0).astype(float)
            out_rows.append(np.asarray(counts, dtype=float).reshape(-1))
            if time_vec is None:
                n_bins = out_rows[-1].size
                time_vec = np.linspace(min_time, max_time, n_bins, dtype=float)
        if time_vec is None:
            time_vec = np.array([], dtype=float)
            mat = np.zeros((0, 0), dtype=float)
        else:
            n_bins = time_vec.size
            mat = np.zeros((len(out_rows), n_bins), dtype=float)
            for i, row in enumerate(out_rows):
                if row.size == n_bins:
                    mat[i, :] = row
                elif row.size > n_bins:
                    mat[i, :] = row[:n_bins]
                else:
                    mat[i, : row.size] = row
        return time_vec, mat

    def merge(self, other: _SpikeTrainCollection) -> _SpikeTrainCollection:
        merged = super().merge(other)
        return nstColl(merged.trains)

    def getFirstSpikeTime(self) -> float:
        return float(min(train.t_start for train in self.trains))

    def getLastSpikeTime(self) -> float:
        return float(max(train.t_end if train.t_end is not None else train.t_start for train in self.trains))

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
        _time, mat = self.getBinnedMatrix(binSize_s=binSize_s, mode=mode)
        return mat.T

    def toSpikeTrain(self, name: str = "merged") -> nspikeTrain:
        selected = self._selected_indices()
        if not selected:
            selected = list(range(self.n_units))
        delta = 1.0 / max(float(self.findMaxSampleRate()), 1.0)
        spike_times: list[np.ndarray] = []
        offset = 0.0
        first_train = self.getNST(selected[0])
        trial_name = first_train.name if first_train.name else name
        spike_times.append(np.asarray(first_train.spike_times, dtype=float).reshape(-1))
        for i in range(1, len(selected)):
            prev = self.getNST(selected[i - 1])
            prev_max = float(prev.t_end) if prev.t_end is not None else float(prev.t_start)
            offset = offset + prev_max + delta
            curr = self.getNST(selected[i])
            spike_times.append(np.asarray(curr.spike_times, dtype=float).reshape(-1) + offset)
        merged_vec = np.concatenate(spike_times) if spike_times else np.array([], dtype=float)
        min_time = float(first_train.t_start)
        max_time = float(max(train.t_end if train.t_end is not None else train.t_start for train in self.trains))
        return nspikeTrain(
            spike_times=np.asarray(merged_vec, dtype=float),
            t_start=min_time,
            t_end=max_time * len(selected),
            name=str(trial_name),
        )

    def shiftTime(self, offset_s: float) -> _SpikeTrainCollection:
        self.shift_time(offset_s)
        return self

    def setMinTime(self, t_min: float) -> _SpikeTrainCollection:
        for train in self.trains:
            train.t_start = float(t_min)
        return self

    def setMaxTime(self, t_max: float) -> _SpikeTrainCollection:
        for train in self.trains:
            train.t_end = float(t_max)
        return self

    def toStructure(self) -> dict[str, Any]:
        trains = [self.getNST(i).toStructure() for i in range(self.n_units)]
        min_time = float(min(train.t_start for train in self.trains))
        max_time = float(max(train.t_end if train.t_end is not None else train.t_start for train in self.trains))
        sample_rate = float(max(getattr(train, "_sample_rate_hz", 1000.0) for train in self.trains))
        neuron_mask = np.ones(self.n_units, dtype=float)
        if self._neuron_mask is not None:
            neuron_mask = np.zeros(self.n_units, dtype=float)
            neuron_mask[np.asarray(self._neuron_mask, dtype=int)] = 1.0
        out = {
            "trains": trains,
            # MATLAB-compatible fields
            "nstrain": trains,
            "numSpikeTrains": int(self.n_units),
            "minTime": min_time,
            "maxTime": max_time,
            "sampleRate": sample_rate,
            "neuronMask": neuron_mask,
            "neuronNames": [str(train.name) for train in self.trains],
            "neighbors": self._neighbors if self._neighbors is not None else [],
        }
        return out

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> _SpikeTrainCollection:
        if hasattr(payload, "_fieldnames"):
            payload = {name: getattr(payload, name) for name in payload._fieldnames}
        source = payload.get("trains", payload.get("nstrain", []))

        def _iter_train_entries(node: Any) -> list[Any]:
            if isinstance(node, nspikeTrain):
                return [node]
            if hasattr(node, "_fieldnames") or isinstance(node, dict):
                return [node]
            if isinstance(node, np.ndarray):
                out: list[Any] = []
                for item in node.reshape(-1):
                    out.extend(_iter_train_entries(item))
                return out
            if isinstance(node, (list, tuple)):
                out = []
                for item in node:
                    out.extend(_iter_train_entries(item))
                return out
            return []

        rows = _iter_train_entries(source)
        trains: list[_SpikeTrain] = []
        for i, row in enumerate(rows):
            if isinstance(row, nspikeTrain):
                trains.append(row.nstCopy())
                continue
            if hasattr(row, "_fieldnames"):
                row_dict = {name: getattr(row, name) for name in row._fieldnames}
            elif isinstance(row, dict):
                row_dict = row
            else:
                continue
            trains.append(cast(_SpikeTrain, nspikeTrain.fromStructure(row_dict)))
        if not trains:
            raise ValueError("fromStructure requires at least one train")
        coll = nstColl(cast(list[_SpikeTrain], trains))
        neigh = payload.get("neighbors")
        if neigh is not None and np.asarray(neigh, dtype=object).size:
            coll.setNeighbors(neigh)
        mask = payload.get("neuronMask")
        if mask is not None and np.asarray(mask, dtype=float).size:
            mask_arr = np.asarray(mask, dtype=float).reshape(-1)
            if mask_arr.size == coll.n_units:
                coll._neuron_mask = list(np.where(mask_arr > 0)[0].astype(int))
        return coll

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
        _, mat = self.getBinnedMatrix(binSize_s=binSize_s, mode="count")
        return bool(np.all((mat == 0) | (mat == 1)))

    def BinarySigRep(self, binSize_s: float = 0.001) -> bool:
        return self.isSigRepBinary(binSize_s=binSize_s)

    def psth(self, binSize_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        if binSize_s <= 0.0:
            raise ValueError("binSize_s must be positive")
        selected = self._selected_indices()
        if not selected:
            selected = list(range(self.n_units))
        min_time = float(min(self.trains[i].t_start for i in selected))
        max_time = float(max(self.trains[i].t_end if self.trains[i].t_end is not None else self.trains[i].t_start for i in selected))
        window_times = np.arange(min_time, max_time + float(binSize_s), float(binSize_s), dtype=float)
        if window_times.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        if not np.any(np.isclose(window_times, max_time)):
            window_times = np.append(window_times, max_time)
        psth_counts = np.zeros(max(window_times.size - 1, 0), dtype=float)
        for i in selected:
            spikes = np.asarray(self.trains[i].spike_times, dtype=float).reshape(-1)
            if spikes.size:
                counts, _ = np.histogram(spikes, bins=window_times)
                psth_counts = psth_counts + counts.astype(float)
        denom = float(binSize_s) * max(len(selected), 1)
        psth_rate = psth_counts / denom
        time_centers = 0.5 * (window_times[1:] + window_times[:-1])
        return time_centers, psth_rate

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
        vals: list[float] = []
        for train in self.trains:
            vals.append(float(getattr(train, "_sample_rate_hz", 1000.0)))
        return float(np.max(np.asarray(vals, dtype=float))) if vals else float("-inf")

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
            if len(idx_raw) == self.n_units and all(v in (0, 1) for v in idx_raw):
                idx = [i for i, flag in enumerate(idx_raw) if flag == 1]
                self._neuron_mask = idx
                return self
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
        min_time = float(min(train.t_start for train in self.trains))
        max_time = float(max(train.t_end if train.t_end is not None else train.t_start for train in self.trains))
        for i, train in enumerate(self.trains):
            if isinstance(train, nspikeTrain):
                curr = train
            else:
                curr = nspikeTrain(
                    spike_times=np.asarray(train.spike_times, dtype=float).copy(),
                    t_start=float(train.t_start),
                    t_end=float(train.t_end) if train.t_end is not None else None,
                    name=str(train.name),
                )
                self.trains[i] = curr
            curr.resample(float(sampleRate))
            curr.setMinTime(min_time)
            curr.setMaxTime(max_time)
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
        _t, mat = self.getBinnedMatrix(binSize_s=binSize_s, mode="count")
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
        # Supports both Python form:
        #   generateUnitImpulseBasis(basisWidth_s, sampleRate_hz, totalTime_s=1.0, name=...)
        # and MATLAB form:
        #   generateUnitImpulseBasis(basisWidth_s, minTime_s, maxTime_s, sampleRate_hz)
        name = str(kwargs.pop("name", "unit_impulse_basis"))
        min_time = float(kwargs.pop("minTime_s", kwargs.pop("min_time_s", 0.0)))
        sample_rate: float | None = kwargs.pop("sampleRate_hz", kwargs.pop("sample_rate_hz", None))
        total_time = kwargs.pop("totalTime_s", kwargs.pop("total_time_s", None))
        max_time = kwargs.pop("maxTime_s", kwargs.pop("max_time_s", None))
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unknown keyword arguments: {unknown}")

        if len(args) == 0:
            pass
        elif len(args) == 1:
            sample_rate = float(args[0])
        elif len(args) == 2:
            # MATLAB form without sampleRate:
            #   generateUnitImpulseBasis(basisWidth, minTime, maxTime)
            min_time = float(args[0])
            max_time = float(args[1])
            total_time = float(max_time) - float(min_time)
        elif len(args) == 3:
            min_time = float(args[0])
            max_time = float(args[1])
            # MATLAB form with explicit sampleRate:
            #   generateUnitImpulseBasis(basisWidth, minTime, maxTime, sampleRate)
            sample_rate = float(args[2])
            total_time = float(max_time) - float(min_time)
        else:
            raise TypeError("generateUnitImpulseBasis accepts at most 4 positional arguments")

        if max_time is not None and total_time is None:
            total_time = float(max_time) - float(min_time)
        if total_time is None:
            total_time = 1.0
        if sample_rate is None:
            sample_rate = 1000.0

        if basisWidth_s <= 0.0:
            raise ValueError("basisWidth_s must be positive")
        if sample_rate <= 0.0:
            raise ValueError("sampleRate_hz must be positive")
        start = float(min_time)
        stop = float(min_time) + float(total_time)
        step = float(basisWidth_s)
        window_times = np.arange(start, stop + 0.5 * step, step, dtype=float)
        if window_times.size == 0:
            window_times = np.array([start, stop], dtype=float)
        if not np.any(np.isclose(window_times, stop)):
            window_times = np.append(window_times, stop)

        dt = 1.0 / float(sample_rate)
        time = np.arange(start, stop + 0.5 * dt, dt, dtype=float)
        num_basis = max(int(window_times.size - 1), 1)
        basis = np.zeros((time.size, num_basis), dtype=float)
        for j in range(num_basis):
            lo = float(window_times[j])
            hi = float(window_times[j + 1])
            if j == (num_basis - 1):
                mask = (time >= lo) & (time <= hi)
            else:
                mask = (time >= lo) & (time < hi)
            basis[mask, j] = 1.0
        labels = [f"basis_{j+1}" for j in range(num_basis)]
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
        self.covShift = 0.0
        self.originalSampleRate = float(self.covariates[0].sample_rate_hz)
        self.originalMinTime = float(np.min(self.covariates[0].time))
        self.originalMaxTime = float(np.max(self.covariates[0].time))
        self._refresh_covcoll_state()

    def _refresh_covcoll_state(self) -> None:
        self.covArray = list(self.covariates)
        self.numCov = int(len(self.covariates))
        self.covDimensions = [int(cov.n_channels) for cov in self.covariates]
        self.sampleRate = float(self.covariates[0].sample_rate_hz)
        self.minTime = float(min(np.min(cov.time) for cov in self.covariates)) + float(self._cov_shift)
        self.maxTime = float(max(np.max(cov.time) for cov in self.covariates)) + float(self._cov_shift)
        active = list(getattr(self, "_cov_mask", list(range(self.numCov))))
        self.covMask = []
        for i, cov in enumerate(self.covariates):
            if i in active:
                self.covMask.append(np.ones((cov.n_channels,), dtype=int).tolist())
            else:
                self.covMask.append(np.zeros((cov.n_channels,), dtype=int).tolist())

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
        # CovColl stores shift at the collection level; expose shifted time.
        return np.asarray(self.time, dtype=float) + float(self._cov_shift)

    def getDesignMatrix(self) -> tuple[np.ndarray, list[str]]:
        return self.design_matrix()

    def copy(self) -> "CovColl":
        copied = super().copy()
        out = CovColl(copied.covariates)
        out._cov_shift = float(self._cov_shift)
        out.covShift = float(self.covShift)
        out.originalSampleRate = float(self.originalSampleRate)
        out.originalMinTime = float(self.originalMinTime)
        out.originalMaxTime = float(self.originalMaxTime)
        out._refresh_covcoll_state()
        return out

    def addToColl(self, cov: _Covariate) -> "CovColl":
        self.add_to_coll(cov)
        self._refresh_covcoll_state()
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
        self._refresh_covcoll_state()
        return self

    def addCovCollection(self, other: _CovariateCollection) -> "CovColl":
        for cov in other.covariates:
            self.add_to_coll(cov)
        self._refresh_covcoll_state()
        return self

    def setMinTime(self, t_min: float) -> "CovColl":
        for cov in self.covariates:
            cov.set_min_time(t_min)
        self._refresh_covcoll_state()
        return self

    def setMaxTime(self, t_max: float) -> "CovColl":
        for cov in self.covariates:
            cov.set_max_time(t_max)
        self._refresh_covcoll_state()
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
        self._refresh_covcoll_state()
        return self

    def resample(self, sampleRate: float) -> "CovColl":
        return self.setSampleRate(sampleRate)

    def enforceSampleRate(self, sampleRate: float) -> "CovColl":
        return self.setSampleRate(sampleRate)

    def updateTimes(self) -> "CovColl":
        _ = self.time
        return self

    def toStructure(self) -> dict[str, Any]:
        self.resetMask()
        self._refresh_covcoll_state()
        cov_structs = [cov.to_structure() for cov in self.covariates]
        return {
            "covArray": cov_structs,
            "covDimensions": list(self.covDimensions),
            "numCov": int(self.numCov),
            "minTime": float(self.minTime),
            "maxTime": float(self.maxTime),
            "covMask": [list(mask) for mask in self.covMask],
            "covShift": float(self.covShift),
            "sampleRate": float(self.sampleRate),
            "originalSampleRate": float(self.originalSampleRate),
            "originalMinTime": float(self.originalMinTime),
            "originalMaxTime": float(self.originalMaxTime),
            # Backward-compatible alias used by existing Python tests.
            "covariates": cov_structs,
        }

    def dataToStructure(self) -> dict[str, Any]:
        return self.toStructure()

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> "CovColl":
        rows = payload.get("covArray", payload.get("covariates", []))

        def _iter_cov_entries(node: Any) -> list[Any]:
            if isinstance(node, dict):
                return [node]
            if hasattr(node, "_fieldnames"):
                return [{name: getattr(node, name) for name in node._fieldnames}]
            if isinstance(node, np.ndarray):
                out: list[Any] = []
                for item in node.reshape(-1):
                    out.extend(_iter_cov_entries(item))
                return out
            if isinstance(node, (list, tuple)):
                out: list[Any] = []
                for item in node:
                    out.extend(_iter_cov_entries(item))
                return out
            return []

        rows_py = _to_python_cell(rows)
        rows_flat = _iter_cov_entries(rows_py)
        covs = [Covariate.fromStructure(cast(dict[str, Any], row)) for row in rows_flat]
        if not covs:
            raise ValueError("fromStructure requires at least one covariate")
        out = CovColl(cast(list[_Covariate], covs))
        if "minTime" in payload and not _is_empty_like(payload.get("minTime")):
            out.setMinTime(float(np.asarray(payload["minTime"], dtype=float).reshape(-1)[0]))
        if "maxTime" in payload and not _is_empty_like(payload.get("maxTime")):
            out.setMaxTime(float(np.asarray(payload["maxTime"], dtype=float).reshape(-1)[0]))
        if "covShift" in payload and not _is_empty_like(payload.get("covShift")):
            out._cov_shift = float(np.asarray(payload["covShift"], dtype=float).reshape(-1)[0])
            out.covShift = float(out._cov_shift)
        out._refresh_covcoll_state()
        return out

    def setMask(self, selector: list[int] | list[str]) -> "CovColl":
        if selector and isinstance(selector[0], str):
            idx = self.get_cov_indices_from_names(cast(list[str], selector))
        else:
            idx = [int(i) for i in cast(list[int], selector)]
        self._cov_mask = idx
        self._refresh_covcoll_state()
        return self

    def resetMask(self) -> "CovColl":
        self._cov_mask = list(range(len(self.covariates)))
        self._refresh_covcoll_state()
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
            self._refresh_covcoll_state()
            return self
        idx = self.get_cov_ind_from_name(selector)
        del self.covariates[idx]
        self._refresh_covcoll_state()
        return self

    def removeFromColl(self, selector: int | str) -> "CovColl":
        return self.removeCovariate(selector)

    def removeFromCollByIndices(self, indices: list[int]) -> "CovColl":
        for i in sorted(set(indices), reverse=True):
            del self.covariates[i]
        self._refresh_covcoll_state()
        return self

    def maskAwayCov(self, selector: int | str | list[int] | list[str] | np.ndarray) -> "CovColl":
        remaining = self.generateRemainingIndex(selector)
        self._cov_mask = remaining
        self._refresh_covcoll_state()
        return self

    def maskAwayOnlyCov(self, selector: int | str | list[int] | list[str] | np.ndarray) -> "CovColl":
        return self.maskAwayCov(selector)

    def maskAwayAllExcept(self, selector: int | str | list[int] | list[str] | np.ndarray) -> "CovColl":
        self._cov_mask = self.covIndFromSelector(selector)
        self._refresh_covcoll_state()
        return self

    def setCovShift(self, shift_s: float) -> "CovColl":
        shift = float(shift_s)
        self.resetCovShift()
        self._cov_shift = shift
        self.covShift = shift
        self._refresh_covcoll_state()
        return self

    def resetCovShift(self) -> "CovColl":
        self._cov_shift = 0.0
        self.covShift = 0.0
        self._refresh_covcoll_state()
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
        self.covShift = 0.0
        self._cov_mask = list(range(len(self.covariates)))
        if not _is_empty_like(self.originalSampleRate):
            self.setSampleRate(float(self.originalSampleRate))
        if not _is_empty_like(self.originalMinTime):
            self.setMinTime(float(self.originalMinTime))
        else:
            self.setMinTime(float(self.findMinTime()))
        if not _is_empty_like(self.originalMaxTime):
            self.setMaxTime(float(self.originalMaxTime))
        else:
            self.setMaxTime(float(self.findMaxTime()))
        self._refresh_covcoll_state()
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
        covMask: Any | None = None,
        sampleRate: Any | None = None,
        history: Any | None = None,
        ensCovHist: Any | None = None,
        ensCovMask: Any | None = None,
        covLag: Any | None = None,
        name: str = "",
        *,
        covariateLabels: list[str] | None = None,
        Fs: Any | None = None,
        fitType: str = "poisson",
        **kwargs: Any,
    ) -> None:
        # MATLAB reference: TrialConfig.m constructor
        # (covMask,sampleRate,history,ensCovHist,ensCovMask,covLag,name)
        # Also keep Python-side keyword aliases used throughout nSTAT-python.
        if covMask is None:
            covMask = kwargs.pop("covariate_labels", covariateLabels)
        if sampleRate is None:
            sampleRate = kwargs.pop("sample_rate_hz", Fs)
        fit_type = str(kwargs.pop("fit_type", fitType))

        self.covMask = [] if _is_empty_like(covMask) else _to_python_cell(covMask)
        self.sampleRate = [] if _is_empty_like(sampleRate) else float(np.asarray(sampleRate).reshape(-1)[0])
        self.history = [] if _is_empty_like(history) else _to_python_cell(history)
        self.ensCovHist = [] if _is_empty_like(ensCovHist) else _to_python_cell(ensCovHist)
        self.ensCovMask = [] if _is_empty_like(ensCovMask) else _to_python_cell(ensCovMask)
        self.covLag = [] if _is_empty_like(covLag) else _to_python_cell(covLag)
        self.name = str(name)

        covariate_labels = self._coerce_covariate_labels(self.covMask)
        sample_rate_hz = float(self.sampleRate) if not _is_empty_like(self.sampleRate) else 1000.0
        super().__init__(
            covariate_labels=covariate_labels,
            sample_rate_hz=sample_rate_hz,
            fit_type=fit_type,
            name=self.name,
        )

    def getFitType(self) -> str:
        return self.fit_type

    def getSampleRate(self) -> Any:
        return self.sampleRate

    def getCovariateLabels(self) -> list[str]:
        return self._coerce_covariate_labels(self.covMask)

    def getName(self) -> str:
        return self.name

    def setName(self, name: str) -> "TrialConfig":
        self.name = str(name)
        return self

    def toStructure(self) -> dict[str, Any]:
        return {
            "covMask": self._to_structure_cell(self.covMask),
            "sampleRate": [] if _is_empty_like(self.sampleRate) else float(self.sampleRate),
            "history": self._to_structure_cell(self.history),
            "ensCovHist": self._to_structure_cell(self.ensCovHist),
            "ensCovMask": self._to_structure_cell(self.ensCovMask),
            "covLag": self._to_structure_cell(self.covLag),
            "name": self.name,
        }

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> "TrialConfig":
        if isinstance(payload, list):
            if len(payload) == 1 and isinstance(payload[0], dict):
                payload = payload[0]
            else:
                raise TypeError("TrialConfig.fromStructure expects a dict-like payload")
        # MATLAB reference: TrialConfig.m fromStructure static method.
        # NOTE: MATLAB currently calls TrialConfig with six args and therefore
        # shifts ensCovMask/covLag positions:
        # TrialConfig(covMask,sampleRate,history,ensCovHist,covLag,name)
        # We preserve this behavior for strict parity.
        return TrialConfig(
            payload.get("covMask", []),
            payload.get("sampleRate", []),
            payload.get("history", []),
            payload.get("ensCovHist", []),
            payload.get("covLag", []),
            payload.get("name", ""),
        )

    def setConfig(self, trial: "Trial") -> "TrialConfig":
        if not _is_empty_like(self.history):
            trial.setHistory(self.history)
        else:
            trial.resetHistory()

        if not _is_empty_like(self.sampleRate):
            trial_sample_rate = getattr(trial, "sampleRate", None)
            if trial_sample_rate is None or not np.isclose(float(trial_sample_rate), float(self.sampleRate)):
                trial.setSampleRate(float(self.sampleRate))

        trial.setCovMask(self.covMask)

        if not _is_empty_like(self.covLag):
            trial.shiftCovariates(float(np.asarray(self.covLag).reshape(-1)[0]))

        if not _is_empty_like(self.ensCovHist):
            trial.setEnsCovHist(self.ensCovHist)
            trial.setEnsCovMask(self.ensCovMask)
        else:
            trial.setEnsCovHist([])
            trial.resetEnsCovMask()
        return self

    @staticmethod
    def _coerce_covariate_labels(cov_mask: Any) -> list[str]:
        if _is_empty_like(cov_mask):
            return []
        labels: list[str] = []
        values = _to_python_cell(cov_mask)
        for item in values:
            if isinstance(item, (list, tuple, np.ndarray)):
                inner = _to_python_cell(item)
                labels.extend(str(v) for v in inner)
            else:
                labels.append(str(item))
        return labels

    @staticmethod
    def _to_structure_cell(value: Any) -> Any:
        if _is_empty_like(value):
            return []
        return _to_python_cell(value)


class ConfigColl(_ConfigCollection):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.numConfigs = int(len(self.configs))
        self.configArray = list(self.configs)
        self.configNames = [
            str(cfg.name) if str(cfg.name) != "" else f"Fit {i+1}"
            for i, cfg in enumerate(self.configs)
        ]

    @staticmethod
    def ConfigColl(*args: Any, **kwargs: Any) -> _ConfigCollection:
        if len(args) == 1 and isinstance(args[0], dict):
            return ConfigColl.fromStructure(args[0])
        return ConfigColl(*args, **kwargs)

    @staticmethod
    def fromStructure(payload: dict[str, Any] | list[dict[str, Any]]) -> _ConfigCollection:
        if isinstance(payload, dict):
            raw_entries = list(payload.get("configArray", payload.get("configs", [])))
        else:
            raw_entries = list(payload)
        entries: list[dict[str, Any]] = []
        for entry in raw_entries:
            parsed = _to_python_cell(entry)
            if isinstance(parsed, list):
                if parsed and all(isinstance(v, dict) for v in parsed):
                    entries.extend(cast(list[dict[str, Any]], parsed))
                elif len(parsed) == 1 and isinstance(parsed[0], dict):
                    entries.append(cast(dict[str, Any], parsed[0]))
            elif isinstance(parsed, dict):
                entries.append(parsed)
        if not entries:
            raise ValueError("fromStructure requires at least one configuration entry")
        configs = cast(list[_TrialConfig], [TrialConfig.fromStructure(entry) for entry in entries])
        out = ConfigColl(configs)
        # MATLAB fromStructure ignores stored configNames and rebuilds names
        # from TrialConfig objects via constructor/addConfig logic.
        return out

    def toStructure(self) -> dict[str, Any]:
        config_array = [
            TrialConfig(
                covMask=getattr(cfg, "covMask", list(cfg.covariate_labels)),
                sampleRate=getattr(cfg, "sampleRate", float(cfg.sample_rate_hz)),
                history=getattr(cfg, "history", []),
                ensCovHist=getattr(cfg, "ensCovHist", []),
                ensCovMask=getattr(cfg, "ensCovMask", []),
                covLag=getattr(cfg, "covLag", []),
                fitType=str(cfg.fit_type),
                name=str(cfg.name),
            ).toStructure()
            for cfg in self.configs
        ]
        return {
            "numConfigs": int(len(self.configs)),
            "configNames": list(self.getConfigNames()),
            "configArray": config_array,
            # Backward-compatible alias used by existing Python tests/utilities.
            "configs": config_array,
        }

    def addConfig(self, config: _TrialConfig) -> _ConfigCollection:
        if str(getattr(config, "name", "")) == "":
            config.name = f"Fit {len(self.configs) + 1}"
        self.configs.append(config)
        self.numConfigs = int(len(self.configs))
        self.configArray = list(self.configs)
        self.configNames.append(str(config.name) if str(config.name) != "" else f"Fit {self.numConfigs}")
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
        self.configArray[idx - 1] = config
        if str(getattr(config, "name", "")) != "":
            self.configNames[idx - 1] = str(config.name)
        return self

    def getConfigNames(self) -> list[str]:
        return list(self.configNames)

    def setConfigNames(self, names: list[str]) -> _ConfigCollection:
        if len(names) != len(self.configs):
            raise ValueError("names length must match number of configs")
        self.configNames = [str(name) for name in names]
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
        return list(self.configArray)


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
        self._ensure_trial_state()
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
        self._ensure_trial_state()
        spikes_struct = nstColl(self.spikes.trains).toStructure()
        cov_struct = CovColl(self.covariates.covariates).toStructure()

        cov_mask_idx = list(getattr(self, "_cov_mask", list(range(len(self.covariates.covariates)))))
        cov_mask = []
        for i, cov in enumerate(self.covariates.covariates):
            dim = int(cov.n_channels)
            if i in cov_mask_idx:
                cov_mask.append(np.ones((dim,), dtype=int).tolist())
            else:
                cov_mask.append(np.zeros((dim,), dtype=int).tolist())

        neuron_mask_idx = list(getattr(self, "_neuron_mask", list(range(self.spikes.n_units))))
        neuron_mask = np.zeros((self.spikes.n_units,), dtype=int)
        if neuron_mask_idx:
            neuron_mask[np.asarray(neuron_mask_idx, dtype=int)] = 1

        partition = self.getTrialPartition()
        if isinstance(partition, dict) and partition:
            training_window = list(partition.get("training", (self.findMinTime(), self.findMaxTime())))
            validation_window = list(partition.get("validation", (self.findMaxTime(), self.findMaxTime())))
        else:
            training_window = [self.findMinTime(), self.findMaxTime()]
            validation_window = [self.findMaxTime(), self.findMaxTime()]

        ev_obj = self.getEvents()
        ev_payload: Any = []
        if ev_obj is not None and hasattr(ev_obj, "toStructure"):
            ev_payload = ev_obj.toStructure()

        hist_payload: Any = []
        if getattr(self, "_history", None) is not None and hasattr(self._history, "toStructure"):
            hist_payload = self._history.toStructure()

        ens_hist_payload: Any = []
        if getattr(self, "_ens_cov_hist", None) is not None and hasattr(self._ens_cov_hist, "toStructure"):
            ens_hist_payload = self._ens_cov_hist.toStructure()

        return {
            # Python-native keys
            "spikes": spikes_struct,
            "covariates": cov_struct,
            "trial_partition": partition,
            # MATLAB-style keys
            "nspikeColl": spikes_struct,
            "covarColl": cov_struct,
            "ev": ev_payload,
            "history": hist_payload,
            "ensCovHist": ens_hist_payload,
            "sampleRate": float(self.findMinSampleRate()),
            "minTime": float(self.findMinTime()),
            "maxTime": float(self.findMaxTime()),
            "covMask": cov_mask,
            "ensCovMask": getattr(self, "_ens_cov_mask", list(range(self.spikes.n_units))),
            "neuronMask": neuron_mask,
            "trainingWindow": training_window,
            "validationWindow": validation_window,
        }

    @staticmethod
    def fromStructure(payload: dict[str, Any]) -> "Trial":
        def _unwrap_single(node: Any) -> Any:
            if isinstance(node, list) and len(node) == 1:
                return node[0]
            if isinstance(node, np.ndarray):
                arr = np.asarray(node, dtype=object).reshape(-1)
                if arr.size == 1:
                    return arr[0]
            return node

        if hasattr(payload, "_fieldnames"):
            payload = {name: getattr(payload, name) for name in payload._fieldnames}
        spikes_payload = _unwrap_single(payload.get("spikes", payload.get("nspikeColl")))
        covs_payload = _unwrap_single(payload.get("covariates", payload.get("covarColl")))
        if spikes_payload is None or covs_payload is None:
            raise ValueError("fromStructure requires spikes/nspikeColl and covariates/covarColl")

        spikes = nstColl.fromStructure(spikes_payload)
        covs = CovColl.fromStructure(covs_payload)
        trial = Trial(spikes=spikes, covariates=covs)

        if "minTime" in payload and not _is_empty_like(payload.get("minTime")):
            trial.setMinTime(float(np.asarray(payload["minTime"], dtype=float).reshape(-1)[0]))
        if "maxTime" in payload and not _is_empty_like(payload.get("maxTime")):
            trial.setMaxTime(float(np.asarray(payload["maxTime"], dtype=float).reshape(-1)[0]))

        if "trial_partition" in payload and isinstance(payload["trial_partition"], dict):
            trial.setTrialPartition(dict(payload["trial_partition"]))
        elif ("trainingWindow" in payload) and ("validationWindow" in payload):
            training = np.asarray(payload.get("trainingWindow"), dtype=float).reshape(-1)
            validation = np.asarray(payload.get("validationWindow"), dtype=float).reshape(-1)
            if training.size >= 2 and validation.size >= 2:
                trial.setTrialPartition(
                    {
                        "training": (float(training[0]), float(training[1])),
                        "validation": (float(validation[0]), float(validation[1])),
                    }
                )

        if "covMask" in payload and not _is_empty_like(payload.get("covMask")):
            raw_cov_mask = _to_python_cell(payload["covMask"])
            if isinstance(raw_cov_mask, list):
                cov_idx: list[int] = []
                for i, row in enumerate(raw_cov_mask):
                    arr = np.asarray(row, dtype=float).reshape(-1)
                    if arr.size and np.any(arr > 0):
                        cov_idx.append(i)
                if cov_idx:
                    trial._cov_mask = cov_idx

        if "neuronMask" in payload and not _is_empty_like(payload.get("neuronMask")):
            arr = np.asarray(payload["neuronMask"], dtype=float).reshape(-1)
            if arr.size == trial.spikes.n_units:
                trial._neuron_mask = [int(i) for i in np.where(arr > 0)[0]]

        if "ensCovMask" in payload and not _is_empty_like(payload.get("ensCovMask")):
            ens = _to_python_cell(payload["ensCovMask"])
            trial._ens_cov_mask = ens if isinstance(ens, list) else trial._ens_cov_mask

        if "ev" in payload and not _is_empty_like(payload.get("ev")):
            trial.setTrialEvents(Events.fromStructure(payload["ev"]))
        if "history" in payload and not _is_empty_like(payload.get("history")):
            trial.setHistory(History.fromStructure(payload["history"]))
        if "ensCovHist" in payload and not _is_empty_like(payload.get("ensCovHist")):
            trial.setEnsCovHist(History.fromStructure(payload["ensCovHist"]))
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
        Xmat = self._coerce_stim_input(X)
        vals = self.evaluate(Xmat)
        coeffs = self.coefficients.reshape(1, -1)
        if self.link == "poisson":
            out = vals[:, None] * coeffs
        else:
            out = (vals * (1.0 - vals))[:, None] * coeffs
        return out[0] if out.shape[0] == 1 else out

    def evalGradientLog(self, X: np.ndarray) -> np.ndarray:
        Xmat = self._coerce_stim_input(X)
        vals = self.evaluate(Xmat)
        coeffs = self.coefficients.reshape(1, -1)
        if self.link == "poisson":
            out = np.repeat(coeffs, Xmat.shape[0], axis=0)
        else:
            out = (1.0 - vals)[:, None] * coeffs
        return out[0] if out.shape[0] == 1 else out

    def evalJacobian(self, X: np.ndarray) -> np.ndarray:
        Xmat = self._coerce_stim_input(X)
        vals = self.evaluate(Xmat)
        outer = np.outer(self.coefficients, self.coefficients)
        if self.link == "poisson":
            out = vals[:, None, None] * outer[None, :, :]
        else:
            factor = vals * (1.0 - vals) * (1.0 - 2.0 * vals)
            out = factor[:, None, None] * outer[None, :, :]
        return out[0] if out.shape[0] == 1 else out

    def evalJacobianLog(self, X: np.ndarray) -> np.ndarray:
        Xmat = self._coerce_stim_input(X)
        vals = self.evaluate(Xmat)
        outer = np.outer(self.coefficients, self.coefficients)
        if self.link == "poisson":
            out = np.zeros((Xmat.shape[0], outer.shape[0], outer.shape[1]), dtype=float)
        else:
            factor = -(vals * (1.0 - vals))
            out = factor[:, None, None] * outer[None, :, :]
        return out[0] if out.shape[0] == 1 else out

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

    @staticmethod
    def _coerce_stim_input(X: np.ndarray | float | list[float]) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError("stimulus input must be scalar, 1D, or 2D")
        return arr

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
            finite = arr[np.isfinite(arr)]
            vals.append(float(np.mean(finite)) if finite.size else np.nan)
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
            finite = np.abs(arr[np.isfinite(arr)])
            vals.append(float(np.mean(finite)) if finite.size else np.nan)
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
    def computeSpikeRateCIs(spike_matrix: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
