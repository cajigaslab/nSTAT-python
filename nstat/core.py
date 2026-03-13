from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must be array-like.")
    if array.ndim > 2 or (array.ndim == 2 and min(array.shape) != 1):
        raise ValueError(f"{name} can only have one dimension.")
    return array.reshape(-1)


def _normalize_signal_matrix(data: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, n_time: int) -> np.ndarray:
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim == 0:
        raise ValueError("Data must be array-like.")
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    elif matrix.ndim != 2:
        raise ValueError("Data must be one- or two-dimensional.")

    if matrix.shape[0] == n_time:
        return matrix.astype(float, copy=True)
    if matrix.shape[1] == n_time:
        return matrix.T.astype(float, copy=True)
    raise ValueError("Data dimensions do not match the time vector specified.")


def _coerce_1based_indices(values: Sequence[int] | np.ndarray, upper: int) -> list[int]:
    out: list[int] = []
    for raw in np.asarray(values).reshape(-1):
        index = int(raw)
        if index < 1 or index > upper:
            raise IndexError("Signal index out of range. Indexing is 1-based.")
        out.append(index)
    return out


def _roundn(values: Sequence[float] | np.ndarray, decimals: int) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), decimals=max(int(decimals), 0))


def _matlab_mode_1d(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return np.nan
    unique, counts = np.unique(array, return_counts=True)
    best = np.flatnonzero(counts == np.max(counts))
    return float(unique[int(best[0])])


def _nearest_sample_matrix(target_time: np.ndarray, source_time: np.ndarray, source_data: np.ndarray) -> np.ndarray:
    target = np.asarray(target_time, dtype=float).reshape(-1)
    source_t = np.asarray(source_time, dtype=float).reshape(-1)
    source = np.asarray(source_data, dtype=float)
    if source.ndim == 1:
        source = source[:, None]
    if source_t.size == 0:
        return np.zeros((target.size, source.shape[1]), dtype=float)
    right = np.searchsorted(source_t, target, side="left")
    right = np.clip(right, 0, source_t.size - 1)
    left = np.clip(right - 1, 0, source_t.size - 1)
    choose_right = np.abs(source_t[right] - target) <= np.abs(source_t[left] - target)
    indices = np.where(choose_right, right, left)
    return source[indices]


class SignalObj:
    """Multi-dimensional time-series signal object (Matlab ``SignalObj``).

    ``SignalObj`` is the foundational data container in nSTAT.  It stores
    one or more signal channels sampled on a common time axis, along with
    metadata (name, units, labels) and supports element-wise arithmetic,
    resampling, filtering, correlation analysis, and spectral estimation.

    Parameters
    ----------
    time : array_like
        Monotonically increasing time vector of length *n*.
    data : array_like
        Signal values.  Shape ``(n,)`` for a scalar signal or ``(n, d)``
        for a *d*-dimensional signal.
    name : str, optional
        Human-readable signal name (used as y-axis label in plots).
    xlabelval : str, optional
        X-axis label string (default ``'time'``).
    xunits : str, optional
        X-axis unit string (default ``'s'``).
    yunits : str, optional
        Y-axis unit string.
    dataLabels : sequence of str or str, optional
        Per-dimension labels.  A single string is broadcast to all
        dimensions.
    plotProps : sequence or str, optional
        Per-dimension Matplotlib format strings.

    See Also
    --------
    Covariate : SignalObj subclass with confidence-interval support.
    nspikeTrain : Point-process (spike train) companion class.
    """

    def __init__(
        self,
        time: Sequence[float],
        data: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        name: str = "",
        xlabelval: str = "time",
        xunits: str = "s",
        yunits: str = "",
        dataLabels: Sequence[str] | str | None = None,
        plotProps: Sequence[Any] | str | None = None,
        **kwargs,
    ) -> None:
        if "xlabel" in kwargs and "xlabelval" not in kwargs:
            xlabelval = kwargs.pop("xlabel")
        if "data_labels" in kwargs and dataLabels is None:
            dataLabels = kwargs.pop("data_labels")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self.time = _as_1d_float(time, "Time vector")
        self.data = _normalize_signal_matrix(data, self.time.size)
        self.name = str(name)
        self.xlabelval = str(xlabelval)
        self.xunits = str(xunits)
        self.yunits = str(yunits)
        self.minTime = float(np.min(self.time)) if self.time.size else 0.0
        self.maxTime = float(np.max(self.time)) if self.time.size else 0.0

        if self.time.size > 1:
            delta_t = float(np.mean(np.diff(self.time)))
        else:
            delta_t = np.nan
        if not np.isfinite(delta_t) or delta_t <= 0:
            delta_t = 0.001
        self.sampleRate = float(1.0 / delta_t)
        self.origSampleRate = float(self.sampleRate)
        self.originalTime = self.time.copy()
        self.originalData = self.data.copy()
        self.dataMask = np.ones(self.dimension, dtype=int)
        self.plotProps: list[Any] = []
        self.setPlotProps(plotProps)
        self.setDataLabels(dataLabels if dataLabels is not None else "")
        self.conf_interval: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def dimension(self) -> int:
        """Number of signal channels (columns in the data matrix)."""
        return int(self.data.shape[1])

    @property
    def values(self) -> np.ndarray:
        """Signal data as a 1-D array (scalar) or 2-D matrix."""
        if self.dimension == 1:
            return self.data[:, 0]
        return self.data

    @property
    def units(self) -> str:
        """Y-axis unit string (alias for ``yunits``)."""
        return self.yunits

    @property
    def sample_rate(self) -> float:
        """Sampling rate in Hz (alias for ``sampleRate``)."""
        return float(self.sampleRate)

    def _spawn(
        self,
        time: np.ndarray,
        data: np.ndarray,
        *,
        data_labels: Sequence[str] | None = None,
        plot_props: Sequence[Any] | None = None,
    ) -> "SignalObj":
        labels = list(self.dataLabels) if data_labels is None else list(data_labels)
        props = list(self.plotProps) if plot_props is None else list(plot_props)
        return self.__class__(
            np.asarray(time, dtype=float).copy(),
            np.asarray(data, dtype=float).copy(),
            self.name,
            self.xlabelval,
            self.xunits,
            self.yunits,
            labels,
            props,
        )

    def copySignal(self) -> "SignalObj":
        """Return a deep copy of this signal (Matlab ``copySignal``)."""
        copied = self._spawn(self.time, self.data)
        if self.conf_interval is not None:
            copied.conf_interval = (
                np.asarray(self.conf_interval[0], dtype=float).copy(),
                np.asarray(self.conf_interval[1], dtype=float).copy(),
            )
        copied.dataMask = np.asarray(self.dataMask, dtype=int).copy()
        copied.originalTime = self.originalTime.copy()
        copied.originalData = self.originalData.copy()
        copied.sampleRate = float(self.sampleRate)
        copied.origSampleRate = float(self.origSampleRate)
        copied.minTime = float(self.minTime)
        copied.maxTime = float(self.maxTime)
        return copied

    def _binary_operand_matrix(self, other) -> tuple[np.ndarray, list[str]]:
        if isinstance(other, SignalObj):
            if self.time.shape != other.time.shape or np.max(np.abs(self.time - other.time)) > 1e-9:
                raise ValueError("Signals must share an identical time grid for arithmetic operations.")
            data = np.asarray(other.data, dtype=float)
            if data.shape[1] == 1 and self.dimension > 1:
                data = np.repeat(data, self.dimension, axis=1)
                labels = list(self.dataLabels)
            elif self.dimension == 1 and data.shape[1] > 1:
                labels = list(other.dataLabels)
            elif data.shape[1] == self.dimension:
                labels = list(self.dataLabels)
            else:
                raise ValueError("Signal dimensions must match for arithmetic operations.")
            return data, labels

        values = np.asarray(other, dtype=float)
        if values.ndim == 0:
            data = np.full(self.data.shape, float(values), dtype=float)
            return data, list(self.dataLabels)
        if values.ndim == 1:
            if values.size == self.time.size:
                return values.reshape(-1, 1), list(self.dataLabels if self.dimension == 1 else [self.dataLabels[0]])
            if values.size == self.dimension:
                return np.tile(values.reshape(1, -1), (self.time.size, 1)), list(self.dataLabels)
        if values.ndim == 2 and values.shape[0] == self.time.size:
            return values, list(self.dataLabels[: values.shape[1]])
        raise ValueError("Unsupported arithmetic operand for SignalObj")

    def _binary_op(self, other, op) -> "SignalObj":
        other_matrix, labels = self._binary_operand_matrix(other)
        left = self.data
        if left.shape[1] == 1 and other_matrix.shape[1] > 1:
            left = np.repeat(left, other_matrix.shape[1], axis=1)
            labels = labels if labels else list(self.dataLabels[: other_matrix.shape[1]])
        if other_matrix.shape[1] == 1 and left.shape[1] > 1:
            other_matrix = np.repeat(other_matrix, left.shape[1], axis=1)
            labels = list(self.dataLabels)
        result = op(left, other_matrix)
        return self._spawn(self.time, result, data_labels=labels)

    def setName(self, name: str) -> None:
        """Set the signal name (y-axis label)."""
        if not isinstance(name, str):
            raise TypeError("Name must be a string!")
        self.name = name

    def setXlabel(self, name: str) -> None:
        """Set the x-axis label string."""
        self.xlabelval = str(name)

    def setYLabel(self, name: str) -> None:
        """Set the y-axis label (alias for ``setName``)."""
        self.setName(name)

    def setUnits(self, xUnits: str, yUnits: str | None = None) -> None:
        """Set x-axis and optionally y-axis units."""
        if yUnits is not None:
            self.setYUnits(yUnits)
        self.setXUnits(xUnits)

    def setXUnits(self, units: str) -> None:
        """Set the x-axis unit string."""
        if isinstance(units, str):
            self.xunits = units

    def setYUnits(self, units: str) -> None:
        """Set the y-axis unit string."""
        if isinstance(units, str):
            self.yunits = units

    def setSampleRate(self, sampleRate: float) -> None:
        """Set the sample rate, resampling the data if it differs from current."""
        requested = float(sampleRate)
        current = float(self.sampleRate)
        if abs(round(requested, 3) - round(current, 3)) > 0:
            self.resampleMe(requested)

    def setDataLabels(self, dataLabels: Sequence[str] | str | None) -> None:
        """Set per-dimension data labels.

        A single string is broadcast to all dimensions.  A sequence must
        have length equal to ``dimension``.
        """
        if dataLabels is None or (isinstance(dataLabels, str) and dataLabels == ""):
            self.dataLabels = ["" for _ in range(self.dimension)]
            return

        if isinstance(dataLabels, str):
            self.dataLabels = [dataLabels for _ in range(self.dimension)]
            return

        labels = [str(label) for label in dataLabels]
        if len(labels) != self.dimension:
            raise ValueError("Need the number of labels to match the number of dimensions of the SignalObj")
        self.dataLabels = labels

    def setPlotProps(self, plotProps: Sequence[Any] | str | None, index: int | None = None) -> None:
        """Set per-dimension Matplotlib format strings.

        When *index* (1-based) is given, only that dimension is updated.
        """
        if index is None:
            if plotProps is None:
                self.plotProps = [None for _ in range(self.dimension)]
            elif isinstance(plotProps, str):
                self.plotProps = [plotProps for _ in range(self.dimension)]
            else:
                props = list(plotProps)
                if len(props) == 0:
                    self.plotProps = [None for _ in range(self.dimension)]
                    return
                if len(props) == 1 and self.dimension > 1:
                    props = props * self.dimension
                if len(props) != self.dimension:
                    raise ValueError("plotProps length must match signal dimension.")
                self.plotProps = props
            return

        indices = _coerce_1based_indices([index], self.dimension)
        target = indices[0] - 1
        if not self.plotProps:
            self.plotProps = [None for _ in range(self.dimension)]
        if isinstance(plotProps, Sequence) and not isinstance(plotProps, str):
            props = list(plotProps)
            self.plotProps[target] = props[0] if props else None
        else:
            self.plotProps[target] = plotProps

    def setDataMask(self, dataMask: Sequence[int] | np.ndarray) -> None:
        """Set binary data mask (1 = visible, 0 = hidden) for each dimension."""
        mask = np.asarray(dataMask, dtype=int).reshape(-1)
        if mask.size != self.dimension:
            raise ValueError("dataMask must match the number of signal dimensions.")
        if np.any((mask != 0) & (mask != 1)):
            raise ValueError("dataMask must be binary.")
        self.dataMask = mask

    def setMaskByInd(self, index: Sequence[int] | np.ndarray) -> None:
        """Enable only the dimensions at the given 1-based indices."""
        selected = _coerce_1based_indices(index, self.dimension)
        mask = np.zeros(self.dimension, dtype=int)
        mask[np.asarray(selected, dtype=int) - 1] = 1
        self.setDataMask(mask)

    def setMaskByLabels(self, labels: Sequence[str] | str) -> None:
        """Enable only the dimensions whose data labels match *labels*."""
        indices = self.getIndicesFromLabels(labels)
        if isinstance(indices, list) and indices and isinstance(indices[0], list):
            flat = [item for sub in indices for item in sub]
        elif isinstance(indices, list):
            flat = indices
        else:
            flat = [indices]
        self.setMaskByInd(flat)

    def setMask(self, mask: Sequence[int] | Sequence[str] | np.ndarray | None = None) -> None:
        """Flexible mask setter accepting indices, labels, or a binary vector.

        ``None`` clears the mask (all hidden).  A binary vector of length
        ``dimension`` is used directly.  A list of labels or 1-based indices
        enables only those dimensions.
        """
        if mask is None:
            self.setDataMask(np.zeros(self.dimension, dtype=int))
            return

        if isinstance(mask, str):
            self.setMaskByLabels(mask)
            return

        values = list(mask)
        if not values:
            self.setDataMask(np.zeros(self.dimension, dtype=int))
            return

        first = values[0]
        if isinstance(first, str):
            self.setMaskByLabels(values)
            return

        arr = np.asarray(values)
        if arr.size == self.dimension and np.all(np.isin(arr, [0, 1])):
            self.setDataMask(arr.astype(int))
            return
        self.setMaskByInd(arr.astype(int))

    def getTime(self) -> np.ndarray:
        """Return a copy of the time vector."""
        return self.time.copy()

    def getData(self) -> np.ndarray:
        """Return signal data as a matrix (alias for ``dataToMatrix()``)."""
        return self.dataToMatrix()

    def getOriginalData(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(originalTime, originalData)`` copies."""
        return self.originalTime.copy(), self.originalData.copy()

    def getOrigDataSig(self) -> "SignalObj":
        """Return the original (pre-resample) data as a new ``SignalObj``."""
        return self._spawn(self.originalTime, self.originalData)

    def getPlotProps(self, index: int) -> Any:
        """Return the plot property for dimension *index* (1-based)."""
        idx = _coerce_1based_indices([index], self.dimension)[0] - 1
        return self.plotProps[idx]

    def getIndexFromLabel(self, label: str) -> list[int]:
        """Return 1-based indices of dimensions whose label equals *label*."""
        matches = [i + 1 for i, value in enumerate(self.dataLabels) if value == label]
        if not matches:
            raise ValueError("Label does not exist!")
        return matches

    def getIndicesFromLabels(self, label: Sequence[str] | str):
        """Return 1-based index(es) for one or more data-label strings."""
        if isinstance(label, str):
            matches = self.getIndexFromLabel(label)
            return matches[0] if len(matches) == 1 else matches

        out = [self.getIndexFromLabel(str(item)) for item in label]
        counts = [len(item) for item in out]
        if counts and max(counts) == 1:
            return [item[0] for item in out]
        return out

    def areDataLabelsEmpty(self) -> bool:
        """Return ``True`` if all data labels are empty strings.

        Matches Matlab ``SignalObj.areDataLabelsEmpty()``.
        """
        return all(not str(label) for label in self.dataLabels)

    def isLabelPresent(self, label: str) -> bool:
        """Return ``True`` if *label* matches any data label or equals ``'all'``.

        Matches Matlab ``SignalObj.isLabelPresent()``.
        """
        if str(label).lower() == "all":
            return True
        try:
            self.getIndexFromLabel(label)
            return True
        except ValueError:
            return False

    def convertNamesToIndices(self, selectorArray) -> list[int] | np.ndarray:
        """Convert label names (or mixed) to 1-based indices.

        Matches Matlab ``SignalObj.convertNamesToIndices()``.
        """
        if isinstance(selectorArray, str):
            if selectorArray == "all":
                return list(range(1, self.dimension + 1))
            if self.isLabelPresent(selectorArray):
                return self.getIndexFromLabel(selectorArray)
            raise ValueError(f"Specified label '{selectorArray}' does not match data label")
        if isinstance(selectorArray, (int, float, np.integer)):
            return [int(selectorArray)]
        if isinstance(selectorArray, np.ndarray):
            return selectorArray.astype(int).ravel().tolist()
        if isinstance(selectorArray, (list, tuple)):
            result: list[int] = []
            for item in selectorArray:
                if isinstance(item, str):
                    if self.isLabelPresent(item):
                        result.extend(self.getIndexFromLabel(item))
                else:
                    result.append(int(item))
            return result
        return list(range(1, self.dimension + 1))

    def getValueAt(self, x: Sequence[float] | float) -> np.ndarray:
        """Return signal value(s) at time(s) *x* via nearest-neighbour lookup."""
        query = np.asarray(x, dtype=float).reshape(-1)
        out = np.zeros((query.size, self.dimension), dtype=float)
        valid = (query >= self.minTime) & (query <= self.maxTime)
        if np.any(valid):
            q_valid = query[valid]
            right = np.searchsorted(self.time, q_valid, side="left")
            right = np.clip(right, 0, self.time.size - 1)
            left = np.clip(right - 1, 0, self.time.size - 1)
            choose_right = np.abs(self.time[right] - q_valid) <= np.abs(self.time[left] - q_valid)
            indices = np.where(choose_right, right, left)
            out[valid] = self.data[indices]
        return out[0] if np.isscalar(x) else out

    def _selector_to_zero_based(self, selectorArray: Sequence[int] | np.ndarray | None) -> np.ndarray:
        if selectorArray is None:
            if self.isMaskSet():
                selected = self.findIndFromDataMask()
            else:
                selected = list(range(1, self.dimension + 1))
        else:
            if isinstance(selectorArray, str):
                selected = self.getIndicesFromLabels(selectorArray)
            else:
                selected = selectorArray
        indices = np.asarray(selected, dtype=int).reshape(-1)
        if indices.size == 0:
            return np.array([], dtype=int)
        if np.min(indices) < 1 or np.max(indices) > self.dimension:
            raise IndexError("Signal index out of range. Indexing is 1-based.")
        return indices - 1

    def dataToMatrix(self, selectorArray: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """Return signal data as an ``(n, d)`` matrix.

        *selectorArray* is an optional sequence of 1-based dimension
        indices.  When ``None``, the data mask selects visible dimensions.
        """
        indices = self._selector_to_zero_based(selectorArray)
        if indices.size == 0:
            return np.zeros((self.time.size, 0), dtype=float)
        return self.data[:, indices]

    def _labels_for_indices(self, zero_based: np.ndarray) -> list[str]:
        return [self.dataLabels[int(i)] for i in zero_based]

    def _plot_props_for_indices(self, zero_based: np.ndarray) -> list[Any]:
        if not self.plotProps:
            return [None for _ in zero_based]
        return [self.plotProps[int(i)] for i in zero_based]

    def getSubSignalFromInd(self, selectorArray: Sequence[int] | np.ndarray) -> "SignalObj":
        """Return a new ``SignalObj`` with only the selected dimensions (1-based)."""
        indices = self._selector_to_zero_based(selectorArray)
        return self._spawn(
            self.time,
            self.data[:, indices],
            data_labels=self._labels_for_indices(indices),
            plot_props=self._plot_props_for_indices(indices),
        )

    def getSubSignalFromNames(self, labels: Sequence[str] | str) -> "SignalObj":
        """Return a sub-signal selected by data-label name(s)."""
        indices = self.getIndicesFromLabels(labels)
        return self.getSubSignalFromInd(indices if isinstance(indices, list) else [indices])

    def getSubSignal(self, identifier) -> "SignalObj":
        """Return a sub-signal selected by labels, indices, or mixed."""
        if isinstance(identifier, str):
            return self.getSubSignalFromNames(identifier)
        if isinstance(identifier, np.ndarray):
            values = identifier.reshape(-1).tolist()
        elif isinstance(identifier, Sequence):
            values = list(identifier)
        else:
            values = [identifier]
        if values and isinstance(values[0], str):
            return self.getSubSignalFromNames(values)
        return self.getSubSignalFromInd(values)

    def findNearestTimeIndex(self, time: float) -> int:
        """Return the 1-based index of the sample nearest to *time*."""
        value = float(time)
        if value < self.minTime:
            return 1
        if value > self.maxTime:
            return self.time.size
        right = int(np.searchsorted(self.time, value, side="left"))
        if right <= 0:
            return 1
        if right >= self.time.size:
            return self.time.size
        left = right - 1
        if abs(self.time[right] - value) <= abs(self.time[left] - value):
            return right + 1
        return left + 1

    def findNearestTimeIndices(self, times: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return 1-based indices of the samples nearest to each time in *times*."""
        return np.asarray([self.findNearestTimeIndex(value) for value in np.asarray(times, dtype=float).reshape(-1)], dtype=int)

    def setMinTime(self, minTime: float | None = None, holdVals: int = 0) -> None:
        """Extend or trim the signal to start at *minTime*.

        If *holdVals* is 1, endpoint values are held when extending;
        otherwise the signal is zero-padded.
        """
        target = self.time[0] if minTime is None else float(minTime)
        timeVec = self.getTime()
        if target < float(np.min(timeVec)):
            maxTime = float(np.max(timeVec))
            dt = 1.0 / self.sampleRate
            newTime = np.arange(target, maxTime + 0.5 * dt, dt, dtype=float)
            numSamples = int(newTime.size - timeVec.size)
            if holdVals == 1:
                pad = np.tile(self.data[0:1, :], (numSamples, 1))
            else:
                pad = np.zeros((numSamples, self.dimension), dtype=float)
            self.data = np.vstack([pad, self.data])
            self.time = newTime
        elif target > float(np.min(timeVec)):
            startIndex = self.findNearestTimeIndex(target) - 1
            self.time = self.time[startIndex:]
            self.data = self.data[startIndex:, :]
        self.minTime = float(np.min(self.time))

    def setMaxTime(self, maxTime: float | None = None, holdVals: int = 0) -> None:
        """Extend or trim the signal to end at *maxTime*.

        If *holdVals* is 1, endpoint values are held when extending;
        otherwise the signal is zero-padded.
        """
        target = self.time[-1] if maxTime is None else float(maxTime)
        timeVec = self.getTime()
        if float(np.max(timeVec)) < target:
            minTime = float(np.min(timeVec))
            n_samples = int(float(self.sampleRate) * (target - minTime) + 1.0)
            n_samples = max(n_samples, timeVec.size)
            newTime = np.linspace(minTime, target, n_samples, dtype=float)
            numSamples = int(newTime.size - timeVec.size)
            if holdVals == 1:
                pad = np.tile(self.data[-1:, :], (numSamples, 1))
            else:
                pad = np.zeros((numSamples, self.dimension), dtype=float)
            self.data = np.vstack([self.data, pad])
            self.time = newTime
        elif float(np.max(timeVec)) > target:
            endIndex = self.findNearestTimeIndex(target)
            self.time = self.time[:endIndex]
            self.data = self.data[:endIndex, :]
        self.maxTime = float(np.max(self.time))

    def merge(self, other: "SignalObj", holdVals: int = 0) -> "SignalObj":
        """Merge *other* signal columns into *self*.

        Matlab calls ``makeCompatible`` first so that signals with
        different time grids are reconciled automatically.  The Python
        version now does the same.

        Parameters
        ----------
        other : SignalObj
            Signal whose data columns will be appended.
        holdVals : int, optional
            Passed to ``makeCompatible`` – ``1`` holds endpoint values
            when the time range is extended; ``0`` (default) pads with
            zeros.
        """
        s1c, s2c = self.makeCompatible(other, holdVals)
        merged = s1c._spawn(
            s1c.time,
            np.column_stack([s1c.data, s2c.data]),
            data_labels=[*s1c.dataLabels, *list(s2c.dataLabels)],
            plot_props=[*s1c.plotProps, *getattr(s2c, "plotProps", [None for _ in range(s2c.dimension)])],
        )
        return merged

    def __add__(self, other) -> "SignalObj":
        return self._binary_op(other, np.add)

    def __radd__(self, other) -> "SignalObj":
        return self + other

    def __sub__(self, other) -> "SignalObj":
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other) -> "SignalObj":
        other_matrix, labels = self._binary_operand_matrix(other)
        left = other_matrix
        right = self.data
        if left.shape[1] == 1 and right.shape[1] > 1:
            left = np.repeat(left, right.shape[1], axis=1)
            labels = list(self.dataLabels)
        if right.shape[1] == 1 and left.shape[1] > 1:
            right = np.repeat(right, left.shape[1], axis=1)
        return self._spawn(self.time, np.subtract(left, right), data_labels=labels)

    def __pos__(self) -> "SignalObj":
        return self.copySignal()

    def __neg__(self) -> "SignalObj":
        return self._spawn(self.time, -self.data, data_labels=list(self.dataLabels))

    def __mul__(self, other) -> "SignalObj":
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other) -> "SignalObj":
        return self * other

    def __truediv__(self, other) -> "SignalObj":
        return self._binary_op(other, np.divide)

    def __rtruediv__(self, other) -> "SignalObj":
        other_matrix, labels = self._binary_operand_matrix(other)
        left = other_matrix
        right = self.data
        if left.shape[1] == 1 and right.shape[1] > 1:
            left = np.repeat(left, right.shape[1], axis=1)
            labels = list(self.dataLabels)
        if right.shape[1] == 1 and left.shape[1] > 1:
            right = np.repeat(right, left.shape[1], axis=1)
        return self._spawn(self.time, np.divide(left, right), data_labels=labels)

    def __matmul__(self, other) -> "SignalObj":
        """Matrix multiply (``@`` operator).  Matches Matlab ``mtimes``."""
        if isinstance(other, SignalObj):
            return self._spawn(self.time, self.data * other.data, data_labels=list(self.dataLabels))
        other_arr = np.asarray(other, dtype=float)
        result = (self.data.T @ other_arr).T if other_arr.ndim <= 1 else self.data @ other_arr
        return self._spawn(self.time[:result.shape[0]] if result.ndim == 2 else self.time, result)

    def ldivide(self, other) -> "SignalObj":
        r"""Element-wise left division (Matlab ``.\``): ``other ./ self``.

        Matches Matlab ``SignalObj.ldivide()``.
        """
        return self._binary_op(other, lambda a, b: np.divide(b, a))

    @property
    def T(self) -> "SignalObj":
        """Transpose the data matrix.  Matches Matlab ``ctranspose`` / ``transpose``."""
        new_data = self.data.T
        new_time = self.time[:new_data.shape[0]] if new_data.shape[0] != self.time.size else self.time
        return self._spawn(new_time, new_data)

    def clearPlotProps(self, index=None) -> None:
        """Clear plot properties.  Matches Matlab ``clearPlotProps``."""
        if index is None:
            index = list(range(self.dimension))
        else:
            index = [i - 1 for i in np.atleast_1d(index)]
        for i in index:
            if i < len(self.plotProps):
                self.plotProps[i] = None

    def plotPropsSet(self) -> bool:
        """Return ``True`` if any plot property is non-empty.

        Matches Matlab ``SignalObj.plotPropsSet()``.
        """
        for prop in self.plotProps:
            if prop is not None and str(prop) != "":
                return True
        return False

    def getSigInTimeWindow(
        self,
        wMin: Sequence[float] | float | None = None,
        wMax: Sequence[float] | float | None = None,
        holdVals: int = 0,
    ) -> "SignalObj":
        """Extract signal within ``[wMin, wMax]``.

        Multiple windows can be specified by passing equal-length sequences
        for *wMin* and *wMax*; the extracted segments are concatenated as
        additional dimensions (Matlab ``getSigInTimeWindow``).
        """
        if wMax is None:
            wMax = self.maxTime
        if wMin is None:
            wMin = self.minTime

        min_values = np.asarray([wMin] if np.isscalar(wMin) else wMin, dtype=float).reshape(-1)
        max_values = np.asarray([wMax] if np.isscalar(wMax) else wMax, dtype=float).reshape(-1)
        if min_values.size != max_values.size:
            raise ValueError("Window minTimes must contain the same number of elements as window maxTimes")

        if min_values.size == 1 and self.minTime == float(min_values[0]) and self.maxTime == float(max_values[0]):
            return self.copySignal()

        windowed: SignalObj | None = None
        for idx, (left, right) in enumerate(zip(min_values, max_values), start=1):
            current = self.copySignal()
            if left < current.minTime:
                current.setMinTime(left, holdVals)
            if right > current.maxTime:
                current.setMaxTime(right, holdVals)

            start = current.findNearestTimeIndex(left) - 1
            stop = current.findNearestTimeIndex(right)
            current.time = current.time[start:stop]
            current.data = current.data[start:stop, :]
            labels = list(current.dataLabels)
            if min_values.size > 1:
                labels = [f"{label}_{{{idx}}}" for label in labels]
            current.setDataLabels(labels)
            current.setMinTime()
            current.setMaxTime()
            windowed = current if windowed is None else windowed.merge(current)
        return windowed if windowed is not None else self.copySignal()

    def restoreToOriginal(self, rMask: int = 0) -> None:
        """Restore time, data, and sample rate to their original values.

        If *rMask* is 1, the data mask is also reset (all visible).
        """
        self.time = self.originalTime.copy()
        self.data = self.originalData.copy()
        self.minTime = float(np.min(self.time))
        self.maxTime = float(np.max(self.time))
        self.sampleRate = float(self.origSampleRate)
        if rMask == 1:
            self.resetMask()

    def resetMask(self) -> None:
        """Reset the data mask so all dimensions are visible."""
        self.dataMask = np.ones(self.dimension, dtype=int)

    def findIndFromDataMask(self) -> list[int]:
        """Return 1-based indices of dimensions currently visible (mask == 1)."""
        return [int(index) + 1 for index in np.flatnonzero(self.dataMask == 1)]

    def isMaskSet(self) -> bool:
        """Return ``True`` if any dimension is currently masked out."""
        return bool(np.any(self.dataMask == 0))

    def abs(self) -> "SignalObj":
        """Element-wise absolute value (Matlab ``abs``)."""
        labels = [f"|{label}|" if label else "" for label in self.dataLabels]
        return self._spawn(self.time, np.abs(self.data), data_labels=labels).with_metadata(
            name=f"|{self.name}|",
            yunits=self.yunits,
        )

    def __abs__(self) -> "SignalObj":
        return self.abs()

    def log(self) -> "SignalObj":
        """Element-wise natural logarithm (Matlab ``log``)."""
        labels = [f"ln({label})" if label else "" for label in self.dataLabels]
        yunits = f"ln({self.yunits})" if self.yunits else ""
        return self._spawn(self.time, np.log(self.data), data_labels=labels).with_metadata(
            name=f"ln({self.name})",
            yunits=yunits,
        )

    def with_metadata(self, *, name: str | None = None, xlabelval: str | None = None, xunits: str | None = None, yunits: str | None = None) -> "SignalObj":
        """Return a copy with selectively overridden metadata fields."""
        out = self.copySignal()
        if name is not None:
            out.name = str(name)
        if xlabelval is not None:
            out.xlabelval = str(xlabelval)
        if xunits is not None:
            out.xunits = str(xunits)
        if yunits is not None:
            out.yunits = str(yunits)
        return out

    def median(self, axis: int | None = None) -> "SignalObj":
        """Column-wise median (default) or row-wise median of signal data.

        ``median()`` or ``median(0)`` computes the median of each
        component across time.  ``median(1)`` computes the median value at
        each time point across dimensions.
        """
        axis_arg = 0 if axis is None else axis
        median_data = np.median(self.data, axis=axis_arg)
        array = np.asarray(median_data, dtype=float)
        if array.ndim == 1 and array.size == self.dimension:
            labels = [f"median({label})" if label else "" for label in self.dataLabels]
            return self._spawn(
                np.asarray([self.time[0], self.time[-1]], dtype=float),
                np.vstack([array, array]),
                data_labels=labels,
            ).with_metadata(name=f"median({self.name})")
        reshaped = array.reshape(-1, 1)
        return self._spawn(self.time, reshaped, data_labels=[f"median({self.name})"]).with_metadata(name=f"median({self.name})")

    def mode(self, axis: int | None = None) -> "SignalObj":
        """Column-wise mode of signal data (Matlab ``mode``)."""
        axis_arg = 0 if axis is None else axis
        if axis_arg == 0:
            mode_data = np.asarray([_matlab_mode_1d(self.data[:, i]) for i in range(self.dimension)], dtype=float)
        elif axis_arg == 1:
            mode_data = np.asarray([_matlab_mode_1d(row) for row in self.data], dtype=float)
        else:
            raise ValueError("axis must be 0, 1, or None")
        array = np.asarray(mode_data, dtype=float)
        if array.ndim == 1 and array.size == self.dimension:
            labels = [f"mode({label})" if label else "" for label in self.dataLabels]
            return self._spawn(
                np.asarray([self.time[0], self.time[-1]], dtype=float),
                np.vstack([array, array]),
                data_labels=labels,
            ).with_metadata(name=f"mode({self.name})")
        reshaped = array.reshape(-1, 1)
        return self._spawn(self.time, reshaped, data_labels=[f"mode({self.name})"]).with_metadata(name=f"mode({self.name})")

    def mean(self, axis: int | None = None) -> "SignalObj":
        """Column-wise mean (default) or row-wise mean of signal data.

        ``mean()`` or ``mean(0)`` computes the mean of each component
        across time.  ``mean(1)`` computes the mean value at each time
        point across dimensions.
        """
        axis_arg = 0 if axis is None else axis
        mean_data = np.mean(self.data, axis=axis_arg)
        array = np.asarray(mean_data, dtype=float)
        if array.ndim == 1 and array.size == self.dimension:
            labels = [f"\\mu({label})" if label else "" for label in self.dataLabels]
            return self._spawn(
                np.asarray([self.time[0], self.time[-1]], dtype=float),
                np.vstack([array, array]),
                data_labels=labels,
            )
        reshaped = array.reshape(-1, 1)
        return self._spawn(self.time, reshaped, data_labels=[f"\\mu({self.name})"])

    def std(self, axis: int | None = None) -> "SignalObj":
        """Column-wise standard deviation (sample, ddof=1) of signal data.

        ``std()`` or ``std(0)`` computes std of each component across
        time.  ``std(1)`` computes std at each time point across dimensions.
        """
        axis_arg = 0 if axis is None else axis
        std_data = np.std(self.data, axis=axis_arg, ddof=1)
        array = np.asarray(std_data, dtype=float)
        if array.ndim == 1 and array.size == self.dimension:
            labels = [f"\\sigma({label})" if label else "" for label in self.dataLabels]
            return self._spawn(
                np.asarray([self.time[0], self.time[-1]], dtype=float),
                np.vstack([array, array]),
                data_labels=labels,
            )
        reshaped = array.reshape(-1, 1)
        return self._spawn(self.time, reshaped, data_labels=[f"\\sigma({self.name})"])

    def max(self, axis: int | None = None):
        """Return ``(values, indices, times)`` of column-wise maxima."""
        axis_arg = 0 if axis is None else axis
        values = np.max(self.data, axis=axis_arg)
        indices = np.argmax(self.data, axis=axis_arg)
        time = self.time[np.asarray(indices, dtype=int)]
        return values, indices, time

    def min(self, axis: int | None = None):
        """Return ``(values, indices, times)`` of column-wise minima."""
        axis_arg = 0 if axis is None else axis
        values = np.min(self.data, axis=axis_arg)
        indices = np.argmin(self.data, axis=axis_arg)
        time = self.time[np.asarray(indices, dtype=int)]
        return values, indices, time

    def resample(self, sample_rate: float) -> "SignalObj":
        """Return a resampled copy at *sample_rate* Hz."""
        copied = self.copySignal()
        copied.resampleMe(sample_rate)
        return copied

    def resampleMe(self, newSampleRate: float) -> None:
        """Resample data in-place to *newSampleRate* Hz via cubic interpolation."""
        try:
            from scipy.interpolate import interp1d
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for SignalObj.resampleMe") from exc

        rate = float(newSampleRate)
        if rate <= 0:
            raise ValueError("sampleRate must be > 0.")
        if self.sampleRate == rate:
            return
        self.restoreToOriginal()
        dt = 1.0 / rate
        newTime = np.arange(self.time[0], self.time[-1] + 0.5 * dt, dt, dtype=float)
        if self.data.shape[0] > 1:
            columns = []
            if self.time.size >= 4:
                interp_kind = "cubic"
            elif self.time.size == 3:
                interp_kind = "quadratic"
            else:
                interp_kind = "linear"
            for index in range(self.dimension):
                interpolator = interp1d(
                    self.time,
                    self.data[:, index],
                    kind=interp_kind,
                    bounds_error=False,
                    fill_value=0.0,
                )
                columns.append(np.asarray(interpolator(newTime), dtype=float))
            newData = np.column_stack(columns)
        else:
            newData = np.asarray(self.data, dtype=float).copy()
        self.time = newTime
        self.data = newData
        self.sampleRate = rate
        self.minTime = float(np.min(newTime))
        self.maxTime = float(np.max(newTime))

    @property
    def derivative(self) -> "SignalObj":
        deriv = np.zeros_like(self.data, dtype=float)
        if self.data.shape[0] > 1:
            deriv[1:, :] = np.diff(self.data, axis=0) * float(self.sampleRate)
        deriv[~np.isfinite(deriv)] = 0.0
        labels = [f"d_{label}" if label else "" for label in self.dataLabels]
        return self._spawn(self.time, deriv, data_labels=labels)

    def derivativeAt(self, x0: Sequence[float] | float):
        """Return the derivative value(s) at time(s) *x0*."""
        deriv = self.derivative
        values = deriv.getValueAt(x0)
        return values

    def integral(self, t0: float | None = None, tf: float | None = None) -> "SignalObj":
        """Cumulative integral of the signal from *t0* to *tf*.

        Computed via a causal IIR accumulator:
        ``y[n] = y[n-1] + x[n] * deltaT``.  If *t0* / *tf* are not
        specified, ``minTime`` / ``maxTime`` are used.
        """
        start = self.minTime if t0 is None else float(t0)
        stop = self.maxTime if tf is None else float(tf)
        integrated = self.getSigInTimeWindow(start, stop)
        dt = 1.0 / max(float(integrated.sampleRate), 1e-12)
        integrated = integrated.filter([dt], [1.0, -1.0])
        if integrated.yunits and integrated.xunits:
            integrated.setYUnits(f"{integrated.yunits}*{integrated.xunits}")
        elif integrated.xunits:
            integrated.setYUnits(integrated.xunits)
        dtstr = " d\\tau"
        integrated.setName(f"\\int_{integrated.minTime:g}^{integrated.xlabelval[:1]}\\!\\!{{{integrated.name}{dtstr}}}")
        labels_empty = all(not str(label) for label in integrated.dataLabels)
        if not labels_empty:
            updated_labels: list[str] = []
            for label in self.dataLabels:
                if label:
                    updated_labels.append(f"\\int_{integrated.minTime:g}^{integrated.xlabelval[:1]}\\!\\!{{{label}{dtstr}}}")
                else:
                    updated_labels.append("")
            integrated.setDataLabels(updated_labels)
        return integrated

    def filter(self, B, A=1) -> "SignalObj":
        """Apply a causal IIR/FIR filter ``(B, A)`` to each dimension.

        Equivalent to ``scipy.signal.lfilter(B, A, data)``.
        """
        try:
            from scipy.signal import lfilter
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for SignalObj.filter") from exc

        b = np.asarray(B, dtype=float).reshape(-1)
        a = np.asarray(A, dtype=float).reshape(-1)
        filtered = np.column_stack([lfilter(b, a, self.data[:, index]) for index in range(self.dimension)])
        return self._spawn(self.time, filtered, data_labels=list(self.dataLabels))

    def filtfilt(self, B, A=1) -> "SignalObj":
        """Apply a zero-phase IIR/FIR filter ``(B, A)`` to each dimension.

        Equivalent to ``scipy.signal.filtfilt(B, A, data)``.
        """
        try:
            from scipy.signal import filtfilt
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for SignalObj.filtfilt") from exc

        b = np.asarray(B, dtype=float).reshape(-1)
        a = np.asarray(A, dtype=float).reshape(-1)
        filtered = np.column_stack([filtfilt(b, a, self.data[:, index]) for index in range(self.dimension)])
        return self._spawn(self.time, filtered, data_labels=list(self.dataLabels))

    def makeCompatible(self, other: "SignalObj", holdVals: int = 0) -> tuple["SignalObj", "SignalObj"]:
        if (
            self.minTime == other.minTime
            and self.maxTime == other.maxTime
            and round(float(self.sampleRate), 9) == round(float(other.sampleRate), 9)
            and self.time.shape == other.time.shape
            and np.max(np.abs(self.time - other.time)) <= 1e-9
        ):
            return self, other

        s1c = self.copySignal()
        s2c = other.copySignal()
        min_time = min(s1c.minTime, s2c.minTime)
        max_time = max(s1c.maxTime, s2c.maxTime)
        sample_rate = max(float(s1c.sampleRate), float(s2c.sampleRate))
        s1c.setSampleRate(sample_rate)
        s2c.setSampleRate(sample_rate)
        s1c.setMinTime(min_time, holdVals)
        s2c.setMinTime(min_time, holdVals)
        s1c.setMaxTime(max_time, holdVals)
        s2c.setMaxTime(max_time, holdVals)
        s2c.data = _nearest_sample_matrix(s1c.time, s2c.time, s2c.data)
        s2c.time = s1c.time.copy()
        s2c.minTime = float(np.min(s2c.time))
        s2c.maxTime = float(np.max(s2c.time))
        return s1c, s2c

    def autocorrelation(self) -> "SignalObj":
        """Normalized auto-correlation for each signal dimension.

        Returns a new ``SignalObj`` whose time axis is lag (in the original
        x-units) and whose data are the correlation coefficients normalised
        to unity at lag zero (Matlab ``autocorrelation``).
        """
        centered = self.data - np.mean(self.data, axis=0, keepdims=True)
        columns: list[np.ndarray] = []
        lags: np.ndarray | None = None
        for index in range(self.dimension):
            series = centered[:, index]
            denom = float(np.dot(series, series))
            corr = np.correlate(series, series, mode="full")
            if denom > 0:
                corr = corr / denom
            else:
                corr = np.zeros_like(corr, dtype=float)
            if lags is None:
                lags = np.arange(-series.size + 1, series.size, dtype=float) / max(float(self.sampleRate), 1e-12)
            columns.append(np.asarray(corr, dtype=float))
        data = np.column_stack(columns) if columns else np.zeros((0, 0), dtype=float)
        return self.__class__(
            lags if lags is not None else np.array([], dtype=float),
            data,
            f"ACF({self.name})",
            "Lag",
            self.xunits,
            f"{self.yunits}^2" if self.yunits else "",
            list(self.dataLabels),
            list(self.plotProps),
        )

    def crosscorrelation(self, other: "SignalObj") -> "SignalObj":
        """Normalized cross-correlation between two scalar signals.

        Both signals must be one-dimensional.  The result is normalised
        so that the peak equals the Pearson correlation coefficient
        (Matlab ``crosscorrelation``).
        """
        if self.dimension != 1 or other.dimension != 1:
            raise ValueError("crosscorrelation only supports one-dimensional signals")
        s1c, s2c = self.makeCompatible(other)
        x = s1c.data[:, 0] - float(np.mean(s1c.data[:, 0]))
        y = s2c.data[:, 0] - float(np.mean(s2c.data[:, 0]))
        denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
        corr = np.correlate(x, y, mode="full")
        if denom > 0:
            corr = corr / denom
        else:
            corr = np.zeros_like(corr, dtype=float)
        lags = np.arange(-x.size + 1, x.size, dtype=float) / max(float(s1c.sampleRate), 1e-12)
        return self.__class__(
            lags,
            corr,
            f"XCORF({self.name})",
            "Lag",
            self.xunits,
            f"{self.yunits}^2" if self.yunits else "",
            list(self.dataLabels[:1]),
            list(self.plotProps[:1]),
        )

    def xcorr(self, other: "SignalObj" | None = None, maxlag: int | None = None) -> "SignalObj":
        """Raw (un-normalised) cross-correlation (Matlab ``xcorr``).

        Computes pairwise cross-correlation for all dimension pairs.
        When *other* is ``None`` (auto-correlation), only non-negative
        lags are returned.  *maxlag* truncates to ``|lag| ≤ maxlag``
        samples.
        """
        s2 = self if other is None else other
        s1c, s2c = self.makeCompatible(s2)
        data_columns: list[np.ndarray] = []
        data_labels: list[str] = []
        lag_index: np.ndarray | None = None
        for left_index in range(s1c.dimension):
            for right_index in range(s2c.dimension):
                corr = np.correlate(s1c.data[:, left_index], s2c.data[:, right_index], mode="full")
                lags = np.arange(-s1c.data.shape[0] + 1, s1c.data.shape[0], dtype=int)
                if maxlag is not None:
                    keep = np.abs(lags) <= int(maxlag)
                    corr = corr[keep]
                    lags = lags[keep]
                if other is None:
                    keep = lags >= 0
                    corr = corr[keep]
                    lags = lags[keep]
                if lag_index is None:
                    lag_index = lags.astype(float) / max(float(s1c.sampleRate), 1e-12)
                data_columns.append(np.asarray(corr, dtype=float))
                left_label = s1c.dataLabels[left_index] if left_index < len(s1c.dataLabels) else str(left_index + 1)
                right_label = s2c.dataLabels[right_index] if right_index < len(s2c.dataLabels) else str(right_index + 1)
                data_labels.append(f"corr({left_label},{right_label})")
        data = np.column_stack(data_columns) if data_columns else np.zeros((0, 0), dtype=float)
        name = f"corr({self.name},{s2.name})"
        return self.__class__(
            lag_index if lag_index is not None else np.array([], dtype=float),
            data,
            name,
            "\\Delta \\tau",
            self.xunits,
            f"{self.yunits}^2" if self.yunits else "",
            data_labels,
        )

    # ------------------------------------------------------------------
    # Time-shift / alignment helpers (match Matlab SignalObj)
    # ------------------------------------------------------------------
    def shift(self, deltaT: float, updateLabels: bool = False) -> "SignalObj":
        """Return a copy with time shifted by *deltaT* seconds."""
        new_time = self.time + float(deltaT)
        out = self.__class__(
            new_time,
            self.data.copy(),
            self.name,
            self.xlabelval,
            self.xunits,
            self.yunits,
            list(self.dataLabels),
            list(self.plotProps),
        )
        if updateLabels:
            out.name = f"{self.name} shifted by {deltaT}"
        return out

    def shiftMe(self, deltaT: float, updateLabels: bool = False) -> None:
        """In-place time shift by *deltaT* seconds (Matlab ``shiftMe``)."""
        self.time = self.time + float(deltaT)
        self.minTime = float(np.min(self.time)) if self.time.size else 0.0
        self.maxTime = float(np.max(self.time)) if self.time.size else 0.0
        if updateLabels:
            self.name = f"{self.name} shifted by {deltaT}"

    def alignTime(self, timeMarker: float, newTime: float = 0.0) -> None:
        """Shift so that *timeMarker* becomes *newTime* (Matlab ``alignTime``).

        Only shifts if *timeMarker* falls within ``[minTime, maxTime]``,
        matching the Matlab implementation's bounds check.
        """
        if self.minTime <= float(timeMarker) <= self.maxTime:
            self.shiftMe(float(newTime) - float(timeMarker))

    # ------------------------------------------------------------------
    # Element-wise arithmetic helpers (match Matlab SignalObj)
    # ------------------------------------------------------------------
    def power(self, exponent: float) -> "SignalObj":
        """Element-wise power ``data ** exponent`` (Matlab ``power``)."""
        return self.__class__(
            self.time.copy(),
            np.power(self.data, float(exponent)),
            f"{self.name}^{exponent}",
            self.xlabelval,
            self.xunits,
            self.yunits,
            list(self.dataLabels),
            list(self.plotProps),
        )

    def sqrt(self) -> "SignalObj":
        """Element-wise square root (Matlab ``sqrt``)."""
        return self.power(0.5)

    # ------------------------------------------------------------------
    # Peak-finding helpers (match Matlab SignalObj)
    # ------------------------------------------------------------------
    def findPeaks(
        self,
        peak_type: str = "maxima",
        minDistance: int | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Find local peaks in each signal dimension.

        Parameters
        ----------
        peak_type : ``'maxima'`` or ``'minima'``
        minDistance : minimum sample distance between peaks (default:
            ``sampleRate * duration / 10``).

        Returns
        -------
        indices, values : lists of arrays (one per dimension).

        Note: The Matlab original has a bug where the ``'minima'`` branch
        does not negate the data before calling ``findpeaks``.  This Python
        port fixes that.
        """
        from scipy.signal import find_peaks as _find_peaks

        data = np.atleast_2d(self.data)
        if data.shape[0] == 1:
            data = data.T
        N = data.shape[0]
        if minDistance is None:
            duration = float(self.maxTime - self.minTime)
            minDistance = max(1, int(round(self.sampleRate * duration / 10)))

        all_indices: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        for col in range(data.shape[1]):
            sig = data[:, col]
            if peak_type == "minima":
                sig = -sig
            idx, _ = _find_peaks(sig, distance=minDistance)
            all_indices.append(idx)
            all_values.append(data[idx, col])  # always return actual values
        return all_indices, all_values

    def findMaxima(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Convenience wrapper: ``findPeaks('maxima')``."""
        return self.findPeaks("maxima")

    def findMinima(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Convenience wrapper: ``findPeaks('minima')``."""
        return self.findPeaks("minima")

    def findGlobalPeak(
        self, peak_type: str = "maxima"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the global max or min across each dimension.

        Returns
        -------
        times : 1-D array of times at which the peak occurs (one per dim).
        values : 1-D array of peak values (one per dim).

        Note: The Matlab original has a typo (``sOBj`` instead of ``sObj``)
        in the minima branch.  This Python port fixes that.
        """
        data = np.atleast_2d(self.data)
        if data.shape[0] == 1:
            data = data.T
        if peak_type == "maxima":
            idx = np.argmax(data, axis=0)
        else:
            idx = np.argmin(data, axis=0)
        times = self.time[idx]
        values = data[idx, np.arange(data.shape[1])]
        return np.atleast_1d(times), np.atleast_1d(values)

    # ------------------------------------------------------------------
    # Alignment / windowing (match Matlab SignalObj)
    # ------------------------------------------------------------------
    def alignToMax(self) -> tuple["SignalObj", float]:
        """Align all dimensions so their peaks coincide at the mean peak time.

        Returns ``(aligned_signal, mean_peak_time)``.
        Matches Matlab ``SignalObj.alignToMax()``.
        """
        peak_times, _ = self.findGlobalPeak("maxima")
        mean_time = float(np.mean(peak_times))
        delta_t = -(peak_times - mean_time)
        aligned = self.getSubSignal(1).shift(float(delta_t[0]))
        for i in range(1, self.dimension):
            aligned = aligned.merge(self.getSubSignal(i + 1).shift(float(delta_t[i])))
        return aligned, mean_time

    def windowedSignal(self, windowTimes) -> "SignalObj":
        """Extract and concatenate windowed segments.

        Matches Matlab ``SignalObj.windowedSignal()``.
        """
        windowTimes = np.asarray(windowTimes, dtype=float).ravel()
        result = None
        for i in range(len(windowTimes) - 1):
            seg = self.getSigInTimeWindow(float(windowTimes[i]), float(windowTimes[i + 1]))
            if i == 0:
                result = seg
            else:
                seg = seg.shift(-float(windowTimes[i]))
                result = result.merge(seg)
        return result if result is not None else self.copySignal()

    def normWindowedSignal(
        self,
        windowTimes,
        numPoints: int = 100,
        lbound: float | None = None,
        ubound: float | None = None,
    ) -> "SignalObj":
        """Normalize windowed signal segments to a common time axis.

        Matches Matlab ``SignalObj.normWindowedSignal()``.
        """
        windowTimes = np.asarray(windowTimes, dtype=float).ravel()
        columns: list[np.ndarray] = []
        for i in range(len(windowTimes) - 1):
            minT = float(windowTimes[i])
            maxT = float(windowTimes[i + 1])
            dur = abs(maxT - minT)
            if lbound is not None and ubound is not None:
                if dur > ubound or dur < lbound:
                    continue
            seg = self.getSigInTimeWindow(minT, maxT)
            norm_time = np.linspace(minT, maxT, numPoints)
            # Matlab uses interp1(..., 'nearest', 0) — nearest-neighbor with 0-fill
            from scipy.interpolate import interp1d as _interp1d
            _ifn = _interp1d(seg.time, seg.data[:, 0], kind="nearest",
                             bounds_error=False, fill_value=0.0)
            interp_data = _ifn(norm_time)
            columns.append(interp_data)

        if not columns:
            return self.copySignal()
        data = np.column_stack(columns)
        act_time = np.arange(numPoints, dtype=float) / float(numPoints)
        labels = list(self.dataLabels[:1]) * data.shape[1]
        return self.__class__(act_time, data, self.name, self.xlabelval, "%", self.yunits, labels)

    def getSubSignalsWithinNStd(self, nStd: float = 2.0) -> tuple["SignalObj", np.ndarray]:
        """Return sub-signals within *nStd* standard deviations of the mean.

        Returns ``(filtered_signal, selected_indices)``.
        Matches Matlab ``SignalObj.getSubSignalsWithinNStd()``.
        """
        mean_sig = np.mean(self.data, axis=1)
        std_sig = np.std(self.data, axis=1, ddof=1)
        min_val = mean_sig - nStd * std_sig
        max_val = mean_sig + nStd * std_sig
        # A column passes if ALL rows are within [minVal, maxVal]
        above_min = np.all(self.data >= min_val[:, None], axis=0)
        below_max = np.all(self.data <= max_val[:, None], axis=0)
        sig_index = np.flatnonzero(above_min & below_max)
        if sig_index.size == 0:
            return self.copySignal(), sig_index
        # 1-based indices for getSubSignal
        return self.getSubSignal((sig_index + 1).tolist()), sig_index

    # ------------------------------------------------------------------
    # Variability plots (match Matlab SignalObj)
    # ------------------------------------------------------------------
    def plotAllVariability(
        self,
        faceColor=None,
        linewidth: float = 3.0,
        ciUpper: float | np.ndarray = 1.96,
        ciLower: float | np.ndarray | None = None,
        ax=None,
    ):
        """Plot mean ± CI shaded area.  Matches Matlab ``plotAllVariability``.

        Parameters
        ----------
        faceColor : color, optional
            Fill colour (default: tab:blue).
        linewidth : float
            Width of mean line.
        ciUpper, ciLower : float or array
            Number of std-devs (scalar) or explicit bounds (array).
        ax : matplotlib Axes, optional
        """
        import matplotlib.pyplot as plt

        if faceColor is None:
            faceColor = "tab:blue"
        if ciLower is None:
            ciLower = ciUpper
        if ax is None:
            ax = plt.gca()

        mean_sig = np.mean(self.data, axis=1)
        std_sig = np.std(self.data, axis=1, ddof=1)

        ciUpper_arr = np.atleast_1d(ciUpper).ravel()
        ciLower_arr = np.atleast_1d(ciLower).ravel()
        if ciUpper_arr.size == 1:
            ci_top = mean_sig + float(ciUpper_arr[0]) * std_sig
        else:
            ci_top = mean_sig + ciUpper_arr[:len(mean_sig)]
        if ciLower_arr.size == 1:
            ci_bottom = mean_sig - float(ciLower_arr[0]) * std_sig
        else:
            ci_bottom = mean_sig - ciLower_arr[:len(mean_sig)]

        ax.fill_between(self.time, ci_bottom, ci_top, color=faceColor, alpha=0.5, edgecolor="none")
        (h,) = ax.plot(self.time, mean_sig, "k-", linewidth=linewidth)
        return h

    def plotVariability(self, selectorArray=None, ax=None):
        """Plot mean ± CI for each label group.  Matches Matlab ``plotVariability``.

        Parameters
        ----------
        selectorArray : list of list[int] or list[int], optional
        ax : matplotlib Axes, optional
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        if selectorArray is None:
            if not self.areDataLabelsEmpty():
                unique_labels = list(dict.fromkeys(self.dataLabels))
                selectorArray = [self.getIndexFromLabel(lbl) for lbl in unique_labels]
            else:
                selectorArray = list(range(1, self.dimension + 1))

        _TAB_COLORS = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        ]
        handles = []
        if isinstance(selectorArray, list) and selectorArray and isinstance(selectorArray[0], (list, tuple, np.ndarray)):
            for i, sel in enumerate(selectorArray):
                h = self.getSubSignal(sel).plotAllVariability(
                    faceColor=_TAB_COLORS[i % len(_TAB_COLORS)], ax=ax
                )
                handles.append(h)
        else:
            h = self.getSubSignal(selectorArray).plotAllVariability(ax=ax)
            handles.append(h)
        return handles

    # ------------------------------------------------------------------
    # Cross-covariance (match Matlab SignalObj.xcov)
    # ------------------------------------------------------------------
    def xcov(self, other: "SignalObj | None" = None, maxlag: int | None = None,
             scaleOpt: str = "none") -> "SignalObj":
        """Cross-covariance (mean-removed xcorr).  Matches Matlab ``xcov``.

        When called with no *other* argument (auto-covariance), only
        non-negative lags are returned — matching Matlab behaviour where
        ``data=tempC(M-1:end,index)`` and ``lags=tempLags(M-1:end)``.
        """
        auto = other is None
        s1 = self
        s2 = self if auto else other
        s1c, s2c = s1.makeCompatible(s2)

        data_columns: list[np.ndarray] = []
        data_labels: list[str] = []
        lag_index: np.ndarray | None = None

        for li in range(s1c.dimension):
            for ri in range(s2c.dimension):
                x = s1c.data[:, li] - np.mean(s1c.data[:, li])
                y = s2c.data[:, ri] - np.mean(s2c.data[:, ri])
                corr = np.correlate(x, y, mode="full")
                N = len(x)
                lags = np.arange(-N + 1, N, dtype=int)

                # scale
                if scaleOpt == "biased":
                    corr = corr / N
                elif scaleOpt == "unbiased":
                    denom = N - np.abs(lags)
                    denom[denom <= 0] = 1
                    corr = corr / denom
                elif scaleOpt == "coeff":
                    corr = corr / corr[N - 1] if corr[N - 1] != 0 else corr

                if maxlag is not None:
                    keep = np.abs(lags) <= int(maxlag)
                    corr = corr[keep]
                    lags = lags[keep]

                # Matlab returns only non-negative lags for auto-covariance
                if auto:
                    nonneg = lags >= 0
                    corr = corr[nonneg]
                    lags = lags[nonneg]

                if lag_index is None:
                    lag_index = lags.astype(float) / max(float(s1c.sampleRate), 1e-12)
                data_columns.append(np.asarray(corr, dtype=float))
                ll = s1c.dataLabels[li] if li < len(s1c.dataLabels) else str(li + 1)
                rl = s2c.dataLabels[ri] if ri < len(s2c.dataLabels) else str(ri + 1)
                data_labels.append(f"xcov({ll},{rl})")

        data = np.column_stack(data_columns) if data_columns else np.zeros((0, 0), dtype=float)
        return self.__class__(
            lag_index if lag_index is not None else np.array([], dtype=float),
            data,
            f"xcov({self.name},{s2.name})",
            "\\Delta \\tau",
            self.xunits,
            f"{self.yunits}^2" if self.yunits else "",
            data_labels,
        )

    # ------------------------------------------------------------------
    # Spectral methods (match Matlab SignalObj)
    # ------------------------------------------------------------------
    def periodogram(self, NFFT: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Power spectral density via periodogram (Matlab ``periodogram``).

        Loops over all signal dimensions like the Matlab implementation.

        Returns ``(frequencies, psd)`` where *psd* has shape
        ``(nfreqs,)`` for 1-D signals or ``(nfreqs, dimension)`` for
        multi-dimensional signals.
        """
        from scipy.signal import periodogram as _periodogram

        fs = float(self.sampleRate)
        psd_cols: list[np.ndarray] = []
        f_out: np.ndarray | None = None
        ndim = self.dimension
        for i in range(ndim):
            x = self.data[:, i] if self.data.ndim == 2 else self.data
            f, Pxx = _periodogram(x, fs=fs, nfft=NFFT, window="boxcar",
                                  scaling="density")
            if f_out is None:
                f_out = f
            psd_cols.append(Pxx)
        if ndim == 1:
            return f_out, psd_cols[0]
        return f_out, np.column_stack(psd_cols)

    def MTMspectrum(self, NW: float = 4.0, NFFT: int | None = None,
                    Pval: float = 0.95,
                    Kmax: int | None = None,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Multi-taper spectral estimate (Matlab ``MTMspectrum``).

        Uses discrete prolate spheroidal sequences (DPSS / Slepian tapers).
        Loops over all signal dimensions like the Matlab implementation.

        Parameters
        ----------
        NW : float
            Time-bandwidth product (default 4).
        NFFT : int, optional
            FFT length (default next power of 2 >= N).
        Pval : float, optional
            Confidence level for the chi-squared confidence interval
            (default 0.95).  Set to ``None`` to skip CI computation.
        Kmax : int, optional
            Number of tapers (default ``2*NW - 1``).

        Returns
        -------
        frequencies : ndarray
        psd : ndarray
            Shape ``(nfreqs,)`` for 1-D or ``(nfreqs, dimension)``.
        psd_ci : ndarray or None
            Shape ``(nfreqs, 2)`` for 1-D or ``(nfreqs, 2*dimension)``
            containing ``[lower, upper]`` columns per dimension.
            ``None`` when *Pval* is ``None``.
        """
        from scipy.signal.windows import dpss

        N = self.data.shape[0]
        fs = float(self.sampleRate)
        if Kmax is None:
            Kmax = int(2 * NW - 1)
        if NFFT is None:
            NFFT = int(2 ** np.ceil(np.log2(N)))

        tapers, eigenvalues = dpss(N, NW, Kmax, return_ratios=True)
        frequencies = np.fft.rfftfreq(NFFT, d=1.0 / fs)
        nfreqs = len(frequencies)

        # chi-squared CI bounds (degrees of freedom = 2*Kmax)
        ci_lo_factor = ci_hi_factor = None
        if Pval is not None:
            from scipy.stats import chi2
            dof = 2 * Kmax
            alpha = 1.0 - Pval
            ci_lo_factor = dof / chi2.ppf(1.0 - alpha / 2.0, dof)
            ci_hi_factor = dof / chi2.ppf(alpha / 2.0, dof)

        ndim = self.dimension
        psd_cols: list[np.ndarray] = []
        ci_cols: list[np.ndarray] = []

        for di in range(ndim):
            x = self.data[:, di] if self.data.ndim == 2 else self.data
            Sk = np.zeros((Kmax, nfreqs))
            for k in range(Kmax):
                xw = x * tapers[k]
                Xf = np.fft.rfft(xw, n=NFFT)
                Sk[k] = np.abs(Xf) ** 2

            weights = eigenvalues / eigenvalues.sum()
            psd = np.dot(weights, Sk) * (2.0 / fs)
            psd[0] /= 2.0
            if NFFT % 2 == 0:
                psd[-1] /= 2.0
            psd_cols.append(psd)

            if Pval is not None:
                ci_cols.append(psd * ci_lo_factor)
                ci_cols.append(psd * ci_hi_factor)

        if ndim == 1:
            psd_out = psd_cols[0]
            ci_out = np.column_stack(ci_cols) if ci_cols else None
        else:
            psd_out = np.column_stack(psd_cols)
            ci_out = np.column_stack(ci_cols) if ci_cols else None

        return frequencies, psd_out, ci_out

    def spectrogram(self, nperseg: int = 256, noverlap: int | None = None,
                    NFFT: int | None = None,
                    window: str = "hann") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Short-time Fourier transform spectrogram (Matlab ``spectrogram``).

        Returns ``(frequencies, times, Sxx)``.
        """
        from scipy.signal import spectrogram as _spectrogram

        x = self.data[:, 0] if self.data.ndim == 2 else self.data
        fs = float(self.sampleRate)
        if noverlap is None:
            noverlap = nperseg // 2
        if NFFT is None:
            NFFT = nperseg
        f, t, Sxx = _spectrogram(x, fs=fs, window=window,
                                  nperseg=nperseg, noverlap=noverlap,
                                  nfft=NFFT)
        # offset times to match signal start
        t = t + self.minTime
        return f, t, Sxx

    def setConfInterval(self, bounds: tuple[np.ndarray, np.ndarray]) -> None:
        """Attach ``(lower, upper)`` confidence bounds aligned with time."""
        low, high = bounds
        low_arr = np.asarray(low, dtype=float)
        high_arr = np.asarray(high, dtype=float)
        if low_arr.shape[0] != self.time.shape[0] or high_arr.shape[0] != self.time.shape[0]:
            raise ValueError("confidence interval bounds must align with time.")
        self.conf_interval = (low_arr, high_arr)

    def dataToStructure(self, selectorArray: Sequence[int] | np.ndarray | None = None) -> dict[str, Any]:
        """Serialize signal data to a plain dict (Matlab ``dataToStructure``)."""
        data = self.dataToMatrix(selectorArray)
        plot_props = list(self.plotProps)
        if all(prop is None for prop in plot_props):
            plot_props = []
        return {
            "time": self.time.tolist(),
            "data": data.tolist(),
            "name": self.name,
            "xlabelval": self.xlabelval,
            "xunits": self.xunits,
            "yunits": self.yunits,
            "dataLabels": list(self.dataLabels),
            "plotProps": plot_props,
        }

    def toStructure(self) -> dict[str, Any]:
        """Serialize the full signal to a plain dict (Matlab ``toStructure``)."""
        return self.dataToStructure()

    @staticmethod
    def signalFromStruct(structure: dict[str, Any]) -> "SignalObj":
        """Reconstruct a ``SignalObj`` from a dict (Matlab ``signalFromStruct``)."""
        return SignalObj(
            structure["time"],
            structure["data"],
            structure.get("name", ""),
            structure.get("xlabelval", "time"),
            structure.get("xunits", "s"),
            structure.get("yunits", ""),
            structure.get("dataLabels"),
            structure.get("plotProps"),
        )

    def plot(self, selectorArray=None, plotPropsIn=None, handle=None):
        """Plot selected signal dimensions (Matlab ``plot``).

        Parameters
        ----------
        selectorArray : optional
            Dimension selector (labels, 1-based indices, or ``None`` for all
            visible dimensions).
        plotPropsIn : optional
            Override Matplotlib format strings for each dimension.
        handle : matplotlib Axes, optional
            Axes to draw into; defaults to ``plt.gca()``.

        Returns
        -------
        list of Line2D
        """
        import matplotlib.pyplot as plt
        from .confidence_interval import MATLAB_COLOR_ORDER

        ax = plt.gca() if handle is None else handle
        signal = self.getSubSignal(selectorArray) if selectorArray is not None else self.getSubSignal(self.findIndFromDataMask() or list(range(1, self.dimension + 1)))
        props = signal.plotProps if plotPropsIn is None else list(plotPropsIn)
        if len(props) == 1 and signal.dimension > 1:
            props = props * signal.dimension
        if not props:
            props = [None for _ in range(signal.dimension)]

        lines = []
        for index in range(signal.dimension):
            kwargs = {}
            prop = props[index]
            if isinstance(prop, str) and prop:
                kwargs["fmt"] = prop
            elif prop is None:
                kwargs["color"] = MATLAB_COLOR_ORDER[index % MATLAB_COLOR_ORDER.shape[0]]
            if "fmt" in kwargs:
                fmt = kwargs.pop("fmt")
                line = ax.plot(signal.time, signal.data[:, index], fmt, **kwargs)
            else:
                line = ax.plot(signal.time, signal.data[:, index], **kwargs)
            lines.extend(line)

        if handle is None:
            xunits = f" [{signal.xunits}]" if signal.xunits else ""
            yunits = f" [{signal.yunits}]" if signal.yunits else ""
            ax.set_xlabel(f"{signal.xlabelval}{xunits}")
            ax.set_ylabel(f"{signal.name}{yunits}")
        return lines


class Covariate(SignalObj):
    """Signal with per-dimension confidence intervals (Matlab ``Covariate``).

    ``Covariate`` extends :class:`SignalObj` with a list of
    :class:`~nstat.confidence_interval.ConfidenceInterval` objects (one per
    dimension) and propagates those intervals through ``+`` and ``-``
    arithmetic.  It also provides ``'zero-mean'`` and ``'standard'``
    signal representations used by the GLM design-matrix builder.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to :class:`SignalObj`.  The keyword aliases ``values``
        (→ ``data``) and ``units`` (→ ``yunits``) are accepted for
        convenience.

    See Also
    --------
    SignalObj : Base time-series container.
    ConfidenceInterval : CI storage class used by ``ci``.
    """

    def __init__(self, *args, **kwargs) -> None:
        if "values" in kwargs and "data" not in kwargs:
            kwargs["data"] = kwargs.pop("values")
        if "units" in kwargs and "yunits" not in kwargs:
            kwargs["yunits"] = kwargs.pop("units")
        super().__init__(*args, **kwargs)
        self.ci: list[Any] | None = None

    @property
    def mu(self) -> SignalObj:
        """Column-wise mean as a ``SignalObj`` (Matlab ``mu`` property)."""
        return self.mean()

    @property
    def sigma(self) -> SignalObj:
        """Column-wise standard deviation as a ``SignalObj`` (Matlab ``sigma``)."""
        return self.std()

    def computeMeanPlusCI(self, alphaVal: float = 0.05) -> "Covariate":
        """Compute row-wise mean with empirical confidence intervals.

        Treats each column as a replicate.  Returns a scalar ``Covariate``
        whose CI bounds are the *alphaVal*/2 and 1−*alphaVal*/2 quantiles
        of the empirical CDF across replicates (Matlab ``computeMeanPlusCI``).
        """
        from .confidence_interval import ConfidenceInterval

        sorted_data = np.sort(self.data, axis=1)
        n_rep = sorted_data.shape[1]
        if n_rep == 0:
            raise ValueError("Covariate must contain at least one column to compute confidence intervals.")
        ecdf = np.arange(1, n_rep + 1, dtype=float) / float(n_rep)
        lower = np.empty(sorted_data.shape[0], dtype=float)
        upper = np.empty(sorted_data.shape[0], dtype=float)
        for row_idx in range(sorted_data.shape[0]):
            row = sorted_data[row_idx]
            lower_idx = np.flatnonzero(ecdf < (alphaVal / 2.0))
            upper_idx = np.flatnonzero(ecdf > (1.0 - alphaVal / 2.0))
            lower[row_idx] = row[int(lower_idx[-1])] if lower_idx.size else row[0]
            upper[row_idx] = row[int(upper_idx[0])] if upper_idx.size else row[-1]
        confInt = ConfidenceInterval(self.time, np.column_stack([lower, upper]))
        mean_signal = np.mean(self.data, axis=1)
        newCov = Covariate(
            self.time.copy(),
            mean_signal,
            self.name,
            self.xlabelval,
            self.xunits,
            self.yunits,
            [f"\\mu({self.name})" if self.name else "\\mu"],
        )
        newCov.setConfInterval(confInt)
        return newCov

    def getSigRep(self, repType: str = "standard") -> "Covariate":
        """Return a signal representation of this covariate.

        Parameters
        ----------
        repType : str
            ``'standard'`` returns ``self`` unchanged.
            ``'zero-mean'`` returns ``self - mean(self)`` with confidence
            intervals propagated (Matlab parity: uses operator overload so
            CIs shift by the same constant).
        """
        rep = str(repType).strip().lower()
        if rep == "standard":
            return self
        if rep == "zero-mean":
            # Build a constant Covariate holding the per-column mean so that
            # the CI-propagating __sub__ is invoked (Matlab: ``self - self.mu``).
            mu_vals = np.mean(self.data, axis=0, keepdims=True)
            mu_broadcast = np.repeat(mu_vals, len(self.time), axis=0)
            mu_cov = Covariate(
                self.time.copy(),
                mu_broadcast,
                self.name,
                self.xlabelval,
                self.xunits,
                self.yunits,
                list(self.dataLabels),
                list(self.plotProps),
            )
            return self - mu_cov
        raise ValueError("repType must be either 'zero-mean' or 'standard'")

    def plot(self, selectorArray=None, plotPropsIn=None, handle=None):
        """Plot signal dimensions with shaded confidence intervals."""
        lines = super().plot(selectorArray, plotPropsIn, handle)
        if self.isConfIntervalSet():
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            ax = plt.gca() if handle is None else handle
            selectors = self.findIndFromDataMask() if selectorArray is None else (
                self.getIndicesFromLabels(selectorArray) if isinstance(selectorArray, str) else list(np.asarray(selectorArray).reshape(-1))
            )
            if not isinstance(selectors, list):
                selectors = [selectors]
            if selectors and isinstance(selectors[0], list):
                selectors = [item[0] for item in selectors]
            for line_index, selector in enumerate(selectors):
                color = getattr(lines[line_index], "get_color", lambda: "b")()
                if isinstance(color, (str, bytes)):
                    color = mcolors.to_rgb(color)
                self.ci[selector - 1].plot(color, ax=ax)
        return lines

    def isConfIntervalSet(self) -> bool:
        """Return ``True`` if at least one dimension has a CI attached."""
        return bool(self.ci)

    def setConfInterval(self, ciObj) -> None:
        """Attach one or more ``ConfidenceInterval`` objects to this covariate."""
        if isinstance(ciObj, list):
            self.ci = list(ciObj)
        else:
            self.ci = [ciObj]

    def copySignal(self) -> "Covariate":
        """Deep-copy including confidence intervals (Matlab ``copySignal``)."""
        copied = Covariate(
            self.time.copy(),
            self.data.copy(),
            self.name,
            self.xlabelval,
            self.xunits,
            self.yunits,
            list(self.dataLabels),
            list(self.plotProps),
        )
        copied.dataMask = np.asarray(self.dataMask, dtype=int).copy()
        copied.originalTime = self.originalTime.copy()
        copied.originalData = self.originalData.copy()
        copied.sampleRate = float(self.sampleRate)
        copied.origSampleRate = float(self.origSampleRate)
        copied.minTime = float(self.minTime)
        copied.maxTime = float(self.maxTime)
        copied.ci = None if not self.ci else list(self.ci)
        if self.conf_interval is not None:
            copied.conf_interval = (
                np.asarray(self.conf_interval[0], dtype=float).copy(),
                np.asarray(self.conf_interval[1], dtype=float).copy(),
            )
        return copied

    def getSubSignal(self, identifier) -> "Covariate":
        """Return a sub-covariate preserving matching CIs."""
        sub = super().getSubSignal(identifier)
        cov = Covariate(
            sub.time,
            sub.data,
            sub.name,
            sub.xlabelval,
            sub.xunits,
            sub.yunits,
            list(sub.dataLabels),
            list(sub.plotProps),
        )
        if self.isConfIntervalSet():
            selected: list[int] = []
            for label in cov.dataLabels:
                if label:
                    match = next((i for i, original in enumerate(self.dataLabels) if original == label), None)
                    if match is None:
                        raise ValueError("Unable to align Covariate confidence interval with sub-signal labels.")
                    selected.append(match)
                else:
                    selected.append(len(selected))
            cov.setConfInterval([self.ci[index] for index in selected])
        return cov

    def __add__(self, other):
        """Add two covariates, propagating confidence intervals."""
        covOut = super().__add__(other)
        if isinstance(other, Covariate):
            if self.isConfIntervalSet() and not other.isConfIntervalSet():
                covOut.setConfInterval([self.ci[index] + other.getSubSignal(index + 1) for index in range(self.dimension)])
            elif self.isConfIntervalSet() and other.isConfIntervalSet():
                covOut.setConfInterval([self.ci[index] + other.ci[index] for index in range(self.dimension)])
            elif (not self.isConfIntervalSet()) and other.isConfIntervalSet():
                covOut.setConfInterval([other.ci[index] + self.getSubSignal(index + 1) for index in range(other.dimension)])
        return covOut

    def __sub__(self, other):
        """Subtract two covariates, propagating confidence intervals."""
        covOut = super().__sub__(other)
        if isinstance(other, Covariate):
            if self.isConfIntervalSet() and not other.isConfIntervalSet():
                covOut.setConfInterval([self.ci[index] - other.getSubSignal(index + 1) for index in range(self.dimension)])
            elif self.isConfIntervalSet() and other.isConfIntervalSet():
                covOut.setConfInterval([self.ci[index] - other.ci[index] for index in range(self.dimension)])
            elif (not self.isConfIntervalSet()) and other.isConfIntervalSet():
                covOut.setConfInterval([self.getSubSignal(index + 1) - other.ci[index] for index in range(other.dimension)])
        return covOut

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a dict, including CI payload if present."""
        structure = super().toStructure()
        if self.isConfIntervalSet():
            ci_payload: list[dict[str, Any]] = []
            for item in self.ci or []:
                if hasattr(item, "dataToStructure"):
                    ci_payload.append(item.dataToStructure())
            if ci_payload:
                structure["ci"] = ci_payload[0] if len(ci_payload) == 1 else ci_payload
        return structure

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "Covariate":
        """Reconstruct a ``Covariate`` (with optional CIs) from a dict."""
        from .confidence_interval import ConfidenceInterval

        cov = Covariate(
            structure["time"],
            structure["data"],
            structure.get("name", ""),
            structure.get("xlabelval", "time"),
            structure.get("xunits", "s"),
            structure.get("yunits", ""),
            structure.get("dataLabels"),
            structure.get("plotProps"),
        )
        ci_payload = structure.get("ci")
        if ci_payload is None:
            return cov
        if isinstance(ci_payload, list):
            cov.setConfInterval([ConfidenceInterval.fromStructure(item) for item in ci_payload])
        elif isinstance(ci_payload, tuple):
            cov.setConfInterval([ConfidenceInterval.fromStructure(item) for item in ci_payload])
        else:
            cov.setConfInterval(ConfidenceInterval.fromStructure(ci_payload))
        return cov


class nspikeTrain:
    """Point-process (spike train) object (Matlab ``nspikeTrain``).

    Stores an array of event times (spikes) and converts them on demand
    into a binned ``SignalObj`` signal representation (``sigRep``).  Burst
    statistics, ISI analysis, and raster plotting are built in.

    Parameters
    ----------
    spikeTimes : array_like
        Spike times in seconds.
    name : str, optional
        Neuron / channel label.
    binwidth : float, optional
        Bin width in seconds for the signal representation (default 1 ms).
    minTime, maxTime : float, optional
        Observation window.  Defaults to ``min/max(spikeTimes)``.
    xlabelval, xunits, yunits : str, optional
        Axis label and unit strings.
    dataLabels : str or sequence of str, optional
        Label(s) for the spike-train dimension.
    makePlots : int, optional
        ``0`` — compute statistics silently (default);
        ``1`` — compute and plot;
        ``< 0`` — skip statistics entirely (fast construction).

    See Also
    --------
    SignalObj : Continuous time-series container returned by ``getSigRep``.
    SpikeTrainCollection : Multi-neuron collection.
    """

    def __init__(
        self,
        spikeTimes,
        name: str = "",
        binwidth: float = 0.001,
        minTime: float | None = None,
        maxTime: float | None = None,
        xlabelval: str = "time",
        xunits: str = "s",
        yunits: str = "",
        dataLabels: str | Sequence[str] | None = "",
        makePlots: int = 0,
    ) -> None:
        if spikeTimes is None:
            raise ValueError("nspikeTrain requires a spikeTimes array as input to create an object")
        spikes = np.asarray(spikeTimes, dtype=float).reshape(-1)
        self.spikeTimes = np.sort(spikes)
        self.originalSpikeTimes = self.spikeTimes.copy()
        self.name = str(name)
        self.sampleRate = float(1.0 / float(binwidth))
        self.originalSampleRate = float(self.sampleRate)
        if minTime is None:
            minTime = float(np.min(self.spikeTimes)) if self.spikeTimes.size else 0.0
        if maxTime is None:
            maxTime = float(np.max(self.spikeTimes)) if self.spikeTimes.size else 0.0
        self.minTime = float(minTime)
        self.maxTime = float(maxTime)
        self.originalMinTime = float(self.minTime)
        self.originalMaxTime = float(self.maxTime)
        self.xlabelval = str(xlabelval)
        self.xunits = str(xunits)
        self.yunits = str(yunits)
        self.dataLabels = dataLabels if dataLabels is not None else ""
        self.sigRep: SignalObj | None = None
        self.isSigRepBin: bool | None = None
        self._sigrep_cache_key: tuple[float, float, float] | None = None
        self.MER = None
        if makePlots >= 0:
            self.computeStatistics(makePlots)
        else:
            self.avgFiringRate = None
            self.B = None
            self.An = None
            self.burstTimes = None
            self.burstRate = None
            self.burstDuration = None
            self.burstSig = None
            self.burstIndex = None
            self.numBursts = None
            self.numSpikesPerBurst = None
            self.avgSpikesPerBurst = None
            self.stdSpikesPerBurst = None
            self.Lstatistic = None

    @property
    def times(self) -> np.ndarray:
        """Alias for ``spikeTimes``."""
        return self.spikeTimes

    @property
    def n_spikes(self) -> int:
        """Number of spikes in the train."""
        return int(self.spikeTimes.size)

    @property
    def duration(self) -> float:
        """Observation window duration ``maxTime − minTime`` in seconds."""
        return float(self.maxTime - self.minTime)

    @property
    def firing_rate_hz(self) -> float:
        """Average firing rate (spikes / duration) in Hz."""
        if self.duration <= 0:
            return 0.0
        return float(self.n_spikes / self.duration)

    def setMER(self, MERSig: SignalObj) -> None:
        """Attach a micro-electrode recording signal to this spike train."""
        if isinstance(MERSig, SignalObj):
            self.MER = MERSig

    def setName(self, name: str) -> None:
        """Set the neuron / channel name."""
        self.name = str(name)

    def computeStatistics(self, makePlots: int = 0) -> None:
        """Compute ISI, burst, and regularity statistics (Matlab ``computeStatistics``)."""
        self.avgFiringRate = self.firing_rate_hz
        isi = self.getISIs()
        # Filter spike times to [minTime, maxTime] so burst statistics
        # remain valid after setMinTime / setMaxTime (Matlab parity).
        spike_times = self.getSpikeTimes(self.minTime, self.maxTime)
        mode_isi = _matlab_mode_1d(isi)
        self.burstIndex = float(1.0 / mode_isi / self.avgFiringRate) if np.isfinite(mode_isi) and self.avgFiringRate > 0 else np.nan
        self.B = np.nan
        self.An = np.nan
        self.burstTimes = np.array([], dtype=float)
        self.burstRate = np.array([], dtype=float)
        self.burstDuration = np.array([], dtype=float)
        self.burstSig = None
        self.numBursts = 0
        self.numSpikesPerBurst = np.array([], dtype=float)
        self.avgSpikesPerBurst = np.nan
        self.stdSpikesPerBurst = np.nan
        self.Lstatistic = np.nan

        if isi.size:
            sigma = float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0
            mu = float(np.mean(isi))
            if np.isfinite(mu) and mu > 0:
                r = sigma / mu
                self.B = float((r - 1.0) / (r + 1.0))
                n = float(spike_times.size)
                self.An = float((np.sqrt(n + 2.0) * r - np.sqrt(n)) / (((np.sqrt(n + 2.0) - 2.0) * r) + np.sqrt(n)))

                ln = isi[isi < mu]
                ml = float(np.mean(ln)) if ln.size else np.nan
                if np.isfinite(ml):
                    burst_isi = (isi < ml).astype(float)
                    shifted = np.concatenate([burst_isi[1:], [0.0]]) if burst_isi.size else np.array([], dtype=float)
                    y = (burst_isi + shifted) > 1.0
                    diff_sig = np.concatenate([[0.0], np.diff(y.astype(float))]) if y.size else np.array([], dtype=float)
                    burst_start = np.flatnonzero(diff_sig == 1.0)
                    burst_end = np.flatnonzero(diff_sig == -1.0) + 1
                    if burst_start.size == 0:
                        burst_end = np.array([], dtype=int)
                    if burst_end.size > burst_start.size and burst_end.size:
                        first = np.flatnonzero(y[: burst_end[0]] == 1)
                        if first.size:
                            burst_start = np.concatenate([[int(first[0])], burst_start])
                    if burst_start.size > burst_end.size and burst_start.size:
                        last = np.flatnonzero(y[burst_start[-1] :] == 1)
                        if last.size:
                            burst_end = np.concatenate([burst_end, [int(last[-1])]])
                    if burst_start.size and burst_end.size:
                        burst_data = np.zeros(spike_times.size, dtype=float)
                        for start, end in zip(burst_start, burst_end, strict=False):
                            burst_data[int(start) : int(end) + 1] = 1.0
                        self.burstDuration = spike_times[burst_end] - spike_times[burst_start]
                        self.burstSig = SignalObj(spike_times, burst_data, "Burst Signal")
                        self.burstTimes = spike_times[burst_start]
                        self.numBursts = int(burst_start.size)
                        duration = self.maxTime - self.minTime
                        self.burstRate = float(self.numBursts / duration) if duration > 0 else np.nan
                        self.numSpikesPerBurst = (burst_end - burst_start + 1).astype(float)
                        self.avgSpikesPerBurst = float(np.mean(self.numSpikesPerBurst + 1.0))
                        if self.numSpikesPerBurst.size > 1:
                            self.stdSpikesPerBurst = float(np.std(self.numSpikesPerBurst + 1.0, ddof=1))
                        elif self.numSpikesPerBurst.size == 1:
                            self.stdSpikesPerBurst = 0.0

        self.Lstatistic = self.getLStatistic()
        if makePlots == 1:
            self.plot()

    def getLStatistic(self) -> float:
        """Return the L-statistic (number of unique bin counts in ``sigRep``)."""
        isi = self.getISIs()
        if isi.size == 0:
            return np.nan
        mean_isi = float(np.mean(isi))
        if not np.isfinite(mean_isi) or mean_isi <= 0:
            return np.nan
        duration = self.maxTime - self.minTime
        if not np.isfinite(duration) or duration <= 0:
            return np.nan
        approx = self.getSigRep(mean_isi)
        return float(np.unique(approx.data[:, 0]).size)

    def _cache_key(self, binwidth: float, minTime: float, maxTime: float) -> tuple[float, float, float]:
        return (round(float(binwidth), 12), round(float(minTime), 12), round(float(maxTime), 12))

    def _build_sigrep(self, binwidth: float, minTime: float, maxTime: float) -> SignalObj:
        if binwidth <= 0:
            raise ValueError("binwidth must be > 0")
        if maxTime < minTime:
            raise ValueError("maxTime must be >= minTime")

        max_bins = int(1e6)
        precision = max(0, int(2 * np.ceil(np.log10(1.0 / binwidth))))
        bw = float(_roundn([binwidth], precision)[0])
        duration = float(maxTime - minTime)
        if np.isfinite(duration) and duration > 0 and np.isfinite(bw) and bw > 0:
            est_bins = duration / bw + 1.0
            if not np.isfinite(est_bins) or est_bins > max_bins:
                bw = duration / float(max_bins - 1)
                precision = max(0, int(2 * np.ceil(np.log10(1.0 / bw))))
                bw = float(_roundn([bw], precision)[0])
        if not np.isfinite(bw) or bw <= 0:
            bw = duration / float(max_bins - 1) if np.isfinite(duration) and duration > 0 else 1.0 / max(self.sampleRate, 1.0)

        numBins = int(np.floor(duration / bw + 1.0)) if np.isfinite(duration) else 2
        if numBins < 2:
            numBins = 2
        if numBins > max_bins:
            numBins = max_bins
        timeVec = np.linspace(minTime, maxTime, numBins, dtype=float)
        if timeVec.size > 1:
            bw = float(np.mean(np.diff(timeVec)))
        windowTimes = np.concatenate([[minTime - bw / 2.0], timeVec + bw / 2.0])

        spikeTimes = _roundn(self.spikeTimes, precision)
        rounded_windows = _roundn(windowTimes, precision + 1)
        counts = np.zeros(timeVec.size, dtype=float)
        split_index = int(np.floor(rounded_windows.size / 2.0))
        for idx in range(timeVec.size):
            left = rounded_windows[idx]
            right = rounded_windows[idx + 1]
            if idx == rounded_windows.size - 2:
                temp = spikeTimes[spikeTimes >= left]
                counts[idx] = float(np.sum(temp <= right))
            elif idx + 1 > split_index:
                temp = spikeTimes[spikeTimes >= left]
                counts[idx] = float(np.sum(temp < right))
            else:
                temp = spikeTimes[spikeTimes < right]
                counts[idx] = float(np.sum(temp >= left))

        label = self.dataLabels if isinstance(self.dataLabels, str) else ""
        sig = SignalObj(timeVec, counts.astype(float), self.name, self.xlabelval, self.xunits, self.yunits, label)
        self.isSigRepBin = bool(np.all(counts <= 1))
        return sig

    def setSigRep(self, binwidth: float | None = None, minTime: float | None = None, maxTime: float | None = None) -> SignalObj:
        """Build the binned signal representation and store it in-place."""
        sig = self.getSigRep(binwidth, minTime, maxTime)
        self.sigRep = sig.copySignal()
        self.sampleRate = float(sig.sampleRate)
        self.isSigRepBin = bool(np.max(np.asarray(sig.data, dtype=float)) <= 1.0)
        # Keep the freshly-built cached representation alive instead of
        # clearing it through the public min/max setters.
        self.minTime = float(sig.minTime)
        self.maxTime = float(sig.maxTime)
        self.computeStatistics(0)
        return self.sigRep

    def clearSigRep(self) -> None:
        """Invalidate the cached signal representation."""
        self.sigRep = None
        self._sigrep_cache_key = None
        self.isSigRepBin = None

    def setMinTime(self, minTime: float) -> None:
        """Set the observation-window start and recompute statistics."""
        self.minTime = float(minTime)
        self.clearSigRep()
        self.computeStatistics(0)

    def setMaxTime(self, maxTime: float) -> None:
        """Set the observation-window end and recompute statistics."""
        self.maxTime = float(maxTime)
        self.clearSigRep()
        self.computeStatistics(0)

    def resample(self, sampleRate: float) -> "nspikeTrain":
        """Rebuild the signal representation at *sampleRate* Hz."""
        self.setSigRep(1.0 / float(sampleRate), self.minTime, self.maxTime)
        self.sampleRate = float(sampleRate)
        return self

    def getSpikeTimes(self, minTime: float | None = None, maxTime: float | None = None) -> np.ndarray:
        """Return spike times within ``[minTime, maxTime]``."""
        start = self.minTime if minTime is None else float(minTime)
        stop = self.maxTime if maxTime is None else float(maxTime)
        spikes = self.spikeTimes[(self.spikeTimes >= start) & (self.spikeTimes <= stop)]
        return spikes.copy()

    def getISIs(self, minTime: float | None = None, maxTime: float | None = None) -> np.ndarray:
        """Return inter-spike intervals within the given time window."""
        spikes = self.getSpikeTimes(minTime, maxTime)
        if spikes.size < 2:
            return np.array([], dtype=float)
        return np.diff(spikes)

    def getMinISI(self, minTime: float | None = None, maxTime: float | None = None) -> float:
        """Return the minimum ISI (refractory period estimate)."""
        isi = self.getISIs(minTime, maxTime)
        if isi.size == 0:
            return float("nan")
        return float(np.min(isi))

    def getSigRep(
        self,
        binwidth: float | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> SignalObj:
        """Return the binned signal representation, using cache when possible.

        The result is a ``SignalObj`` of spike counts on a regular grid
        with bin width *binwidth* (default ``1/sampleRate``).
        """
        bw = (1.0 / self.sampleRate) if binwidth is None else float(binwidth)
        start = self.minTime if minTime is None else float(minTime)
        stop = self.maxTime if maxTime is None else float(maxTime)
        key = self._cache_key(bw, start, stop)
        if self.sigRep is not None and self._sigrep_cache_key == key:
            return self.sigRep.copySignal()
        sig = self._build_sigrep(bw, start, stop)
        self.sigRep = sig.copySignal()
        self._sigrep_cache_key = key
        return sig

    def getMaxBinSizeBinary(self) -> float:
        """Return the largest bin width that keeps the ``sigRep`` binary."""
        isi = self.getISIs()
        if isi.size == 0:
            return np.inf
        return float(np.min(isi))

    def isSigRepBinary(self) -> bool:
        """Return ``True`` if every bin in the default ``sigRep`` has ≤ 1 spike."""
        default_key = self._cache_key(1.0 / float(self.sampleRate), float(self.minTime), float(self.maxTime))
        if self._sigrep_cache_key != default_key or self.isSigRepBin is None:
            self.getSigRep(1.0 / float(self.sampleRate), float(self.minTime), float(self.maxTime))
        return bool(self.isSigRepBin)

    def computeRate(self) -> SignalObj:
        """Return firing rate ``sigRep × sampleRate`` in spikes/sec."""
        sig = self.getSigRep()
        if self.sampleRate <= 0:
            return sig
        rate = np.asarray(sig.data[:, 0], dtype=float) * float(self.sampleRate)
        return SignalObj(sig.time, rate, self.name, sig.xlabelval, sig.xunits, "spikes/sec", sig.dataLabels)

    def restoreToOriginal(self) -> None:
        """Reset spike times and time bounds to original values."""
        self.spikeTimes = self.originalSpikeTimes.copy()
        self.minTime = float(np.min(self.spikeTimes)) if self.spikeTimes.size else 0.0
        self.maxTime = float(np.max(self.spikeTimes)) if self.spikeTimes.size else 0.0
        self.clearSigRep()

    def partitionNST(
        self,
        windowTimes: Sequence[float],
        normalizeTime: int | bool | None = None,
        lbound: float | None = None,
        ubound: float | None = None,
    ):
        """Partition into per-trial spike trains (Matlab ``partitionNST``).

        Parameters
        ----------
        windowTimes : sequence of float
            Edge times defining trial boundaries (N edges → N−1 trials).
        normalizeTime : bool, optional
            If ``True``, rescale each trial's spikes to [0, 1].
        lbound, ubound : float, optional
            Accept only windows whose duration falls in ``[lbound, ubound]``.

        Returns
        -------
        nstColl
        """
        from .nstColl import nstColl

        windows = np.asarray(windowTimes, dtype=float).reshape(-1)
        if windows.size <= 1:
            return nstColl([])
        if ubound is None:
            ubound = lbound

        normalize = bool(normalizeTime) if normalizeTime is not None else False
        partitions: list[nspikeTrain] = []
        for index, (window_start, window_stop) in enumerate(zip(windows[:-1], windows[1:]), start=1):
            window_start = round(float(window_start) * self.sampleRate) / self.sampleRate
            window_stop = round(float(window_stop) * self.sampleRate) / self.sampleRate
            duration = float(window_stop - window_start)
            if lbound is not None and ubound is not None and not (float(lbound) <= abs(duration) <= float(ubound)):
                continue
            if index == windows.size - 1:
                subset = self.spikeTimes[(self.spikeTimes >= window_start) & (self.spikeTimes <= window_stop)]
            else:
                subset = self.spikeTimes[(self.spikeTimes >= window_start) & (self.spikeTimes < window_stop)]
            subset = subset - float(window_start)
            if normalize and duration != 0:
                subset = subset / duration
            partitions.append(nspikeTrain(subset, self.name, makePlots=-1))

        coll = nstColl(partitions)
        if normalize:
            coll.setMinTime(0.0)
            coll.setMaxTime(1.0)
        return coll

    def getFieldVal(self, fieldName: str):
        """Return the value of attribute *fieldName* (Matlab ``getFieldVal``)."""
        return getattr(self, fieldName, [])

    def plotISISpectrumFunction(self):
        """Plot ISI vs. time (Matlab ``plotISISpectrumFunction``)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
        isi = self.getISIs()
        if isi.size:
            (line,) = ax.plot(self.spikeTimes[1:], isi, ".")
        else:
            (line,) = ax.plot([], [], ".")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("ISI [s]")
        return line

    def plotJointISIHistogram(self):
        """Joint ISI scatter plot: ISI(t) vs ISI(t+1) on log-log axes."""
        import matplotlib.pyplot as plt

        ax = plt.subplots(1, 1, figsize=(4.5, 4.0))[1]
        isi = self.getISIs()
        if isi.size >= 2:
            xvals = np.asarray(isi[:-1], dtype=float).reshape(-1)
            yvals = np.asarray(isi[1:], dtype=float).reshape(-1)
            ax.loglog(xvals, yvals, ".")
            mean_isi = float(np.mean(isi))
            ln = isi[isi < mean_isi]
            ml = float(np.mean(ln)) if ln.size else np.nan
            if np.isfinite(ml) and ml > 0:
                ymin = float(np.min(yvals))
                ymax = float(np.max(yvals))
                xmin = float(np.min(xvals))
                xmax = float(np.max(xvals))
                ax.loglog([ml, ml], [ymin, ymax], "k--")
                ax.loglog([xmin, xmax], [ml, ml], "k--")
        ax.set_xlabel("ISI(t) [s]")
        ax.set_ylabel("ISI(t+1) [s]")
        return ax

    def plotISIHistogram(self, minTime: float | None = None, maxTime: float | None = None, numBins: int | None = None, handle=None):
        """Plot ISI histogram (Matlab ``plotISIHistogram``).

        Parameters
        ----------
        minTime, maxTime : float, optional
            Time window for ISIs.  Defaults to the spike train bounds.
        numBins : int, optional
            Number of histogram bins.  When *None* the bin width defaults to
            1 ms (Matlab default behaviour).
        handle : matplotlib Axes, optional
            Axes to plot into.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if handle is None else handle
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        isi = self.getISIs(minTime, maxTime)
        counts = np.array([], dtype=float)
        bins = np.array([], dtype=float)
        if isi.size:
            isi_max = float(np.max(isi))
            if numBins is not None and int(numBins) > 0:
                # Linearly-spaced bins when numBins is specified (Matlab parity).
                n = int(numBins)
                bin_width = max(isi_max / n, 1e-12)
                bins = np.linspace(0.0, isi_max, n + 1, dtype=float)
            else:
                # Default: 1 ms bin width.
                bin_width = 0.001
                bins = np.arange(0.0, isi_max + bin_width, bin_width, dtype=float)
            if bins.size < 2:
                bins = np.array([0.0, bin_width], dtype=float)
            idx = np.searchsorted(bins, isi, side="right") - 1
            idx = np.where(
                np.isclose(isi, bins[-1], rtol=0.0, atol=max(1e-12, bin_width * 1e-9)),
                bins.size - 1,
                idx,
            )
            idx = np.clip(idx, 0, bins.size - 1)
            counts = np.bincount(idx, minlength=bins.size).astype(float)
            centers = bins[:counts.size] if bins.size > counts.size else bins
            ax.bar(
                centers,
                counts[:centers.size],
                width=bin_width,
                align="edge",
                edgecolor="none",
                linewidth=2.0,
                color=(0.831372559070587, 0.815686285495758, 0.7843137383461),
            )
        ax.set_xlabel("ISI [sec]")
        ax.set_ylabel("Spike Counts")
        ax.autoscale(enable=True, axis="x", tight=True)
        return counts

    def plotProbPlot(self, minTime: float | None = None, maxTime: float | None = None, handle=None):
        """Exponential probability plot of ISIs (Matlab ``plotProbPlot``)."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if handle is None else handle
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        isi = self.getISIs(minTime, maxTime)
        ax.clear()
        if isi.size:
            sorted_isi = np.sort(np.asarray(isi, dtype=float).reshape(-1))
            n = sorted_isi.size
            p = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)
            exp_quantiles = -np.log(1.0 - p)
            ax.plot(sorted_isi, exp_quantiles, linestyle="none", marker=".")
        return ax

    def plotExponentialFit(self, minTime: float | None = None, maxTime: float | None = None, numBins: int | None = None, handle=None):
        """ISI histogram + exponential prob-plot side by side."""
        import matplotlib.pyplot as plt

        fig = handle if handle is not None else plt.figure(figsize=(10.0, 4.0))
        fig.clear()
        axes = fig.subplots(1, 2)
        self.plotISIHistogram(minTime, maxTime, numBins, axes[0])
        self.plotProbPlot(minTime, maxTime, axes[1])
        fig.tight_layout()
        return fig

    def plot(self, dHeight: float = 1.0, yOffset: float = 0.5, currentHandle=None, handle=None):
        """Raster plot: vertical tick per spike (Matlab ``plot``).

        Parameters
        ----------
        dHeight : float
            Tick height (default 1.0).
        yOffset : float
            Vertical centre of ticks (default 0.5).
        currentHandle, handle : matplotlib Axes, optional
            Axes to draw into.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if (currentHandle is None and handle is None) else (currentHandle or handle)
        lines = []
        for spike_time in self.spikeTimes:
            (line,) = ax.plot(
                [spike_time, spike_time],
                [yOffset - dHeight / 2.0, yOffset + dHeight / 2.0],
                "k",
            )
            lines.append(line)
        if currentHandle is None and handle is None:
            xunits = f" [{self.xunits}]" if self.xunits else ""
            yunits = f" [{self.yunits}]" if self.yunits else ""
            ax.set_xlabel(f"{self.xlabelval}{xunits}")
            ax.set_ylabel(f"{self.name}{yunits}")
            if self.minTime != self.maxTime:
                ax.set_xlim(self.minTime, self.maxTime)
        return lines

    def nstCopy(self) -> "nspikeTrain":
        """Return a deep copy (Matlab ``nstCopy``).

        Matlab's ``nstCopy`` builds the copy's sigRep and calls
        ``computeStatistics(0)`` so the copy has valid burst parameters.
        """
        return nspikeTrain(
            self.spikeTimes.copy(),
            self.name,
            1.0 / self.sampleRate if self.sampleRate > 0 else 0.001,
            self.minTime,
            self.maxTime,
            self.xlabelval,
            self.xunits,
            self.yunits,
            self.dataLabels,
            0,
        )

    def to_binned_counts(self, bin_edges: Sequence[float]) -> np.ndarray:
        """Histogram spike times into *bin_edges* and return count vector."""
        edges = np.asarray(bin_edges, dtype=float).reshape(-1)
        counts, _ = np.histogram(self.spikeTimes, bins=edges)
        return counts.astype(float)

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a plain dict (Matlab ``toStructure``)."""
        return {
            "spikeTimes": self.spikeTimes.tolist(),
            "name": self.name,
            "sampleRate": self.sampleRate,
            "minTime": self.minTime,
            "maxTime": self.maxTime,
            "xlabelval": self.xlabelval,
            "xunits": self.xunits,
            "yunits": self.yunits,
            "dataLabels": self.dataLabels,
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "nspikeTrain":
        """Reconstruct an ``nspikeTrain`` from a dict."""
        sampleRate = float(structure.get("sampleRate", 1000.0))
        binwidth = 1.0 / sampleRate if sampleRate > 0 else 0.001
        return nspikeTrain(
            structure.get("spikeTimes", []),
            structure.get("name", ""),
            binwidth,
            structure.get("minTime"),
            structure.get("maxTime"),
            structure.get("xlabelval", "time"),
            structure.get("xunits", "s"),
            structure.get("yunits", ""),
            structure.get("dataLabels", ""),
            -1,
        )


# Backward-compatible alias used by earlier Python scaffolding.
SpikeTrain = nspikeTrain
