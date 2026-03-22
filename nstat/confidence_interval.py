from __future__ import annotations

import numpy as np


MATLAB_COLOR_ORDER = np.asarray(
    [
        [0.0660, 0.4430, 0.7450],
        [0.8660, 0.3290, 0.0000],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ],
    dtype=float,
)


class ConfidenceInterval:
    """Confidence interval for a time series or Covariate.

    Stores a pair of (lower, upper) bound traces over a shared time
    axis, with plotting support for both line and shaded-patch styles.

    Parameters
    ----------
    time : array_like
        Time vector.
    bounds : array_like, shape (n, 2)
        Lower and upper bounds at each time point.
    *args
        Positional metadata: ``(name, xlabelval, xunits, yunits,
        dataLabels, plotProps)``.  If a single short string is passed
        it is interpreted as the colour (Matlab compatibility).
    color : str or None
        Line / patch colour.  Default ``'b'``.
    value : float, default 0.95
        Confidence level (e.g. 0.95 for 95 %).
    """

    def __init__(self, time, bounds, *args, color: str | None = None, value: float = 0.95) -> None:
        t = np.asarray(time, dtype=float).reshape(-1)
        b = np.asarray(bounds, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must have shape (n, 2)")
        if b.shape[0] != t.shape[0]:
            raise ValueError("bounds rows must match time length")
        use_color_alias = bool(
            len(args) == 1
            and color is None
            and isinstance(args[0], str)
            and len(args[0]) <= 2
        )
        self.name = "" if use_color_alias or len(args) < 1 else str(args[0])
        self.xlabelval = "time" if use_color_alias or len(args) < 2 else str(args[1])
        self.xunits = "s" if use_color_alias or len(args) < 3 else str(args[2])
        self.yunits = "" if use_color_alias or len(args) < 4 else str(args[3])
        self.dataLabels = (
            ["lower", "upper"]
            if use_color_alias or len(args) < 5 or args[4] is None
            else self._normalize_text_sequence(args[4])
        )
        self.plotProps = (
            []
            if use_color_alias or len(args) < 6 or args[5] is None
            else self._normalize_plot_props(args[5])
        )
        self.time = t
        self.bounds = b
        self.data = self.bounds
        self.dimension = int(self.bounds.shape[1])
        self.minTime = float(self.time[0]) if self.time.size else float("nan")
        self.maxTime = float(self.time[-1]) if self.time.size else float("nan")
        self.sampleRate = (
            float(1.0 / np.mean(np.diff(self.time)))
            if self.time.size > 1 and np.all(np.diff(self.time) != 0)
            else float("nan")
        )
        self.dataMask = np.ones(self.dimension, dtype=int)
        self.color = str(args[0]) if use_color_alias else ("b" if color is None else str(color))
        self.value = float(value)
        if len(self.plotProps) == 1 and self.dimension > 1:
            self.plotProps = self.plotProps * self.dimension

    @staticmethod
    def _normalize_text_sequence(values) -> list[str]:
        if isinstance(values, (str, bytes)):
            return [str(values)]
        arr = np.asarray(values, dtype=object)
        if arr.shape == ():
            return [str(arr.item())]
        return [str(item) for item in arr.reshape(-1)]

    @staticmethod
    def _normalize_plot_props(values) -> list:
        if isinstance(values, (str, bytes)):
            return [str(values)]
        arr = np.asarray(values, dtype=object)
        if arr.shape == ():
            return [arr.item()]
        return [item.item() if isinstance(item, np.generic) else item for item in arr.reshape(-1)]

    @property
    def lower(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def upper(self) -> np.ndarray:
        return self.bounds[:, 1]

    def setColor(self, color: str) -> None:
        """Set the plot colour."""
        self.color = str(color)

    def setValue(self, value: float) -> None:
        """Set the confidence level (e.g. 0.95 for 95 %)."""
        self.value = float(value)

    def dataToMatrix(self) -> np.ndarray:
        """Return the bounds as an (n, 2) numpy array."""
        return np.asarray(self.bounds, dtype=float)

    def dataToStructure(self) -> dict:
        """Serialise to a plain dictionary (matches Matlab ``dataToStructure``)."""
        return {
            "time": self.time.tolist(),
            "signals": {"values": self.bounds.tolist(), "dimensions": self.dimension},
            "name": self.name,
            "dimension": self.dimension,
            "minTime": self.minTime,
            "maxTime": self.maxTime,
            "xlabelval": self.xlabelval,
            "xunits": self.xunits,
            "yunits": self.yunits,
            "dataLabels": list(self.dataLabels),
            "dataMask": self.dataMask.tolist(),
            "sampleRate": self.sampleRate,
            "plotProps": list(self.plotProps),
        }

    def _coerce_signal_values(self, other) -> np.ndarray:
        if hasattr(other, "time") and hasattr(other, "data"):
            other_time = np.asarray(other.time, dtype=float).reshape(-1)
            if other_time.shape != self.time.shape or np.max(np.abs(other_time - self.time)) > 1e-9:
                raise ValueError("ConfidenceInterval operations require matching time grids")
            values = np.asarray(other.data, dtype=float)
            if values.ndim == 2:
                if values.shape[1] != 1:
                    raise ValueError("ConfidenceInterval arithmetic expects a scalar signal per operation")
                values = values[:, 0]
            return values.reshape(-1)
        values = np.asarray(other, dtype=float)
        if values.ndim == 0:
            return np.full(self.time.shape, float(values), dtype=float)
        return values.reshape(-1)

    def __add__(self, other):
        if isinstance(other, ConfidenceInterval):
            if other.time.shape != self.time.shape or np.max(np.abs(other.time - self.time)) > 1e-9:
                raise ValueError("ConfidenceInterval operations require matching time grids")
            bounds = np.column_stack([self.lower + other.lower, self.upper + other.upper])
            return ConfidenceInterval(self.time, bounds, self.color)
        offset = self._coerce_signal_values(other)
        return ConfidenceInterval(self.time, self.bounds + offset[:, None], self.color)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ConfidenceInterval):
            if other.time.shape != self.time.shape or np.max(np.abs(other.time - self.time)) > 1e-9:
                raise ValueError("ConfidenceInterval operations require matching time grids")
            bounds = np.column_stack([self.lower - other.upper, self.upper - other.lower])
            return ConfidenceInterval(self.time, bounds, self.color)
        offset = self._coerce_signal_values(other)
        return ConfidenceInterval(self.time, self.bounds - offset[:, None], self.color)

    def __rsub__(self, other):
        offset = self._coerce_signal_values(other)
        bounds = np.column_stack([offset - self.upper, offset - self.lower])
        return ConfidenceInterval(self.time, bounds, self.color)

    def __neg__(self):
        return ConfidenceInterval(self.time, np.column_stack([-self.upper, -self.lower]), self.color)

    # ------------------------------------------------------------------
    # SignalObj-compatible interface methods
    # In MATLAB, ConfidenceInterval < SignalObj, inheriting ~70 methods.
    # We provide the most commonly used ones here for API parity.
    # ------------------------------------------------------------------

    def getSigInTimeWindow(self, wMin: float, wMax: float, holdVals: int = 0):
        """Return a new ConfidenceInterval restricted to [wMin, wMax]."""
        mask = (self.time >= wMin) & (self.time <= wMax)
        return ConfidenceInterval(self.time[mask], self.bounds[mask], self.color, value=self.value)

    def windowedSignal(self, windowTimes):
        """Return a ConfidenceInterval restricted to the given time window."""
        wt = np.asarray(windowTimes, dtype=float).reshape(-1)
        return self.getSigInTimeWindow(float(wt[0]), float(wt[-1]))

    def shift(self, deltaT: float, updateLabels: bool = False):
        """Return a time-shifted copy."""
        return ConfidenceInterval(self.time + deltaT, self.bounds.copy(), self.color, value=self.value)

    def resample(self, sample_rate: float):
        """Resample the CI bounds to a new sample rate via linear interpolation."""
        new_time = np.arange(self.minTime, self.maxTime, 1.0 / sample_rate)
        new_lower = np.interp(new_time, self.time, self.lower)
        new_upper = np.interp(new_time, self.time, self.upper)
        return ConfidenceInterval(new_time, np.column_stack([new_lower, new_upper]), self.color, value=self.value)

    @property
    def derivative(self):
        """Numerical derivative of both bounds."""
        dt = np.diff(self.time)
        dt[dt == 0] = 1.0
        d_lower = np.concatenate([np.diff(self.lower) / dt, [0.0]])
        d_upper = np.concatenate([np.diff(self.upper) / dt, [0.0]])
        return ConfidenceInterval(self.time, np.column_stack([d_lower, d_upper]), self.color, value=self.value)

    def merge(self, other, holdVals: int = 0):
        """Merge with another ConfidenceInterval along the time axis."""
        new_time = np.concatenate([self.time, other.time])
        new_bounds = np.vstack([self.bounds, other.bounds])
        order = np.argsort(new_time)
        return ConfidenceInterval(new_time[order], new_bounds[order], self.color, value=self.value)

    def copySignal(self):
        """Return a deep copy."""
        return ConfidenceInterval(self.time.copy(), self.bounds.copy(), self.color, value=self.value)

    def getSubSignal(self, identifier):
        """Return a single column (lower=0 or upper=1) as a 1-column CI."""
        idx = int(identifier)
        col = self.bounds[:, idx : idx + 1]
        return ConfidenceInterval(self.time, np.column_stack([col, col]), self.color, value=self.value)

    def setMinTime(self, minTime=None, holdVals: int = 0):
        """Restrict to times >= minTime."""
        if minTime is None:
            return
        mask = self.time >= minTime
        self.time = self.time[mask]
        self.bounds = self.bounds[mask]
        self.minTime = float(self.time[0]) if self.time.size else float("nan")

    def setMaxTime(self, maxTime=None, holdVals: int = 0):
        """Restrict to times <= maxTime."""
        if maxTime is None:
            return
        mask = self.time <= maxTime
        self.time = self.time[mask]
        self.bounds = self.bounds[mask]
        self.maxTime = float(self.time[-1]) if self.time.size else float("nan")

    def toStructure(self) -> dict:
        """Alias for :meth:`dataToStructure`."""
        return self.dataToStructure()

    @staticmethod
    def fromStructure(structure: dict) -> "ConfidenceInterval":
        """Reconstruct a ConfidenceInterval from a dictionary."""
        signals = structure.get("signals", {})
        values = signals.get("values", structure.get("data"))
        ci = ConfidenceInterval(
            structure["time"],
            values,
            structure.get("name", ""),
            structure.get("xlabelval", "time"),
            structure.get("xunits", "s"),
            structure.get("yunits", ""),
            structure.get("dataLabels"),
            structure.get("plotProps"),
        )
        if "dataMask" in structure:
            ci.dataMask = np.asarray(structure["dataMask"], dtype=int).reshape(-1)
        if "sampleRate" in structure:
            ci.sampleRate = float(structure["sampleRate"])
        if "minTime" in structure:
            ci.minTime = float(structure["minTime"])
        if "maxTime" in structure:
            ci.maxTime = float(structure["maxTime"])
        return ci

    def plot(self, color: str | None = None, alphaVal: float = 0.2, drawPatches: int = 0, ax=None):
        """Plot the confidence interval.

        Parameters
        ----------
        color : str or None
            Override colour (default: ``self.color``).
        alphaVal : float, default 0.2
            Transparency for shaded patches.
        drawPatches : int, default 0
            If ``1``, draw a shaded ``fill_between`` region instead of
            lines.
        ax : Axes or None
            Matplotlib axes.  If ``None``, uses ``plt.gca()``.

        Returns
        -------
        PolyCollection or list of Line2D
            Plot handles.
        """
        import matplotlib.pyplot as plt

        axis = plt.gca() if ax is None else ax
        plot_color = self.color if color is None else color
        if drawPatches:
            return axis.fill_between(self.time, self.lower, self.upper, color=plot_color, edgecolor="none", alpha=alphaVal)
        lines = axis.plot(self.time, self.bounds)
        for line in lines:
            if plot_color is not None and not isinstance(plot_color, (str, bytes)):
                line.set_color(plot_color)
        if plot_color is None or isinstance(plot_color, (str, bytes)):
            for index, line in enumerate(lines):
                line.set_color(MATLAB_COLOR_ORDER[index % MATLAB_COLOR_ORDER.shape[0]])
        return lines
