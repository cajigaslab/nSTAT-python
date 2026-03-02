"""Signal and covariate data containers.

These classes provide explicit, typed wrappers around time-indexed arrays.
The implementation intentionally keeps validation strict because many
nSTAT algorithms rely on aligned, monotonic time grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import lfilter as scipy_lfilter
from scipy.signal import filtfilt as scipy_filtfilt


ArrayLike = np.ndarray


def _safe_zero_phase_filter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply filtfilt with a conservative padlen fallback for short vectors."""

    b_arr = np.asarray(b, dtype=float).reshape(-1)
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    if x_arr.size < 2:
        return scipy_lfilter(b_arr, a_arr, x_arr)

    ntaps = max(int(a_arr.size), int(b_arr.size))
    padlen = min(3 * ntaps, int(x_arr.size) - 1)
    try:
        return scipy_filtfilt(b_arr, a_arr, x_arr, padlen=padlen)
    except ValueError:
        # Fallback for pathological short-signal/filter combinations.
        fwd = scipy_lfilter(b_arr, a_arr, x_arr)
        bwd = scipy_lfilter(b_arr, a_arr, fwd[::-1])
        return bwd[::-1]


@dataclass(slots=True)
class Signal:
    """Continuous signal sampled on a 1D time grid.

    Parameters
    ----------
    time:
        Strictly increasing time samples (seconds by convention).
    data:
        Signal values. Can be 1D (`n_time`) or 2D (`n_time`, `n_channels`).
    name:
        Human-readable signal name.
    units:
        Optional unit label.

    Notes
    -----
    MATLAB `SignalObj` supports broad plotting and metadata utilities.
    This base class keeps the core numerical contract minimal and explicit,
    then downstream classes layer workflow-specific behavior.
    """

    time: ArrayLike
    data: ArrayLike
    name: str = "signal"
    units: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    x_units: str | None = None
    y_units: str | None = None
    plot_props: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.data = np.asarray(self.data, dtype=float)

        if self.time.ndim != 1:
            raise ValueError("time must be a 1D array")
        if self.time.size < 2:
            raise ValueError("time must contain at least two samples")

        dtime = np.diff(self.time)
        if np.any(dtime <= 0.0):
            raise ValueError("time must be strictly increasing")

        if self.data.ndim == 1:
            if self.data.shape[0] != self.time.size:
                raise ValueError("1D data length must match time length")
        elif self.data.ndim == 2:
            if self.data.shape[0] != self.time.size:
                raise ValueError("2D data first dimension must match time length")
        else:
            raise ValueError("data must be 1D or 2D")

    @property
    def n_samples(self) -> int:
        """Number of time samples."""

        return int(self.time.size)

    @property
    def n_channels(self) -> int:
        """Number of channels (1 for a vector signal)."""

        if self.data.ndim == 1:
            return 1
        return int(self.data.shape[1])

    @property
    def sample_rate_hz(self) -> float:
        """Estimated sample rate in Hertz using median time delta."""

        dt = float(np.median(np.diff(self.time)))
        return 1.0 / dt

    @property
    def duration_s(self) -> float:
        """Signal duration in seconds."""

        return float(self.time[-1] - self.time[0])

    def copy(self) -> "Signal":
        """Return a deep copy."""

        return Signal(
            time=self.time.copy(),
            data=self.data.copy(),
            name=self.name,
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def copy_signal(self) -> "Signal":
        """MATLAB-style alias for :meth:`copy`."""

        return self.copy()

    def set_name(self, name: str) -> "Signal":
        self.name = name
        return self

    def set_xlabel(self, label: str) -> "Signal":
        self.x_label = label
        return self

    def set_ylabel(self, label: str) -> "Signal":
        self.y_label = label
        return self

    def set_units(self, units: str) -> "Signal":
        self.units = units
        return self

    def set_x_units(self, units: str) -> "Signal":
        self.x_units = units
        return self

    def set_y_units(self, units: str) -> "Signal":
        self.y_units = units
        return self

    def set_plot_props(self, props: dict[str, Any]) -> "Signal":
        self.plot_props = dict(props)
        return self

    def clear_plot_props(self) -> "Signal":
        self.plot_props = {}
        return self

    def set_min_time(self, min_time: float) -> "Signal":
        mask = self.time >= float(min_time)
        if not np.any(mask):
            raise ValueError("set_min_time removed all samples")
        self.time = self.time[mask]
        if self.data.ndim == 1:
            self.data = self.data[mask]
        else:
            self.data = self.data[mask, :]
        return self

    def set_max_time(self, max_time: float) -> "Signal":
        mask = self.time <= float(max_time)
        if not np.any(mask):
            raise ValueError("set_max_time removed all samples")
        self.time = self.time[mask]
        if self.data.ndim == 1:
            self.data = self.data[mask]
        else:
            self.data = self.data[mask, :]
        return self

    def restrict_to_time_window(self, min_time: float, max_time: float) -> "Signal":
        return self.set_min_time(min_time).set_max_time(max_time)

    def shift_time(self, offset_s: float) -> "Signal":
        self.time = self.time + float(offset_s)
        return self

    def align_time(self, new_zero_time: float = 0.0) -> "Signal":
        """Shift time so that current first sample equals ``new_zero_time``."""

        offset = float(new_zero_time) - float(self.time[0])
        return self.shift_time(offset)

    def derivative(self) -> "Signal":
        """Numerical first derivative over time."""

        grad = np.gradient(self.data, self.time, axis=0)
        return Signal(
            time=self.time.copy(),
            data=grad,
            name=f"d/dt {self.name}",
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def integral(self) -> np.ndarray:
        """Cumulative trapezoidal integral of each channel over time."""

        dt = np.diff(self.time)
        if self.data.ndim == 1:
            y = self.data
            acc = np.zeros_like(y)
            acc[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
            return acc
        y = self.data
        acc = np.zeros_like(y)
        acc[1:, :] = np.cumsum(0.5 * (y[:-1, :] + y[1:, :]) * dt[:, None], axis=0)
        return acc

    def data_to_matrix(self) -> np.ndarray:
        """Return data in 2D matrix form `(n_time, n_channels)`."""

        if self.data.ndim == 1:
            return self.data[:, None]
        return self.data.copy()

    def get_sub_signal(self, selector: int | list[int] | np.ndarray) -> "Signal":
        """Select one or more channels by index."""

        mat = self.data_to_matrix()
        selected = mat[:, selector]
        if selected.ndim == 2 and selected.shape[1] == 1:
            selected = selected[:, 0]
        return Signal(
            time=self.time.copy(),
            data=selected,
            name=self.name,
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def merge(self, other: "Signal") -> "Signal":
        """Merge channels from two signals that share the same time grid."""

        if self.time.shape != other.time.shape or not np.allclose(self.time, other.time):
            raise ValueError("Signals must share identical time grids to merge")
        merged = np.hstack([self.data_to_matrix(), other.data_to_matrix()])
        return Signal(
            time=self.time.copy(),
            data=merged,
            name=f"{self.name}+{other.name}",
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def resample(self, sample_rate_hz: float) -> "Signal":
        """Resample signal by linear interpolation to a new sample rate."""

        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        dt = 1.0 / float(sample_rate_hz)
        t_new = np.arange(self.time[0], self.time[-1] + 0.5 * dt, dt)
        mat = self.data_to_matrix()
        y_new = np.column_stack([np.interp(t_new, self.time, mat[:, i]) for i in range(mat.shape[1])])
        if y_new.shape[1] == 1:
            y_out: np.ndarray = y_new[:, 0]
        else:
            y_out = y_new
        return Signal(
            time=t_new,
            data=y_out,
            name=self.name,
            units=self.units,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )


@dataclass(slots=True)
class Covariate(Signal):
    """Named design covariate with optional per-column labels.

    Parameters
    ----------
    labels:
        Optional labels for each covariate column. For vector covariates,
        one label is expected.

    Notes
    -----
    This class is intentionally lightweight; higher-level composition and
    selection live in :mod:`nstat.trial`.
    """

    labels: list[str] = field(default_factory=list)
    conf_interval: Any | None = None

    def __post_init__(self) -> None:
        Signal.__post_init__(self)

        if not self.labels:
            if self.n_channels == 1:
                self.labels = [self.name]
            else:
                self.labels = [f"{self.name}_{i}" for i in range(self.n_channels)]

        if len(self.labels) != self.n_channels:
            raise ValueError("labels length must match number of channels")

    def get_labels(self) -> list[str]:
        return list(self.labels)

    def is_conf_interval_set(self) -> bool:
        return self.conf_interval is not None

    def set_conf_interval(self, interval: Any) -> "Covariate":
        self.conf_interval = interval
        return self

    def get_sub_signal(self, selector: Any) -> "Covariate":
        """Return selected covariate channels by index or label."""

        if isinstance(selector, str):
            signal_selector: int | list[int] | np.ndarray = [self.labels.index(selector)]
        elif isinstance(selector, list) and selector and isinstance(selector[0], str):
            signal_selector = [self.labels.index(str(item)) for item in selector]
        else:
            signal_selector = selector

        sub = Signal.get_sub_signal(self, signal_selector)
        sub_mat = sub.data_to_matrix()
        selector_idx = np.asarray(np.atleast_1d(signal_selector), dtype=int)
        sub_labels = [self.labels[i] for i in selector_idx]
        if sub_mat.shape[1] == 1:
            sub_data: np.ndarray = sub_mat[:, 0]
        else:
            sub_data = sub_mat
        return Covariate(
            time=sub.time,
            data=sub_data,
            name=self.name,
            units=self.units,
            labels=sub_labels,
            conf_interval=self.conf_interval,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def compute_mean_plus_ci(self, axis: int = 1, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean and Gaussian CI for multi-channel/trial covariate data."""

        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0, 1)")
        mat = self.data_to_matrix()
        mu = np.mean(mat, axis=axis)
        std = np.std(mat, axis=axis, ddof=1 if mat.shape[axis] > 1 else 0)
        n = mat.shape[axis]
        z = 1.959963984540054 if np.isclose(level, 0.95) else 1.0
        half = z * std / max(np.sqrt(n), 1.0)
        return mu, mu - half, mu + half

    def filtfilt(self, b: np.ndarray, a: np.ndarray) -> "Covariate":
        """Zero-phase filter each covariate channel."""

        mat = self.data_to_matrix()
        filtered = np.column_stack([_safe_zero_phase_filter(b, a, mat[:, i]) for i in range(mat.shape[1])])
        if filtered.shape[1] == 1:
            out_data: np.ndarray = filtered[:, 0]
        else:
            out_data = filtered
        return Covariate(
            time=self.time.copy(),
            data=out_data,
            name=self.name,
            units=self.units,
            labels=list(self.labels),
            conf_interval=self.conf_interval,
            x_label=self.x_label,
            y_label=self.y_label,
            x_units=self.x_units,
            y_units=self.y_units,
            plot_props=dict(self.plot_props),
        )

    def to_structure(self) -> dict[str, Any]:
        """Serialize to plain python structure."""

        return {
            "time": self.time.copy(),
            "data": self.data.copy(),
            "name": self.name,
            "units": self.units,
            "labels": list(self.labels),
        }

    @staticmethod
    def from_structure(payload: dict[str, Any]) -> "Covariate":
        """Deserialize from :meth:`to_structure` payload."""

        return Covariate(
            time=np.asarray(payload["time"], dtype=float),
            data=np.asarray(payload["data"], dtype=float),
            name=str(payload.get("name", "covariate")),
            units=payload.get("units"),
            labels=[str(item) for item in payload.get("labels", [])],
        )
