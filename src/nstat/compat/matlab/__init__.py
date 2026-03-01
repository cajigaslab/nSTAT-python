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


class Covariate(_Covariate):
    def computeMeanPlusCI(self, axis: int = 1, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.compute_mean_plus_ci(axis=axis, level=level)

    def getSubSignal(self, selector: int | str | list[int] | list[str]) -> "Covariate":
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

    def setConfInterval(self, interval: Any) -> "Covariate":
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
    def fromStructure(payload: dict[str, Any]) -> "Covariate":
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


class ConfidenceInterval(_ConfidenceInterval):
    def getWidth(self) -> np.ndarray:
        return self.width()


class Events(_Events):
    def getTimes(self) -> np.ndarray:
        return self.times


class History(_HistoryBasis):
    def getNumBins(self) -> int:
        return self.n_bins

    def getDesignMatrix(self, spike_times_s: np.ndarray, time_grid_s: np.ndarray) -> np.ndarray:
        return self.design_matrix(spike_times_s=spike_times_s, time_grid_s=time_grid_s)


class nspikeTrain(_SpikeTrain):
    def getSpikeTimes(self) -> np.ndarray:
        return self.spike_times

    def getDuration(self) -> float:
        return self.duration_s()

    def getFiringRate(self) -> float:
        return self.firing_rate_hz()

    def shiftTime(self, offset_s: float) -> "nspikeTrain":
        self.shift_time(offset_s)
        return self

    def setMinTime(self, t_min: float) -> "nspikeTrain":
        self.set_min_time(t_min)
        return self

    def setMaxTime(self, t_max: float) -> "nspikeTrain":
        self.set_max_time(t_max)
        return self


class nstColl(_SpikeTrainCollection):
    def getNumUnits(self) -> int:
        return self.n_units

    def getBinnedMatrix(
        self, binSize_s: float, mode: Literal["binary", "count"] = "binary"
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.to_binned_matrix(bin_size_s=binSize_s, mode=mode)

    def merge(self, other: _SpikeTrainCollection) -> "nstColl":
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

    def addToColl(self, train: _SpikeTrain) -> "nstColl":
        self.add_to_coll(train)
        return self

    def addSingleSpikeToColl(self, unitInd: int, spikeTime: float) -> "nstColl":
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

    def shiftTime(self, offset_s: float) -> "nstColl":
        self.shift_time(offset_s)
        return self

    def setMinTime(self, t_min: float) -> "nstColl":
        self.set_min_time(t_min)
        return self

    def setMaxTime(self, t_max: float) -> "nstColl":
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
    def fromStructure(payload: dict[str, Any]) -> "nstColl":
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

    def updateTimes(self) -> "nstColl":
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


class CovColl(_CovariateCollection):
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

    def updateTimes(self) -> "CovColl":
        _ = self.time
        return self

    def toStructure(self) -> dict[str, Any]:
        return {"covariates": [cov.to_structure() for cov in self.covariates]}

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
    def getConfigs(self) -> list[_TrialConfig]:
        return self.configs


class Trial(_Trial):
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

    def toStructure(self) -> dict[str, np.ndarray | float | str]:
        return self.to_structure()

    @staticmethod
    def fromStructure(payload: dict[str, np.ndarray | float | str]) -> "CIF":
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
        trains = cast(
            list[_SpikeTrain],
            [
            nspikeTrain(spike_times=sp, t_start=float(time[0]), t_end=float(time[-1]), name=f"unit_{i+1}")
            for i, sp in enumerate(spikes)
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


class FitResult(_FitResult):
    @staticmethod
    def FitResult(structure: dict[str, Any]) -> _FitResult:
        return _FitResult.from_structure(structure)

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> _FitResult:
        return _FitResult.from_structure(structure)

    @staticmethod
    def CellArrayToStructure(results: list[_FitResult]) -> list[dict[str, Any]]:
        return _FitResult.cell_array_to_structure(results)

    def toStructure(self) -> dict[str, Any]:
        return self.to_structure()

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
