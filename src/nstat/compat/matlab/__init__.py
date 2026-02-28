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
