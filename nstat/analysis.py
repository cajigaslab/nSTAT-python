from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2, norm

from .SignalObj import SignalObj
from .fit import FitResult, _SingleFit, _matlab_compute_ks_arrays
from .glm import fit_binomial_glm, fit_poisson_glm
from .signal import Covariate
from .trial import ConfigCollection, SpikeTrainCollection, Trial


def psth(spike_trains: Sequence[object], bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be 1D and length >= 2")

    counts = np.zeros(edges.size - 1, dtype=float)
    if len(spike_trains) == 0:
        return counts.copy(), counts

    for tr in spike_trains:
        spikes = np.asarray(getattr(tr, "spikeTimes"), dtype=float).reshape(-1)
        c, _ = np.histogram(spikes, bins=edges)
        counts += c

    widths = np.diff(edges)
    mean_rate_hz = counts / (len(spike_trains) * widths)
    return mean_rate_hz, counts


def _as_neuron_indices(trial: Trial, neuron_selector) -> list[int]:
    if isinstance(neuron_selector, str):
        return [int(idx) for idx in trial.getNeuronIndFromName(neuron_selector)]
    if isinstance(neuron_selector, (int, np.integer, float, np.floating)):
        return [int(neuron_selector)]
    if isinstance(neuron_selector, Sequence) and not isinstance(neuron_selector, (bytes, bytearray)):
        out: list[int] = []
        for item in neuron_selector:
            out.extend(_as_neuron_indices(trial, item))
        return out
    raise TypeError("neuron selector must be a MATLAB-style one-based index, name, or sequence of either")


def _restore_trial_partition(trial: Trial, original_partition: np.ndarray, original_window: np.ndarray | None = None) -> None:
    trial.restoreToOriginal()
    if original_partition.size:
        trial.setTrialPartition(original_partition)
        if original_window is None or original_window.size != 2:
            trial.setTrialTimesFor("training")
            return
        training = original_partition[:2] if original_partition.size >= 2 else None
        validation = original_partition[2:4] if original_partition.size >= 4 else None
        if training is not None and training.size == 2 and np.allclose(original_window, training, rtol=0.0, atol=1e-12):
            trial.setTrialTimesFor("training")
        elif validation is not None and validation.size == 2 and np.allclose(original_window, validation, rtol=0.0, atol=1e-12):
            trial.setTrialTimesFor("validation")
        else:
            trial.setMinTime(float(original_window[0]))
            trial.setMaxTime(float(original_window[1]))


def _time_rescaled_z(counts: np.ndarray, lam_per_bin: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(counts, dtype=float).reshape(-1)
    lam = np.asarray(lam_per_bin, dtype=float).reshape(-1)
    if y_arr.shape != lam.shape:
        raise ValueError("counts and lam_per_bin must have matching shapes")
    z_values: list[float] = []
    accum = 0.0
    for count, lam_i in zip(y_arr, lam, strict=False):
        accum += float(max(lam_i, 1e-12))
        if count >= 1.0:
            repeats = max(int(round(count)), 1)
            for _ in range(repeats):
                z_values.append(accum)
                accum = 0.0
    return np.asarray(z_values, dtype=float)


def _fit_lambda_matrix_to_covariate(lambda_time: np.ndarray, lambda_columns: list[np.ndarray], lambda_index: int) -> Covariate:
    data = np.column_stack([np.asarray(col, dtype=float).reshape(-1) for col in lambda_columns]) if lambda_columns else np.zeros((lambda_time.size, 0), dtype=float)
    data_labels = [f"\\lambda_{{{idx}}}" for idx in range(1, data.shape[1] + 1)]
    return Covariate(
        lambda_time,
        data,
        "\\lambda(t)",
        "time",
        "s",
        "Hz",
        data_labels if data_labels else [f"\\lambda_{{{lambda_index}}}"],
    )


def _glm_deviance(y: np.ndarray, mean_counts: np.ndarray, distribution: str) -> float:
    observed = np.asarray(y, dtype=float).reshape(-1)
    expected = np.clip(np.asarray(mean_counts, dtype=float).reshape(-1), 1e-12, None)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected counts must have matching shapes")

    dist = str(distribution).lower()
    if dist == "poisson":
        ratio = np.ones_like(observed)
        positive = observed > 0.0
        ratio[positive] = observed[positive] / expected[positive]
        return float(2.0 * np.sum(observed * np.log(ratio) - (observed - expected)))

    if dist == "binomial":
        prob = np.clip(expected, 1e-12, 1.0 - 1e-12)
        return float(
            2.0
            * np.sum(
                observed * np.log(np.clip(observed / prob, 1e-12, None))
                + (1.0 - observed) * np.log(np.clip((1.0 - observed) / (1.0 - prob), 1e-12, None))
            )
        )

    raise ValueError(f"Unsupported GLM distribution for deviance: {distribution}")


def _benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(p_values, dtype=float).reshape(-1)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, p.size + 1, dtype=float) / float(p.size))
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros(p.size, dtype=bool)
    cutoff = np.max(np.flatnonzero(passed))
    keep = np.zeros(p.size, dtype=bool)
    keep[order[: cutoff + 1]] = True
    return keep


class Analysis:
    """Canonical analysis entry points preserving MATLAB-facing workflow semantics."""

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    @staticmethod
    def _collapse_spike_input(nspikeObj):
        if isinstance(nspikeObj, SpikeTrainCollection):
            return nspikeObj.toSpikeTrain()
        if isinstance(nspikeObj, Sequence) and not hasattr(nspikeObj, "spikeTimes"):
            if len(nspikeObj) == 0:
                raise ValueError("Spike input sequence must not be empty")
            if any(not hasattr(item, "spikeTimes") for item in nspikeObj):
                raise ValueError("Python Analysis expects sequences of MATLAB-style nspikeTrain objects")
            if len(nspikeObj) == 1:
                return nspikeObj[0]
            return SpikeTrainCollection(list(nspikeObj)).toSpikeTrain()
        return nspikeObj

    @staticmethod
    def psth(spike_trains: Sequence[object], bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return psth(spike_trains, bin_edges)

    @staticmethod
    def GLMFit(
        tObj: Trial,
        neuronNumber,
        lambdaIndex: int,
        Algorithm: str = "GLM",
        *,
        l2: float = 0.0,
        max_iter: int = 120,
    ):
        algorithm = str(Algorithm or "GLM").upper()
        if algorithm not in {"GLM", "BNLRCG"}:
            raise ValueError("Algorithm not supported!")

        indices = _as_neuron_indices(tObj, neuronNumber)
        if not indices:
            raise ValueError("No neurons matched the MATLAB-style selector")

        binary_rep = all(tObj.nspikeColl.getNST(idx).isSigRepBinary() for idx in indices)
        if algorithm == "BNLRCG" and not binary_rep:
            raise ValueError("To use BNLRCG Algorithm, spikeTrain must have a binary representation. Increase sampleRate and try again")

        stacked_y: list[np.ndarray] = []
        stacked_x: list[np.ndarray] = []
        lambda_segments: list[np.ndarray] = []
        lambda_time_segments: list[np.ndarray] = []
        time_offset = 0.0

        for index in indices:
            x = np.asarray(tObj.getDesignMatrix(index), dtype=float)
            lambda_time = np.asarray(tObj.getCov(1).time, dtype=float).reshape(-1)
            sample_rate = float(tObj.sampleRate)
            dt = 1.0 / max(sample_rate, 1e-12)
            bin_edges = np.concatenate([lambda_time, [lambda_time[-1] + dt]])
            y = np.asarray(tObj.nspikeColl.getNST(index).to_binned_counts(bin_edges), dtype=float).reshape(-1)

            n_obs = min(x.shape[0], y.shape[0], lambda_time.shape[0])
            x = x[:n_obs, :]
            y = y[:n_obs]
            lambda_time = lambda_time[:n_obs]

            stacked_x.append(x)
            stacked_y.append(y)
            lambda_time_segments.append(lambda_time + time_offset)
            time_offset = float(lambda_time[-1] + dt) if lambda_time.size else time_offset

        X = np.vstack(stacked_x) if stacked_x else np.zeros((0, 0), dtype=float)
        y = np.concatenate(stacked_y) if stacked_y else np.array([], dtype=float)
        lambda_time_full = np.concatenate(lambda_time_segments) if lambda_time_segments else np.array([], dtype=float)
        sample_rate = float(tObj.sampleRate)

        if algorithm == "BNLRCG":
            glm_res = fit_binomial_glm(X, y, include_intercept=False, l2=l2, max_iter=max_iter)
            lambda_delta = np.clip(glm_res.predict_probability(X), 1e-12, 1.0 - 1e-9)
            rate_hz = lambda_delta * sample_rate
            distribution = "binomial"
            b = np.asarray(glm_res.coefficients, dtype=float).reshape(-1)
            dev = _glm_deviance(y, lambda_delta, distribution)
        else:
            glm_res = fit_poisson_glm(X, y, include_intercept=False, l2=l2, max_iter=max_iter)
            lambda_delta = glm_res.predict_rate(X)
            rate_hz = lambda_delta * sample_rate
            distribution = "poisson"
            b = np.asarray(glm_res.coefficients, dtype=float).reshape(-1)
            dev = _glm_deviance(y, lambda_delta, distribution)

        # MATLAB stores logLL using the legacy per-bin convention
        # `sum(y.*log(data*delta) + (1-y).*(1-data*delta))` for both GLM branches.
        matlab_bin_mass = np.maximum(rate_hz / max(sample_rate, 1e-12), 1e-12)
        logLL = float(np.sum(y * np.log(matlab_bin_mass) + (1.0 - y) * (1.0 - matlab_bin_mass)))

        n_params = int(b.size)
        AIC = float(2.0 * n_params + dev)
        BIC = float(np.log(max(y.shape[0], 1)) * n_params + dev)

        start = 0
        for x_seg in stacked_x:
            stop = start + x_seg.shape[0]
            lambda_segments.append(rate_hz[start:stop])
            start = stop

        lambda_sig = _fit_lambda_matrix_to_covariate(lambda_time_full, lambda_segments, int(lambdaIndex))
        stats = {
            "intercept": float(glm_res.intercept),
            "n_iter": int(glm_res.n_iter),
            "converged": bool(glm_res.converged),
        }
        return lambda_sig, b, dev, stats, AIC, BIC, logLL, distribution

    @staticmethod
    def run_analysis_for_neuron(
        trial: Trial,
        neuron_index: int,
        config_collection: ConfigCollection,
        *,
        algorithm: str = "GLM",
        l2: float = 0.0,
        max_iter: int = 120,
    ) -> FitResult:
        if neuron_index < 0:
            raise IndexError("neuron_index must be >= 0")

        original_partition = np.asarray(trial.getTrialPartition(), dtype=float).reshape(-1)
        original_window = np.asarray([trial.minTime, trial.maxTime], dtype=float).reshape(-1)
        neuron_number = int(neuron_index) + 1
        labels: list[list[str]] = []
        lambda_parts: list[Covariate] = []
        b: list[np.ndarray] = []
        dev: list[float] = []
        stats: list[dict[str, float | int | bool]] = []
        AIC: list[float] = []
        BIC: list[float] = []
        logLL: list[float] = []
        numHist: list[int] = []
        histObjects: list[object] = []
        ensHistObjects: list[object] = []
        fits: list[_SingleFit] = []
        xvalData: list[np.ndarray] = []
        xvalTime: list[np.ndarray] = []
        distributions: list[str] = []

        spike_train = trial.nspikeColl.getNST(neuron_number).nstCopy()
        if not spike_train.name:
            spike_train.setName(str(neuron_number))

        for cfg_index in range(1, config_collection.numConfigs + 1):
            trial.restoreToOriginal()
            if original_partition.size:
                trial.setTrialPartition(original_partition)
                trial.setTrialTimesFor("training")
            config_collection.setConfig(trial, cfg_index)

            current_labels = trial.getLabelsFromMask(neuron_number)
            labels.append(list(current_labels))
            numHist.append(len(trial.getHistLabels()))
            histObjects.append(trial.history)
            ensHistObjects.append(trial.ensCovHist)

            lambda_signal, coeff, deviance, stat_dict, aic, bic, log_likelihood, distribution = Analysis.GLMFit(
                trial,
                neuron_number,
                cfg_index,
                algorithm,
                l2=l2,
                max_iter=max_iter,
            )

            fit_name = config_collection.getConfigNames([cfg_index])[0]
            lambda_signal.setDataLabels([fit_name])
            lambda_parts.append(lambda_signal)
            b.append(np.asarray(coeff, dtype=float).reshape(-1))
            dev.append(float(deviance))
            stats.append(stat_dict)
            AIC.append(float(aic))
            BIC.append(float(bic))
            logLL.append(float(log_likelihood))
            distributions.append(str(distribution))
            fits.append(
                _SingleFit(
                    name=fit_name,
                    coefficients=np.asarray(coeff, dtype=float).reshape(-1),
                    intercept=0.0,
                    log_likelihood=float(log_likelihood),
                    aic=float(aic),
                    bic=float(bic),
                    stats=stat_dict,
                )
            )

            partition = np.asarray(trial.getTrialPartition(), dtype=float).reshape(-1)
            if partition.size >= 4 and partition[2] < partition[3]:
                trial.setTrialTimesFor("validation")
                xvalData.append(np.asarray(trial.getDesignMatrix(neuron_number), dtype=float))
                xvalTime.append(np.asarray(trial.covarColl.getCov(1).time, dtype=float).copy())
                trial.setTrialTimesFor("training")
            else:
                xvalData.append(np.zeros((0, len(current_labels)), dtype=float))
                xvalTime.append(np.array([], dtype=float))

        merged_lambda = lambda_parts[0]
        for part in lambda_parts[1:]:
            merged_lambda = merged_lambda.merge(part)

        _restore_trial_partition(trial, original_partition, original_window)
        fit_result = FitResult(
            spike_train,
            labels,
            numHist,
            histObjects,
            ensHistObjects,
            merged_lambda,
            b,
            dev,
            stats,
            AIC,
            BIC,
            logLL,
            config_collection,
            xvalData,
            xvalTime,
            distributions,
            fits=fits,
        )
        # MATLAB returns fits with KS diagnostics already populated, and
        # downstream summary classes read those cached fields directly.
        fit_result.computeKSStats()
        return fit_result

    @staticmethod
    def run_analysis_for_all_neurons(
        trial: Trial,
        config_collection: ConfigCollection,
        *,
        algorithm: str = "GLM",
        l2: float = 0.0,
        max_iter: int = 120,
    ) -> list[FitResult]:
        out: list[FitResult] = []
        for i in range(trial.spike_collection.num_spike_trains):
            out.append(
                Analysis.run_analysis_for_neuron(
                    trial,
                    i,
                    config_collection,
                    algorithm=algorithm,
                    l2=l2,
                    max_iter=max_iter,
                )
            )
        return out

    @staticmethod
    def RunAnalysisForNeuron(tObj: Trial, neuronNumber, configColl: ConfigCollection, makePlot=1, Algorithm="GLM", DTCorrection=1, batchMode=0):
        del DTCorrection, batchMode
        indices = _as_neuron_indices(tObj, neuronNumber)
        fits = [Analysis.run_analysis_for_neuron(tObj, idx - 1, configColl, algorithm=Algorithm) for idx in indices]
        if makePlot and len(fits) == 1:
            fits[0].plotResults()
        return fits[0] if len(fits) == 1 else fits

    @staticmethod
    def RunAnalysisForAllNeurons(tObj: Trial, configs: ConfigCollection, makePlot=1, Algorithm="GLM", DTCorrection=1, batchMode=0):
        del DTCorrection, batchMode
        fits = Analysis.run_analysis_for_all_neurons(tObj, configs, algorithm=Algorithm)
        if makePlot and len(fits) == 1:
            fits[0].plotResults()
        return fits[0] if len(fits) == 1 else fits

    @staticmethod
    def computeKSStats(nspikeObj, lambdaInput: Covariate, DTCorrection: int = 1, *, random_values=None):
        nspikeObj = Analysis._collapse_spike_input(nspikeObj)
        return _matlab_compute_ks_arrays(nspikeObj, lambdaInput, dt_correction=DTCorrection, random_values=random_values)

    @staticmethod
    def computeInvGausTrans(Z):
        z = np.asarray(Z, dtype=float)
        if z.ndim == 1:
            z = z[:, None]
        U = 1.0 - np.exp(-z)
        U = np.clip(U, 1e-6, 1.0 - 1e-6)
        X = norm.ppf(U)
        if X.shape[0] <= 1:
            lags = np.asarray([], dtype=float)
            rho = np.zeros((0, X.shape[1]), dtype=float)
            conf = np.zeros((0, 2), dtype=float)
        else:
            lags = np.arange(1, X.shape[0], dtype=float)
            rho = np.zeros((lags.size, X.shape[1]), dtype=float)
            for col in range(X.shape[1]):
                centered = X[:, col] - np.mean(X[:, col])
                corr = np.correlate(centered, centered, mode="full")
                corr = corr[corr.size // 2 :]
                if corr[0] != 0.0:
                    corr = corr / corr[0]
                rho[:, col] = corr[1 : lags.size + 1]
            conf_bound = 1.96 / np.sqrt(float(X.shape[0]))
            conf = np.column_stack([np.full(lags.size, conf_bound), np.full(lags.size, -conf_bound)])
        rhoSig = SignalObj(lags, rho, "ACF[ \\Phi^-1(u_i) ]", "Lag \\Delta \\tau", "sec")
        confBoundSig = SignalObj(lags, conf, "ACF[ \\Phi^-1(u_i) ]", "\\Delta \\tau", "sec")
        return X, rhoSig, confBoundSig

    @staticmethod
    def computeFitResidual(nspikeObj, lambdaInput: Covariate, windowSize: float = 0.01):
        nspikeObj = Analysis._collapse_spike_input(nspikeObj)

        nCopy = nspikeObj.nstCopy()
        nCopy.resample(lambdaInput.sampleRate)
        nCopy.setMinTime(lambdaInput.minTime)
        nCopy.setMaxTime(lambdaInput.maxTime)

        # MATLAB's static Analysis.computeFitResidual ultimately operates on
        # the resampled spike-count grid, even when a finer windowSize is
        # requested. Preserve that canonical helper behavior here.
        sumSpikes = nCopy.getSigRep(1.0 / float(nCopy.sampleRate), float(nCopy.minTime), float(nCopy.maxTime))
        windowTimes = np.linspace(float(nCopy.minTime), float(nCopy.maxTime), sumSpikes.time.size, dtype=float)
        if np.isfinite(windowSize) and windowSize > 0:
            origin = float(nCopy.minTime)
            windowTimes = origin + np.round((windowTimes - origin) / float(windowSize)) * float(windowSize)
            windowTimes = np.round(windowTimes, decimals=12)
        lambdaInt = lambdaInput.integral()
        lambdaIntVals = (
            lambdaInt.getValueAt(windowTimes[1:]).reshape(-1, lambdaInt.dimension)
            - lambdaInt.getValueAt(windowTimes[:-1]).reshape(-1, lambdaInt.dimension)
        )
        spike_window_data = np.asarray(sumSpikes.data, dtype=float)
        if lambdaIntVals.shape[0] == spike_window_data.shape[0]:
            sumSpikesOverWindow = spike_window_data
        else:
            sumSpikesOverWindow = spike_window_data[1:, :]
        mdata = np.asarray(sumSpikesOverWindow, dtype=float) - np.asarray(lambdaIntVals, dtype=float)
        out = np.vstack([np.zeros((1, mdata.shape[1]), dtype=float), mdata])
        return Covariate(
            windowTimes,
            out,
            "M(t_k)",
            lambdaInt.xlabelval,
            lambdaInt.xunits,
            lambdaInt.yunits,
            list(lambdaInput.dataLabels),
        )

    @staticmethod
    def KSPlot(fitResults: FitResult, DTCorrection: int = 1, makePlot: int = 1):
        del DTCorrection
        fitResults.computeKSStats()
        return fitResults.KSPlot() if makePlot else []

    @staticmethod
    def plotFitResidual(fitResults: FitResult, windowSize: float = 0.01, makePlot: int = 1):
        fitResults.computeFitResidual(window_size=windowSize)
        return fitResults.plotResidual() if makePlot else []

    @staticmethod
    def plotInvGausTrans(fitResults: FitResult, makePlot: int = 0):
        fitResults.computeInvGausTrans()
        return fitResults.plotInvGausTrans() if makePlot else []

    @staticmethod
    def plotSeqCorr(fitResults: FitResult):
        fitResults.computeInvGausTrans()
        return fitResults.plotSeqCorr()

    @staticmethod
    def plotCoeffs(fitResults: FitResult):
        return fitResults.plotCoeffs()

    @staticmethod
    def computeHistLag(tObj: Trial, neuronNum=None, windowTimes=None, CovLabels=None, Algorithm="GLM", batchMode=0, sampleRate=None, makePlot=1, histMinTimes=None, histMaxTimes=None):
        del batchMode, histMinTimes, histMaxTimes
        if windowTimes is None:
            raise ValueError("Must specify a vector of windowTimes")
        if neuronNum is None:
            neuronNum = tObj.getNeuronIndFromMask()
        if sampleRate is None:
            sampleRate = tObj.sampleRate
        cov_labels = [] if CovLabels is None else CovLabels
        windows = np.asarray(windowTimes, dtype=float).reshape(-1)
        if windows.size < 2:
            raise ValueError("windowTimes must contain at least two entries")

        configs = []
        from .trial import TrialConfig

        configs.append(TrialConfig(cov_labels, sampleRate, [], [], name="Baseline"))
        for i in range(2, windows.size + 1):
            cfg = TrialConfig(cov_labels, sampleRate, windows[:i], [], name=f"Window{i - 1}")
            configs.append(cfg)
        tcc = ConfigCollection(configs)
        fitResults = Analysis.RunAnalysisForNeuron(tObj, neuronNum, tcc, makePlot, Algorithm)
        return fitResults, tcc

    @staticmethod
    def computeHistLagForAll(tObj: Trial, windowTimes, CovLabels=None, Algorithm="GLM", batchMode=0, sampleRate=None, makePlot=1, histMinTimes=None, histMaxTimes=None):
        results = []
        for neuron_idx in tObj.getNeuronIndFromMask():
            fit, _ = Analysis.computeHistLag(
                tObj,
                neuron_idx,
                windowTimes,
                CovLabels,
                Algorithm,
                batchMode,
                sampleRate,
                makePlot,
                histMinTimes,
                histMaxTimes,
            )
            results.append(fit)
        return results

    @staticmethod
    def compHistEnsCoeff(tObj: Trial, history, neuronNum=None, neighbors=None, ensembleCov=None, makePlot=1):
        from .trial import TrialConfig

        neuron_index = _as_neuron_indices(tObj, neuronNum if neuronNum is not None else tObj.getNeuronIndFromMask()[0])[0]
        if neighbors is None or (isinstance(neighbors, Sequence) and not neighbors):
            neighbors = tObj.getNeuronNeighbors(neuron_index)
        if ensembleCov is None:
            ensembleCov = tObj.getEnsembleNeuronCovariates(neuron_index, neighbors, history)

        ensemble_trial = Trial(tObj.nspikeColl, ensembleCov)
        tc = TrialConfig("all", ensemble_trial.sampleRate, [], [], [], [], name="EnsembleHistory")
        tcc = ConfigCollection(tc)
        fitResults = Analysis.RunAnalysisForNeuron(ensemble_trial, neuron_index, tcc, makePlot)
        return fitResults, ensembleCov, tcc

    @staticmethod
    def compHistEnsCoeffForAll(tObj: Trial, history, makePlot=1):
        neuron_indices = tObj.getNeuronIndFromMask()
        if not neuron_indices:
            return [], None, []
        fit_results = []
        config_collections = []
        ensemble_cov = None
        for neuron_index in neuron_indices:
            fit, ensemble_cov_current, tcc = Analysis.compHistEnsCoeff(
                tObj,
                history,
                neuron_index,
                tObj.getNeuronNeighbors(neuron_index),
                None,
                makePlot,
            )
            fit_results.append(fit)
            config_collections.append(tcc)
            if ensemble_cov is None:
                ensemble_cov = ensemble_cov_current
        return fit_results, ensemble_cov, config_collections

    @staticmethod
    def computeGrangerCausalityMatrix(tObj: Trial, Algorithm="GLM", confidenceInterval=0.95, batchMode=0):
        del batchMode
        neuron_indices = tObj.getNeuronIndFromMask()
        n_neurons = tObj.nspikeColl.numSpikeTrains
        gammaMat = np.zeros((n_neurons, n_neurons), dtype=float)
        phiMat = np.zeros_like(gammaMat)
        devianceMat = np.zeros_like(gammaMat)
        sigMat = np.zeros_like(gammaMat, dtype=int)
        fitResults: list[list[FitResult]] = [[] for _ in neuron_indices]

        ens_hist = tObj.ensCovHist if tObj.isEnsCovHistSet() else tObj.history
        if ens_hist is None or (isinstance(ens_hist, np.ndarray) and ens_hist.size == 0) or (
            isinstance(ens_hist, Sequence) and not isinstance(ens_hist, (str, bytes, np.ndarray)) and len(ens_hist) == 0
        ):
            raise ValueError("Trial must define history or ensemble-history before computing Granger causality")

        cov_mask = tObj.covMask
        sample_rate = tObj.sampleRate
        ens_mask = np.asarray(tObj.ensCovMask, dtype=int) if np.asarray(tObj.ensCovMask).size else (
            np.ones((n_neurons, n_neurons), dtype=int) - np.eye(n_neurons, dtype=int)
        )

        from .trial import TrialConfig

        p_vals: list[float] = []
        p_coords: list[tuple[int, int]] = []
        alpha = 1.0 - float(confidenceInterval)

        for target_offset, neuron_index in enumerate(neuron_indices):
            baseline_cfg = TrialConfig(cov_mask, sample_rate, tObj.history, ens_hist, ens_mask, name="Baseline")
            neighbors = np.flatnonzero(ens_mask[:, neuron_index - 1] == 1) + 1
            for neighbor in neighbors:
                reduced_mask = ens_mask.copy()
                reduced_mask[neighbor - 1, neuron_index - 1] = 0
                excluded_cfg = TrialConfig(
                    cov_mask,
                    sample_rate,
                    tObj.history,
                    ens_hist,
                    reduced_mask,
                    name=f"{neighbor}excluded from {neuron_index}",
                )
                fit = Analysis.RunAnalysisForNeuron(tObj, neuron_index, ConfigCollection([baseline_cfg, excluded_cfg]), 0, Algorithm)
                fitResults[target_offset].append(fit)
                gamma = float(np.asarray(fit.logLL, dtype=float)[1] - np.asarray(fit.logLL, dtype=float)[0])
                gammaMat[neighbor - 1, neuron_index - 1] = gamma
                deviance = float(max(-2.0 * gamma, 0.0))
                devianceMat[neighbor - 1, neuron_index - 1] = deviance
                dim_diff = max(int(abs(np.diff(np.asarray(fit.numCoeffs, dtype=int))[0])), 1)
                p_val = float(chi2.sf(deviance, dim_diff))
                p_vals.append(p_val)
                p_coords.append((neighbor - 1, neuron_index - 1))
                coeffs = fit.getHistCoeffs(2) if np.any(np.asarray(fit.numHist, dtype=int) > 0) else np.array([], dtype=float)
                if coeffs.size:
                    phiMat[neighbor - 1, neuron_index - 1] = -float(np.sign(np.sum(coeffs))) * gamma

        if p_vals:
            keep = _benjamini_hochberg(np.asarray(p_vals, dtype=float), alpha=max(alpha, 1e-6))
            for include, (row, col) in zip(keep, p_coords, strict=False):
                sigMat[row, col] = int(include)

        return fitResults, gammaMat, phiMat, devianceMat, sigMat

    @staticmethod
    def computeNeighbors(tObj: Trial, neuronNum=None, sampleRate=None, windowTimes=None, makePlot=1):
        if windowTimes is None:
            raise ValueError("Must specify a vector of windowTimes")
        neuron_index = _as_neuron_indices(tObj, neuronNum if neuronNum is not None else tObj.getNeuronIndFromMask()[0])[0]
        if sampleRate is None:
            sampleRate = tObj.sampleRate

        windows = np.asarray(windowTimes, dtype=float).reshape(-1)
        if windows.size < 2:
            raise ValueError("windowTimes must contain at least two entries")

        from .trial import TrialConfig

        neighbor_mask = np.zeros((tObj.nspikeColl.numSpikeTrains, tObj.nspikeColl.numSpikeTrains), dtype=int)
        neighbors = np.asarray(tObj.getNeuronNeighbors(neuron_index), dtype=int).reshape(-1)
        if neighbors.size:
            neighbor_mask[neighbors - 1, neuron_index - 1] = 1

        configs = [TrialConfig([], sampleRate, [], [], [], [], name="Baseline")]
        for i in range(2, windows.size + 1):
            configs.append(TrialConfig([], sampleRate, [], windows[:i], neighbor_mask, [], name=f"Window{i - 1}"))
        tcc = ConfigCollection(configs)
        fitResults = Analysis.RunAnalysisForNeuron(tObj, neuron_index, tcc, makePlot)
        return fitResults, tcc

    @staticmethod
    def spikeTrigAvg(tObj: Trial, neuronNum, windowSize):
        from .trial import CovariateCollection

        train = tObj.getNeuron(neuronNum).nstCopy()
        spike_times = np.asarray(train.getSpikeTimes(), dtype=float).reshape(-1)
        time_axis = np.arange(-float(windowSize) / 2.0, float(windowSize) / 2.0 + 1.0 / float(tObj.sampleRate), 1.0 / float(tObj.sampleRate))
        covariates = []
        for cov_index in range(1, tObj.covarColl.numCov + 1):
            cov = tObj.getCov(cov_index)
            if spike_times.size == 0:
                samples = np.zeros((time_axis.size, 0, cov.dimension), dtype=float)
            else:
                sampled = [cov.getValueAt(spike_time + time_axis) for spike_time in spike_times]
                samples = np.stack(sampled, axis=1)
            for dim_index, label in enumerate(cov.dataLabels, start=1):
                data = samples[:, :, dim_index - 1] if samples.size else np.zeros((time_axis.size, 0), dtype=float)
                covariates.append(
                    Covariate(
                        time_axis,
                        data,
                        label,
                        cov.xlabelval,
                        cov.xunits,
                        cov.yunits,
                        [f"{label}_spike_{idx}" for idx in range(1, data.shape[1] + 1)] or [label],
                    )
                )
        return CovariateCollection(covariates)


RunAnalysisForNeuron = Analysis.RunAnalysisForNeuron
RunAnalysisForAllNeurons = Analysis.RunAnalysisForAllNeurons
GLMFit = Analysis.GLMFit
KSPlot = Analysis.KSPlot
plotFitResidual = Analysis.plotFitResidual
computeFitResidual = Analysis.computeFitResidual
computeKSStats = Analysis.computeKSStats
plotInvGausTrans = Analysis.plotInvGausTrans
plotSeqCorr = Analysis.plotSeqCorr
plotCoeffs = Analysis.plotCoeffs
computeHistLag = Analysis.computeHistLag
computeHistLagForAll = Analysis.computeHistLagForAll
compHistEnsCoeff = Analysis.compHistEnsCoeff
compHistEnsCoeffForAll = Analysis.compHistEnsCoeffForAll
computeGrangerCausalityMatrix = Analysis.computeGrangerCausalityMatrix
computeNeighbors = Analysis.computeNeighbors
spikeTrigAvg = Analysis.spikeTrigAvg


__all__ = [
    "Analysis",
    "GLMFit",
    "KSPlot",
    "RunAnalysisForAllNeurons",
    "RunAnalysisForNeuron",
    "compHistEnsCoeff",
    "compHistEnsCoeffForAll",
    "computeFitResidual",
    "computeGrangerCausalityMatrix",
    "computeHistLag",
    "computeHistLagForAll",
    "computeKSStats",
    "computeNeighbors",
    "plotCoeffs",
    "plotFitResidual",
    "plotInvGausTrans",
    "plotSeqCorr",
    "psth",
    "spikeTrigAvg",
]
