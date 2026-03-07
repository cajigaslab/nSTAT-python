from __future__ import annotations

from typing import Sequence

import numpy as np

from .fit import FitResult, _SingleFit
from .glm import fit_poisson_glm
from .signal import Covariate
from .trial import ConfigCollection, Trial


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


class Analysis:
    """Canonical analysis entry points preserving MATLAB-facing workflow semantics."""

    @staticmethod
    def psth(spike_trains: Sequence[object], bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return psth(spike_trains, bin_edges)

    @staticmethod
    def run_analysis_for_neuron(
        trial: Trial,
        neuron_index: int,
        config_collection: ConfigCollection,
        *,
        l2: float = 1e-6,
        max_iter: int = 120,
    ) -> FitResult:
        if neuron_index < 0:
            raise IndexError("neuron_index must be >= 0")

        original_partition = trial.getTrialPartition().copy()
        trial.restoreToOriginal()
        if original_partition.size:
            trial.setTrialPartition(original_partition)
            trial.setTrialTimesFor("training")

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
            X = trial.getDesignMatrix(neuron_number)
            time = trial.covarColl.getCov(1).time
            dt = float(np.median(np.diff(time))) if time.shape[0] > 1 else max(1.0 / trial.sampleRate, 1e-12)
            edges = np.concatenate([time, [time[-1] + dt]])
            y = trial.nspikeColl.getNST(neuron_number).to_binned_counts(edges)
            offset = np.full(y.shape[0], np.log(max(dt, 1e-12)), dtype=float)

            glm_res = fit_poisson_glm(X, y, offset=offset, l2=l2, max_iter=max_iter)
            n_params = X.shape[1] + 1
            aic = float(2.0 * n_params - 2.0 * glm_res.log_likelihood)
            bic = float(np.log(max(y.shape[0], 1)) * n_params - 2.0 * glm_res.log_likelihood)
            fit_name = config_collection.getConfigNames([cfg_index])[0]
            coeff = np.concatenate([[glm_res.intercept], np.asarray(glm_res.coefficients, dtype=float).reshape(-1)])

            rate = glm_res.predict_rate(X, offset=offset)
            lambda_signal = Covariate(
                time,
                rate,
                fit_name if fit_name else f"lambda_{cfg_index}",
                "time",
                "s",
                "spikes/sec",
                [fit_name if fit_name else f"lambda_{cfg_index}"],
            )

            labels.append(list(current_labels))
            lambda_parts.append(lambda_signal)
            b.append(coeff)
            dev.append(float(-2.0 * glm_res.log_likelihood))
            stats.append(
                {
                    "intercept": float(glm_res.intercept),
                    "n_iter": int(glm_res.n_iter),
                    "converged": bool(glm_res.converged),
                }
            )
            AIC.append(aic)
            BIC.append(bic)
            logLL.append(float(glm_res.log_likelihood))
            numHist.append(len(trial.getHistLabels()))
            histObjects.append(trial.history)
            ensHistObjects.append(trial.ensCovHist)
            fits.append(
                _SingleFit(
                    name=fit_name,
                    coefficients=np.asarray(glm_res.coefficients, dtype=float),
                    intercept=float(glm_res.intercept),
                    log_likelihood=float(glm_res.log_likelihood),
                    aic=aic,
                    bic=bic,
                    stats=stats[-1],
                )
            )
            distributions.append("poisson")

            partition = trial.getTrialPartition()
            if partition.size >= 4 and partition[2] < partition[3]:
                trial.setTrialTimesFor("validation")
                xvalData.append(trial.getDesignMatrix(neuron_number))
                xvalTime.append(trial.covarColl.getCov(1).time.copy())
                trial.setTrialTimesFor("training")
            else:
                xvalData.append(np.zeros((0, X.shape[1]), dtype=float))
                xvalTime.append(np.array([], dtype=float))

        merged_lambda = lambda_parts[0]
        for part in lambda_parts[1:]:
            merged_lambda = merged_lambda.merge(part)

        trial.restoreToOriginal()
        if original_partition.size:
            trial.setTrialPartition(original_partition)
            trial.setTrialTimesFor("training")

        return FitResult(
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

    @staticmethod
    def run_analysis_for_all_neurons(
        trial: Trial,
        config_collection: ConfigCollection,
        *,
        l2: float = 1e-6,
        max_iter: int = 120,
    ) -> list[FitResult]:
        out: list[FitResult] = []
        for i in range(trial.spike_collection.num_spike_trains):
            out.append(
                Analysis.run_analysis_for_neuron(
                    trial,
                    i,
                    config_collection,
                    l2=l2,
                    max_iter=max_iter,
                )
            )
        return out

    @staticmethod
    def RunAnalysisForNeuron(tObj: Trial, neuronNumber: int, configColl: ConfigCollection, *_):
        return Analysis.run_analysis_for_neuron(tObj, neuronNumber - 1, configColl)

    @staticmethod
    def RunAnalysisForAllNeurons(tObj: Trial, configs: ConfigCollection, *_):
        return Analysis.run_analysis_for_all_neurons(tObj, configs)


__all__ = ["Analysis", "psth"]
