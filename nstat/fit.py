from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from .core import Covariate, nspikeTrain


def _ordered_unique(labels: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(label) for label in labels))


def _parse_neuron_number(spike_obj: nspikeTrain | Sequence[nspikeTrain]) -> str | float:
    if isinstance(spike_obj, Sequence) and not isinstance(spike_obj, nspikeTrain):
        names = [str(item.name) for item in spike_obj if getattr(item, "name", "")]
        unique = _ordered_unique(names)
        return unique[0] if unique else ""
    name = str(getattr(spike_obj, "name", ""))
    if not name:
        return ""
    try:
        return float(name)
    except ValueError:
        return name


def _pad_rows(rows: Sequence[np.ndarray], fill_value: float = np.nan) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=float)
    max_len = max(row.size for row in rows)
    out = np.full((len(rows), max_len), fill_value, dtype=float)
    for idx, row in enumerate(rows):
        out[idx, : row.size] = row
    return out


def _autocorrelation(values: np.ndarray, max_lag: int = 25) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(values, dtype=float).reshape(-1) - float(np.mean(values))
    if centered.size < 2 or float(np.var(centered)) <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr / corr[0]
    lags = np.arange(corr.shape[0], dtype=float)
    max_lag = int(min(max_lag, corr.shape[0] - 1))
    return lags[1 : max_lag + 1], corr[1 : max_lag + 1]


def _time_rescaled_uniforms(y: np.ndarray, lam_per_bin: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    lam = np.asarray(lam_per_bin, dtype=float).reshape(-1)
    if y_arr.shape != lam.shape:
        raise ValueError("y and lam_per_bin must have the same shape")
    if np.sum(y_arr) <= 1:
        return np.asarray([], dtype=float)

    uniforms: list[float] = []
    accum = 0.0
    for count, lam_i in zip(y_arr, lam, strict=False):
        accum += float(max(lam_i, 1e-12))
        if count >= 1.0:
            for _ in range(int(round(count))):
                uniforms.append(1.0 - np.exp(-accum))
                accum = 0.0
    return np.asarray(uniforms, dtype=float)


def _ks_curve(uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.sort(np.asarray(uniforms, dtype=float).reshape(-1))
    if u.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    ideal = np.linspace(1.0 / u.size, 1.0, u.size, dtype=float)
    ci = np.full(u.size, 1.36 / np.sqrt(float(u.size)), dtype=float)
    return ideal, u, ci


@dataclass
class _SingleFit:
    name: str
    coefficients: np.ndarray
    intercept: float
    log_likelihood: float
    aic: float
    bic: float
    stats: Any | None = None


class FitResult:
    """MATLAB-facing fit result container with Python compatibility aliases."""

    def __init__(self, neuralSpikeTrain: nspikeTrain | Sequence[nspikeTrain], *args, **kwargs) -> None:
        if args and isinstance(args[0], Covariate):
            self._init_simplified(neuralSpikeTrain, args[0], args[1] if len(args) > 1 else [])
            return

        covLabels = args[0] if len(args) > 0 else kwargs.get("covLabels", [])
        numHist = args[1] if len(args) > 1 else kwargs.get("numHist", [])
        histObjects = args[2] if len(args) > 2 else kwargs.get("histObjects", [])
        ensHistObj = args[3] if len(args) > 3 else kwargs.get("ensHistObj", [])
        lambda_signal = args[4] if len(args) > 4 else kwargs.get("lambda_signal")
        b = args[5] if len(args) > 5 else kwargs.get("b", [])
        dev = args[6] if len(args) > 6 else kwargs.get("dev", [])
        stats = args[7] if len(args) > 7 else kwargs.get("stats", [])
        AIC = args[8] if len(args) > 8 else kwargs.get("AIC", [])
        BIC = args[9] if len(args) > 9 else kwargs.get("BIC", [])
        logLL = args[10] if len(args) > 10 else kwargs.get("logLL", [])
        configColl = args[11] if len(args) > 11 else kwargs.get("configColl")
        XvalData = args[12] if len(args) > 12 else kwargs.get("XvalData", [])
        XvalTime = args[13] if len(args) > 13 else kwargs.get("XvalTime", [])
        distribution = args[14] if len(args) > 14 else kwargs.get("distribution", "poisson")
        fits = kwargs.get("fits")
        self._init_matlab_style(
            neuralSpikeTrain,
            covLabels,
            numHist,
            histObjects,
            ensHistObj,
            lambda_signal,
            b,
            dev,
            stats,
            AIC,
            BIC,
            logLL,
            configColl,
            XvalData,
            XvalTime,
            distribution,
            fits=fits,
        )

    def _init_common(self) -> None:
        self.Z = np.array([], dtype=float)
        self.U = np.array([], dtype=float)
        self.X = np.array([], dtype=float)
        self.Residual = None
        self._diagnostic_cache: dict[int, dict[str, np.ndarray | float]] = {}
        self.KSStats = np.zeros((self.numResults, 1), dtype=float)
        self.KSPvalues = np.full(self.numResults, np.nan, dtype=float)
        self.withinConfInt = np.zeros(self.numResults, dtype=float)
        self.invGausStats = {"rhoSig": [], "confBoundSig": []}
        self.plotParams = {
            "bAct": _pad_rows([np.asarray(coeffs, dtype=float).reshape(-1) for coeffs in self.b]).T if self.b else np.zeros((0, 0)),
            "seAct": np.zeros((len(self.uniqueCovLabels), self.numResults), dtype=float),
            "sigIndex": np.zeros((len(self.uniqueCovLabels), self.numResults), dtype=float),
            "xLabels": list(self.uniqueCovLabels),
            "numResultsCoeffPresent": np.sum(self.flatMask, axis=1) if self.flatMask.size else np.array([], dtype=int),
        }
        self.validation = None

    def _init_simplified(self, neuralSpikeTrain: nspikeTrain | Sequence[nspikeTrain], lambda_signal: Covariate, fits: Sequence[_SingleFit]) -> None:
        from .trial import ConfigCollection, TrialConfig

        self.neuralSpikeTrain = neuralSpikeTrain
        self.neuronNumber = _parse_neuron_number(neuralSpikeTrain)
        self.lambda_signal = lambda_signal
        self.lambda_ = lambda_signal
        self.numResults = len(list(fits))
        self.fits = list(fits)
        self.b = [np.concatenate([[fit.intercept], np.asarray(fit.coefficients, dtype=float).reshape(-1)]) for fit in self.fits]
        self.dev = np.zeros(self.numResults, dtype=float)
        self.AIC = np.asarray([fit.aic for fit in self.fits], dtype=float)
        self.BIC = np.asarray([fit.bic for fit in self.fits], dtype=float)
        self.logLL = np.asarray([fit.log_likelihood for fit in self.fits], dtype=float)
        self.stats = [fit.stats for fit in self.fits]
        self.configNames = [fit.name for fit in self.fits]
        self.configs = ConfigCollection([TrialConfig(name=name) for name in self.configNames])
        labels = list(lambda_signal.dataLabels) if getattr(lambda_signal, "dataLabels", None) else ["lambda"]
        self.covLabels = [labels[:] for _ in range(self.numResults)]
        self.uniqueCovLabels = _ordered_unique(labels)
        self.indicesToUniqueLabels = [list(range(1, len(labels) + 1)) for _ in range(self.numResults)]
        self.numHist = [0 for _ in range(self.numResults)]
        self.histObjects = [None for _ in range(self.numResults)]
        self.ensHistObjects = [None for _ in range(self.numResults)]
        self.fitType = ["poisson" for _ in range(self.numResults)]
        self.numCoeffs = np.asarray([coeff.shape[0] for coeff in self.b], dtype=int)
        self.flatMask = np.ones((len(self.uniqueCovLabels), self.numResults), dtype=int)
        self.XvalData = []
        self.XvalTime = []
        self.minTime = float(lambda_signal.minTime)
        self.maxTime = float(lambda_signal.maxTime)
        self._init_common()

    def _init_matlab_style(
        self,
        neuralSpikeTrain: nspikeTrain | Sequence[nspikeTrain],
        covLabels,
        numHist,
        histObjects,
        ensHistObj,
        lambda_signal: Covariate | None,
        b,
        dev,
        stats,
        AIC,
        BIC,
        logLL,
        configColl,
        XvalData,
        XvalTime,
        distribution,
        *,
        fits: Sequence[_SingleFit] | None = None,
    ) -> None:
        self.neuralSpikeTrain = neuralSpikeTrain
        self.neuronNumber = _parse_neuron_number(neuralSpikeTrain)
        self.lambda_signal = lambda_signal if lambda_signal is not None else Covariate([], [], "lambda")
        self.lambda_ = self.lambda_signal
        self.covLabels = [list(labels) for labels in covLabels]
        self.uniqueCovLabels = _ordered_unique([label for labels in self.covLabels for label in labels])
        self.indicesToUniqueLabels = []
        self.flatMask = np.zeros((len(self.uniqueCovLabels), max(len(self.covLabels), 1)), dtype=int)
        for fit_idx, labels in enumerate(self.covLabels):
            indices = [self.uniqueCovLabels.index(label) + 1 for label in labels]
            self.indicesToUniqueLabels.append(indices)
            if indices:
                self.flatMask[np.asarray(indices, dtype=int) - 1, fit_idx] = 1

        self.numHist = list(numHist)
        self.histObjects = list(histObjects)
        if ensHistObj is None or ensHistObj == []:
            self.ensHistObjects = [None for _ in range(len(self.covLabels))]
        elif isinstance(ensHistObj, Sequence) and not isinstance(ensHistObj, (str, bytes)):
            self.ensHistObjects = list(ensHistObj)
        else:
            self.ensHistObjects = [ensHistObj for _ in range(len(self.covLabels))]
        self.b = [np.asarray(coeff, dtype=float).reshape(-1) for coeff in b]
        self.dev = np.asarray(dev, dtype=float).reshape(-1)
        self.AIC = np.asarray(AIC, dtype=float).reshape(-1)
        self.BIC = np.asarray(BIC, dtype=float).reshape(-1)
        self.logLL = np.asarray(logLL, dtype=float).reshape(-1)
        self.stats = list(stats)
        self.configs = configColl
        self.configNames = configColl.getConfigNames() if configColl is not None else [f"Fit {i}" for i in range(1, len(self.b) + 1)]
        if isinstance(distribution, str):
            self.fitType = [distribution for _ in range(len(self.b))]
        else:
            self.fitType = list(distribution)
        self.numResults = len(self.b)
        self.numCoeffs = np.asarray([coeff.shape[0] for coeff in self.b], dtype=int)
        self.XvalData = list(XvalData) if isinstance(XvalData, Sequence) and not isinstance(XvalData, (str, bytes, np.ndarray)) else []
        self.XvalTime = list(XvalTime) if isinstance(XvalTime, Sequence) and not isinstance(XvalTime, (str, bytes, np.ndarray)) else []
        self.minTime = float(getattr(self.lambda_signal, "minTime", np.nan))
        self.maxTime = float(getattr(self.lambda_signal, "maxTime", np.nan))
        if fits is not None:
            self.fits = list(fits)
        else:
            self.fits = []
            for idx in range(self.numResults):
                coeff = self.b[idx]
                intercept = float(coeff[0]) if coeff.size else 0.0
                beta = coeff[1:] if coeff.size > 1 else np.array([], dtype=float)
                self.fits.append(
                    _SingleFit(
                        name=self.configNames[idx],
                        coefficients=beta,
                        intercept=intercept,
                        log_likelihood=float(self.logLL[idx]),
                        aic=float(self.AIC[idx]),
                        bic=float(self.BIC[idx]),
                        stats=self.stats[idx] if idx < len(self.stats) else None,
                    )
                )
        self._init_common()

    @property
    def lambdaSignal(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambda_sig(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambdaCov(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambdaObj(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambda_data(self) -> np.ndarray:
        return np.asarray(self.lambda_signal.data, dtype=float)

    @property
    def lambda_values(self) -> np.ndarray:
        return np.asarray(self.lambda_signal.data, dtype=float)

    @property
    def lambda_time(self) -> np.ndarray:
        return np.asarray(self.lambda_signal.time, dtype=float)

    @property
    def lambda_rate(self) -> np.ndarray:
        return np.asarray(self.lambda_signal.data, dtype=float)

    def getCoeffs(self, fit_num: int = 1) -> np.ndarray:
        return self.b[fit_num - 1].copy()

    def getHistCoeffs(self, fit_num: int = 1) -> np.ndarray:
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        coeff = self.getCoeffs(fit_num)
        if num_hist <= 0:
            return np.array([], dtype=float)
        return coeff[-num_hist:]

    def mergeResults(self, other: "FitResult") -> "FitResult":
        from .trial import ConfigCollection

        if isinstance(self.lambda_signal, Covariate) and isinstance(other.lambda_signal, Covariate):
            lambda_signal = self.lambda_signal.merge(other.lambda_signal)
        else:
            lambda_signal = self.lambda_signal
        configs = ConfigCollection(
            [*(self.configs.configArray if self.configs is not None else []), *(other.configs.configArray if other.configs is not None else [])]
        )
        return FitResult(
            self.neuralSpikeTrain,
            [*self.covLabels, *other.covLabels],
            [*self.numHist, *other.numHist],
            [*self.histObjects, *other.histObjects],
            [*self.ensHistObjects, *other.ensHistObjects],
            lambda_signal,
            [*self.b, *other.b],
            np.concatenate([self.dev, other.dev]),
            [*self.stats, *other.stats],
            np.concatenate([self.AIC, other.AIC]),
            np.concatenate([self.BIC, other.BIC]),
            np.concatenate([self.logLL, other.logLL]),
            configs,
            [*self.XvalData, *other.XvalData],
            [*self.XvalTime, *other.XvalTime],
            [*self.fitType, *other.fitType],
            fits=[*self.fits, *other.fits],
        )

    def _lambda_series(self, fit_num: int = 1) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(self.lambda_signal.time, dtype=float).reshape(-1)
        data = np.asarray(self.lambda_signal.data, dtype=float)
        if data.ndim == 1:
            rate = data.reshape(-1)
        else:
            idx = min(max(fit_num - 1, 0), data.shape[1] - 1)
            rate = data[:, idx].reshape(-1)
        if time.shape[0] != rate.shape[0]:
            raise ValueError("lambda signal time and data lengths do not match")
        return time, rate

    def _primary_spike_train(self) -> nspikeTrain:
        if isinstance(self.neuralSpikeTrain, nspikeTrain):
            return self.neuralSpikeTrain
        if isinstance(self.neuralSpikeTrain, Sequence) and self.neuralSpikeTrain:
            return self.neuralSpikeTrain[0]
        raise TypeError("FitResult does not contain a MATLAB-style neural spike train")

    def _compute_diagnostics(self, fit_num: int = 1) -> dict[str, np.ndarray | float]:
        if fit_num in self._diagnostic_cache:
            return self._diagnostic_cache[fit_num]

        time, rate_hz = self._lambda_series(fit_num)
        dt = float(np.median(np.diff(time))) if time.size > 1 else 1.0
        edges = np.concatenate([time, [time[-1] + dt]])
        counts = self._primary_spike_train().to_binned_counts(edges)
        lam_per_bin = rate_hz * dt
        residual = counts - lam_per_bin
        residual_std = residual / np.sqrt(np.maximum(lam_per_bin, 1e-12))
        uniforms = _time_rescaled_uniforms(counts, lam_per_bin)
        ideal, empirical, ci = _ks_curve(uniforms)
        ks_stat = float(np.max(np.abs(empirical - ideal))) if ideal.size else 0.0
        ks_pvalue = float(kstest(uniforms, "uniform").pvalue) if uniforms.size else np.nan
        within = float(np.mean(np.abs(empirical - ideal) <= ci)) if ideal.size else np.nan
        lags, acf = _autocorrelation(uniforms, max_lag=25)
        acf_ci = 1.96 / np.sqrt(float(uniforms.size)) if uniforms.size else np.nan
        gauss = np.clip(uniforms, 1e-6, 1.0 - 1e-6)
        coeffs = self.getCoeffs(fit_num)
        coeff_labels = ["Intercept", *self.covLabels[fit_num - 1]] if fit_num - 1 < len(self.covLabels) else ["Intercept"]
        diagnostics: dict[str, np.ndarray | float] = {
            "time": time,
            "rate_hz": rate_hz,
            "counts": counts,
            "lambda_per_bin": lam_per_bin,
            "residual": residual,
            "residual_std": residual_std,
            "uniforms": uniforms,
            "ks_ideal": ideal,
            "ks_empirical": empirical,
            "ks_ci": ci,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "within_conf_int": within,
            "acf_lags": lags,
            "acf_values": acf,
            "acf_ci": acf_ci,
            "gaussianized": gauss,
            "coefficients": coeffs,
            "coeff_labels": np.asarray(coeff_labels, dtype=object),
        }
        self._diagnostic_cache[fit_num] = diagnostics
        self.KSStats[fit_num - 1, 0] = ks_stat
        self.KSPvalues[fit_num - 1] = ks_pvalue
        self.withinConfInt[fit_num - 1] = within
        self.U = uniforms
        self.Z = gauss
        self.X = time
        self.Residual = {
            "time": time,
            "residual": residual,
            "standardized": residual_std,
        }
        self.invGausStats = {"rhoSig": acf.tolist(), "confBoundSig": [acf_ci]}
        return diagnostics

    def computeKSStats(self, fit_num: int = 1) -> dict[str, float]:
        diag = self._compute_diagnostics(fit_num)
        return {
            "ks_stat": float(diag["ks_stat"]),
            "ks_pvalue": float(diag["ks_pvalue"]),
            "within_conf_int": float(diag["within_conf_int"]),
        }

    def computeInvGausTrans(self, fit_num: int = 1) -> np.ndarray:
        return np.asarray(self._compute_diagnostics(fit_num)["uniforms"], dtype=float)

    def computeFitResidual(self, fit_num: int = 1) -> Covariate:
        diag = self._compute_diagnostics(fit_num)
        return Covariate(
            np.asarray(diag["time"], dtype=float),
            np.asarray(diag["residual"], dtype=float),
            "fit residual",
            "time",
            "s",
            "counts/bin",
            ["residual"],
        )

    def plotResults(self, fit_num: int = 1, handle=None):
        fig = handle if handle is not None else plt.figure(figsize=(11.5, 8.0))
        fig.clear()
        axes = fig.subplots(2, 2)
        self.KSPlot(fit_num=fit_num, handle=axes[0, 0])
        self.plotInvGausTrans(fit_num=fit_num, handle=axes[0, 1])
        self.plotSeqCorr(fit_num=fit_num, handle=axes[1, 0])
        self.plotCoeffs(fit_num=fit_num, handle=axes[1, 1])
        fig.tight_layout()
        return fig

    def KSPlot(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 4.0))[1]
        ideal = np.asarray(diag["ks_ideal"], dtype=float)
        empirical = np.asarray(diag["ks_empirical"], dtype=float)
        ci = np.asarray(diag["ks_ci"], dtype=float)
        if ideal.size:
            ax.plot(ideal, empirical, color="tab:blue", linewidth=1.5)
            ax.plot([0.0, 1.0], [0.0, 1.0], color="0.3", linewidth=1.0, linestyle="--")
            ax.plot(ideal, np.clip(ideal + ci, 0.0, 1.0), color="tab:red", linewidth=1.0)
            ax.plot(ideal, np.clip(ideal - ci, 0.0, 1.0), color="tab:red", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Ideal Uniform CDF")
        ax.set_ylabel("Empirical CDF")
        ax.set_title("KS Plot")
        return ax

    def plotResidual(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        ax.plot(np.asarray(diag["time"], dtype=float), np.asarray(diag["residual"], dtype=float), color="tab:purple", linewidth=1.0)
        ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("count residual")
        ax.set_title("Fit Residual")
        return ax

    def plotInvGausTrans(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        u = np.asarray(diag["uniforms"], dtype=float)
        if u.size:
            ax.plot(np.arange(1, u.size + 1), u, color="tab:green", linewidth=1.0)
            ax.axhline(0.5, color="0.4", linewidth=1.0, linestyle="--")
        ax.set_xlabel("event index")
        ax.set_ylabel("time-rescaled transform")
        ax.set_title("Inverse-Gaussian/Uniform Transform")
        return ax

    def plotSeqCorr(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        lags = np.asarray(diag["acf_lags"], dtype=float)
        acf = np.asarray(diag["acf_values"], dtype=float)
        if lags.size:
            ax.vlines(lags, 0.0, acf, color="tab:orange", linewidth=1.4)
            ax.axhline(float(diag["acf_ci"]), color="tab:red", linewidth=1.0)
            ax.axhline(-float(diag["acf_ci"]), color="tab:red", linewidth=1.0)
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_xlabel("lag")
        ax.set_ylabel("autocorrelation")
        ax.set_title("Sequential Correlation of Rescaled ISIs")
        return ax

    def plotCoeffs(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        coeffs = np.asarray(diag["coefficients"], dtype=float)
        labels = list(np.asarray(diag["coeff_labels"], dtype=object))
        xpos = np.arange(coeffs.size, dtype=float)
        ax.axhline(0.0, color="0.6", linewidth=1.0)
        ax.plot(xpos, coeffs, "o-", color="tab:blue", linewidth=1.0)
        ax.set_xticks(xpos, labels, rotation=45, ha="right")
        ax.set_ylabel("coefficient value")
        ax.set_title("GLM Coefficients")
        return ax

    @property
    def lambda_obj(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambda_model(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambda_result(self) -> Covariate:
        return self.lambda_signal

    def toStructure(self) -> dict[str, Any]:
        return {
            "covLabels": [list(labels) for labels in self.covLabels],
            "numHist": list(self.numHist),
            "lambda_time": self.lambda_signal.time.tolist(),
            "lambda_data": self.lambda_signal.data.tolist(),
            "lambda_name": self.lambda_signal.name,
            "b": [coeff.tolist() for coeff in self.b],
            "dev": self.dev.tolist(),
            "AIC": self.AIC.tolist(),
            "BIC": self.BIC.tolist(),
            "logLL": self.logLL.tolist(),
            "configNames": list(self.configNames),
            "fitType": list(self.fitType),
            "neural_spike_times": (
                self.neuralSpikeTrain.spikeTimes.tolist()
                if isinstance(self.neuralSpikeTrain, nspikeTrain)
                else [train.spikeTimes.tolist() for train in self.neuralSpikeTrain]
            ),
            "neural_name": (
                self.neuralSpikeTrain.name
                if isinstance(self.neuralSpikeTrain, nspikeTrain)
                else [train.name for train in self.neuralSpikeTrain]
            ),
            "neural_min_time": (
                self.neuralSpikeTrain.minTime
                if isinstance(self.neuralSpikeTrain, nspikeTrain)
                else [train.minTime for train in self.neuralSpikeTrain]
            ),
            "neural_max_time": (
                self.neuralSpikeTrain.maxTime
                if isinstance(self.neuralSpikeTrain, nspikeTrain)
                else [train.maxTime for train in self.neuralSpikeTrain]
            ),
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "FitResult":
        from .trial import ConfigCollection, TrialConfig

        spike_times = structure["neural_spike_times"]
        neural_name = structure.get("neural_name", "")
        neural_min_time = structure.get("neural_min_time", None)
        neural_max_time = structure.get("neural_max_time", None)
        if spike_times and isinstance(spike_times[0], list):
            train: nspikeTrain | list[nspikeTrain] = []
            for st, name, min_t, max_t in zip(spike_times, neural_name, neural_min_time, neural_max_time):
                train.append(nspikeTrain(st, name=name, minTime=min_t, maxTime=max_t, makePlots=-1))
        else:
            train = nspikeTrain(spike_times, name=neural_name, minTime=neural_min_time, maxTime=neural_max_time, makePlots=-1)
        lam = Covariate(
            structure["lambda_time"],
            np.asarray(structure["lambda_data"], dtype=float),
            structure.get("lambda_name", "lambda"),
            "time",
            "s",
            "spikes/sec",
        )
        configColl = ConfigCollection([TrialConfig(name=name) for name in structure.get("configNames", [])])
        return FitResult(
            train,
            structure.get("covLabels", []),
            structure.get("numHist", []),
            [],
            [],
            lam,
            [np.asarray(coeff, dtype=float) for coeff in structure.get("b", [])],
            structure.get("dev", []),
            [None for _ in structure.get("b", [])],
            structure.get("AIC", []),
            structure.get("BIC", []),
            structure.get("logLL", []),
            configColl,
            [],
            [],
            structure.get("fitType", "poisson"),
        )


class FitSummary:
    """Cross-fit summary statistics for one or more FitResult objects."""

    def __init__(self, fit_results: FitResult | Iterable[FitResult]) -> None:
        if isinstance(fit_results, FitResult):
            self.fitResCell = [fit_results]
        else:
            self.fitResCell = list(fit_results)
            if not self.fitResCell:
                raise ValueError("FitSummary requires at least one FitResult")

        self.numNeurons = len(self.fitResCell)
        self.numResults = max(fr.numResults for fr in self.fitResCell)
        self.fitNames = self.fitResCell[max(range(self.numNeurons), key=lambda idx: self.fitResCell[idx].numResults)].configNames
        self.neuronNumbers = [fr.neuronNumber for fr in self.fitResCell]

        aic = _pad_rows([np.asarray(fr.AIC, dtype=float).reshape(-1) for fr in self.fitResCell])
        bic = _pad_rows([np.asarray(fr.BIC, dtype=float).reshape(-1) for fr in self.fitResCell])
        logll = _pad_rows([np.asarray(fr.logLL, dtype=float).reshape(-1) for fr in self.fitResCell])
        ks = _pad_rows([np.asarray(fr.KSStats, dtype=float).reshape(-1) for fr in self.fitResCell], fill_value=np.nan)

        self.AIC = np.nanmean(aic, axis=0)
        self.BIC = np.nanmean(bic, axis=0)
        self.logLL = np.nanmean(logll, axis=0)
        self.KSStats = np.column_stack([np.nanmean(ks, axis=0), np.nanstd(ks, axis=0)])

    def getDiffAIC(self, idx: int = 1) -> np.ndarray:
        base = self.AIC[idx - 1]
        return self.AIC - base

    def getDiffBIC(self, idx: int = 1) -> np.ndarray:
        base = self.BIC[idx - 1]
        return self.BIC - base

    def getDifflogLL(self, idx: int = 1) -> np.ndarray:
        base = self.logLL[idx - 1]
        return self.logLL - base

    def plotSummary(self, handle=None):
        fig = handle if handle is not None else plt.figure(figsize=(10.0, 4.5))
        fig.clear()
        axes = fig.subplots(1, 3)
        x = np.arange(self.numResults, dtype=float)
        labels = list(self.fitNames)
        for ax, values, title in zip(
            axes,
            (self.AIC, self.BIC, self.logLL),
            ("AIC", "BIC", "log likelihood"),
            strict=False,
        ):
            ax.bar(x, np.asarray(values, dtype=float), color="tab:blue", alpha=0.8)
            ax.set_xticks(x, labels, rotation=30, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig


class FitResSummary(FitSummary):
    """MATLAB-compatible alias for FitSummary."""


__all__ = ["FitResult", "FitSummary", "FitResSummary", "_SingleFit"]
