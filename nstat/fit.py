from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .core import Covariate, nspikeTrain


def _ordered_unique(labels: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(label) for label in labels))


def _matlab_unique(labels: Sequence[str]) -> list[str]:
    return sorted({str(label) for label in labels})


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


def _ksdiscrete(
    pk: np.ndarray,
    st: np.ndarray,
    spikeflag: str,
    *,
    random_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Port of MATLAB Analysis.m local ksdiscrete helper."""

    pk_arr = np.asarray(pk, dtype=float).reshape(-1)
    if np.any(pk_arr < 0.0) or np.any(pk_arr > 1.0):
        raise ValueError("all values for pk must be within [0,1]")

    if spikeflag == "spiketrain":
        st_arr = np.asarray(st, dtype=float).reshape(-1)
        if pk_arr.shape[0] != st_arr.shape[0]:
            raise ValueError("pk and spike train must be same length")
        spike_indices = np.flatnonzero(st_arr == 1.0) + 1
    elif spikeflag == "spikeind":
        st_arr = np.asarray(st, dtype=float).reshape(-1)
        spike_indices = np.unique(np.asarray(st_arr, dtype=int))
    else:
        raise ValueError("spikeflag must be 'spiketrain' or 'spikeind'")

    if spike_indices.size == 0:
        rst = pk_arr.copy()
        return rst, np.sort(rst), np.asarray([], dtype=float), np.nan, np.sort(rst)

    if spike_indices[0] < 1:
        raise ValueError("There is at least one spike with index less than 0")
    if spike_indices[-1] > pk_arr.shape[0]:
        raise ValueError("There is at least one spike with an index greater than the length of pk")

    with np.errstate(divide="ignore", invalid="ignore"):
        qk = -np.log(1.0 - pk_arr)
    n_spikes = int(spike_indices.size)
    rst = np.zeros(max(n_spikes - 1, 0), dtype=float)
    rstold = np.zeros_like(rst)

    if random_values is None:
        draws = np.random.random_sample(rst.shape[0])
    else:
        draws = np.asarray(random_values, dtype=float).reshape(-1)
        if draws.shape[0] != rst.shape[0]:
            raise ValueError("random_values must match the number of inter-spike intervals")

    for r in range(rst.shape[0]):
        ind1 = int(spike_indices[r])
        ind2 = int(spike_indices[r + 1])
        total = float(np.sum(qk[ind1: ind2 - 1]))
        qk_ind2 = float(qk[ind2 - 1])
        delta = -(1.0 / qk_ind2) * np.log(1.0 - float(draws[r]) * (1.0 - np.exp(-qk_ind2)))
        if delta != 0.0:
            total += qk_ind2 * delta
        rst[r] = total
        rstold[r] = float(np.sum(qk[ind1:ind2]))

    rstsort = np.sort(rst)
    inrst = 1.0 / float(max(n_spikes - 1, 1))
    xrst = np.arange(0.5 * inrst, 1.0 - 0.5 * inrst + 0.5 * inrst, inrst, dtype=float)
    cb = 1.36 * np.sqrt(inrst)
    rstoldsort = np.sort(rstold)
    return rst, rstsort, xrst, cb, rstoldsort


def _matlab_compute_ks_arrays(
    spike_obj: nspikeTrain,
    lambda_input: Covariate,
    *,
    dt_correction: int = 1,
    random_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | np.ndarray]:
    """Port of MATLAB Analysis.computeKSStats for a single spike train."""

    n_copy = spike_obj.nstCopy()
    lambda_signal = Covariate(
        np.asarray(lambda_input.time, dtype=float).reshape(-1),
        np.asarray(lambda_input.data, dtype=float),
        lambda_input.name,
        lambda_input.xlabelval,
        lambda_input.xunits,
        lambda_input.yunits,
        list(lambda_input.dataLabels) if getattr(lambda_input, "dataLabels", None) else [],
    )

    n_copy.resample(lambda_signal.sampleRate)
    n_copy.setMinTime(lambda_signal.minTime)
    n_copy.setMaxTime(lambda_signal.maxTime)

    rep_bin = n_copy.isSigRepBinary()
    if not rep_bin:
        lambda_signal = lambda_signal.resample(2.0 * lambda_signal.sampleRate)
        n_copy.resample(lambda_signal.sampleRate)

    lambda_data = np.asarray(lambda_signal.data, dtype=float)
    if lambda_data.ndim == 1:
        lambda_data = lambda_data[:, None]

    n_dims = int(lambda_data.shape[1])
    if random_values is None:
        random_cols = [None] * n_dims
    else:
        rv = np.asarray(random_values, dtype=float)
        if rv.ndim == 1:
            random_cols = [rv.reshape(-1)] * n_dims
        elif rv.ndim == 2 and rv.shape[1] == n_dims:
            random_cols = [rv[:, idx].reshape(-1) for idx in range(n_dims)]
        else:
            raise ValueError("random_values must be 1D or have one column per lambda dimension")

    z_cols: list[np.ndarray] = []
    u_cols: list[np.ndarray] = []
    x_cols: list[np.ndarray] = []
    ks_cols: list[np.ndarray] = []
    stats: list[float] = []

    if int(dt_correction) == 1 and rep_bin:
        pk = np.maximum(lambda_data * (1.0 / max(float(lambda_signal.sampleRate), 1e-12)), 1e-10)
        spike_signal = np.asarray(n_copy.getSigRep().data, dtype=float).reshape(-1)
        min_dim = min(pk.shape[0], spike_signal.shape[0])
        pk = pk[:min_dim, :]
        spike_signal = spike_signal[:min_dim]

        int_cols: list[np.ndarray] = []
        for idx in range(n_dims):
            pk_col = np.clip(pk[:, idx], 0.0, 1.0)
            z_col, _, _, _, _ = _ksdiscrete(
                pk_col,
                spike_signal,
                "spiketrain",
                random_values=random_cols[idx],
            )
            int_cols.append(np.asarray(z_col, dtype=float).reshape(-1))

        if int_cols:
            Z = _pad_rows(int_cols, fill_value=np.nan).T
        else:
            Z = np.zeros((0, n_dims), dtype=float)
    else:
        lambda_pos = np.maximum(lambda_data, 0.0)
        lambda_cov = Covariate(
            lambda_signal.time,
            lambda_pos,
            lambda_signal.name,
            lambda_signal.xlabelval,
            lambda_signal.xunits,
            lambda_signal.yunits,
            list(lambda_signal.dataLabels),
        )
        lambda_int = lambda_cov.integral()
        if n_copy.isSigRepBinary():
            spike_times = np.concatenate([[0.0], np.asarray(n_copy.getSpikeTimes(), dtype=float).reshape(-1)])
        else:
            nst_signal = n_copy.getSigRep()
            spike_times = np.concatenate([[0.0], np.asarray(nst_signal.time[np.asarray(nst_signal.data).reshape(-1) != 0], dtype=float).reshape(-1)])

        if spike_times.size:
            temp_vals = np.asarray(lambda_int.getValueAt(spike_times), dtype=float)
            Z = temp_vals[1:, :] - temp_vals[:-1, :]
        else:
            Z = np.zeros((1, n_dims), dtype=float)

    U = 1.0 - np.exp(-Z)
    if U.ndim == 1:
        U = U[:, None]

    for idx in range(U.shape[1]):
        ks_sorted = np.sort(np.asarray(U[:, idx], dtype=float).reshape(-1))
        n_events = int(ks_sorted.shape[0])
        if n_events:
            x_axis = ((np.arange(1, n_events + 1, dtype=float) - 0.5) / float(n_events))
            ks_stat = float(np.max(np.abs(ks_sorted - x_axis)))
        else:
            x_axis = np.asarray([], dtype=float)
            ks_stat = 1.0
        z_cols.append(np.asarray(Z[:, idx], dtype=float).reshape(-1))
        u_cols.append(np.asarray(U[:, idx], dtype=float).reshape(-1))
        x_cols.append(x_axis)
        ks_cols.append(ks_sorted)
        stats.append(ks_stat)

    Z_out = _pad_rows(z_cols, fill_value=np.nan).T if z_cols else np.zeros((0, n_dims), dtype=float)
    U_out = _pad_rows(u_cols, fill_value=np.nan).T if u_cols else np.zeros((0, n_dims), dtype=float)
    x_out = _pad_rows(x_cols, fill_value=np.nan).T if x_cols else np.zeros((0, n_dims), dtype=float)
    ks_out = _pad_rows(ks_cols, fill_value=np.nan).T if ks_cols else np.zeros((0, n_dims), dtype=float)
    ks_stat = np.asarray(stats, dtype=float)
    if ks_stat.size == 1:
        return Z_out, U_out, x_out, ks_out, float(ks_stat[0])
    return Z_out, U_out, x_out, ks_out, ks_stat


def _ks_curve(uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.sort(np.asarray(uniforms, dtype=float).reshape(-1))
    if u.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    ideal = (np.arange(1, u.size + 1, dtype=float) - 0.5) / float(u.size)
    ci = np.full(u.size, 1.36 / np.sqrt(float(u.size)), dtype=float)
    return ideal, u, ci


def _matlab_kstest2(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alpha: float = 0.05,
    tail: str = "unequal",
) -> tuple[bool, float, float]:
    """Port of MATLAB's asymptotic kstest2 implementation."""

    sample1 = np.asarray(x1, dtype=float).reshape(-1)
    sample2 = np.asarray(x2, dtype=float).reshape(-1)
    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]
    if sample1.size == 0 or sample2.size == 0:
        raise ValueError("kstest2 requires non-empty samples")

    tail_key = str(tail).lower()
    if tail_key not in {"unequal", "smaller", "larger"}:
        raise ValueError("tail must be 'unequal', 'smaller', or 'larger'")

    bin_edges = np.concatenate(([-np.inf], np.sort(np.concatenate((sample1, sample2))), [np.inf]))
    cdf1 = np.histogram(sample1, bins=bin_edges)[0].cumsum(dtype=float) / float(sample1.size)
    cdf2 = np.histogram(sample2, bins=bin_edges)[0].cumsum(dtype=float) / float(sample2.size)

    if tail_key == "unequal":
        delta_cdf = np.abs(cdf1 - cdf2)
    elif tail_key == "smaller":
        delta_cdf = cdf2 - cdf1
    else:
        delta_cdf = cdf1 - cdf2

    ks_statistic = float(np.max(delta_cdf))
    n1 = float(sample1.size)
    n2 = float(sample2.size)
    n = (n1 * n2) / (n1 + n2)
    lam = max((np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks_statistic, 0.0)

    if tail_key != "unequal":
        p_value = float(np.exp(-2.0 * lam * lam))
    else:
        j = np.arange(1.0, 102.0, dtype=float)
        p_value = float(2.0 * np.sum(((-1.0) ** (j - 1.0)) * np.exp(-2.0 * lam * lam * j * j)))
        p_value = min(max(p_value, 0.0), 1.0)

    h = bool(alpha >= p_value)
    return h, p_value, ks_statistic


def _extract_stat_component(stat: Any, candidates: Sequence[str]) -> Any:
    if stat is None:
        return None
    if isinstance(stat, dict):
        for key in candidates:
            if key in stat:
                return stat[key]
        return None
    for key in candidates:
        if hasattr(stat, key):
            return getattr(stat, key)
    return None


def _extract_standard_errors(stat: Any, size: int) -> np.ndarray:
    values = _extract_stat_component(stat, ("se", "std_err", "stderr", "standard_error", "standard_errors"))
    if values is None:
        return np.full(size, np.nan, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == size:
        return arr
    out = np.full(size, np.nan, dtype=float)
    out[: min(size, arr.size)] = arr[: min(size, arr.size)]
    return out


def _extract_significance_mask(stat: Any, coeffs: np.ndarray, standard_errors: np.ndarray) -> np.ndarray:
    out = np.zeros(coeffs.size, dtype=float)
    valid = np.isfinite(standard_errors) & (np.abs(standard_errors) > 0.0) & (np.abs(standard_errors) < 100.0)
    if np.any(valid):
        lower = coeffs[valid] - standard_errors[valid]
        upper = coeffs[valid] + standard_errors[valid]
        out[valid] = ((np.sign(lower) * np.sign(upper)) > 0).astype(float)
        return out
    pvalues = _extract_stat_component(stat, ("p", "p_values", "pvalues", "pValues"))
    if pvalues is not None:
        p_arr = np.asarray(pvalues, dtype=float).reshape(-1)
        out[: min(coeffs.size, p_arr.size)] = (p_arr[: min(coeffs.size, p_arr.size)] < 0.05).astype(float)
    return out


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
        self.uniqueCovLabels = _matlab_unique([label for labels in self.covLabels for label in labels])
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
                labels = self.covLabels[idx] if idx < len(self.covLabels) else []
                if coeff.size == len(labels):
                    intercept = 0.0
                    beta = coeff.copy()
                else:
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
    def lambda_obj(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambda_model(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambda_result(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambdaObj(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambdaCov(self) -> Covariate:
        return self.lambda_signal

    @property
    def lambda_sig(self) -> Covariate:
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

    def __getattr__(self, name: str):
        if name == "lambda":
            return self.lambda_signal
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def setNeuronName(self, name: str):
        if isinstance(self.neuralSpikeTrain, nspikeTrain):
            self.neuralSpikeTrain.setName(str(name))
        elif isinstance(self.neuralSpikeTrain, Sequence):
            for train in self.neuralSpikeTrain:
                if hasattr(train, "setName"):
                    train.setName(str(name))
        self.neuronNumber = str(name)
        return self

    def mapCovLabelsToUniqueLabels(self):
        self.uniqueCovLabels = _matlab_unique([label for labels in self.covLabels for label in labels])
        self.indicesToUniqueLabels = []
        self.flatMask = np.zeros((len(self.uniqueCovLabels), max(len(self.covLabels), 1)), dtype=int)
        for fit_idx, labels in enumerate(self.covLabels):
            indices = [self.uniqueCovLabels.index(label) + 1 for label in labels]
            self.indicesToUniqueLabels.append(indices)
            if indices:
                self.flatMask[np.asarray(indices, dtype=int) - 1, fit_idx] = 1
        return self

    def getSubsetFitResult(self, subfits) -> "FitResult":
        indices = np.asarray(subfits if isinstance(subfits, Sequence) and not isinstance(subfits, (str, bytes)) else [subfits], dtype=int).reshape(-1)
        zero_based = [int(idx) - 1 for idx in indices]
        from .trial import ConfigCollection

        config_items = []
        if self.configs is not None and hasattr(self.configs, "configArray"):
            config_items = [self.configs.configArray[idx] for idx in zero_based]
        subset = FitResult(
            self.neuralSpikeTrain,
            [self.covLabels[idx] for idx in zero_based],
            [self.numHist[idx] for idx in zero_based],
            [self.histObjects[idx] for idx in zero_based],
            [self.ensHistObjects[idx] for idx in zero_based],
            self.lambda_signal,
            [self.b[idx] for idx in zero_based],
            self.dev[zero_based],
            [self.stats[idx] for idx in zero_based],
            self.AIC[zero_based],
            self.BIC[zero_based],
            self.logLL[zero_based],
            ConfigCollection(config_items),
            [self.XvalData[idx] for idx in zero_based] if self.XvalData else [],
            [self.XvalTime[idx] for idx in zero_based] if self.XvalTime else [],
            [self.fitType[idx] for idx in zero_based],
            fits=[self.fits[idx] for idx in zero_based],
        )
        subset.validation = self.validation
        return subset

    def addParamsToFit(self, neuronNum, lambda_signal, b, dev, stats, AIC, BIC, logLL, configColl):
        del neuronNum
        merged = self.mergeResults(
            FitResult(
                self.neuralSpikeTrain,
                [list(labels) for labels in getattr(configColl, "configNames", [])] if False else self.covLabels[:0],
                [],
                [],
                [],
                lambda_signal,
                b,
                dev,
                stats,
                AIC,
                BIC,
                logLL,
                configColl,
                [],
                [],
                self.fitType[0] if self.fitType else "poisson",
            )
        )
        self.__dict__.update(merged.__dict__)
        return self

    def getCoeffs(self, fit_num: int = 1) -> np.ndarray:
        return self.b[fit_num - 1].copy()

    def getHistCoeffs(self, fit_num: int = 1) -> np.ndarray:
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        coeff = self.getCoeffs(fit_num)
        if num_hist <= 0:
            return np.array([], dtype=float)
        return coeff[-num_hist:]

    def getCoeffIndex(self, fit_num: int = 1, sortByEpoch: int = 0):
        if not self.uniqueCovLabels:
            self.mapCovLabelsToUniqueLabels()
        hist_index, _hist_epoch_id, _ = self.getHistIndex(fit_num, sortByEpoch)
        all_index = np.arange(1, len(self.uniqueCovLabels) + 1, dtype=int)
        hist_set = set(np.asarray(hist_index, dtype=int).reshape(-1).tolist())
        act_coeff_index = np.asarray([idx for idx in all_index if idx not in hist_set], dtype=int)
        all_coeff_terms = [str(self.uniqueCovLabels[idx - 1]) for idx in act_coeff_index]
        epoch_ids_all = np.zeros(act_coeff_index.size, dtype=int)
        epochs_exist = False
        for idx, label in enumerate(all_coeff_terms):
            match = re.search(r"_\{(\d+)\}", label)
            if match:
                epochs_exist = True
                epoch_ids_all[idx] = int(match.group(1))
        all_coeff_positions = list(range(act_coeff_index.size))
        non_epoch_positions = [idx for idx, epoch_id in enumerate(epoch_ids_all) if epoch_id == 0]
        if epochs_exist and not sortByEpoch:
            coeff_positions = list(non_epoch_positions)
            epoch_id = np.zeros(len(non_epoch_positions), dtype=int)
            for epoch in sorted({int(value) for value in epoch_ids_all.tolist() if int(value) != 0}):
                matches = [idx for idx, value in enumerate(epoch_ids_all) if int(value) == epoch]
                coeff_positions.extend(matches)
                epoch_id = np.concatenate((epoch_id, epoch * np.ones(len(matches), dtype=int)))
            coeff_index = act_coeff_index[np.asarray(coeff_positions, dtype=int)] if coeff_positions else np.array([], dtype=int)
        elif epochs_exist and sortByEpoch:
            coeff_index = act_coeff_index[np.asarray(all_coeff_positions, dtype=int)]
            epoch_id = np.asarray(epoch_ids_all, dtype=int)
        else:
            coeff_index = act_coeff_index[np.asarray(all_coeff_positions, dtype=int)]
            epoch_id = np.zeros(len(all_coeff_positions), dtype=int)
        num_epochs = int(np.unique(epoch_id).size) if epoch_id.size else 0
        return np.asarray(coeff_index, dtype=int), np.asarray(epoch_id, dtype=int), num_epochs

    def getHistIndex(self, fit_num: int = 1, sortByEpoch: int = 0):
        del fit_num
        if not self.uniqueCovLabels:
            self.mapCovLabelsToUniqueLabels()
        all_hist_index: list[int] = []
        epoch_ids_all: dict[int, int] = {}
        epochs_exist = False
        for idx, label in enumerate(self.uniqueCovLabels, start=1):
            label_str = str(label)
            if not label_str.startswith("["):
                continue
            all_hist_index.append(idx)
            epoch_match = re.search(r"\]_\{(\d+)\}", label_str)
            if epoch_match:
                epochs_exist = True
                epoch_ids_all[idx] = int(epoch_match.group(1))
        if not all_hist_index:
            return np.array([], dtype=int), np.array([], dtype=int), 0
        if epochs_exist and not sortByEpoch:
            hist_index: list[int] = []
            epoch_id: list[int] = []
            for epoch in sorted(set(epoch_ids_all.values())):
                matches = [idx for idx in all_hist_index if epoch_ids_all.get(idx) == epoch]
                hist_index.extend(matches)
                epoch_id.extend([epoch] * len(matches))
        elif epochs_exist and sortByEpoch:
            hist_index = list(all_hist_index)
            epoch_id = [epoch_ids_all.get(idx, 0) for idx in all_hist_index]
        else:
            hist_index = list(all_hist_index)
            epoch_id = [0] * len(all_hist_index)
        num_epochs = len(set(epoch_id)) if epoch_id else 0
        return np.asarray(hist_index, dtype=int), np.asarray(epoch_id, dtype=int), int(num_epochs)

    def getParam(self, paramNames, fit_num: int = 1):
        names = [paramNames] if isinstance(paramNames, str) else list(paramNames)
        coeffs, labels, se = self.getCoeffsWithLabels(fit_num)
        sig = _extract_significance_mask(self.stats[fit_num - 1] if fit_num - 1 < len(self.stats) else None, coeffs, se)
        indices = [labels.index(name) for name in names if name in labels]
        return coeffs[indices], se[indices], sig[indices]

    def getCoeffsWithLabels(self, fit_num: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        coeffs = self.getCoeffs(fit_num)
        labels = list(self.covLabels[fit_num - 1]) if fit_num - 1 < len(self.covLabels) else [f"b_{idx + 1}" for idx in range(coeffs.size)]
        if coeffs.size == len(labels) + 1:
            labels = ["Intercept", *labels]
        elif coeffs.size != len(labels):
            labels = [f"b_{idx + 1}" for idx in range(coeffs.size)]
        se = _extract_standard_errors(self.stats[fit_num - 1] if fit_num - 1 < len(self.stats) else None, coeffs.size)
        return coeffs, labels, se

    def computePlotParams(self, fit_num: int | None = None):
        del fit_num
        if not self.uniqueCovLabels:
            self.mapCovLabelsToUniqueLabels()
            return self.plotParams

        index = np.where(np.sum(self.flatMask, axis=1) > 0)[0]
        b_act = np.full((len(index), self.numResults), np.nan, dtype=float)
        se_act = np.full((len(index), self.numResults), np.nan, dtype=float)
        sig_index = np.zeros((len(index), self.numResults), dtype=float)
        for result_index in range(1, self.numResults + 1):
            coeffs, labels, se = self.getCoeffsWithLabels(result_index)
            criteria = np.where(np.asarray(se, dtype=float).reshape(-1) < 100.0)[0]
            indices_for_fit = (
                np.asarray(self.indicesToUniqueLabels[result_index - 1], dtype=int).reshape(-1) - 1
                if result_index - 1 < len(self.indicesToUniqueLabels)
                else np.array([], dtype=int)
            )
            if criteria.size and indices_for_fit.size:
                valid = criteria[criteria < indices_for_fit.size]
                mapped_rows = indices_for_fit[valid]
                b_act[mapped_rows, result_index - 1] = coeffs[valid]
                se_act[mapped_rows, result_index - 1] = se[valid]
                temp = np.sign(np.column_stack((b_act[:, result_index - 1] - se_act[:, result_index - 1], b_act[:, result_index - 1] + se_act[:, result_index - 1])))
                product_of_signs = temp[:, 0] * temp[:, 1]
                sig_index[:, result_index - 1] = ((product_of_signs > 0) & (se_act[:, result_index - 1] != 0)).astype(float)
        temp_val = np.sum(self.flatMask, axis=1)
        self.plotParams = {
            "bAct": b_act,
            "seAct": se_act,
            "sigIndex": sig_index,
            "xLabels": [self.uniqueCovLabels[idx] for idx in index],
            "numResultsCoeffPresent": temp_val[index].astype(int),
        }
        return self.plotParams

    def getPlotParams(self):
        return self.computePlotParams()

    def isValDataPresent(self) -> bool:
        if not self.XvalTime or not self.XvalData:
            return False
        for time in self.XvalTime:
            arr = np.asarray(time, dtype=float).reshape(-1)
            if arr.size >= 2 and arr[-1] > arr[0]:
                return True
        return False

    def plotValidation(self):
        if self.validation is not None:
            return self.validation.plotResults()
        return None

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

        lambda_labels = list(self.lambda_signal.dataLabels) if getattr(self.lambda_signal, "dataLabels", None) else []
        if lambda_labels:
            idx = min(max(fit_num - 1, 0), len(lambda_labels) - 1)
            selected_labels = [str(lambda_labels[idx])]
        else:
            selected_labels = [f"\\lambda_{{{fit_num}}}"]

        lambda_signal = Covariate(
            time,
            rate_hz,
            "\\lambda(t)",
            self.lambda_signal.xlabelval,
            self.lambda_signal.xunits,
            self.lambda_signal.yunits,
            selected_labels,
        )
        Z, U, xAxis, KSSorted, _ = _matlab_compute_ks_arrays(self._primary_spike_train(), lambda_signal, dt_correction=1)
        z = np.asarray(Z[:, 0], dtype=float).reshape(-1) if np.asarray(Z).size else np.asarray([], dtype=float)
        uniforms = np.asarray(U[:, 0], dtype=float).reshape(-1) if np.asarray(U).size else np.asarray([], dtype=float)
        ideal = np.asarray(xAxis[:, 0], dtype=float).reshape(-1) if np.asarray(xAxis).size else np.asarray([], dtype=float)
        empirical = np.asarray(KSSorted[:, 0], dtype=float).reshape(-1) if np.asarray(KSSorted).size else np.asarray([], dtype=float)
        ci = np.full(ideal.size, 1.36 / np.sqrt(float(ideal.size)), dtype=float) if ideal.size else np.asarray([], dtype=float)
        ks_curve_stat = float(np.max(np.abs(empirical - ideal))) if ideal.size else 1.0
        if ideal.size:
            different, ks_pvalue, ks_stat = _matlab_kstest2(ideal, empirical)
            within = float(not different)
        else:
            ks_stat = 1.0
            ks_pvalue = np.nan
            within = np.nan
        gaussianized = norm.ppf(np.clip(uniforms, 1e-6, 1.0 - 1e-6))
        lags, acf = _autocorrelation(gaussianized, max_lag=25)
        acf_ci = 1.96 / np.sqrt(float(gaussianized.size)) if gaussianized.size else np.nan
        coeffs = self.getCoeffs(fit_num)
        labels = self.covLabels[fit_num - 1] if fit_num - 1 < len(self.covLabels) else []
        if coeffs.size == len(labels):
            coeff_labels = list(labels)
        elif coeffs.size == len(labels) + 1:
            coeff_labels = ["Intercept", *labels]
        else:
            coeff_labels = [f"b_{idx + 1}" for idx in range(coeffs.size)]
        diagnostics: dict[str, np.ndarray | float] = {
            "time": time,
            "rate_hz": rate_hz,
            "counts": counts,
            "lambda_per_bin": lam_per_bin,
            "residual": residual,
            "residual_std": residual_std,
            "z": z,
            "uniforms": uniforms,
            "ks_ideal": ideal,
            "ks_empirical": empirical,
            "ks_ci": ci,
            "ks_stat": ks_stat,
            "ks_curve_stat": ks_curve_stat,
            "ks_pvalue": ks_pvalue,
            "within_conf_int": within,
            "acf_lags": lags,
            "acf_values": acf,
            "acf_ci": acf_ci,
            "gaussianized": gaussianized,
            "coefficients": coeffs,
            "coeff_labels": np.asarray(coeff_labels, dtype=object),
        }
        self._diagnostic_cache[fit_num] = diagnostics
        self.setKSStats(z, uniforms, ideal, empirical, np.asarray([ks_stat], dtype=float))
        self.KSPvalues[fit_num - 1] = ks_pvalue
        self.withinConfInt[fit_num - 1] = within
        self.X = gaussianized
        self.Residual = {
            "time": time,
            "residual": residual,
            "standardized": residual_std,
        }
        self.invGausStats = {"X": gaussianized, "rhoSig": acf.tolist(), "confBoundSig": [acf_ci]}
        return diagnostics

    def computeKSStats(self, fit_num: int = 1) -> dict[str, float]:
        diag = self._compute_diagnostics(fit_num)
        return {
            "ks_stat": float(diag["ks_stat"]),
            "ks_pvalue": float(diag["ks_pvalue"]),
            "within_conf_int": float(diag["within_conf_int"]),
        }

    def computeInvGausTrans(self, fit_num: int = 1) -> np.ndarray:
        return np.asarray(self._compute_diagnostics(fit_num)["gaussianized"], dtype=float)

    def computeFitResidual(self, fit_num: int = 1, window_size: float | None = None) -> Covariate:
        time, rate_hz = self._lambda_series(fit_num)
        if time.size == 0:
            residual = Covariate([], [], "M(t_k)", "time", "s", "counts/bin", ["residual"])
            self.setFitResidual(residual)
            return residual

        if window_size is None:
            window_size = float(np.median(np.diff(time))) if time.size > 1 else 1.0
        else:
            window_size = float(window_size)
        spike_train = self._primary_spike_train().nstCopy()
        spike_train.resample(1.0 / max(window_size, 1e-12))
        spike_train.setMinTime(float(time[0]))
        spike_train.setMaxTime(float(time[-1]))
        sum_spikes = spike_train.getSigRep(window_size, float(time[0]), float(time[-1]))
        window_times = np.linspace(float(time[0]), float(time[-1]), sum_spikes.time.size, dtype=float)
        if np.isfinite(window_size) and window_size > 0:
            origin = float(time[0])
            window_times = origin + np.round((window_times - origin) / float(window_size)) * float(window_size)
            window_times = np.round(window_times, decimals=12)

        lambda_signal = Covariate(
            time,
            rate_hz,
            "\\lambda(t)",
            self.lambda_signal.xlabelval,
            self.lambda_signal.xunits,
            self.lambda_signal.yunits,
            (
                [str(self.lambda_signal.dataLabels[min(max(fit_num - 1, 0), len(self.lambda_signal.dataLabels) - 1)])]
                if getattr(self.lambda_signal, "dataLabels", None)
                else ["\\lambda"]
            ),
        )
        lambda_int = lambda_signal.integral()
        lambda_int_vals = (
            lambda_int.getValueAt(window_times[1:]).reshape(-1, lambda_int.dimension)
            - lambda_int.getValueAt(window_times[:-1]).reshape(-1, lambda_int.dimension)
        )

        spike_window_data = np.asarray(sum_spikes.data, dtype=float)
        if lambda_int_vals.shape[0] == spike_window_data.shape[0]:
            sum_spikes_over_window = spike_window_data
        else:
            sum_spikes_over_window = spike_window_data[1:, :]

        mdata = np.asarray(sum_spikes_over_window, dtype=float) - np.asarray(lambda_int_vals, dtype=float)
        residual_data = np.vstack([np.zeros((1, mdata.shape[1]), dtype=float), mdata])
        residual = Covariate(
            window_times,
            residual_data,
            "M(t_k)",
            lambda_int.xlabelval,
            lambda_int.xunits,
            lambda_int.yunits,
            list(lambda_signal.dataLabels),
        )
        self.setFitResidual(residual)
        return residual

    def evalLambda(self, fit_num: int = 1, newData=None) -> np.ndarray:
        coeffs = self.getCoeffs(fit_num)
        x = np.asarray(newData if newData is not None else [], dtype=float)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x[:, None]
        if isinstance(newData, list):
            arrays = [np.asarray(item, dtype=float).reshape(-1) for item in newData]
            x = np.column_stack(arrays)
        n_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        if x.shape[1] >= coeffs.size:
            eta = x[:, : coeffs.size] @ coeffs
        elif x.shape[1] >= max(coeffs.size - 1, 0):
            eta = coeffs[0] + x[:, : coeffs.size - 1] @ coeffs[1:]
        elif n_hist and x.shape[1] >= coeffs.size - n_hist:
            eta = x[:, : coeffs.size - n_hist] @ coeffs[: coeffs.size - n_hist]
        elif n_hist and x.shape[1] >= coeffs.size - n_hist - 1:
            use = coeffs.size - n_hist - 1
            eta = coeffs[0] + x[:, :use] @ coeffs[1 : use + 1]
        else:
            raise ValueError("newData does not align with the fitted coefficient count")
        rate = np.exp(np.clip(eta, -20.0, 20.0)) * float(self.lambda_signal.sampleRate)
        return rate.reshape(np.asarray(newData[0] if isinstance(newData, list) else x[:, 0]).shape) if x.size else rate

    def plotResults(self, fit_num: int = 1, handle=None):
        fig = handle if handle is not None else plt.figure(figsize=(11.5, 8.0))
        fig.clear()
        grid = fig.add_gridspec(2, 4)
        ks_ax = fig.add_subplot(grid[0, 0:2])
        inv_ax = fig.add_subplot(grid[0, 2])
        seq_ax = fig.add_subplot(grid[0, 3])
        coeff_ax = fig.add_subplot(grid[1, 0:2])
        residual_ax = fig.add_subplot(grid[1, 2:4])
        self.KSPlot(fit_num=fit_num, handle=ks_ax)
        self.plotInvGausTrans(fit_num=fit_num, handle=inv_ax)
        self.plotSeqCorr(fit_num=fit_num, handle=seq_ax)
        self.plotCoeffs(fit_num=fit_num, handle=coeff_ax)
        self.plotResidual(fit_num=fit_num, handle=residual_ax)
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
        ax.set_title("KS Plot of Rescaled ISIs\nwith 95% Confidence Intervals")
        return ax

    def plotResidual(self, fit_num: int | Sequence[int] | None = None, handle=None):
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        if fit_num is None:
            fit_indices = list(range(1, self.numResults + 1))
        elif np.isscalar(fit_num):
            fit_indices = [int(fit_num)]
        else:
            fit_indices = [int(item) for item in fit_num]

        for fit_idx in fit_indices:
            residual = self.computeFitResidual(fit_idx)
            residual_data = np.asarray(residual.data, dtype=float)
            if residual_data.ndim == 1:
                residual_data = residual_data[:, None]
            ax.plot(
                np.asarray(residual.time, dtype=float),
                residual_data[:, 0],
                linewidth=1.0,
                label=f"\\lambda_{{{fit_idx}}}",
            )
        ax.set_xlabel("time [s]")
        ax.set_ylabel(r"$M(t_k)\; [Hz*s]$")
        ax.set_title("Point Process Residual")
        ymax = max(abs(value) for value in ax.get_ylim())
        if ymax == 0.0:
            ymax = 1.0
        ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
        legend = ax.legend(loc="upper right")
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(14)
        return ax

    def plotInvGausTrans(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        lags = np.asarray(diag["acf_lags"], dtype=float)
        acf = np.asarray(diag["acf_values"], dtype=float)
        if lags.size:
            ax.vlines(lags, 0.0, acf, color="tab:green", linewidth=1.2)
            ax.axhline(float(diag["acf_ci"]), color="tab:red", linewidth=1.0)
            ax.axhline(-float(diag["acf_ci"]), color="tab:red", linewidth=1.0)
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_xlabel(r"$\Delta \tau\; [sec]$")
        ax.set_ylabel(r"$ACF[ \Phi^{-1}(u_i) ]$")
        ax.set_title("Autocorrelation Function\nof Rescaled ISIs\nwith 95% CIs")
        return ax

    def plotSeqCorr(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        uniforms = np.asarray(diag["uniforms"], dtype=float)
        if uniforms.size >= 2:
            ax.plot(
                uniforms[:-1],
                uniforms[1:],
                ".",
                color="tab:orange",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("u_j")
        ax.set_ylabel("u_{j+1}")
        ax.set_title("Sequential Correlation of\nRescaled ISIs")
        return ax

    def plotCoeffs(self, fit_num: int = 1, handle=None):
        diag = self._compute_diagnostics(fit_num)
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        coeffs = np.asarray(diag["coefficients"], dtype=float)
        labels = list(np.asarray(diag["coeff_labels"], dtype=object))
        xpos = np.arange(coeffs.size, dtype=float)
        ax.plot(xpos, coeffs, "o-", color="tab:blue", linewidth=1.0, label=f"\\lambda_{{{fit_num}}}")
        ax.set_xticks(xpos, labels, rotation=45, ha="right")
        ax.set_ylabel("GLM Fit Coefficients")
        ax.set_title("GLM Coefficients with 95% CIs (* p<0.05)")
        ax.legend(loc="lower right")
        return ax

    def plotCoeffsWithoutHistory(self, fit_num: int = 1, sortByEpoch: int = 0, plotSignificance: int = 1, handle=None):
        del sortByEpoch, plotSignificance
        coeffs, labels, _ = self.getCoeffsWithLabels(fit_num)
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        if num_hist > 0:
            coeffs = coeffs[:-num_hist]
            labels = labels[:-num_hist]
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        xpos = np.arange(coeffs.size, dtype=float)
        ax.plot(xpos, coeffs, "o-", color="tab:blue", linewidth=1.0)
        ax.set_xticks(xpos, labels, rotation=45, ha="right")
        ax.set_ylabel("GLM Fit Coefficients")
        ax.set_title("GLM Coefficients with 95% CIs (* p<0.05)")
        return ax

    def plotHistCoeffs(self, fit_num: int = 1, sortByEpoch: int = 0, plotSignificance: int = 1, handle=None):
        del sortByEpoch, plotSignificance
        coeffs = self.getHistCoeffs(fit_num)
        labels = list(self.covLabels[fit_num - 1])[-coeffs.size :] if coeffs.size and fit_num - 1 < len(self.covLabels) else [f"hist_{idx + 1}" for idx in range(coeffs.size)]
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        xpos = np.arange(coeffs.size, dtype=float)
        if coeffs.size:
            ax.plot(xpos, coeffs, "o-", color="tab:orange", linewidth=1.0)
            ax.set_xticks(xpos, labels, rotation=45, ha="right")
        ax.set_ylabel("GLM Fit Coefficients")
        ax.set_title("GLM Coefficients with 95% CIs (* p<0.05)")
        return ax

    def setKSStats(self, Z, U, xAxis, KSSorted, ks_stat):
        self.Z = np.asarray(Z, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.KSXAxis = np.asarray(xAxis, dtype=float)
        self.KSSorted = np.asarray(KSSorted, dtype=float)
        x_axis = np.asarray(xAxis, dtype=float)
        ks_sorted = np.asarray(KSSorted, dtype=float)
        if x_axis.ndim == 1:
            x_axis = x_axis[:, None]
        if ks_sorted.ndim == 1:
            ks_sorted = ks_sorted[:, None]

        if x_axis.size and ks_sorted.size:
            n_cols = min(x_axis.shape[1], ks_sorted.shape[1], self.numResults)
            for idx in range(n_cols):
                different, p_value, stat = _matlab_kstest2(x_axis[:, idx], ks_sorted[:, idx])
                self.KSStats[idx, 0] = stat
                self.KSPvalues[idx] = p_value
                self.withinConfInt[idx] = float(not different)
        else:
            value = np.asarray(ks_stat, dtype=float).reshape(-1)
            self.KSStats[: value.size, 0] = value
        return self

    def setInvGausStats(self, X, rhoSig, confBoundSig):
        self.invGausStats = {"X": np.asarray(X, dtype=float), "rhoSig": rhoSig, "confBoundSig": confBoundSig}
        return self

    def setFitResidual(self, M):
        self.Residual = M
        return self

    def toStructure(self) -> dict[str, Any]:
        lambda_structure = (
            self.lambda_signal.toStructure()
            if hasattr(self.lambda_signal, "toStructure")
            else {
                "time": self.lambda_signal.time.tolist(),
                "data": self.lambda_signal.data.tolist(),
                "name": self.lambda_signal.name,
                "xlabelval": self.lambda_signal.xlabelval,
                "xunits": self.lambda_signal.xunits,
                "yunits": self.lambda_signal.yunits,
                "dataLabels": list(getattr(self.lambda_signal, "dataLabels", [])),
                "plotProps": list(getattr(self.lambda_signal, "plotProps", [])),
            }
        )
        neural_structure = (
            self.neuralSpikeTrain.toStructure()
            if isinstance(self.neuralSpikeTrain, nspikeTrain)
            else [train.toStructure() if hasattr(train, "toStructure") else train for train in self.neuralSpikeTrain]
        )
        configs_structure = self.configs.toStructure() if self.configs is not None else None
        return {
            "covLabels": [list(labels) for labels in self.covLabels],
            "numHist": list(self.numHist),
            "lambda": lambda_structure,
            "lambda_time": self.lambda_signal.time.tolist(),
            "lambda_data": self.lambda_signal.data.tolist(),
            "lambda_name": self.lambda_signal.name,
            "b": [coeff.tolist() for coeff in self.b],
            "dev": self.dev.tolist(),
            "AIC": self.AIC.tolist(),
            "BIC": self.BIC.tolist(),
            "logLL": self.logLL.tolist(),
            "configs": configs_structure,
            "configNames": list(self.configNames),
            "fitType": list(self.fitType),
            "neuralSpikeTrain": neural_structure,
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
            "XvalData": [
                np.asarray(item, dtype=float).tolist() if not isinstance(item, list) else item
                for item in self.XvalData
            ],
            "XvalTime": [np.asarray(item, dtype=float).tolist() for item in self.XvalTime],
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "FitResult":
        from .trial import ConfigCollection, TrialConfig

        neural_structure = structure.get("neuralSpikeTrain")
        if isinstance(neural_structure, dict):
            train: nspikeTrain | list[nspikeTrain] = nspikeTrain.fromStructure(neural_structure)
        elif isinstance(neural_structure, list) and neural_structure and isinstance(neural_structure[0], dict):
            train = [nspikeTrain.fromStructure(item) for item in neural_structure]
        else:
            spike_times = structure["neural_spike_times"]
            neural_name = structure.get("neural_name", "")
            neural_min_time = structure.get("neural_min_time", None)
            neural_max_time = structure.get("neural_max_time", None)
            if spike_times and isinstance(spike_times[0], list):
                train = []
                for st, name, min_t, max_t in zip(spike_times, neural_name, neural_min_time, neural_max_time):
                    train.append(nspikeTrain(st, name=name, minTime=min_t, maxTime=max_t, makePlots=-1))
            else:
                train = nspikeTrain(spike_times, name=neural_name, minTime=neural_min_time, maxTime=neural_max_time, makePlots=-1)

        lambda_structure = structure.get("lambda")
        if isinstance(lambda_structure, dict):
            lam = Covariate.fromStructure(lambda_structure)
        else:
            lam = Covariate(
                structure["lambda_time"],
                np.asarray(structure["lambda_data"], dtype=float),
                structure.get("lambda_name", "lambda"),
                "time",
                "s",
                "spikes/sec",
            )

        configs_structure = structure.get("configs")
        if isinstance(configs_structure, dict):
            configColl = ConfigCollection.fromStructure(configs_structure)
        else:
            configColl = ConfigCollection([TrialConfig(name=name) for name in structure.get("configNames", [])])
        config_names = list(structure.get("configNames", []))
        if config_names:
            configColl.setConfigNames(config_names, list(range(1, len(config_names) + 1)))
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
            structure.get("XvalData", []),
            structure.get("XvalTime", []),
            structure.get("fitType", "poisson"),
        )

    @staticmethod
    def CellArrayToStructure(fitResObjCell):
        return [fit.toStructure() for fit in fitResObjCell]


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
        self.maxNumIndex = int(max(range(self.numNeurons), key=lambda idx: self.fitResCell[idx].numResults) + 1)
        self.fitNames = self.fitResCell[max(range(self.numNeurons), key=lambda idx: self.fitResCell[idx].numResults)].configNames
        self.neuronNumbers = [fr.neuronNumber for fr in self.fitResCell]

        self.dev = _pad_rows([np.asarray(fr.dev, dtype=float).reshape(-1) for fr in self.fitResCell])
        self.AIC = _pad_rows([np.asarray(fr.AIC, dtype=float).reshape(-1) for fr in self.fitResCell])
        self.BIC = _pad_rows([np.asarray(fr.BIC, dtype=float).reshape(-1) for fr in self.fitResCell])
        self.logLL = _pad_rows([np.asarray(fr.logLL, dtype=float).reshape(-1) for fr in self.fitResCell])
        self.KSStats = _pad_rows([np.asarray(fr.KSStats, dtype=float).reshape(-1) for fr in self.fitResCell], fill_value=np.nan)
        self.KSPvalues = _pad_rows([np.asarray(fr.KSPvalues, dtype=float).reshape(-1) for fr in self.fitResCell], fill_value=np.nan)
        self.withinConfInt = _pad_rows([np.asarray(fr.withinConfInt, dtype=float).reshape(-1) for fr in self.fitResCell], fill_value=np.nan)
        self.meanAIC = np.nanmean(self.AIC, axis=0)
        self.meanBIC = np.nanmean(self.BIC, axis=0)
        self.meanlogLL = np.nanmean(self.logLL, axis=0)
        self.meanKSStats = np.nanmean(self.KSStats, axis=0)
        self.stdKSStats = np.nanstd(self.KSStats, axis=0)
        self.uniqueCovLabels: list[str] = []
        self.coeffMin = np.nan
        self.coeffMax = np.nan
        self.plotParams: dict[str, Any] = {}
        self.mapCovLabelsToUniqueLabels()

    def getDiffAIC(self, idx: int = 1) -> np.ndarray:
        if self.numResults > 1:
            keep = [col for col in range(self.AIC.shape[1]) if col != (idx - 1)]
            return self.AIC[:, keep] - self.AIC[:, [idx - 1]]
        return self.AIC.copy()

    def getDiffBIC(self, idx: int = 1) -> np.ndarray:
        if self.numResults > 1:
            keep = [col for col in range(self.BIC.shape[1]) if col != (idx - 1)]
            return self.BIC[:, keep] - self.BIC[:, [idx - 1]]
        return self.BIC.copy()

    def getDifflogLL(self, idx: int = 1) -> np.ndarray:
        if self.numResults > 1:
            keep = [col for col in range(self.logLL.shape[1]) if col != (idx - 1)]
            return self.logLL[:, keep] - self.logLL[:, [idx - 1]]
        return self.logLL.copy()

    def mapCovLabelsToUniqueLabels(self):
        self.uniqueCovLabels = _matlab_unique(
            [label for fit in self.fitResCell for labels in fit.covLabels for label in labels]
        )
        return self.uniqueCovLabels

    def computePlotParams(self):
        labels = list(self.uniqueCovLabels)
        flat_mask = np.zeros((len(labels), self.numResults, self.numNeurons), dtype=int)
        b_act = np.full((len(labels), self.numResults, self.numNeurons), np.nan, dtype=float)
        se_act = np.full_like(b_act, np.nan)
        sig_index = np.zeros_like(b_act, dtype=float)
        for neuron_idx, fit in enumerate(self.fitResCell):
            for fit_idx in range(1, self.numResults + 1):
                if fit_idx > fit.numResults:
                    continue
                curr_labels = fit.covLabels[fit_idx - 1] if fit_idx - 1 < len(fit.covLabels) else []
                index = [labels.index(label) for label in curr_labels if label in labels]
                if index:
                    flat_mask[np.asarray(index, dtype=int), fit_idx - 1, neuron_idx] = 1
                fit_plot_params = fit.getPlotParams()
                orig_index = (
                    np.asarray(fit.indicesToUniqueLabels[fit_idx - 1], dtype=int).reshape(-1) - 1
                    if fit_idx - 1 < len(fit.indicesToUniqueLabels)
                    else np.array([], dtype=int)
                )
                if index and orig_index.size:
                    mapped = np.asarray(index, dtype=int)
                    valid = orig_index < fit_plot_params["bAct"].shape[0]
                    mapped = mapped[valid]
                    orig_index = orig_index[valid]
                    b_act[mapped, fit_idx - 1, neuron_idx] = np.asarray(fit_plot_params["bAct"], dtype=float)[orig_index, fit_idx - 1]
                    se_act[mapped, fit_idx - 1, neuron_idx] = np.asarray(fit_plot_params["seAct"], dtype=float)[orig_index, fit_idx - 1]
                    sig_index[mapped, fit_idx - 1, neuron_idx] = np.asarray(fit_plot_params["sigIndex"], dtype=float)[orig_index, fit_idx - 1]
        self.plotParams = {
            "bAct": b_act,
            "seAct": se_act,
            "sigIndex": sig_index,
            "xLabels": labels,
            "numResultsCoeffPresent": np.sum(flat_mask, axis=(1, 2)).astype(int),
        }
        return self.plotParams

    def setCoeffRange(self, minVal, maxVal):
        self.coeffMin = float(minVal)
        self.coeffMax = float(maxVal)
        return self

    def getCoeffs(self, fitNum: int = 1):
        fit_idx = int(fitNum)
        coeff_index, epoch_id, num_epochs = self.getCoeffIndex(fit_idx)
        coeff_index = np.asarray(coeff_index, dtype=int).reshape(-1)
        epoch_id = np.asarray(epoch_id, dtype=int).reshape(-1)
        if coeff_index.size == 0:
            return np.array([], dtype=float), [], np.array([], dtype=float)

        coeff_strings = [str(self.uniqueCovLabels[idx - 1]) for idx in coeff_index]
        base_strings = [re.sub(r"_\{\d+\}$", "", label) for label in coeff_strings]
        unique_coeffs = _matlab_unique(base_strings)
        min_epoch = int(np.min(epoch_id)) if epoch_id.size else 0
        num_epochs = int(num_epochs) if int(num_epochs) > 0 else 1
        plot_params = self.computePlotParams()
        coeff_mat = np.full((len(unique_coeffs), num_epochs, self.numNeurons), np.nan, dtype=float)
        se_mat = np.full_like(coeff_mat, np.nan)
        labels: list[list[str]] = [["" for _ in range(num_epochs)] for _ in unique_coeffs]

        for row_idx, base_label in enumerate(unique_coeffs):
            matches = [idx for idx, curr in enumerate(base_strings) if curr == base_label]
            coeff_str_index = coeff_index[matches]
            curr_epoch_id = epoch_id[matches]
            epoch_positions = curr_epoch_id + 1 if min_epoch == 0 else curr_epoch_id
            for coeff_label_index, epoch_position in zip(coeff_str_index, epoch_positions, strict=False):
                label = str(self.uniqueCovLabels[int(coeff_label_index) - 1])
                labels[row_idx][int(epoch_position) - 1] = label
                coeff_mat[row_idx, int(epoch_position) - 1, :] = np.asarray(
                    plot_params["bAct"][int(coeff_label_index) - 1, fit_idx - 1, :],
                    dtype=float,
                )
                se_mat[row_idx, int(epoch_position) - 1, :] = np.asarray(
                    plot_params["seAct"][int(coeff_label_index) - 1, fit_idx - 1, :],
                    dtype=float,
                )

        if self.numNeurons == 1:
            coeff_out = coeff_mat[:, :, 0].T
            se_out = se_mat[:, :, 0].T
        elif num_epochs == 1:
            coeff_out = coeff_mat[:, 0, :]
            se_out = se_mat[:, 0, :]
        else:
            coeff_out = coeff_mat
            se_out = se_mat

        if num_epochs == 1:
            label_out: list[str] | list[list[str]] = [row[0] for row in labels]
        else:
            label_out = labels
        return np.asarray(coeff_out, dtype=float), label_out, np.asarray(se_out, dtype=float)

    def getHistCoeffs(self, fitNum: int = 1):
        fit_idx = int(fitNum)
        hist_index, epoch_id, num_epochs = self.getHistIndex(fit_idx)
        hist_index = np.asarray(hist_index, dtype=int).reshape(-1)
        epoch_id = np.asarray(epoch_id, dtype=int).reshape(-1)
        if hist_index.size == 0:
            return np.array([], dtype=float), [], np.array([], dtype=float)

        hist_strings = [str(self.uniqueCovLabels[idx - 1]) for idx in hist_index]
        base_strings = [re.sub(r"_\{\d+\}$", "", label) for label in hist_strings]
        unique_coeffs = _matlab_unique(base_strings)
        min_epoch = int(np.min(epoch_id)) if epoch_id.size else 0
        num_epochs = int(num_epochs) if int(num_epochs) > 0 else 1
        plot_params = self.computePlotParams()
        hist_mat = np.full((len(unique_coeffs), num_epochs, self.numNeurons), np.nan, dtype=float)
        labels: list[list[str]] = [["" for _ in range(num_epochs)] for _ in unique_coeffs]

        for row_idx, base_label in enumerate(unique_coeffs):
            matches = [idx for idx, curr in enumerate(base_strings) if curr == base_label]
            hist_str_index = hist_index[matches]
            curr_epoch_id = epoch_id[matches]
            epoch_positions = curr_epoch_id + 1 if min_epoch == 0 else curr_epoch_id
            for coeff_label_index, epoch_position in zip(hist_str_index, epoch_positions, strict=False):
                label = str(self.uniqueCovLabels[int(coeff_label_index) - 1])
                labels[row_idx][int(epoch_position) - 1] = label
                hist_mat[row_idx, int(epoch_position) - 1, :] = np.asarray(
                    plot_params["bAct"][int(coeff_label_index) - 1, fit_idx - 1, :],
                    dtype=float,
                )

        if self.numNeurons == 1:
            hist_out = hist_mat[:, :, 0].T
            se_out = np.full_like(hist_out, np.nan, dtype=float)
        elif num_epochs == 1:
            hist_out = hist_mat[:, 0, :]
            se_out = np.full_like(hist_out, np.nan, dtype=float)
        else:
            hist_out = hist_mat
            se_out = np.full_like(hist_out, np.nan, dtype=float)

        if num_epochs == 1:
            label_out: list[str] | list[list[str]] = [row[0] for row in labels]
        else:
            label_out = labels
        return np.asarray(hist_out, dtype=float), label_out, np.asarray(se_out, dtype=float)

    def getSigCoeffs(self, fitNum: int = 1):
        labels = list(self.computePlotParams().get("xLabels", []))
        sig = np.full((len(labels), self.numNeurons), np.nan, dtype=float)
        for row_idx, fit in enumerate(self.fitResCell):
            coeffs, fit_labels, se = fit.getCoeffsWithLabels(fitNum)
            mask = _extract_significance_mask(
                fit.stats[fitNum - 1] if fitNum - 1 < len(fit.stats) else None,
                coeffs,
                se,
            )
            for coeff, label, value in zip(coeffs, fit_labels, mask, strict=False):
                if label in labels:
                    sig[labels.index(label), row_idx] = float(coeff) * float(value)
        return sig

    def binCoeffs(self, minVal, maxVal, binSize):
        plot_params = self.computePlotParams()
        edges = np.arange(float(minVal), float(maxVal) + float(binSize), float(binSize), dtype=float)
        if edges.size < 2:
            edges = np.array([float(minVal), float(maxVal)], dtype=float)
        num_labels = len(plot_params["xLabels"])
        N = np.zeros((edges.size, num_labels), dtype=float)
        percent_sig = np.zeros(num_labels, dtype=float)
        for idx in range(num_labels):
            sig_vals = np.asarray(plot_params["bAct"][idx, :, :], dtype=float)
            sig_mask = np.asarray(plot_params["sigIndex"][idx, :, :], dtype=float) == 1
            vals = sig_vals[sig_mask]
            vals = vals[np.isfinite(vals)]
            counts = np.zeros(edges.size, dtype=float)
            if vals.size:
                bin_index = np.searchsorted(edges, vals, side="right") - 1
                exact_last = np.isclose(vals, edges[-1])
                bin_index[exact_last] = edges.size - 1
                valid = (vals >= edges[0]) & ((vals < edges[-1]) | exact_last) & (bin_index >= 0) & (bin_index < edges.size)
                if np.any(valid):
                    counts = np.bincount(bin_index[valid], minlength=edges.size).astype(float)
                total = counts.sum()
                if total > 0:
                    N[:, idx] = counts / total
            denom = float(plot_params["numResultsCoeffPresent"][idx]) if idx < len(plot_params["numResultsCoeffPresent"]) else 0.0
            if denom > 0:
                percent_sig[idx] = counts.sum() / denom
        return N, edges, percent_sig

    def plot2dCoeffSummary(self, h=None):
        if not np.isfinite(self.coeffMin) or not np.isfinite(self.coeffMax):
            self.setCoeffRange(-12.0, 12.0)
        N, edges, percent_sig = self.binCoeffs(self.coeffMin, self.coeffMax, 0.1)
        ax = h if h is not None else plt.subplots(1, 1, figsize=(8.0, 4.0))[1]
        handles = []
        for idx, label in enumerate(self.plotParams.get("xLabels", []), start=1):
            (line,) = ax.plot(edges, N[:, idx - 1] + idx, linewidth=1.0)
            handles.append(line)
            ax.text(
                float(self.coeffMax),
                float(idx),
                f"{percent_sig[idx - 1] * 100:.0f}%_{{sig}}",
                fontsize=6,
                ha="right",
                va="center",
            )
        ax.set_yticks(np.arange(1, len(self.plotParams.get("xLabels", [])) + 1))
        ax.set_yticklabels(self.plotParams.get("xLabels", []), fontsize=6)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("")
        ax.set_xlabel("")
        return ax

    def plot3dCoeffSummary(self, h=None):
        if not np.isfinite(self.coeffMin) or not np.isfinite(self.coeffMax):
            self.setCoeffRange(-12.0, 12.0)
        N, edges, _ = self.binCoeffs(self.coeffMin, self.coeffMax, 0.1)
        if h is None:
            fig = plt.figure(figsize=(8.0, 5.0))
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = h
        x = np.asarray(edges, dtype=float)
        y = np.arange(1, N.shape[1] + 1, dtype=float)
        X, Y = np.meshgrid(x, y, indexing="ij")
        ax.plot_surface(X, Y, N, edgecolor="none", alpha=0.6)
        ax.view_init(elev=28, azim=71.5)
        ax.grid(True)
        ax.set_yticks(y)
        ax.set_yticklabels(self.plotParams.get("xLabels", []))
        return ax

    def getHistIndex(self, fitNum: int | Sequence[int] | None = None, sortByEpoch: int = 0):
        del sortByEpoch
        if fitNum is None:
            fit_indices = list(range(1, self.numResults + 1))
        elif np.isscalar(fitNum):
            fit_indices = [int(fitNum)]
        else:
            fit_indices = [int(item) for item in fitNum]

        hist_index: list[int] = []
        epoch_id: list[int] = []
        for idx, label in enumerate(self.uniqueCovLabels, start=1):
            if not isinstance(label, str):
                continue
            label_lower = label.lower()
            if (
                label.startswith("[")
                or "*hist" in label_lower
                or "history" in label_lower
            ):
                present = False
                for fit_idx in fit_indices:
                    if fit_idx - 1 >= len(self.fitResCell[0].covLabels):
                        continue
                    fit_labels = [
                        str(item)
                        for fit in self.fitResCell
                        if fit_idx - 1 < len(fit.covLabels)
                        for item in fit.covLabels[fit_idx - 1]
                    ]
                    if label in fit_labels:
                        present = True
                        break
                if present:
                    hist_index.append(idx)
                    epoch_id.append(0)
        hist_array = np.asarray(hist_index, dtype=int)
        epoch_array = np.asarray(epoch_id, dtype=int)
        if hist_array.size:
            plot_params = self.computePlotParams()
            fit_zero = [fit_idx - 1 for fit_idx in fit_indices if 0 < fit_idx <= self.numResults]
            if fit_zero:
                b_act = np.asarray(plot_params["bAct"][:, fit_zero, :], dtype=float).reshape(len(self.uniqueCovLabels), -1)
                non_nan_index = np.where(np.sum(~np.isnan(b_act), axis=1) >= 1)[0] + 1
                if non_nan_index.size == 0:
                    fallback = []
                    for idx, label in enumerate(self.uniqueCovLabels, start=1):
                        present = False
                        for fit_idx in fit_indices:
                            for fit in self.fitResCell:
                                if fit_idx - 1 < len(fit.covLabels) and label in fit.covLabels[fit_idx - 1]:
                                    present = True
                                    break
                            if present:
                                break
                        if present:
                            fallback.append(idx)
                    non_nan_index = np.asarray(fallback, dtype=int)
                valid = np.isin(hist_array, non_nan_index)
                hist_array = hist_array[valid]
                epoch_array = epoch_array[valid]
        num_epochs = int(np.unique(epoch_array).size) if epoch_array.size else 0
        return hist_array, epoch_array, num_epochs

    def getCoeffIndex(self, fitNum: int | Sequence[int] | None = None, sortByEpoch: int = 0):
        hist_index, _, _ = self.getHistIndex(fitNum)
        hist_set = set(hist_index.tolist())
        if fitNum is None:
            fit_indices = list(range(1, self.numResults + 1))
        elif np.isscalar(fitNum):
            fit_indices = [int(fitNum)]
        else:
            fit_indices = [int(item) for item in fitNum]
        plot_params = self.computePlotParams()
        fit_zero = [fit_idx - 1 for fit_idx in fit_indices if 0 < fit_idx <= self.numResults]
        if fit_zero:
            b_act = np.asarray(plot_params["bAct"][:, fit_zero, :], dtype=float).reshape(len(self.uniqueCovLabels), -1)
            non_nan_index = np.where(np.sum(~np.isnan(b_act), axis=1) >= 1)[0] + 1
        else:
            non_nan_index = np.array([], dtype=int)
        if non_nan_index.size == 0:
            fallback = []
            for idx, label in enumerate(self.uniqueCovLabels, start=1):
                present = False
                for fit_idx in fit_indices:
                    for fit in self.fitResCell:
                        if fit_idx - 1 < len(fit.covLabels) and label in fit.covLabels[fit_idx - 1]:
                            present = True
                            break
                    if present:
                        break
                if present:
                    fallback.append(idx)
            non_nan_index = np.asarray(fallback, dtype=int)
        coeff_index = [idx for idx in range(1, len(self.uniqueCovLabels) + 1) if idx not in hist_set and idx in set(non_nan_index.tolist())]
        epoch_id = np.zeros(len(coeff_index), dtype=int)
        num_epochs = int(np.unique(epoch_id).size) if epoch_id.size else 0
        return np.asarray(coeff_index, dtype=int), epoch_id, num_epochs

    def plotIC(self, handle=None):
        fig = handle if handle is not None else plt.figure(figsize=(9.0, 3.5))
        fig.clear()
        axes = fig.subplots(1, 3)
        self.plotAIC(handle=axes[0])
        self.plotBIC(handle=axes[1])
        self.plotlogLL(handle=axes[2])
        fig.tight_layout()
        return fig

    def plotAIC(self, handle=None):
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 3.5))[1]
        ax.boxplot(self.AIC, labels=self.fitNames)
        ax.set_ylabel("AIC")
        ax.set_title("AIC Across Neurons")
        return ax

    def plotBIC(self, handle=None):
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 3.5))[1]
        ax.boxplot(self.BIC, labels=self.fitNames)
        ax.set_ylabel("BIC")
        ax.set_title("BIC Across Neurons")
        return ax

    def plotlogLL(self, handle=None):
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 3.5))[1]
        ax.boxplot(self.logLL, labels=self.fitNames)
        ax.set_ylabel("log likelihood")
        ax.set_title("log likelihood Across Neurons")
        return ax

    def plotResidualSummary(self, handle=None):
        fig = handle if handle is not None else plt.figure(figsize=(8.0, 3.5))
        fig.clear()
        num_neurons = max(int(self.numNeurons), 1)
        if num_neurons <= 4:
            nrows, ncols = 2, 2
        elif num_neurons <= 8:
            nrows, ncols = 2, 4
        elif num_neurons <= 12:
            nrows, ncols = 3, 4
        elif num_neurons <= 16:
            nrows, ncols = 4, 4
        elif num_neurons <= 20:
            nrows, ncols = 5, 4
        elif num_neurons <= 24:
            nrows, ncols = 6, 4
        elif num_neurons <= 40:
            nrows, ncols = 10, 4
        else:
            nrows, ncols = 10, 10

        axes = [fig.add_subplot(nrows, ncols, idx + 1) for idx in range(num_neurons)]
        for idx, fit in enumerate(self.fitResCell[:num_neurons]):
            ax = axes[idx]
            fit.plotResidual(handle=ax)
            legend = ax.get_legend()
            if idx != num_neurons - 1:
                if legend is not None:
                    legend.remove()
            elif legend is not None:
                legend.set_loc("center left")
                legend.set_bbox_to_anchor((1.02, 0.5))
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title("")
        fig.tight_layout()
        return fig

    def plotAllCoeffs(
        self,
        h=None,
        fitNum: int | Sequence[int] | None = None,
        plotProps=None,
        plotSignificance: int = 1,
        subIndex: Sequence[int] | None = None,
        legendLabels: Sequence[str] | None = None,
    ):
        del plotProps, plotSignificance
        ax = h if h is not None else plt.subplots(1, 1, figsize=(9.0, 4.0))[1]
        if fitNum is None:
            fit_indices = list(range(1, self.numResults + 1))
        elif np.isscalar(fitNum):
            fit_indices = [int(fitNum)]
        else:
            fit_indices = [int(item) for item in fitNum]

        coeff_labels = list(self.uniqueCovLabels)
        if subIndex is None:
            sub_labels = coeff_labels
        else:
            sub_zero = [int(idx) - 1 if int(idx) >= 1 else int(idx) for idx in subIndex]
            sub_labels = [coeff_labels[idx] for idx in sub_zero if 0 <= idx < len(coeff_labels)]
        x = np.arange(1, len(sub_labels) + 1, dtype=float)

        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for fit_idx in fit_indices:
            coeffs, labels, se = self.getCoeffs(fit_idx)
            label_map = {label: idx for idx, label in enumerate(labels)}
            coeffs = np.asarray(coeffs, dtype=float)
            se = np.asarray(se, dtype=float)
            if coeffs.ndim == 1:
                if coeffs.size == self.numNeurons and len(labels) == 1:
                    coeffs = coeffs.reshape(self.numNeurons, 1)
                    se = se.reshape(self.numNeurons, 1)
                else:
                    coeffs = coeffs.reshape(1, -1)
                    se = se.reshape(1, -1)
            elif coeffs.ndim == 2 and coeffs.shape == (len(labels), self.numNeurons):
                coeffs = coeffs.T
                se = se.T
            coeff_view = np.full((self.numNeurons, len(sub_labels)), np.nan, dtype=float)
            se_view = np.full_like(coeff_view, np.nan)
            for col, label in enumerate(sub_labels):
                src = label_map.get(label)
                if src is not None:
                    coeff_view[:, col] = coeffs[:, src]
                    se_view[:, col] = se[:, src]
            handle = None
            for neuron_idx in range(self.numNeurons):
                eb = ax.errorbar(
                    x,
                    coeff_view[neuron_idx, :],
                    yerr=se_view[neuron_idx, :],
                    fmt=".",
                    linewidth=1.0,
                    markersize=6.0,
                    alpha=0.9,
                )
                if handle is None:
                    handle = eb.lines[0]
            if handle is not None:
                legend_handles.append(handle)
                if legendLabels is not None and fit_idx - 1 < len(legendLabels):
                    legend_labels.append(str(legendLabels[fit_idx - 1]))
                else:
                    legend_labels.append(f"\\lambda_{{{fit_idx}}}")

        ax.set_ylabel("Fit Coefficients")
        ax.set_xticks(x, sub_labels, rotation=90 if len(sub_labels) > 1 else 0)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.02)
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=10)
        ymin, ymax = ax.get_ylim()
        self.setCoeffRange(ymin, ymax)
        return ax

    def plotCoeffsWithoutHistory(
        self,
        fitNum: int | Sequence[int] | None = None,
        sortByEpoch: int = 0,
        plotSignificance: int = 1,
        handle=None,
    ):
        coeff_index, _, _ = self.getCoeffIndex(fitNum, sortByEpoch)
        return self.plotAllCoeffs(
            h=handle,
            fitNum=fitNum,
            plotSignificance=plotSignificance,
            subIndex=coeff_index.tolist() if coeff_index.size else [],
        )

    def plotHistCoeffs(
        self,
        fitNum: int | Sequence[int] | None = None,
        sortByEpoch: int = 0,
        plotSignificance: int = 1,
        handle=None,
    ):
        hist_index, _, _ = self.getHistIndex(fitNum, sortByEpoch)
        return self.plotAllCoeffs(
            h=handle,
            fitNum=fitNum,
            plotSignificance=plotSignificance,
            subIndex=hist_index.tolist() if hist_index.size else [],
        )

    def plotSummary(self, handle=None):
        fig = handle if handle is not None else plt.figure(figsize=(12.0, 7.0))
        fig.clear()
        gs = fig.add_gridspec(2, 4)
        coeff_ax = fig.add_subplot(gs[:, :2])
        self.plotAllCoeffs(h=coeff_ax, legendLabels=self.fitNames)
        coeff_ax.grid(False)
        coeff_ax.set_title("GLM Coefficients Across Neurons\nwith 95% CIs (* p<0.05)")

        ks_ax = fig.add_subplot(gs[0, 2:])
        ks_ax.boxplot(self.KSStats, labels=self.fitNames)
        ks_ax.set_ylabel("KS Statistics")
        ks_ax.set_title("KS Statistics Across Neurons")

        aic_ax = fig.add_subplot(gs[1, 2])
        self.boxPlot(self.getDiffAIC(1), diffIndex=1, h=aic_ax)
        aic_ax.set_ylabel("\\Delta AIC")
        aic_ax.set_title("Change in AIC Across Neurons")
        aic_ax.tick_params(axis="x", rotation=90)

        bic_ax = fig.add_subplot(gs[1, 3])
        self.boxPlot(self.getDiffBIC(1), diffIndex=1, h=bic_ax)
        bic_ax.set_ylabel("\\Delta BIC")
        bic_ax.set_title("Change in BIC Across Neurons")
        bic_ax.tick_params(axis="x", rotation=90)

        fig.tight_layout()
        return fig

    def boxPlot(self, X, diffIndex: int = 1, h=None, dataLabels=None, **kwargs):
        del kwargs
        ax = h if h is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if dataLabels is not None:
            labels = list(dataLabels)
        elif values.shape[1] == len(self.fitNames):
            labels = list(self.fitNames)
        elif values.shape[1] == max(len(self.fitNames) - 1, 1):
            labels = [
                f"{name} - {self.fitNames[diffIndex - 1]}"
                for idx, name in enumerate(self.fitNames, start=1)
                if idx != diffIndex
            ]
        else:
            labels = list(self.fitNames[: values.shape[1]])
        ax.boxplot(values, labels=labels)
        return ax

    def toStructure(self) -> dict[str, Any]:
        return {
            "fitResCell": FitResult.CellArrayToStructure(self.fitResCell),
            "numNeurons": self.numNeurons,
            "numResults": self.numResults,
            "maxNumIndex": self.maxNumIndex,
            "neuronNumbers": list(self.neuronNumbers),
            "fitNames": list(self.fitNames),
            "dev": self.dev.tolist(),
            "AIC": self.AIC.tolist(),
            "BIC": self.BIC.tolist(),
            "logLL": self.logLL.tolist(),
            "KSStats": self.KSStats.tolist(),
            "KSPvalues": self.KSPvalues.tolist(),
            "withinConfInt": self.withinConfInt.tolist(),
            "covLabels": [list(labels) for labels in getattr(self, "covLabels", [])],
            "uniqueCovLabels": list(getattr(self, "uniqueCovLabels", [])),
            "indicesToUniqueLabels": [
                [np.asarray(item, dtype=float).reshape(-1).tolist() for item in row]
                for row in getattr(self, "indicesToUniqueLabels", [])
            ]
            if getattr(self, "indicesToUniqueLabels", None)
            else [],
            "flatMask": np.asarray(getattr(self, "flatMask", np.zeros((0, 0, 0), dtype=float)), dtype=float).tolist(),
            "bAct": np.asarray(getattr(self, "bAct", np.zeros((0, 0, 0), dtype=float)), dtype=float).tolist(),
            "seAct": np.asarray(getattr(self, "seAct", np.zeros((0, 0, 0), dtype=float)), dtype=float).tolist(),
            "sigIndex": np.asarray(getattr(self, "sigIndex", np.zeros((0, 0, 0), dtype=float)), dtype=float).tolist(),
            "numCoeffs": int(getattr(self, "numCoeffs", 0)),
            "numResultsCoeffPresent": np.asarray(
                getattr(self, "numResultsCoeffPresent", np.zeros(0, dtype=float)),
                dtype=float,
            ).reshape(-1).tolist(),
            "coeffRange": [] if getattr(self, "coeffRange", None) in (None, []) else np.asarray(self.coeffRange, dtype=float).reshape(-1).tolist(),
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "FitSummary":
        fits = [FitResult.fromStructure(item) for item in structure.get("fitResCell", [])]
        summary = FitSummary(fits)
        summary.fitNames = [f"Fit {idx + 1}" for idx in range(summary.numResults)]
        return summary


class FitResSummary(FitSummary):
    """MATLAB-compatible alias for FitSummary."""


__all__ = ["FitResult", "FitSummary", "FitResSummary", "_SingleFit"]
