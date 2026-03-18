from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, pearsonr

from .core import Covariate, nspikeTrain


def _ordered_unique(labels: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(label) for label in labels))


def _ensure_mathtext(label: str) -> str:
    """Wrap a label in ``$...$`` if it contains LaTeX commands but isn't already wrapped."""
    s = str(label)
    if not s:
        return s
    # Already wrapped in math delimiters — leave as-is
    if s.startswith("$") and s.endswith("$"):
        return s
    # Contains LaTeX commands (e.g. \lambda, \rho) — wrap in $...$
    if re.search(r"\\[a-zA-Z]", s):
        return f"${s}$"
    return s


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


def _autocorrelation(values: np.ndarray, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized autocorrelation (xcov/xcov[0]) for lags 1..max_lag.

    Matches MATLAB ``xcov`` normalization: ``rho(k) = xcov(k) / xcov(0)``.
    When *max_lag* is None (default) the full range is returned, matching MATLAB.
    """
    centered = np.asarray(values, dtype=float).reshape(-1) - float(np.mean(values))
    if centered.size < 2 or float(np.var(centered)) <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr / corr[0]
    lags = np.arange(corr.shape[0], dtype=float)
    end = corr.shape[0] - 1 if max_lag is None else int(min(max_lag, corr.shape[0] - 1))
    return lags[1 : end + 1], corr[1 : end + 1]


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

    U = 1.0 - np.exp(-np.clip(Z, -700, 700))  # FIX: clip to avoid overflow → -inf in uniforms
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
    pvalues = _extract_stat_component(stat, ("p", "p_values", "pvalues", "pValues"))
    if pvalues is not None:
        p_arr = np.asarray(pvalues, dtype=float).reshape(-1)
        out = np.zeros(coeffs.size, dtype=float)
        out[: min(coeffs.size, p_arr.size)] = (p_arr[: min(coeffs.size, p_arr.size)] < 0.05).astype(float)
        return out
    valid = np.isfinite(standard_errors) & (np.abs(standard_errors) > 0.0)
    out = np.zeros(coeffs.size, dtype=float)
    if np.any(valid):
        out[valid] = (np.abs(coeffs[valid] / standard_errors[valid]) >= 1.96).astype(float)
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
    """GLM fit results for one neuron across one or more model configs (Matlab ``FitResult``).

    Stores coefficients, deviance, AIC/BIC, log-likelihood, fitted λ signal,
    and KS-test diagnostics for each configuration in a
    :class:`~nstat.trial.ConfigCollection`.  Provides coefficient accessors,
    residual analysis, and Matlab-compatible plot methods.

    Parameters
    ----------
    neuralSpikeTrain : nspikeTrain or sequence of nspikeTrain
        The observed spike train(s) that were fitted.
    *args, **kwargs
        Positional / keyword construction matching the Matlab
        ``FitResult(nst, covLabels, numHist, …)`` signature, or
        the simplified ``FitResult(nst, lambdaCov, fits)`` form.

    See Also
    --------
    FitSummary : Aggregate summary across multiple neurons.
    Analysis.RunAnalysisForAllNeurons : Main entry point that produces
        ``FitResult`` objects.
    """

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
        """Rename the neuron on the underlying spike train(s)."""
        if isinstance(self.neuralSpikeTrain, nspikeTrain):
            self.neuralSpikeTrain.setName(str(name))
        elif isinstance(self.neuralSpikeTrain, Sequence):
            for train in self.neuralSpikeTrain:
                if hasattr(train, "setName"):
                    train.setName(str(name))
        self.neuronNumber = str(name)
        return self

    def mapCovLabelsToUniqueLabels(self):
        """Rebuild the unique-label map and ``flatMask`` from ``covLabels``."""
        self.uniqueCovLabels = _ordered_unique([label for labels in self.covLabels for label in labels])
        self.indicesToUniqueLabels = []
        self.flatMask = np.zeros((len(self.uniqueCovLabels), max(len(self.covLabels), 1)), dtype=int)
        for fit_idx, labels in enumerate(self.covLabels):
            indices = [self.uniqueCovLabels.index(label) + 1 for label in labels]
            self.indicesToUniqueLabels.append(indices)
            if indices:
                self.flatMask[np.asarray(indices, dtype=int) - 1, fit_idx] = 1
        return self

    def getSubsetFitResult(self, subfits) -> "FitResult":
        """Return a new ``FitResult`` with only the selected fit indices (1-based)."""
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
        """Append a new fit configuration's results (Matlab ``addParamsToFit``)."""
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

    def _rawCoeffs(self, fit_num: int = 1) -> np.ndarray:
        """Return the raw coefficient vector for *fit_num* (1-based)."""
        return self.b[fit_num - 1].copy()

    def getCoeffs(self, fit_num: int = 1) -> np.ndarray:
        """Return the coefficient vector for *fit_num* (1-based).

        In Matlab ``[coeffMat, labels, SEMat] = getCoeffs(fitObj, fitNum)``
        returns multiple outputs.  Use :meth:`getCoeffsWithLabels` to obtain
        the full ``(coeffMat, labels, SEMat)`` tuple.
        """
        return self._rawCoeffs(fit_num)

    def getHistCoeffs(self, fit_num: int = 1) -> np.ndarray:
        """Return the history-coefficient vector for *fit_num* (1-based).

        In Matlab ``[histMat, labels, SEMat] = getHistCoeffs(fitObj, fitNum)``
        returns multiple outputs.  Use :meth:`getHistCoeffsWithLabels` to
        obtain the full ``(histMat, labels, SEMat)`` tuple.
        """
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        if num_hist <= 0:
            return np.array([], dtype=float)
        return self._rawCoeffs(fit_num)[-num_hist:]

    def getHistCoeffsWithLabels(self, fit_num: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return ``(histMat, labels, SEMat)`` — Matlab multi-output form.

        Matlab: ``[histMat, labels, SEMat] = getHistCoeffs(fitObj, fitNum)``
        """
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        coeffs, labels, se = self.getCoeffsWithLabels(fit_num)
        if num_hist <= 0:
            return np.array([], dtype=float), [], np.array([], dtype=float)
        return coeffs[-num_hist:], labels[-num_hist:], se[-num_hist:]

    def getCoeffIndex(self, fit_num: int = 1, sortByEpoch: int = 0):
        """Return ``(indices, epochIds, numEpochs)`` for non-history coefficients."""
        del sortByEpoch
        labels = list(self.covLabels[fit_num - 1]) if fit_num - 1 < len(self.covLabels) else []
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        non_hist_count = max(len(labels) - num_hist, 0)
        coeff_index = np.arange(1, non_hist_count + 1, dtype=int)
        epoch_id = np.zeros(coeff_index.size, dtype=int)
        return coeff_index, epoch_id, 0

    def getHistIndex(self, fit_num: int = 1, sortByEpoch: int = 0):
        """Return ``(indices, epochIds, numEpochs)`` for history coefficients."""
        del sortByEpoch
        labels = list(self.covLabels[fit_num - 1]) if fit_num - 1 < len(self.covLabels) else []
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        if num_hist <= 0:
            return np.array([], dtype=int), np.array([], dtype=int), 0
        start = max(len(labels) - num_hist, 0)
        hist_index = np.arange(start + 1, len(labels) + 1, dtype=int)
        epoch_id = np.zeros(hist_index.size, dtype=int)
        return hist_index, epoch_id, 0

    def getParam(self, paramNames, fit_num: int = 1):
        """Return ``(coeffs, SE, significance)`` for named parameters."""
        names = [paramNames] if isinstance(paramNames, str) else list(paramNames)
        coeffs, labels, se = self.getCoeffsWithLabels(fit_num)
        sig = _extract_significance_mask(self.stats[fit_num - 1] if fit_num - 1 < len(self.stats) else None, coeffs, se)
        indices = [labels.index(name) for name in names if name in labels]
        return coeffs[indices], se[indices], sig[indices]

    def getCoeffsWithLabels(self, fit_num: int = 1) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return ``(coefficients, labels, standardErrors)`` for *fit_num*."""
        coeffs = self._rawCoeffs(fit_num)
        labels = list(self.covLabels[fit_num - 1]) if fit_num - 1 < len(self.covLabels) else [f"b_{idx + 1}" for idx in range(coeffs.size)]
        if coeffs.size == len(labels) + 1:
            labels = ["Intercept", *labels]
        elif coeffs.size != len(labels):
            labels = [f"b_{idx + 1}" for idx in range(coeffs.size)]
        se = _extract_standard_errors(self.stats[fit_num - 1] if fit_num - 1 < len(self.stats) else None, coeffs.size)
        return coeffs, labels, se

    def computePlotParams(self, fit_num: int | None = None):
        """Compute the aligned coefficient / SE / significance arrays for plotting."""
        del fit_num
        if not self.uniqueCovLabels:
            self.mapCovLabelsToUniqueLabels()
            return self.plotParams

        b_act = np.full((len(self.uniqueCovLabels), self.numResults), np.nan, dtype=float)
        se_act = np.full((len(self.uniqueCovLabels), self.numResults), np.nan, dtype=float)
        sig_index = np.zeros((len(self.uniqueCovLabels), self.numResults), dtype=float)
        for result_index in range(1, self.numResults + 1):
            coeffs, labels, se = self.getCoeffsWithLabels(result_index)
            sig = _extract_significance_mask(self.stats[result_index - 1] if result_index - 1 < len(self.stats) else None, coeffs, se)
            for coeff_value, coeff_se, coeff_sig, label in zip(coeffs, se, sig, labels, strict=False):
                if label not in self.uniqueCovLabels:
                    continue
                row = self.uniqueCovLabels.index(label)
                b_act[row, result_index - 1] = coeff_value
                se_act[row, result_index - 1] = coeff_se
                sig_index[row, result_index - 1] = coeff_sig
        self.plotParams = {
            "bAct": b_act,
            "seAct": se_act,
            "sigIndex": sig_index,
            "xLabels": list(self.uniqueCovLabels),
            "numResultsCoeffPresent": np.sum(np.isfinite(b_act), axis=1).astype(int),
        }
        return self.plotParams

    def getPlotParams(self):
        """Alias for :meth:`computePlotParams`."""
        return self.computePlotParams()

    def isValDataPresent(self) -> bool:
        """Return ``True`` if cross-validation data was stored."""
        if not self.XvalTime or not self.XvalData:
            return False
        for time in self.XvalTime:
            arr = np.asarray(time, dtype=float).reshape(-1)
            if arr.size >= 2 and arr[-1] > arr[0]:
                return True
        return False

    def plotValidation(self):
        """Plot validation fit results (if present)."""
        if self.validation is not None:
            return self.validation.plotResults()
        return None

    def mergeResults(self, other: "FitResult") -> "FitResult":
        """Concatenate another ``FitResult``'s configs into this one."""
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

    def _compute_diagnostics(self, fit_num: int = 1, *, dt_correction: int = 1) -> dict[str, np.ndarray | float]:
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
        Z, U, xAxis, KSSorted, _ = _matlab_compute_ks_arrays(self._primary_spike_train(), lambda_signal, dt_correction=dt_correction)
        z = np.asarray(Z[:, 0], dtype=float).reshape(-1) if np.asarray(Z).size else np.asarray([], dtype=float)
        uniforms = np.asarray(U[:, 0], dtype=float).reshape(-1) if np.asarray(U).size else np.asarray([], dtype=float)
        ideal = np.asarray(xAxis[:, 0], dtype=float).reshape(-1) if np.asarray(xAxis).size else np.asarray([], dtype=float)
        empirical = np.asarray(KSSorted[:, 0], dtype=float).reshape(-1) if np.asarray(KSSorted).size else np.asarray([], dtype=float)
        ci = np.full(ideal.size, 1.36 / np.sqrt(float(ideal.size)), dtype=float) if ideal.size else np.asarray([], dtype=float)
        # MATLAB's setKSStats (FitResult.m:1434) recomputes the KS stat
        # via kstest2(xAxis, KSSorted) — a two-sample KS test.  The
        # curve-level max deviation is kept separately for plotting.
        ks_curve_stat = float(np.max(np.abs(empirical - ideal))) if ideal.size else 1.0
        if ideal.size:
            different, ks_pvalue, ks_stat = _matlab_kstest2(ideal, empirical)
            within = float(not different)
        else:
            ks_stat = 1.0
            ks_pvalue = np.nan
            within = np.nan
        gaussianized = norm.ppf(np.clip(uniforms, 1e-6, 1.0 - 1e-6))
        lags, acf = _autocorrelation(gaussianized)
        acf_ci = 1.96 / np.sqrt(float(gaussianized.size)) if gaussianized.size else np.nan
        coeffs = self._rawCoeffs(fit_num)
        se = _extract_standard_errors(self.stats[fit_num - 1] if fit_num - 1 < len(self.stats) else None, coeffs.size)
        sig_mask = _extract_significance_mask(self.stats[fit_num - 1] if fit_num - 1 < len(self.stats) else None, coeffs, se)
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
            "coeff_se": se,
            "coeff_sig": sig_mask,
            "coeff_labels": np.asarray(coeff_labels, dtype=object),
        }
        self._diagnostic_cache[fit_num] = diagnostics
        # Write KS stat to the correct index (fit_num is 1-based).
        # We avoid calling setKSStats here because it overwrites the
        # multi-column Z/U/KSXAxis/KSSorted arrays and always writes
        # the ks_stat scalar to index 0.  Instead, write directly to
        # the correct row so that multi-fit sweeps accumulate properly.
        idx = fit_num - 1
        if idx < self.KSStats.shape[0]:
            self.KSStats[idx, 0] = ks_stat
        # For the last fit, store Z/U/etc. so legacy callers that
        # expect those arrays still see something useful.
        self.Z = np.asarray(z, dtype=float)[:, None] if z.size else np.array([], dtype=float)
        self.U = np.asarray(uniforms, dtype=float)[:, None] if uniforms.size else np.array([], dtype=float)
        self.KSXAxis = np.asarray(ideal, dtype=float)[:, None] if ideal.size else np.array([], dtype=float)
        self.KSSorted = np.asarray(empirical, dtype=float)[:, None] if empirical.size else np.array([], dtype=float)
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

    def computeKSStats(self, fit_num: int = 1, *, dt_correction: int = 1) -> dict[str, float]:
        """Return KS statistic, p-value, and within-CI flag for *fit_num*."""
        diag = self._compute_diagnostics(fit_num, dt_correction=dt_correction)
        return {
            "ks_stat": float(diag["ks_stat"]),
            "ks_pvalue": float(diag["ks_pvalue"]),
            "within_conf_int": float(diag["within_conf_int"]),
        }

    def computeInvGausTrans(self, fit_num: int = 1) -> np.ndarray:
        """Return Gaussianized (inverse-normal-transformed) rescaled ISIs."""
        return np.asarray(self._compute_diagnostics(fit_num)["gaussianized"], dtype=float)

    def computeFitResidual(self, fit_num: int = 1, *, windowSize: float | None = None) -> Covariate:
        """Compute the martingale residual M(t) = N(t) − Λ(t) (Matlab ``computeFitResidual``)."""
        time, rate_hz = self._lambda_series(fit_num)
        if time.size == 0:
            residual = Covariate([], [], "M(t_k)", "time", "s", "counts/bin", ["residual"])
            self.setFitResidual(residual)
            return residual

        window_size = float(windowSize) if windowSize is not None else (float(np.median(np.diff(time))) if time.size > 1 else 1.0)
        spike_train = self._primary_spike_train().nstCopy()
        spike_train.resample(1.0 / max(window_size, 1e-12))
        spike_train.setMinTime(float(time[0]))
        spike_train.setMaxTime(float(time[-1]))
        sum_spikes = spike_train.getSigRep(window_size, float(time[0]), float(time[-1]))
        window_times = np.linspace(float(time[0]), float(time[-1]), sum_spikes.time.size, dtype=float)

        # Use the label for the specific fit_num, not all labels
        all_labels = self.lambda_signal.dataLabels if getattr(self.lambda_signal, "dataLabels", None) else ["\\lambda"]
        idx = min(max(fit_num - 1, 0), len(all_labels) - 1)
        fit_label = [all_labels[idx]]
        lambda_signal = Covariate(
            time,
            rate_hz,
            "\\lambda(t)",
            self.lambda_signal.xlabelval,
            self.lambda_signal.xunits,
            self.lambda_signal.yunits,
            fit_label,
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
        """Evaluate λ(t) = exp(X·β) · sampleRate on *newData* (Matlab ``evalLambda``)."""
        coeffs = self._rawCoeffs(fit_num)
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

    def computeValLambda(self) -> tuple[Covariate, np.ndarray]:
        """Compute the conditional intensity on validation data (Matlab ``computeValLambda``).

        Returns
        -------
        lambda_val : Covariate
            The validation-set conditional intensity function.
        logLL : np.ndarray
            Log-likelihood for each fit configuration on the validation data.
        """
        if not self.XvalTime or not self.XvalData:
            raise ValueError("No validation data available (XvalData / XvalTime are empty)")

        time_vec = np.asarray(self.XvalTime[0], dtype=float).reshape(-1)
        lambda_data = np.zeros((time_vec.size, self.numResults), dtype=float)
        for i in range(self.numResults):
            xval = self.XvalData[i] if i < len(self.XvalData) else self.XvalData[0]
            lambda_data[:, i] = self.evalLambda(i + 1, xval)

        lambda_val = Covariate(
            time_vec,
            lambda_data,
            "\\lambda(t)",
            self.lambda_signal.xlabelval,
            self.lambda_signal.xunits,
            "Hz",
            list(self.lambda_signal.dataLabels),
        )

        delta = 1.0 / max(float(lambda_val.sampleRate), 1e-12)
        y = self.neuralSpikeTrain.getSigRep().dataToMatrix().reshape(-1)
        # Truncate or pad y to match validation lambda length
        n = min(y.size, lambda_data.shape[0])
        logLL_arr = np.zeros(self.numResults, dtype=float)
        for col in range(self.numResults):
            lam = np.maximum(lambda_data[:n, col] * delta, 1e-30)
            y_trunc = y[:n]
            logLL_arr[col] = float(np.sum(
                y_trunc * np.log(lam) + (1.0 - y_trunc) * np.log(np.maximum(1.0 - lam, 1e-30))
            ))

        return lambda_val, logLL_arr

    def plotResults(self, fit_num: int = 1, handle=None):
        """Matlab-matching 2x4 subplot layout with 5 diagnostic panels.

        Layout (matching Matlab ``subplot(2,4,...)``):
            [1,2]  KSPlot (double-wide)    [3] InvGausTrans  [4] SeqCorr
            [5,6]  plotCoeffs (double-wide) [7,8] plotResidual (double-wide)
        """
        import matplotlib.gridspec as gridspec

        fig = handle if handle is not None else plt.figure(figsize=(14.0, 8.0))
        fig.clear()
        gs = gridspec.GridSpec(2, 4, figure=fig)

        ax_ks = fig.add_subplot(gs[0, 0:2])
        ax_ig = fig.add_subplot(gs[0, 2])
        ax_sc = fig.add_subplot(gs[0, 3])
        ax_co = fig.add_subplot(gs[1, 0:2])
        ax_re = fig.add_subplot(gs[1, 2:4])

        self.KSPlot(fit_num=None, handle=ax_ks)
        # Add neuron number label (matching Matlab)
        ax_ks.text(
            0.45, 0.95, f"Neuron: {self.neuronNumber}",
            transform=ax_ks.transAxes, fontweight="bold", fontsize=10,
            verticalalignment="top",
        )
        self.plotInvGausTrans(fit_num=None, handle=ax_ig)
        self.plotSeqCorr(fit_num=None, handle=ax_sc)
        self.plotCoeffs(fit_num=None, handle=ax_co)
        self.plotResidual(fit_num=None, handle=ax_re)
        fig.tight_layout()
        return fig

    # MATLAB color cycle used by Analysis.colors: b, g, r, c, m, y, k
    _MATLAB_KS_COLORS = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "k"]

    def KSPlot(self, fit_num: int | list[int] | None = None, handle=None):
        """KS goodness-of-fit plot with 95 % confidence bands (Matlab ``KSPlot``).

        Parameters
        ----------
        fit_num : int, list of int, or None
            Which model(s) to plot.  ``None`` (default) plots all models
            (``1:numResults``), matching the MATLAB default behaviour.
            A single int plots one model; a list plots the specified subset.
        handle : matplotlib Axes, optional
            Axes to draw on.  A new figure is created when *None*.
        """
        if fit_num is None:
            fit_nums = list(range(1, self.numResults + 1))
        elif isinstance(fit_num, int):
            fit_nums = [fit_num]
        else:
            fit_nums = list(fit_num)

        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 4.0))[1]

        # Draw reference diagonal and confidence bands from the first model
        first_diag = self._compute_diagnostics(fit_nums[0])
        ideal_ref = np.asarray(first_diag["ks_ideal"], dtype=float)
        ci_ref = np.asarray(first_diag["ks_ci"], dtype=float)
        if ideal_ref.size:
            ax.plot([0.0, 1.0], [0.0, 1.0], "k-.", linewidth=1.0)
            ax.plot(ideal_ref, np.clip(ideal_ref + ci_ref, 0.0, 1.0), "r", linewidth=1.0)
            ax.plot(ideal_ref, np.clip(ideal_ref - ci_ref, 0.0, 1.0), "r", linewidth=1.0)

        # Plot each model's empirical CDF (matching MATLAB colour cycle)
        labels_for_legend: list[str] = []
        handles_for_legend: list[object] = []
        data_labels = list(self.lambda_signal.dataLabels) if getattr(self.lambda_signal, "dataLabels", None) else []
        for i, fn in enumerate(fit_nums):
            diag = self._compute_diagnostics(fn)
            ideal = np.asarray(diag["ks_ideal"], dtype=float)
            empirical = np.asarray(diag["ks_empirical"], dtype=float)
            color = self._MATLAB_KS_COLORS[i % len(self._MATLAB_KS_COLORS)]
            raw_label = data_labels[fn - 1] if fn - 1 < len(data_labels) else f"Model {fn}"
            label = _ensure_mathtext(raw_label)
            if ideal.size:
                h, = ax.plot(ideal, empirical, color=color, linewidth=2.0)
                handles_for_legend.append(h)
                labels_for_legend.append(label)

        if handles_for_legend:
            ax.legend(handles_for_legend, labels_for_legend, loc="lower right", fontsize=14)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Ideal Uniform CDF", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_ylabel("Empirical CDF", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_title("KS Plot of Rescaled ISIs\nwith 95% Confidence Intervals",
                      fontweight="bold", fontsize=11, fontname="Arial")
        ax.tick_params(length=6, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        return ax

    def plotResidual(self, fit_num: int | list[int] | None = None, handle=None):
        """Plot the martingale residual M(t) for one or more fits.

        Matches Matlab ``plotResidual``: plots all residuals with per-fit
        colours and a legend using ``lambda.dataLabels``.
        """
        if fit_num is None:
            fit_nums = list(range(1, self.numResults + 1))
        elif isinstance(fit_num, int):
            fit_nums = [fit_num]
        else:
            fit_nums = list(fit_num)

        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        _SEQ_COLORS = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "k"]
        data_labels = (
            list(self.lambda_signal.dataLabels)
            if getattr(self.lambda_signal, "dataLabels", None)
            else []
        )
        for i, fn in enumerate(fit_nums):
            residual = self.computeFitResidual(fn)
            color = _SEQ_COLORS[i % len(_SEQ_COLORS)]
            label = _ensure_mathtext(
                data_labels[fn - 1] if fn - 1 < len(data_labels) else f"Model {fn}"
            )
            ax.plot(
                np.asarray(residual.time, dtype=float),
                np.asarray(residual.data[:, 0], dtype=float),
                color=color, linewidth=1.0, label=label,
            )
        ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        if len(fit_nums) > 1:
            ax.legend(loc="upper right", fontsize=14)
        ax.set_xlabel("time [s]", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_ylabel("count residual", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_title("Point Process Residual",
                      fontweight="bold", fontsize=11, fontname="Arial")
        # Match MATLAB: symmetric y-axis with 10% margin
        ylims = ax.get_ylim()
        max_y = max(abs(ylims[0]), abs(ylims[1])) * 1.1
        ax.set_ylim(-max_y, max_y)
        ax.tick_params(length=6, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        return ax

    def plotInvGausTrans(self, fit_num: int | list[int] | None = None, handle=None):
        """Plot ACF of gaussianized rescaled ISIs with 95% CIs.

        Matlab: plotInvGausTrans computes X_j = Φ⁻¹(U_j) and plots the
        autocorrelation function of X_j with 95% confidence bounds.
        Supports multi-fit overlay with per-fit colours matching KS/SeqCorr.
        """
        if fit_num is None:
            fit_nums = list(range(1, self.numResults + 1))
        elif isinstance(fit_num, int):
            fit_nums = [fit_num]
        else:
            fit_nums = list(fit_num)

        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        data_labels = (
            list(self.lambda_signal.dataLabels)
            if getattr(self.lambda_signal, "dataLabels", None)
            else []
        )
        _SEQ_COLORS = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "k"]
        legend_handles: list[object] = []
        legend_labels: list[str] = []
        ci_val = None

        for i, fn in enumerate(fit_nums):
            diag = self._compute_diagnostics(fn)
            lags = np.asarray(diag["acf_lags"], dtype=float)
            acf = np.asarray(diag["acf_values"], dtype=float)
            color = _SEQ_COLORS[i % len(_SEQ_COLORS)]
            base_label = _ensure_mathtext(
                data_labels[fn - 1] if fn - 1 < len(data_labels) else f"Model {fn}"
            )
            if lags.size:
                h, = ax.plot(lags, acf, ".", color=color, markersize=4.0)
                legend_handles.append(h)
                legend_labels.append(base_label)
                if ci_val is None:
                    ci_val = float(diag["acf_ci"])

        # Plot 95% CI lines (solid red, matching MATLAB)
        if ci_val is not None:
            ax.axhline(ci_val, color="r", linewidth=1.0)
            ax.axhline(-ci_val, color="r", linewidth=1.0)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=14)
        ax.set_xlabel(r"$\Delta\tau$ [sec]", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"$ACF(\Phi^{-1}(u_n))$", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_title("Autocorrelation Function\nof Rescaled ISIs\nwith 95% CIs",
                      fontweight="bold", fontsize=11, fontname="Arial")
        ax.tick_params(length=6, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        return ax

    def plotSeqCorr(self, fit_num: int | list[int] | None = None, handle=None):
        """Plot U_j vs U_{j+1} scatter with correlation coefficient and p-value.

        Matlab: plotSeqCorr plots the sequential correlation scatter of
        U_j (uniform-transformed rescaled ISIs) to detect serial dependence.
        When multiple models are present, each is plotted with a different
        colour and a legend entry showing ``label, ρ=X.XX (p=Y.YY)``.

        Parameters
        ----------
        fit_num : int, list of int, or None
            Which model(s) to plot.  ``None`` (default) plots all models,
            matching the MATLAB default behaviour.
        handle : matplotlib Axes, optional
            Axes to draw on.  A new figure is created when *None*.
        """
        if fit_num is None:
            fit_nums = list(range(1, self.numResults + 1))
        elif isinstance(fit_num, int):
            fit_nums = [fit_num]
        else:
            fit_nums = list(fit_num)

        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        data_labels = (
            list(self.lambda_signal.dataLabels)
            if getattr(self.lambda_signal, "dataLabels", None)
            else []
        )
        _SEQ_COLORS = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "k"]
        legend_labels: list[str] = []
        legend_handles: list[object] = []

        for i, fn in enumerate(fit_nums):
            diag = self._compute_diagnostics(fn)
            u = np.asarray(diag.get("uniforms", []), dtype=float)
            base_label = _ensure_mathtext(
                data_labels[fn - 1] if fn - 1 < len(data_labels) else f"Model {fn}"
            )
            color = _SEQ_COLORS[i % len(_SEQ_COLORS)]
            if u.size > 1:
                uj = u[:-1]
                uj1 = u[1:]
                h, = ax.plot(uj, uj1, ".", color=color, markersize=4.0)
                # FIX: filter non-finite values before correlation
                finite = np.isfinite(uj) & np.isfinite(uj1)
                uj_f, uj1_f = uj[finite], uj1[finite]
                if uj_f.size > 2 and np.std(uj_f) > 0 and np.std(uj1_f) > 0:
                    rho, pval = pearsonr(uj_f, uj1_f)
                    label = f"{base_label}, $\\rho$={rho:.2g} (p={pval:.2g})"
                else:
                    label = base_label
                legend_handles.append(h)
                legend_labels.append(label)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=14)

        ax.set_title("Sequential Correlation of\nRescaled ISIs",
                      fontweight="bold", fontsize=11, fontname="Arial")
        ax.set_xlabel("$u_j$", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_ylabel("$u_{j+1}$", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(length=6, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        return ax

    def plotCoeffs(self, fit_num: int | list[int] | None = None, handle=None, plotSignificance: int = 1):
        """Plot GLM coefficients with error bars and significance markers.

        Matches Matlab FitResult.plotCoeffs: when *fit_num* is ``None``
        (default) all fits are overlaid with per-fit colours, errorbar
        plots with ±1 SE, and asterisks (*) above significant coefficients
        (p < 0.05).
        """
        if fit_num is None:
            fit_nums = list(range(1, self.numResults + 1))
        elif isinstance(fit_num, int):
            fit_nums = [fit_num]
        else:
            fit_nums = list(fit_num)

        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        _SEQ_COLORS = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "k"]
        ax.axhline(0.0, color="0.6", linewidth=1.0)

        # Collect all labels across fits to build a unified x-axis
        all_labels: list[str] = []
        for fn in fit_nums:
            diag = self._compute_diagnostics(fn)
            for lbl in np.asarray(diag["coeff_labels"], dtype=object):
                if lbl not in all_labels:
                    all_labels.append(lbl)
        label_to_x = {lbl: float(j + 1) for j, lbl in enumerate(all_labels)}
        xpos_all = np.arange(1, len(all_labels) + 1, dtype=float)

        for i, fn in enumerate(fit_nums):
            diag = self._compute_diagnostics(fn)
            coeffs = np.asarray(diag["coefficients"], dtype=float)
            se = np.asarray(diag["coeff_se"], dtype=float)
            sig = np.asarray(diag["coeff_sig"], dtype=float)
            fit_labels = list(np.asarray(diag["coeff_labels"], dtype=object))
            xpos = np.array([label_to_x[lbl] for lbl in fit_labels])
            color = _SEQ_COLORS[i % len(_SEQ_COLORS)]
            valid_se = np.where(np.isfinite(se), se, 0.0)
            ax.errorbar(xpos, coeffs, yerr=valid_se, fmt=".", color=color,
                         linewidth=1.0, markersize=8.0, capsize=3.0)
            if plotSignificance and np.any(sig > 0):
                ylims = ax.get_ylim()
                y_star = 0.8 * ylims[1] - i * 0.1
                sig_idx = xpos[sig.astype(bool)]
                ax.plot(sig_idx, np.full(sig_idx.size, y_star), "*", color=color, markersize=10.0)

        ax.set_xticks(xpos_all)
        ax.set_xticklabels(all_labels, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("GLM Fit Coefficients", fontname="Arial", fontsize=12, fontweight="bold")
        ax.set_title("GLM Coefficients with 95% CIs (* p<0.05)",
                      fontweight="bold", fontsize=11, fontname="Arial")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(length=6, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        return ax

    def plotCoeffsWithoutHistory(self, fit_num: int = 1, sortByEpoch: int = 0, plotSignificance: int = 1, handle=None):
        """Plot non-history (stimulus/baseline) coefficients only."""
        del sortByEpoch, plotSignificance
        coeffs, labels, _ = self.getCoeffsWithLabels(fit_num)
        num_hist = int(self.numHist[fit_num - 1]) if fit_num - 1 < len(self.numHist) else 0
        if num_hist > 0:
            coeffs = coeffs[:-num_hist]
            labels = labels[:-num_hist]
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        xpos = np.arange(coeffs.size, dtype=float)
        ax.axhline(0.0, color="0.6", linewidth=1.0)
        ax.plot(xpos, coeffs, "o-", color="tab:blue", linewidth=1.0)
        ax.set_xticks(xpos, labels, rotation=45, ha="right")
        ax.set_ylabel("coefficient value")
        ax.set_title("GLM Coefficients Without History")
        return ax

    def plotHistCoeffs(self, fit_num: int = 1, sortByEpoch: int = 0, plotSignificance: int = 1, handle=None):
        """Plot history-filter coefficients (Matlab ``plotHistCoeffs``)."""
        del sortByEpoch, plotSignificance
        coeffs, labels, _se = self.getHistCoeffsWithLabels(fit_num)
        if not labels:
            labels = [f"hist_{idx + 1}" for idx in range(coeffs.size)]
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(6.0, 3.5))[1]
        xpos = np.arange(coeffs.size, dtype=float)
        ax.axhline(0.0, color="0.6", linewidth=1.0)
        if coeffs.size:
            ax.plot(xpos, coeffs, "o-", color="tab:orange", linewidth=1.0)
            ax.set_xticks(xpos, labels, rotation=45, ha="right")
        ax.set_ylabel("history coefficient")
        ax.set_title("History Coefficients")
        return ax

    def setKSStats(self, Z, U, xAxis, KSSorted, ks_stat):
        """Store pre-computed KS-test arrays (Matlab ``setKSStats``)."""
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
        """Store pre-computed inverse-Gaussian transform statistics."""
        self.invGausStats = {"X": np.asarray(X, dtype=float), "rhoSig": rhoSig, "confBoundSig": confBoundSig}
        return self

    def setFitResidual(self, M):
        """Store the pre-computed fit residual ``Covariate``."""
        self.Residual = M
        return self

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (Matlab ``toStructure``)."""
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
            "XvalData": [
                np.asarray(item, dtype=float).tolist() if not isinstance(item, list) else item
                for item in self.XvalData
            ],
            "XvalTime": [np.asarray(item, dtype=float).tolist() for item in self.XvalTime],
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "FitResult":
        """Reconstruct a ``FitResult`` from a dict."""
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
            structure.get("XvalData", []),
            structure.get("XvalTime", []),
            structure.get("fitType", "poisson"),
        )

    @staticmethod
    def CellArrayToStructure(fitResObjCell):
        """Serialize a list of ``FitResult`` objects to a list of dicts."""
        return [fit.toStructure() for fit in fitResObjCell]


class FitSummary:
    """Population-level summary across multiple neurons (Matlab ``FitResSummary``).

    Aggregates AIC, BIC, log-likelihood, KS statistics, and coefficients
    from a collection of :class:`FitResult` objects, providing box-plots,
    coefficient histograms, and 2-D/3-D coefficient surfaces.

    Parameters
    ----------
    fit_results : FitResult or iterable of FitResult
        One or more per-neuron fit results to summarise.

    See Also
    --------
    FitResult : Per-neuron fit container.
    """

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
        """Return ΔAIC relative to config *idx* (1-based)."""
        if self.numResults > 1:
            keep = [col for col in range(self.AIC.shape[1]) if col != (idx - 1)]
            return self.AIC[:, keep] - self.AIC[:, [idx - 1]]
        return self.AIC.copy()

    def getDiffBIC(self, idx: int = 1) -> np.ndarray:
        """Return ΔBIC relative to config *idx* (1-based)."""
        if self.numResults > 1:
            keep = [col for col in range(self.BIC.shape[1]) if col != (idx - 1)]
            return self.BIC[:, keep] - self.BIC[:, [idx - 1]]
        return self.BIC.copy()

    def getDifflogLL(self, idx: int = 1) -> np.ndarray:
        """Return Δlog-likelihood relative to config *idx* (1-based)."""
        if self.numResults > 1:
            keep = [col for col in range(self.logLL.shape[1]) if col != (idx - 1)]
            return self.logLL[:, keep] - self.logLL[:, [idx - 1]]
        return self.logLL.copy()

    def mapCovLabelsToUniqueLabels(self):
        """Rebuild the union of covariate labels across all neurons."""
        self.uniqueCovLabels = _ordered_unique(
            [label for fit in self.fitResCell for labels in fit.covLabels for label in labels]
        )
        return self.uniqueCovLabels

    def setCoeffRange(self, minVal, maxVal):
        """Set the coefficient range used by ``binCoeffs``."""
        self.coeffMin = float(minVal)
        self.coeffMax = float(maxVal)
        return self

    def getCoeffs(self, fitNum: int = 1):
        """Return ``(coeffMat, labels, seMat)`` aligned to unique labels."""
        labels = self.uniqueCovLabels
        coeff_rows = []
        se_rows = []
        for fit in self.fitResCell:
            coeffs, fit_labels, se = fit.getCoeffsWithLabels(fitNum)
            row = np.full(len(labels), np.nan, dtype=float)
            se_row = np.full(len(labels), np.nan, dtype=float)
            for coeff, coeff_se, label in zip(coeffs, se, fit_labels, strict=False):
                if label in labels:
                    idx = labels.index(label)
                    row[idx] = coeff
                    se_row[idx] = coeff_se
            coeff_rows.append(row)
            se_rows.append(se_row)
        return np.asarray(coeff_rows, dtype=float), labels, np.asarray(se_rows, dtype=float)

    def getHistCoeffs(self, fitNum: int = 1):
        """Return ``(histMat, labels, seMat)`` for history coefficients."""
        labels = _ordered_unique(
            [label for fit in self.fitResCell for label in fit.covLabels[fitNum - 1][-int(fit.numHist[fitNum - 1]) :] if fitNum - 1 < len(fit.covLabels) and int(fit.numHist[fitNum - 1]) > 0]
        )
        if not labels:
            return np.zeros((self.numNeurons, 0), dtype=float), [], np.zeros((self.numNeurons, 0), dtype=float)
        coeff_rows = []
        se_rows = []
        for fit in self.fitResCell:
            coeffs, fit_labels, se_hist = fit.getHistCoeffsWithLabels(fitNum)
            if not fit_labels:
                fit_labels = list(fit.covLabels[fitNum - 1])[-coeffs.size :] if coeffs.size and fitNum - 1 < len(fit.covLabels) else []
                se_all = _extract_standard_errors(fit.stats[fitNum - 1] if fitNum - 1 < len(fit.stats) else None, fit._rawCoeffs(fitNum).size)
                se_hist = se_all[-coeffs.size :] if coeffs.size else np.array([], dtype=float)
            row = np.full(len(labels), np.nan, dtype=float)
            se_row = np.full(len(labels), np.nan, dtype=float)
            for coeff, coeff_se, label in zip(coeffs, se_hist, fit_labels, strict=False):
                if label in labels:
                    idx = labels.index(label)
                    row[idx] = coeff
                    se_row[idx] = coeff_se
            coeff_rows.append(row)
            se_rows.append(se_row)
        return np.asarray(coeff_rows, dtype=float), labels, np.asarray(se_rows, dtype=float)

    def getSigCoeffs(self, fitNum: int = 1):
        """Return (nNeurons × nCov) binary significance matrix."""
        coeff_mat, labels, se_mat = self.getCoeffs(fitNum)
        sig = np.zeros_like(coeff_mat, dtype=float)
        for row_idx, fit in enumerate(self.fitResCell):
            coeffs, fit_labels, se = fit.getCoeffsWithLabels(fitNum)
            mask = _extract_significance_mask(fit.stats[fitNum - 1] if fitNum - 1 < len(fit.stats) else None, coeffs, se)
            for label, value in zip(fit_labels, mask, strict=False):
                if label in labels:
                    sig[row_idx, labels.index(label)] = value
        return sig

    def binCoeffs(self, minVal=-12.0, maxVal=12.0, binSize=0.1):
        """Histogram of regression coefficients per covariate.

        Matches Matlab FitResSummary.binCoeffs: for each unique covariate,
        bins the significant coefficient values across all neurons/fits,
        normalizes to a PDF, and computes the fraction of times each
        covariate was significant.

        Returns
        -------
        N : (nBins, nCovariates) per-covariate normalized histograms (PDFs)
        edges : (nBins + 1,) bin edges
        percentSig : (nCovariates,) fraction of times each covariate was significant
        """
        edges = np.arange(float(minVal), float(maxVal) + float(binSize), float(binSize), dtype=float)
        if edges.size < 2:
            edges = np.array([float(minVal), float(maxVal)], dtype=float)

        # Build per-covariate data across all fits
        # bAct: (nNeurons, nCov), sigIdx: (nNeurons, nCov)
        coeff_mat, labels, se_mat = self.getCoeffs(1)  # (nNeurons, nCov)
        sig_mat = self.getSigCoeffs(1)  # (nNeurons, nCov) boolean

        nCov = len(labels)
        N = np.zeros((edges.size - 1, nCov), dtype=float)
        percentSig = np.zeros(nCov, dtype=float)

        for i in range(nCov):
            vals = coeff_mat[:, i]
            sig = sig_mat[:, i].astype(bool)
            valid = np.isfinite(vals)
            numPresent = float(np.sum(valid))
            # Take only significant values
            sig_vals = vals[sig & valid]
            Ntemp, _ = np.histogram(sig_vals, bins=edges)
            numSig = float(Ntemp.sum())
            percentSig[i] = numSig / max(numPresent, 1.0)
            if numSig > 0:
                N[:, i] = Ntemp.astype(float) / numSig  # normalize to PDF

        return N, edges, percentSig

    def plotIC(self, handle=None):
        """Plot AIC, BIC, and log-likelihood box-plots side by side."""
        fig = handle if handle is not None else plt.figure(figsize=(9.0, 3.5))
        fig.clear()
        axes = fig.subplots(1, 3)
        self.plotAIC(handle=axes[0])
        self.plotBIC(handle=axes[1])
        self.plotlogLL(handle=axes[2])
        fig.tight_layout()
        return fig

    def plotAIC(self, handle=None):
        """Box-plot of AIC across neurons (Matlab ``plotAIC``)."""
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 3.5))[1]
        ax.boxplot(self.AIC, tick_labels=self.fitNames)
        ax.set_ylabel("AIC")
        ax.set_title("AIC Across Neurons")
        return ax

    def plotBIC(self, handle=None):
        """Box-plot of BIC across neurons (Matlab ``plotBIC``)."""
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 3.5))[1]
        ax.boxplot(self.BIC, tick_labels=self.fitNames)
        ax.set_ylabel("BIC")
        ax.set_title("BIC Across Neurons")
        return ax

    def plotlogLL(self, handle=None):
        """Box-plot of log-likelihood across neurons (Matlab ``plotlogLL``)."""
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(5.0, 3.5))[1]
        ax.boxplot(self.logLL, tick_labels=self.fitNames)
        ax.set_ylabel("log likelihood")
        ax.set_title("log likelihood Across Neurons")
        return ax

    def plotResidualSummary(self, handle=None):
        """Overlay all neurons' martingale residuals (Matlab ``plotResidualSummary``)."""
        fig = handle if handle is not None else plt.figure(figsize=(8.0, 3.5))
        fig.clear()
        ax = fig.subplots(1, 1)
        for fit in self.fitResCell:
            residual = fit.computeFitResidual().dataToMatrix().reshape(-1)
            ax.plot(residual, alpha=0.6)
        ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        ax.set_title("Residual Summary")
        ax.set_ylabel("count residual")
        fig.tight_layout()
        return fig

    def plotSummary(self, handle=None):
        """Bar chart of mean AIC, BIC, and log-likelihood across configs."""
        fig = handle if handle is not None else plt.figure(figsize=(10.0, 4.5))
        fig.clear()
        axes = fig.subplots(1, 3)
        x = np.arange(self.numResults, dtype=float)
        labels = list(self.fitNames)
        for ax, values, title in zip(
            axes,
            (self.meanAIC, self.meanBIC, self.meanlogLL),
            ("AIC", "BIC", "log likelihood"),
            strict=False,
        ):
            ax.bar(x, np.asarray(values, dtype=float), color="tab:blue", alpha=0.8)
            ax.set_xticks(x, labels, rotation=30, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    def boxPlot(self, X, diffIndex: int = 1, h=None, dataLabels=None, **kwargs):
        """General-purpose box-plot of *X* columns with fit-name labels."""
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
            labels = [name for idx, name in enumerate(self.fitNames, start=1) if idx != diffIndex]
        else:
            labels = list(self.fitNames[: values.shape[1]])
        ax.boxplot(values, tick_labels=labels)
        return ax

    # ------------------------------------------------------------------
    # Coefficient plotting (match Matlab FitResSummary)
    # ------------------------------------------------------------------
    def plotAllCoeffs(self, fitNum: int | list[int] | None = None,
                      plotSignificance: bool = True,
                      subIndex: list[int] | None = None,
                      handle=None):
        """Errorbar plot of GLM coefficients across neurons (Matlab ``plotAllCoeffs``)."""
        if fitNum is None:
            fitNum = list(range(1, self.numResults + 1))
        if isinstance(fitNum, int):
            fitNum = [fitNum]
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(10, 5))[1]

        coeff_mat, labels, se_mat = self.getCoeffs(fitNum[0])
        if subIndex is not None:
            labels = [labels[i] for i in subIndex]
            coeff_mat = coeff_mat[:, subIndex]
            se_mat = se_mat[:, subIndex]

        x = np.arange(1, len(labels) + 1)
        for n_idx in range(self.numNeurons):
            ax.errorbar(x, coeff_mat[n_idx], yerr=se_mat[n_idx], fmt=".",
                        alpha=0.7, capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_ylabel("Fit Coefficients")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
        return ax

    def _is_hist_label(self, label: str) -> bool:
        """Return True if *label* looks like a history window term (e.g. ``[0.001,0.01]``)."""
        return bool(re.match(r"^\[", label))

    def getHistIndex(self, fitNum: int | list[int] | None = None) -> list[int]:
        """Return 0-based indices into *uniqueCovLabels* that are history terms."""
        if fitNum is None:
            fitNum = list(range(1, self.numResults + 1))
        if isinstance(fitNum, int):
            fitNum = [fitNum]
        coeff_mat, labels, _ = self.getCoeffs(fitNum[0])
        hist_indices: list[int] = []
        for idx, label in enumerate(labels):
            if self._is_hist_label(label):
                # Only include if at least one neuron has a non-NaN value
                if np.any(np.isfinite(coeff_mat[:, idx])):
                    hist_indices.append(idx)
        return hist_indices

    def getCoeffIndex(self, fitNum: int | list[int] | None = None) -> list[int]:
        """Return 0-based indices into *uniqueCovLabels* that are NOT history terms."""
        if fitNum is None:
            fitNum = list(range(1, self.numResults + 1))
        if isinstance(fitNum, int):
            fitNum = [fitNum]
        coeff_mat, labels, _ = self.getCoeffs(fitNum[0])
        hist_set = set(self.getHistIndex(fitNum))
        coeff_indices: list[int] = []
        for idx, label in enumerate(labels):
            if idx not in hist_set:
                if np.any(np.isfinite(coeff_mat[:, idx])):
                    coeff_indices.append(idx)
        return coeff_indices

    def plotCoeffsWithoutHistory(self, fitNum: int | list[int] | None = None,
                                 plotSignificance: bool = True,
                                 handle=None):
        """Plot coefficients excluding history terms (Matlab ``plotCoeffsWithoutHistory``)."""
        coeffIndex = self.getCoeffIndex(fitNum)
        return self.plotAllCoeffs(fitNum=fitNum, plotSignificance=plotSignificance,
                                  subIndex=coeffIndex, handle=handle)

    def plotHistCoeffs(self, fitNum: int | list[int] | None = None,
                       plotSignificance: bool = True,
                       handle=None):
        """Plot only the history coefficients (Matlab ``plotHistCoeffs``)."""
        histIndex = self.getHistIndex(fitNum)
        return self.plotAllCoeffs(fitNum=fitNum, plotSignificance=plotSignificance,
                                  subIndex=histIndex, handle=handle)

    def plot3dCoeffSummary(self, handle=None):
        """3D ribbon plot of binned coefficient distributions (Matlab ``plot3dCoeffSummary``)."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        N, edges, _percentSig = self.binCoeffs(-12, 12, 0.1)
        labels = self.uniqueCovLabels
        fig = plt.figure(figsize=(10, 7)) if handle is None else handle
        if hasattr(fig, "add_subplot"):
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig

        for i in range(N.shape[1] if N.ndim == 2 else 0):
            xs = edges[:-1]
            ys = np.full_like(xs, i)
            zs = N[:len(xs), i] if N.shape[0] > len(xs) else N[:, i]
            ax.bar(xs, zs, zs=i, zdir="y", alpha=0.6, width=(edges[1] - edges[0]))

        ax.set_xlabel("Coefficient value")
        ax.set_ylabel("Covariate index")
        ax.set_zlabel("Density")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        return ax

    def plot2dCoeffSummary(self, handle=None):
        """Stacked line plot of binned coefficient distributions (Matlab ``plot2dCoeffSummary``)."""
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(8, 6))[1]
        N, edges, percentSig = self.binCoeffs(-12, 12, 0.1)
        labels = self.uniqueCovLabels
        num_coeffs = N.shape[1] if N.ndim == 2 else 0

        for i in range(num_coeffs):
            offset = i + 1
            vals = N[:len(edges), i] if N.shape[0] >= len(edges) else N[:, i]
            ax.plot(edges[:len(vals)], vals + offset)

        ax.set_yticks(range(1, num_coeffs + 1))
        ax.set_yticklabels(labels[:num_coeffs], fontsize=6)
        # Annotate significance percentages
        for i in range(num_coeffs):
            if i < len(percentSig):
                pct = float(percentSig) if np.isscalar(percentSig) else float(percentSig[i]) if hasattr(percentSig, "__getitem__") else 0.0
                ax.annotate(f"{pct*100:.0f}%sig", xy=(0.98, (i + 1)),
                            xycoords=("axes fraction", "data"),
                            fontsize=6, ha="right")
        return ax

    def plotKSSummary(self, neurons: list[int] | None = None, handle=None):
        """Subplot grid of KS plots per neuron (Matlab ``plotKSSummary``)."""
        if neurons is None:
            neurons = list(range(self.numNeurons))
        n = len(neurons)
        if n <= 1:
            nrows, ncols = 1, 1
        elif n <= 2:
            nrows, ncols = 1, 2
        elif n <= 4:
            nrows, ncols = 2, 2
        elif n <= 8:
            nrows, ncols = 2, 4
        elif n <= 12:
            nrows, ncols = 3, 4
        elif n <= 16:
            nrows, ncols = 4, 4
        elif n <= 20:
            nrows, ncols = 5, 4
        elif n <= 24:
            nrows, ncols = 6, 4
        elif n <= 40:
            nrows, ncols = 10, 4
        else:
            nrows, ncols = 10, 10

        fig = handle if handle is not None else plt.figure(figsize=(3 * ncols, 2.5 * nrows))
        if hasattr(fig, "subplots"):
            fig.clear()
            axes = fig.subplots(nrows, ncols, squeeze=False)
        else:
            return fig

        for cnt, neuron_idx in enumerate(neurons):
            row, col = divmod(cnt, ncols)
            ax = axes[row][col]
            fit = self.fitResCell[neuron_idx]
            fit.KSPlot(handle=ax)
            ax.set_title(f"N{neuron_idx + 1}", fontsize=8)
            if cnt < n - 1:
                ax.get_legend().set_visible(False) if ax.get_legend() else None
                ax.set_xlabel("")
                ax.set_ylabel("")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

        # Hide unused subplots
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        return fig

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "fitResCell": FitResult.CellArrayToStructure(self.fitResCell),
            "numNeurons": self.numNeurons,
            "numResults": self.numResults,
            "fitNames": list(self.fitNames),
            "dev": self.dev.tolist(),
            "AIC": self.AIC.tolist(),
            "BIC": self.BIC.tolist(),
            "logLL": self.logLL.tolist(),
            "KSStats": self.KSStats.tolist(),
            "KSPvalues": self.KSPvalues.tolist(),
            "withinConfInt": self.withinConfInt.tolist(),
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "FitSummary":
        """Reconstruct a ``FitSummary`` from a dict."""
        fits = [FitResult.fromStructure(item) for item in structure.get("fitResCell", [])]
        return FitSummary(fits)


class FitResSummary(FitSummary):
    """MATLAB-compatible alias for FitSummary."""


__all__ = ["FitResult", "FitSummary", "FitResSummary", "_SingleFit"]
