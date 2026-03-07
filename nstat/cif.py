from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .history import History
from .signal import Covariate
from .simulation import simulate_poisson_from_rate
from .trial import SpikeTrainCollection
from .nspikeTrain import nspikeTrain


def _as_1d_float(values) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=float).reshape(-1)


def _extract_kernel_coeffs(model_like) -> np.ndarray:
    if model_like is None:
        return np.zeros(0, dtype=float)
    if isinstance(model_like, (int, float, np.integer, np.floating)):
        return np.asarray([float(model_like)], dtype=float)
    if isinstance(model_like, np.ndarray):
        return np.asarray(model_like, dtype=float).reshape(-1)
    if isinstance(model_like, Sequence) and not isinstance(model_like, (str, bytes)):
        return np.asarray(model_like, dtype=float).reshape(-1)
    if hasattr(model_like, "num"):
        num = np.asarray(getattr(model_like, "num"), dtype=float).reshape(-1)
        return num.copy()
    raise TypeError("CIF simulation kernels must be array-like, scalar, or transfer-function-like objects")


def _extract_kernel_bank(model_like, input_dim: int) -> list[np.ndarray]:
    if input_dim < 1:
        return []
    if model_like is None:
        return [np.zeros(1, dtype=float) for _ in range(input_dim)]
    if hasattr(model_like, "num") or isinstance(model_like, (int, float, np.integer, np.floating, np.ndarray)):
        coeffs = _extract_kernel_coeffs(model_like)
        if input_dim == 1:
            return [coeffs]
        if coeffs.size == input_dim:
            return [np.asarray([coeff], dtype=float) for coeff in coeffs]
        if coeffs.size == 1:
            return [coeffs.copy() for _ in range(input_dim)]
        raise ValueError("simulation kernels must align with the input dimension")
    if isinstance(model_like, Sequence) and not isinstance(model_like, (str, bytes)):
        items = list(model_like)
        if len(items) != input_dim:
            raise ValueError("simulation kernels must align with the input dimension")
        return [_extract_kernel_coeffs(item) for item in items]
    raise TypeError("simulation kernels must be array-like, scalar, sequence-aligned, or transfer-function-like objects")


def _compute_filtered_drive(inputs: np.ndarray, kernels: list[np.ndarray], output_length: int) -> np.ndarray:
    if output_length < 1:
        return np.zeros(0, dtype=float)
    if not kernels:
        return np.zeros(output_length, dtype=float)
    data = np.asarray(inputs, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] != len(kernels):
        raise ValueError("kernel bank must align with the input dimension")
    drive = np.zeros(output_length, dtype=float)
    for dim, kernel in enumerate(kernels):
        kernel_vec = np.asarray(kernel, dtype=float).reshape(-1)
        if kernel_vec.size == 0:
            continue
        drive += np.convolve(data[:, dim], kernel_vec, mode="full")[:output_length]
    return drive


def _check_kernel_sample_time(model_like, dt: float) -> None:
    if hasattr(model_like, "Ts"):
        ts = float(getattr(model_like, "Ts"))
        if not np.isclose(ts, dt):
            raise ValueError("History and Stimulus Transfer functions be discrete and have 'Ts' equal to 1/inputStimSignal.sampleRate")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))


@dataclass
class CIFModel:
    """Conditional intensity function abstraction used by standalone workflows."""

    time: np.ndarray
    rate_hz: np.ndarray
    name: str = "lambda"

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.rate_hz = np.asarray(self.rate_hz, dtype=float).reshape(-1)
        if self.time.shape[0] != self.rate_hz.shape[0]:
            raise ValueError("time and rate_hz length mismatch")

    def to_covariate(self) -> Covariate:
        return Covariate(self.time, self.rate_hz, self.name, "time", "s", "spikes/sec", [self.name])

    def simulate(self, num_realizations: int = 1, *, seed: int | None = None) -> SpikeTrainCollection:
        if num_realizations < 1:
            raise ValueError("num_realizations must be >= 1")
        rng = np.random.default_rng(seed)
        trains = []
        for i in range(num_realizations):
            st = simulate_poisson_from_rate(self.time, self.rate_hz, rng=rng)
            st.setName(str(i + 1))
            trains.append(st)
        return SpikeTrainCollection(trains)

    @classmethod
    def from_linear_terms(
        cls,
        time: np.ndarray,
        intercept: float,
        coefficients: np.ndarray,
        design_matrix: np.ndarray,
        dt: float,
        name: str = "lambda",
    ) -> "CIFModel":
        eta = intercept + np.asarray(design_matrix, dtype=float) @ np.asarray(coefficients, dtype=float)
        p = np.exp(np.clip(eta, -20.0, 20.0))
        p = p / (1.0 + p)
        rate = p / max(float(dt), 1e-12)
        return cls(np.asarray(time, dtype=float).reshape(-1), rate, name)


class CIF:
    """MATLAB-facing CIF object plus native Python simulation helpers."""

    def __init__(
        self,
        beta: Sequence[float] | np.ndarray | None = None,
        Xnames: Sequence[str] | None = None,
        stimNames: Sequence[str] | None = None,
        fitType: str = "poisson",
        histCoeffs: Sequence[float] | np.ndarray | None = None,
        historyObj=None,
        nst=None,
    ) -> None:
        self.b = _as_1d_float(beta)
        self.varIn = list(Xnames or [])
        self.stimVars = list(stimNames or [])
        self.fitType = str(fitType).lower()
        self.histCoeffs = _as_1d_float(histCoeffs)
        self.history = None
        self.historyMat = np.zeros((0, 0), dtype=float)
        self.spikeTrain = None
        if historyObj is not None:
            self.setHistory(historyObj)
        if nst is not None:
            self.setSpikeTrain(nst)

    def CIFCopy(self):
        copied = CIF(
            beta=np.asarray(self.b, dtype=float).copy(),
            Xnames=list(self.varIn),
            stimNames=list(self.stimVars),
            fitType=self.fitType,
            histCoeffs=np.asarray(self.histCoeffs, dtype=float).copy(),
        )
        if self.history is not None:
            copied.history = History(self.history.windowTimes, self.history.minTime, self.history.maxTime, self.history.name)
        if self.spikeTrain is not None:
            copied.setSpikeTrain(self.spikeTrain)
        elif self.historyMat.size:
            copied.historyMat = np.asarray(self.historyMat, dtype=float).copy()
        return copied

    def setSpikeTrain(self, spikeTrain) -> None:
        if not isinstance(spikeTrain, nspikeTrain):
            spikeTrain = getattr(spikeTrain, "nstCopy", lambda: spikeTrain)()
        self.spikeTrain = spikeTrain.nstCopy()
        if self.history is not None:
            self.historyMat = np.asarray(self.history.computeHistory(self.spikeTrain).dataToMatrix(), dtype=float)
        else:
            self.historyMat = np.zeros((0, 0), dtype=float)

    def setHistory(self, histObj) -> None:
        if isinstance(histObj, History):
            self.history = History(histObj.windowTimes, histObj.minTime, histObj.maxTime, histObj.name)
        elif isinstance(histObj, (np.ndarray, Sequence)) and not isinstance(histObj, (str, bytes)):
            self.history = History(histObj)
        else:
            raise ValueError("History can only be set by passing in a History Object or a vector of windowTimes")
        if self.spikeTrain is not None:
            self.historyMat = np.asarray(self.history.computeHistory(self.spikeTrain).dataToMatrix(), dtype=float)

    def _split_coefficients(self, stim_dim: int) -> tuple[float, np.ndarray]:
        coeffs = np.asarray(self.b, dtype=float).reshape(-1)
        if coeffs.size == stim_dim:
            return 0.0, coeffs.copy()
        if coeffs.size == stim_dim + 1:
            return float(coeffs[0]), coeffs[1:].copy()
        if coeffs.size == 1 and stim_dim == 0:
            return float(coeffs[0]), np.zeros(0, dtype=float)
        raise ValueError("stimulus design does not align with CIF coefficients")

    def _history_values(self, time_index: int | None = None, nst: nspikeTrain | None = None) -> np.ndarray:
        if self.history is None or self.histCoeffs.size == 0:
            return np.zeros(0, dtype=float)
        if nst is not None:
            hist = np.asarray(self.history.computeHistory(nst).dataToMatrix(), dtype=float)
            return hist[-1, :].reshape(-1)
        if self.historyMat.size == 0:
            return np.zeros(self.histCoeffs.size, dtype=float)
        if time_index is None:
            return self.historyMat[-1, :].reshape(-1)
        idx = max(int(time_index) - 1, 0)
        idx = min(idx, self.historyMat.shape[0] - 1)
        return self.historyMat[idx, :].reshape(-1)

    def _eta(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None) -> tuple[float, np.ndarray, np.ndarray, float]:
        stim = _as_1d_float(stimVal)
        intercept, stim_coeffs = self._split_coefficients(stim.size)
        hist_vals = self._history_values(time_index=time_index, nst=nst)
        hist_coeffs = self.histCoeffs.copy()
        if gamma is not None and hist_coeffs.size:
            gamma_arr = _as_1d_float(gamma)
            if gamma_arr.size == 1:
                hist_coeffs = hist_coeffs * float(gamma_arr[0])
            elif gamma_arr.size == hist_coeffs.size:
                hist_coeffs = hist_coeffs * gamma_arr
            else:
                raise ValueError("gamma must be scalar or align with histCoeffs")
        eta = intercept
        if stim_coeffs.size:
            eta += float(stim @ stim_coeffs)
        if hist_coeffs.size:
            eta += float(hist_vals @ hist_coeffs)
        return eta, stim_coeffs, hist_coeffs, intercept

    def _lambda_delta(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None) -> float:
        eta, _, _, _ = self._eta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        if self.fitType == "binomial":
            return float(_sigmoid(np.asarray([eta], dtype=float))[0])
        if self.fitType == "poisson":
            return float(np.exp(np.clip(eta, -20.0, 20.0)))
        raise ValueError("fitType must be either 'poisson' or 'binomial'")

    def _gradient(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None, log: bool = False) -> np.ndarray:
        lambda_delta = self._lambda_delta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        _, stim_coeffs, _, _ = self._eta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        if self.fitType == "binomial":
            scale = 1.0 - lambda_delta if log else lambda_delta * (1.0 - lambda_delta)
            return (scale * stim_coeffs).reshape(1, -1)
        return stim_coeffs.reshape(1, -1) if log else (lambda_delta * stim_coeffs).reshape(1, -1)

    def _jacobian(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None, log: bool = False) -> np.ndarray:
        lambda_delta = self._lambda_delta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        _, stim_coeffs, _, _ = self._eta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        outer = np.outer(stim_coeffs, stim_coeffs)
        if self.fitType == "binomial":
            if log:
                return -lambda_delta * (1.0 - lambda_delta) * outer
            return lambda_delta * (1.0 - lambda_delta) * (1.0 - 2.0 * lambda_delta) * outer
        return np.zeros_like(outer) if log else lambda_delta * outer

    def evalLambdaDelta(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        return self._lambda_delta(stimVal, time_index=time_index, nst=nst)

    def evalGradient(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        return self._gradient(stimVal, time_index=time_index, nst=nst)

    def evalGradientLog(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        return self._gradient(stimVal, time_index=time_index, nst=nst, log=True)

    def evalJacobian(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        return self._jacobian(stimVal, time_index=time_index, nst=nst)

    def evalJacobianLog(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        return self._jacobian(stimVal, time_index=time_index, nst=nst, log=True)

    def evalLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        return self._lambda_delta(stimVal, time_index=time_index, nst=nst, gamma=gamma)

    def evalLogLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        return float(np.log(np.clip(self.evalLDGamma(stimVal, time_index=time_index, nst=nst, gamma=gamma), 1e-12, None)))

    def evalGradientLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        return self._gradient(stimVal, time_index=time_index, nst=nst, gamma=gamma)

    def evalGradientLogLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        return self._gradient(stimVal, time_index=time_index, nst=nst, gamma=gamma, log=True)

    def evalJacobianLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        return self._jacobian(stimVal, time_index=time_index, nst=nst, gamma=gamma)

    def evalJacobianLogLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        return self._jacobian(stimVal, time_index=time_index, nst=nst, gamma=gamma, log=True)

    def isSymBeta(self) -> bool:
        beta = np.asarray(self.b)
        if beta.dtype == object:
            return True
        return any(type(item).__module__.startswith("sympy") for item in beta.reshape(-1))

    def evaluate(self, design_matrix: np.ndarray, *, delta: float = 1.0, history_matrix: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(design_matrix, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        intercept, stim_coeffs = self._split_coefficients(x.shape[1])
        eta = intercept + x @ stim_coeffs
        if history_matrix is not None and self.histCoeffs.size:
            hist = np.asarray(history_matrix, dtype=float)
            if hist.ndim == 1:
                hist = hist[:, None]
            if hist.shape[1] != self.histCoeffs.size:
                raise ValueError("history_matrix column count must match histCoeffs length")
            eta = eta + hist @ self.histCoeffs
        if self.fitType == "binomial":
            lambda_delta = _sigmoid(eta)
        elif self.fitType == "poisson":
            lambda_delta = np.exp(np.clip(eta, -20.0, 20.0))
        else:
            raise ValueError("fitType must be either 'poisson' or 'binomial'")
        return lambda_delta / max(float(delta), 1e-12)

    def to_covariate(self, time: Sequence[float], design_matrix: np.ndarray, *, delta: float = 1.0, name: str = "lambda") -> Covariate:
        rate = self.evaluate(design_matrix, delta=delta)
        return Covariate(time, rate, name, "time", "s", "spikes/sec", [name])

    @staticmethod
    def simulateCIFByThinningFromLambda(
        lambda_covariate: Covariate,
        numRealizations: int = 1,
        maxTimeRes: float | None = None,
        *,
        seed: int | None = None,
    ) -> SpikeTrainCollection:
        model = CIFModel(lambda_covariate.time, np.asarray(lambda_covariate.data, dtype=float).reshape(-1), getattr(lambda_covariate, "name", "lambda"))
        coll = model.simulate(num_realizations=numRealizations, seed=seed)
        if maxTimeRes is not None:
            rounded = []
            for idx in range(1, coll.numSpikeTrains + 1):
                train = coll.getNST(idx).nstCopy()
                spikes = np.unique(np.ceil(train.spikeTimes / float(maxTimeRes)) * float(maxTimeRes))
                rounded.append(nspikeTrain(spikes, name=train.name, minTime=lambda_covariate.minTime, maxTime=lambda_covariate.maxTime, makePlots=-1))
            coll = SpikeTrainCollection(rounded)
        coll.setMinTime(lambda_covariate.minTime)
        coll.setMaxTime(lambda_covariate.maxTime)
        return coll

    @staticmethod
    def simulateCIFByThinning(
        mu,
        hist,
        stim,
        ens,
        inputStimSignal: Covariate,
        inputEnsSignal: Covariate,
        numRealizations: int = 1,
        simType: str = "binomial",
        *,
        seed: int | None = None,
        return_lambda: bool = False,
    ):
        return CIF.simulateCIF(
            mu,
            hist,
            stim,
            ens,
            inputStimSignal,
            inputEnsSignal,
            numRealizations,
            simType,
            seed=seed,
            return_lambda=return_lambda,
        )

    @staticmethod
    def simulateCIF(
        mu,
        hist,
        stim,
        ens,
        inputStimSignal: Covariate,
        inputEnsSignal: Covariate,
        numRealizations: int = 1,
        simType: str = "binomial",
        *,
        seed: int | None = None,
        return_lambda: bool = False,
    ):
        if int(numRealizations) < 1:
            raise ValueError("numRealizations must be >= 1")
        time = np.asarray(inputStimSignal.time, dtype=float).reshape(-1)
        if time.size < 2:
            raise ValueError("inputStimSignal must contain at least two time points")
        ens_time = np.asarray(inputEnsSignal.time, dtype=float).reshape(-1)
        if ens_time.shape != time.shape or np.max(np.abs(ens_time - time)) > 1e-9:
            raise ValueError("inputStimSignal and inputEnsSignal must share the same time grid")

        dt = float(np.median(np.diff(time)))
        _check_kernel_sample_time(hist, dt)
        _check_kernel_sample_time(stim, dt)
        _check_kernel_sample_time(ens, dt)

        hist_kernel = _extract_kernel_coeffs(hist)
        hist_kernel = hist_kernel.reshape(-1)

        stim_input = np.asarray(inputStimSignal.data, dtype=float)
        ens_input = np.asarray(inputEnsSignal.data, dtype=float)
        if stim_input.ndim == 1:
            stim_input = stim_input[:, None]
        if ens_input.ndim == 1:
            ens_input = ens_input[:, None]
        stim_kernels = _extract_kernel_bank(stim, stim_input.shape[1])
        ens_kernels = _extract_kernel_bank(ens, ens_input.shape[1])
        stim_drive = _compute_filtered_drive(stim_input, stim_kernels, time.size)
        ens_drive = _compute_filtered_drive(ens_input, ens_kernels, time.size)

        fit_type = str(simType or "binomial").lower()
        if fit_type not in {"binomial", "poisson"}:
            raise ValueError("simType must be either poisson or binomial")

        lambda_data = np.zeros((time.size, int(numRealizations)), dtype=float)
        trains: list[nspikeTrain] = []
        rng = np.random.default_rng(seed)
        mu_val = float(np.asarray(mu, dtype=float).reshape(-1)[0])

        for realization in range(int(numRealizations)):
            spikes = np.zeros(time.size, dtype=float)
            for idx in range(time.size):
                hist_effect = 0.0
                for lag, coeff in enumerate(hist_kernel, start=1):
                    if idx - lag >= 0:
                        hist_effect += float(coeff) * float(spikes[idx - lag])
                eta = mu_val + float(stim_drive[idx]) + float(ens_drive[idx]) + hist_effect
                if fit_type == "binomial":
                    lambda_delta = float(_sigmoid(np.asarray([eta], dtype=float))[0])
                    rate_hz = lambda_delta / max(dt, 1e-12)
                    spikes[idx] = float(rng.random() < lambda_delta)
                else:
                    rate_hz = float(np.exp(np.clip(eta, -20.0, 20.0)))
                    lambda_delta = 1.0 - np.exp(-rate_hz * dt)
                    spikes[idx] = float(rng.random() < np.clip(lambda_delta, 0.0, 1.0))
                lambda_data[idx, realization] = rate_hz
            spike_times = time[spikes > 0.5]
            train = nspikeTrain(spike_times, name=str(realization + 1), minTime=float(time[0]), maxTime=float(time[-1]), makePlots=-1)
            trains.append(train)

        spikeTrainColl = SpikeTrainCollection(trains)
        spikeTrainColl.setMinTime(float(time[0]))
        spikeTrainColl.setMaxTime(float(time[-1]))
        lambda_cov = Covariate(time, lambda_data, "\\lambda(t|H_t)", "time", "s", "Hz")
        return (spikeTrainColl, lambda_cov) if return_lambda else spikeTrainColl

    @staticmethod
    def from_linear_terms(
        time: np.ndarray,
        intercept: float,
        coefficients: np.ndarray,
        design_matrix: np.ndarray,
        dt: float,
        name: str = "lambda",
    ) -> Covariate:
        return CIFModel.from_linear_terms(time, intercept, coefficients, design_matrix, dt, name).to_covariate()


__all__ = ["CIFModel", "CIF"]
