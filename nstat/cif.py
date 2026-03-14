from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from .history import History
from .signal import Covariate
from .simulation import simulate_poisson_from_rate
from .trial import SpikeTrainCollection
from .nspikeTrain import nspikeTrain


_SYMPY_TRANSFORMS = standard_transformations + (convert_xor,)


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


def _prepare_uniform_matrix(
    values,
    num_realizations: int,
    num_draws: int,
    *,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        if arr.size < num_draws:
            raise ValueError(f"{name} must contain at least {num_draws} draws")
        return np.tile(arr.reshape(1, -1), (int(num_realizations), 1))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if arr.shape[0] == int(num_realizations) and arr.shape[1] >= int(num_draws):
        return arr
    if arr.shape[1] == int(num_realizations) and arr.shape[0] >= int(num_draws):
        return arr.T
    raise ValueError(
        f"{name} must have shape ({num_realizations}, >= {num_draws}) or "
        f"(>= {num_draws}, {num_realizations})"
    )


def _prepare_optional_uniform_rows(values, num_realizations: int, *, name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.tile(arr.reshape(1, -1), (int(num_realizations), 1))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if arr.shape[0] == int(num_realizations):
        return arr
    if arr.shape[1] == int(num_realizations):
        return arr.T
    raise ValueError(f"{name} must align with numRealizations={num_realizations}")


def _as_single_lambda_trace(lambda_covariate: Covariate) -> np.ndarray:
    lambda_data = np.asarray(lambda_covariate.data, dtype=float)
    if lambda_data.ndim == 1:
        return lambda_data.reshape(-1)
    if lambda_data.ndim == 2 and lambda_data.shape[1] == 1:
        return lambda_data[:, 0].reshape(-1)
    raise ValueError("simulateCIFByThinningFromLambda requires a single lambda trace")


def _ordered_independent_names(
    xnames: Sequence[str] | None,
    stim_names: Sequence[str] | None,
) -> tuple[str, ...]:
    ordered: list[str] = []
    for values in (stim_names or [], xnames or []):
        for raw in values:
            expr = str(raw).strip()
            if not expr:
                continue
            parsed = parse_expr(expr, transformations=_SYMPY_TRANSFORMS, evaluate=True)
            for symbol in sorted(parsed.free_symbols, key=lambda item: str(item)):
                name = str(symbol)
                if name not in ordered:
                    ordered.append(name)
    return tuple(ordered)


def _zero_row(width: int) -> np.ndarray:
    return np.zeros((1, int(width)), dtype=float)


def _zero_square(size: int) -> np.ndarray:
    return np.zeros((int(size), int(size)), dtype=float)


def _reshape_row(result, width: int) -> np.ndarray:
    if width == 0:
        return _zero_row(0)
    arr = np.asarray(result, dtype=float).reshape(-1)
    if arr.size != width:
        raise ValueError("Compiled CIF gradient width mismatch")
    return arr.reshape(1, width)


def _reshape_square(result, size: int) -> np.ndarray:
    if size == 0:
        return _zero_square(0)
    arr = np.asarray(result, dtype=float)
    if arr.ndim == 0:
        if size != 1:
            raise ValueError("Compiled CIF Hessian size mismatch")
        return np.asarray([[float(arr)]], dtype=float)
    arr = arr.reshape(size, size)
    return arr


def _compile_cif_surface(
    beta: np.ndarray,
    xnames: Sequence[str] | None,
    stim_names: Sequence[str] | None,
    fit_type: str,
    hist_coeffs: np.ndarray,
) -> dict[str, Any] | None:
    beta_vec = np.asarray(beta)
    if beta_vec.dtype == object or beta_vec.size == 0:
        return None
    beta_vec = np.asarray(beta_vec, dtype=float).reshape(-1)
    xnames_list = [str(name) for name in (xnames or [])]
    if len(xnames_list) != beta_vec.size:
        return None

    independent_names = _ordered_independent_names(xnames_list, stim_names)
    symbol_map = {name: sp.Symbol(name, real=True) for name in independent_names}
    stim_var_names = [str(name) for name in (stim_names or [])]
    for name in stim_var_names:
        symbol_map.setdefault(name, sp.Symbol(name, real=True))

    term_exprs = [
        parse_expr(str(name).strip(), local_dict=symbol_map, transformations=_SYMPY_TRANSFORMS, evaluate=True)
        for name in xnames_list
    ]
    independent_symbols = tuple(symbol_map[name] for name in independent_names)
    stim_symbols = tuple(symbol_map[name] for name in stim_var_names)

    eta = sum(sp.Float(float(coeff)) * expr for coeff, expr in zip(beta_vec, term_exprs, strict=True))

    hist_coeff_vec = np.asarray(hist_coeffs, dtype=float).reshape(-1)
    hist_symbols = tuple(sp.Symbol(f"h{idx}", real=True) for idx in range(hist_coeff_vec.size))
    gamma_symbols = tuple(sp.Symbol(f"g{idx}", real=True) for idx in range(hist_coeff_vec.size))
    hist_term = sum(sp.Float(float(coeff)) * symbol for coeff, symbol in zip(hist_coeff_vec, hist_symbols, strict=True))
    gamma_term = sum(symbol * hist_symbol for symbol, hist_symbol in zip(gamma_symbols, hist_symbols, strict=True))

    eta_hist = eta + hist_term
    eta_gamma = eta + gamma_term

    if str(fit_type).lower() == "binomial":
        lambda_expr = sp.exp(eta_hist) / (1 + sp.exp(eta_hist))
        lambda_gamma_expr = sp.exp(eta_gamma) / (1 + sp.exp(eta_gamma))
    elif str(fit_type).lower() == "poisson":
        lambda_expr = sp.exp(eta_hist)
        lambda_gamma_expr = sp.exp(eta_gamma)
    else:
        return None

    gradient_expr = sp.Matrix([sp.diff(lambda_expr, symbol) for symbol in stim_symbols]) if stim_symbols else sp.Matrix([])
    gradient_log_expr = sp.Matrix([sp.diff(sp.log(lambda_expr), symbol) for symbol in stim_symbols]) if stim_symbols else sp.Matrix([])
    jacobian_expr = sp.hessian(lambda_expr, stim_symbols) if stim_symbols else sp.Matrix([])
    jacobian_log_expr = sp.hessian(sp.log(lambda_expr), stim_symbols) if stim_symbols else sp.Matrix([])

    if gamma_symbols:
        gradient_gamma_expr = sp.Matrix([sp.diff(lambda_gamma_expr, symbol) for symbol in gamma_symbols])
        gradient_log_gamma_expr = sp.Matrix([sp.diff(sp.log(lambda_gamma_expr), symbol) for symbol in gamma_symbols])
        jacobian_gamma_expr = sp.hessian(lambda_gamma_expr, gamma_symbols)
        jacobian_log_gamma_expr = sp.hessian(sp.log(lambda_gamma_expr), gamma_symbols)
    else:
        gradient_gamma_expr = sp.Matrix([])
        gradient_log_gamma_expr = sp.Matrix([])
        jacobian_gamma_expr = sp.Matrix([])
        jacobian_log_gamma_expr = sp.Matrix([])

    args = independent_symbols + hist_symbols
    gamma_args = independent_symbols + hist_symbols + gamma_symbols

    return {
        "independent_names": independent_names,
        "stim_names": tuple(stim_var_names),
        "lambda_expr": lambda_expr,
        "lambda_gamma_expr": lambda_gamma_expr if gamma_symbols else None,
        "log_lambda_gamma_expr": sp.log(lambda_gamma_expr) if gamma_symbols else None,
        "gradient_expr": gradient_expr,
        "gradient_log_expr": gradient_log_expr,
        "jacobian_expr": jacobian_expr,
        "jacobian_log_expr": jacobian_log_expr,
        "gradient_gamma_expr": gradient_gamma_expr if gamma_symbols else None,
        "gradient_log_gamma_expr": gradient_log_gamma_expr if gamma_symbols else None,
        "jacobian_gamma_expr": jacobian_gamma_expr if gamma_symbols else None,
        "jacobian_log_gamma_expr": jacobian_log_gamma_expr if gamma_symbols else None,
        "lambda_fn": sp.lambdify(args, lambda_expr, modules="numpy"),
        "gradient_fn": sp.lambdify(args, gradient_expr, modules="numpy") if stim_symbols else None,
        "gradient_log_fn": sp.lambdify(args, gradient_log_expr, modules="numpy") if stim_symbols else None,
        "jacobian_fn": sp.lambdify(args, jacobian_expr, modules="numpy") if stim_symbols else None,
        "jacobian_log_fn": sp.lambdify(args, jacobian_log_expr, modules="numpy") if stim_symbols else None,
        "lambda_gamma_fn": sp.lambdify(gamma_args, lambda_gamma_expr, modules="numpy") if gamma_symbols else None,
        "log_lambda_gamma_fn": sp.lambdify(gamma_args, sp.log(lambda_gamma_expr), modules="numpy") if gamma_symbols else None,
        "gradient_gamma_fn": sp.lambdify(gamma_args, gradient_gamma_expr, modules="numpy") if gamma_symbols else None,
        "gradient_log_gamma_fn": sp.lambdify(gamma_args, gradient_log_gamma_expr, modules="numpy") if gamma_symbols else None,
        "jacobian_gamma_fn": sp.lambdify(gamma_args, jacobian_gamma_expr, modules="numpy") if gamma_symbols else None,
        "jacobian_log_gamma_fn": sp.lambdify(gamma_args, jacobian_log_gamma_expr, modules="numpy") if gamma_symbols else None,
    }


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
        rate = np.exp(np.clip(eta, -20.0, 20.0))  # Poisson log link: lambda = exp(eta)
        return cls(np.asarray(time, dtype=float).reshape(-1), rate, name)


class CIF:
    """Conditional Intensity Function for point-process modelling.

    Encapsulates the regression coefficients, variable names, link function,
    and optional spike-history terms that define a conditional intensity
    function (CIF).  Supports symbolic differentiation (gradient / Jacobian)
    for use in point-process adaptive filters and decoders.

    Parameters
    ----------
    beta : array_like or None
        Regression coefficients.
    Xnames : sequence of str or None
        Names of the variables in the order they appear in *beta*.
    stimNames : sequence of str or None
        Names of the subset of variables that define the stimulus.
    fitType : {'poisson', 'binomial'}, default ``'poisson'``
        Link function.  For Poisson: ``λΔ = exp(Xβ)``.
        For binomial: ``λΔ = exp(Xβ) / (1 + exp(Xβ))``.
    histCoeffs : array_like or None
        Coefficients for each history window defined in *historyObj*.
    historyObj : History or array_like or None
        History object (or window-times vector) defining the spiking-
        history basis.
    nst : nspikeTrain or None
        Spike train used to pre-compute history values.

    See Also
    --------
    History, DecodingAlgorithms
    """

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
        self.indepVars = list(_ordered_independent_names(self.varIn, self.stimVars))
        self.history = None
        self.historyMat = np.zeros((0, 0), dtype=float)
        self.spikeTrain = None
        self.lambdaDelta = None
        self.lambdaDeltaGamma = None
        self.LogLambdaDeltaGamma = None
        self.gradientLambdaDelta = None
        self.gradientLogLambdaDelta = None
        self.gradientLambdaDeltaGamma = None
        self.gradientLogLambdaDeltaGamma = None
        self.jacobianLambdaDelta = None
        self.jacobianLogLambdaDelta = None
        self.jacobianLambdaDeltaGamma = None
        self.jacobianLogLambdaDeltaGamma = None
        self._expression_surface = _compile_cif_surface(self.b, self.varIn, self.stimVars, self.fitType, self.histCoeffs)
        if self._expression_surface is not None:
            self.indepVars = list(self._expression_surface["independent_names"])
            self.lambdaDelta = self._expression_surface["lambda_expr"]
            self.lambdaDeltaGamma = self._expression_surface["lambda_gamma_expr"]
            self.LogLambdaDeltaGamma = self._expression_surface["log_lambda_gamma_expr"]
            self.gradientLambdaDelta = self._expression_surface["gradient_expr"]
            self.gradientLogLambdaDelta = self._expression_surface["gradient_log_expr"]
            self.gradientLambdaDeltaGamma = self._expression_surface["gradient_gamma_expr"]
            self.gradientLogLambdaDeltaGamma = self._expression_surface["gradient_log_gamma_expr"]
            self.jacobianLambdaDelta = self._expression_surface["jacobian_expr"]
            self.jacobianLogLambdaDelta = self._expression_surface["jacobian_log_expr"]
            self.jacobianLambdaDeltaGamma = self._expression_surface["jacobian_gamma_expr"]
            self.jacobianLogLambdaDeltaGamma = self._expression_surface["jacobian_log_gamma_expr"]
        if historyObj is not None:
            self.setHistory(historyObj)
        if nst is not None:
            self.setSpikeTrain(nst)

    def CIFCopy(self):
        """Return a deep copy of this CIF object."""
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
        """Attach a spike train and pre-compute the history matrix."""
        if not isinstance(spikeTrain, nspikeTrain):
            spikeTrain = getattr(spikeTrain, "nstCopy", lambda: spikeTrain)()
        self.spikeTrain = spikeTrain.nstCopy()
        if self.history is not None:
            self.historyMat = np.asarray(self.history.computeHistory(self.spikeTrain).dataToMatrix(), dtype=float)
        else:
            self.historyMat = np.zeros((0, 0), dtype=float)

    def setHistory(self, histObj) -> None:
        """Set the History object for this CIF.

        Parameters
        ----------
        histObj : History or array_like
            A ``History`` object, or a vector of window-times from which
            one will be created.
        """
        if isinstance(histObj, History):
            self.history = History(histObj.windowTimes, histObj.minTime, histObj.maxTime, histObj.name)
        elif isinstance(histObj, (np.ndarray, Sequence)) and not isinstance(histObj, (str, bytes)):
            self.history = History(histObj)
        else:
            raise ValueError("History can only be set by passing in a History Object or a vector of windowTimes")
        if self.spikeTrain is not None:
            self.historyMat = np.asarray(self.history.computeHistory(self.spikeTrain).dataToMatrix(), dtype=float)

    def _stimulus_values(self, stimVal) -> np.ndarray:
        stim = _as_1d_float(stimVal)
        if self._expression_surface is None:
            return stim
        expected = len(self._expression_surface["independent_names"])
        if stim.size != expected:
            raise ValueError(
                f"Expected {expected} independent variable values for CIF evaluation, received {stim.size}"
            )
        return stim

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

    def _surface_args(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None) -> tuple[float, ...]:
        stim = self._stimulus_values(stimVal)
        hist = self._history_values(time_index=time_index, nst=nst)
        return tuple(stim.tolist() + hist.tolist())

    def _surface_gamma_args(
        self,
        stimVal,
        time_index: int | None = None,
        nst: nspikeTrain | None = None,
        gamma=None,
    ) -> tuple[float, ...]:
        args = list(self._surface_args(stimVal, time_index=time_index, nst=nst))
        gamma_arr = _as_1d_float(gamma)
        if self.histCoeffs.size == 0:
            if gamma_arr.size:
                raise ValueError("gamma is only valid for history-dependent CIFs")
            return tuple(args)
        if gamma_arr.size == 1:
            gamma_arr = np.repeat(gamma_arr, self.histCoeffs.size)
        if gamma_arr.size != self.histCoeffs.size:
            raise ValueError("gamma must be scalar or align with histCoeffs")
        return tuple(args + gamma_arr.tolist())

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
        if self._expression_surface is not None and gamma is None:
            return float(np.asarray(self._expression_surface["lambda_fn"](*self._surface_args(stimVal, time_index=time_index, nst=nst)), dtype=float).reshape(-1)[0])
        if self._expression_surface is not None and gamma is not None and self._expression_surface["lambda_gamma_fn"] is not None:
            return float(
                np.asarray(
                    self._expression_surface["lambda_gamma_fn"](
                        *self._surface_gamma_args(stimVal, time_index=time_index, nst=nst, gamma=gamma)
                    ),
                    dtype=float,
                ).reshape(-1)[0]
            )
        eta, _, _, _ = self._eta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        if self.fitType == "binomial":
            return float(_sigmoid(np.asarray([eta], dtype=float))[0])
        if self.fitType == "poisson":
            return float(np.exp(np.clip(eta, -20.0, 20.0)))
        raise ValueError("fitType must be either 'poisson' or 'binomial'")

    def _gradient(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None, log: bool = False) -> np.ndarray:
        if self._expression_surface is not None and gamma is None:
            fn = self._expression_surface["gradient_log_fn" if log else "gradient_fn"]
            if fn is None:
                return _zero_row(0)
            width = len(self._expression_surface["stim_names"])
            return _reshape_row(fn(*self._surface_args(stimVal, time_index=time_index, nst=nst)), width)
        lambda_delta = self._lambda_delta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        _, stim_coeffs, _, _ = self._eta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        if self.fitType == "binomial":
            scale = 1.0 - lambda_delta if log else lambda_delta * (1.0 - lambda_delta)
            return (scale * stim_coeffs).reshape(1, -1)
        return stim_coeffs.reshape(1, -1) if log else (lambda_delta * stim_coeffs).reshape(1, -1)

    def _jacobian(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None, log: bool = False) -> np.ndarray:
        if self._expression_surface is not None and gamma is None:
            fn = self._expression_surface["jacobian_log_fn" if log else "jacobian_fn"]
            if fn is None:
                return _zero_square(0)
            size = len(self._expression_surface["stim_names"])
            return _reshape_square(fn(*self._surface_args(stimVal, time_index=time_index, nst=nst)), size)
        lambda_delta = self._lambda_delta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        _, stim_coeffs, _, _ = self._eta(stimVal, time_index=time_index, nst=nst, gamma=gamma)
        outer = np.outer(stim_coeffs, stim_coeffs)
        if self.fitType == "binomial":
            if log:
                return -lambda_delta * (1.0 - lambda_delta) * outer
            return lambda_delta * (1.0 - lambda_delta) * (1.0 - 2.0 * lambda_delta) * outer
        return np.zeros_like(outer) if log else lambda_delta * outer

    def evalLambdaDelta(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        """Evaluate λΔ at the given stimulus values.

        Parameters
        ----------
        stimVal : array_like
            Stimulus variable values.
        time_index : int or None
            1-based time index into the pre-computed history matrix.
        nst : nspikeTrain or None
            Spike train for on-the-fly history computation.

        Returns
        -------
        float
            Scalar value of λΔ.
        """
        return self._lambda_delta(stimVal, time_index=time_index, nst=nst)

    def evalGradient(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        """Gradient of λΔ with respect to the stimulus variables.

        Returns
        -------
        ndarray, shape (1, n_stim)
            Row vector of partial derivatives.
        """
        return self._gradient(stimVal, time_index=time_index, nst=nst)

    def evalGradientLog(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        """Gradient of log(λΔ) with respect to the stimulus variables.

        Returns
        -------
        ndarray, shape (1, n_stim)
            Row vector of partial derivatives.
        """
        return self._gradient(stimVal, time_index=time_index, nst=nst, log=True)

    def evalJacobian(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        """Hessian of λΔ with respect to the stimulus variables.

        Returns
        -------
        ndarray, shape (n_stim, n_stim)
            Second-derivative matrix.
        """
        return self._jacobian(stimVal, time_index=time_index, nst=nst)

    def evalJacobianLog(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None):
        """Hessian of log(λΔ) with respect to the stimulus variables.

        Returns
        -------
        ndarray, shape (n_stim, n_stim)
            Second-derivative matrix.
        """
        return self._jacobian(stimVal, time_index=time_index, nst=nst, log=True)

    def evalLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        """Evaluate λΔ with gamma-scaled history coefficients.

        Parameters
        ----------
        stimVal : array_like
            Stimulus variable values.
        time_index : int or None
            1-based time index into the history matrix.
        nst : nspikeTrain or None
            Spike train for on-the-fly history computation.
        gamma : array_like or None
            Scaling factors applied to the history coefficients.

        Returns
        -------
        float
            Scalar value of λΔ.
        """
        return self._lambda_delta(stimVal, time_index=time_index, nst=nst, gamma=gamma)

    def evalLogLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        """Evaluate log(λΔ) with gamma-scaled history coefficients.

        Returns
        -------
        float
            Scalar value of log(λΔ).
        """
        if self._expression_surface is not None and self._expression_surface["log_lambda_gamma_fn"] is not None:
            return float(
                np.asarray(
                    self._expression_surface["log_lambda_gamma_fn"](
                        *self._surface_gamma_args(stimVal, time_index=time_index, nst=nst, gamma=gamma)
                    ),
                    dtype=float,
                ).reshape(-1)[0]
            )
        return float(np.log(np.clip(self.evalLDGamma(stimVal, time_index=time_index, nst=nst, gamma=gamma), 1e-12, None)))

    def evalGradientLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        """Gradient of λΔ w.r.t. gamma (history-coefficient scaling)."""
        if self._expression_surface is not None and self._expression_surface["gradient_gamma_fn"] is not None:
            return _reshape_row(
                self._expression_surface["gradient_gamma_fn"](
                    *self._surface_gamma_args(stimVal, time_index=time_index, nst=nst, gamma=gamma)
                ),
                self.histCoeffs.size,
            )
        return self._gradient(stimVal, time_index=time_index, nst=nst, gamma=gamma)

    def evalGradientLogLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        """Gradient of log(λΔ) w.r.t. gamma (history-coefficient scaling)."""
        if self._expression_surface is not None and self._expression_surface["gradient_log_gamma_fn"] is not None:
            return _reshape_row(
                self._expression_surface["gradient_log_gamma_fn"](
                    *self._surface_gamma_args(stimVal, time_index=time_index, nst=nst, gamma=gamma)
                ),
                self.histCoeffs.size,
            )
        return self._gradient(stimVal, time_index=time_index, nst=nst, gamma=gamma, log=True)

    def evalJacobianLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        """Hessian of λΔ w.r.t. gamma (history-coefficient scaling)."""
        if self._expression_surface is not None and self._expression_surface["jacobian_gamma_fn"] is not None:
            return _reshape_square(
                self._expression_surface["jacobian_gamma_fn"](
                    *self._surface_gamma_args(stimVal, time_index=time_index, nst=nst, gamma=gamma)
                ),
                self.histCoeffs.size,
            )
        return self._jacobian(stimVal, time_index=time_index, nst=nst, gamma=gamma)

    def evalJacobianLogLDGamma(self, stimVal, time_index: int | None = None, nst: nspikeTrain | None = None, gamma=None):
        """Hessian of log(λΔ) w.r.t. gamma (history-coefficient scaling)."""
        if self._expression_surface is not None and self._expression_surface["jacobian_log_gamma_fn"] is not None:
            return _reshape_square(
                self._expression_surface["jacobian_log_gamma_fn"](
                    *self._surface_gamma_args(stimVal, time_index=time_index, nst=nst, gamma=gamma)
                ),
                self.histCoeffs.size,
            )
        return self._jacobian(stimVal, time_index=time_index, nst=nst, gamma=gamma, log=True)

    def isSymBeta(self) -> bool:
        """Return ``True`` if the coefficients contain symbolic expressions."""
        beta = np.asarray(self.b)
        if beta.dtype == object:
            return True
        return any(type(item).__module__.startswith("sympy") for item in beta.reshape(-1))

    def evaluate(self, design_matrix: np.ndarray, *, delta: float = 1.0, history_matrix: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the CIF on a full design matrix (vectorised).

        Parameters
        ----------
        design_matrix : ndarray, shape (T, n_vars)
            Design matrix with one row per time step.
        delta : float, default 1.0
            Bin width (seconds).  The returned rate is λΔ / Δ.
        history_matrix : ndarray or None
            Pre-computed history matrix, shape (T, n_hist).

        Returns
        -------
        ndarray, shape (T,)
            Firing rate in Hz.
        """
        x = np.asarray(design_matrix, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if self._expression_surface is not None and x.shape[1] == len(self._expression_surface["independent_names"]):
            if history_matrix is None:
                hist = np.zeros((x.shape[0], self.histCoeffs.size), dtype=float)
            else:
                hist = np.asarray(history_matrix, dtype=float)
                if hist.ndim == 1:
                    hist = hist[:, None]
                if hist.shape[1] != self.histCoeffs.size:
                    raise ValueError("history_matrix column count must match histCoeffs length")
            args = [x[:, idx] for idx in range(x.shape[1])] + [hist[:, idx] for idx in range(hist.shape[1])]
            lambda_delta = np.asarray(self._expression_surface["lambda_fn"](*args), dtype=float).reshape(-1)
            return lambda_delta / max(float(delta), 1e-12)
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
        """Evaluate the CIF and return the result as a :class:`Covariate`."""
        rate = self.evaluate(design_matrix, delta=delta)
        return Covariate(time, rate, name, "time", "s", "spikes/sec", [name])

    @staticmethod
    def simulateCIFByThinningFromLambda(
        lambda_covariate: Covariate,
        numRealizations: int = 1,
        maxTimeRes: float | None = None,
        *,
        seed: int | None = None,
        random_values: np.ndarray | None = None,
        thinning_values: np.ndarray | None = None,
        return_details: bool = False,
    ) -> SpikeTrainCollection:
        """Simulate spike trains from a given λ(t) via thinning.

        Uses the thinning (rejection) algorithm: propose spikes from a
        homogeneous Poisson process at the bound rate, then accept each
        candidate with probability λ(t) / λ_max.

        Parameters
        ----------
        lambda_covariate : Covariate
            Conditional intensity function time series (Hz).
        numRealizations : int, default 1
            Number of independent spike-train realisations.
        maxTimeRes : float or None
            Minimum inter-spike interval (seconds).  Spikes closer than
            this are merged.
        seed : int or None
            Random seed for reproducibility.
        random_values : ndarray or None
            Pre-drawn uniform random values for the proposal process.
        thinning_values : ndarray or None
            Pre-drawn uniform random values for thinning acceptance.
        return_details : bool, default False
            If ``True``, return ``(collection, details_dict)`` instead.

        Returns
        -------
        SpikeTrainCollection
            Collection of simulated spike trains.
        """
        if int(numRealizations) < 1:
            raise ValueError("numRealizations must be >= 1")

        lambda_trace = _as_single_lambda_trace(lambda_covariate)
        Tmax = float(lambda_covariate.maxTime)
        lambda_bound = float(np.max(lambda_trace)) if lambda_trace.size else 0.0
        proposal_count = int(np.ceil(lambda_bound * (1.5 * Tmax))) if lambda_bound > 0.0 and Tmax > 0.0 else 0

        arrival_uniforms = _prepare_uniform_matrix(
            random_values,
            int(numRealizations),
            proposal_count,
            name="random_values",
        )
        thinning_uniforms = _prepare_optional_uniform_rows(
            thinning_values,
            int(numRealizations),
            name="thinning_values",
        )
        rng = np.random.default_rng(seed)

        trains: list[nspikeTrain] = []
        details = {
            "lambda": lambda_trace.copy(),
            "time": np.asarray(lambda_covariate.time, dtype=float).reshape(-1).copy(),
            "lambda_bound": float(lambda_bound),
            "proposal_count": int(proposal_count),
            "arrival_uniforms": [],
            "interarrival_times": [],
            "candidate_spike_times": [],
            "lambda_ratio": [],
            "thinning_uniforms": [],
            "accepted_spike_times": [],
        }

        for realization in range(int(numRealizations)):
            if proposal_count <= 0:
                arrival = np.zeros(0, dtype=float)
                interarrival = np.zeros(0, dtype=float)
                candidate_spike_times = np.zeros(0, dtype=float)
                lambda_ratio = np.zeros(0, dtype=float)
                thinning = np.zeros(0, dtype=float)
                accepted_spike_times = np.zeros(0, dtype=float)
            else:
                arrival = (
                    np.asarray(arrival_uniforms[realization, :proposal_count], dtype=float).reshape(-1)
                    if arrival_uniforms is not None
                    else rng.random(proposal_count, dtype=float)
                )
                arrival = np.clip(arrival, np.finfo(float).tiny, 1.0)
                interarrival = -np.log(arrival) / lambda_bound
                candidate_spike_times = np.cumsum(interarrival)
                candidate_spike_times = candidate_spike_times[candidate_spike_times <= Tmax]

                if candidate_spike_times.size:
                    lambda_ratio = np.asarray(lambda_covariate.getValueAt(candidate_spike_times), dtype=float).reshape(-1)
                    lambda_ratio = np.clip(lambda_ratio / lambda_bound, 0.0, 1.0)
                    if thinning_uniforms is not None:
                        if thinning_uniforms.shape[1] < lambda_ratio.size:
                            raise ValueError(
                                "thinning_values must contain at least as many draws as surviving proposal times"
                            )
                        thinning = np.asarray(
                            thinning_uniforms[realization, : lambda_ratio.size],
                            dtype=float,
                        ).reshape(-1)
                    else:
                        thinning = rng.random(lambda_ratio.size, dtype=float)
                    thinning = np.clip(thinning, 0.0, 1.0)
                    accepted_spike_times = candidate_spike_times[lambda_ratio >= thinning]
                else:
                    lambda_ratio = np.zeros(0, dtype=float)
                    thinning = np.zeros(0, dtype=float)
                    accepted_spike_times = np.zeros(0, dtype=float)

            if maxTimeRes is not None and accepted_spike_times.size:
                accepted_spike_times = np.unique(
                    np.ceil(accepted_spike_times / float(maxTimeRes)) * float(maxTimeRes)
                )

            train = nspikeTrain(
                accepted_spike_times,
                name="1",
                minTime=lambda_covariate.minTime,
                maxTime=lambda_covariate.maxTime,
                makePlots=-1,
            )
            train.setName("1")
            trains.append(train)

            details["arrival_uniforms"].append(arrival.copy())
            details["interarrival_times"].append(interarrival.copy())
            details["candidate_spike_times"].append(candidate_spike_times.copy())
            details["lambda_ratio"].append(lambda_ratio.copy())
            details["thinning_uniforms"].append(thinning.copy())
            details["accepted_spike_times"].append(accepted_spike_times.copy())

        coll = SpikeTrainCollection(trains)
        coll.setMinTime(lambda_covariate.minTime)
        coll.setMaxTime(lambda_covariate.maxTime)
        if not return_details:
            return coll

        if int(numRealizations) == 1:
            detail_payload = {
                key: (value[0] if isinstance(value, list) else value)
                for key, value in details.items()
            }
            return coll, detail_payload
        return coll, details

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
        backend: str = "auto",
    ):
        """Simulate a point process via the thinning algorithm.

        Alias for :meth:`simulateCIF`.  See that method for full
        parameter documentation.
        """
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
            backend=backend,
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
        random_values: np.ndarray | None = None,
        return_details: bool = False,
        backend: str = "auto",
    ):
        """Simulate a point process from component kernels and inputs.

        Constructs λΔ from the input terms
        ``μ + stim ∗ inputStimSignal + hist ∗ spikeHistory + ens ∗ inputEnsSignal``
        and generates spike trains via Bernoulli draws at each time step.

        Parameters
        ----------
        mu : float
            Baseline (mean) log-rate of the process.
        hist : transfer-function-like or array_like
            History kernel convolved with the process's own spiking.
        stim : transfer-function-like or array_like
            Stimulus kernel convolved with *inputStimSignal*.
        ens : transfer-function-like or array_like
            Ensemble kernel convolved with *inputEnsSignal*.
        inputStimSignal : Covariate
            Stimulus time series.
        inputEnsSignal : Covariate
            Ensemble activity time series.
        numRealizations : int, default 1
            Number of independent realisations.
        simType : {'binomial', 'poisson'}, default ``'binomial'``
            Link function for computing λΔ.
        seed : int or None
            Random seed (Python backend only).
        return_lambda : bool, default False
            If ``True``, return ``(collection, lambda_array)``.
        random_values : ndarray or None
            Pre-drawn uniform random values for reproducibility
            (Python backend only).
        return_details : bool, default False
            If ``True``, return ``(collection, details_dict)``
            (Python backend only).
        backend : {'auto', 'matlab', 'python'}, default ``'auto'``
            Simulation backend.  ``'auto'`` uses MATLAB/Simulink when
            available and falls back to the native Python implementation
            with a :class:`~nstat.matlab_engine.MatlabFallbackWarning`.
            ``'matlab'`` forces Simulink (raises if unavailable).
            ``'python'`` forces the native implementation with no warning.

        Returns
        -------
        SpikeTrainCollection
            Simulated spike trains (or tuple if *return_lambda* /
            *return_details* is ``True``).
        """
        # ---- Backend selection ----
        from . import matlab_engine as _meng

        if backend == "auto":
            use_matlab = (
                _meng.is_matlab_available()
                and _meng.get_matlab_nstat_path() is not None
            )
        elif backend == "matlab":
            if not _meng.is_matlab_available():
                raise RuntimeError(
                    "backend='matlab' requested but MATLAB Engine is not "
                    "available.  Install MATLAB and the MATLAB Engine API "
                    "for Python, or use backend='auto' / backend='python'."
                )
            if _meng.get_matlab_nstat_path() is None:
                raise RuntimeError(
                    "backend='matlab' requested but the MATLAB nSTAT repo "
                    "could not be found.  Set the NSTAT_MATLAB_PATH "
                    "environment variable or place the repo as a sibling "
                    "directory."
                )
            use_matlab = True
        elif backend == "python":
            use_matlab = False
        else:
            raise ValueError("backend must be 'auto', 'matlab', or 'python'")

        if use_matlab:
            try:
                return CIF._simulateCIF_matlab(
                    mu, hist, stim, ens,
                    inputStimSignal, inputEnsSignal,
                    numRealizations, simType,
                    return_lambda=return_lambda,
                )
            except Exception:
                if backend == "matlab":
                    raise  # User explicitly asked for MATLAB — propagate
                # auto mode — fall back to Python
                _meng.warn_fallback()

        elif backend == "auto":
            # MATLAB not available — warn the user
            _meng.warn_fallback()

        # ---- Native Python path ----
        return CIF._simulateCIF_python(
            mu, hist, stim, ens,
            inputStimSignal, inputEnsSignal,
            numRealizations, simType,
            seed=seed,
            return_lambda=return_lambda,
            random_values=random_values,
            return_details=return_details,
        )

    # ------------------------------------------------------------------ #
    # MATLAB/Simulink backend
    # ------------------------------------------------------------------ #

    @staticmethod
    def _simulateCIF_matlab(
        mu, hist, stim, ens,
        inputStimSignal: Covariate,
        inputEnsSignal: Covariate,
        numRealizations: int = 1,
        simType: str = "binomial",
        *,
        return_lambda: bool = False,
    ):
        """Run the simulation through ``PointProcessSimulation.slx``."""
        from . import matlab_engine as _meng

        time = np.asarray(inputStimSignal.time, dtype=float).reshape(-1)
        dt = float(np.median(np.diff(time)))

        hist_kernel = _extract_kernel_coeffs(hist).reshape(-1)
        stim_input = np.asarray(inputStimSignal.data, dtype=float)
        if stim_input.ndim == 1:
            stim_input = stim_input[:, None]
        ens_input = np.asarray(inputEnsSignal.data, dtype=float)
        if ens_input.ndim == 1:
            ens_input = ens_input[:, None]

        stim_kernels = _extract_kernel_bank(stim, stim_input.shape[1])
        ens_kernels = _extract_kernel_bank(ens, ens_input.shape[1])

        spike_times_list, lambda_data = _meng.simulateCIF_via_simulink(
            mu=float(np.asarray(mu, dtype=float).reshape(-1)[0]),
            hist_kernel=hist_kernel,
            stim_kernel_bank=stim_kernels,
            ens_kernel_bank=ens_kernels,
            stim_time=time,
            stim_data=stim_input[:, 0],
            ens_time=np.asarray(inputEnsSignal.time, dtype=float).reshape(-1),
            ens_data=ens_input[:, 0],
            num_realizations=int(numRealizations),
            sim_type=str(simType).lower(),
            dt=dt,
        )

        trains = []
        for i, st in enumerate(spike_times_list):
            train = nspikeTrain(
                st, name=str(i + 1),
                minTime=float(time[0]), maxTime=float(time[-1]),
                makePlots=-1,
            )
            trains.append(train)

        from .trial import SpikeTrainCollection
        coll = SpikeTrainCollection(trains)
        coll.setMinTime(float(time[0]))
        coll.setMaxTime(float(time[-1]))

        if return_lambda:
            lambda_cov = Covariate(
                time, lambda_data,
                "\\lambda(t|H_t)", "time", "s", "Hz",
            )
            return coll, lambda_cov
        return coll

    # ------------------------------------------------------------------ #
    # Native Python backend
    # ------------------------------------------------------------------ #

    @staticmethod
    def _simulateCIF_python(
        mu, hist, stim, ens,
        inputStimSignal: Covariate,
        inputEnsSignal: Covariate,
        numRealizations: int = 1,
        simType: str = "binomial",
        *,
        seed: int | None = None,
        return_lambda: bool = False,
        random_values: np.ndarray | None = None,
        return_details: bool = False,
    ):
        """Pure-NumPy discrete-time Bernoulli simulation."""
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
        lambda_delta_data = np.zeros((time.size, int(numRealizations)), dtype=float)
        hist_effect_data = np.zeros((time.size, int(numRealizations)), dtype=float)
        eta_data = np.zeros((time.size, int(numRealizations)), dtype=float)
        if random_values is None:
            rng = np.random.default_rng(seed)
            draws = rng.random((time.size, int(numRealizations)))
        else:
            draws = np.asarray(random_values, dtype=float)
            if draws.shape != (time.size, int(numRealizations)):
                raise ValueError("random_values must have shape (len(time), numRealizations)")
        trains: list[nspikeTrain] = []
        spike_matrix = np.zeros((time.size, int(numRealizations)), dtype=float)
        mu_val = float(np.asarray(mu, dtype=float).reshape(-1)[0])

        for realization in range(int(numRealizations)):
            spikes = np.zeros(time.size, dtype=float)
            for idx in range(time.size):
                hist_effect = 0.0
                for lag, coeff in enumerate(hist_kernel, start=1):
                    if idx - lag >= 0:
                        hist_effect += float(coeff) * float(spikes[idx - lag])
                eta = mu_val + float(stim_drive[idx]) + float(ens_drive[idx]) + hist_effect
                hist_effect_data[idx, realization] = hist_effect
                eta_data[idx, realization] = eta
                if fit_type == "binomial":
                    lambda_delta = float(_sigmoid(np.asarray([eta], dtype=float))[0])
                    rate_hz = lambda_delta / max(dt, 1e-12)
                    spikes[idx] = float(draws[idx, realization] < lambda_delta)
                else:
                    rate_hz = float(np.exp(np.clip(eta, -20.0, 20.0)))
                    lambda_delta = 1.0 - np.exp(-rate_hz * dt)
                    spikes[idx] = float(draws[idx, realization] < np.clip(lambda_delta, 0.0, 1.0))
                lambda_data[idx, realization] = rate_hz
                lambda_delta_data[idx, realization] = lambda_delta
                spike_matrix[idx, realization] = spikes[idx]
            spike_times = time[spikes > 0.5]
            train = nspikeTrain(spike_times, name=str(realization + 1), minTime=float(time[0]), maxTime=float(time[-1]), makePlots=-1)
            trains.append(train)

        spikeTrainColl = SpikeTrainCollection(trains)
        spikeTrainColl.setMinTime(float(time[0]))
        spikeTrainColl.setMaxTime(float(time[-1]))
        lambda_cov = Covariate(time, lambda_data, "\\lambda(t|H_t)", "time", "s", "Hz")
        if return_details:
            details = {
                "time": time,
                "stimulus_drive": stim_drive.copy(),
                "ensemble_drive": ens_drive.copy(),
                "history_effect": hist_effect_data,
                "eta": eta_data,
                "lambda_delta": lambda_delta_data,
                "uniform_values": draws,
                "spike_indicator": spike_matrix,
            }
            if return_lambda:
                return spikeTrainColl, lambda_cov, details
            return spikeTrainColl, details
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
