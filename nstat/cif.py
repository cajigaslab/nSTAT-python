from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .signal import Covariate
from .simulation import simulate_poisson_from_rate
from .trial import SpikeTrainCollection


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
    """MATLAB-facing CIF object plus static convenience APIs."""

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
        self.b = np.asarray(beta if beta is not None else [], dtype=float).reshape(-1)
        self.varIn = list(Xnames or [])
        self.stimVars = list(stimNames or [])
        self.fitType = str(fitType)
        self.histCoeffs = np.asarray(histCoeffs if histCoeffs is not None else [], dtype=float).reshape(-1)
        self.history = historyObj
        self.spikeTrain = None if nst is None else getattr(nst, "nstCopy", lambda: nst)()

    def evaluate(self, design_matrix: np.ndarray, *, delta: float = 1.0, history_matrix: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(design_matrix, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        beta = self.b
        if x.shape[1] != beta.size:
            raise ValueError("design_matrix column count must match number of CIF coefficients")
        eta = x @ beta
        if history_matrix is not None and self.histCoeffs.size:
            hist = np.asarray(history_matrix, dtype=float)
            if hist.ndim == 1:
                hist = hist[:, None]
            if hist.shape[1] != self.histCoeffs.size:
                raise ValueError("history_matrix column count must match histCoeffs length")
            eta = eta + hist @ self.histCoeffs
        if self.fitType == "poisson":
            lambda_delta = np.exp(np.clip(eta, -20.0, 20.0))
        elif self.fitType == "binomial":
            exp_eta = np.exp(np.clip(eta, -20.0, 20.0))
            lambda_delta = exp_eta / (1.0 + exp_eta)
        else:
            raise ValueError("fitType must be either 'poisson' or 'binomial'")
        return lambda_delta / max(float(delta), 1e-12)

    def to_covariate(self, time: Sequence[float], design_matrix: np.ndarray, *, delta: float = 1.0, name: str = "lambda") -> Covariate:
        rate = self.evaluate(design_matrix, delta=delta)
        return Covariate(time, rate, name, "time", "s", "spikes/sec", [name])

    @staticmethod
    def simulateCIFByThinningFromLambda(lambda_covariate: Covariate, numRealizations: int = 1) -> SpikeTrainCollection:
        model = CIFModel(lambda_covariate.time, lambda_covariate.data[:, 0], getattr(lambda_covariate, "name", "lambda"))
        return model.simulate(num_realizations=numRealizations)

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
