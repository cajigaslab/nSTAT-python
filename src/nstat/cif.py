"""Conditional intensity function models and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, gammaln


@dataclass(slots=True)
class CIFModel:
    """Generalized linear conditional intensity model.

    Parameters
    ----------
    coefficients:
        Model coefficients, one per feature.
    intercept:
        Scalar intercept term.
    link:
        `poisson` (log link) or `binomial` (logit link).
    """

    coefficients: np.ndarray
    intercept: float = 0.0
    link: str = "poisson"

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if self.coefficients.ndim != 1:
            raise ValueError("coefficients must be 1D")
        if self.link not in {"poisson", "binomial"}:
            raise ValueError("link must be 'poisson' or 'binomial'")

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        """Compute linear predictor ``eta = intercept + X @ coefficients``."""

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.coefficients.size:
            raise ValueError("X feature dimension mismatch")
        return self.intercept + X @ self.coefficients

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate point-wise intensity (Poisson) or probability (binomial)."""

        eta = self.linear_predictor(X)
        if self.link == "poisson":
            # Clip for numerical stability in high-magnitude regimes.
            return np.exp(np.clip(eta, -50.0, 50.0))
        return expit(eta)

    def eval_lambda_delta(self, X: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Evaluate ``lambda * dt`` for Poisson, or Bernoulli probability for binomial."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        lam = self.evaluate(X)
        if self.link == "poisson":
            return lam * float(dt)
        return lam

    def log_likelihood(self, y: np.ndarray, X: np.ndarray, dt: float = 1.0) -> float:
        """Return independent-bin log-likelihood under the configured link.

        Parameters
        ----------
        y:
            Observation vector.
            - Poisson: non-negative integer counts per bin.
            - Binomial: Bernoulli outcomes in ``{0, 1}``.
        X:
            Design matrix with shape ``(n_samples, n_features)``.
        dt:
            Bin width in seconds. Poisson intensity is interpreted in Hz, so
            expected counts are ``lambda * dt``.
        """

        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D vector")
        mu = self.evaluate(X)
        if mu.shape[0] != y.size:
            raise ValueError("X and y sample dimensions must match")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if self.link == "poisson":
            lam = np.clip(mu * dt, 1e-12, None)
            return float(np.sum(y * np.log(lam) - lam - gammaln(y + 1.0)))

        p = np.clip(mu, 1e-9, 1.0 - 1e-9)
        return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def simulate_by_thinning(self, time: np.ndarray, X: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate spike times on a fixed grid.

        Notes
        -----
        For Poisson models this performs exact simulation under a
        piecewise-constant intensity assumption over each time bin.
        For binomial models this performs one Bernoulli draw per bin.
        """

        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time must be a 1D grid with >=2 samples")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing")

        rng = rng or np.random.default_rng()
        values = self.evaluate(X)
        if values.shape[0] != time.size:
            raise ValueError("X must have one row per time sample")

        dt = np.diff(time)
        dt = np.concatenate([dt, [float(np.median(dt))]])
        if self.link == "poisson":
            expected_counts = np.clip(values * dt, 0.0, None)
            n_per_bin = rng.poisson(expected_counts)
            spikes: list[float] = []
            for i, n_spikes in enumerate(n_per_bin):
                if n_spikes <= 0:
                    continue
                bin_start = time[i]
                bin_end = bin_start + dt[i]
                # Uniform within-bin placement gives an exact sample for
                # homogeneous intensity inside each discretized interval.
                spikes.extend(rng.uniform(bin_start, bin_end, size=int(n_spikes)).tolist())
            if not spikes:
                return np.array([], dtype=float)
            return np.sort(np.asarray(spikes, dtype=float))

        p = np.clip(values, 0.0, 1.0)
        draws = rng.random(time.size) < p
        return time[draws]

    def compute_plot_params(self, X: np.ndarray) -> dict[str, float]:
        """Return compact summary stats useful for help/notebook plots."""

        vals = np.asarray(self.evaluate(X), dtype=float)
        return {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    def to_structure(self) -> dict[str, np.ndarray | float | str]:
        """Serialize model parameters to a plain structure."""

        return {
            "coefficients": self.coefficients.copy(),
            "intercept": float(self.intercept),
            "link": self.link,
        }

    @staticmethod
    def from_structure(payload: dict[str, np.ndarray | float | str]) -> "CIFModel":
        """Deserialize from :meth:`to_structure` payload."""

        return CIFModel(
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            intercept=float(payload["intercept"]),
            link=str(payload["link"]),
        )

    @staticmethod
    def simulate_cif_by_thinning_from_lambda(
        time: np.ndarray,
        lambda_values: np.ndarray,
        num_realizations: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        """Simulate spike times from explicit lambda(t) traces.

        This is a Python-native counterpart to MATLAB's
        `CIF.simulateCIFByThinningFromLambda` workflow.
        """

        time = np.asarray(time, dtype=float)
        lam = np.asarray(lambda_values, dtype=float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time must be a 1D grid with at least two samples")
        if lam.ndim != 1 or lam.shape[0] != time.size:
            raise ValueError("lambda_values must be 1D and match time length")
        if np.any(lam < 0.0):
            raise ValueError("lambda_values must be non-negative")
        if num_realizations <= 0:
            raise ValueError("num_realizations must be positive")

        rng = rng or np.random.default_rng()
        dt = np.diff(time)
        dt = np.concatenate([dt, [float(np.median(dt))]])
        expected = np.clip(lam * dt, 0.0, None)

        out: list[np.ndarray] = []
        for _ in range(num_realizations):
            n_per_bin = rng.poisson(expected)
            spikes: list[float] = []
            for i, n_spikes in enumerate(n_per_bin):
                if n_spikes <= 0:
                    continue
                t0 = float(time[i])
                t1 = t0 + float(dt[i])
                spikes.extend(rng.uniform(t0, t1, size=int(n_spikes)).tolist())
            if spikes:
                out.append(np.sort(np.asarray(spikes, dtype=float)))
            else:
                out.append(np.array([], dtype=float))
        return out
