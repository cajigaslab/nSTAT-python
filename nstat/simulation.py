"""Standalone Poisson and log-linear CIF spike-train simulators.

This module provides the lightweight Python simulators used by tutorials,
notebooks, and unit tests:

- :func:`simulate_poisson_from_rate` — Bernoulli-thinned Poisson draws
  from an arbitrary time-varying rate ``r(t)`` (Hz) on the grid ``t``.
- :func:`simulate_cif_from_stimulus` — convenience wrapper that builds a
  log-linear CIF ``λ(t) = exp(β₀ + β₁ · x(t))`` (Hz) from a stimulus
  ``x(t)`` and a pair of GLM coefficients, then simulates spikes.

These are pure-Python counterparts to the Simulink-driven simulation
pipeline used in :mod:`nstat.cif` and :mod:`nstat.simulators`.  No
MATLAB counterpart at the module level — the underlying algorithm is the
standard Bernoulli-thinning approximation for an inhomogeneous Poisson
process (Lewis & Shedler 1979; Ogata 1981).  Time is in **seconds**,
rates in **Hz**.
"""
from __future__ import annotations

import numpy as np

from .core import nspikeTrain


# Backward-compatible alias used by earlier Python code.
SpikeTrain = nspikeTrain


def simulate_poisson_from_rate(
    time: np.ndarray,
    rate_hz: np.ndarray,
    rng: np.random.Generator | None = None,
) -> nspikeTrain:
    t = np.asarray(time, dtype=float).reshape(-1)
    r = np.asarray(rate_hz, dtype=float).reshape(-1)
    if t.shape[0] != r.shape[0]:
        raise ValueError("time and rate_hz length mismatch")
    if t.shape[0] < 2:
        return nspikeTrain(np.asarray([], dtype=float))

    if rng is None:
        rng = np.random.default_rng()

    dt = np.diff(t)
    dt = np.concatenate([dt, [dt[-1]]])
    p = 1.0 - np.exp(-np.clip(r, 0.0, np.inf) * dt)
    p = np.clip(p, 0.0, 1.0)
    keep = rng.random(t.shape[0]) < p
    return nspikeTrain(spikeTimes=t[keep])


def simulate_cif_from_stimulus(
    time: np.ndarray,
    stimulus: np.ndarray,
    beta0: float,
    beta1: float,
    rng: np.random.Generator | None = None,
) -> tuple[nspikeTrain, np.ndarray, np.ndarray]:
    """Simulate a spike train from a log-linear CIF driven by a stimulus.

    Computes the conditional intensity ``lambda(t) = exp(beta0 + beta1 * x(t))``
    in spikes/second, then draws spikes via Bernoulli thinning at the time-grid
    resolution implied by ``time``.

    Parameters
    ----------
    time : ndarray, shape (T,)
        Time vector in seconds.
    stimulus : ndarray, shape (T,)
        Stimulus values at each time sample.
    beta0 : float
        Log-baseline rate (intercept of the log-linear CIF).
    beta1 : float
        Stimulus coefficient.
    rng : np.random.Generator, optional
        Random generator.  Defaults to ``np.random.default_rng()``.

    Returns
    -------
    spike_train : nspikeTrain
        Simulated spike train.
    rate_hz : ndarray, shape (T,)
        The instantaneous CIF in spikes per second.
    log_rate : ndarray, shape (T,)
        The log-rate ``beta0 + beta1 * x(t)`` (useful for diagnostics).
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    x = np.asarray(stimulus, dtype=float).reshape(-1)
    if t.shape[0] != x.shape[0]:
        raise ValueError("time and stimulus length mismatch")
    log_rate = beta0 + beta1 * x
    rate_hz = np.exp(log_rate)
    spike_train = simulate_poisson_from_rate(t, rate_hz, rng=rng)
    return spike_train, rate_hz, log_rate


__all__ = [
    "SpikeTrain",
    "simulate_cif_from_stimulus",
    "simulate_poisson_from_rate",
]
