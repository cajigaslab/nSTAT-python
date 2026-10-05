"""Higher-level point-process and network simulation result containers.

This module provides richer simulation outputs than the bare-bones
generators in :mod:`nstat.simulation`.  Each result dataclass carries
the time grid, the rate, the realised spikes, and the intermediate
quantities (latent drive, ``λΔ`` per bin, history / ensemble effects,
uniform variates) that callers may want to inspect or reuse for
unit-test reproducibility.

Exported symbols
----------------
- :class:`PointProcessSimulation` — single-neuron Poisson-thinning
  output bundle.
- :class:`NetworkSimulationResult` — multi-neuron network simulation
  output bundle (carries adjacency, history/ensemble kernels, etc.).
- :func:`simulate_point_process` and friends — generators that return
  the dataclasses above.

No direct MATLAB counterpart; the MATLAB toolbox returns plain ``struct``
arrays instead.  Time is in **seconds**, rates in **Hz**.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spikes import SpikeTrain, SpikeTrainCollection


@dataclass
class PointProcessSimulation:
    time: np.ndarray
    rate_hz: np.ndarray
    spikes: SpikeTrain
    lambda_delta: np.ndarray | None = None
    spike_indicator: np.ndarray | None = None
    uniform_values: np.ndarray | None = None


@dataclass
class NetworkSimulationResult:
    time: np.ndarray
    latent_drive: np.ndarray
    lambda_delta: np.ndarray
    spikes: SpikeTrainCollection
    actual_network: np.ndarray
    history_kernel: np.ndarray
    stimulus_kernel: np.ndarray
    ensemble_kernel: np.ndarray
    baseline_mu: np.ndarray
    eta: np.ndarray | None = None
    history_effect: np.ndarray | None = None
    ensemble_effect: np.ndarray | None = None
    spike_indicator: np.ndarray | None = None
    uniform_values: np.ndarray | None = None


def simulate_point_process(
    time: np.ndarray,
    rate_hz: np.ndarray,
    *,
    seed: int | None = None,
    uniform_values: np.ndarray | None = None,
) -> PointProcessSimulation:
    """Simulate one spike train from a sampled rate by Bernoulli thinning.

    In each time bin the spike probability is ``1 - exp(-rate_hz * dt)``
    (``dt`` is the local bin width; the last bin reuses the previous width).
    A spike is emitted, stamped at the bin's time value, when a uniform draw
    falls below that probability.

    Parameters
    ----------
    time : array_like, shape (n,)
        Monotonic time grid in seconds.
    rate_hz : array_like, shape (n,)
        Conditional intensity in Hz on the same grid.  Negative values are
        treated as zero.
    seed : int, optional
        Seed for ``numpy.random.default_rng`` when ``uniform_values`` is not
        given.
    uniform_values : array_like, shape (n,), optional
        Pre-drawn uniform variates in ``[0, 1)``, for deterministic
        reproduction.  When given, ``seed`` is ignored.

    Returns
    -------
    PointProcessSimulation
        Bundle with the time grid, rate, realised ``spikes``, per-bin spike
        probability ``lambda_delta``, binary ``spike_indicator`` and the
        ``uniform_values`` actually used.  When fewer than two time samples
        are given, the spike train is empty.

    Raises
    ------
    ValueError
        If ``time`` and ``rate_hz`` differ in length, or ``uniform_values``
        does not match the length of ``time``.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    r = np.asarray(rate_hz, dtype=float).reshape(-1)
    if t.shape[0] != r.shape[0]:
        raise ValueError("time and rate_hz length mismatch")
    if t.shape[0] < 2:
        return PointProcessSimulation(t, r, SpikeTrain(np.array([], dtype=float)))

    dt = np.diff(t)
    dt = np.concatenate([dt, [dt[-1]]])
    # Case C: use -expm1 for accurate 1 - exp(-x) at small x*dt
    # (preserves precision at low rates / small bins).
    p = -np.expm1(-np.clip(r, 0.0, np.inf) * dt)
    p = np.clip(p, 0.0, 1.0)

    if uniform_values is None:
        rng = np.random.default_rng(seed)
        draws = rng.random(t.shape[0])
    else:
        draws = np.asarray(uniform_values, dtype=float).reshape(-1)
        if draws.shape[0] != t.shape[0]:
            raise ValueError("uniform_values must match the length of time")
    keep = draws < p
    return PointProcessSimulation(
        t,
        r,
        SpikeTrain(t[keep]),
        lambda_delta=p,
        spike_indicator=keep.astype(float),
        uniform_values=draws,
    )


def simulate_two_neuron_network(
    duration_s: float = 50.0,
    dt: float = 0.001,
    baseline_mu: tuple[float, float] = (-3.0, -3.0),
    history_kernel: tuple[float, ...] = (-4.0, -2.0, -1.0),
    stimulus_kernel: tuple[float, float] = (1.0, -1.0),
    ensemble_kernel: tuple[float, float] = (1.0, -4.0),
    stimulus_frequency_hz: float = 1.0,
    seed: int | None = 13,
    uniform_values: np.ndarray | None = None,
    backend: str = "auto",
) -> NetworkSimulationResult:
    """Standalone Python replacement for the MATLAB/Simulink 2-neuron NetworkTutorial.

    Simulates two coupled neurons on the grid ``arange(0, duration_s + dt, dt)``.
    In each bin the log-odds of spiking for neuron ``i`` is the baseline
    plus self-history, a sinusoidal stimulus drive, and the other neuron's
    previous-bin spike; the spike probability is the logistic function of that
    sum.

    Parameters
    ----------
    duration_s : float, default 50.0
        Simulation length in seconds.  Must be positive.
    dt : float, default 0.001
        Time step (bin width) in seconds.  Must be positive.
    baseline_mu : tuple of float, default (-3.0, -3.0)
        Per-neuron baseline log-odds (one value per neuron).
    history_kernel : tuple of float, default (-4.0, -2.0, -1.0)
        Self-history filter coefficients shared by both neurons, ordered from
        the most recent bin (lag 1) backwards.
    stimulus_kernel : tuple of float, default (1.0, -1.0)
        Per-neuron gain applied to the sinusoidal stimulus.
    ensemble_kernel : tuple of float, default (1.0, -4.0)
        Coupling coefficients: the first entry is the effect of neuron 2's
        previous-bin spike on neuron 1, the second the effect of neuron 1 on
        neuron 2.
    stimulus_frequency_hz : float, default 1.0
        Frequency of the sinusoidal stimulus ``sin(2*pi*f*t)``.
    seed : int or None, default 13
        Seed for ``numpy.random.default_rng``; ignored when
        ``uniform_values`` is given.
    uniform_values : numpy.ndarray, optional
        Pre-drawn uniform variates of shape ``(len(time), 2)`` for
        deterministic tests.  Used by the native Python backend only.
    backend : {'auto', 'matlab', 'python'}, default ``'auto'``
        Simulation backend.  ``'auto'`` uses MATLAB/Simulink when
        available and falls back to native Python with a warning.
        ``'matlab'`` forces Simulink (raises if unavailable).
        ``'python'`` forces the native implementation.

    Returns
    -------
    NetworkSimulationResult
        Time grid, latent drive, per-bin spike probabilities, the two
        simulated spike trains (``neuron_1``, ``neuron_2``), the coupling
        matrix and the kernels used.  The per-bin ``eta``, ``history_effect``,
        ``ensemble_effect``, ``spike_indicator`` and ``uniform_values``
        diagnostics are populated by the Python backend only.

    Raises
    ------
    ValueError
        If ``duration_s`` or ``dt`` is not positive, ``backend`` is not one
        of the accepted values, or ``uniform_values`` has the wrong shape.
    RuntimeError
        If ``backend='matlab'`` is requested but MATLAB or the nSTAT MATLAB
        path is unavailable.
    """
    if duration_s <= 0 or dt <= 0:
        raise ValueError("duration_s and dt must be > 0")

    # ---- Backend selection ----
    from .matlab_engine import (
        MatlabFallbackWarning as _MFW,  # noqa: F401
        is_matlab_available as _is_avail,
        get_matlab_nstat_path as _get_path,
        simulate_network_via_simulink as _sim_net_sl,
        warn_fallback as _warn_fb,
    )

    if backend == "auto":
        _use_matlab = _is_avail() and _get_path() is not None
    elif backend == "matlab":
        if not _is_avail():
            raise RuntimeError(
                "backend='matlab' requested but MATLAB Engine is not "
                "available.  Install MATLAB and the MATLAB Engine API "
                "for Python, or use backend='auto' / backend='python'."
            )
        if _get_path() is None:
            raise RuntimeError(
                "backend='matlab' requested but the MATLAB nSTAT repo "
                "could not be found.  Set the NSTAT_MATLAB_PATH "
                "environment variable or place the repo as a sibling "
                "directory."
            )
        _use_matlab = True
    elif backend == "python":
        _use_matlab = False
    else:
        raise ValueError("backend must be 'auto', 'matlab', or 'python'")

    if _use_matlab:
        try:
            time = np.arange(0.0, duration_s + dt, dt)
            drive = np.sin(2.0 * np.pi * float(stimulus_frequency_hz) * time)
            hist_arr = np.asarray(history_kernel, dtype=float).reshape(-1)
            spike_times_list, lambda_data = _sim_net_sl(
                stim_time=time,
                stim_data=drive,
                baseline_mu=baseline_mu,
                history_kernel=hist_arr,
                stimulus_kernel=stimulus_kernel,
                ensemble_kernel=ensemble_kernel,
                dt=dt,
            )
            coll = SpikeTrainCollection([
                SpikeTrain(spike_times_list[0], name="neuron_1"),
                SpikeTrain(spike_times_list[1], name="neuron_2"),
            ])
            return NetworkSimulationResult(
                time=time,
                latent_drive=drive,
                lambda_delta=lambda_data,
                spikes=coll,
                actual_network=np.array([
                    [0.0, float(ensemble_kernel[0])],
                    [float(ensemble_kernel[1]), 0.0],
                ], dtype=float),
                history_kernel=hist_arr,
                stimulus_kernel=np.asarray(stimulus_kernel, dtype=float),
                ensemble_kernel=np.asarray(ensemble_kernel, dtype=float),
                baseline_mu=np.asarray(baseline_mu, dtype=float),
            )
        except Exception:
            if backend == "matlab":
                raise  # User explicitly asked for MATLAB — propagate
            # auto mode — fall back to Python with warning
            _warn_fb()
    elif backend == "auto":
        _warn_fb()

    time = np.arange(0.0, duration_s + dt, dt)
    drive = np.sin(2.0 * np.pi * float(stimulus_frequency_hz) * time)
    baseline_mu_arr = np.asarray(baseline_mu, dtype=float).reshape(2)
    history_kernel_arr = np.asarray(history_kernel, dtype=float).reshape(-1)
    stimulus_kernel_arr = np.asarray(stimulus_kernel, dtype=float).reshape(2)
    ensemble_kernel_arr = np.asarray(ensemble_kernel, dtype=float).reshape(2)
    actual_network = np.array(
        [
            [0.0, ensemble_kernel_arr[0]],
            [ensemble_kernel_arr[1], 0.0],
        ],
        dtype=float,
    )

    spikes = np.zeros((time.shape[0], 2), dtype=float)
    lambda_delta = np.zeros_like(spikes)
    eta_trace = np.zeros_like(spikes)
    history_effect = np.zeros_like(spikes)
    ensemble_effect = np.zeros_like(spikes)
    if uniform_values is None:
        rng = np.random.default_rng(seed)
        draws = rng.random(spikes.shape)
    else:
        draws = np.asarray(uniform_values, dtype=float)
        if draws.shape != spikes.shape:
            raise ValueError("uniform_values must have shape (len(time), 2)")
    for i in range(time.shape[0]):
        hist_self = np.zeros(2, dtype=float)
        for lag, coeff in enumerate(history_kernel_arr, start=1):
            if i - lag >= 0:
                hist_self[0] += float(coeff) * float(spikes[i - lag, 0])
                hist_self[1] += float(coeff) * float(spikes[i - lag, 1])
        ens_effect = np.zeros(2, dtype=float)
        if i - 1 >= 0:
            ens_effect[0] = ensemble_kernel_arr[0] * float(spikes[i - 1, 1])
            ens_effect[1] = ensemble_kernel_arr[1] * float(spikes[i - 1, 0])
        eta = baseline_mu_arr + hist_self + (stimulus_kernel_arr * float(drive[i])) + ens_effect
        history_effect[i] = hist_self
        ensemble_effect[i] = ens_effect
        eta_trace[i] = eta
        lambda_delta[i] = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        spikes[i, 0] = 1.0 if draws[i, 0] < lambda_delta[i, 0] else 0.0
        spikes[i, 1] = 1.0 if draws[i, 1] < lambda_delta[i, 1] else 0.0

    t1 = time[spikes[:, 0] > 0.5]
    t2 = time[spikes[:, 1] > 0.5]
    coll = SpikeTrainCollection([SpikeTrain(t1, name="neuron_1"), SpikeTrain(t2, name="neuron_2")])
    return NetworkSimulationResult(
        time=time,
        latent_drive=drive,
        lambda_delta=lambda_delta,
        spikes=coll,
        actual_network=actual_network,
        history_kernel=history_kernel_arr,
        stimulus_kernel=stimulus_kernel_arr,
        ensemble_kernel=ensemble_kernel_arr,
        baseline_mu=baseline_mu_arr,
        eta=eta_trace,
        history_effect=history_effect,
        ensemble_effect=ensemble_effect,
        spike_indicator=spikes,
        uniform_values=draws,
    )


__all__ = ["PointProcessSimulation", "NetworkSimulationResult", "simulate_point_process", "simulate_two_neuron_network"]
