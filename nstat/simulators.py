from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spikes import SpikeTrain, SpikeTrainCollection


@dataclass
class PointProcessSimulation:
    time: np.ndarray
    rate_hz: np.ndarray
    spikes: SpikeTrain


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


def simulate_point_process(time: np.ndarray, rate_hz: np.ndarray, *, seed: int | None = None) -> PointProcessSimulation:
    t = np.asarray(time, dtype=float).reshape(-1)
    r = np.asarray(rate_hz, dtype=float).reshape(-1)
    if t.shape[0] != r.shape[0]:
        raise ValueError("time and rate_hz length mismatch")
    if t.shape[0] < 2:
        return PointProcessSimulation(t, r, SpikeTrain(np.array([], dtype=float)))

    dt = np.diff(t)
    dt = np.concatenate([dt, [dt[-1]]])
    p = 1.0 - np.exp(-np.clip(r, 0.0, np.inf) * dt)
    p = np.clip(p, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    keep = rng.random(t.shape[0]) < p
    return PointProcessSimulation(t, r, SpikeTrain(t[keep]))


def simulate_two_neuron_network(
    duration_s: float = 50.0,
    dt: float = 0.001,
    baseline_mu: tuple[float, float] = (-3.0, -3.0),
    history_kernel: tuple[float, ...] = (-4.0, -2.0, -1.0),
    stimulus_kernel: tuple[float, float] = (1.0, -1.0),
    ensemble_kernel: tuple[float, float] = (1.0, -4.0),
    stimulus_frequency_hz: float = 1.0,
    seed: int | None = 13,
) -> NetworkSimulationResult:
    """Standalone Python replacement for the MATLAB/Simulink 2-neuron NetworkTutorial."""
    if duration_s <= 0 or dt <= 0:
        raise ValueError("duration_s and dt must be > 0")

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

    rng = np.random.default_rng(seed)
    spikes = np.zeros((time.shape[0], 2), dtype=float)
    lambda_delta = np.zeros_like(spikes)
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
        lambda_delta[i] = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        spikes[i, 0] = 1.0 if rng.random() < lambda_delta[i, 0] else 0.0
        spikes[i, 1] = 1.0 if rng.random() < lambda_delta[i, 1] else 0.0

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
    )


__all__ = ["PointProcessSimulation", "NetworkSimulationResult", "simulate_point_process", "simulate_two_neuron_network"]
