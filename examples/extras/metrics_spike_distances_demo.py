"""Demo: ISI / SPIKE / SPIKE-synchronization metrics on a 5-train population.

Generates 5 inhomogeneous-Poisson spike trains with shared rate
modulation (so they're partially synchronous), then computes the
pairwise SPIKE-distance matrix via PySpike.

Demonstrates :mod:`nstat.extras.metrics.spike_distances`:

- :func:`isi_distance`, :func:`spike_distance`, :func:`spike_synchronization`
  for pairwise scalar metrics.
- :func:`pairwise_spike_distance_matrix` for population-level analysis.

Run::

    pip install nstat-toolbox[metrics]
    python examples/extras/metrics_spike_distances_demo.py
"""
from __future__ import annotations

import numpy as np

from nstat import nspikeTrain


def _simulate_shared_modulation(
    n_trains: int = 5, duration: float = 5.0, rng_seed: int = 0
) -> list[nspikeTrain]:
    """Five trains sharing a sinusoidal rate modulation (partially synchronous)."""
    rng = np.random.default_rng(rng_seed)
    dt = 0.001
    t = np.arange(0.0, duration, dt)
    base_rate = 5.0 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t))  # 0.5 Hz modulation
    trains: list[nspikeTrain] = []
    for i in range(n_trains):
        # Per-train jitter on top of the shared modulation.
        per_train_rate = base_rate * (1.0 + 0.1 * rng.standard_normal(t.size))
        spikes_per_bin = rng.poisson(per_train_rate * dt)
        spike_times = t[spikes_per_bin > 0]
        trains.append(
            nspikeTrain(
                spikeTimes=spike_times,
                name=f"unit_{i}",
                sampleRate=1000.0,
                minTime=0.0,
                maxTime=duration,
            )
        )
    return trains


def main() -> int:
    try:
        from nstat.extras.metrics.spike_distances import (
            isi_distance,
            spike_distance,
            spike_synchronization,
            pairwise_spike_distance_matrix,
        )
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    trains = _simulate_shared_modulation()
    print(f"Population    : {len(trains)} trains over "
          f"[{trains[0].minTime}, {trains[0].maxTime}] s, "
          f"{[len(t.spikeTimes) for t in trains]} spikes each")

    # --- Pairwise scalar metrics for trains 0 and 1 --------------------
    d_isi = isi_distance(trains[0], trains[1])
    d_spike = spike_distance(trains[0], trains[1])
    s_sync = spike_synchronization(trains[0], trains[1])
    print(f"\nTrain 0 vs 1:")
    print(f"  ISI-distance              = {d_isi:.4f}  (0=identical timing)")
    print(f"  SPIKE-distance            = {d_spike:.4f}  (in [0, 1])")
    print(f"  SPIKE-synchronization     = {s_sync:.4f}  (1=perfectly sync)")

    # --- Population-level pairwise distance matrix ---------------------
    D = pairwise_spike_distance_matrix(trains)
    print(f"\nPairwise SPIKE-distance matrix ({D.shape[0]}×{D.shape[1]}):")
    for row in D:
        print("  " + "  ".join(f"{x:.3f}" for x in row))
    print(f"\nMean off-diagonal SPIKE-distance: "
          f"{D[~np.eye(len(trains), dtype=bool)].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
