from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import SpikeTrain


def spike_indicator(spike_train: SpikeTrain, time: Sequence[float]) -> np.ndarray:
    """Convert spike times to a binary per-sample indicator on a time grid."""

    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("time must include at least two samples.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing.")

    dt = np.diff(t)
    edges = np.concatenate([t, [t[-1] + dt[-1]]])
    counts = spike_train.to_binned_counts(edges)
    return (counts > 0).astype(float)


def psth(
    spike_trains: Sequence[SpikeTrain], bin_edges: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin mean firing rate (PSTH) across trials."""

    if len(spike_trains) == 0:
        raise ValueError("spike_trains must contain at least one trial.")

    edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    if edges.size < 2:
        raise ValueError("bin_edges must contain at least 2 values.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")

    counts = np.vstack([trial.to_binned_counts(edges) for trial in spike_trains])
    mean_rate_hz = counts.mean(axis=0) / np.diff(edges)
    return mean_rate_hz, counts

