"""Spike-train distance metrics — ISI, SPIKE, SPIKE-synchronization.

Thin wrappers around PySpike (https://github.com/mariomulansky/PySpike),
a BSD-2 library with C/Cython acceleration that implements the
Kreuz / Mulansky family of time-resolved spike-train distances:

- **ISI-distance** (Kreuz 2007) — instantaneous dissimilarity in ISIs.
- **SPIKE-distance** (Kreuz 2013) — instantaneous dissimilarity in
  spike timing.  Bounded in [0, 1].
- **SPIKE-synchronization** (Kreuz 2015) — fraction of "synchronous"
  spikes; symmetric in [0, 1].

All three are *parameter-free* — no kernel bandwidth, no binning.
That's their main advantage over rate-based or kernel-density measures.

Install:
    pip install nstat-toolbox[metrics]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nstat import nspikeTrain

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


_IMPORT_ERROR_MSG = (
    "nstat.extras.metrics.spike_distances requires the 'pyspike' package. "
    "Install with: pip install nstat-toolbox[metrics]"
)


def _require_pyspike():
    try:
        import pyspike as spk
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    return spk


def _to_pyspike(nst: nspikeTrain):
    """Convert an :class:`nspikeTrain` to a ``pyspike.SpikeTrain``."""
    spk = _require_pyspike()
    return spk.SpikeTrain(
        spike_times=np.asarray(nst.spikeTimes, dtype=float),
        edges=(float(nst.minTime), float(nst.maxTime)),
    )


def isi_distance(a: nspikeTrain, b: nspikeTrain) -> float:
    """ISI-distance between two spike trains (Kreuz 2007).

    Returns
    -------
    float
        Scalar ISI-distance in [0, 1].
    """
    spk = _require_pyspike()
    return float(spk.isi_distance(_to_pyspike(a), _to_pyspike(b)))


def spike_distance(a: nspikeTrain, b: nspikeTrain) -> float:
    """SPIKE-distance between two spike trains (Kreuz 2013).

    Returns
    -------
    float
        Scalar SPIKE-distance in [0, 1].
    """
    spk = _require_pyspike()
    return float(spk.spike_distance(_to_pyspike(a), _to_pyspike(b)))


def spike_synchronization(a: nspikeTrain, b: nspikeTrain) -> float:
    """SPIKE-synchronization between two spike trains (Kreuz 2015).

    Returns
    -------
    float
        Scalar synchronization in [0, 1] — fraction of "coincident"
        spikes (1.0 = perfectly synchronous, 0.0 = no coincidences).
    """
    spk = _require_pyspike()
    return float(spk.spike_sync(_to_pyspike(a), _to_pyspike(b)))


def pairwise_spike_distance_matrix(
    trains: list[nspikeTrain],
) -> np.ndarray:
    """N×N matrix of pairwise SPIKE-distances over a population.

    Parameters
    ----------
    trains
        List of nstat spike trains; all must share the same recording
        window (``minTime`` / ``maxTime``).

    Returns
    -------
    numpy.ndarray
        Symmetric matrix of shape ``(N, N)``; diagonal is zero.
    """
    spk = _require_pyspike()
    pyspike_trains = [_to_pyspike(t) for t in trains]
    return np.asarray(spk.spike_distance_matrix(pyspike_trains), dtype=float)


__all__ = [
    "isi_distance",
    "spike_distance",
    "spike_synchronization",
    "pairwise_spike_distance_matrix",
]
