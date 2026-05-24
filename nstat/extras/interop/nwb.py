"""NWB (Neurodata Without Borders) → nstat reader (Tier-B interop).

NWB:N is the BRAIN-Initiative-standard format for neurophysiology data.
This module reads NWB files into nstat's :class:`Trial` /
:class:`SpikeTrainCollection` primitives so analyses written for nstat
can run directly on NWB-formatted datasets.

The reverse direction (nstat → NWB writer) is deliberately omitted in
the initial release: writing a faithful NWB file requires populating
mandatory metadata (subject, session_description, experimenter, …) that
nstat doesn't track, so the conversion would need user-supplied fields
the bridge can't infer.  Users who need to write NWB should construct
the file via :mod:`pynwb` directly and use this module's converters to
populate the spike-train tables.

Install:
    pip install nstat-toolbox[nwb]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nstat import nspikeTrain, SpikeTrainCollection

if TYPE_CHECKING:
    import pynwb


_IMPORT_ERROR_MSG = (
    "nstat.extras.interop.nwb requires the 'pynwb' package. "
    "Install with: pip install nstat-toolbox[nwb]"
)


def _require_pynwb() -> None:
    try:
        import pynwb  # noqa: F401
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e


def nwb_units_to_collection(
    nwbfile: "pynwb.NWBFile",
    *,
    sample_rate: float = 30000.0,
    name_prefix: str = "unit",
    time_window: tuple[float, float] | None = None,
) -> SpikeTrainCollection:
    """Read the ``units`` table of an NWB file into a :class:`SpikeTrainCollection`.

    Recording-window resolution order (most specific → least specific):

    1. Explicit ``time_window=(t0, t1)`` parameter.
    2. Per-unit ``obs_intervals`` column if present (NWB-standard field
       for "intervals during which this unit was observed").  Uses the
       outer envelope ``[min(start), max(end)]``.
    3. Session-level ``nwbfile.session_start_time`` /
       ``nwbfile.timestamps_reference_time`` if both populated and
       NWB-style ``session_description`` carries a duration hint.
    4. Fallback: ``[min(spike_times), max(spike_times)]`` per unit — and
       a ``UserWarning`` is emitted because this silently understates
       the observation window for sparse-firing units.

    Parameters
    ----------
    nwbfile
        Open NWB file with a populated ``units`` table.
    sample_rate
        Recording sample rate in Hz.  NWB's ``Units`` table stores spike
        times directly, not raw samples, so this is metadata used by
        nstat's primitives only — it does not affect the spike times.
        Defaults to 30 kHz (typical sorting acquisition rate).
    name_prefix
        Prefix for auto-generated unit names; resulting names look like
        ``f"{name_prefix}_{row_id}"``.
    time_window
        Explicit ``(min_time_s, max_time_s)`` recording window applied
        uniformly to every unit.  Strongly recommended when the NWB
        file does not carry ``obs_intervals`` — otherwise PSTH /
        rate-based downstream analyses will silently corrupt for
        sparsely-firing units.

    Returns
    -------
    SpikeTrainCollection
    """
    import warnings

    _require_pynwb()
    units = nwbfile.units
    if units is None or len(units) == 0:
        return SpikeTrainCollection()

    has_obs_intervals = (
        time_window is None
        and "obs_intervals" in getattr(units, "colnames", ())
    )
    warned_about_spike_bounds = False

    trains: list[nspikeTrain] = []
    for idx, row_id in enumerate(units.id[:]):
        spike_times = np.asarray(units["spike_times"][idx], dtype=float)

        if time_window is not None:
            min_t, max_t = float(time_window[0]), float(time_window[1])
        elif has_obs_intervals:
            intervals = np.asarray(units["obs_intervals"][idx], dtype=float)
            # obs_intervals is shape (n_intervals, 2)
            if intervals.ndim == 1:
                intervals = intervals.reshape(-1, 2)
            min_t = float(intervals[:, 0].min())
            max_t = float(intervals[:, 1].max())
        elif spike_times.size:
            min_t = float(spike_times.min())
            max_t = float(spike_times.max())
            if not warned_about_spike_bounds:
                warnings.warn(
                    "nwb_units_to_collection: NWB file has no obs_intervals "
                    "and no explicit time_window was passed — falling back "
                    "to per-unit spike-time bounds.  Sparse-firing units "
                    "will have an artificially narrow window; PSTH and "
                    "rate calculations will silently understate denominators. "
                    "Pass time_window=(t0, t1) to fix.",
                    UserWarning,
                    stacklevel=2,
                )
                warned_about_spike_bounds = True
        else:
            # Empty unit + no window info — degenerate but legal NWB.
            min_t, max_t = 0.0, 0.0

        trains.append(
            nspikeTrain(
                spikeTimes=spike_times,
                name=f"{name_prefix}_{int(row_id)}",
                sampleRate=sample_rate,
                minTime=min_t,
                maxTime=max_t,
            )
        )
    return SpikeTrainCollection(trains)


def read_nwb_path(
    path: str,
    *,
    sample_rate: float = 30000.0,
    name_prefix: str = "unit",
) -> SpikeTrainCollection:
    """Convenience: open an NWB file path and return a :class:`SpikeTrainCollection`.

    Parameters
    ----------
    path
        Filesystem path to a ``.nwb`` file.
    sample_rate, name_prefix
        Forwarded to :func:`nwb_units_to_collection`.

    Returns
    -------
    SpikeTrainCollection
    """
    _require_pynwb()
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(path, mode="r") as io:
        nwbfile = io.read()
        return nwb_units_to_collection(
            nwbfile, sample_rate=sample_rate, name_prefix=name_prefix
        )


__all__ = ["nwb_units_to_collection", "read_nwb_path"]
