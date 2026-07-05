"""Demo: read an NWB file into an nstat :class:`SpikeTrainCollection`.

NWB (Neurodata Without Borders; Teeters et al. 2015, Rübel et al. 2022) is
the standard container behind the DANDI Archive's human intracranial
datasets. One such dataset is **AJILE12** (Peterson et al. 2022, Sci Data
9:184; DANDI dandiset 000055): naturalistic, days-long chronic ECoG paired
with upper-body pose tracking, recorded from 12 human subjects during
clinical epilepsy monitoring, stored in NWB on DANDI. AJILE12 itself
stores continuous 500 Hz ECoG voltage as a ``pynwb.ecephys.ElectricalSeries``
(macro-electrode LFP-band data), not spike-sorted units in an NWB
``Units`` table -- ECoG at that scale doesn't yield sortable single
units -- but it is a concrete example of why a robust, standardized NWB
reader matters for clinical/BCI neurophysiology data sharing.

This demo does **not** download AJILE12 or any other DANDI dataset. It
exercises the general NWB ``units``/``obs_intervals`` ingestion path --
the pattern used by spike-sorted (single/multi-unit) NWB datasets, not
AJILE12's continuous ECoG data -- on a small synthetic in-memory NWB
file with per-unit spike times plus ``obs_intervals`` recording windows
that vary in duration across units, then converts it to nstat primitives
using both the ``obs_intervals`` resolution path and the explicit
``time_window=`` override. The file itself stays fully synthetic so the
demo runs with no download and no new dependency.

Demonstrates :mod:`nstat.extras.interop.nwb`:

- :func:`nwb_units_to_collection` with implicit per-unit observation
  windows derived from ``obs_intervals``.
- The same call with an explicit ``time_window=(t0, t1)`` that
  overrides obs_intervals and silences the fallback warning.

References: Teeters JL et al. (2015). Neurodata Without Borders: Creating
a Common Data Format for Neurophysiology. Neuron 88:629. Rübel O et al.
(2022). The Neurodata Without Borders ecosystem for neurophysiological
data science. eLife 11:e78362. Peterson SM et al. (2022). AJILE12: Long-
term naturalistic human intracranial neural recordings and pose. Scientific
Data 9:184 (DANDI dandiset 000055).

Run::

    pip install nstat-toolbox[nwb]
    python examples/extras/interop_nwb_demo.py
"""
from __future__ import annotations

from datetime import datetime, timezone


def main() -> int:
    try:
        from pynwb import NWBFile
        from nstat.extras.interop.nwb import nwb_units_to_collection
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    # --- Build a synthetic NWB file with 3 units + obs_intervals -------
    nwb = NWBFile(
        session_description="demo session",
        identifier="demo-001",
        session_start_time=datetime.now(tz=timezone.utc),
    )
    # Each unit observed [0, 10] s; three with varying spike rates.
    nwb.add_unit_column(
        name="obs_intervals",
        description="time intervals during which the unit was observed",
        index=True,
    )
    for spike_times in ([0.1, 0.5, 0.9], [0.2, 0.8, 1.5, 4.0], [3.3, 5.5, 7.7, 9.9]):
        nwb.add_unit(spike_times=spike_times, obs_intervals=[[0.0, 10.0]])

    # --- Convert using obs_intervals (the standard path) ----------------
    coll = nwb_units_to_collection(nwb, sample_rate=30_000.0)
    print(f"From obs_intervals:")
    for nst in coll:
        print(f"  {nst.name}: {len(nst.spikeTimes)} spikes, "
              f"window=[{nst.minTime}, {nst.maxTime}] s")

    # --- Same call with explicit time_window= override ------------------
    coll_override = nwb_units_to_collection(
        nwb, sample_rate=30_000.0, time_window=(0.0, 20.0)
    )
    print(f"\nWith explicit time_window=(0.0, 20.0):")
    for nst in coll_override:
        print(f"  {nst.name}: window=[{nst.minTime}, {nst.maxTime}] s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
