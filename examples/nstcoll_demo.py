"""nstColl parity demo."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import nspikeTrain
from nstat.compat.matlab import nstColl


def main() -> None:
    st1 = nspikeTrain(spike_times=np.array([0.10, 0.20, 0.25, 0.90]), t_start=0.0, t_end=1.0, name="u1")
    st2 = nspikeTrain(spike_times=np.array([0.15, 0.40, 0.80]), t_start=0.0, t_end=1.0, name="u2")
    st1.resample(10.0)
    st2.resample(10.0)

    coll = nstColl([st1, st2])
    print("First/last support:", coll.getFirstSpikeTime(), coll.getLastSpikeTime())
    print("Names:", coll.getNSTnames())
    print("Count matrix shape:", coll.dataToMatrix(0.1, "count").shape)
    print("PSTH:", coll.psth(0.1)[1])
    print("Merged spikes:", coll.toSpikeTrain().spike_times)


if __name__ == "__main__":
    main()
