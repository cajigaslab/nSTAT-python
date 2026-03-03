"""nspikeTrain parity demo."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import nspikeTrain


def main() -> None:
    st = nspikeTrain(spike_times=np.array([0.10, 0.20, 0.25, 0.90]), t_start=0.0, t_end=1.0, name="u1")
    st.resample(10.0)

    sig = st.getSigRep(binSize_s=0.1, mode="count", minTime_s=0.0, maxTime_s=1.0)
    print("Count representation:", sig)
    print("ISIs:", st.getISIs())
    print("Min ISI:", st.getMinISI())
    print("Max binary bin size:", st.getMaxBinSizeBinary())
    print("Firing rate:", st.computeRate())
    print("L-statistic:", st.getLStatistic())


if __name__ == "__main__":
    main()
