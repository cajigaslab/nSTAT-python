"""Covariate parity demo.

This mirrors key MATLAB Covariate operations used in class-level parity tests.
"""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import ConfidenceInterval
from nstat.compat.matlab import Covariate


def main() -> None:
    time = np.linspace(0.0, 1.0, 6)
    data = np.column_stack([time, time**2, 0.5 + 0.75 * time])

    cov = Covariate(
        time=time,
        data=data,
        name="stim",
        units="u",
        labels=["c1", "c2", "c3"],
        x_label="time",
        x_units="s",
        y_units="u",
    )

    zero_mean = cov.getSigRep("zero-mean")
    mean_ci = cov.computeMeanPlusCI(0.10)

    print("Covariate shape:", cov.dataToMatrix().shape)
    print("Zero-mean channel means:", np.mean(zero_mean.dataToMatrix(), axis=0))
    print("Mean+CI shape:", mean_ci.dataToMatrix().shape)

    cov1 = Covariate(time=time, data=time, name="a", units="u", labels=["a"], x_label="time", x_units="s", y_units="u")
    ci = ConfidenceInterval(time=time, lower=time - 0.1, upper=time + 0.2)
    cov1.setConfInterval(ci)

    shifted = cov1.plus(0.5)
    print("Shifted cov first sample:", float(shifted.dataToMatrix()[0, 0]))


if __name__ == "__main__":
    main()
