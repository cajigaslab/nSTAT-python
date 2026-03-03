"""CovColl demo aligned to MATLAB helpfiles/CovCollExamples workflows."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import CovColl, Covariate


def run_demo() -> CovColl:
    time = np.arange(0.0, 1.0 + 1e-12, 0.1)
    cov1 = Covariate(time=time, data=np.sin(2.0 * np.pi * time), name="sine", labels=["sine"])
    cov2 = Covariate(time=time, data=np.column_stack([time, time**2]), name="poly", labels=["t", "t2"])
    coll = CovColl([cov1, cov2])
    return coll


if __name__ == "__main__":
    cov_coll = run_demo()
    X, labels = cov_coll.dataToMatrix()
    print("CovColl labels:", labels)
    print("CovColl matrix shape:", X.shape)
