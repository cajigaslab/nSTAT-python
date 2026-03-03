"""Trial demo aligned to MATLAB TrialExamples core setup."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import CovColl, Covariate, Trial, nspikeTrain, nstColl


def main() -> None:
    time = np.arange(0.0, 1.0 + 1e-12, 0.1)
    cov1 = Covariate(time=time, data=np.sin(2.0 * np.pi * time), name="sine", labels=["sine"])
    cov2 = Covariate(time=time, data=np.cos(2.0 * np.pi * time), name="ctx", labels=["ctx"])
    covs = CovColl([cov1, cov2])

    st1 = nspikeTrain(spike_times=np.array([0.10, 0.30, 0.70]), t_start=0.0, t_end=1.0, name="u1")
    st2 = nspikeTrain(spike_times=np.array([0.20, 0.40, 0.80]), t_start=0.0, t_end=1.0, name="u2")
    spikes = nstColl([st1, st2])

    trial = Trial(spikes=spikes, covariates=covs)
    t_bins, y, X = trial.getAlignedBinnedObservation(0.1, unitIndex=0, mode="count")

    print("Trial cov labels:", trial.getAllCovLabels())
    print("Trial neuron names:", trial.getNeuronNames())
    print("Aligned bins:", t_bins.shape)
    print("Spike vector shape:", y.shape)
    print("Design matrix shape:", X.shape)


if __name__ == "__main__":
    main()
