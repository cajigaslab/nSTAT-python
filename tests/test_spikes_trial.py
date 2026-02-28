import numpy as np

from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial


def test_spike_train_binarize() -> None:
    st = SpikeTrain(spike_times=np.array([0.1, 0.2, 0.21]), t_start=0.0, t_end=1.0)
    t, y = st.binarize(0.1)
    assert t.ndim == 1
    assert y.ndim == 1
    assert y.max() <= 1.0

    tc, counts = st.bin_counts(0.1)
    assert tc.shape == t.shape
    assert np.sum(counts) == 3.0


def test_trial_alignment() -> None:
    t = np.linspace(0.0, 1.0, 101)
    cov = Covariate(time=t, data=np.sin(2 * np.pi * t), name="stim")
    covs = CovariateCollection([cov])

    trains = SpikeTrainCollection([
        SpikeTrain(spike_times=np.array([0.1, 0.3, 0.9]), t_start=0.0, t_end=1.0)
    ])
    trial = Trial(spikes=trains, covariates=covs)

    tb, y, X = trial.aligned_binned_observation(0.01)
    assert tb.shape[0] == y.shape[0]
    assert X.shape[0] == y.shape[0]

    _, yc, _ = trial.aligned_binned_observation(0.01, mode="count")
    assert np.all(yc >= 0.0)
