import numpy as np

from nstat.analysis import Analysis
from nstat.decoding import DecodingAlgorithms
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial, TrialConfig


def _trial() -> Trial:
    t = np.linspace(0.0, 1.0, 1001)
    stim = np.sin(2 * np.pi * 5 * t)
    cov = Covariate(time=t, data=stim, name="stim")

    spikes = np.array([0.12, 0.26, 0.41, 0.6, 0.77, 0.91])
    st = SpikeTrain(spike_times=spikes, t_start=0.0, t_end=1.0)

    return Trial(
        spikes=SpikeTrainCollection([st]),
        covariates=CovariateCollection([cov]),
    )


def test_fit_trial_poisson() -> None:
    trial = _trial()
    cfg = TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson")
    res = Analysis.fit_trial(trial, cfg)
    assert res.n_parameters >= 1
    assert np.isfinite(res.aic())


def test_decode_algorithms() -> None:
    rng = np.random.default_rng(0)
    mat = rng.binomial(1, 0.1, size=(8, 200)).astype(float)
    rates, prob, sig = DecodingAlgorithms.compute_spike_rate_cis(mat)
    assert rates.shape == (8,)
    assert prob.shape == (8, 8)
    assert sig.shape == (8, 8)
