from __future__ import annotations

import numpy as np

from nstat import Analysis, CIFModel, ConfigCollection, Covariate, CovariateCollection, FitSummary, Trial, TrialConfig
from nstat.nspikeTrain import nspikeTrain
from nstat.nstColl import nstColl


def test_trial_analysis_pipeline() -> None:
    t = np.arange(0.0, 1.0, 0.001)
    stim = np.sin(2 * np.pi * 2 * t)
    cov = Covariate(t, stim, "stim", "time", "s", "a.u.", ["stim"])

    model = CIFModel(t, 10.0 + 5.0 * np.maximum(stim, 0.0), name="lambda")
    spikes = model.simulate(num_realizations=3, seed=2)

    trial = Trial(spike_collection=spikes, covariate_collection=CovariateCollection([cov]))
    cfgs = ConfigCollection([TrialConfig(covMask=["stim"], sampleRate=1000.0, name="stim_model")])

    fits = Analysis.run_analysis_for_all_neurons(trial, cfgs)
    assert len(fits) == 3
    assert fits[0].AIC.shape == (1,)

    summary = FitSummary(fits)
    assert summary.numNeurons == 3
    assert summary.AIC.shape == (3, 1)
    assert summary.meanAIC.shape == (1,)


def test_analysis_helpers_accept_multi_trial_spike_inputs_like_matlab() -> None:
    time = np.arange(0.0, 1.1, 0.1)
    lam = Covariate(time, np.full(time.shape, 2.0), "\\lambda(t)", "time", "s", "Hz", ["lambda"])

    st1 = nspikeTrain([0.1, 0.3], "1", 10.0, 0.0, 0.5, "time", "s", "", "", -1)
    st2 = nspikeTrain([0.2], "1", 10.0, 0.0, 0.5, "time", "s", "", "", -1)
    coll = nstColl([st1, st2])
    collapsed = coll.toSpikeTrain()

    multi_ks = Analysis.computeKSStats([st1, st2], lam, DTCorrection=0)
    collapsed_ks = Analysis.computeKSStats(collapsed, lam, DTCorrection=0)
    for left, right in zip(multi_ks, collapsed_ks, strict=False):
        np.testing.assert_allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=1e-8, atol=1e-10)

    multi_residual = Analysis.computeFitResidual([st1, st2], lam, windowSize=0.1)
    collapsed_residual = Analysis.computeFitResidual(collapsed, lam, windowSize=0.1)
    np.testing.assert_allclose(multi_residual.time, collapsed_residual.time, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(multi_residual.data, collapsed_residual.data, rtol=1e-8, atol=1e-10)
