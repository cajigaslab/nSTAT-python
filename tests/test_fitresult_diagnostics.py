from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from nstat import Analysis, CIFModel, ConfigCollection, Covariate, CovariateCollection, FitSummary, Trial, TrialConfig


def _build_fit_result():
    t = np.arange(0.0, 1.0, 0.001)
    stim = np.sin(2 * np.pi * 2 * t)
    cov = Covariate(t, stim, "stim", "time", "s", "a.u.", ["stim"])

    model = CIFModel(t, 12.0 + 4.0 * np.maximum(stim, 0.0), name="lambda")
    spikes = model.simulate(num_realizations=2, seed=7)

    trial = Trial(spike_collection=spikes, covariate_collection=CovariateCollection([cov]))
    cfgs = ConfigCollection([TrialConfig(covMask=["stim"], sampleRate=1000.0, name="stim_model")])
    return Analysis.run_analysis_for_all_neurons(trial, cfgs)[0]


def test_fitresult_diagnostics_populate_ks_and_residual_fields() -> None:
    fit = _build_fit_result()

    ks = fit.computeKSStats()
    residual = fit.computeFitResidual()
    inv = fit.computeInvGausTrans()

    assert "ks_stat" in ks
    assert fit.KSStats.shape == (1, 1)
    assert residual.name == "fit residual"
    assert np.asarray(inv, dtype=float).ndim == 1
    assert fit.Residual is not None


def test_fitresult_plotting_methods_return_matplotlib_objects() -> None:
    fit = _build_fit_result()

    fig = fit.plotResults()
    ax1 = fit.KSPlot()
    ax2 = fit.plotResidual()
    ax3 = fit.plotInvGausTrans()
    ax4 = fit.plotSeqCorr()
    ax5 = fit.plotCoeffs()

    assert len(fig.axes) == 4
    for ax in (ax1, ax2, ax3, ax4, ax5):
        assert hasattr(ax, "plot")
    plt.close("all")


def test_fitsummary_plotsummary_returns_figure() -> None:
    fit = _build_fit_result()
    summary = FitSummary([fit])
    fig = summary.plotSummary()
    assert len(fig.axes) == 3
    plt.close("all")
