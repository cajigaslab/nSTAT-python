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
    cfgs = ConfigCollection([TrialConfig(covMask=[["stim", "stim"]], sampleRate=1000.0, name="stim_model")])
    return Analysis.run_analysis_for_all_neurons(trial, cfgs)[0]


def test_fitresult_diagnostics_populate_ks_and_residual_fields() -> None:
    fit = _build_fit_result()

    ks = fit.computeKSStats()
    residual = fit.computeFitResidual()
    inv = fit.computeInvGausTrans()

    assert "ks_stat" in ks
    assert fit.KSStats.shape == (1, 1)
    assert residual.name == "M(t_k)"
    assert np.asarray(inv, dtype=float).ndim == 1
    assert np.all(np.isfinite(np.asarray(inv, dtype=float)))
    np.testing.assert_allclose(np.asarray(inv, dtype=float), np.asarray(fit.X, dtype=float))
    assert fit.Residual is not None


def test_fitresult_plotting_methods_return_matplotlib_objects() -> None:
    fit = _build_fit_result()

    fig = fit.plotResults()
    ax1 = fit.KSPlot()
    ax2 = fit.plotResidual()
    ax3 = fit.plotInvGausTrans()
    ax4 = fit.plotSeqCorr()
    ax5 = fit.plotCoeffs()

    assert len(fig.axes) == 5  # 2x4 layout: KS, InvGaus, SeqCorr, Coeffs, Residual
    for ax in (ax1, ax2, ax3, ax4, ax5):
        assert hasattr(ax, "plot")
    plt.close("all")


def test_fitresult_matlab_style_helpers_expose_plot_params_and_subsets() -> None:
    fit = _build_fit_result()

    fit.setNeuronName("unitA")
    plot_params = fit.getPlotParams()
    coeffs, labels, se = fit.getCoeffsWithLabels(0)
    param_vals, param_se, param_sig = fit.getParam(labels[0], 0)
    subset = fit.getSubsetFitResult([0])
    fit.setKSStats(np.array([0.1]), np.array([0.2]), np.array([0.3]), np.array([0.4]), np.array([0.5]))
    fit.setInvGausStats(np.array([0.1]), np.array([0.2]), np.array([0.3]))
    fit.setFitResidual({"value": 1})

    assert fit.neuronNumber == "unitA"
    assert plot_params["bAct"].shape[1] == fit.numResults
    assert len(labels) == coeffs.size == se.size
    assert param_vals.size == param_se.size == param_sig.size == 1
    assert subset.numResults == 1
    assert np.isfinite(fit.KSStats[0, 0])
    assert np.isfinite(fit.KSPvalues[0])
    assert fit.invGausStats["X"].shape == (1,)
    assert fit.Residual == {"value": 1}

    ax1 = fit.plotCoeffsWithoutHistory()
    ax2 = fit.plotHistCoeffs()
    assert hasattr(ax1, "plot")
    assert hasattr(ax2, "plot")
    plt.close("all")


def test_fitsummary_plotsummary_returns_figure() -> None:
    fit = _build_fit_result()
    summary = FitSummary([fit])
    fig = summary.plotSummary()
    assert len(fig.axes) == 3
    plt.close("all")


def test_fitsummary_matlab_style_helpers_cover_ic_and_coeff_views() -> None:
    fit = _build_fit_result()
    summary = FitSummary([fit])

    coeff_mat, labels, se_mat = summary.getCoeffs(0)
    sig = summary.getSigCoeffs(0)
    bins, edges, percent_sig = summary.binCoeffs(-5.0, 5.0, 1.0)
    summary.setCoeffRange(-2.0, 2.0)

    assert coeff_mat.shape == se_mat.shape
    assert coeff_mat.shape[0] == summary.numNeurons
    assert sig.shape == coeff_mat.shape
    assert len(labels) == coeff_mat.shape[1]
    assert bins.ndim == 2  # (nBins, nCovariates) — per-covariate histograms
    assert edges.ndim == 1
    assert percent_sig.ndim == 1  # one value per covariate
    assert np.all((0.0 <= percent_sig) & (percent_sig <= 1.0))
    assert summary.coeffMin == -2.0
    assert summary.coeffMax == 2.0

    fig1 = summary.plotIC()
    ax1 = summary.plotAIC()
    ax2 = summary.plotBIC()
    ax3 = summary.plotlogLL()
    fig2 = summary.plotResidualSummary()
    ax4 = summary.boxPlot(coeff_mat, dataLabels=labels)
    restored = FitSummary.fromStructure(summary.toStructure())

    assert len(fig1.axes) == 3
    assert summary.AIC.shape == (1, 1)
    assert summary.meanAIC.shape == (1,)
    assert hasattr(ax1, "boxplot")
    assert hasattr(ax2, "boxplot")
    assert hasattr(ax3, "boxplot")
    assert len(fig2.axes) == 1
    assert hasattr(ax4, "boxplot")
    assert restored.numNeurons == summary.numNeurons
    plt.close("all")


class _CoeffStubSummary(FitSummary):
    """FitSummary whose per-fit coefficient matrices are given directly.

    Rows are neurons, columns follow ``uniqueCovLabels``.
    """

    def __init__(self, per_fit: list[np.ndarray], labels: list[str]) -> None:  # noqa: D107
        self._per_fit = per_fit
        self.uniqueCovLabels = labels
        self.numResults = len(per_fit)
        self.numNeurons = per_fit[0].shape[0]

    def getCoeffs(self, fitNum: int = 0):
        mat = self._per_fit[fitNum]
        return mat, list(self.uniqueCovLabels), np.full_like(mat, np.nan)


def test_fitsummary_hist_and_coeff_index_keep_labels_finite_in_any_fit_and_neuron() -> None:
    """Mirror of cajigaslab/nSTAT@d37a07a (FitResSummary.getHistIndex).

    MATLAB kept a label only if neuron 1 had it (``bAct(:,fitNum)`` on a 3-D
    array).  The fix keeps a label that is finite for ANY requested fit and
    ANY neuron.  Python already looked at every neuron, but only at the first
    requested fit.
    """
    nan = np.nan
    labels = ["constant", "stim", "[0,0.001]", "[0.001,0.002]"]
    # fit 0: neuron 0 has no history and no stim; neuron 1 has the first window only.
    fit0 = np.array([[0.1, nan, nan, nan],
                     [0.2, nan, 1.4, nan]])
    # fit 1: only neuron 1 has the second window and the stim term.
    fit1 = np.array([[0.1, nan, nan, nan],
                     [0.2, 0.5, nan, 1.3]])
    summary = _CoeffStubSummary([fit0, fit1], labels)

    # Any neuron (the MATLAB neuron-1 defect): neuron 1 alone keeps window 1.
    assert summary.getHistIndex(0) == [2]
    assert summary.getCoeffIndex(0) == [0]
    # Any requested fit: the union over fits 0 and 1.
    assert summary.getHistIndex([0, 1]) == [2, 3]
    assert summary.getCoeffIndex([0, 1]) == [0, 1]
    # Default = every fit.
    assert summary.getHistIndex() == [2, 3]


def test_fitsummary_hist_index_empty_when_no_history_is_finite() -> None:
    nan = np.nan
    labels = ["constant", "[0,0.001]"]
    summary = _CoeffStubSummary([np.array([[0.1, nan], [0.2, nan]])], labels)
    assert summary.getHistIndex() == []
    assert summary.getCoeffIndex() == [0]
