"""Expanded test coverage for post-v0.3.0 — edge cases, serialization, plotting, and analysis helpers.

This file fills coverage gaps identified after the v0.3.0 release:
- Edge cases: empty spike trains, single-neuron collections, zero-rate scenarios
- Serialization round-trips: Trial, FitResult, FitResSummary
- FitResult/FitResSummary plotting: all plot methods
- Analysis helpers: computeHistLag, computeHistLagForAll, Granger, spikeTrigAvg
- Kalman and PP EM: basic smoke tests
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nstat
from nstat import (
    Analysis,
    CIF,
    ConfigColl,
    CovColl,
    Covariate,
    DecodingAlgorithms,
    Events,
    FitResult,
    FitSummary,
    History,
    SignalObj,
    Trial,
    TrialConfig,
    nspikeTrain,
    nstColl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_trial():
    """Build a minimal Trial with 2 neurons, 1 covariate, 1000 Hz."""
    t = np.arange(0.0, 1.0, 0.001)
    stim = np.sin(2 * np.pi * 5 * t)
    cov = Covariate(t, stim, "stim", "time", "s", "a.u.", ["stim"])

    np.random.seed(42)
    spikes1 = np.sort(np.random.uniform(0, 1, 30))
    spikes2 = np.sort(np.random.uniform(0, 1, 25))
    n1 = nspikeTrain(spikes1, "n1", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
    n2 = nspikeTrain(spikes2, "n2", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)

    trial = Trial(nstColl([n1, n2]), CovColl([cov]))
    return trial


@pytest.fixture
def fit_result(simple_trial):
    """Run a simple GLM analysis and return a FitResult."""
    cfgs = ConfigColl([TrialConfig([["stim", "stim"]], sampleRate=1000.0, name="m1")])
    results = Analysis.RunAnalysisForAllNeurons(simple_trial, cfgs, makePlot=0)
    return results[0]


@pytest.fixture
def fit_summary(fit_result):
    """Build a FitSummary from a single FitResult."""
    return FitSummary([fit_result])


# ---------------------------------------------------------------------------
# 1. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_spike_train_statistics(self) -> None:
        """Empty spike train should not error on basic statistics."""
        nst = nspikeTrain([], "empty", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
        assert nst.n_spikes == 0
        isis = nst.getISIs()
        assert len(isis) == 0

    def test_single_spike_train_collection(self) -> None:
        """nstColl with one spike train should work without error."""
        nst = nspikeTrain([0.1, 0.5, 0.9], "only", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
        coll = nstColl([nst])
        assert coll.numSpikeTrains == 1
        assert coll.getNST(1).name == "only"
        mat = coll.dataToMatrix([1], 0.1, 0.0, 1.0)
        assert mat.shape[1] == 1

    def test_single_spike_train_psth(self) -> None:
        """PSTH on single-neuron collection should still work."""
        nst = nspikeTrain([0.1, 0.5, 0.9], "n1", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
        coll = nstColl([nst])
        psth = coll.psth(0.1, [1], 0.0, 1.0)
        # nstColl.psth returns nstat.core.Covariate; check by class name to avoid module alias issues
        assert psth.__class__.__name__ == "Covariate"
        assert psth.data.shape[0] == len(psth.time)

    def test_spike_train_with_one_spike(self) -> None:
        """Spike train with exactly one spike should compute statistics safely."""
        nst = nspikeTrain([0.5], "one", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
        assert nst.n_spikes == 1
        assert len(nst.getISIs()) == 0

    def test_covariate_collection_empty(self) -> None:
        """Empty CovariateCollection should not error."""
        coll = CovColl()
        assert coll.numCov == 0

    def test_trial_with_no_covariates(self) -> None:
        """Trial requires CovColl — verify clear error when omitted."""
        nst = nspikeTrain([0.1], "n1", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
        with pytest.raises(ValueError, match="CovColl is a required argument"):
            Trial(nstColl([nst]))

    def test_signal_obj_zero_length(self) -> None:
        """SignalObj validates that dataLabels match data dimensions."""
        # Zero-length but with matching 1-D label
        sig = SignalObj(np.array([]), np.array([]).reshape(-1, 1), "empty", "time", "s", "u", ["x"])
        assert sig.data.size == 0


# ---------------------------------------------------------------------------
# 2. Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerializationRoundTrips:
    def test_trial_tostructure_fromstructure(self, simple_trial) -> None:
        """Trial should survive toStructure/fromStructure round-trip."""
        structure = simple_trial.toStructure()
        restored = Trial.fromStructure(structure)

        assert restored.getNumUniqueNeurons() == simple_trial.getNumUniqueNeurons()
        assert restored.covarColl.numCov == simple_trial.covarColl.numCov
        np.testing.assert_allclose(
            restored.spikeColl.getNST(1).spikeTimes,
            simple_trial.spikeColl.getNST(1).spikeTimes,
            rtol=1e-12,
        )

    def test_fitresult_tostructure_fromstructure(self, fit_result) -> None:
        """FitResult should survive toStructure/fromStructure round-trip."""
        structure = fit_result.toStructure()
        restored = FitResult.fromStructure(structure)

        assert restored.numResults == fit_result.numResults
        np.testing.assert_allclose(
            restored.AIC.reshape(-1),
            fit_result.AIC.reshape(-1),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            restored.BIC.reshape(-1),
            fit_result.BIC.reshape(-1),
            rtol=1e-8,
        )

    def test_fitsummary_tostructure_fromstructure(self, fit_summary) -> None:
        """FitSummary should survive toStructure/fromStructure round-trip."""
        structure = fit_summary.toStructure()
        restored = FitSummary.fromStructure(structure)

        assert restored.numNeurons == fit_summary.numNeurons
        np.testing.assert_allclose(
            restored.AIC.reshape(-1),
            fit_summary.AIC.reshape(-1),
            rtol=1e-8,
        )

    def test_nstcoll_tostructure_fromstructure(self) -> None:
        """nstColl should survive toStructure/fromStructure round-trip."""
        n1 = nspikeTrain([0.1, 0.5], "n1", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
        n2 = nspikeTrain([0.2, 0.8], "n2", 0.001, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
        coll = nstColl([n1, n2])
        structure = coll.toStructure()
        restored = nstColl.fromStructure(structure)
        assert restored.numSpikeTrains == 2
        np.testing.assert_allclose(
            restored.getNST(1).spikeTimes,
            n1.spikeTimes,
            rtol=1e-12,
        )

    def test_events_tostructure_fromstructure(self) -> None:
        """Events should survive round-trip."""
        ev = Events([0.1, 0.5, 0.9], ["start", "mid", "end"], "red")
        structure = ev.toStructure()
        restored = Events.fromStructure(structure)
        np.testing.assert_allclose(restored.eventTimes, ev.eventTimes, rtol=1e-12)
        assert restored.eventLabels == ev.eventLabels

    def test_history_tostructure_fromstructure(self) -> None:
        """History should survive round-trip."""
        h = History([0.0, 0.01, 0.02])
        structure = h.toStructure()
        restored = History.fromStructure(structure)
        np.testing.assert_allclose(restored.windowTimes, h.windowTimes, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. FitResult / FitResSummary plotting completeness
# ---------------------------------------------------------------------------

class TestFitResultPlotting:
    def test_plotresults_returns_figure(self, fit_result) -> None:
        fig = fit_result.plotResults()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3
        plt.close("all")

    def test_ksplot_returns_axes(self, fit_result) -> None:
        ax = fit_result.KSPlot()
        assert hasattr(ax, "plot")
        plt.close("all")

    def test_plotresidual_returns_axes(self, fit_result) -> None:
        ax = fit_result.plotResidual()
        assert hasattr(ax, "plot")
        plt.close("all")

    def test_plotinvgaustrans_returns_axes(self, fit_result) -> None:
        ax = fit_result.plotInvGausTrans()
        assert hasattr(ax, "plot")
        plt.close("all")

    def test_plotseqcorr_returns_axes(self, fit_result) -> None:
        ax = fit_result.plotSeqCorr()
        assert hasattr(ax, "plot")
        plt.close("all")

    def test_plotcoeffs_returns_axes(self, fit_result) -> None:
        ax = fit_result.plotCoeffs()
        assert hasattr(ax, "plot") or hasattr(ax, "bar")
        plt.close("all")

    def test_plotcoeffswithouthistory_returns_axes(self, fit_result) -> None:
        ax = fit_result.plotCoeffsWithoutHistory()
        assert hasattr(ax, "plot") or hasattr(ax, "bar")
        plt.close("all")


class TestFitSummaryPlotting:
    def test_plotsummary_returns_figure(self, fit_summary) -> None:
        fig = fit_summary.plotSummary()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plotIC_returns_figure(self, fit_summary) -> None:
        fig = fit_summary.plotIC()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plotAIC_returns_axes(self, fit_summary) -> None:
        ax = fit_summary.plotAIC()
        assert hasattr(ax, "boxplot") or hasattr(ax, "plot")
        plt.close("all")

    def test_plotBIC_returns_axes(self, fit_summary) -> None:
        ax = fit_summary.plotBIC()
        assert hasattr(ax, "boxplot") or hasattr(ax, "plot")
        plt.close("all")

    def test_plotlogLL_returns_axes(self, fit_summary) -> None:
        ax = fit_summary.plotlogLL()
        assert hasattr(ax, "boxplot") or hasattr(ax, "plot")
        plt.close("all")

    def test_plotResidualSummary_returns_figure(self, fit_summary) -> None:
        fig = fit_summary.plotResidualSummary()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_boxPlot_returns_axes(self, fit_summary) -> None:
        coeffs, labels, se = fit_summary.getCoeffs(1)
        ax = fit_summary.boxPlot(coeffs, dataLabels=labels)
        assert hasattr(ax, "boxplot") or hasattr(ax, "plot")
        plt.close("all")

    def test_binCoeffs_returns_valid(self, fit_summary) -> None:
        bins, edges, percent_sig = fit_summary.binCoeffs(-5.0, 5.0, 1.0)
        assert bins.ndim == 2
        assert edges.ndim == 1
        assert np.all((0.0 <= percent_sig) & (percent_sig <= 1.0))

    def test_plotAllCoeffs_returns_axes(self, fit_summary) -> None:
        ax = fit_summary.plotAllCoeffs()
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# 4. Analysis helper methods
# ---------------------------------------------------------------------------

class TestAnalysisHelpers:
    def test_computeHistLag_basic(self, simple_trial) -> None:
        """computeHistLag should run without error and return (FitResult, ConfigColl)."""
        result, tcc = Analysis.computeHistLag(
            simple_trial,
            neuronNum=1,
            windowTimes=[0.0, 0.01],
            CovLabels=[["stim", "stim"]],
            Algorithm="GLM",
            batchMode=1,
            sampleRate=1000.0,
            makePlot=0,
        )
        assert result is not None
        assert result.__class__.__name__ == "FitResult"
        assert tcc.__class__.__name__ == "ConfigCollection"

    def test_computeHistLagForAll_basic(self, simple_trial) -> None:
        """computeHistLagForAll should run for all neurons."""
        results = Analysis.computeHistLagForAll(
            simple_trial,
            windowTimes=[0.0, 0.01],
            CovLabels=[["stim", "stim"]],
            Algorithm="GLM",
            batchMode=1,
            sampleRate=1000.0,
            makePlot=0,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_spikeTrigAvg_basic(self, simple_trial) -> None:
        """spikeTrigAvg should return a cross-correlation-like signal."""
        cc = Analysis.spikeTrigAvg(simple_trial, neuronNum=1, windowSize=0.02)
        assert cc is not None

    def test_psth_function(self) -> None:
        """Analysis.psth should return counts and centers."""
        spikes = [nspikeTrain([0.1, 0.3, 0.5, 0.7, 0.9], "n1", 0.001, 0.0, 1.0)]
        bins = np.arange(0.0, 1.1, 0.2)
        counts, centers = Analysis.psth(spikes, bins)
        assert counts.shape[0] == len(bins) - 1
        assert centers.shape == counts.shape


# ---------------------------------------------------------------------------
# 5. Kalman and PP EM smoke tests
# ---------------------------------------------------------------------------

class TestEMSmoke:
    def test_kf_em_basic(self) -> None:
        """KF_EM should run a simple linear Gaussian state-space model."""
        np.random.seed(123)
        T = 100
        A0 = np.eye(1) * 0.95
        Q0 = np.eye(1) * 0.1
        C0 = np.eye(1)
        R0 = np.eye(1) * 0.5
        alpha0 = np.zeros((1, 1))
        x0 = np.zeros((1, 1))
        Px0 = np.eye(1) * 1.0

        # Simulate
        x_true = np.zeros((1, T))
        y = np.zeros((1, T))
        for t in range(1, T):
            x_true[:, t] = A0 @ x_true[:, t - 1] + np.random.randn(1) * np.sqrt(0.1)
            y[:, t] = C0 @ x_true[:, t] + np.random.randn(1) * np.sqrt(0.5)

        constraints = DecodingAlgorithms.KF_EMCreateConstraints()
        result = DecodingAlgorithms.KF_EM(
            y, A0, Q0, C0, R0, alpha0, x0, Px0, constraints
        )
        # Should return a tuple of results
        assert isinstance(result, tuple)
        assert len(result) >= 10
        xK = result[0]
        assert xK.shape[1] == T

    def test_pp_em_create_constraints(self) -> None:
        """PP_EMCreateConstraints should return a dict-like object."""
        c = DecodingAlgorithms.PP_EMCreateConstraints()
        assert c is not None
        # Should have standard fields
        assert hasattr(c, "__getitem__") or isinstance(c, dict)


# ---------------------------------------------------------------------------
# 6. CIF additional coverage
# ---------------------------------------------------------------------------

class TestCIFCoverage:
    def test_cif_copy_preserves_state(self) -> None:
        """CIFCopy should create an independent copy."""
        b = np.array([1.0, 0.5])
        cif = CIF(b, ["const", "stim"], ["stim"], "poisson")
        copy = cif.CIFCopy()
        np.testing.assert_allclose(copy.b, cif.b)
        # Modify original, copy should be unchanged
        cif.b[0] = 99.0
        assert copy.b[0] != 99.0

    def test_cif_eval_gradient_and_jacobian(self) -> None:
        """CIF gradient and Jacobian methods should return arrays."""
        b = np.array([2.0, 0.3])
        cif = CIF(b, ["const", "stim"], ["stim"], "poisson")
        # CIF with 2 params (const, stim) expects 2 stimulus values
        stim = np.array([1.0, 0.5])

        ld = cif.evalLambdaDelta(stim)
        assert np.isfinite(ld)

        grad = cif.evalGradient(stim)
        assert grad.size >= 1

        jac = cif.evalJacobian(stim)
        assert jac.ndim == 2

    def test_cif_log_gradient_and_jacobian(self) -> None:
        """Log variants of gradient/Jacobian should also work."""
        b = np.array([2.0, 0.3])
        cif = CIF(b, ["const", "stim"], ["stim"], "poisson")
        # CIF with 2 params (const, stim) expects 2 stimulus values
        stim = np.array([1.0, 0.5])

        grad_log = cif.evalGradientLog(stim)
        assert grad_log.size >= 1

        jac_log = cif.evalJacobianLog(stim)
        assert jac_log.ndim == 2


# ---------------------------------------------------------------------------
# 7. SignalObj additional coverage
# ---------------------------------------------------------------------------

class TestSignalObjCoverage:
    def test_signalobj_shift_and_align(self) -> None:
        """shift and alignTime should produce correct time offsets."""
        t = np.arange(0.0, 1.0, 0.01)
        sig = SignalObj(t, np.sin(t), "test", "time", "s", "u", ["x"])
        shifted = sig.shift(0.5)
        assert shifted.time[0] == pytest.approx(0.5, abs=1e-10)

    def test_signalobj_power_and_sqrt(self) -> None:
        """power and sqrt should preserve signal structure."""
        t = np.arange(0.0, 1.0, 0.01)
        data = np.abs(np.sin(t)) + 0.1  # positive values
        sig = SignalObj(t, data, "test", "time", "s", "u", ["x"])
        sq = sig.power(2)
        assert sq.data.shape == sig.data.shape
        rt = sig.sqrt()
        assert rt.data.shape == sig.data.shape
        np.testing.assert_allclose(rt.data[:, 0] ** 2, sig.data[:, 0], rtol=1e-10)

    def test_signalobj_xcov(self) -> None:
        """xcov (cross-covariance) should return a signal."""
        t = np.arange(0.0, 1.0, 0.01)
        sig1 = SignalObj(t, np.sin(t), "s1", "time", "s", "u", ["x"])
        sig2 = SignalObj(t, np.cos(t), "s2", "time", "s", "u", ["x"])
        xcov_result = sig1.xcov(sig2, 10)
        assert xcov_result is not None

    def test_mtmspectrum_returns_psd(self) -> None:
        """MTMspectrum should return (freqs, psd, tapers) with correct shapes."""
        t = np.arange(0.0, 1.0, 0.001)
        sig = SignalObj(t, np.sin(2 * np.pi * 50 * t), "test", "time", "s", "u", ["x"])
        freqs, psd, tapers = sig.MTMspectrum()
        assert freqs.shape == psd.shape
        assert freqs.size > 0
        assert tapers.ndim == 2

    def test_spectrogram_returns_three_arrays(self) -> None:
        """spectrogram should return (f, t, Sxx)."""
        t = np.arange(0.0, 1.0, 0.001)
        sig = SignalObj(t, np.sin(2 * np.pi * 50 * t), "test", "time", "s", "u", ["x"])
        f, t_spec, sxx = sig.spectrogram()
        assert f.size > 0
        assert t_spec.size > 0
        assert sxx.shape == (f.size, t_spec.size)

    def test_periodogram_returns_psd(self) -> None:
        """periodogram should return (freqs, psd)."""
        t = np.arange(0.0, 1.0, 0.001)
        sig = SignalObj(t, np.sin(2 * np.pi * 50 * t), "test", "time", "s", "u", ["x"])
        freqs, psd = sig.periodogram()
        assert freqs.shape == psd.shape


# ---------------------------------------------------------------------------
# 8. Trial plotting
# ---------------------------------------------------------------------------

class TestTrialPlotting:
    def test_trial_plot_returns_axes(self, simple_trial) -> None:
        ax = simple_trial.plot()
        assert ax is not None
        plt.close("all")

    def test_trial_plotraster_returns_axes(self, simple_trial) -> None:
        ax = simple_trial.plotRaster()
        assert ax is not None
        plt.close("all")

    def test_trial_plotcovariates_returns_axes(self, simple_trial) -> None:
        result = simple_trial.plotCovariates()
        assert result is not None
        plt.close("all")
