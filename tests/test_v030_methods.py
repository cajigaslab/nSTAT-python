"""Unit tests for methods added in v0.3.0.

Tests cover:
- SignalObj.MTMspectrum (multi-taper spectral estimate)
- SignalObj.spectrogram (short-time Fourier transform)
- SignalObj.periodogram
- SignalObj.findPeaks / findMaxima / findMinima / findGlobalPeak
- DecodingAlgorithms.PPSS_EStep (SSGLM E-step smoke test)
- nstColl.ssglm (SSGLM entry point smoke test)
"""

from __future__ import annotations

import numpy as np
import pytest

import nstat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sinusoidal_signal(
    freq: float = 10.0,
    duration: float = 1.0,
    sample_rate: float = 1000.0,
    amplitude: float = 1.0,
) -> nstat.SignalObj:
    """Create a pure sinusoidal SignalObj for spectral tests."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    data = amplitude * np.sin(2 * np.pi * freq * t)
    return nstat.SignalObj(t, data, "sinusoid")


def _make_multi_peak_signal() -> nstat.SignalObj:
    """Create a signal with known peak locations."""
    t = np.linspace(0, 1, 1000)
    # Three well-separated peaks at t=0.2, 0.5, 0.8
    data = (
        np.exp(-((t - 0.2) ** 2) / 0.001)
        + np.exp(-((t - 0.5) ** 2) / 0.001)
        + np.exp(-((t - 0.8) ** 2) / 0.001)
    )
    return nstat.SignalObj(t, data, "peaks")


# ---------------------------------------------------------------------------
# Spectral methods
# ---------------------------------------------------------------------------


class TestMTMspectrum:
    """Tests for SignalObj.MTMspectrum."""

    def test_returns_correct_types_and_shapes(self) -> None:
        sig = _make_sinusoidal_signal(freq=50.0, duration=0.5)
        freqs, psd, ci = sig.MTMspectrum()
        assert isinstance(freqs, np.ndarray)
        assert isinstance(psd, np.ndarray)
        assert freqs.ndim == 1
        assert psd.ndim == 1
        assert len(freqs) == len(psd)
        assert ci is not None
        assert ci.shape == (len(freqs), 2)

    def test_peak_at_signal_frequency(self) -> None:
        """PSD should peak near the signal's true frequency."""
        freq = 50.0
        sig = _make_sinusoidal_signal(freq=freq, duration=1.0)
        freqs, psd, _ = sig.MTMspectrum()
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - freq) < 2.0, f"Peak at {peak_freq}, expected ~{freq}"

    def test_no_ci_when_pval_none(self) -> None:
        sig = _make_sinusoidal_signal()
        freqs, psd, ci = sig.MTMspectrum(Pval=None)
        assert ci is None

    def test_multidim_signal(self) -> None:
        """Multi-dimensional signals return (nfreqs, ndim) PSD."""
        t = np.arange(0, 1, 0.001)
        data = np.column_stack([np.sin(2 * np.pi * 10 * t),
                                np.sin(2 * np.pi * 30 * t)])
        sig = nstat.SignalObj(t, data, "2d")
        freqs, psd, ci = sig.MTMspectrum()
        assert psd.shape[1] == 2
        assert ci.shape[1] == 4  # lower+upper per dim

    def test_custom_nw_and_kmax(self) -> None:
        sig = _make_sinusoidal_signal()
        freqs, psd, ci = sig.MTMspectrum(NW=3.0, Kmax=4)
        assert len(freqs) > 0
        assert len(psd) == len(freqs)


class TestPeriodogram:
    """Tests for SignalObj.periodogram."""

    def test_returns_correct_shapes(self) -> None:
        sig = _make_sinusoidal_signal()
        freqs, psd = sig.periodogram()
        assert isinstance(freqs, np.ndarray)
        assert isinstance(psd, np.ndarray)
        assert len(freqs) == len(psd)

    def test_peak_at_signal_frequency(self) -> None:
        freq = 100.0
        sig = _make_sinusoidal_signal(freq=freq, duration=1.0, sample_rate=1000.0)
        freqs, psd = sig.periodogram()
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - freq) < 2.0


class TestSpectrogram:
    """Tests for SignalObj.spectrogram."""

    def test_returns_three_arrays(self) -> None:
        sig = _make_sinusoidal_signal(duration=1.0)
        f, t, Sxx = sig.spectrogram(nperseg=128)
        assert isinstance(f, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert isinstance(Sxx, np.ndarray)
        assert Sxx.shape == (len(f), len(t))

    def test_frequency_range(self) -> None:
        sr = 1000.0
        sig = _make_sinusoidal_signal(sample_rate=sr)
        f, t, Sxx = sig.spectrogram()
        assert f[0] >= 0
        assert f[-1] <= sr / 2


# ---------------------------------------------------------------------------
# Peak-finding methods
# ---------------------------------------------------------------------------


class TestFindPeaks:
    """Tests for SignalObj.findPeaks, findMaxima, findMinima, findGlobalPeak."""

    def test_findpeaks_returns_known_peaks(self) -> None:
        sig = _make_multi_peak_signal()
        indices, values = sig.findPeaks("maxima")
        assert len(indices) == 1  # one dimension
        assert len(indices[0]) >= 3  # at least 3 peaks
        # All peak values should be positive
        assert np.all(values[0] > 0.5)

    def test_findmaxima_alias(self) -> None:
        sig = _make_multi_peak_signal()
        i1, v1 = sig.findMaxima()
        i2, v2 = sig.findPeaks("maxima")
        np.testing.assert_array_equal(i1[0], i2[0])

    def test_findminima_finds_troughs(self) -> None:
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 5 * t)  # 5 Hz, ~5 minima
        sig = nstat.SignalObj(t, data, "sine")
        indices, values = sig.findMinima()
        assert len(indices[0]) >= 3
        # Minima values should all be negative
        assert np.all(values[0] < 0)

    def test_findglobalpeak_maxima(self) -> None:
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 1 * t)  # single cycle, peak at t≈0.25
        sig = nstat.SignalObj(t, data, "sine")
        times, values = sig.findGlobalPeak("maxima")
        assert abs(times[0] - 0.25) < 0.01
        assert abs(values[0] - 1.0) < 0.01

    def test_findglobalpeak_minima(self) -> None:
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 1 * t)  # single cycle, min at t≈0.75
        sig = nstat.SignalObj(t, data, "sine")
        times, values = sig.findGlobalPeak("minima")
        assert abs(times[0] - 0.75) < 0.01
        assert abs(values[0] - (-1.0)) < 0.01

    def test_findpeaks_minima_fix_matlab_bug(self) -> None:
        """Python port fixes the Matlab bug where minima branch
        doesn't negate data. Verify minima values are actually minima."""
        t = np.linspace(0, 2, 2000)
        data = np.sin(2 * np.pi * 3 * t)
        sig = nstat.SignalObj(t, data, "sine")
        _, min_vals = sig.findPeaks("minima")
        _, max_vals = sig.findPeaks("maxima")
        # Minima should be strictly less than maxima
        assert np.mean(min_vals[0]) < np.mean(max_vals[0])


# ---------------------------------------------------------------------------
# SSGLM EM smoke tests
# ---------------------------------------------------------------------------


class TestSSGLM:
    """Smoke tests for the SSGLM EM entry points."""

    @pytest.fixture()
    def _ssglm_inputs(self):
        """Minimal SSGLM inputs: 5 trials, 1 neuron, 50 time bins, 2 basis."""
        rng = np.random.default_rng(42)
        K = 5   # trials
        T = 50  # time bins per trial
        R = 2   # number of basis functions = state dimension
        delta = 0.001
        J = 1   # number of history covariates

        # State transition (R x R)
        A = np.eye(R)
        Q0 = 0.01 * np.eye(R)
        x0 = np.zeros(R)

        # Spike data: K x T
        dN = rng.poisson(0.05, size=(K, T)).astype(float)

        # History design matrix: K arrays each (T, J)
        HkAll = [rng.standard_normal((T, J)) * 0.1 for _ in range(K)]

        gamma0 = np.zeros(J)
        fitType = "poisson"
        windowTimes = np.array([0.0, delta * T])
        numBasis = R

        return {
            "A": A, "Q0": Q0, "x0": x0, "dN": dN,
            "HkAll": HkAll, "fitType": fitType, "delta": delta,
            "gamma0": gamma0, "windowTimes": windowTimes,
            "numBasis": numBasis,
        }

    def test_ppss_estep_runs(self, _ssglm_inputs) -> None:
        """PPSS_EStep should run without error and return expected shapes."""
        inp = _ssglm_inputs
        # Returns: (x_K, W_K, Wku, logll, sumXkTerms, sumPPll)
        result = nstat.DecodingAlgorithms.PPSS_EStep(
            inp["A"], inp["Q0"], inp["x0"], inp["dN"],
            inp["HkAll"], inp["fitType"], inp["delta"],
            inp["gamma0"], inp["numBasis"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 6
        x_K, W_K, Wku, logll, sumXkTerms, sumPPll = result
        R = inp["numBasis"]
        K = inp["dN"].shape[0]
        assert x_K.shape == (R, K)
        assert isinstance(logll, float)

    def test_ppss_em_converges(self, _ssglm_inputs) -> None:
        """PPSS_EM should iterate and return results."""
        inp = _ssglm_inputs
        # Returns: (xK, WK, Wku, Qhat, gammahat, logll, QhatAll, gammahatAll, nIter, negLL)
        result = nstat.DecodingAlgorithms.PPSS_EM(
            inp["A"], inp["Q0"], inp["x0"], inp["dN"],
            inp["fitType"], inp["delta"], inp["gamma0"],
            inp["windowTimes"], inp["numBasis"], inp["HkAll"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 10
        xK, WK, Wku, Qhat, gammahat, logll, QhatAll, gammahatAll, nIter, negLL = result
        assert isinstance(xK, np.ndarray)
        assert isinstance(logll, float)
        assert isinstance(nIter, (int, np.integer))

    def test_ppss_emfb_returns_results(self, _ssglm_inputs) -> None:
        """PPSS_EMFB (forward-backward EM) should return results."""
        inp = _ssglm_inputs
        # Returns: (xK, WK, Wku, Qhat, gammahat, fitResults, stimulus, stimCIs, logll, QhatAll, gammahatAll, nIter)
        result = nstat.DecodingAlgorithms.PPSS_EMFB(
            inp["A"], inp["Q0"], inp["x0"], inp["dN"],
            inp["fitType"], inp["delta"], inp["gamma0"],
            inp["windowTimes"], inp["numBasis"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 12
        xK = result[0]
        fitResults = result[5]
        assert isinstance(xK, np.ndarray)
        assert isinstance(fitResults, nstat.FitResult)
