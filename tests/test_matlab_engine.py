"""Tests for the MATLAB Engine Simulink bridge.

These tests verify:
- MATLAB availability detection returns a bool and is cached
- Path configuration resolves correctly
- Fallback warnings are issued when MATLAB is absent
- backend='python' never warns
- backend='matlab' raises when MATLAB is unavailable
- Integration tests (skipped without MATLAB)
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from nstat import CIF, nspikeTrain, nstColl
from nstat.matlab_engine import (
    MatlabFallbackWarning,
    get_matlab_nstat_path,
    is_matlab_available,
    set_matlab_nstat_path,
)
from nstat.signal import Covariate
from nstat.simulators import simulate_two_neuron_network


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_covariates():
    """Return (stim, ens) Covariates on a 100-ms, 1 kHz grid."""
    t = np.arange(0.0, 0.1, 0.001)
    stim_data = np.sin(2 * np.pi * 10 * t)
    ens_data = np.zeros_like(t)
    stim = Covariate(t, stim_data, "stim", "time", "s", "V", ["x"])
    ens = Covariate(t, ens_data, "ens", "time", "s", "V", ["n"])
    return stim, ens


# ---------------------------------------------------------------------------
# 1. MATLAB availability detection
# ---------------------------------------------------------------------------

class TestMatlabAvailability:
    def test_is_matlab_available_returns_bool(self) -> None:
        result = is_matlab_available()
        assert isinstance(result, bool)

    def test_is_matlab_available_is_cached(self) -> None:
        """Calling twice returns the same value (no re-probe)."""
        r1 = is_matlab_available()
        r2 = is_matlab_available()
        assert r1 is r2


# ---------------------------------------------------------------------------
# 2. Path configuration
# ---------------------------------------------------------------------------

class TestPathConfiguration:
    def test_get_matlab_nstat_path_returns_path_or_none(self) -> None:
        result = get_matlab_nstat_path()
        assert result is None or result.is_dir()

    def test_set_matlab_nstat_path_via_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("NSTAT_MATLAB_PATH", str(tmp_path))
        result = get_matlab_nstat_path()
        assert result == tmp_path

    def test_nonexistent_env_path_ignored(self, monkeypatch) -> None:
        monkeypatch.setenv("NSTAT_MATLAB_PATH", "/nonexistent/path/abc123")
        # Should fall through to sibling detection or None
        result = get_matlab_nstat_path()
        # Result is either None or a valid sibling path — not the bad env path
        if result is not None:
            assert result.is_dir()


# ---------------------------------------------------------------------------
# 3. Fallback warning behaviour
# ---------------------------------------------------------------------------

class TestFallbackWarning:
    def test_simulateCIF_auto_warns_when_matlab_absent(self) -> None:
        """When MATLAB is not available, backend='auto' should warn."""
        if is_matlab_available():
            pytest.skip("MATLAB is available — fallback not triggered")
        stim, ens = _make_simple_covariates()
        with pytest.warns(MatlabFallbackWarning):
            CIF.simulateCIF(
                -3.0, [0.0], [1.0], [0.0],
                stim, ens, 1, "binomial",
                seed=0, backend="auto",
            )

    def test_simulateCIF_python_backend_no_warning(self) -> None:
        """backend='python' should never issue MatlabFallbackWarning."""
        stim, ens = _make_simple_covariates()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CIF.simulateCIF(
                -3.0, [0.0], [1.0], [0.0],
                stim, ens, 1, "binomial",
                seed=0, backend="python",
            )
        fallback_warnings = [
            x for x in w if issubclass(x.category, MatlabFallbackWarning)
        ]
        assert len(fallback_warnings) == 0

    def test_simulateCIF_matlab_backend_raises_when_unavailable(self) -> None:
        """backend='matlab' should raise if MATLAB is not installed."""
        if is_matlab_available():
            pytest.skip("MATLAB is available — would not raise")
        stim, ens = _make_simple_covariates()
        with pytest.raises(RuntimeError):
            CIF.simulateCIF(
                -3.0, [0.0], [1.0], [0.0],
                stim, ens, 1, "binomial",
                backend="matlab",
            )

    def test_simulateCIF_invalid_backend_raises(self) -> None:
        """Invalid backend value should raise ValueError."""
        stim, ens = _make_simple_covariates()
        with pytest.raises(ValueError, match="backend must be"):
            CIF.simulateCIF(
                -3.0, [0.0], [1.0], [0.0],
                stim, ens, 1, "binomial",
                backend="invalid",
            )

    def test_simulate_network_auto_warns_when_matlab_absent(self) -> None:
        """Network simulator with backend='auto' should warn when MATLAB absent."""
        if is_matlab_available():
            pytest.skip("MATLAB is available — fallback not triggered")
        with pytest.warns(MatlabFallbackWarning):
            simulate_two_neuron_network(
                duration_s=0.1, dt=0.001, seed=0, backend="auto",
            )

    def test_simulate_network_python_backend_no_warning(self) -> None:
        """backend='python' on network simulator should never warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simulate_two_neuron_network(
                duration_s=0.1, dt=0.001, seed=0, backend="python",
            )
        fallback_warnings = [
            x for x in w if issubclass(x.category, MatlabFallbackWarning)
        ]
        assert len(fallback_warnings) == 0


# ---------------------------------------------------------------------------
# 4. Functional smoke tests (always run, using Python backend)
# ---------------------------------------------------------------------------

class TestPythonBackendSmoke:
    def test_simulateCIF_python_returns_spike_collection(self) -> None:
        """Explicit Python backend should produce valid results."""
        stim, ens = _make_simple_covariates()
        result = CIF.simulateCIF(
            -3.0, [0.0], [1.0], [0.0],
            stim, ens, 2, "binomial",
            seed=42, backend="python",
        )
        assert result.__class__.__name__ == "SpikeTrainCollection"
        assert result.numSpikeTrains == 2

    def test_simulateCIF_python_with_return_lambda(self) -> None:
        stim, ens = _make_simple_covariates()
        coll, lam = CIF.simulateCIF(
            -3.0, [0.0], [1.0], [0.0],
            stim, ens, 1, "binomial",
            seed=42, backend="python",
            return_lambda=True,
        )
        assert coll.numSpikeTrains == 1
        assert lam.__class__.__name__ == "Covariate"

    def test_simulate_network_python_returns_result(self) -> None:
        result = simulate_two_neuron_network(
            duration_s=0.1, dt=0.001, seed=42, backend="python",
        )
        assert result.spikes is not None
        assert result.lambda_delta.shape[1] == 2


# ---------------------------------------------------------------------------
# 5. Integration tests — skipped without MATLAB
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not is_matlab_available(),
    reason="MATLAB Engine not installed",
)
class TestMatlabEngineIntegration:
    def test_simulateCIF_matlab_returns_spike_collection(self) -> None:
        stim, ens = _make_simple_covariates()
        result = CIF.simulateCIF(
            -3.0, [0.0], [1.0], [0.0],
            stim, ens, 1, "binomial",
            backend="matlab",
        )
        assert result.__class__.__name__ == "SpikeTrainCollection"

    def test_simulateCIF_matlab_lambda_trace_is_finite(self) -> None:
        stim, ens = _make_simple_covariates()
        coll, lam = CIF.simulateCIF(
            -3.0, [0.0], [1.0], [0.0],
            stim, ens, 1, "binomial",
            backend="matlab",
            return_lambda=True,
        )
        assert np.all(np.isfinite(np.asarray(lam.data)))

    def test_simulate_network_matlab_returns_result(self) -> None:
        result = simulate_two_neuron_network(
            duration_s=0.5, dt=0.001, backend="matlab",
        )
        assert result.spikes is not None
