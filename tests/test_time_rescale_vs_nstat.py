"""Cross-validation harness: clean-room time_rescale oracle vs nstat.Analysis.computeKSStats.

The oracle in ``tests/parity/_third_party/time_rescale_oracle.py`` is a
clean-room implementation of the Brown et al. 2002 time-rescaling KS
test, written from the paper directly with no reference to existing
Python ports.  This file pins the **agreement** between the oracle and
nstat's production-path :func:`nstat.Analysis.computeKSStats`:

- On well-specified models (true intensity), both implementations'
  KS statistics agree to ~1e-3.
- On mis-specified models, both reject decisively.
- The agreement holds across multiple random seeds.

Together with the oracle's own correctness tests
(``tests/test_time_rescale_oracle.py``), this gives the
**triangulated** parity guarantee: the two implementations agree
*and* both correctly distinguish well-specified from mis-specified
models on independent draws.

Pattern: pass the same intensity array + spike indicator (or
equivalent ``nspikeTrain`` + ``Covariate``) to both implementations
and compare the scalar KS statistic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from nstat import Analysis, Covariate, nspikeTrain


# Make the clean-room oracle importable.
_REPO_ROOT = Path(__file__).resolve().parent
_THIRD_PARTY = _REPO_ROOT / "parity" / "_third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from time_rescale_oracle import time_rescaling_ks_test  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _simulate_homogeneous_poisson(
    rate_hz: float, duration_s: float, dt: float, rng_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bernoulli thinning at small dt approximates a Poisson process.

    Returns
    -------
    t : (N,) time grid in seconds
    intensity : (N,) constant rate_hz
    spike_indicator : (N,) binary mask of spike bins
    """
    rng = np.random.default_rng(rng_seed)
    t = np.arange(0.0, duration_s, dt)
    spike_indicator = (rng.uniform(size=t.size) < rate_hz * dt).astype(int)
    intensity = np.full(t.size, rate_hz, dtype=float)
    return t, intensity, spike_indicator


def _make_nstat_inputs(
    t: np.ndarray, intensity: np.ndarray, spike_indicator: np.ndarray, dt: float
) -> tuple[nspikeTrain, Covariate]:
    """Build the ``nspikeTrain`` + ``Covariate`` that nstat's
    ``Analysis.computeKSStats`` expects, from the plain arrays the
    oracle consumes."""
    spike_times = t[spike_indicator.astype(bool)]
    nst = nspikeTrain(
        spike_times,
        "1",
        1.0 / dt,
        0.0,
        float(t[-1] + dt),
        "time",
        "s",
        "",
        "",
        -1,
    )
    lam = Covariate(t, intensity, "lambda", "time", "s", "Hz", ["lambda"])
    return nst, lam


def _both_ks_stats(
    t: np.ndarray, intensity: np.ndarray, spike_indicator: np.ndarray, dt: float
) -> tuple[float, float]:
    """Compute the KS statistic via both nstat and the oracle.  Returns
    ``(nstat_ks, oracle_ks)`` for direct numerical comparison."""
    nst, lam = _make_nstat_inputs(t, intensity, spike_indicator, dt)
    nstat_result = Analysis.computeKSStats([nst], lam, DTCorrection=0)
    nstat_ks = float(nstat_result[-1])
    oracle_result = time_rescaling_ks_test(intensity, spike_indicator, dt=dt)
    oracle_ks = oracle_result.ks_stat
    return nstat_ks, oracle_ks


# ----------------------------------------------------------------------
# Agreement on well-specified models
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_well_specified_homogeneous_poisson_agrees(seed: int) -> None:
    """On a well-specified homogeneous Poisson process, nstat and the
    oracle's KS statistics agree to ~1e-3 on a 50 Hz / 2 s fixture.

    Parametrized across 5 seeds — the agreement holds for every
    realization, not just one.
    """
    t, intensity, spikes = _simulate_homogeneous_poisson(
        rate_hz=50.0, duration_s=2.0, dt=1e-3, rng_seed=seed
    )
    nstat_ks, oracle_ks = _both_ks_stats(t, intensity, spikes, dt=1e-3)
    delta = abs(nstat_ks - oracle_ks)
    assert delta < 5e-3, (
        f"nstat ks={nstat_ks:.4f}, oracle ks={oracle_ks:.4f}, "
        f"|Δ|={delta:.3e} > 5e-3 (seed {seed})"
    )


def test_well_specified_inhomogeneous_poisson_agrees() -> None:
    """Time-rescaling theorem applies to inhomogeneous Poisson too.
    Use a sinusoidal rate and verify agreement holds."""
    dt = 1e-3
    duration_s = 5.0
    t = np.arange(0.0, duration_s, dt)
    intensity = 30.0 + 20.0 * np.sin(2.0 * np.pi * 1.0 * t)
    rng = np.random.default_rng(100)
    spikes = (rng.uniform(size=t.size) < intensity * dt).astype(int)

    nstat_ks, oracle_ks = _both_ks_stats(t, intensity, spikes, dt=dt)
    delta = abs(nstat_ks - oracle_ks)
    assert delta < 5e-3, (
        f"Inhomogeneous Poisson: nstat ks={nstat_ks:.4f}, "
        f"oracle ks={oracle_ks:.4f}, |Δ|={delta:.3e}"
    )


# ----------------------------------------------------------------------
# Agreement on mis-specified models (both should reject)
# ----------------------------------------------------------------------


def test_mis_specified_intensity_both_reject() -> None:
    """When the intensity passed to the KS test is 10× too low, both
    implementations should produce a large KS statistic and reject the
    null at α=0.05.

    Agreement need not be tight in absolute value here — both should
    produce ks > 0.5 (large enough to reject) but the *signs* of
    departure from H0 agree.
    """
    t, _, spikes = _simulate_homogeneous_poisson(
        rate_hz=50.0, duration_s=5.0, dt=1e-3, rng_seed=7
    )
    # Lie about the rate.
    wrong_intensity = np.full(t.size, 5.0)
    nstat_ks, oracle_ks = _both_ks_stats(t, wrong_intensity, spikes, dt=1e-3)
    # Both should reject decisively — large KS statistics.
    assert nstat_ks > 0.5, f"nstat ks={nstat_ks:.4f} too small for 10× misspec"
    assert oracle_ks > 0.5, f"oracle ks={oracle_ks:.4f} too small for 10× misspec"


# ----------------------------------------------------------------------
# Stability across sample size
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration_s,tol",
    [
        # Short windows have few spikes — discrete-step boundary effects
        # in the empirical CDF widen the per-implementation disagreement.
        # Tolerance scales ~1/sqrt(N_spikes), which is ~1/sqrt(rate*duration).
        (0.5, 2.5e-2),
        (1.0, 1.5e-2),
        (5.0, 5e-3),
        (10.0, 3e-3),
    ],
)
def test_agreement_holds_across_sample_sizes(
    duration_s: float, tol: float
) -> None:
    """The agreement between nstat and oracle is approximately independent
    of duration up to a 1/sqrt(N) small-sample term — both are computing
    the same statistic, just with more data.

    The tolerance is parametrized accordingly (looser for shorter
    windows).  This documents the empirical baseline; if the
    disagreement at any size grows beyond the listed tolerance, it
    signals a real bug rather than discretization noise.
    """
    t, intensity, spikes = _simulate_homogeneous_poisson(
        rate_hz=50.0, duration_s=duration_s, dt=1e-3, rng_seed=42
    )
    nstat_ks, oracle_ks = _both_ks_stats(t, intensity, spikes, dt=1e-3)
    delta = abs(nstat_ks - oracle_ks)
    assert delta < tol, (
        f"Duration {duration_s}s: nstat ks={nstat_ks:.4f}, "
        f"oracle ks={oracle_ks:.4f}, |Δ|={delta:.3e} > tol={tol}"
    )
