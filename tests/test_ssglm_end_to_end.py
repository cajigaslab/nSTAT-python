"""End-to-end numerical tests for the state-space GLM (``ssglmFB``).

The rest of the suite only signature-checks the SSGLM (see
``test_audit_bug_fixes.py``). These tests actually *fit* an SSGLM from spike
data, so the EM initialization + forward-backward path is exercised in CI.

They are regression tests for the three MATLAB->Python port bugs fixed in
``nstat/trial.py`` (cajigaslab/nSTAT-python#248):
  1. ``_psth_glm_coeffs`` read fit index 1 instead of 0;
  2. ``estimateVarianceAcrossTrials`` indexed columns by realization number in
     the backward loop;
  3. ``estimateVarianceAcrossTrials`` propagated -120 empty-bin sentinels into Q,
     exploding the state-noise variance;
and for the ``PPSS_EStep`` covariance-update regularization that keeps the EM
from aborting on an ``eigh`` non-convergence for high-rate / sparse units.
"""

from __future__ import annotations

import numpy as np
import pytest

from nstat import nspikeTrain, nstColl


def _poisson_trials(rate_hz, n_trials, dur=1.5, sample_rate=1000.0, seed=0):
    """A collection of homogeneous-Poisson trials (movements), one nspikeTrain each."""
    rng = np.random.default_rng(seed)
    trains = [
        nspikeTrain(
            np.sort(rng.uniform(0.0, dur, rng.poisson(rate_hz * dur))),
            sampleRate=sample_rate,
            minTime=0.0,
            maxTime=dur,
        )
        for _ in range(n_trials)
    ]
    return nstColl(trains)


def test_estimate_variance_across_trials_is_physical():
    """Q must be finite and O(1) on clean data (bug 3: it was ~1e3-1e4)."""
    coll = _poisson_trials(rate_hz=12.0, n_trials=40, seed=1)
    Q = np.asarray(coll.estimateVarianceAcrossTrials(8, None, 4, "poisson"))
    assert Q.shape == (8, 8)
    assert np.all(np.isfinite(Q))
    # State-noise variance for a stationary unit should be small, not exploding.
    assert np.median(np.diag(Q)) < 10.0, np.diag(Q)


def test_ssglmFB_runs_and_recovers_rate():
    """Full ssglmFB fit: documented return shape + a sane recovered state.

    Exercises all three fixed bugs (init crashes) plus a stable EM.
    """
    rate_hz, n_trials, dur, sr, nb = 14.0, 40, 1.5, 1000.0, 6
    coll = _poisson_trials(rate_hz, n_trials, dur=dur, sample_rate=sr, seed=2)

    out = coll.ssglmFB(windowTimes=None, numBasis=nb, numVarEstIter=4)
    assert len(out) == 12, "ssglmFB should return the documented 12-tuple"

    xK = np.asarray(out[0], dtype=float)      # across-trial log-rate state
    Qhat = np.asarray(out[3], dtype=float)
    assert xK.shape == (nb, n_trials)
    assert np.all(np.isfinite(xK))
    assert np.all(np.isfinite(Qhat))

    # Median recovered rate (exp(state) / delta) should land near the truth.
    recovered_hz = float(np.exp(np.median(xK)) * sr)
    assert 0.4 * rate_hz < recovered_hz < 2.5 * rate_hz, recovered_hz


def test_ssglmFB_does_not_abort_on_high_rate_bursty_unit():
    """Regularized E-step must not raise eigh non-convergence on a hard unit.

    High rate + a sparse silent bin is the configuration that previously made
    ``np.linalg.eigh`` throw "Eigenvalues did not converge" mid-EM.
    """
    rng = np.random.default_rng(3)
    dur, sr = 1.5, 1000.0
    trains = []
    for _ in range(30):
        # Bursty high-rate spikes in the first half, silent second half (sparse bins).
        burst = np.sort(rng.uniform(0.0, 0.6, rng.poisson(120 * 0.6)))
        trains.append(nspikeTrain(burst, sampleRate=sr, minTime=0.0, maxTime=dur))
    coll = nstColl(trains)

    out = coll.ssglmFB(windowTimes=None, numBasis=6, numVarEstIter=4)
    xK = np.asarray(out[0], dtype=float)
    assert np.all(np.isfinite(xK))
    # State stays within the physical bound enforced by the regularization.
    assert np.max(np.abs(xK)) <= 50.0 + 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
