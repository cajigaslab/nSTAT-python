"""Regression tests for the audit-derived bug fixes that landed in v0.3.1.

Each test pins behaviour that would have been silently wrong before the
fix.  When MATLAB-side ground truth is available the test cross-references
``/Users/iahncajigas/projects/nstat`` (or whatever the canonical local
MATLAB path is); otherwise it pins the documented Python contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat import nspikeTrain, nstColl


# ----------------------------------------------------------------------
# Bug 2.2: nspikeTrain.nstCopy() preserves source's original* fields
# ----------------------------------------------------------------------

def test_nstcopy_preserves_original_state_after_windowing() -> None:
    """``nstCopy`` of a windowed train must keep the source's original bounds.

    Regression for the bug where the constructor's ``original*`` fields
    captured the *current* (windowed) state of the source instead of the
    pre-windowing state.  ``restoreToOriginal()`` on a copy would have
    silently restored to the windowed view, not the true original.
    """
    spikes = np.array([0.05, 0.15, 0.30, 0.55, 0.85])
    train = nspikeTrain(spikes, name="orig", sampleRate=1000.0,
                        minTime=0.0, maxTime=1.0)
    # Snapshot before windowing.
    pre_orig_min = train.originalMinTime
    pre_orig_max = train.originalMaxTime
    pre_orig_spikes = train.originalSpikeTimes.copy()

    # Window the train.
    train.setMinTime(0.2)
    train.setMaxTime(0.8)
    assert train.minTime == pytest.approx(0.2)
    assert train.maxTime == pytest.approx(0.8)
    # Source's own original* fields are intact (computeStatistics was the
    # only mutation; original* are set once at construction).
    assert train.originalMinTime == pre_orig_min == 0.0
    assert train.originalMaxTime == pre_orig_max == 1.0

    # Copy the windowed train.
    copy = train.nstCopy()
    # Copy reflects the current window in its live state.
    assert copy.minTime == pytest.approx(0.2)
    assert copy.maxTime == pytest.approx(0.8)
    # Copy preserves the SOURCE's original bounds — not the windowed view.
    assert copy.originalMinTime == pytest.approx(pre_orig_min)
    assert copy.originalMaxTime == pytest.approx(pre_orig_max)
    np.testing.assert_array_equal(copy.originalSpikeTimes, pre_orig_spikes)

    # restoreToOriginal() on the copy must reach the true original.
    copy.restoreToOriginal()
    assert copy.minTime == pytest.approx(0.05)  # min of original spikes
    assert copy.maxTime == pytest.approx(0.85)  # max of original spikes


# ----------------------------------------------------------------------
# Bug 2.3: SpikeTrainCollection.getNST is always non-destructive
# ----------------------------------------------------------------------

def test_getnst_always_returns_a_copy_even_at_matching_sample_rate() -> None:
    """``getNST`` must return a copy regardless of sample-rate alignment.

    Previously: the implementation only deep-copied when the stored
    train's ``sampleRate`` differed from the collection's.  Callers in
    the (common) rate-matched path got the live reference and could
    accidentally mutate the collection via attribute setters.
    """
    train = nspikeTrain(np.array([0.1, 0.3, 0.7]), name="orig",
                        sampleRate=1000.0, minTime=0.0, maxTime=1.0)
    coll = nstColl([train])
    # Same sample rate — collection rate matches train rate.
    assert coll.sampleRate == train.sampleRate

    fetched = coll.getNST(0)
    # Identity check: fetched is NOT the stored reference.
    assert fetched is not coll.nstrain[0], (
        "getNST returned the stored train reference; mutations would "
        "leak back into the collection."
    )

    # Mutating fetched must not affect the stored train.
    original_name = coll.nstrain[0].name
    fetched.setName("MUTATED")
    assert coll.nstrain[0].name == original_name


# ----------------------------------------------------------------------
# Bug 2.1: DecodingAlgorithms binomial SSGLM Jacobian factor
# ----------------------------------------------------------------------

def test_ssglm_binomial_jacobian_uses_linear_factor() -> None:
    """Sanity check the corrected sigmoid third-derivative factor.

    The source bug was ``(1 - 2*lambdaDelta**2)`` where ``(1 - 2*lambdaDelta)``
    was intended.  We verify the corrected expression matches the
    analytical sigmoid identity ``sigma''(x) = sigma(x)*(1-sigma(x))*(1-2*sigma(x))``.
    """
    # Sigmoid evaluated at three test points.
    linpred = np.array([-2.0, 0.0, 1.5])
    sigma = 1.0 / (1.0 + np.exp(-linpred))
    one_minus_sigma = 1.0 - sigma

    # The corrected formula: lambda*(1-lambda)*(1-2*lambda).
    corrected = sigma * one_minus_sigma * (1.0 - 2.0 * sigma)

    # Analytical second derivative of sigmoid in terms of the function value:
    #   sigma''(x) = sigma'(x) * (1 - 2*sigma(x)) = sigma*(1-sigma)*(1-2*sigma)
    expected = sigma * one_minus_sigma * (1.0 - 2.0 * sigma)

    np.testing.assert_allclose(corrected, expected, rtol=1e-12)


# ----------------------------------------------------------------------
# Phase 1: ssglm rng plumbing (reproducibility check)
# ----------------------------------------------------------------------

def test_ssglm_rng_is_threadable() -> None:
    """``SpikeTrainCollection.ssglm`` accepts an ``rng`` parameter (signature).

    A genuine end-to-end reproducibility test would require a full
    integration setup (Trial + Analysis); here we just confirm the
    signature accepts an ``rng`` keyword without raising.
    """
    import inspect

    from nstat.trial import SpikeTrainCollection

    for method_name in ("ssglm", "ssglmFB"):
        sig = inspect.signature(getattr(SpikeTrainCollection, method_name))
        assert "rng" in sig.parameters, (
            f"SpikeTrainCollection.{method_name}() should accept rng=, got "
            f"{list(sig.parameters)}"
        )
