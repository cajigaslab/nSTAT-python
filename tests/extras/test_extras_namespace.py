"""Smoke + contract tests for the ``nstat.extras`` namespace.

These tests assert the **architectural** invariants of the extras
namespace.  They run in every CI matrix (no opt-deps required):

- Top-level ``nstat.extras`` imports cleanly without any opt-dep.
- Each submodule docstring + ``__all__`` is present.
- Each submodule raises a *clear, actionable* ``ImportError`` when its
  backing library is absent (the helpful-error contract documented in
  ``nstat/extras/__init__.py``).
- When the backing library IS present, the bridge round-trips a small
  fixture without loss (covered in the per-bridge functional tests below).

The functional tests are gated on the optional import via
``pytest.importorskip`` so they no-op silently in environments that
don't have the corresponding library.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import nstat.extras as extras


# ----------------------------------------------------------------------
# Architectural invariants
# ----------------------------------------------------------------------


def test_extras_namespace_imports_without_optional_deps() -> None:
    """``import nstat.extras`` must succeed with zero opt-deps installed."""
    assert hasattr(extras, "__all__")
    assert isinstance(extras.__all__, list)


def test_extras_subpackages_import_without_optional_deps() -> None:
    """Each subpackage's __init__ is import-safe — no eager opt-dep loads."""
    for name in ("nstat.extras.interop", "nstat.extras.validation", "nstat.extras.metrics"):
        mod = importlib.import_module(name)
        assert hasattr(mod, "__all__")


def test_extras_independence_no_matlab_runtime_imports() -> None:
    """Sanity: nothing in extras should import matlab.engine.

    Companion to ``tests/test_cleanroom_boundary.py``; included here so
    a regression in the extras-namespace docstrings or imports surfaces
    in the per-extras test run as well.
    """
    import re
    from pathlib import Path

    pattern = re.compile(r"\bmatlab\.engine\b")
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root / "nstat" / "extras").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not pattern.search(text), f"matlab.engine reference in {path}"


# ----------------------------------------------------------------------
# Tier-B: interop bridges — import-error pathway
# ----------------------------------------------------------------------


def _call_with_missing_dep_emits_install_hint(modname: str, callable_name: str) -> None:
    """Helper: when the backing lib is absent, calling a function in the
    module must raise ImportError whose message names the pip-install line.

    Skips if the backing lib IS installed (in which case the function
    succeeds and the import-error pathway is unreachable from here).
    """
    mod = importlib.import_module(modname)
    fn = getattr(mod, callable_name)

    # If the underlying library imports successfully, we can't exercise
    # the error path.  Skip rather than fail.
    if modname.endswith(".neo"):
        try:
            import neo  # noqa: F401
            pytest.skip("neo is installed; import-error path unreachable")
        except ImportError:
            pass
    elif modname.endswith(".pynapple"):
        try:
            import pynapple  # noqa: F401
            pytest.skip("pynapple is installed; import-error path unreachable")
        except ImportError:
            pass
    elif modname.endswith(".nwb"):
        try:
            import pynwb  # noqa: F401
            pytest.skip("pynwb is installed; import-error path unreachable")
        except ImportError:
            pass
    elif modname.endswith(".nemos_bridge"):
        try:
            import nemos  # noqa: F401
            pytest.skip("nemos is installed; import-error path unreachable")
        except ImportError:
            pass
    elif modname.endswith(".pykalman_bridge"):
        try:
            import pykalman  # noqa: F401
            pytest.skip("pykalman is installed; import-error path unreachable")
        except ImportError:
            pass
    elif modname.endswith(".spike_distances"):
        try:
            import pyspike  # noqa: F401
            pytest.skip("pyspike is installed; import-error path unreachable")
        except ImportError:
            pass

    # Call with the right number of junk args — we expect ImportError to
    # fire *before* any value validation since the dep check is the
    # first thing the function body does.
    import inspect

    n_required = sum(
        1
        for p in inspect.signature(fn).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    with pytest.raises(ImportError) as excinfo:
        fn(*[None] * n_required)
    msg = str(excinfo.value)
    assert "pip install nstat-toolbox[" in msg, (
        f"ImportError message missing pip-install hint:\n  {msg}"
    )


def test_neo_bridge_emits_install_hint_when_missing() -> None:
    _call_with_missing_dep_emits_install_hint(
        "nstat.extras.interop.neo", "to_neo_spiketrain"
    )


def test_pynapple_bridge_emits_install_hint_when_missing() -> None:
    _call_with_missing_dep_emits_install_hint(
        "nstat.extras.interop.pynapple", "to_pynapple_ts"
    )


def test_nwb_bridge_emits_install_hint_when_missing() -> None:
    _call_with_missing_dep_emits_install_hint(
        "nstat.extras.interop.nwb", "read_nwb_path"
    )


def test_nemos_bridge_emits_install_hint_when_missing() -> None:
    _call_with_missing_dep_emits_install_hint(
        "nstat.extras.validation.nemos_bridge", "cross_validate_poisson_glm"
    )


def test_pykalman_bridge_emits_install_hint_when_missing() -> None:
    _call_with_missing_dep_emits_install_hint(
        "nstat.extras.validation.pykalman_bridge", "cross_validate_kalman"
    )


def test_spike_distances_emits_install_hint_when_missing() -> None:
    _call_with_missing_dep_emits_install_hint(
        "nstat.extras.metrics.spike_distances", "isi_distance"
    )


# ----------------------------------------------------------------------
# Tier-B / C: functional round-trips (skipped when opt-dep absent)
# ----------------------------------------------------------------------


def test_neo_roundtrip_preserves_spike_times_and_window() -> None:
    pytest.importorskip("neo")
    pytest.importorskip("quantities")
    from nstat import nspikeTrain
    from nstat.extras.interop.neo import to_neo_spiketrain, from_neo_spiketrain

    nst_in = nspikeTrain(
        spikeTimes=[0.1, 0.5, 1.2, 1.8],
        name="unit_42",
        sampleRate=30_000.0,
        minTime=0.0,
        maxTime=2.0,
    )
    neo_st = to_neo_spiketrain(nst_in)
    nst_out = from_neo_spiketrain(neo_st)

    np.testing.assert_allclose(nst_out.spikeTimes, nst_in.spikeTimes)
    assert nst_out.minTime == pytest.approx(0.0)
    assert nst_out.maxTime == pytest.approx(2.0)
    assert nst_out.name == "unit_42"


def test_pynapple_roundtrip_preserves_spike_times() -> None:
    pytest.importorskip("pynapple")
    from nstat import nspikeTrain
    from nstat.extras.interop.pynapple import (
        to_pynapple_with_support,
        from_pynapple_ts,
    )

    nst_in = nspikeTrain(
        spikeTimes=[0.05, 0.3, 0.7, 1.4],
        name="cell_7",
        sampleRate=30_000.0,
        minTime=0.0,
        maxTime=2.0,
    )
    ts, support = to_pynapple_with_support(nst_in)
    nst_out = from_pynapple_ts(ts, name="cell_7", sample_rate=30_000.0, support=support)

    np.testing.assert_allclose(nst_out.spikeTimes, nst_in.spikeTimes)
    assert nst_out.minTime == pytest.approx(0.0)
    assert nst_out.maxTime == pytest.approx(2.0)
    assert nst_out.name == "cell_7"


def test_spike_distance_returns_finite_scalar() -> None:
    pytest.importorskip("pyspike")
    from nstat import nspikeTrain
    from nstat.extras.metrics.spike_distances import spike_distance, isi_distance

    a = nspikeTrain([0.1, 0.5, 0.9], minTime=0.0, maxTime=1.0)
    b = nspikeTrain([0.15, 0.55, 0.95], minTime=0.0, maxTime=1.0)
    d_spike = spike_distance(a, b)
    d_isi = isi_distance(a, b)
    assert np.isfinite(d_spike) and 0.0 <= d_spike <= 1.0
    assert np.isfinite(d_isi)


def test_kalman_comparison_assertion_fires_when_tolerance_violated() -> None:
    """The assert_* methods must actually raise — guards against the
    failure path being silently bypassed (e.g., inverted comparison).
    """
    from nstat.extras.validation.pykalman_bridge import KalmanComparison

    cmp = KalmanComparison(
        nstat_filtered_means=np.zeros((2, 2)),
        pykalman_filtered_means=np.zeros((2, 2)),
        nstat_smoothed_means=None,
        pykalman_smoothed_means=None,
        filtered_inf_norm=0.5,
        smoothed_inf_norm=2.0,
    )
    with pytest.raises(AssertionError, match="filtered means disagree"):
        cmp.assert_filtered_agree(atol=1e-3)
    with pytest.raises(AssertionError, match="smoothed means disagree"):
        cmp.assert_smoothed_agree(atol=1e-3)

    # When smoother wasn't computed, assertion should still raise (with
    # a different message) rather than silently passing.
    cmp_no_smooth = KalmanComparison(
        nstat_filtered_means=np.zeros((2, 2)),
        pykalman_filtered_means=np.zeros((2, 2)),
        nstat_smoothed_means=None,
        pykalman_smoothed_means=None,
        filtered_inf_norm=1e-12,
        smoothed_inf_norm=None,
    )
    cmp_no_smooth.assert_filtered_agree(atol=1e-6)  # passes — tiny diff
    with pytest.raises(AssertionError, match="not computed"):
        cmp_no_smooth.assert_smoothed_agree(atol=1.0)


def test_kalman_filtered_means_agree_with_pykalman() -> None:
    """nstat.DecodingAlgorithms.kalman_filter vs pykalman.

    Runs the cross-validation harness on a 100×2 linear-Gaussian
    fixture.  Tolerance is the documented empirical baseline (1e-2 for
    filter, very loose for smoother — see AUDIT D3).
    """
    pytest.importorskip("pykalman")
    from nstat.extras.validation.pykalman_bridge import cross_validate_kalman

    rng = np.random.default_rng(0)
    T, Dx, Dy = 100, 2, 2
    A = np.eye(Dx) * 0.95
    C = np.eye(Dy)
    Q = np.eye(Dx) * 0.01
    R = np.eye(Dy) * 0.1
    x0 = np.zeros(Dx)
    P0 = np.eye(Dx)

    x = np.zeros((T, Dx))
    y = np.zeros((T, Dy))
    x[0] = rng.multivariate_normal(x0, P0)
    y[0] = C @ x[0] + rng.multivariate_normal(np.zeros(Dy), R)
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(Dx), Q)
        y[t] = C @ x[t] + rng.multivariate_normal(np.zeros(Dy), R)

    cmp = cross_validate_kalman(y, A, C, Q, R, x0, P0)
    cmp.assert_filtered_agree(atol=1e-2)
    # Smoother gap is intentionally loose (AUDIT D3) — verify the
    # comparison produced a finite scalar so the harness itself works.
    assert cmp.smoothed_inf_norm is not None
    assert np.isfinite(cmp.smoothed_inf_norm)


def test_nemos_glm_agrees_with_nstat_within_tolerance() -> None:
    pytest.importorskip("nemos")
    from nstat.extras.validation.nemos_bridge import cross_validate_poisson_glm

    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 3))
    beta_true = np.array([0.2, -0.4, 0.1])
    rates = np.exp(0.5 + X @ beta_true)
    y = rng.poisson(rates)

    cmp = cross_validate_poisson_glm(X, y)
    # Loose tolerance — different optimizers, different stopping criteria.
    cmp.assert_agree(atol=5e-2, rtol=5e-2)


# ----------------------------------------------------------------------
# Regression tests for review-#1 bugs (don't reappear)
# ----------------------------------------------------------------------


def test_to_neo_segment_iterates_collection_without_crashing() -> None:
    """Regression: to_neo_segment used to call ``getNST()`` with no arg.

    The fix iterates the collection directly.  This test guards against
    that bug recurring.
    """
    pytest.importorskip("neo")
    from nstat import SpikeTrainCollection, nspikeTrain
    from nstat.extras.interop.neo import to_neo_segment

    coll = SpikeTrainCollection(
        [
            nspikeTrain([0.1, 0.5], minTime=0, maxTime=1),
            nspikeTrain([0.2, 0.6], minTime=0, maxTime=1),
        ]
    )
    segment = to_neo_segment(coll)
    assert len(segment.spiketrains) == 2


def test_to_pynapple_tsgroup_iterates_collection_without_crashing() -> None:
    """Regression: to_pynapple_tsgroup used to call ``getNST()`` with no arg."""
    pytest.importorskip("pynapple")
    from nstat import SpikeTrainCollection, nspikeTrain
    from nstat.extras.interop.pynapple import to_pynapple_tsgroup

    coll = SpikeTrainCollection(
        [
            nspikeTrain([0.1, 0.5], minTime=0, maxTime=1),
            nspikeTrain([0.2, 0.6], minTime=0, maxTime=1),
        ]
    )
    tsgroup = to_pynapple_tsgroup(coll)
    assert len(tsgroup) == 2


def test_from_pynapple_ts_rejects_empty_without_support() -> None:
    """Regression: empty Ts used to silently produce a 0.0/0.0 window.

    The fix raises ValueError; this test guards against the silent-
    corruption path coming back.
    """
    pytest.importorskip("pynapple")
    import pynapple as nap
    from nstat.extras.interop.pynapple import from_pynapple_ts

    empty_ts = nap.Ts(t=np.array([], dtype=float), time_units="s")
    with pytest.raises(ValueError, match="support"):
        from_pynapple_ts(empty_ts)


def test_nwb_units_warns_when_falling_back_to_spike_bounds() -> None:
    """Regression: per-unit window used to silently use spike-time
    bounds without warning the user.

    The fix emits UserWarning once per call when neither obs_intervals
    nor explicit time_window is available.  This test verifies the
    warning fires.
    """
    pynwb = pytest.importorskip("pynwb")
    import warnings
    from datetime import datetime
    from dateutil.tz import tzlocal
    from nstat.extras.interop.nwb import nwb_units_to_collection

    nwbfile = pynwb.NWBFile(
        session_description="test",
        identifier="x",
        session_start_time=datetime.now(tz=tzlocal()),
    )
    nwbfile.add_unit(spike_times=[0.1, 0.5, 0.9])
    nwbfile.add_unit(spike_times=[0.2, 0.6])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coll = nwb_units_to_collection(nwbfile)
        assert any(
            "spike-time bounds" in str(w.message) for w in caught
        ), f"Expected UserWarning about spike-time fallback; got: {caught}"
    assert len(list(coll)) == 2


def test_nwb_units_explicit_time_window_skips_warning() -> None:
    """Companion: passing time_window= silences the fallback warning."""
    pynwb = pytest.importorskip("pynwb")
    import warnings
    from datetime import datetime
    from dateutil.tz import tzlocal
    from nstat.extras.interop.nwb import nwb_units_to_collection

    nwbfile = pynwb.NWBFile(
        session_description="test",
        identifier="x",
        session_start_time=datetime.now(tz=tzlocal()),
    )
    nwbfile.add_unit(spike_times=[0.1, 0.5])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coll = nwb_units_to_collection(nwbfile, time_window=(0.0, 2.0))
        for w in caught:
            assert "spike-time bounds" not in str(w.message)

    nst = list(coll)[0]
    assert nst.minTime == pytest.approx(0.0)
    assert nst.maxTime == pytest.approx(2.0)
