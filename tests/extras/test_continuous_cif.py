"""Tests for nstat.extras.continuous_cif — continuous-time CIF simulator.

Ports the MATLAB Simulink model ``PointProcessSimulationCont.slx``.
Validation strategy (per the plan's "Verified model semantics"):

- With ``H = 0``, ``eta`` / ``lambda_delta`` are deterministic in the
  inputs and are checked bit-close against the MATLAB gold fixture
  ``tests/parity/fixtures/matlab_gold/cif_cont_lambda.mat`` (Task 1).
- With ``H != 0``, ``lambda_delta`` depends on the realised (stochastic)
  spike sequence, so it is validated Python-side with injected
  ``uniform_values`` for exact reproducibility rather than against
  MATLAB (whose DSP Random Source RNG is not reproducible in NumPy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.signal import lsim, lti

from nstat.core import Covariate
from nstat.extras.continuous_cif import simulate_cif_continuous
from nstat.simulators import PointProcessSimulation
from nstat.spikes import SpikeTrain

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "parity" / "fixtures" / "matlab_gold"


def _load_gold():
    return loadmat(FIXTURE_ROOT / "cif_cont_lambda.mat", squeeze_me=True, struct_as_record=False)


def _gold_arrays():
    """Common arrays pulled from the gold fixture, already float-cast."""
    payload = _load_gold()
    t = np.asarray(payload["t"], dtype=float).reshape(-1)
    u = np.asarray(payload["u"], dtype=float).reshape(-1)
    e = np.asarray(payload["e"], dtype=float).reshape(-1)
    mu = float(payload["mu"])
    Ts = float(payload["Ts"])
    Snum = np.atleast_1d(np.asarray(payload["Snum"], dtype=float))
    Sden = np.atleast_1d(np.asarray(payload["Sden"], dtype=float))
    eta_gold = np.asarray(payload["eta"], dtype=float).reshape(-1)
    lambda_poisson = np.asarray(payload["lambda_poisson"], dtype=float).reshape(-1)
    lambda_binom = np.asarray(payload["lambda_binom"], dtype=float).reshape(-1)
    return {
        "t": t, "u": u, "e": e, "mu": mu, "Ts": Ts,
        "Snum": Snum, "Sden": Sden,
        "eta": eta_gold, "lambda_poisson": lambda_poisson, "lambda_binom": lambda_binom,
    }


# ----------------------------------------------------------------------
# (a) continuous S filter matches scipy.signal.lsim and gold eta
# ----------------------------------------------------------------------


def test_lsim_of_S_matches_gold_eta_directly():
    g = _gold_arrays()
    _, y, _ = lsim((g["Snum"], g["Sden"]), g["u"], g["t"])
    eta_direct = g["mu"] + y
    np.testing.assert_allclose(eta_direct, g["eta"], rtol=1e-6, atol=1e-9)


def test_simulate_cif_continuous_eta_matches_gold_via_return_details():
    g = _gold_arrays()
    result = simulate_cif_continuous(
        g["mu"], (g["Snum"], g["Sden"]), 0.0, 0.0,
        (g["t"], g["u"]), (g["t"], g["e"]),
        Ts=g["Ts"], simType="poisson", seed=0, return_details=True,
    )
    np.testing.assert_allclose(result.eta, g["eta"], rtol=1e-6, atol=1e-9)


# ----------------------------------------------------------------------
# (b) deterministic lambda vs gold, both links (H = 0 => spike-independent)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "sim_type, gold_key",
    [("poisson", "lambda_poisson"), ("binomial", "lambda_binom")],
)
def test_deterministic_rate_matches_gold_for_both_links(sim_type, gold_key):
    g = _gold_arrays()
    # H = 0 => lambda_delta does not depend on the realised spikes, so the
    # seed / uniform draws are irrelevant to this comparison.
    result = simulate_cif_continuous(
        g["mu"], (g["Snum"], g["Sden"]), 0.0, 0.0,
        (g["t"], g["u"]), (g["t"], g["e"]),
        Ts=g["Ts"], simType=sim_type, seed=123,
    )
    # Model logs lambda on t[1:] (drops t=0); gold[k] <-> eta[k+1].
    np.testing.assert_allclose(result.rate_hz[1:], g[gold_key], rtol=1e-6, atol=1e-6)


def test_deterministic_rate_is_seed_independent_when_history_is_zero():
    g = _gold_arrays()
    r1 = simulate_cif_continuous(
        g["mu"], (g["Snum"], g["Sden"]), 0.0, 0.0,
        (g["t"], g["u"]), (g["t"], g["e"]),
        Ts=g["Ts"], simType="poisson", seed=1,
    )
    r2 = simulate_cif_continuous(
        g["mu"], (g["Snum"], g["Sden"]), 0.0, 0.0,
        (g["t"], g["u"]), (g["t"], g["e"]),
        Ts=g["Ts"], simType="poisson", seed=999,
    )
    np.testing.assert_array_equal(r1.rate_hz, r2.rate_hz)
    np.testing.assert_array_equal(r1.lambda_delta, r2.lambda_delta)


# ----------------------------------------------------------------------
# (c) injected-uniform history recursion: exact reproducibility + H
#     demonstrably changes eta vs H = 0
# ----------------------------------------------------------------------


def _stim_grid(n=500, Ts=0.001, freq_hz=2.0):
    t = np.arange(n) * Ts
    u = np.sin(2.0 * np.pi * freq_hz * t)
    e = np.zeros_like(t)
    return t, u, e


def test_history_feedback_changes_eta_versus_zero_history():
    t, u, e = _stim_grid()
    rng = np.random.default_rng(0)
    draws = rng.random(t.shape[0])
    stim = (np.array([1.0]), np.array([1.0]))

    zero_hist = simulate_cif_continuous(
        -2.0, stim, 0.0, 0.0, (t, u), (t, e),
        Ts=0.001, simType="binomial", uniform_values=draws, return_details=True,
    )
    nonzero_hist = simulate_cif_continuous(
        -2.0, stim, 0.0, (np.array([3.0]), np.array([0.01, 1.0])), (t, u), (t, e),
        Ts=0.001, simType="binomial", uniform_values=draws, return_details=True,
    )

    assert not np.allclose(zero_hist.eta, nonzero_hist.eta)
    assert not np.array_equal(zero_hist.spike_indicator, nonzero_hist.spike_indicator)
    # H = 0 leaves eta fully explained by mu + stim_drive.
    np.testing.assert_allclose(zero_hist.history_effect, 0.0)
    assert np.abs(nonzero_hist.history_effect).max() > 0.0


def test_injected_uniform_values_reproduce_exact_spike_sequence():
    t, u, e = _stim_grid()
    draws = np.random.default_rng(7).random(t.shape[0])
    stim = (np.array([1.0]), np.array([1.0]))
    hist = (np.array([3.0]), np.array([0.01, 1.0]))

    r1 = simulate_cif_continuous(
        -2.0, stim, 0.0, hist, (t, u), (t, e),
        Ts=0.001, simType="binomial", uniform_values=draws,
    )
    r2 = simulate_cif_continuous(
        -2.0, stim, 0.0, hist, (t, u), (t, e),
        Ts=0.001, simType="binomial", uniform_values=draws,
    )

    np.testing.assert_array_equal(r1.spike_indicator, r2.spike_indicator)
    np.testing.assert_array_equal(r1.spikes.spikeTimes, r2.spikes.spikeTimes)
    np.testing.assert_array_equal(r1.uniform_values, draws)
    # A genuinely stochastic-looking run should have some but not all spikes.
    assert 0 < r1.spike_indicator.sum() < r1.spike_indicator.size


# ----------------------------------------------------------------------
# (d) API / shape / simType / error-path checks
# ----------------------------------------------------------------------


def test_returns_point_process_simulation_with_expected_shapes():
    t, u, e = _stim_grid(n=200)
    result = simulate_cif_continuous(
        -1.0, (np.array([1.0]), np.array([0.05, 1.0])), 0.0, 0.0,
        (t, u), (t, e), Ts=0.001, simType="binomial", seed=3,
    )
    assert isinstance(result, PointProcessSimulation)
    assert isinstance(result.spikes, SpikeTrain)
    for arr in (result.time, result.rate_hz, result.lambda_delta, result.spike_indicator, result.uniform_values):
        assert arr.shape == (200,)
    np.testing.assert_array_equal(result.time, t)
    assert np.all((result.lambda_delta >= 0.0) & (result.lambda_delta <= 1.0))
    assert set(np.unique(result.spike_indicator)).issubset({0.0, 1.0})


def test_return_details_attaches_diagnostics_only_when_requested():
    t, u, e = _stim_grid(n=50)
    plain = simulate_cif_continuous(
        -1.0, (np.array([1.0]), np.array([0.05, 1.0])), 0.0, 0.0,
        (t, u), (t, e), Ts=0.001, seed=1,
    )
    detailed = simulate_cif_continuous(
        -1.0, (np.array([1.0]), np.array([0.05, 1.0])), 0.0, 0.0,
        (t, u), (t, e), Ts=0.001, seed=1, return_details=True,
    )
    for attr in ("eta", "stim_drive", "ens_drive", "history_effect"):
        assert not hasattr(plain, attr)
        assert hasattr(detailed, attr)
        assert getattr(detailed, attr).shape == (50,)


def test_bad_sim_type_raises():
    t, u, e = _stim_grid(n=20)
    with pytest.raises(ValueError):
        simulate_cif_continuous(
            -1.0, 0.0, 0.0, 0.0, (t, u), (t, e), Ts=0.001, simType="bogus",
        )


def test_non_positive_Ts_raises():
    t, u, e = _stim_grid(n=20)
    with pytest.raises(ValueError):
        simulate_cif_continuous(-1.0, 0.0, 0.0, 0.0, (t, u), (t, e), Ts=0.0)


def test_uniform_values_length_mismatch_raises():
    t, u, e = _stim_grid(n=20)
    with pytest.raises(ValueError):
        simulate_cif_continuous(
            -1.0, 0.0, 0.0, 0.0, (t, u), (t, e), Ts=0.001,
            uniform_values=np.zeros(5),
        )


def test_mismatched_stim_ens_grids_raises():
    t, u, _ = _stim_grid(n=20)
    t_other = t + 100.0  # same length, different grid
    e = np.zeros_like(t)
    with pytest.raises(ValueError):
        simulate_cif_continuous(-1.0, 0.0, 0.0, 0.0, (t, u), (t_other, e), Ts=0.001)


def test_invalid_input_stim_type_raises():
    t, u, e = _stim_grid(n=20)
    with pytest.raises(TypeError):
        simulate_cif_continuous(-1.0, 0.0, 0.0, 0.0, [1, 2, 3], (t, e), Ts=0.001)


def test_accepts_scipy_lti_system_equivalently_to_tuple():
    t, u, e = _stim_grid(n=300)
    num, den = np.array([1.0]), np.array([0.05, 1.0])
    from_tuple = simulate_cif_continuous(
        -1.0, (num, den), 0.0, 0.0, (t, u), (t, e), Ts=0.001, simType="poisson", seed=5,
    )
    from_lti = simulate_cif_continuous(
        -1.0, lti(num, den), 0.0, 0.0, (t, u), (t, e), Ts=0.001, simType="poisson", seed=5,
    )
    np.testing.assert_allclose(from_tuple.lambda_delta, from_lti.lambda_delta)


def test_bare_array_stimulus_is_treated_as_fir_numerator_static_gain():
    t, u, e = _stim_grid(n=100)
    gain = 2.0
    k = 10
    assert u[k] != 0.0  # otherwise the gain has nothing to act on
    result = simulate_cif_continuous(
        -1.0, np.array([gain]), 0.0, 0.0, (t, u), (t, e),
        Ts=0.001, simType="poisson", seed=1,
    )
    # A static-gain system (den=[1]) is a pure pointwise multiply: no
    # filter dynamics, no history/ensemble contribution here.
    expected_eta_k = -1.0 + gain * u[k]
    assert result.lambda_delta[k] == pytest.approx(np.exp(expected_eta_k))


def test_accepts_covariate_inputs_matching_tuple_inputs():
    t, u, e = _stim_grid(n=200)
    stim_cov = Covariate(t, u, "Stimulus", "time", "s", "V", ["u"])
    ens_cov = Covariate(t, e, "Ensemble", "time", "s", "spikes", ["e"])
    stim = (np.array([1.0]), np.array([0.05, 1.0]))

    from_tuple = simulate_cif_continuous(
        -1.0, stim, 0.0, 0.0, (t, u), (t, e), Ts=0.001, simType="poisson", seed=2,
    )
    from_covariate = simulate_cif_continuous(
        -1.0, stim, 0.0, 0.0, stim_cov, ens_cov, Ts=0.001, simType="poisson", seed=2,
    )
    np.testing.assert_allclose(from_tuple.lambda_delta, from_covariate.lambda_delta)


# ----------------------------------------------------------------------
# (e) mean-rate statistical sanity check with a fixed seed
# ----------------------------------------------------------------------


def test_mean_rate_matches_theoretical_bernoulli_rate_with_fixed_seed():
    Ts = 0.001
    n = 20_000
    t = np.arange(n) * Ts
    zeros = np.zeros_like(t)
    mu = -2.0

    result = simulate_cif_continuous(
        mu, 0.0, 0.0, 0.0, (t, zeros), (t, zeros),
        Ts=Ts, simType="binomial", seed=42,
    )

    p_theory = 1.0 / (1.0 + np.exp(-mu))
    empirical_rate = float(result.spike_indicator.mean())
    sigma = np.sqrt(p_theory * (1.0 - p_theory) / n)
    assert abs(empirical_rate - p_theory) < 5.0 * sigma
