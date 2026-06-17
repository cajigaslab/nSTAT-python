"""Tests for ``nstat.extras.latents.gpfa_bridge``.

Validation tests (config + shape rejection) run unconditionally; the
elephant-backed functional tests skip cleanly when the optional
dependency is missing.

The bridge wraps :class:`elephant.gpfa.GPFA` (Yu et al. 2009).  Tests
exercise four contracts:

1. :class:`GPFAConfig` validates every numeric argument.
2. :func:`fit_gpfa` rejects malformed and degenerate inputs.
3. Synthetic-trial fit recovers the underlying low-dimensional
   structure (Pearson |r| > 0.4 on at least one latent vs ground
   truth) — the "EM found something" check, not a tight recovery test.
4. ``seed=`` is reproducible AND the caller's RNG state is preserved
   on exit (the isolation contract for the legacy ``np.random.seed``
   wrapper inside the bridge).
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1–3. GPFAConfig validation
# ---------------------------------------------------------------------------


def test_config_validates_x_dim() -> None:
    from nstat.extras.latents import GPFAConfig

    with pytest.raises(ValueError, match="x_dim"):
        GPFAConfig(x_dim=0)
    with pytest.raises(ValueError, match="x_dim"):
        GPFAConfig(x_dim=-1)


def test_config_validates_bin_size() -> None:
    from nstat.extras.latents import GPFAConfig

    with pytest.raises(ValueError, match="bin_size_s"):
        GPFAConfig(x_dim=2, bin_size_s=0.0)
    with pytest.raises(ValueError, match="bin_size_s"):
        GPFAConfig(x_dim=2, bin_size_s=-0.01)


def test_config_validates_em_settings() -> None:
    from nstat.extras.latents import GPFAConfig

    with pytest.raises(ValueError, match="em_max_iter"):
        GPFAConfig(x_dim=2, em_max_iter=0)
    with pytest.raises(ValueError, match="em_tol"):
        GPFAConfig(x_dim=2, em_tol=0.0)
    with pytest.raises(ValueError, match="min_var_frac"):
        GPFAConfig(x_dim=2, min_var_frac=0.0)
    with pytest.raises(ValueError, match="min_var_frac"):
        GPFAConfig(x_dim=2, min_var_frac=1.0)


# ---------------------------------------------------------------------------
# 4–5. Input-shape rejection (no elephant needed; raises before lazy gate)
# ---------------------------------------------------------------------------


def test_fit_gpfa_rejects_single_trial() -> None:
    """A list with a single Trial-like entry trips the >=2 guard before
    the lazy elephant import.  We use a duck-typed stub to avoid
    constructing a full nstat ``Trial`` (which would need spikes +
    covariates) in a pure-validation test.
    """
    from nstat.extras.latents import fit_gpfa

    class _StubTrial:
        nspikeColl = object()

    with pytest.raises(ValueError, match="GPFA requires >= 2 trials"):
        fit_gpfa([_StubTrial()])


def test_fit_gpfa_rejects_unknown_input_shape() -> None:
    from nstat.extras.latents import fit_gpfa

    with pytest.raises(TypeError, match="list\\[Trial\\]"):
        fit_gpfa(np.zeros((3, 4)))
    with pytest.raises(TypeError, match="list\\[Trial\\]"):
        fit_gpfa({"trials": []})


# ---------------------------------------------------------------------------
# 6–8. Functional tests — gated on elephant availability.
# ---------------------------------------------------------------------------


def _make_synthetic_trials(
    *, n_trials: int, n_neurons: int, duration_s: float, seed: int
):
    """Synthetic multi-trial Poisson spike trains driven by two
    sinusoidal latents at different frequencies.

    Returns ``(neo_trials, true_latents)`` where ``true_latents`` is a
    length-``n_trials`` list of ``(n_fine_steps, 2)`` ndarrays sampled on
    a 1 ms grid (used for the recovery check).
    """
    import neo
    import quantities as pq

    rng = np.random.default_rng(seed)
    dt_fine = 0.001  # 1 ms ground-truth grid
    n_steps = int(round(duration_s / dt_fine))
    t = np.arange(n_steps) * dt_fine

    # Two latents shared across trials but with per-trial phase jitter so
    # each trial provides distinct evidence.
    base_freqs = np.array([1.5, 3.0])  # Hz
    # Random per-neuron loading onto each latent.
    loadings = rng.standard_normal((n_neurons, 2)) * 1.0
    # Baseline log-rate per neuron (sets mean firing ~20 Hz).
    baseline = np.log(20.0) * np.ones(n_neurons)

    neo_trials: list[list[neo.SpikeTrain]] = []
    true_latents: list[np.ndarray] = []
    for trial_idx in range(n_trials):
        phase = rng.uniform(-0.3, 0.3, size=2)
        z = np.stack(
            [
                np.sin(2 * np.pi * base_freqs[0] * t + phase[0]),
                np.sin(2 * np.pi * base_freqs[1] * t + phase[1]),
            ],
            axis=1,
        )  # (n_steps, 2)
        true_latents.append(z)
        # Log firing rate for each neuron at each fine step.
        log_rate = z @ loadings.T + baseline  # (n_steps, n_neurons)
        rate = np.exp(log_rate)
        # Poisson thinning: expected spikes per fine bin = rate * dt_fine.
        lam = rate * dt_fine
        spikes = rng.poisson(lam)  # (n_steps, n_neurons)
        trial_sts: list[neo.SpikeTrain] = []
        for n in range(n_neurons):
            # Emit a spike at each fine bin where lam_n[t] sample > 0.
            # For lam << 1 the multiplicity is essentially 0/1; we use
            # the bin centre as the spike time.
            idx = np.nonzero(spikes[:, n])[0]
            counts = spikes[idx, n]
            # Expand multiplicities into individual times with small
            # within-bin jitter so duplicate times don't appear.
            times: list[float] = []
            for ti, c in zip(idx, counts, strict=False):
                for k in range(int(c)):
                    times.append(t[ti] + (k + 0.5) * (dt_fine / max(c, 1)))
            times_arr = np.asarray(sorted(times), dtype=float)
            trial_sts.append(
                neo.SpikeTrain(
                    times_arr * pq.s,
                    t_start=0.0 * pq.s,
                    t_stop=duration_s * pq.s,
                )
            )
        neo_trials.append(trial_sts)

    return neo_trials, true_latents


def test_fit_gpfa_on_synthetic_trials() -> None:
    pytest.importorskip("elephant")
    pytest.importorskip("neo")
    pytest.importorskip("quantities")
    from nstat.extras.latents import GPFAConfig, fit_gpfa

    n_trials = 5
    duration_s = 2.0
    bin_size_s = 0.05
    neo_trials, true_latents = _make_synthetic_trials(
        n_trials=n_trials, n_neurons=8, duration_s=duration_s, seed=20260616,
    )

    cfg = GPFAConfig(x_dim=2, bin_size_s=bin_size_s, em_max_iter=80)
    result = fit_gpfa(neo_trials, config=cfg, seed=42)

    # Shape contracts.
    assert len(result.latent_trajectories) == n_trials
    n_bins = int(round(duration_s / bin_size_s))
    for traj in result.latent_trajectories:
        assert traj.shape == (n_bins, 2)
    assert result.x_dim == 2
    assert result.bin_size_s == bin_size_s
    assert result.n_trials == n_trials
    # The elephant model is exposed.
    assert result.elephant_model is not None

    # Subsample the ground-truth latents onto the same bin grid as the
    # GPFA output for correlation scoring.
    dt_fine = 0.001
    bin_step = int(round(bin_size_s / dt_fine))
    # Pearson |r| > 0.4 between at least one recovered and one true
    # latent on at least one trial — the "EM found *some* low-dim
    # structure" smoke test, not a tight recovery claim.
    max_corr = 0.0
    for traj, z_true in zip(
        result.latent_trajectories, true_latents, strict=False
    ):
        # Bin-centred subsampling of the ground truth.
        offset = bin_step // 2
        # Guard against off-by-one when n_steps is not divisible by bin_step.
        idx = np.arange(n_bins) * bin_step + offset
        idx = np.clip(idx, 0, z_true.shape[0] - 1)
        z_binned = z_true[idx]  # (n_bins, 2)
        for i in range(2):
            for j in range(2):
                r = float(
                    np.corrcoef(traj[:, i], z_binned[:, j])[0, 1]
                )
                if np.isfinite(r):
                    max_corr = max(max_corr, abs(r))
    assert max_corr > 0.4, (
        f"GPFA recovered no latent correlated with the ground truth "
        f"(max |r|={max_corr:.3f}); EM probably did not converge."
    )


def test_fit_gpfa_seed_reproducibility() -> None:
    """Same ``seed=`` must give bit-identical latents.

    Note on the "different seed → different latents" intuition: for
    non-overlapping trials with no cross-validation pass, Elephant's
    GPFA is fully deterministic — sklearn's :class:`FactorAnalysis`
    init uses a deterministic SVD, and the EM E-/M-steps have no
    stochastic component.  The only path that consumes
    ``np.random.*`` is the overlapping-epoch multinomial split in
    ``gpfa_util.segment_by_trial``, which doesn't fire here.  So
    ``seed=42`` and ``seed=99`` may produce identical output on this
    synthetic data; we only assert the strictly stronger
    same-seed-reproducibility contract.
    """
    pytest.importorskip("elephant")
    pytest.importorskip("neo")
    pytest.importorskip("quantities")
    from nstat.extras.latents import GPFAConfig, fit_gpfa

    neo_trials, _ = _make_synthetic_trials(
        n_trials=3, n_neurons=6, duration_s=1.5, seed=11,
    )
    cfg = GPFAConfig(x_dim=2, bin_size_s=0.05, em_max_iter=30)

    a = fit_gpfa(neo_trials, config=cfg, seed=42)
    b = fit_gpfa(neo_trials, config=cfg, seed=42)

    assert np.allclose(
        a.latent_trajectories[0], b.latent_trajectories[0]
    ), "Same seed must produce bit-identical latents."
    # Sanity: unseeded fit on the same data is also deterministic for
    # this codepath.
    c = fit_gpfa(neo_trials, config=cfg, seed=None)
    assert np.allclose(
        a.latent_trajectories[0], c.latent_trajectories[0]
    ), (
        "Elephant's GPFA is deterministic on non-overlapping trials; the "
        "seeded fit should match an unseeded fit on the same data."
    )


def test_fit_gpfa_preserves_caller_rng_state() -> None:
    pytest.importorskip("elephant")
    pytest.importorskip("neo")
    pytest.importorskip("quantities")
    from nstat.extras.latents import GPFAConfig, fit_gpfa

    neo_trials, _ = _make_synthetic_trials(
        n_trials=3, n_neurons=6, duration_s=1.5, seed=22,
    )
    cfg = GPFAConfig(x_dim=2, bin_size_s=0.05, em_max_iter=10)

    # Pin the caller's legacy RNG state, fit (which sets a different
    # internal seed), and assert the caller's state is restored byte-
    # for-byte.  This is the isolation contract for the documented
    # legacy ``np.random.seed`` wrapper inside the bridge.
    np.random.seed(123456)
    before = np.random.get_state()
    fit_gpfa(neo_trials, config=cfg, seed=0)
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2] == after[2]
    assert before[3] == after[3]
    assert before[4] == after[4]
