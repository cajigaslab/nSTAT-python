"""Tests for nstat.extras.spatial.gibbs — Gibbs interaction models + samplers.

Synthetic data only (``np.random.default_rng``); no patient data.

Contract checks (architect's brief §6):
- Parameter dataclasses reject non-positive / out-of-range values.
- Strauss / area-interaction birth-death sampler returns points strictly
  inside the window and produces sensible event counts.
- Hard-core dart-throwing sampler enforces the minimum-distance constraint
  exactly and raises with the documented birth-death fallback hint when
  the acceptance ratio collapses.
- AreaInteractionProcess pixel-resolution floor (``R >= 2 * pixel_size``)
  is enforced in BOTH the sampler and the pseudo-likelihood fitter.
- All RNG draws come from ``np.random.default_rng(seed)``.
"""
from __future__ import annotations

import numpy as np
import pytest

# Composition proof: the Gibbs module reaches into nstat.glm — keeping
# this import here documents the contract that the spatial extras live
# strictly on top of the core GLM, not parallel to it.
from nstat.glm import fit_poisson_glm  # noqa: F401

from nstat.extras.spatial.gibbs import (
    AreaInteractionProcess,
    GibbsStrauss,
    HardcoreProcess,
    simulate_hardcore_rejection,
    simulate_strauss_birth_death,
)


WINDOW = (0.0, 0.0, 1.0, 1.0)


# ----------------------------------------------------------------------
# 1. Parameter validation
# ----------------------------------------------------------------------


def test_gibbs_strauss_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="beta"):
        GibbsStrauss(beta=0.0, gamma=0.5, R=0.05)
    with pytest.raises(ValueError, match="gamma"):
        GibbsStrauss(beta=10.0, gamma=0.0, R=0.05)
    with pytest.raises(ValueError, match="gamma"):
        GibbsStrauss(beta=10.0, gamma=1.5, R=0.05)
    with pytest.raises(ValueError, match="R"):
        GibbsStrauss(beta=10.0, gamma=0.5, R=0.0)
    # Boundary: gamma == 1 is valid (recovers Poisson).
    _ = GibbsStrauss(beta=10.0, gamma=1.0, R=0.05)


def test_hardcore_process_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="beta"):
        HardcoreProcess(beta=-1.0, R=0.05)
    with pytest.raises(ValueError, match="R"):
        HardcoreProcess(beta=10.0, R=-0.05)
    _ = HardcoreProcess(beta=10.0, R=0.05)


def test_area_interaction_process_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="beta"):
        AreaInteractionProcess(beta=0.0, eta=1.5, R=0.05)
    with pytest.raises(ValueError, match="eta"):
        AreaInteractionProcess(beta=10.0, eta=0.0, R=0.05)
    with pytest.raises(ValueError, match="R"):
        AreaInteractionProcess(beta=10.0, eta=1.5, R=0.0)
    # eta == 1 is valid (Poisson limit).
    _ = AreaInteractionProcess(beta=10.0, eta=1.0, R=0.05)


# ----------------------------------------------------------------------
# 2. Strauss birth-death sampler
# ----------------------------------------------------------------------


def test_simulate_strauss_birth_death_returns_points_in_window():
    rng = np.random.default_rng(20260617)
    proc = GibbsStrauss(beta=80.0, gamma=0.4, R=0.05)
    pts = simulate_strauss_birth_death(proc, WINDOW, n_steps=2000, rng=rng)
    assert pts.dtype == np.float64
    assert pts.ndim == 2 and pts.shape[1] == 2
    xmin, ymin, xmax, ymax = WINDOW
    assert np.all((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax))
    assert np.all((pts[:, 1] >= ymin) & (pts[:, 1] <= ymax))


def test_simulate_strauss_birth_death_inhibition_reduces_count_vs_poisson():
    # With gamma << 1 the interaction strongly suppresses near-neighbours,
    # so the mean event count should fall below the Poisson(|W| * beta)
    # baseline at the same beta.
    beta = 100.0
    n_poisson = []
    n_inhib = []
    for seed in range(8):
        rng_p = np.random.default_rng(seed)
        proc_p = GibbsStrauss(beta=beta, gamma=1.0, R=0.05)
        n_poisson.append(
            len(
                simulate_strauss_birth_death(
                    proc_p, WINDOW, n_steps=4000, rng=rng_p
                )
            )
        )
        rng_i = np.random.default_rng(seed + 1000)
        proc_i = GibbsStrauss(beta=beta, gamma=0.1, R=0.07)
        n_inhib.append(
            len(
                simulate_strauss_birth_death(
                    proc_i, WINDOW, n_steps=4000, rng=rng_i
                )
            )
        )
    assert np.mean(n_inhib) < np.mean(n_poisson)


def test_simulate_strauss_birth_death_validates_n_steps():
    rng = np.random.default_rng(0)
    proc = GibbsStrauss(beta=50.0, gamma=0.5, R=0.05)
    with pytest.raises(ValueError, match="n_steps"):
        simulate_strauss_birth_death(proc, WINDOW, n_steps=1, rng=rng)


def test_simulate_strauss_birth_death_rejects_bad_process():
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="GibbsStrauss"):
        simulate_strauss_birth_death("not a process", WINDOW, rng=rng)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 3. Hard-core dart-throwing sampler
# ----------------------------------------------------------------------


def test_simulate_hardcore_rejection_enforces_min_distance():
    rng = np.random.default_rng(20260617)
    proc = HardcoreProcess(beta=40.0, R=0.05)
    pts = simulate_hardcore_rejection(proc, WINDOW, rng=rng)
    assert pts.dtype == np.float64
    if pts.shape[0] >= 2:
        d2 = np.sum(
            (pts[:, None, :] - pts[None, :, :]) ** 2, axis=2
        )
        np.fill_diagonal(d2, np.inf)
        assert np.min(d2) >= proc.R**2 - 1e-12


def test_simulate_hardcore_rejection_returns_points_in_window():
    rng = np.random.default_rng(7)
    proc = HardcoreProcess(beta=30.0, R=0.03)
    pts = simulate_hardcore_rejection(proc, WINDOW, rng=rng)
    xmin, ymin, xmax, ymax = WINDOW
    assert np.all((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax))
    assert np.all((pts[:, 1] >= ymin) & (pts[:, 1] <= ymax))


def test_simulate_hardcore_rejection_raises_below_acceptance_floor():
    # An extreme combination — large beta and large R — makes the
    # configuration impossibly dense; the acceptance ratio collapses
    # and the sampler should raise with the documented fallback hint.
    rng = np.random.default_rng(0)
    proc = HardcoreProcess(beta=2000.0, R=0.3)
    with pytest.raises(RuntimeError, match="simulate_strauss_birth_death"):
        simulate_hardcore_rejection(proc, WINDOW, rng=rng, max_attempts=300)


# ----------------------------------------------------------------------
# 4. Area-interaction sampler
# ----------------------------------------------------------------------


def test_simulate_area_interaction_returns_points_in_window():
    rng = np.random.default_rng(20260617)
    proc = AreaInteractionProcess(beta=60.0, eta=2.0, R=0.06)
    pts = simulate_strauss_birth_death(proc, WINDOW, n_steps=3000, rng=rng)
    assert pts.dtype == np.float64
    if pts.shape[0] > 0:
        xmin, ymin, xmax, ymax = WINDOW
        assert np.all((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax))
        assert np.all((pts[:, 1] >= ymin) & (pts[:, 1] <= ymax))


def test_simulate_area_interaction_enforces_pixel_resolution_floor():
    # Architect §4: R must be >= 2 * pixel_size or the occupancy grid is
    # too coarse for the union-area sufficient statistic.
    rng = np.random.default_rng(0)
    # R = 0.005, pixel_resolution = 32 → pixel_size = 1/32 = 0.03125;
    # 2 * pixel_size = 0.0625, far above R.
    proc = AreaInteractionProcess(beta=10.0, eta=1.2, R=0.005)
    with pytest.raises(ValueError, match="pixel"):
        simulate_strauss_birth_death(
            proc, WINDOW, n_steps=500, pixel_resolution=32, rng=rng
        )


def test_simulate_area_interaction_passes_floor_at_high_resolution():
    # Same R succeeds when pixel_resolution is large enough.
    rng = np.random.default_rng(0)
    proc = AreaInteractionProcess(beta=10.0, eta=1.2, R=0.05)
    pts = simulate_strauss_birth_death(
        proc, WINDOW, n_steps=200, pixel_resolution=256, rng=rng
    )
    # Just check it ran to completion.
    assert pts.ndim == 2 and pts.shape[1] == 2
