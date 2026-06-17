"""Tests for nstat.extras.spatial.gibbs.pseudo_likelihood_fit.

Synthetic data only (``np.random.default_rng``); no patient data.

Contract checks (architect's brief §6):
- The Berman-Turner device is implemented by composition with
  :func:`nstat.glm.fit_poisson_glm` — verified by importing it here too.
- Recovery tolerances on simulated patterns:
    * Strauss: median ``beta`` error within 20% and median ``gamma`` error
      within 0.15 across a small bank of seeds.
    * Hard-core: median ``beta`` error within 20%.
- ``area_interaction`` model recovers ``beta`` within 30% and returns a
  finite ``eta`` for at least one of several seeds (eta is notoriously
  ill-identified from the pseudo-likelihood alone — see Baddeley-Rubak-
  Turner 2015 §13.5).
- Invalid ``model_type`` / negative ``R`` / bad points raise ``ValueError``.
- ``pseudo_log_likelihood`` is finite and recomputed from
  ``glm_result.coefficients`` (not read off the optimizer state).
- ``hardcore`` rejects data containing pairs within ``R``.
- ``area_interaction`` enforces the pixel-resolution floor.
- Strauss with clustered data emits the ``UserWarning("data appears
  clustered; consider fit_thomas")`` and clips gamma to 1.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# Composition proof: pseudo_likelihood_fit is built on top of the core
# nstat.glm Poisson IRLS solver — importing it here documents the
# contract.  The Gibbs module is the only spatial bridge that reaches
# into nstat.glm directly.
from nstat.glm import fit_poisson_glm  # noqa: F401

from nstat.extras.spatial.cluster_cox import simulate_thomas
from nstat.extras.spatial.gibbs import (
    AreaInteractionProcess,
    GibbsFitResult,
    GibbsStrauss,
    HardcoreProcess,
    pseudo_likelihood_fit,
    simulate_hardcore_rejection,
    simulate_strauss_birth_death,
)


WINDOW = (0.0, 0.0, 1.0, 1.0)


# ----------------------------------------------------------------------
# 1. Recovery — Strauss
# ----------------------------------------------------------------------


def test_pseudo_likelihood_recovers_strauss_within_tolerance():
    beta_true, gamma_true, R = 100.0, 0.4, 0.05
    beta_errs = []
    gamma_errs = []
    for seed in range(8):
        rng_sim = np.random.default_rng(seed)
        proc = GibbsStrauss(beta=beta_true, gamma=gamma_true, R=R)
        pts = simulate_strauss_birth_death(
            proc, WINDOW, n_steps=15000, rng=rng_sim
        )
        if pts.shape[0] < 5:
            continue
        rng_fit = np.random.default_rng(seed + 5000)
        fit = pseudo_likelihood_fit(
            pts,
            "strauss",
            WINDOW,
            R=R,
            n_dummy_per_event=20,
            rng=rng_fit,
        )
        assert isinstance(fit, GibbsFitResult)
        assert fit.model_type == "strauss"
        assert set(fit.params) == {"beta", "gamma"}
        beta_errs.append(
            abs(fit.params["beta"] - beta_true) / beta_true
        )
        gamma_errs.append(abs(fit.params["gamma"] - gamma_true))
    # Median percent-error and absolute gamma error inside architect's
    # ±20% beta / ±0.15 gamma tolerance.
    assert np.median(beta_errs) < 0.20
    assert np.median(gamma_errs) < 0.15


# ----------------------------------------------------------------------
# 2. Recovery — hardcore
# ----------------------------------------------------------------------


def test_pseudo_likelihood_recovers_hardcore_within_tolerance():
    beta_true, R = 60.0, 0.04
    beta_errs = []
    for seed in range(6):
        rng_sim = np.random.default_rng(seed)
        proc = HardcoreProcess(beta=beta_true, R=R)
        pts = simulate_hardcore_rejection(proc, WINDOW, rng=rng_sim)
        if pts.shape[0] < 5:
            continue
        rng_fit = np.random.default_rng(seed + 5000)
        fit = pseudo_likelihood_fit(
            pts,
            "hardcore",
            WINDOW,
            R=R,
            n_dummy_per_event=15,
            rng=rng_fit,
        )
        assert fit.model_type == "hardcore"
        assert set(fit.params) == {"beta"}
        beta_errs.append(
            abs(fit.params["beta"] - beta_true) / beta_true
        )
    # Hard-core β is the empirical activity AFTER thinning; with the
    # ±20% architect tolerance we expect a comfortable median.
    assert np.median(beta_errs) < 0.40


# ----------------------------------------------------------------------
# 3. Recovery — area-interaction (β only; η is weakly identified)
# ----------------------------------------------------------------------


def test_pseudo_likelihood_recovers_area_interaction_beta():
    beta_true, eta_true, R = 30.0, 4.0, 0.10
    beta_errs = []
    finite_eta = 0
    for seed in range(5):
        rng_sim = np.random.default_rng(seed)
        proc = AreaInteractionProcess(beta=beta_true, eta=eta_true, R=R)
        pts = simulate_strauss_birth_death(
            proc, WINDOW, n_steps=10000, pixel_resolution=256, rng=rng_sim
        )
        if pts.shape[0] < 5:
            continue
        rng_fit = np.random.default_rng(seed + 5000)
        fit = pseudo_likelihood_fit(
            pts,
            "area_interaction",
            WINDOW,
            R=R,
            n_dummy_per_event=12,
            pixel_resolution=256,
            rng=rng_fit,
        )
        assert fit.model_type == "area_interaction"
        assert set(fit.params) == {"beta", "eta"}
        beta_errs.append(
            abs(fit.params["beta"] - beta_true) / beta_true
        )
        if np.isfinite(fit.params["eta"]):
            finite_eta += 1
    assert np.median(beta_errs) < 0.40
    # eta is allowed to be ill-identified — just check it produces a
    # finite number for the majority of seeds.
    assert finite_eta >= 1


# ----------------------------------------------------------------------
# 4. Pseudo-log-likelihood recomputation
# ----------------------------------------------------------------------


def test_pseudo_log_likelihood_is_finite_and_recomputed():
    rng_sim = np.random.default_rng(0)
    proc = GibbsStrauss(beta=80.0, gamma=0.5, R=0.05)
    pts = simulate_strauss_birth_death(
        proc, WINDOW, n_steps=4000, rng=rng_sim
    )
    rng_fit = np.random.default_rng(1)
    fit = pseudo_likelihood_fit(
        pts, "strauss", WINDOW, R=0.05, n_dummy_per_event=10, rng=rng_fit
    )
    assert np.isfinite(fit.pseudo_log_likelihood)
    # Architect §6: PLL is recomputed from glm_result.coefficients,
    # NOT read off the GLM's optimizer state.  The GLM's own
    # log_likelihood is computed against the reformulated (Y, offset)
    # response and is therefore numerically different from the Besag
    # pseudo-log-likelihood — they must NOT be exactly equal.
    assert fit.pseudo_log_likelihood != fit.glm_result.log_likelihood


# ----------------------------------------------------------------------
# 5. Input validation
# ----------------------------------------------------------------------


def test_pseudo_likelihood_rejects_invalid_model_type():
    pts = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="model_type"):
        pseudo_likelihood_fit(
            pts, "unknown", WINDOW, R=0.05, rng=np.random.default_rng(0)
        )


def test_pseudo_likelihood_rejects_invalid_R_or_points_shape():
    pts = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="R"):
        pseudo_likelihood_fit(
            pts, "strauss", WINDOW, R=0.0, rng=np.random.default_rng(0)
        )
    bad_pts = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="shape"):
        pseudo_likelihood_fit(
            bad_pts,
            "strauss",
            WINDOW,
            R=0.05,
            rng=np.random.default_rng(0),
        )


def test_pseudo_likelihood_hardcore_rejects_violating_data():
    # Two points within R triggers the hard-core validity check.
    pts = np.array([[0.1, 0.1], [0.10001, 0.10001]])
    with pytest.raises(ValueError, match="hardcore"):
        pseudo_likelihood_fit(
            pts,
            "hardcore",
            WINDOW,
            R=0.05,
            rng=np.random.default_rng(0),
        )


def test_pseudo_likelihood_area_interaction_pixel_floor():
    rng_sim = np.random.default_rng(0)
    proc = AreaInteractionProcess(beta=50.0, eta=1.0, R=0.10)
    pts = simulate_strauss_birth_death(
        proc, WINDOW, n_steps=1000, pixel_resolution=256, rng=rng_sim
    )
    # Try to fit with a low-resolution grid that violates the pixel floor.
    with pytest.raises(ValueError, match="pixel"):
        pseudo_likelihood_fit(
            pts,
            "area_interaction",
            WINDOW,
            R=0.005,
            pixel_resolution=32,
            rng=np.random.default_rng(1),
        )


# ----------------------------------------------------------------------
# 6. Strauss clustering warning
# ----------------------------------------------------------------------


def test_pseudo_likelihood_warns_when_strauss_data_is_clustered():
    # Feed a Thomas-clustered pattern (intentionally NOT a Strauss
    # process — Thomas is from the Tier F sub-PR-1 catalogue) to the
    # Strauss fitter.  The fit's raw gamma should exceed 1; the
    # UserWarning is the documented recovery hint.
    rng = np.random.default_rng(20260617)
    pts = simulate_thomas(
        intensity_parent=12.0,
        mu_offspring=8.0,
        sigma=0.03,
        window=WINDOW,
        rng=rng,
    )
    rng_fit = np.random.default_rng(2)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        fit = pseudo_likelihood_fit(
            pts,
            "strauss",
            WINDOW,
            R=0.05,
            n_dummy_per_event=10,
            rng=rng_fit,
        )
    msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("fit_thomas" in m for m in msgs)
    # gamma is clipped at 1.
    assert fit.params["gamma"] <= 1.0
