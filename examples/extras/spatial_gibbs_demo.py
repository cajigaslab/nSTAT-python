#!/usr/bin/env python3
"""Demo: co-active/implanted-site spacing -- CSR vs. minimum-distance
exclusion (hard-core), via Gibbs interaction processes + Berman-Turner
pseudo-likelihood.

**Clinical question.**  When two contacts are implanted (SEEG depth
electrodes, subdural grid contacts, chronic microelectrode arrays) or two
sites are observed to be co-active in the same epoch, is their spacing
consistent with **independent placement** (complete spatial randomness,
CSR), or is it governed by a **minimum-distance repulsion** -- an
enforced exclusion zone below which two sites simply cannot co-occur?
Real implantation is *not* independent: computer-assisted SEEG
trajectory-planning pipelines fold a minimum inter-electrode-distance
safety constraint into their planning criteria alongside vascular-injury
avoidance (Sparks R et al. 2016, Int J Comput Assist Radiol Surg
12(1):123), high-density recording-array design targets an *optimal*
inter-electrode spacing that trades spike-sorting yield against site
count (Meszéna D et al. 2026, Microsyst Nanoeng 12(1):41), and even
naturally-arranged mosaics of a single retinal-neuron type show a real
minimum-distance "exclusion radius" around each cell rather than
independent scatter (Rockhill RL et al. 2000, PNAS 97(5):2303 --
spatial order *within* but not *between* cell types). Seizure-core
recruitment fronts likewise carve out an exclusion zone of suppressed
firing just ahead of the propagating wave (Schevon CA et al. 2012, Nat
Commun 3:1060) -- a related but distinct minimum-distance signature. A
scatter plot of contact/site locations cannot separate "no structure"
from "hard exclusion" by eye alone; this demo asks whether the
Berman-Turner pseudo-likelihood can.

**Modeling correspondence.**  CSR is a homogeneous Poisson process --
sites placed independently, with no constraint on how close two sites
can be. A hard-core Gibbs process (the Strauss ``gamma -> 0`` limit)
instead assigns zero probability to any configuration containing two
points closer than a fixed exclusion radius ``R`` -- exactly the
minimum-inter-electrode/-contact-distance story above. Fitting a Strauss
model (``model_type="strauss"``) to a CSR pattern should recover an
interaction strength ``gamma_hat ~= 1`` (no interaction); for the
hard-core pattern, the *observed minimum inter-point distance* is the
classical estimator of the exclusion radius itself (Ripley & Kelly 1977
note that any candidate ``R`` above this value is inconsistent with the
data), while :func:`~nstat.extras.spatial.pseudo_likelihood_fit` recovers
the hard-core intensity ``beta`` at that (known) radius.

**Scenario.**  Two placements of co-active/implanted sites on the unit
square at the same target intensity ``beta = 60``: (a) CSR -- independent
placement -- and (b) a hard-core process enforcing a minimum inter-site
distance ``R = 0.04`` normalized units. Read literally as a 10 cm x 10 cm
cortical/subdural patch, ``R = 0.04`` is ~4 mm -- inside the range of
minimum inter-contact spacing considered by SEEG trajectory-planning
safety margins and consistent with the Meszéna et al. (2026)
optimal-spacing analysis for high-density arrays. Strauss (mild
inhibition) and area-interaction (clustering) processes are kept as
**secondary**, purely contrastive panels -- the same Gibbs catalogue, a
different second-order signature.

1. Simulate CSR (:meth:`numpy.random.Generator.uniform`) and hard-core
   (:func:`~nstat.extras.spatial.simulate_hardcore_rejection`) patterns
   at the same target intensity ``beta = 60``, exclusion radius
   ``R = 0.04``.
2. Fit both via :func:`nstat.extras.spatial.pseudo_likelihood_fit` --
   ``model_type="strauss"`` for CSR (recovers ``gamma_hat ~= 1``, i.e.
   *no* interaction), ``model_type="hardcore"`` for the hard-core
   pattern (recovers ``beta_hat``, with the documented upward bias --
   see the ``hardcore-bias`` note in the run-table output). The
   exclusion *distance* itself is recovered directly from the data as
   the observed minimum inter-point distance, since
   ``pseudo_likelihood_fit`` takes ``R`` as a known input rather than
   estimating it.
3. Secondary, purely contrastive: a Strauss process (Strauss 1975) at
   ``gamma = 0.4`` (mild inhibition) and an area-interaction process
   (Widom-Rowlinson 1970; Baddeley-van Lieshout 1995) at ``eta = 4.0``
   (clustering), each simulated and refit exactly as in the pre-2026-07
   version of this demo.

See also :mod:`spatial_cluster_cox_demo` -- the opposite second-order
signature (*clustering* around hidden hub sites) rather than the
*repulsion* story here; comparing the two PCFs side by side is the
fastest way to see how attraction (hubs) and repulsion (exclusion zones)
each leave a distinct mark on :math:`g(r)`.

The script is **fully synthetic** -- no figshare dataset access required.

Run::

    python examples/extras/spatial_gibbs_demo.py            # interactive
    python examples/extras/spatial_gibbs_demo.py --no-display
    python examples/extras/spatial_gibbs_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/spatial_gibbs/``) and are NOT committed to the
repository via this flag -- CI never invokes it (the three committed
PNGs are regenerated and reviewed manually when the demo changes).

References:

Clinical motivation (electrode/site spacing, exclusion zones):

- Meszéna D, Fadel W, Tóth R, Paulk AC, Cash SS, Williams Z, Kiss T,
  Stippinger M, Wittner L, Fiáth R, Somogyvári Z (2026). *Optimal
  inter-electrode distances for maximizing single unit yield per
  electrode in neural recordings.* Microsyst Nanoeng 12(1):41.
- Rockhill RL, Euler T, Masland RH (2000). *Spatial order within but not
  between types of retinal neurons.* Proc Natl Acad Sci U S A
  97(5):2303-2307.
- Sparks R, Zombori G, Rodionov R, Nowell M, Vos SB, Zuluaga MA, Diehl B,
  Wehner T, Miserocchi A, McEvoy AW, Duncan JS, Ourselin S (2016).
  *Automated multiple trajectory planning algorithm for the placement of
  stereo-electroencephalography (SEEG) electrodes in epilepsy
  treatment.* Int J Comput Assist Radiol Surg 12(1):123-136. (Generic:
  computer-assisted SEEG trajectory planning folds a minimum
  inter-electrode-distance constraint into its safety criteria.)
- Schevon CA, Weiss SA, McKhann G Jr, Goodman RR, Yuste R, Emerson RG,
  Trevelyan AJ (2012). *Evidence of an inhibitory restraint of seizure
  activity in humans.* Nat Commun 3:1060.

Statistical methods:

- Strauss DJ (1975). *A model for clustering.* Biometrika 62(2):467.
- Besag J (1977). *Some methods of statistical analysis for spatial data.*
  Bull. Inst. Internat. Statist. 47:77.
- Ripley BD, Kelly FP (1977). *Markov point processes.* J. London Math.
  Soc. 15(1):188 (minimum inter-point distance as the hard-core-radius
  estimator).
- Berman M, Turner TR (1992). *Approximating point process likelihoods
  with GLIM.* Appl. Stat. 41(1):31.
- Baddeley A, Turner R (2000). *Practical maximum pseudolikelihood for
  spatial point patterns.* Aust. N. Z. J. Stat. 42(3):283.
- Widom B, Rowlinson JS (1970). *New model for the study of liquid-vapor
  phase transitions.* J. Chem. Phys. 52(4):1670.
- Baddeley AJ, van Lieshout MNM (1995). *Area-interaction point
  processes.* Ann. Inst. Statist. Math. 47(4):601.
- Geyer CJ (1999). *Likelihood inference for spatial point processes.*
- Baddeley A, Rubak E, Turner R (2015). *Spatial Point Patterns:
  Methodology and Applications with R.* CRC §13.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Window / domain conventions: the Gibbs simulators/fitter take a flat
# 4-tuple (xmin, ymin, xmax, ymax); pair_correlation takes the nested
# ((xmin, xmax), (ymin, ymax)) rectangle form.
WINDOW = (0.0, 0.0, 1.0, 1.0)
DOMAIN = ((0.0, 1.0), (0.0, 1.0))

# Physiological grounding for the hard-core exclusion distance below:
# read the unit square as a 10 cm x 10 cm (100 mm) cortical/subdural
# patch, so R = 0.04 normalized units is ~4 mm -- inside the range of
# minimum inter-contact/inter-electrode spacing considered by SEEG
# trajectory-planning safety margins (Sparks et al. 2016) and consistent
# with the optimal inter-electrode-spacing analysis for high-density
# recording arrays (Meszéna et al. 2026).  See the module docstring for
# the full citations.
PATCH_SIDE_MM = 100.0


# Q1 (resolved): ship Strauss + area-interaction at n_steps=5000.  If a
# local 60 s wall budget is exceeded the user can drop to 3000.
N_STEPS_BD = 5000


def _run_csr(rng: np.random.Generator, *, beta: float, R: float) -> dict:
    """Simulate CSR (independent site placement) at the hard-core's
    target intensity, then fit the Strauss model at the hard-core's
    exclusion radius ``R``.

    No genuine repulsion is present here, so the fitted interaction
    strength should recover ``gamma_hat ~= 1`` (no interaction) -- the
    contrast case for the minimum-distance exclusion story in
    :func:`_run_hardcore`.
    """
    from nstat.extras.spatial import pseudo_likelihood_fit

    n = max(int(rng.poisson(beta)), 1)
    points = rng.uniform(0.0, 1.0, size=(n, 2))
    fit = pseudo_likelihood_fit(
        points, model_type="strauss", window=WINDOW, R=R,
        n_dummy_per_event=20, rng=rng,
    )
    return {
        "points": points,
        "beta_true": beta,
        "gamma_true": 1.0,
        "R": R,
        "beta_hat": float(fit.params["beta"]),
        "gamma_hat": float(fit.params["gamma"]),
        "pseudo_log_likelihood": float(fit.pseudo_log_likelihood),
        "n_data": int(fit.n_data),
        "n_dummy": int(fit.n_dummy),
        "fit_converged": bool(fit.glm_result.converged),
    }


def _run_hardcore(rng: np.random.Generator) -> dict:
    """Simulate + fit a hard-core (minimum-distance exclusion) process
    at beta = 60, R = 0.04 (~4 mm on a 10 cm x 10 cm patch; see the
    module docstring's physiological grounding).

    Q3 (resolved) target was 100.  Empirically, at beta = 100 on the unit
    square with R = 0.04 the intercept-only Berman-Turner Poisson GLM
    routinely fails to converge across many seeds — the log-area offset
    drives the IRLS step into numerical overflow.  beta = 60 sits in the
    test-calibrated regime
    (``tests/extras/test_spatial_pseudo_likelihood.py`` uses the same
    value) where the GLM converges and the documented upward bias of
    ``beta_hat`` is finite and visible — i.e. the bias-direction story
    Q3 wanted is demonstrable.  Both beta = 60 and beta = 100 sit well
    below the packing-fraction failure mode of
    :func:`~nstat.extras.spatial.simulate_hardcore_rejection`.
    """
    from nstat.extras.spatial import (
        HardcoreProcess,
        pseudo_likelihood_fit,
        simulate_hardcore_rejection,
    )

    beta_true = 60.0
    R = 0.04

    process = HardcoreProcess(beta=beta_true, R=R)
    points = simulate_hardcore_rejection(process, WINDOW, rng=rng)
    fit = pseudo_likelihood_fit(
        points, model_type="hardcore", window=WINDOW, R=R,
        n_dummy_per_event=15, rng=rng,
    )

    # Recovered exclusion distance: pseudo_likelihood_fit takes R as a
    # *known* input rather than estimating it, but the classical
    # estimator of a hard-core radius is simply the observed minimum
    # inter-point distance (Ripley & Kelly 1977) -- any candidate R
    # larger than this value is inconsistent with the data (indeed,
    # pseudo_likelihood_fit's own hardcore branch raises ValueError if
    # asked to fit at such an R).
    if points.shape[0] >= 2:
        r_hat_min_distance = float(np.min(pdist(points)))
    else:
        r_hat_min_distance = float("nan")

    return {
        "points": points,
        "beta_true": beta_true,
        "R": R,
        "r_hat_min_distance": r_hat_min_distance,
        "beta_hat": float(fit.params["beta"]),
        "pseudo_log_likelihood": float(fit.pseudo_log_likelihood),
        "n_data": int(fit.n_data),
        "n_dummy": int(fit.n_dummy),
        "fit_converged": bool(fit.glm_result.converged),
    }


def _run_strauss(rng: np.random.Generator) -> dict:
    """Simulate + fit a Strauss process at gamma = 0.4 (secondary panel)."""
    from nstat.extras.spatial import (
        GibbsStrauss,
        pseudo_likelihood_fit,
        simulate_strauss_birth_death,
    )

    beta_true = 100.0
    gamma_true = 0.4
    R = 0.05

    process = GibbsStrauss(beta=beta_true, gamma=gamma_true, R=R)
    points = simulate_strauss_birth_death(
        process, WINDOW, n_steps=N_STEPS_BD, rng=rng,
    )
    fit = pseudo_likelihood_fit(
        points, model_type="strauss", window=WINDOW, R=R,
        n_dummy_per_event=20, rng=rng,
    )
    return {
        "points": points,
        "beta_true": beta_true,
        "gamma_true": gamma_true,
        "R": R,
        "beta_hat": float(fit.params["beta"]),
        "gamma_hat": float(fit.params["gamma"]),
        "pseudo_log_likelihood": float(fit.pseudo_log_likelihood),
        "n_data": int(fit.n_data),
        "n_dummy": int(fit.n_dummy),
        "fit_converged": bool(fit.glm_result.converged),
    }


def _run_area_interaction(rng: np.random.Generator) -> dict:
    """Simulate + fit an area-interaction process at eta = 4.0 (secondary panel)."""
    from nstat.extras.spatial import (
        AreaInteractionProcess,
        pseudo_likelihood_fit,
        simulate_strauss_birth_death,
    )

    # Test-calibrated parameters from
    # tests/extras/test_spatial_pseudo_likelihood.py — beta well within
    # both the simulator's stationarity envelope and the fitter's
    # numerical-stability envelope.  eta is notoriously weakly
    # identified by pseudo-likelihood alone (Baddeley-Rubak-Turner
    # 2015 §13.5) — we accept whatever finite estimate falls out.
    beta_true = 30.0
    eta_true = 4.0
    R = 0.10

    process = AreaInteractionProcess(beta=beta_true, eta=eta_true, R=R)
    points = simulate_strauss_birth_death(
        process, WINDOW, n_steps=N_STEPS_BD,
        pixel_resolution=256, rng=rng,
    )
    fit = pseudo_likelihood_fit(
        points, model_type="area_interaction", window=WINDOW, R=R,
        n_dummy_per_event=12, pixel_resolution=256, rng=rng,
    )
    return {
        "points": points,
        "beta_true": beta_true,
        "eta_true": eta_true,
        "R": R,
        "beta_hat": float(fit.params["beta"]),
        "eta_hat": float(fit.params["eta"]),
        "pseudo_log_likelihood": float(fit.pseudo_log_likelihood),
        "n_data": int(fit.n_data),
        "n_dummy": int(fit.n_dummy),
        "fit_converged": bool(fit.glm_result.converged),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_scatter_with_radius(
    ax, points: np.ndarray, R: float | None, title: str
) -> None:
    ax.scatter(
        points[:, 0], points[:, 1],
        s=14, color="tab:blue", alpha=0.8, edgecolor="k", linewidth=0.4,
    )
    # Reference circle in the lower-left corner showing the interaction
    # radius R at the figure's aspect.  Skipped when R is None (CSR has
    # no interaction radius to show).
    if R is not None:
        theta = np.linspace(0.0, 2.0 * np.pi, 60)
        ax.plot(
            0.05 + R * np.cos(theta),
            0.05 + R * np.sin(theta),
            color="tab:red", lw=1.2, alpha=0.9,
        )
        ax.text(
            0.05, 0.05 + R + 0.015,
            f"R = {R}", color="tab:red", fontsize=8, ha="center",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=9.5)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_demo(
    *,
    seed: int = 20260616,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool = True,
    plot_style: str = "legacy",
) -> dict:
    """Run the CSR-vs-hard-core (+ secondary Gibbs catalogue) demo.

    Returns
    -------
    dict
        ``{"csr": ..., "hardcore": ..., "strauss": ..., "area_interaction":
        ..., "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style
    from nstat.extras.spatial import pair_correlation

    print("=" * 72)
    print("Co-active/implanted-site spacing: CSR vs. minimum-distance")
    print("exclusion (hard-core), via Berman-Turner pseudo-likelihood")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    hc = _run_hardcore(rng)
    csr = _run_csr(rng, beta=hc["beta_true"], R=hc["R"])
    st = _run_strauss(rng)
    ai = _run_area_interaction(rng)

    # ---- PRIMARY: CSR vs. hard-core recovery table ----
    print()
    print("-" * 72)
    print("PRIMARY: CSR (independent placement) vs. hard-core (minimum-")
    print("distance exclusion) -- co-active/implanted-site spacing")
    print("-" * 72)

    print()
    print("CSR / complete spatial randomness -- independent site placement")
    print(f"  n_data           : {csr['n_data']}")
    print(f"  target beta      : {csr['beta_true']:.3f}")
    print(f"  estimated beta   : {csr['beta_hat']:.3f}")
    print(f"  target gamma     : {csr['gamma_true']:.3f}  (no interaction, by construction)")
    print(f"  estimated gamma  : {csr['gamma_hat']:.3f}")
    gamma_tol = 0.40
    gamma_err = abs(csr["gamma_hat"] - csr["gamma_true"])
    gamma_ok = gamma_err < gamma_tol
    print(
        f"  |gamma_hat - 1|  : {gamma_err:.3f}  (tol {gamma_tol})  "
        f"{'PASS' if gamma_ok else 'FAIL'}"
    )
    print(f"  GLM converged    : {csr['fit_converged']}")

    print()
    print("Hard-core (minimum-distance exclusion) -- dart-throwing rejection")
    print(f"  n_data                   : {hc['n_data']}")
    print(f"  target beta              : {hc['beta_true']:.3f}")
    print(f"  estimated beta           : {hc['beta_hat']:.3f}")
    print(
        f"  true exclusion R         : {hc['R']:.4f}  "
        f"(~{hc['R'] * PATCH_SIDE_MM:.1f} mm on a "
        f"{PATCH_SIDE_MM / 10:.0f}x{PATCH_SIDE_MM / 10:.0f} cm patch)"
    )
    print(
        f"  recovered R (min NN dist): {hc['r_hat_min_distance']:.4f}  "
        f"(~{hc['r_hat_min_distance'] * PATCH_SIDE_MM:.1f} mm)"
    )
    r_tol_rel = 0.30
    r_err_rel = abs(hc["r_hat_min_distance"] - hc["R"]) / hc["R"]
    r_ok = r_err_rel < r_tol_rel
    print(
        f"  relative R error         : {r_err_rel:.3f}  (tol {r_tol_rel})  "
        f"{'PASS' if r_ok else 'FAIL'}"
    )
    print(f"  GLM converged            : {hc['fit_converged']}")
    print()
    print("  NOTE (hardcore bias): the intercept-only Berman-Turner GLM is")
    print("  upward-biased on the hard-core intensity — median ~40% high")
    print("  at small R for the dart-throwing simulator, because the")
    print("  log-area offset over-attributes activity to the un-excluded")
    print("  quadrature area.  Baddeley-Rubak-Turner (2015) §13.4 give the")
    print("  analytical correction beta_hat / (1 - pi R^2 lambda_hat),")
    print("  which we deliberately do NOT apply here so the demo records")
    print("  the bias direction honestly.  See the test docstring in")
    print("  tests/extras/test_spatial_pseudo_likelihood.py for the full")
    print("  characterisation if you need calibrated intensity recovery.")
    print("  The exclusion *distance* itself (unlike beta) recovers")
    print("  cleanly — it is read directly off the data, not fit by GLM.")

    # ---- SECONDARY: Strauss / area-interaction ----
    print()
    print("-" * 72)
    print("SECONDARY (contrastive, not part of the CSR-vs-hard-core story)")
    print("-" * 72)

    print()
    print("Strauss process (Strauss 1975) — birth-death + pseudo-likelihood")
    print(f"  n_data           : {st['n_data']}")
    print(f"  target beta      : {st['beta_true']:.3f}")
    print(f"  estimated beta   : {st['beta_hat']:.3f}")
    print(f"  target gamma     : {st['gamma_true']:.3f}")
    print(f"  estimated gamma  : {st['gamma_hat']:.3f}")
    print(f"  GLM converged    : {st['fit_converged']}")
    print(f"  pseudo log-lik   : {st['pseudo_log_likelihood']:.3f}")

    print()
    print("Area-interaction process (Baddeley-van Lieshout 1995) — birth-death")
    print(f"  n_data           : {ai['n_data']}")
    print(f"  target beta      : {ai['beta_true']:.3f}")
    print(f"  estimated beta   : {ai['beta_hat']:.3f}")
    print(f"  target eta       : {ai['eta_true']:.3f}")
    print(f"  estimated eta    : {ai['eta_hat']:.3f}")
    print(f"  GLM converged    : {ai['fit_converged']}")
    print(f"  pseudo log-lik   : {ai['pseudo_log_likelihood']:.3f}")

    # ---- Figures ----
    # === FIGURE: fig01_strauss_scatter.png ===
    fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))
    _plot_scatter_with_radius(
        ax1, st["points"], st["R"],
        f"Strauss (secondary) — beta={st['beta_true']}, "
        f"gamma={st['gamma_true']}, R={st['R']} — n={st['n_data']}",
    )
    # === END FIGURE ===

    # === FIGURE: fig02_hardcore_scatter.png ===
    # Primary ground-truth-vs-estimate figure: CSR vs. hard-core patterns
    # side by side, their empirical-vs-fitted pair correlation, and the
    # recovered-vs-true exclusion distance / interaction strength.
    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(11.5, 10.0))

    _plot_scatter_with_radius(
        ax2a, csr["points"], None,
        f"CSR — independent placement — n={csr['n_data']}",
    )
    _plot_scatter_with_radius(
        ax2b, hc["points"], hc["R"],
        f"Hard-core exclusion (R={hc['R']}) — n={hc['n_data']}",
    )

    # Empirical vs. fitted pair correlation g(r).  bw is widened past the
    # library default rule-of-thumb: at n~50-70 points the default
    # bandwidth leaves both curves too noisy near r=0 to read the
    # hard-core dip by eye.
    r_grid = np.linspace(0.01, 4.0 * hc["R"], 40)
    bw_pcf = 0.03
    lam_hc = np.full(hc["n_data"], hc["n_data"] / 1.0)
    lam_csr = np.full(csr["n_data"], csr["n_data"] / 1.0)
    g_hc = pair_correlation(hc["points"], lam_hc, r_grid, domain=DOMAIN, bw=bw_pcf)
    g_csr = pair_correlation(csr["points"], lam_csr, r_grid, domain=DOMAIN, bw=bw_pcf)
    # "Fitted" curves at the recovered parameters: a 0/1 hard-core step
    # at the recovered exclusion distance, and the fitted Strauss
    # gamma_hat (flat gamma_hat below R, 1 above -- for gamma_hat ~= 1
    # this is indistinguishable from the CSR null).
    g_fit_hc = np.where(r_grid < hc["r_hat_min_distance"], 0.0, 1.0)
    g_fit_csr = np.where(r_grid < csr["R"], csr["gamma_hat"], 1.0)

    ax2c.plot(r_grid, g_csr, color="tab:blue", lw=1.6, label="CSR empirical g(r)")
    ax2c.plot(r_grid, g_hc, color="tab:red", lw=1.6, label="hard-core empirical g(r)")
    ax2c.plot(
        r_grid, g_fit_csr, color="tab:blue", lw=1.2, ls="--", alpha=0.85,
        label="CSR fitted (gamma_hat)",
    )
    ax2c.plot(
        r_grid, g_fit_hc, color="tab:red", lw=1.2, ls="--", alpha=0.85,
        label="hard-core fitted step (R_hat)",
    )
    ax2c.axhline(
        1.0, color="0.4", lw=0.8, ls=":", label="theoretical CSR null: g(r)=1",
    )
    ax2c.axvline(hc["R"], color="k", lw=0.8, ls=":", alpha=0.6)
    ax2c.set_xlabel("lag r")
    ax2c.set_ylabel("g(r)")
    ax2c.set_title("Empirical vs. fitted pair correlation", fontsize=9.5)
    ax2c.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    # The CSR curve sits elevated above the theoretical g=1 null at short
    # lags and only settles near it further out -- a finite-sample
    # kernel-bias/edge artifact at n ~ 50-70 with no boundary correction
    # in nstat.extras.spatial.pair_correlation (Baddeley-Rubak-Turner
    # 2015 Sec. 13), NOT evidence of clustering. The diagnostic signal
    # the recovery table quantifies is the *absence* of a toward-zero
    # dip in the CSR curve, unlike the hard-core curve's clean dip to 0
    # below R -- that contrast, not the CSR curve's absolute height,
    # is what separates "no structure" from "hard exclusion" here.
    ax2c.text(
        0.97, 0.97,
        "CSR elevation at small r: finite-sample\n"
        "kernel-bias artifact (no edge correction),\n"
        "not clustering. Diagnostic signal = no\n"
        "toward-zero dip (cf. hard-core below R).",
        transform=ax2c.transAxes, ha="right", va="top", fontsize=6.3,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
    )

    # Recovered-vs-true bar chart: hard-core exclusion distance R on the
    # left axis, CSR interaction strength gamma on a twin right axis
    # (the two quantities live on very different scales).
    ax2d.bar(
        [0, 1], [hc["R"], hc["r_hat_min_distance"]], width=0.6,
        color=["0.55", "tab:red"],
    )
    ax2d.set_ylabel("hard-core exclusion distance R")
    r_ymax = max(hc["R"], hc["r_hat_min_distance"]) * 1.5
    ax2d.set_ylim(0.0, r_ymax)

    ax2d2 = ax2d.twinx()
    ax2d2.bar(
        [2.4, 3.4], [csr["gamma_true"], csr["gamma_hat"]], width=0.6,
        color=["0.55", "tab:blue"],
    )
    ax2d2.set_ylabel("Strauss gamma (CSR fit)")
    ax2d2.set_ylim(0.0, 1.5)

    ax2d.set_xticks([0, 1, 2.4, 3.4])
    ax2d.set_xticklabels(
        ["R\n(true)", "R\n(recovered)", "gamma\n(true, CSR)", "gamma\n(fitted, CSR)"],
        fontsize=7.5,
    )
    ax2d.set_xlim(-0.7, 4.1)
    ax2d.set_title("Recovered vs. true: exclusion distance & CSR interaction", fontsize=9.5)

    fig2.suptitle(
        "Co-active/implanted-site spacing: CSR vs. minimum-distance exclusion",
        fontsize=11,
    )
    # === END FIGURE ===

    # === FIGURE: fig03_area_interaction_scatter.png ===
    fig3, ax3 = plt.subplots(figsize=(5.5, 5.5))
    _plot_scatter_with_radius(
        ax3, ai["points"], ai["R"],
        f"Area-interaction (secondary) — beta={ai['beta_true']}, "
        f"eta={ai['eta_true']}, R={ai['R']} — n={ai['n_data']}",
    )
    # === END FIGURE ===

    figures = [fig1, fig2, fig3]
    fig_names = (
        "fig01_strauss_scatter",
        "fig02_hardcore_scatter",
        "fig03_area_interaction_scatter",
    )
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras" / "spatial_gibbs"
            )
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        for fig, name in zip(figures, fig_names):
            path = export_dir / f"{name}.png"
            fig.savefig(path, dpi=180, facecolor="w", edgecolor="none")
            figure_paths.append(path)
            print(f"  Saved: {path}")

    if visible:
        plt.show()
    else:
        plt.close("all")

    return {
        "csr": csr,
        "hardcore": hc,
        "strauss": st,
        "area_interaction": ai,
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CSR vs. minimum-distance exclusion (hard-core) "
                    "co-active/implanted-site spacing, plus secondary "
                    "Strauss/area-interaction Gibbs processes",
    )
    parser.add_argument(
        "--seed", type=int, default=20260616,
        help="np.random.default_rng seed.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write the three PNGs to --export-dir.",
    )
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="Override the PNG export directory.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write a compact recovery summary as JSON.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without showing figures (headless).",
    )
    parser.add_argument(
        "--plot-style", choices=("modern", "legacy"), default="legacy",
        help="Figure styling forwarded to nstat.apply_plot_style.",
    )
    args = parser.parse_args(argv)

    if args.no_display:
        import matplotlib
        matplotlib.use("Agg")
        visible = False
    else:
        visible = bool(args.show)

    result = run_demo(
        seed=args.seed,
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        visible=visible,
        plot_style=args.plot_style,
    )

    if args.output_json is not None:
        summary = {
            "csr": {
                "n_data": result["csr"]["n_data"],
                "beta_true": result["csr"]["beta_true"],
                "beta_hat": result["csr"]["beta_hat"],
                "gamma_true": result["csr"]["gamma_true"],
                "gamma_hat": result["csr"]["gamma_hat"],
                "fit_converged": result["csr"]["fit_converged"],
            },
            "hardcore": {
                "n_data": result["hardcore"]["n_data"],
                "beta_true": result["hardcore"]["beta_true"],
                "beta_hat": result["hardcore"]["beta_hat"],
                "R": result["hardcore"]["R"],
                "r_hat_min_distance": result["hardcore"]["r_hat_min_distance"],
                "fit_converged": result["hardcore"]["fit_converged"],
            },
            "strauss": {
                "n_data": result["strauss"]["n_data"],
                "beta_true": result["strauss"]["beta_true"],
                "beta_hat": result["strauss"]["beta_hat"],
                "gamma_true": result["strauss"]["gamma_true"],
                "gamma_hat": result["strauss"]["gamma_hat"],
                "fit_converged": result["strauss"]["fit_converged"],
            },
            "area_interaction": {
                "n_data": result["area_interaction"]["n_data"],
                "beta_true": result["area_interaction"]["beta_true"],
                "beta_hat": result["area_interaction"]["beta_hat"],
                "eta_true": result["area_interaction"]["eta_true"],
                "eta_hat": result["area_interaction"]["eta_hat"],
                "fit_converged": result["area_interaction"]["fit_converged"],
            },
            "figure_paths": result["figure_paths"],
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
