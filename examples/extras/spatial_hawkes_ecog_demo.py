#!/usr/bin/env python3
"""Demo: seizure "core" vs. "penumbra" via space-time Hawkes + Cox-Hawkes
on an ECoG grid.

End-to-end exercise of the space-time self-exciting (ETAS-style) Hawkes
process and the Cox-Hawkes background/excitation decomposition shipped in
:mod:`nstat.extras.spatial`, grounded in a realistic **ECoG surface-grid**
recording geometry:

**Clinical question.**  Human intracranial recordings of focal seizures
show a sharp spatial split between a small territory of intense,
hypersynchronous firing where neurons are actively **recruited** into the
seizure (the "ictal core") and a surrounding "ictal penumbra" where
large-amplitude field potentials mask only sparse, unstructured firing,
held back by a fast feedforward inhibitory restraint (Schevon et al.
2012).  That same recruited-core / restrained-penumbra split organizes
seizure activity across spatial scales as a self-propagating wave
(Martinet et al. 2017).  Separately, point-process/GLM analysis of
human single-unit **ensemble** spiking during focal seizures -- the
kind of single-unit ensemble data a spatiotemporal Hawkes model targets
-- found spiking largely irregular and asynchronous during gamma-band
seizures, versus coarse (50-100 ms) phase-locked synchrony -- with only
transient fine (<10 ms) synchrony in the initial ~20 ms of the
spike-wave-complex phase -- during spike-wave-complex seizures
(Truccolo et al. 2014).
This demo asks: **given only observed event
times and positions, can a background/triggering decomposition recover
which events were pulled into the self-exciting recruitment cascade (the
core) vs. which were unrecruited background activity (the penumbra), and
can it also localize *where* on the grid the underlying hyperexcitable
tissue (a seizure-onset zone) sits?**

**Modeling correspondence.**  The self-exciting Hawkes triggering kernel
*is* the recruitment mechanism that builds a core: each discharge raises
the near-term, nearby-in-space probability that its neighbours are
recruited into synchronized firing -- exactly the hypersynchronous-cascade
phenomenology above.  The background intensity -- homogeneous in
catalogue A, an inhomogeneous log-Gaussian-Cox (LGCP) field anchored at a
fixed hyperexcitable "seizure-onset zone" in catalogue B -- is the source
of unrecruited, penumbra-like activity.  Declustering an event catalogue
(branching EM's ``P(background)`` responsibilities) therefore *is* a
core/penumbra classification at the single-event level: low
``P(background)`` events were recruited into a cascade (core); high
``P(background)`` events were not (penumbra).

**Scenario.**  An **8x8 ECoG grid** (1cm pitch) captures self-exciting
spatiotemporal spread of cortical events over a multi-minute continuous
epoch -- the textbook model for interictal-spike propagation or
travelling waves: each event *raises the probability* of a
nearby-and-soon-after event (recruitment into the core), on top of a
background rate that supplies the unrecruited penumbra activity.  Both
catalogues below demonstrate that *same* core(=triggered)/
penumbra(=background) decomposition within a single recording; they
differ only in the background model the estimator must contend with, not
in which one is "core" and which is "penumbra".  Catalogue A gives the
background a **homogeneous** rate everywhere on the grid -- no spatial
landmark to go by, so branching EM's per-event declustering is the only
way to tell core (triggered) from penumbra (background) events apart.
Catalogue B instead anchors the background to a fixed,
spatially-**inhomogeneous** hyperexcitable "seizure-onset zone", so that
both the elevated background rate and the self-exciting cascade it seeds
concentrate near that zone, and the Cox-Hawkes alternating estimator must
separate the slow hyperexcitable-tissue field from the fast recruitment
kernel riding on top of it.  Clinical interictal-spike-propagation studies routinely
analyze continuous multi-minute (not sub-second) epochs precisely because
the temporal/spatial triggering kernel is only well identified once many
recruitment cascades have been observed -- this demo's ~4-8 minute
synthetic epochs are chosen for that reason, not for narrative
convenience.  All catalogues are fully synthetic branching simulations --
no real recording or dataset is used or claimed.

**Novelty.**  Cox-Hawkes (Miscouridou et al. 2022) -- a doubly-stochastic
LGCP-background-plus-Hawkes-triggering model developed in the statistics
literature for seismology/social-media cascades -- has not, to our
knowledge, previously been applied to the epilepsy core/penumbra
decomposition.  This demo presents that combination as a methodological
opportunity for separating recruited-core from restrained-penumbra
activity and localizing the seizure-onset zone, not as a result drawn
from any published epilepsy study.

Demonstrates:

1. :func:`nstat.extras.spatial.simulate_spatial_hawkes` -- catalogue A,
   the **homogeneous-background variant**: a synthetic recruitment-cascade
   catalogue on the grid with a spatially uniform background.
2. :func:`nstat.extras.spatial.em_spatial_hawkes` -- recover
   ``(mu, K_branch, c, sigma_space)`` by branching EM; ``sigma_space`` is
   the **spatial spread of recruitment** (how far, in cm, one discharge's
   influence reaches) and ``1 / c`` is the **temporal recruitment window**
   (how soon recruited offspring tend to follow, in seconds); the
   returned responsibilities also give a per-event core/penumbra
   declustering.
3. :func:`nstat.extras.spatial.simulate_cox_hawkes` /
   :func:`nstat.extras.spatial.fit_cox_hawkes` -- catalogue B, the
   **inhomogeneous LGCP-background variant**: a second catalogue with a
   fixed hyperexcitable "seizure-onset zone" background plus the same
   self-exciting recruitment; the alternating estimator separates the
   slow hyperexcitable-tissue (penumbra-source) field from the fast
   recruitment (core) kernel.

See also :mod:`spatial_gof_ecog_demo` -- asks whether concentrated ECoG
activity is a genuinely propagating wavefront or a static hot-spot, the
question this demo's core/penumbra decomposition presupposes an answer
to; and :mod:`spatial_gibbs_demo` -- a complementary second-order-structure
story (repulsion / exclusion-zone spacing) for electrode and contact
placement rather than event recruitment.

The script is **fully synthetic** -- no figshare dataset access required.

Run::

    python examples/extras/spatial_hawkes_ecog_demo.py            # interactive
    python examples/extras/spatial_hawkes_ecog_demo.py --no-display
    python examples/extras/spatial_hawkes_ecog_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/spatial_hawkes_ecog/``) and are NOT committed to
the repository -- the export flag exists for local inspection only.  CI
never invokes it.

References:

Clinical motivation (human ictal core/penumbra structure):

- Schevon CA, Weiss SA, McKhann G Jr, Goodman RR, Yuste R, Emerson RG,
  Trevelyan AJ (2012). *Evidence of an inhibitory restraint of seizure
  activity in humans.* Nat Commun 3:1060.
- Truccolo W, Ahmed OJ, Harrison MT, Eskandar EN, Cosgrove GR, Madsen JR,
  Blum AS, Potter NS, Hochberg LR, Cash SS (2014). *Neuronal ensemble
  synchrony during human focal seizures.* J Neurosci 34(30):9927-9944.
- Martinet LE, Fiddyment G, Madsen JR, Eskandar EN, Truccolo W, Eden UT,
  Cash SS, Kramer MA (2017). *Human seizures couple across spatial scales
  through travelling wave dynamics.* Nat Commun 8:14896.

Statistical methods:

- Veen A, Schoenberg FP (2008). *Estimation of space-time branching
  process models in seismology using an EM-type algorithm.* JASA
  103(482):614-624.
- Zhuang J, Ogata Y, Vere-Jones D (2002). *Stochastic declustering of
  space-time earthquake occurrences.* JASA 97(458):369-380.
- Ogata Y (1998). *Space-time point-process models for earthquake
  occurrences.* Ann. Inst. Statist. Math. 50(2):379-402.
- Miscouridou X, Bhatt S, Mohler G, Flaxman S, Bhamidi S (2022).
  *Cox-Hawkes: doubly stochastic spatiotemporal Poisson processes.* TMLR.
  (Applying it to the epilepsy core/penumbra decomposition is, to our
  knowledge, novel -- see "Novelty" above.)
- Moller J, Rasmussen JG (2005). *Perfect simulation of Hawkes
  processes.* Adv. Appl. Probab. 37(3):629-646 (branching simulator).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 8x8 ECoG grid geometry (1cm pitch)
# ---------------------------------------------------------------------------

N_SIDE = 8
PITCH_CM = 1.0
MARGIN_CM = 0.5
GRID_EDGE_CM = 2.0 * MARGIN_CM + PITCH_CM * (N_SIDE - 1)  # == 8.0 cm

DOMAIN = ((0.0, GRID_EDGE_CM), (0.0, GRID_EDGE_CM))

# Catalogue A: homogeneous-background variant -- a spatially uniform
# background (no anatomical landmark) plus the same self-exciting
# recruitment cascade (core) as catalogue B.  Branching EM recovers the
# core (triggered) vs. penumbra (background) split for each event via its
# ``P(background)`` responsibilities.  ``c`` (temporal decay, /s) and
# ``T`` are chosen so that T is long relative to the mean recruitment
# delay 1/c -- branching EM only identifies (K_branch, c) well once many
# recruitment cascades have been observed (per the companion test
# ``tests/extras/test_spatial_spatial_hawkes.py::test_em_recovers_all_four_params``,
# which uses the same T >> 1/c regime).
MU_TRUE = 0.4  # events/s
K_BRANCH_TRUE = 0.35
C_TRUE = 1.2  # mean triggering delay 1/c ~= 0.83s
SIGMA_SPACE_TRUE = 0.35
T_A = 500.0  # ~8.3 minute continuous epoch

# Catalogue B: inhomogeneous LGCP-background variant -- a spatially
# varying (Cox) hyperexcitable "seizure-onset zone" background, near
# which the self-exciting recruitment cascade concentrates, plus the
# same self-excitation kernel as catalogue A.  The Cox-Hawkes alternating
# estimator recovers the same core (triggered) vs. penumbra (background)
# split as catalogue A, this time against a non-uniform background field.
#
# ``fit_cox_hawkes`` has no ``length_scale_space``/``length_scale_time``
# passthrough to its internal ``lgcp_st_fit`` call -- it always uses that
# function's defaults (``length_scale_space=0.12``, ``length_scale_time=0.1``),
# which are calibrated for a *unit-square* domain (matching the module's own
# ``DOMAIN_UNIT`` test-suite convention in
# ``tests/extras/test_spatial_cox_hawkes.py``).  Fitting directly on an
# 8cm-wide physical domain would make that length scale ~1.5% of the domain
# width -- far shorter than one grid cell -- and produce an under-smoothed,
# speckled background estimate unrelated to the true field.  Catalogue B is
# therefore simulated and fit on a **normalized unit-square domain**
# (1.0 == ``GRID_EDGE_CM``); physical (x, y) in cm is recovered by
# multiplying by ``GRID_EDGE_CM`` for axis labels / electrode overlays.
DOMAIN_B = ((0.0, 1.0), (0.0, 1.0))
FOCUS_CENTER_CM = np.array([2.0, 6.0])
FOCUS_CENTER = FOCUS_CENTER_CM / GRID_EDGE_CM
FOCUS_SIGMA_CM = 1.2
FOCUS_SIGMA = FOCUS_SIGMA_CM / GRID_EDGE_CM
BASELINE_RATE_CM = 0.03  # Hz/cm^2
AMP_RATE_CM = 0.06  # Hz/cm^2
# Density scales as 1/area under a unit rescale: lambda_unit = lambda_cm * GRID_EDGE_CM^2.
BASELINE_RATE = BASELINE_RATE_CM * GRID_EDGE_CM**2
AMP_RATE = AMP_RATE_CM * GRID_EDGE_CM**2
K_BRANCH_COX_TRUE = 0.35
C_COX_TRUE = 1.2  # temporal decay is unaffected by the spatial rescale
SIGMA_SPACE_COX_TRUE_CM = 0.35
SIGMA_SPACE_COX_TRUE = SIGMA_SPACE_COX_TRUE_CM / GRID_EDGE_CM
T_B = 250.0  # ~4.2 minute continuous epoch
LGCP_GRID = (10, 10, 6)


def _electrode_positions() -> np.ndarray:
    coords = MARGIN_CM + PITCH_CM * np.arange(N_SIDE)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _cox_background_rate(x: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Background rate on the normalized [0, 1]^2 domain (see DOMAIN_B note)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    d2 = np.sum((x - FOCUS_CENTER[None, :]) ** 2, axis=1)
    return BASELINE_RATE + AMP_RATE * np.exp(-d2 / (2.0 * FOCUS_SIGMA**2))


def _clip_to_domain(
    points: np.ndarray, times: np.ndarray, domain=DOMAIN
) -> tuple[np.ndarray, np.ndarray]:
    (xlo, xhi), (ylo, yhi) = domain
    mask = (
        (points[:, 0] >= xlo) & (points[:, 0] <= xhi)
        & (points[:, 1] >= ylo) & (points[:, 1] <= yhi)
    )
    return points[mask], times[mask]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_demo(
    *,
    seed: int = 20260703,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool = True,
    plot_style: str = "legacy",
) -> dict:
    """Run the ECoG seizure core/penumbra space-time Hawkes + Cox-Hawkes demo.

    Returns
    -------
    dict
        ``{"hawkes": ..., "cox_hawkes": ..., "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style
    from nstat.extras.spatial import (
        SpatialHawkesSpec,
        em_spatial_hawkes,
        fit_cox_hawkes,
        simulate_cox_hawkes,
        simulate_spatial_hawkes,
    )

    print("=" * 72)
    print("Seizure core/penumbra decomposition via space-time Hawkes "
          "on a synthetic 8x8 ECoG grid")
    print("=" * 72)
    print(
        f"Grid footprint: {GRID_EDGE_CM:.1f}cm x {GRID_EDGE_CM:.1f}cm, "
        f"{N_SIDE}x{N_SIDE} electrodes, {PITCH_CM:.0f}cm pitch "
        "(fully synthetic events -- no real recording)"
    )

    rng = np.random.default_rng(seed)
    # Catalogue B gets its own decoupled stream (rather than continuing to
    # draw from ``rng`` after catalogue A) so its realization -- and hence
    # the visual quality of the LGCP-background recovery in fig03 -- does
    # not silently shift whenever catalogue A's simulation code changes.
    rng_b = np.random.default_rng(seed + 3)

    # ---- Catalogue A: homogeneous-background variant + the same
    # self-exciting recruitment (core) cascade as catalogue B. ----
    pts_a, t_a = simulate_spatial_hawkes(
        MU_TRUE, K_BRANCH_TRUE, C_TRUE, SIGMA_SPACE_TRUE,
        domain=DOMAIN, T=T_A, rng=rng,
    )
    print(
        f"Catalogue A -- homogeneous-background variant: "
        f"{pts_a.shape[0]} events over {T_A:.1f}s"
    )

    fit_a = em_spatial_hawkes(
        pts_a, t_a, domain=DOMAIN, T=T_A,
        spec=SpatialHawkesSpec(K0=0.3, c0=1.0, sigma0=0.3, max_iter=500),
        return_responsibilities=True,
    )
    print()
    print("Catalogue A recovery (branching EM -- core-cascade kernel + penumbra declustering):")
    print(f"  {'param':>14} | {'true':>8} | {'fitted':>8}")
    print(f"  {'mu (rate, /s)':>14} | {MU_TRUE:8.3f} | {fit_a.mu_hat:8.3f}")
    print(f"  {'K_branch':>14} | {K_BRANCH_TRUE:8.3f} | {fit_a.K_branch_hat:8.3f}")
    print(f"  {'c (1/s)':>14} | {C_TRUE:8.3f} | {fit_a.c_hat:8.3f}")
    print(f"  {'sigma_space (cm)':>14} | {SIGMA_SPACE_TRUE:8.3f} | {fit_a.sigma_space_hat:8.3f}")
    print(
        f"  interpretation: core recruitment reaches ~{fit_a.sigma_space_hat:.2f}cm "
        f"spatially and decays with a ~{1.0 / fit_a.c_hat:.2f}s time window "
        f"(converged={fit_a.converged}, n_iter={fit_a.n_iter})"
    )

    # ---- Catalogue B: inhomogeneous LGCP-background variant
    # (seizure-onset-zone background) + the same self-exciting
    # recruitment cascade as catalogue A. ----
    # Simulated + fit on the normalized unit-square DOMAIN_B (see the
    # module-level note by DOMAIN_B for why).
    pts_b, t_b = simulate_cox_hawkes(
        _cox_background_rate, K_BRANCH_COX_TRUE, C_COX_TRUE, SIGMA_SPACE_COX_TRUE,
        domain=DOMAIN_B, T=T_B, rng=rng_b,
    )
    pts_b, t_b = _clip_to_domain(pts_b, t_b, domain=DOMAIN_B)
    print()
    print(
        f"Catalogue B -- inhomogeneous LGCP-background variant "
        f"(seizure-onset-zone background): {pts_b.shape[0]} events over "
        f"{T_B:.1f}s"
    )

    fit_b = fit_cox_hawkes(
        pts_b, t_b, domain=DOMAIN_B, period=(0.0, T_B), grid=LGCP_GRID,
        max_outer=10, tol=1e-3,
        hawkes_spec=SpatialHawkesSpec(K0=0.3, c0=1.0, sigma0=SIGMA_SPACE_COX_TRUE),
    )
    print()
    print(
        "Catalogue B recovery (Cox-Hawkes alternating estimator -- "
        "seizure-onset-zone field + core-cascade kernel):"
    )
    print(f"  {'param':>14} | {'true':>8} | {'fitted':>8}")
    print(f"  {'K_branch':>14} | {K_BRANCH_COX_TRUE:8.3f} | {fit_b.K_branch_hat:8.3f}")
    print(f"  {'c (1/s)':>14} | {C_COX_TRUE:8.3f} | {fit_b.c_hat:8.3f}")
    print(
        f"  {'sigma_space (cm)':>14} | {SIGMA_SPACE_COX_TRUE_CM:8.3f} | "
        f"{fit_b.sigma_space_hat * GRID_EDGE_CM:8.3f}"
    )
    print(
        f"  background_fraction={fit_b.background_fraction:.3f} (penumbra, "
        f"i.e. background, share of events; converged={fit_b.converged}, "
        f"n_outer={fit_b.n_outer})"
    )

    # ---- Figures ----
    electrodes = _electrode_positions()

    # === FIGURE: fig01_propagation_scatter.png ===
    # A short excerpt (not the full multi-minute catalogue) is shown,
    # coloured by elapsed time, so that visually tight temporal clusters
    # -- suggestive of individual recruitment cascades building a "core"
    # -- stand out against the sparser, scattered "penumbra" background
    # events.  Colour encodes time only, not P(background), so a tight
    # cluster is suggestive of a cascade, not proof of one (temporally
    # close background events can look the same).
    excerpt_s = 30.0
    excerpt_mask = t_a <= excerpt_s
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.6))
    ax1.scatter(
        electrodes[:, 0], electrodes[:, 1],
        marker="s", s=10, color="0.6", zorder=1, label="electrode",
    )
    sc = ax1.scatter(
        pts_a[excerpt_mask, 0], pts_a[excerpt_mask, 1], c=t_a[excerpt_mask],
        s=22, cmap="plasma", alpha=0.85, zorder=2, label="event",
    )
    fig1.colorbar(sc, ax=ax1, label="time (s)")
    ax1.set_xlim(*DOMAIN[0])
    ax1.set_ylim(*DOMAIN[1])
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (cm)")
    ax1.set_ylabel("y (cm)")
    ax1.set_title(
        f"Catalogue A (homogeneous-background variant): first "
        f"{excerpt_s:.0f}s excerpt ({int(excerpt_mask.sum())} of "
        f"{pts_a.shape[0]} events), coloured by time\n"
        f"tight same-colour clusters suggest core recruitment cascades\n"
        f"full catalogue spans {T_A:.0f}s on the {N_SIDE}x{N_SIDE} ECoG grid",
        fontsize=10,
    )
    ax1.legend(loc="upper left", fontsize=8)
    # === END FIGURE ===

    # === FIGURE: fig02_recovered_kernels.png ===
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    t_grid = np.linspace(0.0, 5.0, 200)
    ax2a.plot(t_grid, C_TRUE * np.exp(-C_TRUE * t_grid), color="black", lw=1.4,
              ls=":", label="true g(t)")
    ax2a.plot(
        t_grid, fit_a.c_hat * np.exp(-fit_a.c_hat * t_grid), color="tab:red",
        lw=1.8, label="recovered g(t)",
    )
    ax2a.set_xlabel("elapsed time since parent (s)")
    ax2a.set_ylabel("g(t) = c * exp(-c t)")
    ax2a.set_title("Temporal recruitment (core) kernel")
    ax2a.legend(loc="upper right", fontsize=8)

    r_grid = np.linspace(0.0, 2.0, 200)
    h_true = np.exp(-r_grid**2 / (2 * SIGMA_SPACE_TRUE**2)) / (
        2 * np.pi * SIGMA_SPACE_TRUE**2
    )
    h_hat = np.exp(-r_grid**2 / (2 * fit_a.sigma_space_hat**2)) / (
        2 * np.pi * fit_a.sigma_space_hat**2
    )
    ax2b.plot(r_grid, h_true, color="black", lw=1.4, ls=":", label="true h(r)")
    ax2b.plot(r_grid, h_hat, color="tab:red", lw=1.8, label="recovered h(r)")
    ax2b.set_xlabel("spatial offset r (cm)")
    ax2b.set_ylabel("h(r), isotropic Gaussian")
    ax2b.set_title("Spatial recruitment (core) kernel")
    ax2b.legend(loc="upper right", fontsize=8)
    fig2.suptitle(
        "Recovered self-exciting recruitment-cascade (core) kernels "
        "(catalogue A, homogeneous-background variant)"
    )
    # === END FIGURE ===

    # === FIGURE: fig03_background_excitation_decomposition.png ===
    fig3, axes3 = plt.subplots(1, 3, figsize=(15.5, 5.0))

    resp_diag = np.asarray(fit_a.responsibilities.diagonal()) if (
        fit_a.responsibilities is not None
    ) else np.full(pts_a.shape[0], np.nan)
    sc3 = axes3[0].scatter(
        pts_a[:, 0], pts_a[:, 1], c=resp_diag, s=18, cmap="coolwarm",
        vmin=0.0, vmax=1.0,
    )
    fig3.colorbar(sc3, ax=axes3[0], label="P(background) = P(penumbra)")
    axes3[0].set_xlim(*DOMAIN[0])
    axes3[0].set_ylim(*DOMAIN[1])
    axes3[0].set_aspect("equal")
    axes3[0].set_xlabel("x (cm)")
    axes3[0].set_ylabel("y (cm)")
    axes3[0].set_title("Core vs. penumbra (catalogue A)", fontsize=10)

    # Displayed in physical cm units for readability (see the DOMAIN_B note):
    # rate density lambda_cm = lambda_unit / GRID_EDGE_CM^2, x_cm = x_unit * GRID_EDGE_CM.
    Gx_l, Gy_l, _ = LGCP_GRID
    grid_x_bg = fit_b.background.grid_x
    true_bg_vals = _cox_background_rate(grid_x_bg, np.zeros(grid_x_bg.shape[0]))
    true_bg_vals_cm = true_bg_vals / GRID_EDGE_CM**2
    cm_extent = (0.0, GRID_EDGE_CM, 0.0, GRID_EDGE_CM)
    im_true = axes3[1].imshow(
        true_bg_vals_cm.reshape(Gy_l, Gx_l), origin="lower", cmap="viridis",
        extent=cm_extent, aspect="equal",
    )
    axes3[1].set_xlabel("x (cm)")
    axes3[1].set_ylabel("y (cm)")
    axes3[1].set_title("True seizure-onset-zone rate (catalogue B)")
    fig3.colorbar(im_true, ax=axes3[1], fraction=0.046, pad=0.04, label="Hz/cm^2")

    mean_bg, _lo_bg, _hi_bg = fit_b.background.rate_map(T_B / 2.0, level=0.9)
    mean_bg_cm = mean_bg / GRID_EDGE_CM**2
    im_fit = axes3[2].imshow(
        mean_bg_cm.reshape(Gy_l, Gx_l), origin="lower", cmap="viridis",
        extent=cm_extent, aspect="equal",
    )
    axes3[2].set_xlabel("x (cm)")
    axes3[2].set_ylabel("y (cm)")
    axes3[2].set_title("Fitted seizure-onset-zone rate (catalogue B)")
    fig3.colorbar(im_fit, ax=axes3[2], fraction=0.046, pad=0.04, label="Hz/cm^2")

    fig3.suptitle("Core (self-excited cascade) vs. penumbra (background) decomposition")
    # === END FIGURE ===

    figures = [fig1, fig2, fig3]
    fig_names = (
        "fig01_propagation_scatter",
        "fig02_recovered_kernels",
        "fig03_background_excitation_decomposition",
    )
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras" / "spatial_hawkes_ecog"
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
        "hawkes": fit_a,
        "cox_hawkes": fit_b,
        "n_a": int(pts_a.shape[0]),
        "n_b": int(pts_b.shape[0]),
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seizure core/penumbra decomposition via space-time "
                    "Hawkes + Cox-Hawkes on a synthetic ECoG grid",
    )
    parser.add_argument(
        "--seed", type=int, default=20260703,
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
        hawkes = result["hawkes"]
        cox = result["cox_hawkes"]
        summary = {
            "n_a": result["n_a"],
            "n_b": result["n_b"],
            "hawkes_mu_hat": hawkes.mu_hat,
            "hawkes_K_branch_hat": hawkes.K_branch_hat,
            "hawkes_c_hat": hawkes.c_hat,
            "hawkes_sigma_space_hat": hawkes.sigma_space_hat,
            "hawkes_converged": bool(hawkes.converged),
            "cox_hawkes_K_branch_hat": cox.K_branch_hat,
            "cox_hawkes_c_hat": cox.c_hat,
            # Reported in cm (normalized-unit fit value * GRID_EDGE_CM);
            # see the DOMAIN_B note near the top of this file.
            "cox_hawkes_sigma_space_hat_cm": cox.sigma_space_hat * GRID_EDGE_CM,
            "cox_hawkes_background_fraction": cox.background_fraction,
            "cox_hawkes_converged": bool(cox.converged),
            "figure_paths": result["figure_paths"],
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
