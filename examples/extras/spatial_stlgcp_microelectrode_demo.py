#!/usr/bin/env python3
"""Demo: tracking a spreading-depolarisation / seizure wavefront across a microelectrode array.

End-to-end exercise of the two spatiotemporal rate-estimation modules
shipped in :mod:`nstat.extras.spatial` (space-time KDE + spatiotemporal
LGCP), grounded in a realistic microelectrode-array recording geometry:

**Clinical question.**  Spreading depolarisation (SD) and the fast ictal
wavefronts that accompany human focal seizures sit on the same
pathophysiological continuum: a self-propagating wave of near-complete
neuronal depolarisation that recruits adjacent cortex as it migrates
(Lauritzen et al. 2011).  In the neuro-ICU, subdural strip electrodes
continuously screen traumatic-brain-injury patients for these events,
whose frequency and duration independently predict poor outcome
(Hartings et al. 2011).  In the epilepsy-monitoring-unit setting, an
analogous propagating wavefront -- not a static "hot-spot" -- has been
shown to organise the spread of the ictal discharge itself (Diamond et
al. 2021).  Cortical travelling waves of every speed, from sensory-evoked
responses to these pathological SD/ictal fronts, share the same
observational problem (Muller et al. 2018): the wave itself is never
observed directly, only **discharge events at a sparse set of
electrodes**.  This demo asks: **can a wavefront's trajectory and
propagation speed be recovered from those sparse events alone, without
fitting a parametric wave-equation model to the data?**

**Scenario.**  A **Utah-style 10x10 microelectrode array** (4mm x 4mm
footprint, 400um electrode pitch, 200um edge margin -- a high-density
microelectrode geometry used to probe seizure dynamics at fine spatial
resolution) records population multiunit activity during a single short
recording epoch.  The population's space-time firing-rate field is a
Gaussian wavefront of elevated activity that **translates across the
array at a constant ~1.9 mm/s (~113 mm/min)** -- a speed at the fast,
ictal end of the SD-to-seizure continuum, chosen so the wavefront is
trackable within a short recording window on a 4mm array.  True cortical
spreading depression proper is roughly 20-60x slower (2-5 mm/min, per
Lauritzen et al. 2011) and would take tens of minutes to cross the same
array. All
spikes are drawn from a known, fully synthetic space-time Poisson
intensity (Lewis-Shedler thinning) -- **no real recording or dataset is
used or claimed**.

Demonstrates:

1. :func:`nstat.extras.spatial.intensity_st_kde` -- a boundary-corrected
   space-time kernel estimate lambda_hat(x, t) of the moving wavefront,
   read off at several time slices.
2. :func:`nstat.extras.spatial.lgcp_st_fit` -- a Kronecker-Laplace
   spatiotemporal log-Gaussian Cox process fit, giving a posterior
   *mean* rate map **with credible bands** (:meth:`LGCPSTResult.rate_map`)
   at the same time slices.
3. A recovery table comparing the true, KDE-estimated, and LGCP-estimated
   wavefront centroid (mm) at each slice -- "does the fitted wavefront
   track the true wavefront?" -- plus a **propagation-speed readout**: a
   line fit through the recovered peak (centroid) locations over time
   recovers the wavefront's speed (mm/s), compared against the true,
   by-construction speed.

**Novelty.**  Established methods for characterising cortical travelling
waves fit an explicit parametric wave model to the data up front -- plane-
or spherical-wave-equation fitting to per-channel time-of-arrival, optical
-flow estimation on dense imaging arrays, or pairwise cross-correlation
/time-lag analysis (the methodological toolkit surveyed in Muller et al.
2018).  A spatiotemporal log-Gaussian Cox process instead reconstructs the
underlying intensity surface nonparametrically and only afterward reads a
trajectory/speed off that reconstruction, with no wave-shape assumption
built in.  To our knowledge this LGCP-based approach has not previously
been applied to spreading-depolarisation or seizure-wavefront tracking.
This demo presents that combination as a methodological opportunity, not
as a result drawn from any published SD/epilepsy study.

The script is **fully synthetic** -- no figshare dataset access required.

Run::

    python examples/extras/spatial_stlgcp_microelectrode_demo.py            # interactive
    python examples/extras/spatial_stlgcp_microelectrode_demo.py --no-display
    python examples/extras/spatial_stlgcp_microelectrode_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/spatial_stlgcp_microelectrode/``) and are NOT
committed to the repository -- the export flag exists for local
inspection only.  CI never invokes it.

See also :mod:`spatial_gof_ecog_demo` -- that demo asks *whether* a
recording contains a genuine travelling wavefront at all (a space-time
goodness-of-fit test against an inhomogeneous-Poisson null); this demo
assumes the answer is yes and asks the follow-up question, *what is the
wavefront's trajectory and propagation speed*.

References:

Clinical motivation (spreading depolarisation / seizure wavefronts):

- Muller L, Chavane F, Reynolds J, Sejnowski TJ (2018). *Cortical
  travelling waves: mechanisms and computational principles.* Nat Rev
  Neurosci 19(5):255-268.
- Lauritzen M, Dreier JP, Fabricius M, Hartings JA, Graf R, Strong AJ
  (2011). *Clinical relevance of cortical spreading depression in
  neurological disorders: migraine, malignant stroke, subarachnoid and
  intracranial hemorrhage, and traumatic brain injury.* J Cereb Blood
  Flow Metab 31(1):17-35.
- Hartings JA et al. (2011). *Spreading depolarisations and outcome after
  traumatic brain injury: a prospective observational study.* Lancet
  Neurol 10(12):1058-1064.
- Diamond JM, Diamond BE, Trotta MS, Dembny K, Inati SK, Zaghloul KA
  (2021). *Travelling waves reveal a dynamic seizure source in human
  focal epilepsy.* Brain 144(6):1751-1763.

Statistical methods:

- Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal
  Point Patterns* (3rd ed.). CRC Press, Chapter 7 (space-time kernel
  intensity estimation).
- Moller J, Syversveen AR, Waagepetersen RP (1998). *Log Gaussian Cox
  processes.* Scand. J. Statistics 25(3):451-482.
- Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
  Learning*, Algorithm 3.1 (Laplace/Newton-IRLS posterior mode).
- Saatci Y (2011). *Scalable Inference for Structured Gaussian Process
  Models.* PhD thesis, University of Cambridge, Ch. 5 (Kronecker/GP-grid
  inference used by :func:`lgcp_st_fit`).
- Maynard EM, Nordhausen CT, Normann RA (1997). *The Utah Intracortical
  Electrode Array: a recording structure for potential brain-computer
  interfaces.* Electroencephalography and Clinical Neurophysiology
  102(3):228-239 (the 10x10 / 400um-pitch array geometry used here).
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
# Utah-style 10x10 microelectrode array geometry
# ---------------------------------------------------------------------------

N_SIDE = 10
PITCH_MM = 0.4
MARGIN_MM = 0.2
ARRAY_EDGE_MM = 2.0 * MARGIN_MM + PITCH_MM * (N_SIDE - 1)  # == 4.0 mm

DOMAIN = ((0.0, ARRAY_EDGE_MM), (0.0, ARRAY_EDGE_MM))
PERIOD = (0.0, 1.5)  # a short ictal-wavefront recording epoch

# True wavefront parameters (population multiunit rate density, events
# per mm^2 per s -- a *population*-level density, not a single-unit Hz).
# The wavefront moves along the C0->C1 diagonal at a *constant* velocity,
# so its true propagation speed is the exact analytic value below -- see
# module docstring for why this sits at the fast (ictal), not the slow
# (spreading-depression), end of the SD-to-seizure continuum.
C0_TRUE = np.array([1.0, 1.0])
C1_TRUE = np.array([3.0, 3.0])
SIGMA_BUMP_TRUE = 0.6
BASELINE_TRUE = 40.0
AMP_TRUE = 250.0

TRUE_SPEED_MM_S = float(np.linalg.norm(C1_TRUE - C0_TRUE) / (PERIOD[1] - PERIOD[0]))

SLICE_FRACS = (0.2, 0.5, 0.8)


def _electrode_positions() -> np.ndarray:
    """``(100, 2)`` Utah-array electrode centres (mm)."""
    coords = MARGIN_MM + PITCH_MM * np.arange(N_SIDE)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _bump_center(t: np.ndarray) -> np.ndarray:
    """True bump centre(s) ``(n, 2)`` at time(s) ``t`` (broadcasts)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    frac = np.clip(t / PERIOD[1], 0.0, 1.0)
    return C0_TRUE[None, :] + (C1_TRUE - C0_TRUE)[None, :] * frac[:, None]


def _true_intensity(x: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """True space-time Poisson intensity of the moving wavefront."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    t_arr = np.broadcast_to(np.asarray(t, dtype=float).reshape(-1), (x.shape[0],))
    center = _bump_center(t_arr)
    d2 = np.sum((x - center) ** 2, axis=1)
    return BASELINE_TRUE + AMP_TRUE * np.exp(-d2 / (2.0 * SIGMA_BUMP_TRUE**2))


def _simulate_moving_bump(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Lewis-Shedler thinning simulation of the moving-wavefront intensity."""
    (xlo, xhi), (ylo, yhi) = DOMAIN
    tlo, thi = PERIOD
    vol = (xhi - xlo) * (yhi - ylo) * (thi - tlo)
    lam_max = BASELINE_TRUE + AMP_TRUE
    n_prop = int(rng.poisson(lam_max * vol))
    px = rng.uniform(xlo, xhi, n_prop)
    py = rng.uniform(ylo, yhi, n_prop)
    pt = rng.uniform(tlo, thi, n_prop)
    cand = np.column_stack([px, py])
    vals = _true_intensity(cand, pt)
    keep = rng.uniform(size=n_prop) < (vals / lam_max)
    order = np.argsort(pt[keep], kind="stable")
    return cand[keep][order], pt[keep][order]


def _weighted_centroid(values: np.ndarray, grid_x: np.ndarray) -> np.ndarray:
    """Background-subtracted intensity centroid -- a peak-location estimate.

    Most grid cells sit far from the wavefront and see only the spatially
    -uniform baseline rate; weighting by the *raw* rate (as opposed to the
    rate in excess of that baseline) lets the baseline's domain-wide mass
    dilute the peak location toward the array centre, an increasingly
    severe bias as the true wavefront approaches the array edges.
    Subtracting a robust (median) baseline estimate before weighting
    isolates the excess mass actually attributable to the wavefront.
    """
    values = np.asarray(values, dtype=float)
    excess = np.clip(values - float(np.median(values)), 0.0, None)
    total = float(excess.sum())
    if total <= 0.0:
        return np.full(2, np.nan)
    return (excess[:, None] * grid_x).sum(axis=0) / total


def _fit_propagation_speed(
    times: np.ndarray, centers: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Recover a wavefront's propagation speed from its peak trajectory.

    Fits an independent ordinary-least-squares line ``x(t) = c_x + v_x t``
    and ``y(t) = c_y + v_y t`` to the recovered (or true) peak/centroid
    locations, the same time-of-arrival-regression idea used to read a
    velocity off a travelling-wave trajectory.  Rows with a non-finite
    centroid (``_weighted_centroid`` returns NaN for an all-zero rate map)
    are dropped before fitting.

    Returns
    -------
    speed_mm_s, velocity, intercept : tuple[float, np.ndarray, np.ndarray]
        ``speed_mm_s`` is ``||velocity||``; ``velocity`` and ``intercept``
        are each ``(2,)`` (x, y) OLS coefficients.
    """
    times = np.asarray(times, dtype=float)
    centers = np.asarray(centers, dtype=float)
    finite = np.all(np.isfinite(centers), axis=1)
    times, centers = times[finite], centers[finite]
    if times.size < 2:
        return float("nan"), np.full(2, np.nan), np.full(2, np.nan)
    vx, cx = np.polyfit(times, centers[:, 0], 1)
    vy, cy = np.polyfit(times, centers[:, 1], 1)
    velocity = np.array([vx, vy])
    intercept = np.array([cx, cy])
    return float(np.linalg.norm(velocity)), velocity, intercept


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
    """Run the microelectrode-array space-time KDE + LGCP wavefront-tracking demo.

    Returns
    -------
    dict
        ``{"points", "times", "kde", "lgcp", "recovery", "true_speed_mm_s",
        "kde_speed_mm_s", "lgcp_speed_mm_s", "figure_paths"}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style
    from nstat.extras.spatial import intensity_st_kde, lgcp_st_fit

    print("=" * 72)
    print("Tracking a spreading-depolarisation / seizure wavefront -- ")
    print("space-time KDE + LGCP on a Utah-style 10x10 microelectrode array")
    print("=" * 72)
    print(
        f"Array footprint: {ARRAY_EDGE_MM:.1f}mm x {ARRAY_EDGE_MM:.1f}mm, "
        f"{N_SIDE}x{N_SIDE} electrodes, {PITCH_MM * 1000:.0f}um pitch "
        "(fully synthetic spikes -- no real recording)"
    )

    rng = np.random.default_rng(seed)
    points, times = _simulate_moving_bump(rng)
    n = points.shape[0]
    print(f"Simulated {n} events over a {PERIOD[1]:.2f}s trial")

    kde = intensity_st_kde(
        points, times, domain=DOMAIN, period=PERIOD, grid=(20, 20, 12)
    )
    lgcp = lgcp_st_fit(
        points, times, domain=DOMAIN, period=PERIOD, grid=(12, 12, 8)
    )
    print(
        f"LGCP: converged={lgcp.converged} in {lgcp.n_iter} Newton/IRLS "
        "iterations"
    )

    slice_times = tuple(f * PERIOD[1] for f in SLICE_FRACS)
    # intensity_st_kde was called with grid=(Gx=20, Gy=20, Gt=12); the
    # returned grid_x/intensity columns are flattened (y slow, x fast), so
    # the reshape target for a spatial slice is (Gy, Gx).
    Gy_kde, Gx_kde = 20, 20

    recovery_rows = []
    for t_query in slice_times:
        true_c = _bump_center(t_query)[0]

        kde_idx = int(np.argmin(np.abs(kde.grid_t - t_query)))
        kde_vals = kde.intensity[kde_idx]
        kde_c = _weighted_centroid(kde_vals, kde.grid_x)

        mean, lo, hi = lgcp.rate_map(t_query, level=0.9)
        lgcp_c = _weighted_centroid(mean, lgcp.grid_x)

        recovery_rows.append(
            {
                "t": float(t_query),
                "true_center": true_c,
                "kde_center": kde_c,
                "kde_error_mm": float(np.linalg.norm(kde_c - true_c)),
                "lgcp_center": lgcp_c,
                "lgcp_error_mm": float(np.linalg.norm(lgcp_c - true_c)),
            }
        )

    print()
    print("Recovery table -- does the fitted wavefront track the true wavefront?")
    print(
        f"  {'t (s)':>7} | {'true (x,y)':>14} | {'KDE err (mm)':>13} | "
        f"{'LGCP err (mm)':>14}"
    )
    for row in recovery_rows:
        tc = row["true_center"]
        print(
            f"  {row['t']:7.2f} | ({tc[0]:5.2f},{tc[1]:5.2f}) | "
            f"{row['kde_error_mm']:13.3f} | {row['lgcp_error_mm']:14.3f}"
        )
    mean_kde_err = float(np.mean([r["kde_error_mm"] for r in recovery_rows]))
    mean_lgcp_err = float(np.mean([r["lgcp_error_mm"] for r in recovery_rows]))
    print(f"  mean KDE centroid error : {mean_kde_err:.3f} mm")
    print(f"  mean LGCP centroid error: {mean_lgcp_err:.3f} mm")

    # ---- Propagation-speed recovery ----
    # Fit a line through the recovered peak (centroid) location at every
    # native KDE/LGCP time-grid point (denser than the 3 SLICE_FRACS above
    # used only for the figure panels) and read off each estimator's
    # propagation speed, compared against the true, by-construction speed.
    kde_traj_c = np.array(
        [
            _weighted_centroid(kde.intensity[i], kde.grid_x)
            for i in range(len(kde.grid_t))
        ]
    )
    lgcp_traj_c = np.array(
        [
            _weighted_centroid(lgcp.rate_map(t, level=0.9)[0], lgcp.grid_x)
            for t in lgcp.grid_t
        ]
    )
    kde_speed, kde_vel, _ = _fit_propagation_speed(kde.grid_t, kde_traj_c)
    lgcp_speed, lgcp_vel, _ = _fit_propagation_speed(lgcp.grid_t, lgcp_traj_c)
    # The LGCP posterior mean is a *smoothed* (Bayesian-shrunk) rate map --
    # its Matern prior systematically attenuates a sharp, briefly-visited
    # rate peak's estimated displacement more than the KDE's local,
    # non-parametric kernel estimate does, so the LGCP speed recovery gets
    # a wider (but still discriminating) tolerance.
    KDE_SPEED_REL_TOL = 0.25
    LGCP_SPEED_REL_TOL = 0.40
    kde_speed_rel_err = abs(kde_speed - TRUE_SPEED_MM_S) / TRUE_SPEED_MM_S
    lgcp_speed_rel_err = abs(lgcp_speed - TRUE_SPEED_MM_S) / TRUE_SPEED_MM_S
    kde_speed_ok = kde_speed_rel_err < KDE_SPEED_REL_TOL
    lgcp_speed_ok = lgcp_speed_rel_err < LGCP_SPEED_REL_TOL

    print()
    print(
        "Propagation-speed recovery -- line fit through the recovered peak "
        "trajectory vs. the true wavefront speed"
    )
    print(f"  true wavefront speed : {TRUE_SPEED_MM_S:.3f} mm/s (by construction)")
    print(
        f"  KDE-recovered speed  : {kde_speed:.3f} mm/s  "
        f"(rel. err {kde_speed_rel_err:.3f}, tol {KDE_SPEED_REL_TOL})  "
        f"{'PASS' if kde_speed_ok else 'FAIL'}"
    )
    print(
        f"  LGCP-recovered speed : {lgcp_speed:.3f} mm/s  "
        f"(rel. err {lgcp_speed_rel_err:.3f}, tol {LGCP_SPEED_REL_TOL})  "
        f"{'PASS' if lgcp_speed_ok else 'FAIL'}"
    )

    # ---- Figures ----
    electrodes = _electrode_positions()

    # === FIGURE: fig01_array_scatter.png ===
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.6))
    ax1.scatter(
        electrodes[:, 0], electrodes[:, 1],
        marker="s", s=10, color="0.6", zorder=1, label="electrode",
    )
    sc = ax1.scatter(
        points[:, 0], points[:, 1], c=times, s=8, cmap="viridis",
        alpha=0.8, zorder=2, label="event",
    )
    fig1.colorbar(sc, ax=ax1, label="time (s)")
    ax1.set_xlim(*DOMAIN[0])
    ax1.set_ylim(*DOMAIN[1])
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_title(
        f"Utah-style {N_SIDE}x{N_SIDE} MEA -- {n} synthetic events\n"
        "(spreading-depolarisation / seizure wavefront)",
        fontsize=10,
    )
    ax1.legend(loc="upper left", fontsize=8)
    # === END FIGURE ===

    # === FIGURE: fig02_kde_snapshots.png ===
    # Top row = the GROUND-TRUTH wavefront intensity field at each slice;
    # bottom row = the space-time KDE estimate the model DERIVES from the
    # observed events alone.  The white star marks the true wavefront centre
    # in both rows, so the reader can check that the estimated hot-spot
    # tracks the truth as the wavefront sweeps across the array.
    tf_x = np.linspace(DOMAIN[0][0], DOMAIN[0][1], 60)
    tf_y = np.linspace(DOMAIN[1][0], DOMAIN[1][1], 60)
    tf_gx, tf_gy = np.meshgrid(tf_x, tf_y)
    tf_pts = np.column_stack([tf_gx.ravel(), tf_gy.ravel()])
    fig2, axes2 = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for col, (t_query, row) in enumerate(zip(slice_times, recovery_rows)):
        tc = row["true_center"]

        ax_true = axes2[0, col]
        true_field = _true_intensity(tf_pts, t_query).reshape(tf_gx.shape)
        im_t = ax_true.imshow(
            true_field, origin="lower", cmap="viridis",
            extent=(*DOMAIN[0], *DOMAIN[1]), aspect="equal",
        )
        ax_true.plot(tc[0], tc[1], marker="*", color="white", ms=14, mec="black")
        ax_true.set_title(f"true lambda(x, t={t_query:.2f}s)")
        ax_true.set_xlabel("x (mm)")
        ax_true.set_ylabel("y (mm)")
        fig2.colorbar(im_t, ax=ax_true, fraction=0.046, pad=0.04)

        ax_kde = axes2[1, col]
        idx = int(np.argmin(np.abs(kde.grid_t - t_query)))
        img = kde.intensity[idx].reshape(Gy_kde, Gx_kde)
        im_k = ax_kde.imshow(
            img, origin="lower", cmap="viridis", extent=(*DOMAIN[0], *DOMAIN[1]),
            aspect="equal",
        )
        ax_kde.plot(tc[0], tc[1], marker="*", color="white", ms=14, mec="black")
        ax_kde.set_title(f"KDE lambda_hat(x, t={t_query:.2f}s)")
        ax_kde.set_xlabel("x (mm)")
        ax_kde.set_ylabel("y (mm)")
        fig2.colorbar(im_k, ax=ax_kde, fraction=0.046, pad=0.04)
    fig2.suptitle(
        "Ground-truth wavefront intensity (top) vs the space-time KDE "
        "estimate the model derives (bottom); white star = true wavefront centre"
    )
    # === END FIGURE ===

    # === FIGURE: fig03_lgcp_snapshots.png ===
    # Rows 1-2 (unchanged): LGCP posterior mean + credible-band snapshots.
    # Row 3 (new): the true-vs-recovered wavefront trajectory in the (x, y)
    # plane, spanning the full trial -- not just the 3 snapshot times above
    # -- with the recovered-vs-true propagation-speed readout annotated.
    Gy_l, Gx_l = 12, 12
    fig3 = plt.figure(figsize=(13.5, 11.2))
    gs3 = fig3.add_gridspec(
        3, 3, height_ratios=(1.0, 1.0, 1.3), hspace=0.4, top=0.90, bottom=0.05
    )
    for col, (t_query, row) in enumerate(zip(slice_times, recovery_rows)):
        mean, lo, hi = lgcp.rate_map(t_query, level=0.9)
        band_width = hi - lo
        tc = row["true_center"]

        ax_mean = fig3.add_subplot(gs3[0, col])
        im0 = ax_mean.imshow(
            mean.reshape(Gy_l, Gx_l), origin="lower", cmap="viridis",
            extent=(*DOMAIN[0], *DOMAIN[1]), aspect="equal",
        )
        ax_mean.plot(tc[0], tc[1], marker="*", color="white", ms=14, mec="black")
        ax_mean.plot(
            row["lgcp_center"][0], row["lgcp_center"][1],
            marker="x", color="red", ms=10, mew=2,
        )
        ax_mean.set_title(f"posterior mean, t={t_query:.2f}s")
        ax_mean.set_xlabel("x (mm)")
        ax_mean.set_ylabel("y (mm)")
        fig3.colorbar(im0, ax=ax_mean, fraction=0.046, pad=0.04)

        ax_band = fig3.add_subplot(gs3[1, col])
        im1 = ax_band.imshow(
            band_width.reshape(Gy_l, Gx_l), origin="lower", cmap="magma",
            extent=(*DOMAIN[0], *DOMAIN[1]), aspect="equal",
        )
        ax_band.set_title("90% credible band width")
        ax_band.set_xlabel("x (mm)")
        ax_band.set_ylabel("y (mm)")
        fig3.colorbar(im1, ax=ax_band, fraction=0.046, pad=0.04)

    ax_traj = fig3.add_subplot(gs3[2, :])
    t_fine = np.linspace(PERIOD[0], PERIOD[1], 200)
    true_fine = _bump_center(t_fine)
    ax_traj.plot(
        true_fine[:, 0], true_fine[:, 1], color="0.35", lw=2.5, zorder=1,
        label="true wavefront trajectory",
    )
    ax_traj.plot(
        kde_traj_c[:, 0], kde_traj_c[:, 1], marker="o", ms=5, lw=1.0,
        ls="--", color="tab:blue", zorder=2, label="KDE recovered peak",
    )
    ax_traj.plot(
        lgcp_traj_c[:, 0], lgcp_traj_c[:, 1], marker="x", ms=7, mew=1.5,
        lw=1.0, ls="--", color="red", zorder=2, label="LGCP recovered peak",
    )
    ax_traj.set_xlim(*DOMAIN[0])
    ax_traj.set_ylim(*DOMAIN[1])
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x (mm)")
    ax_traj.set_ylabel("y (mm)")
    ax_traj.set_title(
        "True vs. recovered wavefront trajectory + propagation-speed readout"
    )
    ax_traj.legend(loc="upper left", fontsize=8)
    speed_text = (
        f"true speed: {TRUE_SPEED_MM_S:.2f} mm/s\n"
        f"KDE recovered: {kde_speed:.2f} mm/s "
        f"({kde_speed_rel_err * 100:.0f}% rel. err, "
        f"{'PASS' if kde_speed_ok else 'FAIL'})\n"
        f"LGCP recovered: {lgcp_speed:.2f} mm/s "
        f"({lgcp_speed_rel_err * 100:.0f}% rel. err, "
        f"{'PASS' if lgcp_speed_ok else 'FAIL'})"
    )
    ax_traj.text(
        0.98, 0.03, speed_text, transform=ax_traj.transAxes,
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9),
    )

    fig3.suptitle(
        "Spatiotemporal LGCP posterior mean + credible band (white star ="
        " truth, red x = recovered centroid)\nbottom: true-vs-recovered"
        " wavefront trajectory + propagation speed",
        fontsize=11,
    )
    # === END FIGURE ===

    figures = [fig1, fig2, fig3]
    fig_names = ("fig01_array_scatter", "fig02_kde_snapshots", "fig03_lgcp_snapshots")
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras"
                / "spatial_stlgcp_microelectrode"
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
        "points": points,
        "times": times,
        "kde": kde,
        "lgcp": lgcp,
        "recovery": recovery_rows,
        "mean_kde_error_mm": mean_kde_err,
        "mean_lgcp_error_mm": mean_lgcp_err,
        "true_speed_mm_s": TRUE_SPEED_MM_S,
        "kde_speed_mm_s": kde_speed,
        "lgcp_speed_mm_s": lgcp_speed,
        "kde_speed_rel_err": kde_speed_rel_err,
        "lgcp_speed_rel_err": lgcp_speed_rel_err,
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Space-time KDE + LGCP spreading-depolarisation / "
                    "seizure-wavefront tracking demo on a synthetic "
                    "Utah-style microelectrode array",
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
        summary = {
            "n_events": int(result["points"].shape[0]),
            "mean_kde_error_mm": result["mean_kde_error_mm"],
            "mean_lgcp_error_mm": result["mean_lgcp_error_mm"],
            "true_speed_mm_s": result["true_speed_mm_s"],
            "kde_speed_mm_s": result["kde_speed_mm_s"],
            "lgcp_speed_mm_s": result["lgcp_speed_mm_s"],
            "kde_speed_rel_err": result["kde_speed_rel_err"],
            "lgcp_speed_rel_err": result["lgcp_speed_rel_err"],
            "lgcp_converged": bool(result["lgcp"].converged),
            "figure_paths": result["figure_paths"],
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
