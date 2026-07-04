#!/usr/bin/env python3
"""Demo: time-varying firing-rate field on a Utah-style microelectrode array.

End-to-end exercise of the two spatiotemporal rate-estimation modules
shipped in :mod:`nstat.extras.spatial` (space-time KDE + spatiotemporal
LGCP), grounded in a realistic microelectrode-array recording geometry:

**Scenario.**  A **Utah-style 10x10 microelectrode array** (4mm x 4mm
footprint, 400um electrode pitch, 200um edge margin -- the standard
Blackrock Utah-array layout) records population multiunit activity during
a single ~1.5s reach-like trial.  The population's space-time firing-rate
field is a Gaussian "bump" of elevated activity that **translates across
the array over the trial** -- the textbook signature of a moving
population-activity locus, e.g. a travelling representation of reach
direction in motor cortex (Georgopoulos-style population-vector rotation)
or a propagating sensory-evoked response.  All spikes are drawn from a
known, fully synthetic space-time Poisson intensity (Lewis-Shedler
thinning) -- **no real recording or dataset is used or claimed**.

Demonstrates:

1. :func:`nstat.extras.spatial.intensity_st_kde` -- a boundary-corrected
   space-time kernel estimate lambda_hat(x, t) of the moving bump, read
   off at several time slices.
2. :func:`nstat.extras.spatial.lgcp_st_fit` -- a Kronecker-Laplace
   spatiotemporal log-Gaussian Cox process fit, giving a posterior
   *mean* rate map **with credible bands** (:meth:`LGCPSTResult.rate_map`)
   at the same time slices.
3. A recovery table comparing the true, KDE-estimated, and LGCP-estimated
   bump centroid (mm) at each slice -- "does the fitted bump track the
   true bump?".

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

References:
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
- Georgopoulos AP, Schwartz AB, Kettner RE (1986). *Neuronal population
  coding of movement direction.* Science 233(4771):1416-1419 (the
  moving-population-locus motivation for the synthetic scenario).
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
PERIOD = (0.0, 1.5)  # a ~1.5s reach-like trial

# True moving-bump parameters (population multiunit rate density, events
# per mm^2 per s -- a *population*-level density, not a single-unit Hz).
C0_TRUE = np.array([1.0, 1.0])
C1_TRUE = np.array([3.0, 3.0])
SIGMA_BUMP_TRUE = 0.5
BASELINE_TRUE = 8.0
AMP_TRUE = 50.0

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
    """True space-time Poisson intensity of the moving bump."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    t_arr = np.broadcast_to(np.asarray(t, dtype=float).reshape(-1), (x.shape[0],))
    center = _bump_center(t_arr)
    d2 = np.sum((x - center) ** 2, axis=1)
    return BASELINE_TRUE + AMP_TRUE * np.exp(-d2 / (2.0 * SIGMA_BUMP_TRUE**2))


def _simulate_moving_bump(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Lewis-Shedler thinning simulation of the moving-bump intensity."""
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
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(2, np.nan)
    return (values[:, None] * grid_x).sum(axis=0) / total


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
    """Run the microelectrode-array space-time KDE + LGCP demo.

    Returns
    -------
    dict
        ``{"points", "times", "kde", "lgcp", "recovery", "figure_paths"}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style
    from nstat.extras.spatial import intensity_st_kde, lgcp_st_fit

    print("=" * 72)
    print("Space-time LGCP on a Utah-style 10x10 microelectrode array")
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
    print("Recovery table -- does the fitted bump track the true bump?")
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
        "(moving-bump intensity)",
        fontsize=10,
    )
    ax1.legend(loc="upper left", fontsize=8)
    # === END FIGURE ===

    # === FIGURE: fig02_kde_snapshots.png ===
    fig2, axes2 = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, t_query, row in zip(axes2, slice_times, recovery_rows):
        idx = int(np.argmin(np.abs(kde.grid_t - t_query)))
        img = kde.intensity[idx].reshape(Gy_kde, Gx_kde)
        im = ax.imshow(
            img, origin="lower", cmap="viridis", extent=(*DOMAIN[0], *DOMAIN[1]),
            aspect="equal",
        )
        tc = row["true_center"]
        ax.plot(tc[0], tc[1], marker="*", color="white", ms=14, mec="black")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"KDE lambda_hat(x, t={t_query:.2f}s)")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.suptitle("Space-time KDE rate-field snapshots (white star = true bump centre)")
    # === END FIGURE ===

    # === FIGURE: fig03_lgcp_snapshots.png ===
    Gy_l, Gx_l = 12, 12
    fig3, axes3 = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for col, (t_query, row) in enumerate(zip(slice_times, recovery_rows)):
        mean, lo, hi = lgcp.rate_map(t_query, level=0.9)
        band_width = hi - lo
        tc = row["true_center"]

        ax_mean = axes3[0, col]
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

        ax_band = axes3[1, col]
        im1 = ax_band.imshow(
            band_width.reshape(Gy_l, Gx_l), origin="lower", cmap="magma",
            extent=(*DOMAIN[0], *DOMAIN[1]), aspect="equal",
        )
        ax_band.set_title("90% credible band width")
        ax_band.set_xlabel("x (mm)")
        ax_band.set_ylabel("y (mm)")
        fig3.colorbar(im1, ax=ax_band, fraction=0.046, pad=0.04)
    fig3.suptitle(
        "Spatiotemporal LGCP posterior mean + credible-band snapshots "
        "(white star = truth, red x = recovered centroid)"
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
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Space-time KDE + LGCP demo on a synthetic "
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
            "lgcp_converged": bool(result["lgcp"].converged),
            "figure_paths": result["figure_paths"],
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
