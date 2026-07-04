#!/usr/bin/env python3
"""Demo: is ictal propagation a genuine traveling wave, or a static hot-spot?

End-to-end exercise of the space-time inhomogeneous second-order
goodness-of-fit machinery shipped in :mod:`nstat.extras.spatial`, grounded
in a realistic **ECoG (electrocorticography) surface-grid** recording
geometry:

**Clinical question.**  Intracranial recordings of human seizures show
discharge activity concentrated over part of an electrode grid -- but
that spatial footprint alone does not say *why*.  It could be a
genuinely **propagating ictal wavefront** that recruits cortex ahead of
it as it migrates (Smith et al. 2016; Martinet et al. 2017; Smith et al.
2022), or it could be a **static hot-spot**: a chronically hyperexcitable
patch of cortex whose sites fire independently of each other, with no
true space-time coupling, more consistent with a spatially inhomogeneous
but temporally static source (Diamond et al. 2021 similarly argue for
re-examining what "the seizure source" looks like from raw ictal
activity).  A naive per-electrode firing-rate map cannot tell these two
apart -- both produce a spatially concentrated pattern.  They differ only
in their **second-order** space-time structure: independent per-site
firing (however spatially inhomogeneous) is consistent with an
inhomogeneous-Poisson null, while a genuine travelling wavefront induces
excess space-time clustering along its trajectory that the same Poisson
null cannot explain.

**Scenario.**  An **8x8 ECoG grid** (1cm electrode pitch, the standard
subdural-strip/grid spacing) records two ~1s test epochs, both compared
against the same held-out inhomogeneous-Poisson background:

- a **static hot-spot** epoch, where events really are an inhomogeneous
  Poisson process with a fixed, temporally-static spatial rate bump (no
  space-time clustering beyond the known rate inhomogeneity) --
  chronically hyperexcitable tissue near a seizure focus, but no
  propagation;
- a **traveling wave** epoch, drawn (again by pure thinning -- no
  self-exciting cascade) from a Gaussian intensity bump whose centre
  **translates across the grid at a fixed velocity** over the test
  window -- the second-order signature of a genuinely propagating ictal
  wavefront.

The question the demo answers: **does the second-order space-time
structure of each epoch look like the inhomogeneous-Poisson null implied
by the fitted background rate, or does it reject that null?**  All
spikes are fully synthetic (Lewis-Shedler thinning) -- no real recording
or dataset is used or claimed.

Demonstrates:

1. :func:`nstat.extras.spatial.intensity_st_kde` -- fit a **held-out**
   space-time intensity lambda_hat from a separate "characterization"
   recording (the plug-in-bias caveat documented in the module: reusing
   the *same* pattern's own KDE fit deflates the test's variance).
2. :func:`nstat.extras.spatial.k_st_inhom` /
   :func:`nstat.extras.spatial.pair_correlation_st` -- the SOIRS-reweighted
   space-time K-function / pair correlation, using that held-out
   lambda_hat.
3. :func:`nstat.extras.spatial.global_envelope_st` -- a Monte-Carlo
   global-rank envelope test: the static-hot-spot epoch should stay
   INSIDE the envelope; the traveling-wave epoch should reject (fall
   OUTSIDE).

The script is **fully synthetic** -- no figshare dataset access required.

Run::

    python examples/extras/spatial_gof_ecog_demo.py            # interactive
    python examples/extras/spatial_gof_ecog_demo.py --no-display
    python examples/extras/spatial_gof_ecog_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/spatial_gof_ecog/``) and are NOT committed to the
repository -- the export flag exists for local inspection only.  CI
never invokes it.

See also :mod:`spatial_stlgcp_microelectrode_demo` -- once a pattern is
known to reject the Poisson null as a travelling wave, that demo tracks
the *same kind* of migrating Gaussian bump as a moving intensity surface
(spatiotemporal LGCP) and recovers its trajectory/speed; this demo only
answers "is it a wave at all?", not "what is the wave's trajectory?".

References:

Clinical motivation (human ictal/interictal travelling waves):

- Smith EH, Liou JY, Davis TS, Merricks EM, Kellis SS, Weiss SA, Greger B,
  House PA, McKhann GM, Goodman RR, Emerson RG, Bateman LM, Trevelyan AJ,
  Schevon CA (2016). *The ictal wavefront is the spatiotemporal source of
  discharges during spontaneous human seizures.* Nat Commun 7:11098.
- Martinet LE, Fiddyment G, Madsen JR, Eskandar EN, Truccolo W, Eden UT,
  Cash SS, Kramer MA (2017). *Human seizures couple across spatial scales
  through travelling wave dynamics.* Nat Commun 8:14896.
- Diamond JM, Diamond BE, Trotta MS, Dembny K, Inati SK, Zaghloul KA
  (2021). *Travelling waves reveal a dynamic seizure source in human
  focal epilepsy.* Brain 144(6):1751-1763.
- Smith EH, Liou JY, Merricks EM, Davis T, Thomson K, Greger B, House P,
  Emerson RG, Goodman R, McKhann GM, Sheth S, Schevon C, Rolston JD
  (2022). *Human interictal epileptiform discharges are bidirectional
  traveling waves echoing ictal discharges.* eLife 11:e73541.

Statistical methods:

- Diggle PJ, Chetwynd AG, Haggkvist R, Morris SE (1995). *Second-order
  analysis of space-time clustering.* J. R. Statist. Soc. C 44(1):71-86.
- Gabriel E, Diggle PJ (2009). *Second-order analysis of inhomogeneous
  spatio-temporal point process data.* Statistica Neerlandica 63(1):43-51.
- Moller J, Ghorbani M (2012). *Aspects of second-order analysis of
  structured inhomogeneous spatio-temporal point processes.* Statistica
  Neerlandica 66(4):472-491.
- Myllymaki M, Mrkvicka T, Grabarnik P, Seijo H, Hahn U (2017). *Global
  envelope tests for spatial processes.* J. R. Statist. Soc. B
  79(2):381-404.
- Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal
  Point Patterns* (3rd ed.). CRC Press, Chapter 7 (lambda_hat KDE).
- Miscouridou X, Bhatt S, Mohler G, Flaxman S, Bhamidi S (2022).
  *Cox-Hawkes: doubly stochastic spatiotemporal Poisson processes.*
  TMLR (source of :func:`nstat.extras.spatial.simulate_cox_hawkes`; used
  here with ``K_branch=0`` purely as a convenient thinning-based
  inhomogeneous-Poisson event generator with a time-varying background --
  the demo's subject is goodness-of-fit, not Cox-Hawkes cascade
  estimation -- see ``spatial_hawkes_ecog_demo.py`` for that).
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
# 8x8 ECoG grid geometry (1cm pitch) + inhomogeneous background
# ---------------------------------------------------------------------------

N_SIDE = 8
PITCH_CM = 1.0
MARGIN_CM = 0.5
GRID_EDGE_CM = 2.0 * MARGIN_CM + PITCH_CM * (N_SIDE - 1)  # == 8.0 cm

DOMAIN = ((0.0, GRID_EDGE_CM), (0.0, GRID_EDGE_CM))

FOCUS_CENTER = np.array([5.5, 2.5])  # cm -- e.g. a chronically-active focus
FOCUS_SIGMA = 1.3
BASELINE_RATE = 1.2  # Hz/cm^2
AMP_RATE = 2.5  # Hz/cm^2

T_CALIB = 3.0
T_TEST = 1.0

# -- Traveling-wave epoch: a Gaussian intensity bump migrating at fixed
# velocity across the grid over [0, T_TEST], the second-order signature
# of a genuinely propagating ictal wavefront (Smith 2016; Martinet 2017).
# The total excursion (~3.6cm) is deliberately modest relative to the
# grid (8cm) and comparable to the static hot-spot's own footprint
# (FOCUS_SIGMA=1.3) -- a full corner-to-corner sweep would smear the
# wave's time-averaged rate into an elongated band that a naive
# per-electrode rate map *could* trivially tell apart from the
# hot-spot's compact blob, defeating the point of the second-order test.
WAVE_SIGMA = 1.0  # cm -- spatial half-width of the migrating wavefront
WAVE_PEAK_RATE = 4.0  # Hz/cm^2 -- peak excess rate at the wavefront core
WAVE_START = np.array([2.0, 5.5])  # cm -- wavefront centre at t=0
WAVE_VELOCITY = np.array([3.0, -2.0])  # cm/s -- fixed translation velocity
WAVE_BG_MAX = BASELINE_RATE + WAVE_PEAK_RATE  # exact peak: thinning bound

R_GRID = np.linspace(0.25, 1.75, 6)
T_GRID = np.linspace(0.05, 0.35, 5)
N_SIM = 39
ALPHA = 0.1


def _electrode_positions() -> np.ndarray:
    coords = MARGIN_CM + PITCH_CM * np.arange(N_SIDE)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _background_rate(x: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Time-invariant, spatially inhomogeneous ECoG background (Hz/cm^2).

    The **static hot-spot** ground truth: a fixed rate bump near a
    chronically hyperexcitable focus.  Spatially inhomogeneous, but *not*
    a function of ``t`` -- no propagation, by construction.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    d2 = np.sum((x - FOCUS_CENTER[None, :]) ** 2, axis=1)
    return BASELINE_RATE + AMP_RATE * np.exp(-d2 / (2.0 * FOCUS_SIGMA**2))


def _wave_rate(x: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Migrating Gaussian wavefront intensity (Hz/cm^2).

    The **traveling wave** ground truth: same functional form as
    :func:`_background_rate` (a Gaussian bump on top of the flat
    ``BASELINE_RATE``), except the bump's centre translates linearly in
    ``t`` at ``WAVE_VELOCITY`` -- a genuinely propagating source rather
    than a static hot-spot.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    t = np.broadcast_to(np.asarray(t, dtype=float), (x.shape[0],))
    center = WAVE_START[None, :] + WAVE_VELOCITY[None, :] * t[:, None]
    d2 = np.sum((x - center) ** 2, axis=1)
    return BASELINE_RATE + WAVE_PEAK_RATE * np.exp(-d2 / (2.0 * WAVE_SIGMA**2))


def _traveling_wave_events(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the traveling-wave test epoch by thinning against its own max.

    Reuses :func:`nstat.extras.spatial.simulate_cox_hawkes` with
    ``K_branch=0`` (no Hawkes cascade -- see the module References) as a
    time-varying-background Lewis-Shedler thinning generator: the
    dominating proposal rate ``bg_max=WAVE_BG_MAX`` is the *exact*
    analytic peak of :func:`_wave_rate` (attained at the moving centre),
    not merely a probed estimate, so this is thinning against the
    wavefront's own true max intensity.
    """
    from nstat.extras.spatial import simulate_cox_hawkes

    pts, times = simulate_cox_hawkes(
        _wave_rate, 0.0, 1.0, 1.0,
        domain=DOMAIN, T=T_TEST, rng=rng, bg_max=WAVE_BG_MAX,
    )
    return _clip_to_domain(pts, times)


def _clip_to_domain(
    points: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only events that fall on the observed ECoG grid footprint.

    ``simulate_cox_hawkes``'s offspring offsets are not clipped to the
    background window (matching ``simulate_spatial_hawkes``'s convention),
    but an ECoG grid can only ever record activity from *under* the grid
    -- so events that drift outside the array footprint are dropped here,
    exactly as a real recording would simply not observe them.
    """
    (xlo, xhi), (ylo, yhi) = DOMAIN
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
    """Run the ECoG traveling-wave-vs-static-hot-spot goodness-of-fit demo.

    Returns
    -------
    dict
        ``{"hotspot_inside": ..., "wave_inside": ..., "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style
    from nstat.extras.spatial import (
        global_envelope_st,
        intensity_st_kde,
        k_st_inhom,
        pair_correlation_st,
        simulate_cox_hawkes,
    )

    print("=" * 72)
    print("Traveling wave vs static hot-spot GOF on a synthetic 8x8 ECoG grid")
    print("=" * 72)
    print(
        f"Grid footprint: {GRID_EDGE_CM:.1f}cm x {GRID_EDGE_CM:.1f}cm, "
        f"{N_SIDE}x{N_SIDE} electrodes, {PITCH_CM:.0f}cm pitch "
        "(fully synthetic events -- no real recording)"
    )

    rng = np.random.default_rng(seed)

    # ---- Held-out characterization ("calibration") epoch: pure background. ----
    calib_pts, calib_t = simulate_cox_hawkes(
        _background_rate, 0.0, 1.0, 1.0,
        domain=DOMAIN, T=T_CALIB, rng=rng,
    )
    calib_pts, calib_t = _clip_to_domain(calib_pts, calib_t)
    kde_calib = intensity_st_kde(
        calib_pts, calib_t, domain=DOMAIN, period=(0.0, T_CALIB), grid=(16, 16, 8)
    )
    print(f"Calibration epoch: {calib_pts.shape[0]} events over {T_CALIB:.1f}s")

    # ---- Static hot-spot TEST epoch: fresh draw, inhomogeneous Poisson (null true). ----
    hotspot_pts, hotspot_t = simulate_cox_hawkes(
        _background_rate, 0.0, 1.0, 1.0,
        domain=DOMAIN, T=T_TEST, rng=rng,
    )
    hotspot_pts, hotspot_t = _clip_to_domain(hotspot_pts, hotspot_t)

    # ---- Traveling-wave TEST epoch: migrating Gaussian bump, pure thinning. ----
    wave_pts, wave_t = _traveling_wave_events(rng)

    print(f"Static hot-spot epoch: {hotspot_pts.shape[0]} events over {T_TEST:.1f}s")
    print(f"Traveling-wave epoch:  {wave_pts.shape[0]} events over {T_TEST:.1f}s")

    def _analyze(points, times, label):
        env = global_envelope_st(
            points, times, kde_calib.evaluate, R_GRID, T_GRID,
            n_sim=N_SIM, statistic="kst", alpha=ALPHA,
            domain=DOMAIN, period=(0.0, T_TEST), rng=rng,
        )
        k_obs = k_st_inhom(
            points, times, kde_calib.evaluate, R_GRID, T_GRID,
            domain=DOMAIN, period=(0.0, T_TEST),
        )
        g_obs = pair_correlation_st(
            points, times, kde_calib.evaluate, R_GRID, T_GRID,
            domain=DOMAIN, period=(0.0, T_TEST),
        )
        return {
            "label": label,
            "points": points,
            "times": times,
            "env": env,
            "k_st": k_obs.k_st,
            "g_st": g_obs,
        }

    baseline = _analyze(hotspot_pts, hotspot_t, "static hot-spot (Poisson null)")
    wave = _analyze(wave_pts, wave_t, "traveling wave (propagating wavefront)")

    print()
    print("Global-rank envelope verdict (K_st statistic, alpha=%.2f):" % ALPHA)
    for res in (baseline, wave):
        env = res["env"]
        verdict = "INSIDE (fails to reject null)" if env.inside else "OUTSIDE (rejects null)"
        print(
            f"  {res['label']:38s}: {verdict}  "
            f"p_interval=({env.p_interval[0]:.3f}, {env.p_interval[1]:.3f})"
        )

    # ---- Figures ----
    electrodes = _electrode_positions()
    t_idx = -1  # largest temporal lag: clustering signal is most visible here

    # === FIGURE: fig01_kst_envelope.png ===
    # Top row = the GROUND-TRUTH event patterns each epoch actually produced,
    # overlaid on the true rate field.  The static-hot-spot column shows a
    # time-invariant bump; the traveling-wave column shows the rate field at
    # t=0 plus the wavefront's straight-line path to t=T_TEST, with events
    # coloured by time so the temporal sweep is visible.  Bottom row = the
    # space-time K-function the MODEL DERIVES from each pattern, vs the
    # Monte-Carlo global-rank envelope of the fitted-background null.
    # Reading top-to-bottom shows *why* the test fires: the migrating
    # wavefront's space-time-correlated events push its K_st far outside the
    # envelope, while the static hot-spot -- however spatially concentrated
    # -- stays inside it.
    fig1, axes1 = plt.subplots(2, 2, figsize=(11.5, 9.0))
    gx = np.linspace(DOMAIN[0][0], DOMAIN[0][1], 60)
    gy = np.linspace(DOMAIN[1][0], DOMAIN[1][1], 60)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    bg_field_hotspot = _background_rate(grid_xy, 0.0).reshape(grid_x.shape)
    bg_field_wave = _wave_rate(grid_xy, 0.0).reshape(grid_x.shape)
    wave_end = WAVE_START + WAVE_VELOCITY * T_TEST
    for col, (res, bg_field) in enumerate(
        ((baseline, bg_field_hotspot), (wave, bg_field_wave))
    ):
        ax = axes1[0, col]
        pcm = ax.pcolormesh(grid_x, grid_y, bg_field, cmap="Greys",
                            shading="auto", alpha=0.85)
        ax.scatter(electrodes[:, 0], electrodes[:, 1], marker="s", s=9,
                   facecolors="none", edgecolors="0.4", linewidths=0.5)
        if res is wave:
            ax.annotate(
                "", xy=wave_end, xytext=WAVE_START,
                arrowprops=dict(arrowstyle="-|>", color="tab:blue",
                                lw=1.5, ls="--"),
                zorder=2,
            )
            ax.plot(*WAVE_START, marker="o", mfc="none", mec="tab:blue",
                    ms=8, zorder=2, label="wavefront centre, t=0")
            ax.plot(*wave_end, marker="o", mfc="tab:blue", mec="tab:blue",
                    ms=8, zorder=2, label=f"wavefront centre, t={T_TEST:.1f}s")
            ax.legend(loc="upper left", fontsize=6.5)
        pts = res["points"]
        ax.scatter(pts[:, 0], pts[:, 1], c=res["times"], cmap="autumn", s=18,
                   edgecolors="k", linewidths=0.2, zorder=3)
        ax.set_title(f"ground truth: {res['label']}\n({pts.shape[0]} events)")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_aspect("equal")
        cbar_label = (
            "true rate at t=0 (Hz/cm^2)" if res is wave else "true bg rate (Hz/cm^2)"
        )
        fig1.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    k_ymax = max(
        float(wave["env"].observed[:, t_idx].max()),
        float(wave["env"].hi[:, t_idx].max()),
    )
    for col, res in enumerate((baseline, wave)):
        ax = axes1[1, col]
        env = res["env"]
        ax.fill_between(
            R_GRID, env.lo[:, t_idx], env.hi[:, t_idx],
            color="tab:blue", alpha=0.25, label="MC envelope (null)",
        )
        ax.plot(R_GRID, env.observed[:, t_idx], color="tab:red", lw=1.8,
                marker="o", ms=4, label="observed K_st")
        verdict = "INSIDE" if env.inside else "OUTSIDE (reject)"
        ax.set_title(f"model: K_st test, t={T_GRID[t_idx]:.2f}s -- {verdict}")
        ax.set_xlabel("spatial lag r (cm)")
        ax.set_ylabel("K_st(r, t)")
        ax.set_ylim(0.0, k_ymax * 1.05)
        ax.legend(loc="upper left", fontsize=8)
    fig1.suptitle(
        "Ground-truth event patterns (top) vs the space-time K-function the "
        "model derives and its global-rank envelope test (bottom)"
    )
    # === END FIGURE ===

    # === FIGURE: fig02_pcf_g.png ===
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.8))
    ax2.axhline(1.0, color="gray", lw=1.0, ls=":", label="Poisson null (g=1)")
    ax2.plot(R_GRID, baseline["g_st"][:, t_idx], color="tab:green", lw=1.8,
              marker="o", ms=4, label=baseline["label"])
    ax2.plot(R_GRID, wave["g_st"][:, t_idx], color="tab:red", lw=1.8,
              marker="s", ms=4, label=wave["label"])
    ax2.set_xlabel("spatial lag r (cm)")
    ax2.set_ylabel(f"g(r, t={T_GRID[t_idx]:.2f}s)")
    ax2.set_title("Space-time pair correlation")
    ax2.legend(loc="upper right", fontsize=8)
    # === END FIGURE ===

    # === FIGURE: fig03_verdict.png ===
    fig3, ax3 = plt.subplots(figsize=(7.5, 3.0))
    ax3.axis("off")
    lines = ["epoch                                  verdict                    p_interval",
             "-" * 82]
    for res in (baseline, wave):
        env = res["env"]
        verdict = "INSIDE (fails to reject)" if env.inside else "OUTSIDE (rejects null)"
        lines.append(
            f"{res['label']:38s} {verdict:26s} "
            f"({env.p_interval[0]:.3f}, {env.p_interval[1]:.3f})"
        )
    ax3.text(
        0.02, 0.85, "\n".join(lines), family="monospace", fontsize=10,
        va="top", transform=ax3.transAxes,
    )
    ax3.set_title("Global-rank envelope verdict summary")
    # === END FIGURE ===

    figures = [fig1, fig2, fig3]
    fig_names = ("fig01_kst_envelope", "fig02_pcf_g", "fig03_verdict")
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras" / "spatial_gof_ecog"
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
        "hotspot_inside": bool(baseline["env"].inside),
        "wave_inside": bool(wave["env"].inside),
        "hotspot_p_interval": baseline["env"].p_interval,
        "wave_p_interval": wave["env"].p_interval,
        "n_hotspot": int(hotspot_pts.shape[0]),
        "n_wave": int(wave_pts.shape[0]),
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Traveling-wave vs static-hot-spot GOF demo on a "
                    "synthetic ECoG grid",
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
        help="Write a compact verdict summary as JSON.",
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
            "hotspot_inside": result["hotspot_inside"],
            "wave_inside": result["wave_inside"],
            "hotspot_p_interval": list(result["hotspot_p_interval"]),
            "wave_p_interval": list(result["wave_p_interval"]),
            "n_hotspot": result["n_hotspot"],
            "n_wave": result["n_wave"],
            "figure_paths": result["figure_paths"],
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
