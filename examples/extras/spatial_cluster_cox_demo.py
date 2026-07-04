#!/usr/bin/env python3
"""Demo: hidden epileptogenic hubs vs. random scatter -- recovering Thomas /
Matérn-cluster structure in microdischarge patterns.

End-to-end exercise of the cluster-Cox catalogue and minimum-contrast
inference shipped in :mod:`nstat.extras.spatial` (Tier F sub-PR-1, #195),
grounded in the intracranial "microseizure"/multisite-recruitment literature:

**Clinical question.**  Dense, isolated microelectrode recordings reveal
brief, sparse, spatially-restricted "microseizures" that are invisible to
standard clinical macroelectrodes and are *more frequent in tissue that
goes on to generate clinical seizures* than in non-epileptogenic tissue
(Stead et al. 2010).  Multisite ictal recruitment likewise concentrates
into a handful of discrete, stereotyped "seizure hubs" rather than
spreading uniformly -- many patients thought pre-operatively to have a
single focus in fact show several noncontiguous hubs (Tobochnik et al.
2021).  This demo asks: **are observed microdischarges organised around a
few hidden epileptogenic hub sites, or randomly (uniformly) scattered
across the recording?  And if hubs exist, how many are there and how
tightly do discharges cluster around each?**  A naive count of "how many
electrodes/sites show activity" cannot answer this -- 5 tight hubs and 1
diffuse blob can light up the same number of contacts.

**Modeling correspondence.**  A Thomas process is exactly this hypothesis
made generative: :math:`K` latent, *never directly observed* parent
locations (the epileptogenic hub microdomains) are scattered as a
homogeneous Poisson process at intensity :math:`\\kappa`
(``intensity_parent``/``lambda_p`` in code); each parent independently
seeds :math:`\\mathrm{Poisson}(\\mu)` microdischarge offspring displaced by
an isotropic 2-D Gaussian of standard deviation :math:`\\sigma`
(``sigma``) -- a sub-mm-scale, sharply-decaying dispersion consistent with
Stead's isolated-microelectrode microseizure geometry.  Recovering
:math:`(\\sigma, \\kappa)` from *only* the offspring locations -- the hub
sites themselves are never in the data -- via minimum-contrast
second-order matching is the statistical analogue of asking a clinician to
infer "how many hidden epileptogenic zones, and how tight are they" from a
single sparse microelectrode snapshot.

**Scenario.**  A "few hidden hubs, tight clusters" regime: roughly
``K ~ kappa`` hub sites on the unit square, each seeding several
microdischarges within a tight, sub-mm-scale radius (Thomas process,
Gaussian falloff) -- deliberately *not* the diffuse many-cluster field the
pre-2026-07 version of this demo used, which read as generic point-pattern
clustering rather than "a few identifiable hubs".  A Matérn-cluster
catalogue repeats the same hidden-hub question with an **alternative
offspring geometry**: microdischarges scattered uniformly inside a
hard-edged disc of radius :math:`R` around each hub rather than fading off
smoothly -- demonstrating that minimum-contrast estimation recovers the
hub count and dispersion scale regardless of which offspring geometry
actually generated the data.

1. Simulate a **Thomas process** with isotropic Gaussian offspring
   displacement (Thomas 1949; Møller-Waagepetersen 2003 §5.3), drawn via
   the public generic Neyman-Scott constructor
   (:class:`~nstat.extras.spatial.NeymanScottCox` +
   :func:`~nstat.extras.spatial.simulate_neyman_scott`) configured to
   reproduce :func:`~nstat.extras.spatial.simulate_thomas`'s exact
   parent/offspring recipe -- ``simulate_thomas`` itself does not expose
   the latent hub locations, and this demo's ground-truth figure needs
   them for the true-parent overlay below.
2. Simulate a **Matérn cluster process** with uniform-disc offspring
   (Matérn 1986), via the same generic-constructor route.
3. Recover both parameter pairs ``(sigma, lambda_p)`` / ``(R, lambda_p)``
   -- i.e. :math:`(\\sigma, \\kappa)` -- from the simulated offspring
   patterns *alone* with :func:`nstat.extras.spatial.fit_thomas` and
   :func:`nstat.extras.spatial.fit_matern_cluster` -- Diggle's (2013
   §6.2.1) minimum-contrast estimator on the SOIRS pair correlation,
   which is a rescaling of Ripley's :math:`K`-function
   (:math:`g(r) = K'(r) / (2\\pi r)`) and shares its CSR reference value
   (:math:`g(r) \\equiv 1` under complete spatial randomness, no hubs).
4. Plot the four diagnostic figures -- true hub + offspring scatter,
   empirical-vs-fitted pair correlation with the CSR null overlaid, for
   each process -- and print a recovered-vs-true hub-count/dispersion
   table.

See also :mod:`spatial_gibbs_demo` -- the opposite second-order signature
(minimum-distance *repulsion* between co-active/implanted sites) rather
than the *clustering* story here; comparing the two PCFs side by side is
the fastest way to see how attraction (hubs) and repulsion (exclusion
zones) each leave a distinct mark on :math:`g(r)`.

The script is **fully synthetic** -- no figshare dataset access required.

Run::

    python examples/extras/spatial_cluster_cox_demo.py            # interactive
    python examples/extras/spatial_cluster_cox_demo.py --no-display
    python examples/extras/spatial_cluster_cox_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to ``docs/figures/extras/spatial_cluster_cox/``)
and are NOT committed to the repository — the export flag exists for
local inspection only.  CI never invokes it.

References:

Clinical motivation (hidden epileptogenic microdomains / seizure hubs):

- Stead M, Bower M, Brinkmann BH, Lee K, Marsh WR, Meyer FB, Litt B,
  Van Gompel J, Worrell GA (2010). *Microseizures and the spatiotemporal
  scales of human partial epilepsy.* Brain 133(9):2789-2797.
- Tobochnik S, Bateman LM, Akman CI, et al. (2021). *Tracking Multisite
  Seizure Propagation Using Ictal High-Gamma Activity.* J Clin
  Neurophysiol 39(7):592-601.

Statistical methods:

- Thomas M (1949). *A generalization of Poisson's binomial limit for use
  in ecology.* Biometrika 36(1/2):18.
- Matérn B (1986). *Spatial Variation* (2nd ed.). Springer LNS 36.
- Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal
  Point Patterns* (3rd ed.). CRC §6.2.1.
- Møller J, Waagepetersen RP (2003). *Statistical Inference and
  Simulation for Spatial Point Processes.* Chapman & Hall §5.3, §4.2.
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


# Window / domain conventions:
# - the cluster-Cox simulators take a flat 4-tuple (xmin, ymin, xmax, ymax).
# - pair_correlation / fit_* take ((xmin, xmax), (ymin, ymax)).
WINDOW = (0.0, 0.0, 1.0, 1.0)
DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def _gaussian_offspring_kernel(sigma: float):
    """Isotropic 2-D Gaussian offspring displacement kernel.

    Reproduces :func:`nstat.extras.spatial.cluster_cox._gaussian_offspring`
    exactly (same distribution, same ``rng`` call), so that generating the
    pattern through the public generic Neyman-Scott constructor is
    bit-for-bit identical to what ``simulate_thomas`` would draw for the
    same ``rng`` state.
    """
    def _kernel(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, sigma, size=(n, 2))
    return _kernel


def _uniform_disc_offspring_kernel(radius: float):
    """Uniform-disc offspring displacement kernel of radius ``radius``.

    Reproduces
    :func:`nstat.extras.spatial.cluster_cox._uniform_disc_offspring`
    exactly, for the same reason as :func:`_gaussian_offspring_kernel`
    above.
    """
    def _kernel(n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        r = radius * np.sqrt(u)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return _kernel


def _run_thomas(rng: np.random.Generator) -> dict:
    """Simulate + fit a Thomas process of hidden epileptogenic hubs.

    "Few hidden hubs, tight clusters" regime: ``kappa=8`` (few latent hub
    sites over the unit square), ``mu=10`` (several microdischarge
    offspring per hub), ``sigma=0.02`` (tight, sub-mm-scale dispersion per
    Stead 2010's isolated-microelectrode microseizure geometry) --
    replacing the earlier many-parent/diffuse-cluster regime, which did
    not read as "a few identifiable hidden hubs".
    """
    from nstat.extras.spatial import (
        NeymanScottCox,
        fit_thomas,
        simulate_neyman_scott,
        thomas_pair_correlation,
    )

    sigma_true = 0.02
    lambda_p_true = 8.0
    mu_offspring_true = 10.0

    # simulate_thomas() does not expose the latent parent (hub) locations
    # needed for the ground-truth overlay below, so the identical Thomas
    # generative recipe (homogeneous-Poisson parents on the window padded
    # by 3*sigma, Poisson(mu) Gaussian-displaced offspring) is reproduced
    # through the public generic Neyman-Scott constructor, which *does*
    # return the parents.
    process = NeymanScottCox(
        intensity_parent=lambda_p_true,
        mu_offspring=mu_offspring_true,
        offspring_kernel=_gaussian_offspring_kernel(sigma_true),
        pad=3.0 * sigma_true,
    )
    points, true_parents = simulate_neyman_scott(
        process, WINDOW, rng=rng, return_parents=True
    )

    r_grid = np.linspace(0.01, 0.25, 32)
    fit = fit_thomas(points, DOMAIN, r_grid)
    sigma_hat = float(fit.theta_hat[0])
    lambda_p_hat = float(fit.theta_hat[1])
    g_true = thomas_pair_correlation(
        r_grid, sigma_true, lambda_p_true, mu_offspring_true
    )
    return {
        "points": points,
        "true_parents": true_parents,
        "r_grid": r_grid,
        "g_fit": np.asarray(fit.g_model_at_theta, dtype=float),
        "g_true": np.asarray(g_true, dtype=float),
        "sigma_true": sigma_true,
        "lambda_p_true": lambda_p_true,
        "mu_offspring_true": mu_offspring_true,
        "sigma_hat": sigma_hat,
        "lambda_p_hat": lambda_p_hat,
        "objective_value": float(fit.objective_value),
        "success": bool(fit.success),
        "message": str(fit.message),
        "n_iter": int(fit.n_iter),
    }


def _run_matern(rng: np.random.Generator) -> dict:
    """Simulate + fit a Matérn cluster process -- hidden hubs, alternative
    (hard-edged uniform-disc) offspring geometry.

    Same "few hidden hubs" regime as :func:`_run_thomas`
    (``kappa=8``); the offspring disperse uniformly inside a disc of
    radius ``R=0.07`` instead of fading off as a Gaussian, so this
    catalogue demonstrates the same hub-count/dispersion recovery under a
    qualitatively different dispersion shape.
    """
    from nstat.extras.spatial import (
        NeymanScottCox,
        fit_matern_cluster,
        matern_cluster_pair_correlation,
        simulate_neyman_scott,
    )

    radius_true = 0.07
    lambda_p_true = 8.0
    mu_offspring_true = 9.0

    # simulate_matern_cluster() likewise does not expose the latent
    # parents; reproduce its exact recipe (parents on the window padded
    # by `radius`, uniform-disc offspring) via the generic constructor --
    # see the docstring note in _run_thomas.
    process = NeymanScottCox(
        intensity_parent=lambda_p_true,
        mu_offspring=mu_offspring_true,
        offspring_kernel=_uniform_disc_offspring_kernel(radius_true),
        pad=radius_true,
    )
    points, true_parents = simulate_neyman_scott(
        process, WINDOW, rng=rng, return_parents=True
    )

    r_grid = np.linspace(0.01, 0.25, 32)
    fit = fit_matern_cluster(points, DOMAIN, r_grid)
    radius_hat = float(fit.theta_hat[0])
    lambda_p_hat = float(fit.theta_hat[1])
    g_true = matern_cluster_pair_correlation(
        r_grid, radius_true, lambda_p_true, mu_offspring_true
    )
    return {
        "points": points,
        "true_parents": true_parents,
        "r_grid": r_grid,
        "g_fit": np.asarray(fit.g_model_at_theta, dtype=float),
        "g_true": np.asarray(g_true, dtype=float),
        "radius_true": radius_true,
        "lambda_p_true": lambda_p_true,
        "mu_offspring_true": mu_offspring_true,
        "radius_hat": radius_hat,
        "lambda_p_hat": lambda_p_hat,
        "objective_value": float(fit.objective_value),
        "success": bool(fit.success),
        "message": str(fit.message),
        "n_iter": int(fit.n_iter),
    }


def _empirical_g(points: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """Border-corrected empirical pair correlation on the unit square."""
    from nstat.extras.spatial import pair_correlation

    area = (DOMAIN[0][1] - DOMAIN[0][0]) * (DOMAIN[1][1] - DOMAIN[1][0])
    lam = float(points.shape[0]) / area
    lam_arr = np.full(points.shape[0], lam, dtype=float)
    return np.asarray(
        pair_correlation(
            points, lam_arr, r_grid,
            domain=DOMAIN, edge_correction="border",
        ),
        dtype=float,
    )


def _in_window_count(parents: np.ndarray) -> int:
    """Count latent parents that fall inside the (unpadded) unit square.

    ``simulate_neyman_scott(..., return_parents=True)`` returns parents
    *before* cropping (some hub sites just outside the window can still
    seed offspring inside it); this restricts the "true hub count" used
    for the recovered-vs-true annotation to hubs actually visible in the
    plotted window.
    """
    if parents.shape[0] == 0:
        return 0
    (xlo, xhi), (ylo, yhi) = DOMAIN
    mask = (
        (parents[:, 0] >= xlo) & (parents[:, 0] <= xhi)
        & (parents[:, 1] >= ylo) & (parents[:, 1] <= yhi)
    )
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_scatter(
    ax, points: np.ndarray, true_parents: np.ndarray, title: str,
    *, kappa_true: float, kappa_hat: float,
) -> None:
    ax.scatter(
        points[:, 0], points[:, 1],
        s=10, color="tab:blue", alpha=0.75, zorder=2,
        label="microdischarge (offspring)",
    )
    ax.scatter(
        true_parents[:, 0], true_parents[:, 1],
        s=140, marker="*", color="gold", edgecolor="black", linewidth=0.8,
        zorder=3, label="epileptogenic hub (true parent)",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    n_true_window = _in_window_count(true_parents)
    ax.text(
        0.02, 0.02,
        f"true hubs (window): {n_true_window}  |  recovered "
        f"$\\hat\\kappa$: {kappa_hat:.2f}  (target $\\kappa$={kappa_true:.1f})",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.85),
    )


def _plot_pcf(ax, r_grid, g_emp, g_fit, g_true, *, title: str) -> None:
    ax.plot(r_grid, g_emp, color="tab:blue", lw=1.6, marker="o", ms=4,
            label="empirical (border)")
    ax.plot(r_grid, g_fit, color="tab:red", lw=1.8, ls="--",
            label="fit (min-contrast)")
    ax.plot(r_grid, g_true, color="black", lw=1.0, ls=":",
            label="closed-form (truth)")
    ax.axhline(1.0, color="gray", lw=1.0, alpha=0.7, label="CSR null (no hubs)")
    ax.set_xlabel(r"lag $r$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title(title, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8)


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
    """Run the cluster-Cox + minimum-contrast demo.

    Returns
    -------
    dict
        ``{"thomas": ..., "matern": ..., "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style

    print("=" * 72)
    print("Hidden epileptogenic hubs: Thomas vs. Matérn-cluster microdischarge")
    print("recovery via minimum-contrast estimation")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    th = _run_thomas(rng)
    ma = _run_matern(rng)
    th["g_emp"] = _empirical_g(th["points"], th["r_grid"])
    ma["g_emp"] = _empirical_g(ma["points"], ma["r_grid"])

    # ---- Recovery table (Thomas) ----
    print()
    print("Thomas process (Gaussian offspring) — hidden-hub recovery")
    print("(mu_offspring not identifiable from g(r) alone)")
    print(f"  n_microdischarges : {th['points'].shape[0]}")
    print(f"  true hubs (window): {_in_window_count(th['true_parents'])}")
    print(f"  target sigma      : {th['sigma_true']:.4f}")
    print(f"  estimated sigma   : {th['sigma_hat']:.4f}")
    print(f"  target kappa      : {th['lambda_p_true']:.4f}")
    print(f"  estimated kappa   : {th['lambda_p_hat']:.4f}")
    print(f"  min-contrast S    : {th['objective_value']:.4e}  "
          f"(iters={th['n_iter']}, converged={th['success']})")
    # Recover mu_offspring from a posteriori sufficient statistic
    # mu_hat = n / (lambda_p_hat * |W|).
    win_area = (WINDOW[2] - WINDOW[0]) * (WINDOW[3] - WINDOW[1])
    if th["lambda_p_hat"] > 0:
        mu_recover = th["points"].shape[0] / (th["lambda_p_hat"] * win_area)
        print(f"  recovered mu_hat  : {mu_recover:.4f}  "
              f"(target {th['mu_offspring_true']:.4f})")

    # ---- Recovery table (Matérn) ----
    print()
    print("Matérn cluster process (uniform-disc offspring) — hidden-hub recovery")
    print(f"  n_microdischarges : {ma['points'].shape[0]}")
    print(f"  true hubs (window): {_in_window_count(ma['true_parents'])}")
    print(f"  target radius     : {ma['radius_true']:.4f}")
    print(f"  estimated radius  : {ma['radius_hat']:.4f}")
    print(f"  target kappa      : {ma['lambda_p_true']:.4f}")
    print(f"  estimated kappa   : {ma['lambda_p_hat']:.4f}")
    print(f"  min-contrast S    : {ma['objective_value']:.4e}  "
          f"(iters={ma['n_iter']}, converged={ma['success']})")
    if ma["lambda_p_hat"] > 0:
        mu_recover_m = ma["points"].shape[0] / (
            ma["lambda_p_hat"] * win_area
        )
        print(f"  recovered mu_hat  : {mu_recover_m:.4f}  "
              f"(target {ma['mu_offspring_true']:.4f})")

    # ---- Figures ----
    # === FIGURE: fig01_thomas_scatter.png ===
    fig1, ax1 = plt.subplots(figsize=(5.5, 5.4))
    _plot_scatter(
        ax1, th["points"], th["true_parents"],
        f"Hidden-hub microdischarges (Thomas, Gaussian offspring)\n"
        f"$\\sigma$={th['sigma_true']}, $\\kappa$={th['lambda_p_true']}, "
        f"n={th['points'].shape[0]}",
        kappa_true=th["lambda_p_true"], kappa_hat=th["lambda_p_hat"],
    )
    # === END FIGURE ===

    # === FIGURE: fig02_thomas_pcf.png ===
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
    _plot_pcf(
        ax2, th["r_grid"], th["g_emp"], th["g_fit"], th["g_true"],
        title="Thomas g(r) — hub clustering vs. CSR (no-hub) null",
    )
    # === END FIGURE ===

    # === FIGURE: fig03_matern_scatter.png ===
    fig3, ax3 = plt.subplots(figsize=(5.5, 5.4))
    _plot_scatter(
        ax3, ma["points"], ma["true_parents"],
        f"Hidden-hub microdischarges (Matérn, uniform-disc offspring)\n"
        f"R={ma['radius_true']}, $\\kappa$={ma['lambda_p_true']}, "
        f"n={ma['points'].shape[0]}",
        kappa_true=ma["lambda_p_true"], kappa_hat=ma["lambda_p_hat"],
    )
    # === END FIGURE ===

    # === FIGURE: fig04_matern_pcf.png ===
    fig4, ax4 = plt.subplots(figsize=(6.4, 4.8))
    _plot_pcf(
        ax4, ma["r_grid"], ma["g_emp"], ma["g_fit"], ma["g_true"],
        title="Matérn-cluster g(r) — hub clustering vs. CSR (no-hub) null",
    )
    # === END FIGURE ===

    figures = [fig1, fig2, fig3, fig4]
    fig_names = (
        "fig01_thomas_scatter",
        "fig02_thomas_pcf",
        "fig03_matern_scatter",
        "fig04_matern_pcf",
    )
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras" / "spatial_cluster_cox"
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
        "thomas": th,
        "matern": ma,
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster Cox + minimum-contrast demo "
                    "(Thomas, Matérn-cluster)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260616,
        help="np.random.default_rng seed.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write the four PNGs to --export-dir.",
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

    # Lock matplotlib backend AFTER CLI parsing — never at module top.
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
            "thomas": {
                "n_points": int(result["thomas"]["points"].shape[0]),
                "sigma_true": result["thomas"]["sigma_true"],
                "sigma_hat": result["thomas"]["sigma_hat"],
                "lambda_p_true": result["thomas"]["lambda_p_true"],
                "lambda_p_hat": result["thomas"]["lambda_p_hat"],
                "objective_value": result["thomas"]["objective_value"],
                "success": result["thomas"]["success"],
            },
            "matern": {
                "n_points": int(result["matern"]["points"].shape[0]),
                "radius_true": result["matern"]["radius_true"],
                "radius_hat": result["matern"]["radius_hat"],
                "lambda_p_true": result["matern"]["lambda_p_true"],
                "lambda_p_hat": result["matern"]["lambda_p_hat"],
                "objective_value": result["matern"]["objective_value"],
                "success": result["matern"]["success"],
            },
            "figure_paths": result["figure_paths"],
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
