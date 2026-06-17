#!/usr/bin/env python3
"""Demo: cluster Cox processes + minimum-contrast estimation.

End-to-end exercise of the cluster-Cox catalogue and minimum-contrast
inference shipped in :mod:`nstat.extras.spatial` (Tier F sub-PR-1, #195):

1. Simulate a **Thomas process** with isotropic Gaussian offspring
   displacement (Thomas 1949; Møller-Waagepetersen 2003 §5.3).
2. Simulate a **Matérn cluster process** with uniform-disc offspring
   (Matérn 1986).
3. Recover both parameter pairs ``(sigma, lambda_p)`` / ``(R, lambda_p)``
   from the simulated patterns with
   :func:`nstat.extras.spatial.fit_thomas` and
   :func:`nstat.extras.spatial.fit_matern_cluster` — Diggle's (2013 §6.2.1)
   minimum-contrast estimator on the SOIRS pair correlation.
4. Plot the four diagnostic figures and print a recovery table.

The script is **fully synthetic** — no figshare dataset access required.

Run::

    python examples/extras/spatial_cluster_cox_demo.py            # interactive
    python examples/extras/spatial_cluster_cox_demo.py --no-display
    python examples/extras/spatial_cluster_cox_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to ``docs/figures/extras/spatial_cluster_cox/``)
and are NOT committed to the repository — the export flag exists for
local inspection only.  CI never invokes it.

References:
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


def _run_thomas(rng: np.random.Generator) -> dict:
    """Simulate + fit a Thomas process on the unit square."""
    from nstat.extras.spatial import (
        fit_thomas,
        simulate_thomas,
        thomas_pair_correlation,
    )

    sigma_true = 0.04
    lambda_p_true = 30.0
    mu_offspring_true = 8.0

    points = simulate_thomas(
        intensity_parent=lambda_p_true,
        mu_offspring=mu_offspring_true,
        sigma=sigma_true,
        window=WINDOW,
        rng=rng,
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
    """Simulate + fit a Matérn cluster process on the unit square."""
    from nstat.extras.spatial import (
        fit_matern_cluster,
        matern_cluster_pair_correlation,
        simulate_matern_cluster,
    )

    radius_true = 0.06
    lambda_p_true = 25.0
    mu_offspring_true = 10.0

    points = simulate_matern_cluster(
        intensity_parent=lambda_p_true,
        mu_offspring=mu_offspring_true,
        radius=radius_true,
        window=WINDOW,
        rng=rng,
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_scatter(ax, points: np.ndarray, title: str) -> None:
    ax.scatter(
        points[:, 0], points[:, 1],
        s=10, color="tab:blue", alpha=0.75,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def _plot_pcf(ax, r_grid, g_emp, g_fit, g_true, *, title: str) -> None:
    ax.plot(r_grid, g_emp, color="tab:blue", lw=1.6, marker="o", ms=4,
            label="empirical (border)")
    ax.plot(r_grid, g_fit, color="tab:red", lw=1.8, ls="--",
            label="fit (min-contrast)")
    ax.plot(r_grid, g_true, color="black", lw=1.0, ls=":",
            label="closed-form (truth)")
    ax.axhline(1.0, color="gray", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"lag $r$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title(title)
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
    print("Cluster Cox + minimum-contrast demo — Thomas and Matérn-cluster")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    th = _run_thomas(rng)
    ma = _run_matern(rng)
    th["g_emp"] = _empirical_g(th["points"], th["r_grid"])
    ma["g_emp"] = _empirical_g(ma["points"], ma["r_grid"])

    # ---- Recovery table (Thomas) ----
    print()
    print("Thomas process — recovery (mu_offspring not identifiable from g(r))")
    print(f"  n_points          : {th['points'].shape[0]}")
    print(f"  target sigma      : {th['sigma_true']:.4f}")
    print(f"  estimated sigma   : {th['sigma_hat']:.4f}")
    print(f"  target lambda_p   : {th['lambda_p_true']:.4f}")
    print(f"  estimated lambda_p: {th['lambda_p_hat']:.4f}")
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
    print("Matérn cluster process — recovery")
    print(f"  n_points          : {ma['points'].shape[0]}")
    print(f"  target radius     : {ma['radius_true']:.4f}")
    print(f"  estimated radius  : {ma['radius_hat']:.4f}")
    print(f"  target lambda_p   : {ma['lambda_p_true']:.4f}")
    print(f"  estimated lambda_p: {ma['lambda_p_hat']:.4f}")
    print(f"  min-contrast S    : {ma['objective_value']:.4e}  "
          f"(iters={ma['n_iter']}, converged={ma['success']})")
    if ma["lambda_p_hat"] > 0:
        mu_recover_m = ma["points"].shape[0] / (
            ma["lambda_p_hat"] * win_area
        )
        print(f"  recovered mu_hat  : {mu_recover_m:.4f}  "
              f"(target {ma['mu_offspring_true']:.4f})")

    # ---- Figures ----
    fig1, ax1 = plt.subplots(figsize=(5.5, 5.4))
    _plot_scatter(
        ax1, th["points"],
        f"Thomas pattern (sigma={th['sigma_true']}, "
        f"lambda_p={th['lambda_p_true']}, "
        f"n={th['points'].shape[0]})",
    )

    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
    _plot_pcf(
        ax2, th["r_grid"], th["g_emp"], th["g_fit"], th["g_true"],
        title="Thomas g(r) — empirical vs min-contrast fit vs truth",
    )

    fig3, ax3 = plt.subplots(figsize=(5.5, 5.4))
    _plot_scatter(
        ax3, ma["points"],
        f"Matérn-cluster pattern (R={ma['radius_true']}, "
        f"lambda_p={ma['lambda_p_true']}, "
        f"n={ma['points'].shape[0]})",
    )

    fig4, ax4 = plt.subplots(figsize=(6.4, 4.8))
    _plot_pcf(
        ax4, ma["r_grid"], ma["g_emp"], ma["g_fit"], ma["g_true"],
        title="Matérn-cluster g(r) — empirical vs min-contrast fit vs truth",
    )

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
