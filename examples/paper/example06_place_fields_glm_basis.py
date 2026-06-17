#!/usr/bin/env python3
"""Example 06 — 2-D Place-Field Recovery With a B-Spline GLM Basis.

This example is a Python-only complement to Example 04: it does NOT
modify example04 or its data path.  Where example04 fits Gaussian or
Zernike receptive-field models to recorded place cells, this example
exercises the new :mod:`nstat.extras.spatial` building blocks against a
**simulated** 2-D inhomogeneous Poisson process whose ground-truth
log-intensity is known analytically, so the rate recovery and the
second-order goodness-of-fit can be calibrated directly.

Pipeline:

1. Simulate ~600 events from a sum-of-three-Gaussian-bumps log-rate on
   the unit square via thinning.
2. Bin the events on a 24x24 grid and fit a tensor-product cubic
   B-spline Poisson GLM
   (``BSplineBasis2D.from_grid`` + :func:`nstat.glm.fit_poisson_glm`,
   with the log cell area carried as an ``offset``).
3. Fit a basis-projected LGCP comparator
   (:func:`~nstat.extras.spatial.lgcp.lgcp_fit_glm` with a Matern-5/2
   prior on the spline coefficients evaluated at their Greville
   abscissae — Diggle et al. 2013).
4. Second-order diagnostic: edge-corrected pair correlation
   :math:`\\hat g(r)` (``edge_correction='isotropic'``, Ripley 1976)
   plus the global-rank envelope of Myllymaki et al. 2017 against the
   true intensity.
5. Plot true vs basis recovery, the LGCP posterior-mean rate with a 90%
   log-normal credible band, and the pair-correlation curve with its
   global envelope.

References:
- Baddeley AJ, Moller J, Waagepetersen R (2000). Statistica Neerlandica 54(3):329.
- Myllymaki M, Mrkvicka T, Grabarnik P, Seijo H, Hahn U (2017). JRSS-B 79(2):381.
- Daley DJ, Vere-Jones D (2003). *An Introduction to the Theory of Point Processes*, Vol I.
- Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal Point Patterns*, 3rd ed.
- Kass RE, Eden UT, Brown EN (2014). *Analysis of Neural Data*, Ch. 19.

Expected outputs:
- Figure 1: ground-truth, B-spline-recovered, and LGCP posterior-mean rate maps.
- Figure 2: per-cell LGCP rate slice with the 90% credible band.
- Figure 3: edge-corrected pair correlation with global envelope and the
  Poisson reference line ``g(r) = 1``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat import apply_plot_style  # noqa: E402
from nstat.extras.spatial import (  # noqa: E402
    BSplineBasis2D,
    MaternPrior,
    global_envelope,
    lgcp_fit_glm,
    pair_correlation,
)
from nstat.glm import fit_poisson_glm  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
#  Ground-truth log-rate: sum of three 2-D Gaussian bumps on the unit square.
# ──────────────────────────────────────────────────────────────────────────

_BUMPS = (
    # (x0,  y0,   sx,    sy,    amp)
    (0.30, 0.30, 0.10, 0.10, 1.5),
    (0.70, 0.65, 0.08, 0.12, 1.2),
    (0.45, 0.80, 0.12, 0.07, 0.9),
)
_LOG_BASELINE = np.log(400.0)  # base rate (events / unit area)


def _true_log_rate(xy: np.ndarray) -> np.ndarray:
    """Analytic ground-truth log-intensity on the unit square."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    x = xy[:, 0]
    y = xy[:, 1]
    log_rate = np.full(x.shape, _LOG_BASELINE)
    for x0, y0, sx, sy, amp in _BUMPS:
        log_rate = log_rate + amp * np.exp(
            -0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2)
        )
    return log_rate


def _true_rate(xy: np.ndarray) -> np.ndarray:
    return np.exp(_true_log_rate(xy))


def _simulate_pattern(rng: np.random.Generator) -> np.ndarray:
    """Thinning simulation of the inhomogeneous Poisson process."""
    # Find a tight upper bound for the true rate on the unit square.
    nref = 80
    ax = np.linspace(0.0, 1.0, nref)
    XX, YY = np.meshgrid(ax, ax, indexing="ij")
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    lam_max = float(_true_rate(grid).max()) * 1.05  # safety margin
    n_prop = rng.poisson(lam_max)  # area = 1
    cand = rng.uniform(size=(n_prop, 2))
    keep = rng.uniform(size=n_prop) < (_true_rate(cand) / lam_max)
    return cand[keep]


# ──────────────────────────────────────────────────────────────────────────
#  Fits
# ──────────────────────────────────────────────────────────────────────────


def _fit_basis_glm(pts: np.ndarray, n_grid: int = 24):
    """Tensor-product B-spline Poisson GLM with log-area offset."""
    grid = np.linspace(0.0, 1.0, n_grid)
    basis = BSplineBasis2D.from_grid(
        grid_x=grid, grid_y=grid, n_knots=(8, 8), degree=3
    )
    B = basis.design_matrix()  # (n_grid**2, 64)

    # Bin events on the same (n_grid x n_grid) cell grid in ij flattening,
    # so the row order matches BSplineBasis2D.from_grid.
    edges = np.linspace(0.0, 1.0, n_grid + 1)
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[edges, edges])
    counts = H.ravel().astype(float)
    cell_area = float(1.0 / (n_grid * n_grid))
    offset = np.full(counts.shape, np.log(cell_area))

    # The tensor-product B-spline basis is a partition of unity (rows sum
    # to 1), so adding a separate intercept column makes the design rank-
    # deficient — every basis coefficient is identifiable in absolute log-
    # rate terms.  Drop the intercept and use a small L2 ridge for stability.
    # The unpenalized Newton-IRLS in fit_poisson_glm can overshoot from a
    # zero start when many cells are nearly empty; a moderate diagonal L2
    # damps the early steps without meaningfully biasing the converged fit.
    glm = fit_poisson_glm(
        B, counts, offset=offset, include_intercept=False, l2=0.3,
        max_iter=300,
    )
    eta = B @ glm.coefficients
    rate_hat = np.exp(eta)
    return basis, B, counts, glm, rate_hat


def _fit_lgcp(pts: np.ndarray, basis: BSplineBasis2D, n_grid: int = 24):
    """Basis-projected LGCP with a Matern-5/2 GP prior on the coefficients."""
    prior = MaternPrior(nu=2.5, length_scale=0.12, marginal_var=1.0)
    res = lgcp_fit_glm(
        pts,
        ((0.0, 1.0), (0.0, 1.0)),
        basis,
        prior,
        grid=n_grid,
    )
    mean, lo, hi = res.rate_map(level=0.90)
    return res, mean, lo, hi


# ──────────────────────────────────────────────────────────────────────────
#  Second-order GoF
# ──────────────────────────────────────────────────────────────────────────


def _pcf_with_envelope(pts: np.ndarray, rng: np.random.Generator):
    """Pair correlation g(r) with Ripley isotropic edge correction and a
    99-sim global-rank envelope against the *true* intensity."""
    r_grid = np.linspace(0.02, 0.20, 14)
    bw = 0.04
    domain = ((0.0, 1.0), (0.0, 1.0))

    # SOIRS-reweight by the analytic true intensity at the event locations.
    lam_true = _true_rate(pts)
    g = pair_correlation(
        pts,
        lam_true,
        r_grid,
        bw=bw,
        domain=domain,
        edge_correction="isotropic",
    )
    # global_envelope does NOT accept edge_correction (verified against
    # spatial_gof.py); the envelope uses the default epanechnikov path.
    env = global_envelope(
        pts,
        _true_rate,
        r_grid,
        n_sim=99,
        domain=domain,
        statistic="pcf",
        bw=bw,
        rng=rng,
    )
    return r_grid, g, env


# ──────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────


def _rate_grid_ij(n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell centres in ij flattening matching BSplineBasis2D row order."""
    axis = np.linspace(0.0, 1.0, n_grid)
    XX, YY = np.meshgrid(axis, axis, indexing="ij")
    return axis, XX, YY


def _plot_true_vs_basis(
    pts: np.ndarray, rate_basis: np.ndarray, lgcp_mean: np.ndarray, n_grid: int
):
    """Three side-by-side rate maps: truth, B-spline GLM, LGCP."""
    axis, XX, YY = _rate_grid_ij(n_grid)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    rate_true = _true_rate(grid_pts).reshape(n_grid, n_grid)
    rate_basis_2d = rate_basis.reshape(n_grid, n_grid)
    rate_lgcp_2d = lgcp_mean.reshape(n_grid, n_grid)
    vmax = float(max(rate_true.max(), rate_basis_2d.max(), rate_lgcp_2d.max()))

    # === FIGURE: fig01_true_vs_basis_recovered_rate.png ===
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    titles = ("Ground truth", "B-spline GLM", "LGCP (Matern-5/2)")
    for ax, field, title in zip(
        axes, (rate_true, rate_basis_2d, rate_lgcp_2d), titles
    ):
        im = ax.pcolormesh(
            axis, axis, field.T, shading="auto", cmap="viridis",
            vmin=0.0, vmax=vmax,
        )
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color="w", alpha=0.45,
                   edgecolor="none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Example 06 — true vs basis-recovered rate (~600 events)")
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_lgcp_band(
    pts: np.ndarray,
    lgcp_mean: np.ndarray,
    lgcp_lo: np.ndarray,
    lgcp_hi: np.ndarray,
    n_grid: int,
):
    """Heatmap of the LGCP posterior-mean rate plus a horizontal slice
    through ``y = 0.30`` (the dominant bump) showing the 90% band."""
    axis, _, _ = _rate_grid_ij(n_grid)
    mean_2d = lgcp_mean.reshape(n_grid, n_grid)
    lo_2d = lgcp_lo.reshape(n_grid, n_grid)
    hi_2d = lgcp_hi.reshape(n_grid, n_grid)

    # === FIGURE: fig02_lgcp_glm_credible_band.png ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    im = axes[0].pcolormesh(axis, axis, mean_2d.T, shading="auto",
                            cmap="magma")
    axes[0].scatter(pts[:, 0], pts[:, 1], s=3, color="w", alpha=0.4,
                    edgecolor="none")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("LGCP posterior-mean rate")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axhline(0.30, color="cyan", lw=1.0, ls="--", alpha=0.8,
                    label="slice y = 0.30")
    axes[0].legend(loc="upper right", fontsize=8)

    # Find the row of the analysis grid closest to y = 0.30.
    j = int(np.argmin(np.abs(axis - 0.30)))
    grid_pts = np.column_stack([axis, np.full_like(axis, axis[j])])
    rate_true = _true_rate(grid_pts)
    mean_slice = mean_2d[:, j]
    lo_slice = lo_2d[:, j]
    hi_slice = hi_2d[:, j]
    axes[1].fill_between(axis, lo_slice, hi_slice, color="tab:orange",
                         alpha=0.30, label="90% credible band")
    axes[1].plot(axis, mean_slice, color="tab:orange", lw=1.8,
                 label="LGCP posterior mean")
    axes[1].plot(axis, rate_true, color="k", lw=1.4, ls="--",
                 label="Ground truth")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("rate(x, 0.30)")
    axes[1].set_title("LGCP log-normal credible band (slice)")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Example 06 — basis-projected LGCP with a Matern-5/2 prior")
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_pcf_envelope(r_grid: np.ndarray, g: np.ndarray, env) -> "plt.Figure":
    """Pair correlation g(r) with the global-rank envelope."""
    # === FIGURE: fig03_pcf_with_global_envelope.png ===
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.fill_between(r_grid, env.lo, env.hi, color="gray", alpha=0.35,
                    label=f"global envelope (n_sim={env.n_sim})")
    ax.plot(r_grid, g, color="tab:blue", lw=1.8, marker="o", ms=4,
            label="observed g(r) (isotropic)")
    ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.7,
               label="Poisson null g(r) = 1")
    ax.set_xlabel("lag r")
    ax.set_ylabel("g(r)")
    ax.set_title(
        "Example 06 — edge-corrected pair correlation with global envelope"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    # === END FIGURE ===
    return fig


# ──────────────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────────────


def run_example06(
    *,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool | None = True,
    plot_style: str = "legacy",
) -> dict:
    """Run Example 06: B-spline GLM + LGCP recovery of a 2-D place field.

    Returns
    -------
    dict
        Keys: ``pts``, ``rate_hat_basis``, ``lgcp_mean``, ``lgcp_lo``,
        ``lgcp_hi``, ``r_grid``, ``g``, ``envelope``, ``rmse_basis``,
        ``rmse_lgcp``, ``figure_paths``.
    """
    print("=" * 70)
    print("Example 06: 2-D Place-Field Recovery (B-spline GLM + LGCP)")
    print("=" * 70)

    rng = np.random.default_rng(20260616)
    n_grid = 24

    pts = _simulate_pattern(rng)
    print(f"  simulated n = {pts.shape[0]} events on unit square")

    basis, B, counts, glm, rate_basis = _fit_basis_glm(pts, n_grid=n_grid)
    print(f"  B-spline GLM: K = {B.shape[1]} basis fns, "
          f"{glm.n_iter} IRLS iters, converged = {glm.converged}")

    lgcp_res, lgcp_mean, lgcp_lo, lgcp_hi = _fit_lgcp(pts, basis, n_grid=n_grid)
    print(f"  LGCP-GLM: {lgcp_res.n_iter} IRLS iters, "
          f"converged = {lgcp_res.converged}")

    # RMSE on the analysis grid (ij flattening).
    axis, XX, YY = _rate_grid_ij(n_grid)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    rate_true_flat = _true_rate(grid_pts)
    rmse_basis = float(
        np.sqrt(np.mean((rate_basis - rate_true_flat) ** 2))
    )
    rmse_lgcp = float(
        np.sqrt(np.mean((lgcp_mean - rate_true_flat) ** 2))
    )
    print(f"  RMSE(truth, B-spline) = {rmse_basis:.3f}")
    print(f"  RMSE(truth, LGCP)     = {rmse_lgcp:.3f}")

    r_grid, g, env = _pcf_with_envelope(pts, rng)
    print(f"  pair-correlation envelope: inside = {env.inside}, "
          f"p_interval = {env.p_interval}")

    fig1 = _plot_true_vs_basis(pts, rate_basis, lgcp_mean, n_grid)
    fig2 = _plot_lgcp_band(pts, lgcp_mean, lgcp_lo, lgcp_hi, n_grid)
    fig3 = _plot_pcf_envelope(r_grid, g, env)
    figures = [fig1, fig2, fig3]
    for fig in figures:
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "example06"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        fig_names = (
            "fig01_true_vs_basis_recovered_rate",
            "fig02_lgcp_glm_credible_band",
            "fig03_pcf_with_global_envelope",
        )
        for fig, name in zip(figures, fig_names):
            path = export_dir / f"{name}.png"
            fig.savefig(path, dpi=200, facecolor="w", edgecolor="none")
            figure_paths.append(path)
            print(f"  Saved: {path}")

    if bool(visible):
        plt.show()
    else:
        plt.close("all")

    return {
        "pts": pts,
        "rate_hat_basis": rate_basis,
        "lgcp_mean": lgcp_mean,
        "lgcp_lo": lgcp_lo,
        "lgcp_hi": lgcp_hi,
        "r_grid": r_grid,
        "g": g,
        "envelope": env,
        "rmse_basis": rmse_basis,
        "rmse_lgcp": rmse_lgcp,
        "figure_paths": [str(p) for p in figure_paths],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 06: 2-D Place-Field Recovery (B-spline GLM + LGCP)"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help=("Repository root (used by other paper examples "
                              "for dataset lookup; this script is data-free "
                              "and only uses it to resolve the default "
                              "export-dir under docs/figures/example06)."))
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively.")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without displaying figures (headless).")
    parser.add_argument("--plot-style", choices=("modern", "legacy"),
                        default="legacy",
                        help="Figure styling.")
    args = parser.parse_args()

    if args.no_display:
        visible = False
    else:
        visible = bool(args.show)
    result = run_example06(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        visible=visible,
        plot_style=args.plot_style,
    )
    if args.output_json:
        # The full result holds NumPy arrays; serialise only the scalars.
        summary = {
            "n_events": int(result["pts"].shape[0]),
            "rmse_basis": result["rmse_basis"],
            "rmse_lgcp": result["rmse_lgcp"],
            "envelope_inside": bool(result["envelope"].inside),
            "envelope_p_interval": list(map(float, result["envelope"].p_interval)),
        }
        args.output_json.write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
