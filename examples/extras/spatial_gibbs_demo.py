#!/usr/bin/env python3
"""Demo: Gibbs interaction processes + Berman-Turner pseudo-likelihood.

End-to-end exercise of the Gibbs interaction catalogue and the
Berman-Turner pseudo-likelihood fitter shipped in
:mod:`nstat.extras.spatial` (Tier F sub-PR-2, #196):

1. **Strauss process** (Strauss 1975) — birth-death sampler at
   ``gamma = 0.4`` (mild inhibition), then recover ``(beta, gamma)`` via
   :func:`nstat.extras.spatial.pseudo_likelihood_fit` with
   ``model_type="strauss"``.
2. **Hard-core process** (Strauss ``gamma -> 0`` limit) — dart-throwing
   rejection sampler at ``beta = 60`` (deliberately below the
   packing-fraction failure mode of
   :func:`~nstat.extras.spatial.simulate_hardcore_rejection` AND in the
   numerical envelope where the intercept-only Berman-Turner GLM
   converges), then recover ``beta`` via the intercept-only
   Berman-Turner GLM.  See the ``hardcore-bias`` note at the bottom of
   the run-table output for the expected upward bias and the literature
   reference for the closed-form correction.
3. **Area-interaction process** (Widom-Rowlinson 1970; Baddeley-van
   Lieshout 1995) — birth-death sampler with ``eta = 4.0`` (clustering),
   then recover ``(beta, eta)`` from the union-of-discs sufficient
   statistic.

The script is **fully synthetic** — no figshare dataset access required.

Run::

    python examples/extras/spatial_gibbs_demo.py            # interactive
    python examples/extras/spatial_gibbs_demo.py --no-display
    python examples/extras/spatial_gibbs_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/spatial_gibbs/``) and are NOT committed to the
repository — the export flag exists for local inspection only.  CI never
invokes it.

References:
- Strauss DJ (1975). *A model for clustering.* Biometrika 62(2):467.
- Besag J (1977). *Some methods of statistical analysis for spatial data.*
  Bull. Inst. Internat. Statist. 47:77.
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

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WINDOW = (0.0, 0.0, 1.0, 1.0)


# Q1 (resolved): ship Strauss + area-interaction at n_steps=5000.  If a
# local 60 s wall budget is exceeded the user can drop to 3000.
N_STEPS_BD = 5000


def _run_strauss(rng: np.random.Generator) -> dict:
    """Simulate + fit a Strauss process at gamma = 0.3."""
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


def _run_hardcore(rng: np.random.Generator) -> dict:
    """Simulate + fit a hard-core process at beta = 60.

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
    return {
        "points": points,
        "beta_true": beta_true,
        "R": R,
        "beta_hat": float(fit.params["beta"]),
        "pseudo_log_likelihood": float(fit.pseudo_log_likelihood),
        "n_data": int(fit.n_data),
        "n_dummy": int(fit.n_dummy),
        "fit_converged": bool(fit.glm_result.converged),
    }


def _run_area_interaction(rng: np.random.Generator) -> dict:
    """Simulate + fit an area-interaction process at eta = 4.0."""
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
    ax, points: np.ndarray, R: float, title: str
) -> None:
    ax.scatter(
        points[:, 0], points[:, 1],
        s=14, color="tab:blue", alpha=0.8, edgecolor="k", linewidth=0.4,
    )
    # Reference circle in the lower-left corner to show the interaction
    # radius R at the figure's aspect.
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
    ax.set_title(title)


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
    """Run the Gibbs interaction demo.

    Returns
    -------
    dict
        ``{"strauss": ..., "hardcore": ..., "area_interaction": ...,
        "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style

    print("=" * 72)
    print("Gibbs interaction demo — Strauss, hard-core, area-interaction")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    st = _run_strauss(rng)
    hc = _run_hardcore(rng)
    ai = _run_area_interaction(rng)

    # ---- Recovery table ----
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
    print("Hard-core process (Strauss gamma -> 0 limit) — dart-throwing")
    print(f"  n_data           : {hc['n_data']}")
    print(f"  target beta      : {hc['beta_true']:.3f}")
    print(f"  estimated beta   : {hc['beta_hat']:.3f}")
    print(f"  GLM converged    : {hc['fit_converged']}")
    print(f"  pseudo log-lik   : {hc['pseudo_log_likelihood']:.3f}")
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
        f"Strauss (beta={st['beta_true']}, gamma={st['gamma_true']}, "
        f"R={st['R']}) — n={st['n_data']}",
    )
    # === END FIGURE ===

    # === FIGURE: fig02_hardcore_scatter.png ===
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    _plot_scatter_with_radius(
        ax2, hc["points"], hc["R"],
        f"Hard-core (beta={hc['beta_true']}, R={hc['R']}) — n={hc['n_data']}",
    )
    # === END FIGURE ===

    # === FIGURE: fig03_area_interaction_scatter.png ===
    fig3, ax3 = plt.subplots(figsize=(5.5, 5.5))
    _plot_scatter_with_radius(
        ax3, ai["points"], ai["R"],
        f"Area-interaction (beta={ai['beta_true']}, "
        f"eta={ai['eta_true']}, R={ai['R']}) — n={ai['n_data']}",
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
        "strauss": st,
        "hardcore": hc,
        "area_interaction": ai,
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gibbs interaction demo "
                    "(Strauss, hard-core, area-interaction)",
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
            "strauss": {
                "n_data": result["strauss"]["n_data"],
                "beta_true": result["strauss"]["beta_true"],
                "beta_hat": result["strauss"]["beta_hat"],
                "gamma_true": result["strauss"]["gamma_true"],
                "gamma_hat": result["strauss"]["gamma_hat"],
                "fit_converged": result["strauss"]["fit_converged"],
            },
            "hardcore": {
                "n_data": result["hardcore"]["n_data"],
                "beta_true": result["hardcore"]["beta_true"],
                "beta_hat": result["hardcore"]["beta_hat"],
                "fit_converged": result["hardcore"]["fit_converged"],
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
