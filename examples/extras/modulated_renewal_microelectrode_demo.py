#!/usr/bin/env python3
"""Demo: single-unit spike-timing regularity on a microelectrode.

End-to-end exercise of the rate-modulated renewal (conditional-ISI)
point-process model shipped in :mod:`nstat.extras.spatial`, grounded in a
realistic single-unit microelectrode recording:

**Scenario.**  A **single well-isolated unit** recorded on one channel of
a microelectrode array is driven by a slowly cycling stimulus (e.g. a
periodic sensory drive), so its instantaneous rate is stimulus-modulated
-- but its spike **timing is more regular than a Poisson process**, a
well-documented property of many thalamic and cortical regular-spiking
units (interspike-interval coefficients of variation well below 1,
Barbieri et al. 2001).  The unit is simulated as a Gamma **modulated
-renewal** process (Cox 1955; Barbieri, Quirk, Frank, Wilson & Brown
2001) with a known coefficient of variation.  All spikes are fully
synthetic (time-rescaling inverse simulation) -- no real recording or
dataset is used or claimed.

Demonstrates:

1. :func:`nstat.extras.spatial.simulate_modulated_renewal` -- simulate a
   gamma renewal process with known shape (hence known CV).
2. :func:`nstat.extras.spatial.fit_modulated_renewal` -- recover the
   covariate GLM (the stimulus-driven rate) *and* the renewal shape / CV
   jointly by alternating penalized MLE.
3. :func:`nstat.extras.spatial.renewal_cdf` -- the continuous
   time-rescaling-theorem check: the fitted model's rescaled
   (operational-time) ISIs pass a KS test against Uniform(0,1), whereas
   the same data tested under a **naive Poisson (CV=1) assumption**
   fails badly.
4. :mod:`nstat.extras.spatial.marked_gof` -- the discrete-time-rescaling
   correction (Haslinger, Pipa & Brown 2010) tied to the fitted model's
   own per-bin conditional intensity (:meth:`ModulatedRenewalResult.rate_fn`),
   confirming the fit passes the finite-bin-width-corrected KS test too.

The script is **fully synthetic** -- no figshare dataset access required.

Run::

    python examples/extras/modulated_renewal_microelectrode_demo.py            # interactive
    python examples/extras/modulated_renewal_microelectrode_demo.py --no-display
    python examples/extras/modulated_renewal_microelectrode_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/modulated_renewal_microelectrode/``) and are NOT
committed to the repository -- the export flag exists for local
inspection only.  CI never invokes it.

References:
- Barbieri R, Quirk MC, Frank LM, Wilson MA, Brown EN (2001).
  *Construction and analysis of non-Poisson stimulus-response models of
  neural spiking activity.* J Neurosci Methods 105:25-37.
- Kass RE, Ventura V (2001). *A spike-train probability model.* Neural
  Computation 13:1713-1720.
- Cox DR (1955). *Some statistical methods connected with series of
  events.* J R Stat Soc B 17(2):129-164.
- Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002). *The
  time-rescaling theorem and its application to neural spike train data
  analysis.* Neural Computation 14(2):325-346.
- Haslinger R, Pipa G, Brown E (2010). *Discrete time rescaling theorem:
  determining goodness of fit for discrete time statistical models of
  neural spiking.* Neural Computation 22(10):2477.
- Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005). *A point
  process framework for relating neural spiking activity to spiking
  history, neural ensemble, and extrinsic covariate effects.* J
  Neurophysiol 93:1074-1089.
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


T_TRIAL = 30.0  # seconds of continuous single-unit recording
STIM_FREQ_HZ = 0.5  # slow periodic sensory-drive cycle
BETA0_TRUE = float(np.log(15.0))  # ~15 Hz baseline firing rate
BETA1_TRUE = 0.5  # stimulus-drive modulation depth (log-rate units)
SHAPE_TRUE = 9.0  # gamma renewal shape -> CV = 1/sqrt(9) ~= 0.33 (regular)
RENEWAL = "gamma"

# Discretization for the fit / GOF checks.  A fine dt (Hz*dt << 1) keeps
# the O(rate * dt) discrete-time-rescaling bias (Haslinger, Pipa & Brown
# 2010) small enough for the *naive* continuous-time KS check to agree
# with the discrete-time-corrected one; both are shown below regardless.
DT_FIT = 0.002


def _true_rate_fn(t: np.ndarray) -> np.ndarray:
    """The true stimulus-driven baseline rate lambda_0(t) (Hz)."""
    return np.exp(BETA0_TRUE + BETA1_TRUE * np.cos(2.0 * np.pi * STIM_FREQ_HZ * t))


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
    """Run the microelectrode single-unit modulated-renewal demo.

    Returns
    -------
    dict
        ``{"spike_times", "fit", "ks_true_family", "ks_poisson_assumed",
        "marked_gof", "figure_paths"}``.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    from nstat import apply_plot_style
    from nstat.extras.spatial import (
        fit_modulated_renewal,
        marked_gof,
        renewal_cdf,
        simulate_modulated_renewal,
    )

    print("=" * 72)
    print("Modulated-renewal fit for a single regular-spiking microelectrode unit")
    print("=" * 72)
    print(
        f"True model: gamma renewal, shape={SHAPE_TRUE:.1f} "
        f"(CV={1.0 / np.sqrt(SHAPE_TRUE):.3f}), "
        f"stimulus-modulated baseline ~{np.exp(BETA0_TRUE):.1f} Hz "
        f"(fully synthetic -- no real recording)"
    )

    rng = np.random.default_rng(seed)
    spike_times = simulate_modulated_renewal(
        _true_rate_fn, SHAPE_TRUE, T=T_TRIAL, renewal=RENEWAL, rng=rng, dt=1e-3
    )
    n_spikes = spike_times.size
    print(f"Simulated {n_spikes} spikes over {T_TRIAL:.0f}s "
          f"({n_spikes / T_TRIAL:.2f} Hz mean rate)")

    n_bins = int(np.ceil(T_TRIAL / DT_FIT))
    bin_edges = np.arange(n_bins + 1, dtype=float) * DT_FIT
    bin_centers = bin_edges[:-1] + 0.5 * DT_FIT
    covariates = np.cos(2.0 * np.pi * STIM_FREQ_HZ * bin_centers)[:, None]

    fit = fit_modulated_renewal(
        spike_times, covariates, renewal=RENEWAL, dt=DT_FIT,
        max_iter=100, n_inner=12,
    )
    print()
    print("Recovery table:")
    print(f"  {'param':>18} | {'true':>8} | {'fitted':>8}")
    print(f"  {'beta0 (intercept)':>18} | {BETA0_TRUE:8.3f} | {fit.beta[0]:8.3f}")
    print(f"  {'beta1 (drive)':>18} | {BETA1_TRUE:8.3f} | {fit.beta[1]:8.3f}")
    print(f"  {'renewal shape':>18} | {SHAPE_TRUE:8.3f} | {fit.shape_param:8.3f}")
    print(f"  {'CV':>18} | {1.0 / np.sqrt(SHAPE_TRUE):8.3f} | {fit.cv:8.3f}")
    print(f"  converged={fit.converged}, n_iter={fit.n_iter}")

    # ---- Continuous time-rescaling KS check (true-family fit vs a naive
    # Poisson/CV=1 assumption on the *same* rescaled ISIs). ----
    u_fit = renewal_cdf(fit.rescaled_isis, fit.shape_param, RENEWAL)
    ks_fit = stats.kstest(u_fit, "uniform")
    u_poisson = renewal_cdf(fit.rescaled_isis, 1.0, RENEWAL)
    ks_poisson = stats.kstest(u_poisson, "uniform")
    ks_band = 1.358 / np.sqrt(len(u_fit))  # two-sided KS critical value, alpha=0.05

    print()
    print("Continuous time-rescaling KS test (rescaled ISIs vs Uniform(0,1)):")
    print(
        f"  fitted {RENEWAL} model : D={ks_fit.statistic:.4f}  "
        f"(band={ks_band:.4f})  {'PASS' if ks_fit.statistic < ks_band else 'FAIL'}"
    )
    print(
        f"  naive Poisson (CV=1)  : D={ks_poisson.statistic:.4f}  "
        f"(band={ks_band:.4f})  {'PASS' if ks_poisson.statistic < ks_band else 'FAIL'}"
    )

    # ---- Discrete-time-rescaling tie to marked_gof (Haslinger-Pipa-Brown). ----
    rate_fn_fitted = fit.rate_fn()
    p_k = np.clip(rate_fn_fitted(bin_centers) * DT_FIT, np.finfo(float).eps, 1.0 - 1e-12)
    spike_bins = np.clip(
        np.searchsorted(bin_edges, spike_times, side="right") - 1, 0, n_bins - 1
    )
    gof = marked_gof.marked_time_rescaling(
        spike_bins, None, p_k, rng=np.random.default_rng(seed + 1)
    )
    print()
    print("Discrete-time-rescaling correction (marked_gof.marked_time_rescaling):")
    print(
        f"  uncorrected: D={gof.ks_uncorrected:.4f}  "
        f"{'PASS' if gof.inside_uncorrected else 'FAIL'}"
    )
    print(
        f"  corrected  : D={gof.ks_corrected:.4f}  "
        f"{'PASS' if gof.inside_corrected else 'FAIL'}  (band={gof.ks_band:.4f})"
    )

    # ---- Figures ----
    # === FIGURE: fig01_raster_rate.png ===
    fig1, (ax1a, ax1b) = plt.subplots(
        2, 1, figsize=(9.5, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    show_t = 10.0
    show_mask = spike_times <= show_t
    ax1a.eventplot(
        spike_times[show_mask], lineoffsets=0.5, linelengths=0.8, colors="black",
    )
    ax1a.set_yticks([])
    ax1a.set_ylabel("unit")
    ax1a.set_title(f"Raster (first {show_t:.0f}s of {T_TRIAL:.0f}s trial)")

    t_plot = np.linspace(0.0, show_t, 1000)
    ax1b.plot(t_plot, _true_rate_fn(t_plot), color="black", lw=1.4, ls=":",
              label="true lambda_0(t)")
    fitted_lam0 = np.exp(
        fit.beta[0] + fit.beta[1] * np.cos(2.0 * np.pi * STIM_FREQ_HZ * t_plot)
    )
    ax1b.plot(t_plot, fitted_lam0, color="tab:red", lw=1.8, label="fitted lambda_0(t)")
    ax1b.set_xlabel("time (s)")
    ax1b.set_ylabel("rate (Hz)")
    ax1b.legend(loc="upper right", fontsize=8)
    # === END FIGURE ===

    # === FIGURE: fig02_isi_vs_renewal.png ===
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.8))
    u_grid = np.linspace(1e-4, 3.0, 400)
    h = 1e-4
    pdf_fit = (
        renewal_cdf(u_grid + h, fit.shape_param, RENEWAL)
        - renewal_cdf(np.maximum(u_grid - h, 0.0), fit.shape_param, RENEWAL)
    ) / (2.0 * h)
    pdf_poisson = (
        renewal_cdf(u_grid + h, 1.0, RENEWAL)
        - renewal_cdf(np.maximum(u_grid - h, 0.0), 1.0, RENEWAL)
    ) / (2.0 * h)
    ax2.hist(
        fit.rescaled_isis, bins=30, density=True, color="tab:blue", alpha=0.45,
        label="operational-time ISIs (rescaled)",
    )
    ax2.plot(u_grid, pdf_fit, color="tab:red", lw=1.8,
              label=f"fitted {RENEWAL} density (CV={fit.cv:.2f})")
    ax2.plot(u_grid, pdf_poisson, color="gray", lw=1.4, ls="--",
              label="Exp(1) density (Poisson null)")
    ax2.set_xlabel("operational-time ISI u (mean 1)")
    ax2.set_ylabel("density")
    ax2.set_title("Rescaled-ISI histogram vs fitted renewal density")
    ax2.legend(loc="upper right", fontsize=8)
    # === END FIGURE ===

    # === FIGURE: fig03_ks_rescaled.png ===
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    grid01 = np.linspace(0.0, 1.0, 200)
    ax3a.plot(grid01, grid01, color="black", lw=1.0, ls=":", label="Uniform(0,1)")
    ax3a.plot(
        np.sort(u_fit), np.linspace(0, 1, len(u_fit), endpoint=False),
        color="tab:red", lw=1.8,
        label=f"fitted {RENEWAL} (D={ks_fit.statistic:.3f})",
    )
    ax3a.plot(
        np.sort(u_poisson), np.linspace(0, 1, len(u_poisson), endpoint=False),
        color="tab:gray", lw=1.4, ls="--",
        label=f"assumed Poisson (D={ks_poisson.statistic:.3f})",
    )
    ax3a.set_xlabel("u")
    ax3a.set_ylabel("empirical CDF")
    ax3a.set_title("Continuous time-rescaling KS")
    ax3a.legend(loc="lower right", fontsize=7)

    ax3b.plot(grid01, grid01, color="black", lw=1.0, ls=":", label="Uniform(0,1)")
    ax3b.plot(
        np.sort(gof.u_uncorrected),
        np.linspace(0, 1, len(gof.u_uncorrected), endpoint=False),
        color="tab:orange", lw=1.6,
        label=f"uncorrected (D={gof.ks_uncorrected:.3f})",
    )
    ax3b.plot(
        np.sort(gof.u_corrected),
        np.linspace(0, 1, len(gof.u_corrected), endpoint=False),
        color="tab:green", lw=1.8,
        label=f"discrete-corrected (D={gof.ks_corrected:.3f})",
    )
    ax3b.set_xlabel("u")
    ax3b.set_ylabel("empirical CDF")
    ax3b.set_title("Discrete-time-rescaling correction (marked_gof)")
    ax3b.legend(loc="lower right", fontsize=7)
    fig3.suptitle("Goodness-of-fit: rescaled ISIs vs Uniform(0,1)")
    # === END FIGURE ===

    figures = [fig1, fig2, fig3]
    fig_names = ("fig01_raster_rate", "fig02_isi_vs_renewal", "fig03_ks_rescaled")
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras"
                / "modulated_renewal_microelectrode"
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
        "n_spikes": n_spikes,
        "fit": fit,
        "ks_fit_statistic": float(ks_fit.statistic),
        "ks_fit_pass": bool(ks_fit.statistic < ks_band),
        "ks_poisson_statistic": float(ks_poisson.statistic),
        "ks_poisson_pass": bool(ks_poisson.statistic < ks_band),
        "marked_gof_corrected_pass": bool(gof.inside_corrected),
        "marked_gof_uncorrected_pass": bool(gof.inside_uncorrected),
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Modulated-renewal spike-timing regularity demo on a "
                    "synthetic microelectrode unit",
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
        help="Write a compact recovery/GOF summary as JSON.",
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
        fit = result["fit"]
        summary = {
            "n_spikes": result["n_spikes"],
            "beta_hat": list(fit.beta),
            "shape_param_hat": fit.shape_param,
            "cv_hat": fit.cv,
            "converged": bool(fit.converged),
            "ks_fit_statistic": result["ks_fit_statistic"],
            "ks_fit_pass": result["ks_fit_pass"],
            "ks_poisson_statistic": result["ks_poisson_statistic"],
            "ks_poisson_pass": result["ks_poisson_pass"],
            "marked_gof_corrected_pass": result["marked_gof_corrected_pass"],
            "marked_gof_uncorrected_pass": result["marked_gof_uncorrected_pass"],
            "figure_paths": result["figure_paths"],
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
