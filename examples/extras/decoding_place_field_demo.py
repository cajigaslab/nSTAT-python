"""Demo: population-vector bias vs. maximum-likelihood decode of reach
direction from a non-uniformly tuned M1 population.

**Question** (design spec S4.2, intracortical motor-BCI thread): is the
classic **population vector** (Georgopoulos et al. 1982, 1986) biased when
neurons' preferred directions are **not** uniformly distributed around the
circle, and does a maximum-likelihood/Bayesian tuning-curve decode stay
unbiased where the population vector fails?

Ground truth
------------
An 8-direction **center-out** task (Georgopoulos et al. 1982's own
paradigm: arm movements at 45-degree intervals from a common origin).
A population of ~50-100 M1-like units each has a von-Mises-shaped
directional tuning curve, ``rate(theta) = exp(b0 + kappa * cos(theta -
PD))`` -- the log-linear, always-positive form of the classic cosine
tuning model ``d = b0 + b1 sin(theta) + b2 cos(theta)`` (Georgopoulos et
al. 1982) -- with its own baseline rate, modulation depth ``kappa``, and
preferred direction (PD).  Critically, PDs are **deliberately clustered**
rather than uniform: most units' PDs are drawn from a tight von Mises
distribution around one arc of the circle, with only a uniform minority
covering the rest -- exactly the "nonuniform distribution of preferred
directions" Sanger (1996) analyzes as breaking the population vector.
Each unit's spike count in a trial is an independent Poisson draw from its
tuning curve over a short movement-related epoch.

Recovery
--------
1. Per-neuron tuning curves are fit from training trials with
   :func:`nstat.fit_poisson_glm` -- Poisson regression of spike count on
   ``(cos theta, sin theta)`` -- recovering each unit's baseline rate,
   modulation depth, and PD.  This is nSTAT's own Poisson-GLM machinery;
   only the population-vector/ML *decoders* below are novel to this demo.
2. **Population vector**: on held-out test trials, decode
   ``theta_hat = angle(sum_i (r_i - baseline_i) * PD_unit_vector_i)``
   (Georgopoulos et al. 1986; Kettner et al. 1988) -- a vector sum over
   fitted preferred directions.  Because the sum implicitly assumes PDs
   uniformly span the circle, it is systematically pulled toward the
   dense PD region when they do not.
3. **Maximum likelihood**: on the same held-out trials, grid-search the
   Poisson log-likelihood of the observed population count vector against
   every neuron's fitted tuning curve (Sanger 1996's probability-density
   estimator) -- no assumption about how PDs are distributed, so it stays
   unbiased.

Mean absolute angular decode error is reported for both decoders across
all held-out center-out trials; the ML decoder's error is smaller,
reproducing Sanger's central result.

References
----------
- Georgopoulos AP, Kalaska JF, Caminiti R, Massey JT (1982). *On the
  relations between the direction of two-dimensional arm movements and
  cell discharge in primate motor cortex.* J Neurosci 2(11):1527-1537.
  -- the 8-direction center-out task and the cosine tuning-curve model.
- Georgopoulos AP, Schwartz AB, Kettner RE (1986). *Neuronal population
  coding of movement direction.* Science 233(4771):1416-1419. -- the
  population-vector construction.
- Kettner RE, Schwartz AB, Georgopoulos AP (1988). *Primate motor cortex
  and free arm movements to visual targets in three-dimensional space.
  III. Positional gradients and population coding of movement direction
  from various movement origins.* J Neurosci 8(8):2938-2947.
- Sanger TD (1996). *Probability density estimation for the
  interpretation of neural population codes.* J Neurophysiol
  76(4):2790-2793. -- shows by simulation that density/ML estimation
  "correctly finds movement directions for nonuniform distributions of
  preferred directions ... whereas the population vector method fails
  for these cases"; the direct basis for this demo's recovery story.

Cross-links
-----------
``examples/extras/decoding_clusterless_demo.py`` -- a different motor-BCI
decoding failure mode (spike-sorting degradation) on the same
intracortical-BCI thread.  ``examples/extras/latents_gpfa_demo.py`` --
the same M1/PMd reaching substrate viewed through population-level
low-dimensional structure rather than a per-trial direction decode.

Run::

    python examples/extras/decoding_place_field_demo.py
    python examples/extras/decoding_place_field_demo.py --export-figures
    python examples/extras/decoding_place_field_demo.py --no-display
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 8-direction center-out task (Georgopoulos et al. 1982): targets at 45-degree
# intervals from a common origin.
# ---------------------------------------------------------------------------
N_TARGETS = 8
TARGET_ANGLES = np.arange(N_TARGETS) * (2.0 * np.pi / N_TARGETS)

N_NEURONS = 80
# Fraction of the population whose preferred directions are drawn from a
# tight von Mises cluster (deliberately non-uniform, per Sanger 1996's
# critique); the remainder are uniform so every target direction still has
# some coverage.
CLUSTER_FRACTION = 0.65
CLUSTER_CENTER = np.deg2rad(65.0)
CLUSTER_CONCENTRATION = 3.0  # von Mises kappa for the PD *distribution*

BASELINE_RATE_RANGE_HZ = (8.0, 20.0)
MODULATION_KAPPA_RANGE = (0.3, 0.9)  # per-neuron tuning-curve concentration
TRIAL_DURATION_S = 0.5  # movement-related epoch per center-out trial

N_TRAIN_REPS = 30  # training trials per target direction (240 total)
N_TEST_REPS = 50  # held-out test trials per target direction (400 total)

THETA_GRID = np.linspace(-np.pi, np.pi, 361)[:-1]  # 1-degree resolution


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def _simulate_preferred_directions(n_neurons: int, rng: np.random.Generator) -> np.ndarray:
    """Deliberately non-uniform (clustered) preferred-direction population.

    ``CLUSTER_FRACTION`` of units draw their PD from a tight von Mises
    distribution around ``CLUSTER_CENTER``; the rest are uniform on the
    circle.  This is the "nonuniform distribution of preferred directions"
    Sanger (1996) shows breaks the population-vector decode.
    """
    n_clustered = int(round(CLUSTER_FRACTION * n_neurons))
    n_uniform = n_neurons - n_clustered
    pd_clustered = rng.vonmises(CLUSTER_CENTER, CLUSTER_CONCENTRATION, size=n_clustered)
    pd_uniform = rng.uniform(-np.pi, np.pi, size=n_uniform)
    pd = np.concatenate([pd_clustered, pd_uniform])
    rng.shuffle(pd)
    return pd


def _simulate_tuning_params(n_neurons: int, rng: np.random.Generator):
    """Per-neuron baseline log-rate and modulation depth (von Mises kappa)."""
    baseline_rate = rng.uniform(*BASELINE_RATE_RANGE_HZ, size=n_neurons)
    kappa = rng.uniform(*MODULATION_KAPPA_RANGE, size=n_neurons)
    return np.log(baseline_rate), kappa


def _simulate_trials(
    thetas: np.ndarray,
    pd_true: np.ndarray,
    baseline_log_rate: np.ndarray,
    kappa_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Poisson spike counts, shape (n_trials, n_neurons), for each trial's
    movement direction ``thetas`` under the true von-Mises tuning curves.
    """
    log_rate = (
        baseline_log_rate[None, :]
        + kappa_true[None, :] * np.cos(thetas[:, None] - pd_true[None, :])
    )
    mean_count = np.exp(log_rate) * TRIAL_DURATION_S
    return rng.poisson(mean_count)


# ---------------------------------------------------------------------------
# Encoding: per-neuron tuning-curve fit via nstat's Poisson GLM
# ---------------------------------------------------------------------------


def _fit_tuning_curves(theta_train: np.ndarray, counts_train: np.ndarray):
    """Fit each neuron's tuning curve with :func:`nstat.fit_poisson_glm`.

    Regresses spike count on ``(cos theta, sin theta)`` with an
    ``offset=log(TRIAL_DURATION_S)`` so the intercept and coefficients are
    directly in log-rate (Hz) units: ``log rate(theta) = intercept + c1 *
    cos(theta) + c2 * sin(theta)``, the same log-linear cosine/von-Mises
    tuning form used to generate the ground truth.

    Returns
    -------
    intercept_hat, coef_hat (n_neurons, 2), pd_hat, n_converged
    """
    from nstat import fit_poisson_glm

    n_neurons = counts_train.shape[1]
    design = np.column_stack([np.cos(theta_train), np.sin(theta_train)])
    offset = np.full(theta_train.shape[0], np.log(TRIAL_DURATION_S))

    intercept_hat = np.empty(n_neurons)
    coef_hat = np.empty((n_neurons, 2))
    n_converged = 0
    for i in range(n_neurons):
        fit = fit_poisson_glm(design, counts_train[:, i], offset=offset)
        intercept_hat[i] = fit.intercept
        coef_hat[i] = fit.coefficients
        n_converged += int(fit.converged)

    pd_hat = np.arctan2(coef_hat[:, 1], coef_hat[:, 0])
    return intercept_hat, coef_hat, pd_hat, n_converged


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


def _decode_population_vector(
    counts_test: np.ndarray, pd_hat: np.ndarray, baseline_rate_hat: np.ndarray
) -> np.ndarray:
    """Classic Georgopoulos population-vector decode (vectorized over trials).

    ``theta_hat = angle(sum_i (r_i - baseline_i) * (cos PD_i, sin PD_i))``
    -- a vector sum of each neuron's above-baseline firing rate along its
    own fitted preferred direction (Georgopoulos et al. 1986; Kettner et
    al. 1988).  The vector sum implicitly weights the circle by how
    densely populated it is with PDs, so it is unbiased only when PDs are
    (near-)uniform.
    """
    observed_rate = counts_test / TRIAL_DURATION_S  # (n_trials, n_neurons)
    weights = observed_rate - baseline_rate_hat[None, :]
    vx = weights @ np.cos(pd_hat)
    vy = weights @ np.sin(pd_hat)
    return np.arctan2(vy, vx)


def _decode_maximum_likelihood(
    counts_test: np.ndarray, intercept_hat: np.ndarray, coef_hat: np.ndarray
) -> np.ndarray:
    """Grid-search ML/Bayesian decode (Sanger 1996): the direction on
    ``THETA_GRID`` maximizing the Poisson log-likelihood of the observed
    population count vector under every neuron's fitted tuning curve.
    Fully vectorized over the test-trial batch via one matrix product.
    """
    log_offset = np.log(TRIAL_DURATION_S)
    # eta[g, i] = log rate(theta_g) for neuron i, in *count* units (i.e.
    # including the trial-duration offset), shape (n_grid, n_neurons).
    eta = (
        intercept_hat[None, :]
        + np.cos(THETA_GRID)[:, None] * coef_hat[None, :, 0]
        + np.sin(THETA_GRID)[:, None] * coef_hat[None, :, 1]
        + log_offset
    )
    mean_count = np.exp(np.clip(eta, -20.0, 20.0))
    const = mean_count.sum(axis=1)  # (n_grid,), same for every trial
    # loglik[trial, g] = sum_i counts[trial, i] * eta[g, i] - const[g]
    # (dropping the log(count!) term, constant across candidate thetas).
    loglik = counts_test @ eta.T - const[None, :]
    best = np.argmax(loglik, axis=1)
    return THETA_GRID[best]


def _circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed circular difference ``a - b`` wrapped to (-pi, pi]."""
    return np.angle(np.exp(1j * (a - b)))


def _wrap_deg(deg: np.ndarray) -> np.ndarray:
    """Wrap degrees into (-180, 180] (matches ``THETA_GRID``'s range)."""
    return (deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# End-to-end experiment
# ---------------------------------------------------------------------------


def _run_experiment(seed: int) -> dict:
    rng_pd = np.random.default_rng(seed)
    rng_params = np.random.default_rng(seed + 1)
    rng_train = np.random.default_rng(seed + 2)
    rng_test = np.random.default_rng(seed + 3)

    pd_true = _simulate_preferred_directions(N_NEURONS, rng_pd)
    baseline_log_rate, kappa_true = _simulate_tuning_params(N_NEURONS, rng_params)

    theta_train = np.repeat(TARGET_ANGLES, N_TRAIN_REPS)
    counts_train = _simulate_trials(
        theta_train, pd_true, baseline_log_rate, kappa_true, rng_train
    )

    intercept_hat, coef_hat, pd_hat, n_converged = _fit_tuning_curves(
        theta_train, counts_train
    )
    baseline_rate_hat = np.exp(intercept_hat)

    theta_test = np.repeat(TARGET_ANGLES, N_TEST_REPS)
    counts_test = _simulate_trials(
        theta_test, pd_true, baseline_log_rate, kappa_true, rng_test
    )

    pv_decode = _decode_population_vector(counts_test, pd_hat, baseline_rate_hat)
    ml_decode = _decode_maximum_likelihood(counts_test, intercept_hat, coef_hat)

    pv_error = np.abs(_circular_diff(pv_decode, theta_test))
    ml_error = np.abs(_circular_diff(ml_decode, theta_test))

    per_direction = []
    for target in TARGET_ANGLES:
        mask = theta_test == target
        pv_mean = np.arctan2(
            np.sin(pv_decode[mask]).mean(), np.cos(pv_decode[mask]).mean()
        )
        ml_mean = np.arctan2(
            np.sin(ml_decode[mask]).mean(), np.cos(ml_decode[mask]).mean()
        )
        per_direction.append(
            {
                "target": target,
                "pv_mean": pv_mean,
                "ml_mean": ml_mean,
                "pv_bias": _circular_diff(pv_mean, target),
                "ml_bias": _circular_diff(ml_mean, target),
            }
        )

    return {
        "pd_true": pd_true,
        "pd_hat": pd_hat,
        "baseline_log_rate": baseline_log_rate,
        "kappa_true": kappa_true,
        "intercept_hat": intercept_hat,
        "coef_hat": coef_hat,
        "n_converged": n_converged,
        "theta_train": theta_train,
        "counts_train": counts_train,
        "theta_test": theta_test,
        "pv_decode": pv_decode,
        "ml_decode": ml_decode,
        "pv_error": pv_error,
        "ml_error": ml_error,
        "per_direction": per_direction,
        "mean_pv_error": float(pv_error.mean()),
        "mean_ml_error": float(ml_error.mean()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_tuning_curves(result: dict):
    """Per-neuron tuning-curve panel: PD-clustering polar histogram plus a
    handful of example neurons' fitted-vs-true tuning curves.
    """
    import matplotlib.pyplot as plt

    pd_true = result["pd_true"]
    n_neurons = pd_true.shape[0]

    # === FIGURE: fig02_tuning_curves.png ===
    fig = plt.figure(figsize=(14.0, 7.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.3, 1.0, 1.0, 1.0])

    ax_hist = fig.add_subplot(gs[:, 0], projection="polar")
    n_bins = 24
    counts, edges = np.histogram(pd_true, bins=n_bins, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]) * 0.9
    ax_hist.bar(centers, counts, width=width, color="tab:purple", alpha=0.75)
    ax_hist.set_title(
        f"Preferred-direction distribution\n({n_neurons} units, "
        f"{CLUSTER_FRACTION:.0%} clustered near "
        f"{np.degrees(CLUSTER_CENTER):.0f}°)",
        fontsize=10,
    )

    theta_fine = np.linspace(-np.pi, np.pi, 200)
    deg_fine = np.degrees(theta_fine)
    example_idx = np.argsort(pd_true)[
        np.linspace(0, n_neurons - 1, 6).round().astype(int)
    ]
    axes_tc = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(1, 4)]
    for panel, i in zip(axes_tc, example_idx):
        true_rate = np.exp(
            result["baseline_log_rate"][i]
            + result["kappa_true"][i] * np.cos(theta_fine - pd_true[i])
        )
        fitted_rate = np.exp(
            result["intercept_hat"][i]
            + result["coef_hat"][i, 0] * np.cos(theta_fine)
            + result["coef_hat"][i, 1] * np.sin(theta_fine)
        )
        train_mask_by_target = result["theta_train"][:, None] == TARGET_ANGLES[None, :]
        emp_rate = np.array(
            [
                result["counts_train"][train_mask_by_target[:, j], i].mean()
                / TRIAL_DURATION_S
                for j in range(N_TARGETS)
            ]
        )
        panel.plot(deg_fine, true_rate, "k--", lw=1.2, label="true")
        panel.plot(deg_fine, fitted_rate, color="tab:blue", lw=1.6, label="GLM fit")
        panel.scatter(
            _wrap_deg(np.degrees(TARGET_ANGLES)), emp_rate, color="tab:red", s=18,
            zorder=3, label="observed (train mean)",
        )
        panel.set_title(
            f"neuron {i}: PD̂={np.degrees(result['pd_hat'][i]):.0f}° "
            f"(true {np.degrees(pd_true[i]):.0f}°)",
            fontsize=9,
        )
        panel.set_xlabel("direction (deg)", fontsize=8)
        panel.set_ylabel("rate (Hz)", fontsize=8)
        panel.tick_params(labelsize=7)
    axes_tc[0].legend(loc="best", fontsize=7)

    fig.suptitle(
        "M1 directional tuning curves: non-uniform preferred-direction "
        "population + per-neuron cosine/von-Mises GLM fit"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_decode_comparison(result: dict):
    """Polar plot of true vs. population-vector vs. ML decoded direction,
    plus a Cartesian true-vs-decoded view with the unity reference line.
    """
    import matplotlib.pyplot as plt

    theta_test = result["theta_test"]
    pv_decode = result["pv_decode"]
    ml_decode = result["ml_decode"]

    # === FIGURE: fig01_pv_vs_ml_decode.png ===
    fig = plt.figure(figsize=(13.0, 6.0))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_cart = fig.add_subplot(1, 2, 2)

    rng_jitter = np.random.default_rng(0)
    r_pv = 0.9 + rng_jitter.uniform(-0.03, 0.03, size=pv_decode.shape[0])
    r_ml = 1.1 + rng_jitter.uniform(-0.03, 0.03, size=ml_decode.shape[0])
    ax_polar.scatter(pv_decode, r_pv, s=8, color="tab:red", alpha=0.18, linewidths=0)
    ax_polar.scatter(ml_decode, r_ml, s=8, color="tab:green", alpha=0.18, linewidths=0)

    for target in TARGET_ANGLES:
        ax_polar.plot([target, target], [0.0, 1.25], color="0.3", lw=1.0, ls=":")
    ax_polar.scatter(
        TARGET_ANGLES, np.full(N_TARGETS, 1.25), marker="^", s=70, color="k",
        zorder=4, label="true target",
    )
    pv_means = np.array([d["pv_mean"] for d in result["per_direction"]])
    ml_means = np.array([d["ml_mean"] for d in result["per_direction"]])
    ax_polar.scatter(
        pv_means, np.full(N_TARGETS, 0.9), s=130, color="tab:red", edgecolor="k",
        zorder=5, label="PV mean decode",
    )
    ax_polar.scatter(
        ml_means, np.full(N_TARGETS, 1.1), s=130, color="tab:green", marker="s",
        edgecolor="k", zorder=5, label="ML mean decode",
    )
    ax_polar.set_ylim(0.0, 1.35)
    ax_polar.set_yticklabels([])
    ax_polar.set_title(
        "True vs. decoded reach direction\n(population vector vs. ML)", fontsize=10
    )
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    true_deg = np.degrees(theta_test)
    pv_display_deg = true_deg + np.degrees(_circular_diff(pv_decode, theta_test))
    ml_display_deg = true_deg + np.degrees(_circular_diff(ml_decode, theta_test))
    ax_cart.scatter(true_deg, pv_display_deg, s=10, color="tab:red", alpha=0.25,
                     label="PV (per trial)")
    ax_cart.scatter(true_deg, ml_display_deg, s=10, color="tab:green", alpha=0.25,
                     label="ML (per trial)")
    lims = (-45.0, 360.0)
    ax_cart.plot(lims, lims, "k--", lw=1.2, label="perfect decode (y=x)")
    ax_cart.set_xlim(*lims)
    ax_cart.set_ylim(*lims)
    ax_cart.set_aspect("equal")
    ax_cart.set_xlabel("true direction (deg)")
    ax_cart.set_ylabel("decoded direction (deg)")
    ax_cart.set_title(
        f"mean |error|: PV={np.degrees(result['mean_pv_error']):.1f}°, "
        f"ML={np.degrees(result['mean_ml_error']):.1f}°",
        fontsize=10,
    )
    ax_cart.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Population-vector bias vs. maximum-likelihood decode under "
        "non-uniform preferred directions"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Demo of population-vector bias vs. ML decode of reach "
            "direction under non-uniform M1 preferred directions"
        )
    )
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument(
        "--export-dir", type=Path,
        default=Path("docs/figures/extras/decoding_place_field"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    print(
        "Population-vector bias vs. ML decode of reach direction "
        "(non-uniform M1 preferred directions)\n"
    )
    result = _run_experiment(seed=args.seed)

    print(
        f"  {N_NEURONS} units, {CLUSTER_FRACTION:.0%} clustered near "
        f"{np.degrees(CLUSTER_CENTER):.0f}° preferred direction "
        f"(Sanger 1996's nonuniform-PD scenario)"
    )
    print(
        f"  tuning-curve GLM fit converged for {result['n_converged']}/"
        f"{N_NEURONS} units"
    )
    pd_fit_error = np.degrees(
        np.abs(_circular_diff(result["pd_hat"], result["pd_true"])).mean()
    )
    print(f"  mean |PD fit error| = {pd_fit_error:.2f}° (recovered vs. true PD)")
    print()
    print(f"  {'target':>8} | {'PV decode':>10} | {'ML decode':>10} | "
          f"{'PV bias':>9} | {'ML bias':>9}")
    for row in result["per_direction"]:
        print(
            f"  {np.degrees(row['target']):7.0f}° | "
            f"{np.degrees(row['pv_mean']):9.1f}° | "
            f"{np.degrees(row['ml_mean']):9.1f}° | "
            f"{np.degrees(row['pv_bias']):8.1f}° | "
            f"{np.degrees(row['ml_bias']):8.1f}°"
        )
    print()
    mean_pv_deg = np.degrees(result["mean_pv_error"])
    mean_ml_deg = np.degrees(result["mean_ml_error"])
    print(f"  mean |PV angular error| = {mean_pv_deg:.2f}°")
    print(f"  mean |ML angular error| = {mean_ml_deg:.2f}°")
    recovery_ok = result["mean_ml_error"] < result["mean_pv_error"]
    print(
        f"  recovery: ML angular error < PV angular error under "
        f"non-uniform preferred directions -> {recovery_ok}"
    )

    if args.export_figures or args.show or not args.no_display:
        fig_decode = _plot_decode_comparison(result)
        fig_tuning = _plot_tuning_curves(result)
        if args.export_figures:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            fig_decode.savefig(
                args.export_dir / "fig01_pv_vs_ml_decode.png", dpi=160
            )
            fig_tuning.savefig(
                args.export_dir / "fig02_tuning_curves.png", dpi=160
            )
            print(f"  saved figures under {args.export_dir}")
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        elif args.no_display:
            import matplotlib.pyplot as plt
            plt.close("all")

    return 0 if recovery_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
