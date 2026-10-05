#!/usr/bin/env python3
"""Demo: single-trial rotational motor-cortical manifold recovery via GPFA
(Yu et al. 2009 Elephant bridge).

**Question** (design spec S4.4, intracortical motor-BCI thread): do motor
populations occupy a low-dimensional **rotational** neural manifold -- one
shared rotational dynamical system whose oscillation frequency is the
*same* across every reach condition, with conditions differing only in
where they enter that rotation (initial phase/amplitude) -- and can
Gaussian-Process Factor Analysis (GPFA; Yu, Cunningham, Santhanam, Ryu,
Shenoy & Sahani 2009) recover that rotation on **single trials** from
spiking alone, without ever being told the condition label or the
generating equations?

Ground truth
------------
A 2-D **skew-symmetric** rotation generator ``A`` (``A.T == -A``) drives
``dz/dt = A z``, whose solution is a pure rotation ``z(t) = R(omega t)
z0`` at a fixed angular frequency ``omega`` -- the defining signature
Churchland, Cunningham, Kaufman, Foster, Nuyujukian, Ryu & Shenoy (2012)
found in M1/PMd population recordings: once projected onto a shared
low-dimensional subspace, reach-related activity **rotates**, and the
*same* rotational frequency describes every reach condition; only the
initial state (amplitude + phase entering the rotation) is
condition-specific. Six reach conditions are simulated, each with its own
initial amplitude/phase but the identical generator ``A`` (so all six
trace concentric, phase-offset circles in the 2-D latent phase plane).
This latent is projected through a **fixed random loading matrix** into
an ~80-unit synthetic population -- within the 50-100-unit range typical
of a chronic Utah-array recording -- with **condition-locked** per-neuron
baseline-rate offsets (a fixed, condition-specific nuisance shift, layered
underneath the shared rotational drive, standing in for the large
non-rotational condition-tuning component real M1/PMd PSTHs carry
alongside the rotation) and independent Poisson spiking on top.

Recovery
--------
GPFA is fit with ``x_dim=2`` across all trials/conditions pooled (the
condition labels are never given to the fit). Because a factor-analysis
model's latent axes are identified only up to an unknown invertible
linear transform (Yu et al. 2009 Sec. 2), each recovered single-trial
trajectory is realigned onto the true 2-D phase plane by ordinary
least-squares (the *multiple correlation coefficient* this yields is the
rotation-invariant generalization of a raw per-axis Pearson r) before
either scoring or plotting it against the known rotational ground truth.
A second sweep re-runs the full simulate-then-fit pipeline at several
population sizes, confirming that reconstruction fidelity improves as
more neurons are recorded -- exactly the alignment story Gallego, Perich,
Naufel, Ethier, Solla & Miller (2018) and Gallego, Perich, Chowdhury,
Solla & Miller (2020) report for shared low-D manifolds recovered from
larger, more stable cortical populations.

Figure
------
``fig01_latent_trajectories.png``: left panel overlays the ground-truth
rotational phase portrait (one colour per condition, six concentric/
phase-offset circles) with the GPFA-recovered single-trial trajectories
(same colour, realigned onto the true phase plane); right panel plots
mean reconstruction fidelity vs. population size.

References
----------
- Yu BM, Cunningham JP, Santhanam G, Ryu SI, Shenoy KV, Sahani M (2009).
  *Gaussian-process factor analysis for low-dimensional single-trial
  analysis of neural population activity.* J Neurophysiol 102(1):614-635.
- Churchland MM, Cunningham JP, Kaufman MT, Foster JD, Nuyujukian P, Ryu
  SI, Shenoy KV (2012). *Neural population dynamics during reaching.*
  Nature 487(7405):51-56. -- the rotational-dynamics finding this demo's
  ground-truth generator reproduces.
- Gallego JA, Perich MG, Naufel SN, Ethier C, Solla SA, Miller LE (2018).
  *Cortical population activity within a preserved neural manifold
  underlies multiple motor behaviors.* Nat Commun 9(1):4233.
- Gallego JA, Perich MG, Chowdhury RH, Solla SA, Miller LE (2020).
  *Long-term stability of cortical population dynamics underlying
  consistent behavior.* Nat Neurosci 23(2):260-270.

Cross-links
-----------
``examples/extras/decoding_place_field_demo.py`` -- the same M1/PMd
reaching substrate viewed through a per-trial population-vector/ML
direction decode rather than population-level low-dimensional structure.

Run::

    pip install nstat-toolbox[latents]   # pulls Elephant (~50 MB)
    python examples/extras/latents_gpfa_demo.py            # interactive
    python examples/extras/latents_gpfa_demo.py --no-display
    python examples/extras/latents_gpfa_demo.py --export-figures

PNGs from ``--export-figures`` land under ``docs/figures/extras/latents_gpfa/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Ground-truth rotational manifold (Churchland et al. 2012)
# ---------------------------------------------------------------------------
N_CONDITIONS = 6  # reach conditions (a subset of an 8-direction center-out family)
N_TRIALS_PER_CONDITION = 2  # single trials per condition (12 trials total)
N_NEURONS = 80  # main recovery population size (within the Utah-array 50-100 range)
DURATION_S = 1.5
BIN_SIZE_S = 0.05
ROTATION_FREQ_HZ = 2.0  # shared angular frequency (one revolution every 0.5 s)
COND_AMPLITUDE_RANGE = (0.6, 1.4)  # per-condition initial-state radius
COND_VARIABILITY_STD = 0.5  # per-neuron, per-condition log-rate offset std
BASELINE_RATE_HZ = 20.0

# Population-size sweep for the "fidelity improves with population size" panel.
POPULATION_SIZES = (8, 20, 40, 80)


def _condition_initial_states(
    n_conditions: int, rng: np.random.Generator,
) -> np.ndarray:
    """Per-condition initial state ``z0`` -- distinct radius + phase, shared
    rotation generator.  Returns ``(n_conditions, 2)``.
    """
    amplitudes = np.linspace(*COND_AMPLITUDE_RANGE, n_conditions)
    phases = np.linspace(0.0, 2.0 * np.pi, n_conditions, endpoint=False)
    phases = phases + rng.uniform(-0.15, 0.15, size=n_conditions)
    return np.stack(
        [amplitudes * np.cos(phases), amplitudes * np.sin(phases)], axis=1,
    )


def _rotate_trajectory(z0: np.ndarray, omega: float, t: np.ndarray) -> np.ndarray:
    """Analytic solution of ``dz/dt = A z`` for the 2x2 skew-symmetric
    generator: a rotation of ``z0`` at constant angular velocity ``omega``.
    """
    cos_o = np.cos(omega * t)
    sin_o = np.sin(omega * t)
    z1 = cos_o * z0[0] - sin_o * z0[1]
    z2 = sin_o * z0[0] + cos_o * z0[1]
    return np.stack([z1, z2], axis=1)


def _simulate_rotational_population(
    *, n_conditions: int, n_trials_per_condition: int, n_neurons: int,
    duration_s: float, seed: int,
):
    """Synthetic multi-condition Poisson population driven by the shared
    rotational latent.

    Returns ``(neo_trials, true_latents, condition_ids, dt_fine)`` where
    ``true_latents[k]`` is the ``(n_steps, 2)`` fine-resolution rotational
    trajectory for trial ``k``'s condition and ``condition_ids[k]`` is its
    condition index.
    """
    import neo
    import quantities as pq

    rng = np.random.default_rng(seed)
    dt_fine = 0.001
    n_steps = int(round(duration_s / dt_fine))
    t = np.arange(n_steps) * dt_fine
    omega = 2.0 * np.pi * ROTATION_FREQ_HZ

    z0_per_condition = _condition_initial_states(n_conditions, rng)
    # Fixed random loading matrix + baseline: the same population "wiring"
    # is reused across every trial/condition, only the rotational drive
    # and the condition-locked nuisance offset change.
    loadings = rng.standard_normal((n_neurons, 2)) / np.sqrt(2.0)
    baseline = np.log(BASELINE_RATE_HZ) * np.ones(n_neurons)
    # Condition-locked variability: a fixed-per-condition, per-neuron
    # baseline-rate offset -- layered underneath the shared rotational
    # drive, standing in for the large non-rotational condition-tuning
    # component real M1/PMd PSTHs carry.
    cond_offsets = rng.normal(0.0, COND_VARIABILITY_STD, size=(n_conditions, n_neurons))

    neo_trials: list = []
    true_latents: list[np.ndarray] = []
    condition_ids: list[int] = []
    for c in range(n_conditions):
        z = _rotate_trajectory(z0_per_condition[c], omega, t)
        for _ in range(n_trials_per_condition):
            true_latents.append(z)
            condition_ids.append(c)
            log_rate = z @ loadings.T + baseline + cond_offsets[c]
            rate = np.exp(log_rate)
            lam = rate * dt_fine
            spikes = rng.poisson(lam)
            trial_sts: list = []
            for n in range(n_neurons):
                idx = np.nonzero(spikes[:, n])[0]
                counts = spikes[idx, n]
                times: list[float] = []
                for ti, cnt in zip(idx, counts, strict=False):
                    for k in range(int(cnt)):
                        times.append(t[ti] + (k + 0.5) * (dt_fine / max(cnt, 1)))
                times_arr = np.asarray(sorted(times), dtype=float)
                trial_sts.append(
                    neo.SpikeTrain(
                        times_arr * pq.s,
                        t_start=0.0 * pq.s,
                        t_stop=duration_s * pq.s,
                    )
                )
            neo_trials.append(trial_sts)
    return neo_trials, true_latents, condition_ids, dt_fine


def _bin_true_latent(
    z_true: np.ndarray, n_bins: int, *, dt_fine: float, bin_size_s: float,
) -> np.ndarray:
    """Sample the fine-resolution true latent at each GPFA bin center."""
    bin_step = int(round(bin_size_s / dt_fine))
    offset = bin_step // 2
    idx = np.clip(np.arange(n_bins) * bin_step + offset, 0, z_true.shape[0] - 1)
    return z_true[idx]


def _align_recovered_to_truth(
    traj: np.ndarray, z_true_binned: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Realign a recovered ``(n_bins, x_dim)`` GPFA trajectory onto the true
    2-D phase plane via ordinary least squares.

    GPFA's recovered latent axes are only identified up to an unknown
    invertible linear transform (a standard factor-analysis property; Yu
    et al. 2009 Sec. 2) -- so before either plotting the recovered
    trajectory on the same phase-portrait axes as ground truth, or
    scoring recovery quality, we find the linear map (least squares, one
    intercept + one coefficient per recovered dimension) that best
    predicts each true latent dimension from the recovered ones.  The
    resulting *multiple correlation coefficient* (``sqrt(R^2)``)
    generalizes a raw per-axis Pearson r to arbitrary rotations/rescalings
    of the recovered coordinate frame -- essential here since a naive
    axis-by-axis pairing is not a meaningful score for a rotational
    latent (the recovered basis need not line up with the true x/y axes
    even when the full 2-D rotation is recovered near-perfectly).

    Returns
    -------
    z_aligned : (n_bins, 2) ndarray
        Recovered trajectory expressed in true-latent coordinates.
    r_multiple : (2,) ndarray
        Multiple correlation coefficient for each true latent dimension.
    """
    design = np.column_stack([traj, np.ones(traj.shape[0])])
    z_aligned = np.empty_like(z_true_binned)
    r_multiple = np.empty(z_true_binned.shape[1])
    for j in range(z_true_binned.shape[1]):
        y = z_true_binned[:, j]
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        yhat = design @ beta
        z_aligned[:, j] = yhat
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_multiple[j] = np.sqrt(r2)
    return z_aligned, r_multiple


def _reconstruction_fidelity(
    recovered: list[np.ndarray],
    truth: list[np.ndarray],
    *, dt_fine: float, bin_size_s: float,
) -> tuple[list[float], list[np.ndarray]]:
    """Per-trial reconstruction fidelity (mean multiple-correlation across
    the two true latent dimensions) and the corresponding realigned
    trajectories (for plotting).
    """
    fidelity: list[float] = []
    aligned: list[np.ndarray] = []
    for traj, z_true in zip(recovered, truth, strict=False):
        z_true_binned = _bin_true_latent(
            z_true, traj.shape[0], dt_fine=dt_fine, bin_size_s=bin_size_s,
        )
        z_aligned, r_multiple = _align_recovered_to_truth(traj, z_true_binned)
        fidelity.append(float(np.mean(r_multiple)))
        aligned.append(z_aligned)
    return fidelity, aligned


def run_demo(
    *,
    seed: int = 20260616,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool = True,
) -> dict:
    """Run the rotational-manifold GPFA recovery demo and return a result
    dictionary.
    """
    import matplotlib.pyplot as plt

    from nstat.extras.latents import GPFAConfig, fit_gpfa

    print("=" * 72)
    print("GPFA demo — rotational motor-cortical manifold (Churchland et al. 2012)")
    print("=" * 72)

    # ----- Main recovery: pooled fit across all conditions/trials --------
    neo_trials, true_latents, condition_ids, dt_fine = _simulate_rotational_population(
        n_conditions=N_CONDITIONS, n_trials_per_condition=N_TRIALS_PER_CONDITION,
        n_neurons=N_NEURONS, duration_s=DURATION_S, seed=seed,
    )
    cfg = GPFAConfig(x_dim=2, bin_size_s=BIN_SIZE_S, em_max_iter=100)
    result = fit_gpfa(neo_trials, config=cfg, seed=seed)
    fidelity, aligned = _reconstruction_fidelity(
        result.latent_trajectories, true_latents,
        dt_fine=dt_fine, bin_size_s=BIN_SIZE_S,
    )

    print()
    print("Recovery summary")
    print(f"  n_conditions          : {N_CONDITIONS}")
    print(f"  n_trials              : {result.n_trials}")
    print(f"  n_neurons             : {N_NEURONS}")
    print(f"  bin_size_s            : {result.bin_size_s}")
    print(f"  recovered x_dim       : {result.x_dim}")
    print(f"  rotation frequency    : {ROTATION_FREQ_HZ} Hz (shared across all conditions)")
    print(f"  final log-likelihood  : {result.log_likelihood}")
    print("  best |corr| (multiple correlation) per trial, recovered vs. rotational truth:")
    for k, (cond, r) in enumerate(zip(condition_ids, fidelity, strict=False)):
        print(f"    trial {k} (condition {cond}): |r|={r:.3f}")
    print(f"  mean |corr| across all trials: {np.mean(fidelity):.3f}")

    # ----- Reconstruction fidelity vs. population size --------------------
    print()
    print("Reconstruction fidelity vs. population size (same conditions/trials):")
    sweep_results: list[tuple[int, float]] = []
    for n_neurons in POPULATION_SIZES:
        neo_trials_n, true_latents_n, _cond_n, dt_fine_n = _simulate_rotational_population(
            n_conditions=N_CONDITIONS, n_trials_per_condition=N_TRIALS_PER_CONDITION,
            n_neurons=n_neurons, duration_s=DURATION_S, seed=seed,
        )
        cfg_n = GPFAConfig(x_dim=2, bin_size_s=BIN_SIZE_S, em_max_iter=80)
        result_n = fit_gpfa(neo_trials_n, config=cfg_n, seed=seed)
        fidelity_n, _aligned_n = _reconstruction_fidelity(
            result_n.latent_trajectories, true_latents_n,
            dt_fine=dt_fine_n, bin_size_s=BIN_SIZE_S,
        )
        mean_fidelity_n = float(np.mean(fidelity_n))
        sweep_results.append((n_neurons, mean_fidelity_n))
        print(f"    n_neurons={n_neurons:3d}: mean |corr|={mean_fidelity_n:.3f}")

    # === FIGURE: fig01_latent_trajectories.png ===
    fig, (ax_phase, ax_fidelity) = plt.subplots(1, 2, figsize=(12.5, 5.5))

    cmap = plt.get_cmap("tab10")
    plotted_condition: set[int] = set()
    for k, cond in enumerate(condition_ids):
        color = cmap(cond % 10)
        if cond not in plotted_condition:
            z_true = true_latents[k]
            ax_phase.plot(
                z_true[:, 0], z_true[:, 1], color=color, lw=2.2, alpha=0.9,
                label=f"condition {cond} (true)", zorder=2,
            )
            ax_phase.scatter(
                z_true[0, 0], z_true[0, 1], color=color, marker="^", s=60,
                edgecolor="k", zorder=4,
            )
            plotted_condition.add(cond)
        ax_phase.plot(
            aligned[k][:, 0], aligned[k][:, 1], color=color, lw=0.9, ls="-",
            marker="o", markersize=3, alpha=0.7, zorder=3,
        )
    ax_phase.set_xlabel("latent dim 1")
    ax_phase.set_ylabel("latent dim 2")
    ax_phase.set_aspect("equal")
    ax_phase.set_title(
        "Rotational phase portrait: ground truth (solid) vs.\n"
        "GPFA single-trial recovery (dotted), realigned"
    )
    ax_phase.legend(loc="upper right", fontsize=7, ncol=1)

    sweep_n = [n for n, _ in sweep_results]
    sweep_fid = [f for _, f in sweep_results]
    ax_fidelity.plot(sweep_n, sweep_fid, "o-", color="tab:blue", lw=1.8, ms=7)
    ax_fidelity.set_xscale("log")
    ax_fidelity.set_xticks(list(POPULATION_SIZES))
    ax_fidelity.set_xticklabels([str(n) for n in POPULATION_SIZES])
    ax_fidelity.set_ylim(0.0, 1.05)
    ax_fidelity.set_xlabel("population size (n_neurons)")
    ax_fidelity.set_ylabel("mean |corr| (recovered vs. true rotation)")
    ax_fidelity.set_title("Reconstruction fidelity vs. population size")
    ax_fidelity.grid(True, alpha=0.3)

    fig.suptitle(
        "Single-trial rotational motor-cortical manifold recovery via GPFA "
        "(Yu et al. 2009; Churchland et al. 2012)"
    )
    fig.tight_layout()
    # === END FIGURE ===

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras" / "latents_gpfa"
            )
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / "fig01_latent_trajectories.png"
        fig.savefig(path, dpi=180, facecolor="w", edgecolor="none")
        figure_paths.append(path)
        print(f"  Saved: {path}")

    if visible:
        plt.show()
    else:
        plt.close("all")

    return {
        "log_likelihood": result.log_likelihood,
        "fidelity_per_trial": fidelity,
        "mean_fidelity": float(np.mean(fidelity)),
        "population_size_sweep": sweep_results,
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rotational motor-cortical manifold GPFA demo via the "
        "nstat.extras.latents Elephant bridge.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260616,
        help="Random seed for both the simulator and the GPFA fit.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write the latent-trajectory PNG to --export-dir.",
    )
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="Override the PNG export directory.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without showing figures (headless).",
    )
    args = parser.parse_args(argv)

    # Lock matplotlib backend AFTER CLI parsing — never at module top.
    if args.no_display:
        import matplotlib

        matplotlib.use("Agg")
        visible = False
    else:
        visible = bool(args.show)

    try:
        import nstat.extras.latents.gpfa_bridge  # noqa: F401
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    run_demo(
        seed=args.seed,
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        visible=visible,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
