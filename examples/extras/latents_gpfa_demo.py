#!/usr/bin/env python3
"""Demo: Gaussian-Process Factor Analysis (Yu et al. 2009) via the Elephant bridge.

End-to-end exercise of :mod:`nstat.extras.latents.gpfa_bridge`:

1. Simulate 4 multi-trial Poisson-spike datasets driven by a shared
   2-D smooth latent (two sinusoids at different frequencies).
2. Fit GPFA with ``x_dim=2`` using :func:`nstat.extras.latents.fit_gpfa`.
3. Print a recovery summary (final log-likelihood, best |Pearson r| of
   any recovered latent vs the ground-truth latents).
4. Optionally plot ``fig01_latent_trajectories.png`` — one subplot per
   trial, two recovered latent lines per subplot.

Run::

    pip install nstat-toolbox[latents]   # pulls Elephant (~50 MB)
    python examples/extras/latents_gpfa_demo.py            # interactive
    python examples/extras/latents_gpfa_demo.py --no-display
    python examples/extras/latents_gpfa_demo.py --export-figures

PNGs from ``--export-figures`` land under ``docs/figures/extras/latents_gpfa/``
and are NOT committed to the repo (per the established policy).

Reference
---------

Yu BM, Cunningham JP, Santhanam G, Ryu SI, Shenoy KV, Sahani M. (2009).
*Gaussian-process factor analysis for low-dimensional single-trial
analysis of neural population activity.*  J. Neurophysiol. 102(1).
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


def _simulate_synthetic_trials(
    *, n_trials: int, n_neurons: int, duration_s: float, seed: int,
):
    """Synthetic Poisson trials driven by two sinusoidal latents.

    Returns ``(neo_trials, true_latents, dt_fine)``.
    """
    import neo
    import quantities as pq

    rng = np.random.default_rng(seed)
    dt_fine = 0.001
    n_steps = int(round(duration_s / dt_fine))
    t = np.arange(n_steps) * dt_fine
    base_freqs = np.array([1.5, 3.0])
    loadings = rng.standard_normal((n_neurons, 2))
    baseline = np.log(20.0) * np.ones(n_neurons)

    neo_trials: list = []
    true_latents: list[np.ndarray] = []
    for _ in range(n_trials):
        phase = rng.uniform(-0.3, 0.3, size=2)
        z = np.stack(
            [
                np.sin(2 * np.pi * base_freqs[0] * t + phase[0]),
                np.sin(2 * np.pi * base_freqs[1] * t + phase[1]),
            ],
            axis=1,
        )
        true_latents.append(z)
        log_rate = z @ loadings.T + baseline
        rate = np.exp(log_rate)
        lam = rate * dt_fine
        spikes = rng.poisson(lam)
        trial_sts: list = []
        for n in range(n_neurons):
            idx = np.nonzero(spikes[:, n])[0]
            counts = spikes[idx, n]
            times: list[float] = []
            for ti, c in zip(idx, counts, strict=False):
                for k in range(int(c)):
                    times.append(t[ti] + (k + 0.5) * (dt_fine / max(c, 1)))
            times_arr = np.asarray(sorted(times), dtype=float)
            trial_sts.append(
                neo.SpikeTrain(
                    times_arr * pq.s,
                    t_start=0.0 * pq.s,
                    t_stop=duration_s * pq.s,
                )
            )
        neo_trials.append(trial_sts)
    return neo_trials, true_latents, dt_fine


def _best_corr_per_trial(
    recovered: list[np.ndarray],
    truth: list[np.ndarray],
    *, dt_fine: float, bin_size_s: float,
) -> list[float]:
    """Best |Pearson r| of any recovered latent vs any true latent, per trial."""
    bin_step = int(round(bin_size_s / dt_fine))
    offset = bin_step // 2
    out: list[float] = []
    for traj, z_true in zip(recovered, truth, strict=False):
        n_bins = traj.shape[0]
        idx = np.clip(np.arange(n_bins) * bin_step + offset, 0, z_true.shape[0] - 1)
        z_binned = z_true[idx]
        best = 0.0
        for i in range(traj.shape[1]):
            for j in range(z_binned.shape[1]):
                r = float(np.corrcoef(traj[:, i], z_binned[:, j])[0, 1])
                if np.isfinite(r):
                    best = max(best, abs(r))
        out.append(best)
    return out


def run_demo(
    *,
    seed: int = 20260616,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool = True,
) -> dict:
    """Run the GPFA recovery demo and return a result dictionary."""
    import matplotlib.pyplot as plt

    from nstat.extras.latents import GPFAConfig, fit_gpfa

    n_trials = 4
    duration_s = 2.0
    bin_size_s = 0.05

    print("=" * 72)
    print("GPFA demo — Elephant bridge (Yu et al. 2009)")
    print("=" * 72)

    neo_trials, true_latents, dt_fine = _simulate_synthetic_trials(
        n_trials=n_trials, n_neurons=6,
        duration_s=duration_s, seed=seed,
    )
    cfg = GPFAConfig(x_dim=2, bin_size_s=bin_size_s, em_max_iter=80)
    result = fit_gpfa(neo_trials, config=cfg, seed=seed)

    corrs = _best_corr_per_trial(
        result.latent_trajectories, true_latents,
        dt_fine=dt_fine, bin_size_s=bin_size_s,
    )
    print()
    print("Recovery summary")
    print(f"  n_trials              : {result.n_trials}")
    print(f"  bin_size_s            : {result.bin_size_s}")
    print(f"  recovered x_dim       : {result.x_dim}")
    print(f"  final log-likelihood  : {result.log_likelihood}")
    print("  best |Pearson r| per trial (recovered vs truth):")
    for k, r in enumerate(corrs):
        print(f"    trial {k}: |r|={r:.3f}")

    fig, axes = plt.subplots(
        1, n_trials, figsize=(3.4 * n_trials, 3.6), sharey=True,
    )
    if n_trials == 1:
        axes = [axes]
    t_bins = (np.arange(result.latent_trajectories[0].shape[0]) + 0.5) * bin_size_s
    for k, ax in enumerate(axes):
        traj = result.latent_trajectories[k]
        for i in range(traj.shape[1]):
            ax.plot(t_bins, traj[:, i], lw=1.6, label=f"latent {i}")
        ax.set_title(f"trial {k}  |r|={corrs[k]:.2f}")
        ax.set_xlabel("time (s)")
        if k == 0:
            ax.set_ylabel("recovered latent")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

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
        "correlations": corrs,
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GPFA demo via the nstat.extras.latents Elephant bridge.",
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
