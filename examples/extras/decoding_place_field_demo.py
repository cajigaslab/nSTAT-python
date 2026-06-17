"""Demo: 2-D place-field encoding + PPAF decoding via ``fit_place_field_decoder``.

Mirrors the canonical example08 pipeline (B-spline Poisson encoder +
quadratic-CIF refit + PPDecodeFilterLinear) on a fully-synthetic
3-cell place-cell trial — no figshare data, no opt-deps beyond core
``nstat``.

Run::

    python examples/extras/decoding_place_field_demo.py
    python examples/extras/decoding_place_field_demo.py --export-figures
    python examples/extras/decoding_place_field_demo.py --no-display
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _ou_walk(n: int, dt: float, *, seed: int = 0,
             theta: float = 0.3, sigma: float = 0.25) -> np.ndarray:
    """Mean-reverting random walk in [0.05, 0.95]."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    x[0] = 0.5
    sqrt_dt = np.sqrt(dt)
    for i in range(n - 1):
        x[i + 1] = (
            x[i]
            + theta * (0.5 - x[i]) * dt
            + sigma * sqrt_dt * rng.standard_normal()
        )
    return np.clip(x, 0.05, 0.95)


def _build_synthetic_trial(seed: int = 20260617):
    from nstat import (
        Covariate,
        CovariateCollection,
        SpikeTrainCollection,
        Trial,
        nspikeTrain,
    )

    rng = np.random.default_rng(seed)
    duration = 60.0
    fs = 50.0
    n_time = int(duration * fs)
    t = np.arange(n_time, dtype=float) / fs
    x_pos = _ou_walk(n_time, 1.0 / fs, seed=seed + 1)
    y_pos = _ou_walk(n_time, 1.0 / fs, seed=seed + 2)

    bin_width = 0.020
    n_bins = int(duration / bin_width)
    bin_centres = (np.arange(n_bins) + 0.5) * bin_width
    x_bin = np.interp(bin_centres, t, x_pos)
    y_bin = np.interp(bin_centres, t, y_pos)

    cell_centres = [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)]
    sigma_pf = 0.15
    peak = 20.0
    nstrains = []
    for cx, cy in cell_centres:
        rates = peak * np.exp(
            -((x_bin - cx) ** 2 + (y_bin - cy) ** 2) / (2.0 * sigma_pf ** 2)
        )
        counts = rng.poisson(rates * bin_width)
        spike_times: list[float] = []
        for k, c in enumerate(counts):
            for _ in range(int(c)):
                spike_times.append(
                    float(bin_centres[k] + rng.uniform(-bin_width / 2, bin_width / 2))
                )
        nstrains.append(
            nspikeTrain(np.sort(np.asarray(spike_times, dtype=float)),
                        minTime=0.0, maxTime=duration)
        )

    spikes = SpikeTrainCollection(nstrains)
    x_cov = Covariate(t, x_pos, "x_pos", "time", "s", "m", ["x"])
    y_cov = Covariate(t, y_pos, "y_pos", "time", "s", "m", ["y"])
    trial = Trial(
        spike_collection=spikes,
        covariate_collection=CovariateCollection([x_cov, y_cov]),
    )
    # Trial constructs covariates at its own sample grid (the spike-train
    # default), which differs from the 50 Hz array we passed in. Read the
    # covariates back so the position array is aligned to that grid.
    position = np.column_stack([
        np.asarray(trial.covarColl.getCov(i).data, dtype=float).reshape(-1)
        for i in range(2)
    ])
    return trial, position, cell_centres


def _plot_decoded_trajectory(position, result):
    import matplotlib.pyplot as plt

    n_bins = result.decoded_position.shape[0]
    trial_time = np.linspace(0.0, n_bins * result.bin_width_s, n_bins)
    # Interpolate true position onto the decoding lattice for overlay.
    orig_t = np.linspace(0.0, trial_time[-1], position.shape[0])
    true_x = np.interp(trial_time, orig_t, position[:, 0])
    true_y = np.interp(trial_time, orig_t, position[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].plot(true_x, true_y, "k-", lw=1.0, alpha=0.7, label="true")
    axes[0].plot(
        result.decoded_position[:, 0], result.decoded_position[:, 1],
        color="tab:blue", lw=1.0, alpha=0.7, label="decoded",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    axes[0].set_title("Decoded trajectory (PPAF, nonlinear CIF branch)")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(trial_time, true_x, "k-", lw=1.0, label="true x")
    axes[1].plot(
        trial_time, result.decoded_position[:, 0],
        color="tab:blue", lw=1.0, label="decoded x",
    )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("x")
    axes[1].set_title("x(t) trace")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def _plot_place_field_summary(result, cell_centres):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for c_idx, kept in enumerate(result.cell_indices_kept):
        cx, cy = cell_centres[c_idx % len(cell_centres)]
        ax.scatter([cx], [cy], s=160, marker="x", color="tab:red",
                   label="true field centres" if c_idx == 0 else None)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("True place-field centres (synthetic)")
    if result.cell_indices_kept:
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo of nstat.extras.decoding.fit_place_field_decoder"
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

    from nstat.extras.decoding import (
        PlaceFieldDecoderConfig,
        fit_place_field_decoder,
    )

    print("nstat.extras.decoding.place_field_decoder — synthetic 3-cell demo\n")
    trial, position, cell_centres = _build_synthetic_trial(seed=args.seed)
    print(
        f"  trial: minTime={trial.minTime:.2f}s, maxTime={trial.maxTime:.2f}s, "
        f"sampleRate={trial.sampleRate} Hz"
    )

    cfg = PlaceFieldDecoderConfig(decode_filter="nonlinear")
    result = fit_place_field_decoder(trial, position, config=cfg)
    print(
        f"  kept {len(result.cell_indices_kept)} cells "
        f"(skipped {len(result.cell_indices_skipped)})"
    )
    print(f"  mean decoding error = {result.mean_decoding_error:.4f}")

    if args.export_figures or args.show or not args.no_display:
        fig_traj = _plot_decoded_trajectory(position, result)
        fig_fields = _plot_place_field_summary(result, cell_centres)
        if args.export_figures:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            fig_traj.savefig(args.export_dir / "fig01_decoded_trajectory.png",
                             dpi=160)
            fig_fields.savefig(args.export_dir / "fig02_place_fields.png",
                               dpi=160)
            print(f"  saved figures under {args.export_dir}")
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        elif args.no_display:
            import matplotlib.pyplot as plt
            plt.close("all")

    return 0 if np.isfinite(result.mean_decoding_error) else 1


if __name__ == "__main__":
    raise SystemExit(main())
