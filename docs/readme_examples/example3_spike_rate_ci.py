"""README Example 3: spike-rate confidence intervals and significance map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import DecodingAlgorithms


def build_inputs(num_basis: int, num_trials: int, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_idx = np.arange(1, num_basis + 1, dtype=float)[:, None]
    trial_idx = np.arange(1, num_trials + 1, dtype=float)[None, :]
    xk = 0.06 * np.sin(0.37 * basis_idx * trial_idx) + 0.04 * np.cos(0.19 * basis_idx * trial_idx)

    wku = np.zeros((num_basis, num_basis, num_trials, num_trials), dtype=float)
    for r in range(num_basis):
        wku[r, r, :, :] = 0.05 * np.eye(num_trials, dtype=float)

    grid = np.arange(num_trials * n_bins, dtype=float).reshape(num_trials, n_bins)
    d_n = ((np.sin(0.173 * grid) + np.cos(0.037 * grid)) > 1.15).astype(float)
    return xk, wku, d_n


def main() -> None:
    np.random.seed(0)

    num_basis = 5
    num_trials = 6
    n_bins = 160
    delta = 0.01
    t0 = 0.0
    tf = (n_bins - 1) * delta
    mc = 40

    xk, wku, d_n = build_inputs(num_basis=num_basis, num_trials=num_trials, n_bins=n_bins)

    _, prob, sig = DecodingAlgorithms.computeSpikeRateCIs(
        xk,
        wku,
        d_n,
        t0,
        tf,
        "binomial",
        delta,
        0.0,
        [],
        mc,
        0.05,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))
    im1 = ax1.imshow(prob, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    ax1.set_title("Probability Matrix")
    ax1.set_xlabel("trial m")
    ax1.set_ylabel("trial k")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("P(rate_m > rate_k)")

    im2 = ax2.imshow(sig, aspect="auto", origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
    ax2.set_title("Significance Mask")
    ax2.set_xlabel("trial m")
    ax2.set_ylabel("trial k")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("significant")
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "images" / "readme_example3_spike_rate_ci.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
