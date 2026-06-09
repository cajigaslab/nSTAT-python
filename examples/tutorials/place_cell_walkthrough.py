#!/usr/bin/env python3
"""Capstone — a complete analysis on REAL hippocampal place-cell data.

The other tutorials in this folder each isolate one idea on *simulated* data.
This one runs the whole arc end-to-end on a **real recording**: 49 hippocampal
place cells from a rat foraging in an open field for ~24 minutes
(``place_cell_animal1``; Brown et al. 1998, PMID 9736661). The story has four
acts, the same shape as a real analysis:

    LOAD ───► ENCODE ───► CHECK (goodness-of-fit) ───► DECODE
    real      place-       does the model actually    read the animal's
    spikes    field GLM    describe the spiking?       position back out

The lesson is deliberately honest. The place-field model is good enough to
**decode the animal's position several times better than chance** — yet it
**fails the time-rescaling goodness-of-fit test** for almost every cell. A
model can be *useful* and still be *statistically incomplete*: real place cells
also carry spike history, the theta rhythm, and finer spatial structure than a
single Gaussian bump. Goodness-of-fit is what tells you so, and points to the
fixes (history terms → ``spike_trains_and_glms.md``; a richer spatial basis →
Paper Example 04's Zernike fields).

Encoding model (per cell c), a Gaussian place field as a log-linear GLM:

    log λ_c(t) = β0 + β1·x + β2·y + β3·x² + β4·y² + β5·xy   (+ log Δ offset)

Decoder (memoryless Bayesian population reconstruction; Zhang et al. 1998,
PMID 9463459): for each time bin, the most likely position on a grid under the
Poisson place-field likelihood of the observed population spike counts. This is
the static cousin of the PPAF in ``decoding_ppaf.py`` — the PPAF adds a
dynamical prior (the animal moves smoothly) on top of exactly this likelihood.

Concepts pages: docs/concepts/goodness_of_fit_and_decoding.md ·
                docs/concepts/spike_trains_and_glms.md
References (verified against PubMed; see docs/concepts/bibliography.md):
  Brown et al. 1998, J Neurosci 18:7411      PMID 9736661
  Zhang et al. 1998, J Neurophysiol 79:1017  PMID 9463459
  Truccolo et al. 2005, J Neurophysiol 93:1074  PMID 15356183

Data: ``place_cell_animal1`` — downloaded on first run via nSTAT's data
manager (``ensure_example_data``); see docs/data_installation.rst.

Run:
    python examples/tutorials/place_cell_walkthrough.py
    python examples/tutorials/place_cell_walkthrough.py --save-fig out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nstat import (  # noqa: E402
    get_dataset_path, fit_poisson_glm, population_time_rescale,
)


# ---------------------------------------------------------------------------
# Act 1 — Load the real recording
# ---------------------------------------------------------------------------
def load_place_cells():
    """Return (x, y, t, dt, spike_counts) for the foraging recording.

    `spike_counts` is a (T, C) matrix: spikes of each of C cells in each of the
    T position frames (the camera sampled position at ~30 Hz).
    """
    path = get_dataset_path("place_cell_animal1")   # downloads on first call
    d = loadmat(str(path), squeeze_me=True)
    x = np.asarray(d["x"], dtype=float).ravel()
    y = np.asarray(d["y"], dtype=float).ravel()
    t = np.asarray(d["time"], dtype=float).ravel()
    neurons = np.asarray(d["neuron"], dtype=object).ravel()
    dt = float(np.median(np.diff(t)))

    edges = np.concatenate([t, [t[-1] + dt]])
    counts = np.zeros((t.size, neurons.size))
    for c in range(neurons.size):
        st = np.asarray(neurons[c]["spikeTimes"].item(), dtype=float).ravel()
        counts[:, c] = np.histogram(st, bins=edges)[0]
    return x, y, t, dt, counts


# ---------------------------------------------------------------------------
# Act 2 — Encode: fit a place-field GLM to each cell
# ---------------------------------------------------------------------------
def fit_place_fields(x, y, dt, counts, min_spikes=50):
    """Fit a Gaussian place field per (sufficiently active) cell.

    Returns the active-cell indices and, for each, the fitted per-bin intensity
    over time (`lam_t`, T×K) and the GLM coefficient vectors (`coefs`).
    """
    X = np.column_stack([x, y, x ** 2, y ** 2, x * y])
    offset = np.full(x.size, np.log(dt))
    active = np.flatnonzero(counts.sum(axis=0) >= min_spikes)

    lam_t = np.zeros((x.size, active.size))
    coefs = []
    for k, c in enumerate(active):
        fit = fit_poisson_glm(X, counts[:, c], offset=offset,
                              l2=1e-4, max_iter=100, tol=1e-9)
        lam_t[:, k] = np.exp(fit.intercept + X @ fit.coefficients + offset)
        coefs.append((fit.intercept, fit.coefficients))
    return active, lam_t, coefs


# ---------------------------------------------------------------------------
# Act 3 — Check: time-rescaling goodness-of-fit, cell by cell
# ---------------------------------------------------------------------------
def goodness_of_fit(active, counts, lam_t):
    """Per-cell time-rescaling KS test; return the fraction that pass."""
    passes = 0
    for k, c in enumerate(active):
        res = population_time_rescale([counts[:, c]], [lam_t[:, k]])
        if res.ground_ks_pvalue > 0.05:
            passes += 1
    return passes, active.size


# ---------------------------------------------------------------------------
# Act 4 — Decode: Bayesian population reconstruction of position
# ---------------------------------------------------------------------------
def decode_position(x, y, dt, counts, active, coefs, grid=32, window=8):
    """Reconstruct (x, y) over time from population spikes on a position grid.

    For each time bin we sum spikes over a short `window` of frames (~250 ms, as
    is standard for place-cell decoding) and pick the grid position that
    maximises the Poisson population log-likelihood
        Σ_c [ n_c · log λ_c(pos) − λ_c(pos) ].
    """
    gx = np.linspace(x.min(), x.max(), grid)
    gy = np.linspace(y.min(), y.max(), grid)
    GX, GY = np.meshgrid(gx, gy)
    Xg = np.column_stack([GX.ravel(), GY.ravel(), GX.ravel() ** 2,
                          GY.ravel() ** 2, GX.ravel() * GY.ravel()])

    # Place-field intensity (per bin) of each cell at every grid point.
    lam_grid = np.array([np.exp(b0 + Xg @ b + np.log(dt)) for b0, b in coefs])

    # Spike counts summed over a trailing window of frames.
    Na = counts[:, active]
    csum = np.cumsum(np.vstack([np.zeros((1, active.size)), Na]), axis=0)
    Nw = np.empty_like(Na)
    for i in range(x.size):
        Nw[i] = csum[i + 1] - csum[max(0, i - window + 1)]

    loglam = np.log(lam_grid + 1e-12)        # (K, G²)
    lamsum = window * lam_grid.sum(axis=0)   # expected total count per grid pt
    dec = np.empty((x.size, 2))
    for s in range(0, x.size, 4000):          # chunk to bound memory
        e = min(x.size, s + 4000)
        ll = Nw[s:e] @ loglam - lamsum[None, :]
        idx = np.argmax(ll, axis=1)
        dec[s:e, 0] = GX.ravel()[idx]
        dec[s:e, 1] = GY.ravel()[idx]

    err = np.hypot(dec[:, 0] - x, dec[:, 1] - y)
    perm = np.random.default_rng(0).permutation(x.size)
    chance = np.hypot(x[perm] - x, y[perm] - y)
    return dec, float(np.median(err)), float(np.median(chance))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-fig", metavar="PATH", default=None,
                        help="Save a place-field / KS / decode figure (needs matplotlib).")
    args = parser.parse_args()

    print("Capstone: a full analysis on real hippocampal place cells\n")
    x, y, t, dt, counts = load_place_cells()
    print(f"Act 1  loaded {counts.shape[1]} cells, {t.size} position frames "
          f"({t[-1] - t[0]:.0f} s at {1 / dt:.0f} Hz)")

    active, lam_t, coefs = fit_place_fields(x, y, dt, counts)
    print(f"Act 2  fit place-field GLMs for {active.size} active cells")

    n_pass, n_tot = goodness_of_fit(active, counts, lam_t)
    print(f"Act 3  time-rescaling KS: {n_pass}/{n_tot} cells pass (p>0.05)")

    dec, med_err, chance = decode_position(x, y, dt, counts, active, coefs)
    print(f"Act 4  decoded position — median error {med_err:.3f} vs "
          f"chance {chance:.3f} (arena width ~{x.max() - x.min():.2f})")

    print("\nLesson: the place-field model decodes position several times")
    print("better than chance, yet fails goodness-of-fit for almost every")
    print("cell. Useful is not the same as correct. Goodness-of-fit reveals")
    print("what spatial tuning alone misses — spike history, theta, and finer")
    print("place-field shape (see spike_trains_and_glms.md and Paper Example 04).")

    if args.save_fig:
        _save_figure(args.save_fig, x, y, dt, counts, active, lam_t, coefs,
                     dec, med_err, n_pass, n_tot)
        print(f"\nSaved figure to {args.save_fig}")


def _save_figure(path, x, y, dt, counts, active, lam_t, coefs, dec, med_err,
                 n_pass, n_tot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # Panel 1: the strongest place field, shown only where the animal went.
    # (The global quadratic basis explodes in the unvisited corners — exactly
    # the artifact that motivates a local basis like Paper Example 04's Zernike
    # fields — so we mask grid cells the animal never occupied.)
    k_best = int(np.argmax(lam_t.max(axis=0)))
    gx = np.linspace(x.min(), x.max(), 60)
    gy = np.linspace(y.min(), y.max(), 60)
    GX, GY = np.meshgrid(gx, gy)
    Xg = np.column_stack([GX.ravel(), GY.ravel(), GX.ravel() ** 2,
                          GY.ravel() ** 2, GX.ravel() * GY.ravel()])
    b0, b = coefs[k_best]
    field = np.exp(b0 + Xg @ b).reshape(GX.shape)   # spikes/s (offset already binned)
    occ, _, _ = np.histogram2d(x, y, bins=[gx.size, gy.size],
                               range=[[x.min(), x.max()], [y.min(), y.max()]])
    field = np.where(occ.T > 0, field, np.nan)
    ax = axes[0]
    vmax = np.nanpercentile(field, 99)
    pcm = ax.pcolormesh(GX, GY, field, shading="auto", cmap="viridis", vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="firing rate (spikes/s)")
    ax.set(title=f"Fitted place field (cell {active[k_best] + 1})",
           xlabel="x position", ylabel="y position", aspect="equal")

    # Panel 2: KS plot for every cell (most fall outside the band).
    ax = axes[1]
    for k in range(active.size):
        res = population_time_rescale([counts[:, active[k]]], [lam_t[:, k]])
        u = np.sort(np.asarray(res.ground_uniforms))
        n = u.size
        if n < 5:
            continue
        emp = (np.arange(1, n + 1) - 0.5) / n
        ax.plot(u, emp, color="#2c5282", lw=0.5, alpha=0.35)
    n_band = max(int(np.median([counts[:, c].sum() for c in active])), 1)
    band = 1.36 / np.sqrt(n_band)
    xx = np.linspace(0, 1, 100)
    ax.plot([0, 1], [0, 1], color="0.3", lw=1)
    ax.plot(xx, np.clip(xx + band, 0, 1), color="#e53e3e", ls="--", lw=0.9)
    ax.plot(xx, np.clip(xx - band, 0, 1), color="#e53e3e", ls="--", lw=0.9)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel="model quantile (rescaled)", ylabel="empirical quantile",
           title=f"Goodness-of-fit: {n_pass}/{n_tot} cells pass")

    # Panel 3: decoded vs true trajectory over a 30-second window.
    ax = axes[2]
    fs = 1.0 / dt
    sl = slice(0, int(30 * fs))
    ax.plot(x[sl], y[sl], color="#dd6b20", lw=2.0, label="true path")
    ax.plot(dec[sl, 0], dec[sl, 1], color="#2c5282", lw=0.7, alpha=0.7,
            label="Bayesian decode")
    ax.set(title=f"Decoded position (median err {med_err:.2f})",
           xlabel="x position", ylabel="y position", aspect="equal")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
