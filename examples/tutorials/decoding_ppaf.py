#!/usr/bin/env python3
"""Teaching example — decoding a stimulus from a neural population (PPAF).

The companion to ``encoding_to_goodness_of_fit.py``. There we did *encoding*
(stimulus → spikes) and checked the fit. Here we do the inverse — **decoding**:
given only the spikes of a population, reconstruct the hidden stimulus that
drove them. This is the math behind brain–machine interfaces.

The tool is the **point-process adaptive filter (PPAF)**, the spiking analogue
of the Kalman filter (Eden et al. 2004, PMID 15070506). At each time step it:

  1. PREDICTS the state forward using a simple dynamics model, then
  2. CORRECTS that prediction using the spikes just observed, weighting each
     neuron by its tuning (its fitted CIF).

Because spikes are all-or-none point events (not Gaussian measurements), the
correction uses the point-process likelihood rather than the Kalman gain — but
the predict/correct rhythm is identical.

We also illustrate a key lesson: **more neurons → a better decode.** A single
neuron is ambiguous; a population pins the stimulus down.

Concepts page: docs/concepts/goodness_of_fit_and_decoding.md
References (verified against PubMed; see docs/concepts/bibliography.md):
  Eden et al. 2004, Neural Comput 16:971   PMID 15070506
  Truccolo et al. 2005, J Neurophysiol 93:1074  PMID 15356183

Run:
    python examples/tutorials/decoding_ppaf.py
    python examples/tutorials/decoding_ppaf.py --save-fig out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nstat import DecodingAlgorithms  # noqa: E402


def encode_population(n_cells, time, stim, rng, delta):
    """Simulate a population whose cells are each (randomly) tuned to `stim`.

    Each cell c has a logistic CIF: p_c(t) = sigmoid(b0_c + b1_c * stim(t)).
    b1_c is the cell's tuning (how strongly/which direction the stimulus moves
    its firing); b0_c sets its baseline rate.  Returns the spike matrix dN
    (n_cells x T) plus the encoding parameters the decoder will use.
    """
    T = time.size
    b1 = rng.standard_normal(n_cells)                       # tuning per cell
    b0 = np.log(10.0 * delta) + rng.standard_normal(n_cells)  # baseline per cell
    dN = np.zeros((n_cells, T))
    for c in range(n_cells):
        eta = np.clip(b0[c] + b1[c] * stim, -20.0, 20.0)
        p = np.exp(eta) / (1.0 + np.exp(eta))
        dN[c, :] = (rng.random(T) < p).astype(float)
    return dN, b0, b1


def decode(dN, b0, b1, stim, delta):
    """Run the PPAF and return the decoded stimulus, its 95% band, and RMSE."""
    # State-space dynamics for the latent stimulus: a slow random walk
    #   x(t+1) = A x(t) + noise,   here A = 1 (persistence).
    A = np.array([[1.0]])
    Q = np.array([[float(np.std(np.diff(stim)))]])   # process noise ~ stim wiggle
    x0 = np.array([0.0])
    Pi0 = 0.5 * np.eye(1)
    beta = b1.reshape(1, -1)                          # (1 state, C cells)

    # PPAF: predict/correct over the whole recording.
    _, _, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilterLinear(
        A, Q, dN, b0, beta, "binomial", delta, None, None, x0, Pi0
    )
    x_hat = x_u[0, :]
    sigma = np.sqrt(np.maximum(W_u[0, 0, :], 0.0))
    rmse = float(np.sqrt(np.mean((x_hat - stim) ** 2)))
    return x_hat, sigma, rmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-fig", metavar="PATH", default=None,
                        help="Save a decoded-vs-true figure to PATH (needs matplotlib).")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    delta = 0.001
    time = np.arange(0.0, 1.0 + delta, delta)
    stim = np.sin(2.0 * np.pi * 2.0 * time)     # the hidden 1-D stimulus

    print("Decoding a hidden stimulus from population spikes (PPAF)\n")
    print(f"{'# cells':>8}   {'decode RMSE':>12}")
    results = {}
    for n_cells in (1, 5, 20, 60):
        dN, b0, b1 = encode_population(n_cells, time, stim, rng, delta)
        x_hat, sigma, rmse = decode(dN, b0, b1, stim, delta)
        results[n_cells] = (x_hat, sigma, rmse)
        print(f"{n_cells:>8}   {rmse:>12.3f}")

    print("\nLesson: RMSE falls as the population grows — a single cell is")
    print("ambiguous, but many cells with different tuning jointly pin down")
    print("the stimulus. This is why decoding is done from populations.")

    if args.save_fig:
        _save_figure(args.save_fig, time, stim, results)
        print(f"\nSaved figure to {args.save_fig}")


def _save_figure(path, time, stim, results) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = 20
    x_hat, sigma, rmse = results[n_show]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(time, stim, color="#dd6b20", lw=2, label="true stimulus")
    ax.plot(time, x_hat, color="#2c5282", lw=1.5, label=f"PPAF decode ({n_show} cells)")
    ax.fill_between(time, x_hat - 1.96 * sigma, x_hat + 1.96 * sigma,
                    color="#2c5282", alpha=0.2, label="95% credible band")
    ax.set(xlabel="time (s)", ylabel="stimulus", xlim=(time[0], time[-1]),
           title=f"Decoding the hidden stimulus from spikes (RMSE={rmse:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
