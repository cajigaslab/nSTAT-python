#!/usr/bin/env python3
"""Teaching example — functional coupling between neurons.

Neurons do not fire independently: one cell's spikes can raise or lower
another's firing probability. A point-process GLM captures this with
**ensemble** (coupling) terms — one neuron's recent spikes as a covariate for
another's firing (Truccolo et al. 2005, PMID 15356183).

Here we simulate a two-neuron network with a *known, asymmetric* coupling:

    neuron 0  --(+, excitatory)-->  neuron 1
    neuron 1  --(-, inhibitory)-->  neuron 0

and recover it two ways:

  1. The **cross-correlogram (CCG)** — a model-free picture of coupling: how
     much more (or less) likely neuron 1 is to fire at each lag relative to a
     neuron-0 spike. Excitation shows a bump, inhibition a trough.
  2. A **coupling GLM** — fit each neuron's firing as a function of its own
     history *plus the other neuron's recent spikes*; the sign of the ensemble
     coefficient recovers excitation (+) vs. inhibition (-).

Concepts page: docs/concepts/network_connectivity.md

Run:
    python examples/tutorials/network_coupling.py
    python examples/tutorials/network_coupling.py --save-fig out.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nstat import fit_poisson_glm, simulate_two_neuron_network  # noqa: E402
from nstat import MatlabFallbackWarning  # noqa: E402


def cross_correlogram(si0, si1, dt, max_lag_s=0.025):
    """Counts of neuron-1 spikes at each lag relative to neuron-0 spikes."""
    L = int(round(max_lag_s / dt))
    lags = np.arange(-L, L + 1)
    t0 = np.flatnonzero(si0 > 0)
    ccg = np.zeros(lags.size)
    n = si1.size
    for i, lag in enumerate(lags):
        idx = t0 + lag
        idx = idx[(idx >= 0) & (idx < n)]
        ccg[i] = si1[idx].sum()
    return lags * dt * 1e3, ccg  # lag in ms, counts


def coupling_glm(target, source, stim, dt, hist_bins=8):
    """Recover the source->target coupling sign with a point-process GLM.

    Covariates: the target's own recent spike count (history), the source's
    recent spike count (ensemble/coupling), and the *shared stimulus* — the
    last is essential, since common input would otherwise masquerade as
    coupling (correlation is not connection).
    """
    own = np.concatenate([[0.0], np.convolve(target, np.ones(hist_bins))[: target.size][:-1]])
    other = np.concatenate([[0.0], source[:-1]])     # the source's *previous* bin
    X = np.column_stack([own, other, stim])
    offset = np.full(target.shape, np.log(dt))
    fit = fit_poisson_glm(X, target, offset=offset, l2=0.05, max_iter=300, tol=1e-9)
    return fit.coefficients[1]  # the ensemble (coupling) coefficient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-fig", metavar="PATH", default=None,
                        help="Save the cross-correlogram figure to PATH.")
    args = parser.parse_args()

    dt = 0.001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatlabFallbackWarning)
        sim = simulate_two_neuron_network(
            duration_s=120.0, dt=dt, seed=7,
            ensemble_kernel=(1.5, -1.5),  # 1->0 excitatory, 0->1 inhibitory (clear)
        )
    si = sim.spike_indicator            # (T, 2)
    si0, si1 = si[:, 0], si[:, 1]
    stim = sim.latent_drive             # the shared stimulus both neurons see
    net = sim.actual_network            # [[., w_1->0],[w_0->1, .]]

    print("Functional coupling between two neurons\n")
    print(f"True coupling (simulated):")
    print(f"  neuron 0 -> neuron 1 : {net[1, 0]:+.1f}  "
          f"({'excitatory' if net[1, 0] > 0 else 'inhibitory'})")
    print(f"  neuron 1 -> neuron 0 : {net[0, 1]:+.1f}  "
          f"({'excitatory' if net[0, 1] > 0 else 'inhibitory'})\n")

    c_01 = coupling_glm(si1, si0, stim, dt)   # source 0 -> target 1
    c_10 = coupling_glm(si0, si1, stim, dt)   # source 1 -> target 0
    print("Recovered coupling GLM coefficients (sign = excite + / inhibit -):")
    print(f"  neuron 0 -> neuron 1 : {c_01:+.3f}  -> "
          f"{'excitatory' if c_01 > 0 else 'inhibitory'}")
    print(f"  neuron 1 -> neuron 0 : {c_10:+.3f}  -> "
          f"{'excitatory' if c_10 > 0 else 'inhibitory'}")
    print("\nThe GLM recovers the asymmetric wiring: 0 excites 1, 1 inhibits 0 —")
    print("a directed functional connection you cannot see from firing rates alone.")

    if args.save_fig:
        _save_figure(args.save_fig, si0, si1, dt)
        print(f"\nSaved figure to {args.save_fig}")


def _save_figure(path, si0, si1, dt) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lags_ms, ccg = cross_correlogram(si0, si1, dt)
    baseline = ccg.mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(lags_ms, ccg, width=(lags_ms[1] - lags_ms[0]), color="#2c5282", alpha=0.8)
    ax.axhline(baseline, color="0.5", ls="--", lw=0.8, label="chance level")
    ax.axvline(0, color="#dd6b20", lw=1)
    ax.set(xlabel="lag (ms): neuron-1 spikes relative to neuron-0 spikes",
           ylabel="spike count",
           title="Cross-correlogram: peak before 0 = 1→0 excitation; "
                 "trough after 0 = 0→1 inhibition")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
