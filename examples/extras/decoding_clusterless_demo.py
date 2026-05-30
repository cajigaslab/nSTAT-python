"""Demo: clusterless point-process decoding + trajectory classification.

Exercises :mod:`nstat.extras.decoding.clusterless_bridge` end-to-end on
a synthetic 1-D back-and-forth trajectory:

1. ``fit_clusterless_decoder``    — single-state continuous decoder
   (clusterless variant of the classic point-process state-space
   decoder, equivalent in spirit to nSTAT's PPAF / PPHF but with marked
   observations).
2. ``fit_clusterless_classifier`` — adds a discrete latent state
   (e.g. *continuous* vs. *fragmented*) on top of the continuous decode.

The bridge wraps `replay_trajectory_classification
<https://github.com/Eden-Kramer-Lab/replay_trajectory_classification>`_
(Denovellis 2021, eLife; MIT).  This Python-only extras module fills
the Tier 2.1 gap in ``parity/methods_roadmap.md``.

Run::

    pip install nstat-toolbox[clusterless]   # pulls JAX (~200 MB)
    python examples/extras/decoding_clusterless_demo.py
"""
from __future__ import annotations

import numpy as np


def _make_synthetic_clusterless(
    n_time: int = 200, n_marks: int = 4, n_electrodes: int = 3, seed: int = 0
):
    """A 1-D back-and-forth trajectory + a sparse Poisson mark cube."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    position = (50.0 + 45.0 * np.sin(2 * np.pi * t / n_time)).reshape(-1, 1)
    multiunits = np.full((n_time, n_marks, n_electrodes), np.nan)
    spike_mask = rng.random(n_time) < 0.3
    for t_i in np.flatnonzero(spike_mask):
        for e in range(n_electrodes):
            if rng.random() < 0.5:
                multiunits[t_i, :, e] = rng.normal(
                    loc=position[t_i, 0] / 20.0, size=n_marks
                )
    return position, multiunits


def _demo_decoder() -> int:
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_decoder

    position, multiunits = _make_synthetic_clusterless()
    result = fit_clusterless_decoder(position, multiunits, place_bin_size=5.0)
    sums = result.posterior.reshape(position.shape[0], -1).sum(axis=1)
    print(f"[Decoder]    posterior shape={result.posterior.shape} "
          f"row-sums in [{sums.min():.4f}, {sums.max():.4f}] (≈1)")
    return 0 if np.all(np.isfinite(result.posterior)) else 1


def _demo_classifier() -> int:
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_classifier

    position, multiunits = _make_synthetic_clusterless(n_time=150)
    result = fit_clusterless_classifier(
        position, multiunits,
        place_bin_size=5.0,
        state_names=["continuous", "fragmented"],
    )
    marginal_means = result.state_probabilities.mean(axis=0)
    print(f"[Classifier] states={result.state_names} "
          f"mean P(state) over time={np.round(marginal_means, 3).tolist()}")
    return 0 if np.allclose(result.state_probabilities.sum(axis=1), 1.0, atol=1e-3) else 1


def main() -> int:
    try:
        import nstat.extras.decoding.clusterless_bridge  # noqa: F401
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    print("nstat.extras.decoding.clusterless_bridge — clusterless decoding demo\n")
    try:
        rc = _demo_decoder()
        rc |= _demo_classifier()
    except ImportError as exc:
        print(f"replay_trajectory_classification missing: {exc}")
        return 1
    print("\nAll clusterless routines ran." if rc == 0
          else "\nA routine reported a problem.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
