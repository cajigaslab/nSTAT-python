"""Decoding demo aligned to MATLAB computeSpikeRateCIs full signature."""

from __future__ import annotations

import numpy as np

from nstat.compat.matlab import DecodingAlgorithms


def main() -> None:
    xK = np.array(
        [
            [0.40, 0.10, -0.20],
            [0.20, -0.10, 0.30],
        ],
        dtype=float,
    )
    num_basis, n_trials = xK.shape
    Wku = np.zeros((num_basis, num_basis, n_trials, n_trials), dtype=float)
    dN = np.array(
        [
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 0],
        ],
        dtype=float,
    )

    t0 = 0.0
    delta = 0.2
    tf = (dN.shape[1] - 1) * delta

    spike_rate_sig, prob_mat, sig_mat = DecodingAlgorithms.computeSpikeRateCIs(
        xK,
        Wku,
        dN,
        t0,
        tf,
        "binomial",
        delta,
        np.array([], dtype=float),
        np.array([], dtype=float),
        40,
        0.05,
    )

    print("Mean spike rates:", np.asarray(spike_rate_sig.dataToMatrix(), dtype=float).reshape(-1).round(6).tolist())
    print("Probability matrix:")
    print(np.asarray(prob_mat, dtype=float).round(6))
    print("Significance matrix:")
    print(np.asarray(sig_mat, dtype=float))


if __name__ == "__main__":
    main()
