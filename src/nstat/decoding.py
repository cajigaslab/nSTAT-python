"""Decoding algorithms.

This module provides foundational decoding utilities used in example
workflows. The initial implementation prioritizes numerical stability and
reproducibility over highly specialized optimizations.
"""

from __future__ import annotations

import numpy as np


class DecodingAlgorithms:
    """Collection of static decoding methods."""

    @staticmethod
    def compute_spike_rate_cis(spike_matrix: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean spike-rate proxy and pairwise significance matrix.

        Parameters
        ----------
        spike_matrix:
            Binary/count matrix shaped `(n_trials, n_time_bins)`.
        alpha:
            Significance threshold for pairwise trial differences.

        Returns
        -------
        spike_rate:
            Trial-wise average spike rate proxy.
        prob_mat:
            Pairwise absolute rate differences normalized to [0,1].
        sig_mat:
            Binary significance matrix based on empirical threshold.
        """

        data = np.asarray(spike_matrix, dtype=float)
        if data.ndim != 2:
            raise ValueError("spike_matrix must be 2D")

        rate = np.mean(data, axis=1)
        diff = np.abs(rate[:, None] - rate[None, :])
        max_diff = np.max(diff)
        prob_mat = diff / max_diff if max_diff > 0 else diff

        # Simple threshold rule chosen for deterministic, dependency-light behavior.
        # More sophisticated tests can be added without changing external API.
        threshold = np.quantile(diff[np.triu_indices(diff.shape[0], k=1)], 1.0 - alpha) if diff.shape[0] > 1 else 0.0
        sig_mat = (diff >= threshold).astype(int)
        np.fill_diagonal(sig_mat, 0)
        return rate, prob_mat, sig_mat

    @staticmethod
    def decode_weighted_center(spike_counts: np.ndarray, tuning_curves: np.ndarray) -> np.ndarray:
        """Decode latent state via weighted center-of-mass estimator.

        Parameters
        ----------
        spike_counts:
            Shape `(n_units, n_time)`.
        tuning_curves:
            Shape `(n_units, n_states)`.
        """

        counts = np.asarray(spike_counts, dtype=float)
        tuning = np.asarray(tuning_curves, dtype=float)
        if counts.ndim != 2 or tuning.ndim != 2:
            raise ValueError("spike_counts and tuning_curves must be 2D")
        if counts.shape[0] != tuning.shape[0]:
            raise ValueError("unit count must match between counts and tuning curves")

        state_axis = np.arange(tuning.shape[1], dtype=float)
        decoded = np.zeros(counts.shape[1], dtype=float)
        eps = 1e-12
        for t in range(counts.shape[1]):
            weights = counts[:, t][:, None] * tuning
            post = np.sum(weights, axis=0)
            post = post / (np.sum(post) + eps)
            decoded[t] = float(np.sum(post * state_axis))
        return decoded
