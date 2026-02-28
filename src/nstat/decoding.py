"""Decoding algorithms.

This module provides foundational decoding utilities used in example
workflows. The initial implementation prioritizes numerical stability and
reproducibility over highly specialized optimizations.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import norm


class DecodingAlgorithms:
    """Collection of static decoding methods."""

    @staticmethod
    def compute_spike_rate_cis(spike_matrix: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute trial rates, pairwise p-values, and FDR-controlled differences.

        Parameters
        ----------
        spike_matrix:
            Binary/count matrix shaped `(n_trials, n_time_bins)`.
        alpha:
            Significance threshold for pairwise trial differences.

        Returns
        -------
        spike_rate:
            Trial-wise average event rate per bin.
        prob_mat:
            Pairwise two-sided p-value matrix.
        sig_mat:
            Binary significance matrix after Benjamini-Hochberg FDR control.
        """

        data = np.asarray(spike_matrix, dtype=float)
        if data.ndim != 2:
            raise ValueError("spike_matrix must be 2D")
        if data.shape[1] < 2:
            raise ValueError("spike_matrix must have at least two time bins")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if np.any(data < 0.0):
            raise ValueError("spike_matrix cannot contain negative counts")

        n_trials, n_bins = data.shape
        counts = np.sum(data, axis=1)
        rate = counts / float(n_bins)

        pvals = np.ones((n_trials, n_trials), dtype=float)
        upper_idx: list[tuple[int, int]] = []
        upper_pvals: list[float] = []
        for i in range(n_trials):
            for j in range(i + 1, n_trials):
                p1 = rate[i]
                p2 = rate[j]
                pooled = (counts[i] + counts[j]) / (2.0 * n_bins)
                se = np.sqrt(max(pooled * (1.0 - pooled) * (2.0 / n_bins), 0.0))
                if se <= 0.0:
                    p = 1.0 if np.isclose(p1, p2) else 0.0
                else:
                    z = (p1 - p2) / se
                    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
                pvals[i, j] = p
                pvals[j, i] = p
                upper_idx.append((i, j))
                upper_pvals.append(p)

        sig = np.zeros((n_trials, n_trials), dtype=int)
        if upper_pvals:
            pvec = np.asarray(upper_pvals, dtype=float)
            order = np.argsort(pvec)
            sorted_p = pvec[order]
            m = sorted_p.size
            thresholds = alpha * (np.arange(1, m + 1) / m)
            passing = np.where(sorted_p <= thresholds)[0]
            if passing.size > 0:
                cutoff = sorted_p[int(np.max(passing))]
                selected = pvec <= cutoff
                for is_sel, (i, j) in zip(selected, upper_idx, strict=False):
                    if is_sel:
                        sig[i, j] = 1
                        sig[j, i] = 1

        np.fill_diagonal(pvals, 1.0)
        np.fill_diagonal(sig, 0)
        return rate, pvals, sig

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

    @staticmethod
    def decode_state_posterior(
        spike_counts: np.ndarray,
        tuning_rates: np.ndarray,
        transition: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode discrete latent state by point-process Bayes filtering.

        Parameters
        ----------
        spike_counts:
            Non-negative count matrix with shape ``(n_units, n_time)``.
        tuning_rates:
            Expected spike counts per bin with shape ``(n_units, n_states)``.
        transition:
            Optional Markov transition matrix ``(n_states, n_states)`` where
            columns index next-state probabilities.
        prior:
            Optional initial state prior. Defaults to a uniform distribution.
        """

        counts = np.asarray(spike_counts, dtype=float)
        rates = np.asarray(tuning_rates, dtype=float)
        if counts.ndim != 2 or rates.ndim != 2:
            raise ValueError("spike_counts and tuning_rates must be 2D")
        if counts.shape[0] != rates.shape[0]:
            raise ValueError("unit dimension mismatch between spike_counts and tuning_rates")
        if np.any(counts < 0.0):
            raise ValueError("spike_counts must be non-negative")
        if np.any(rates <= 0.0):
            raise ValueError("tuning_rates must be strictly positive")

        n_states = rates.shape[1]
        n_time = counts.shape[1]
        if prior is None:
            prior_vec = np.full(n_states, 1.0 / n_states, dtype=float)
        else:
            prior_vec = np.asarray(prior, dtype=float)
            if prior_vec.shape != (n_states,):
                raise ValueError("prior shape mismatch")
            if np.any(prior_vec < 0.0):
                raise ValueError("prior cannot contain negative values")
            prior_sum = np.sum(prior_vec)
            if prior_sum <= 0.0:
                raise ValueError("prior must have positive mass")
            prior_vec = prior_vec / prior_sum

        if transition is not None:
            transition_mat = np.asarray(transition, dtype=float)
            if transition_mat.shape != (n_states, n_states):
                raise ValueError("transition shape mismatch")
            if np.any(transition_mat < 0.0):
                raise ValueError("transition cannot contain negative values")
            col_sums = np.sum(transition_mat, axis=1, keepdims=True)
            if np.any(col_sums <= 0.0):
                raise ValueError("each transition row must have positive mass")
            transition_mat = transition_mat / col_sums
            log_transition = np.log(np.clip(transition_mat, 1e-12, 1.0))
        else:
            log_transition = None

        rates3 = rates[:, :, None]
        counts3 = counts[:, None, :]
        log_emit = np.sum(
            counts3 * np.log(rates3) - rates3 - gammaln(counts3 + 1.0),
            axis=0,
        )
        log_prior = np.log(np.clip(prior_vec, 1e-12, 1.0))

        log_post = np.zeros((n_states, n_time), dtype=float)
        log_post[:, 0] = log_prior + log_emit[:, 0]
        log_post[:, 0] -= logsumexp(log_post[:, 0])

        for t in range(1, n_time):
            if log_transition is None:
                pred = log_post[:, t - 1]
            else:
                pred = np.array(
                    [
                        logsumexp(log_post[:, t - 1] + log_transition[:, s_next])
                        for s_next in range(n_states)
                    ],
                    dtype=float,
                )
            log_post[:, t] = pred + log_emit[:, t]
            log_post[:, t] -= logsumexp(log_post[:, t])

        posterior = np.exp(log_post)
        decoded_state = np.argmax(posterior, axis=0).astype(int)
        return decoded_state, posterior
