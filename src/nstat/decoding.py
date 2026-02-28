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

    @staticmethod
    def compute_spike_rate_diff_cis(
        spike_matrix_a: np.ndarray, spike_matrix_b: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute trial-wise rate differences and Wald-style confidence intervals.

        Parameters
        ----------
        spike_matrix_a:
            First trial matrix, shape `(n_trials, n_time_bins)`.
        spike_matrix_b:
            Second trial matrix, shape `(n_trials, n_time_bins)`.
        alpha:
            Two-sided confidence level parameter.
        """

        a = np.asarray(spike_matrix_a, dtype=float)
        b = np.asarray(spike_matrix_b, dtype=float)
        if a.shape != b.shape:
            raise ValueError("spike_matrix_a and spike_matrix_b must have matching shape")
        if a.ndim != 2:
            raise ValueError("inputs must be 2D trial matrices")
        if np.any(a < 0.0) or np.any(b < 0.0):
            raise ValueError("spike matrices cannot contain negative counts")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        n_bins = float(a.shape[1])
        rate_a = np.sum(a, axis=1) / n_bins
        rate_b = np.sum(b, axis=1) / n_bins
        diff = rate_a - rate_b

        var = np.clip((rate_a * (1.0 - rate_a) + rate_b * (1.0 - rate_b)) / n_bins, 1e-12, None)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        half = z * np.sqrt(var)
        lo = diff - half
        hi = diff + half
        return diff, lo, hi

    @staticmethod
    def compute_stimulus_cis(
        posterior: np.ndarray, state_values: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Approximate posterior mean and confidence intervals of a decoded stimulus."""

        post = np.asarray(posterior, dtype=float)
        values = np.asarray(state_values, dtype=float)
        if post.ndim != 2:
            raise ValueError("posterior must be 2D (n_states, n_time)")
        if values.ndim != 1 or values.size != post.shape[0]:
            raise ValueError("state_values must match number of states")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        normed = np.clip(post, 1e-15, None)
        normed = normed / np.sum(normed, axis=0, keepdims=True)

        mean = values @ normed
        centered = values[:, None] - mean[None, :]
        var = np.sum((centered**2) * normed, axis=0)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        half = z * np.sqrt(np.clip(var, 0.0, None))
        return mean, mean - half, mean + half

    @staticmethod
    def kalman_predict(
        x_prev: np.ndarray, p_prev: np.ndarray, a: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linear-Gaussian Kalman prediction step."""

        x_prev = np.asarray(x_prev, dtype=float)
        p_prev = np.asarray(p_prev, dtype=float)
        a = np.asarray(a, dtype=float)
        q = np.asarray(q, dtype=float)
        x_pred = a @ x_prev
        p_pred = a @ p_prev @ a.T + q
        return x_pred, p_pred

    @staticmethod
    def kalman_update(
        x_pred: np.ndarray,
        p_pred: np.ndarray,
        y_t: np.ndarray,
        h: np.ndarray,
        r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linear-Gaussian Kalman update step."""

        x_pred = np.asarray(x_pred, dtype=float)
        p_pred = np.asarray(p_pred, dtype=float)
        y_t = np.asarray(y_t, dtype=float)
        h = np.asarray(h, dtype=float)
        r = np.asarray(r, dtype=float)

        innov = y_t - (h @ x_pred)
        s = h @ p_pred @ h.T + r
        k = p_pred @ h.T @ np.linalg.inv(s)
        x_filt = x_pred + k @ innov
        p_filt = p_pred - k @ h @ p_pred
        return x_filt, p_filt

    @staticmethod
    def kalman_filter(
        y: np.ndarray,
        a: np.ndarray,
        h: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run Kalman filtering over all time points."""

        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        n_time = y.shape[0]
        n_state = np.asarray(x0).size

        xf = np.zeros((n_time, n_state), dtype=float)
        pf = np.zeros((n_time, n_state, n_state), dtype=float)
        xp = np.zeros((n_time, n_state), dtype=float)
        pp = np.zeros((n_time, n_state, n_state), dtype=float)

        x_prev = np.asarray(x0, dtype=float)
        p_prev = np.asarray(p0, dtype=float)
        for t in range(n_time):
            x_pred, p_pred = DecodingAlgorithms.kalman_predict(x_prev=x_prev, p_prev=p_prev, a=a, q=q)
            x_filt, p_filt = DecodingAlgorithms.kalman_update(
                x_pred=x_pred, p_pred=p_pred, y_t=y[t], h=h, r=r
            )
            xp[t] = x_pred
            pp[t] = p_pred
            xf[t] = x_filt
            pf[t] = p_filt
            x_prev, p_prev = x_filt, p_filt

        return xf, pf, xp, pp

    @staticmethod
    def kalman_fixed_interval_smoother(
        xf: np.ndarray, pf: np.ndarray, xp: np.ndarray, pp: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rauch-Tung-Striebel fixed-interval smoother."""

        xf = np.asarray(xf, dtype=float)
        pf = np.asarray(pf, dtype=float)
        xp = np.asarray(xp, dtype=float)
        pp = np.asarray(pp, dtype=float)
        a = np.asarray(a, dtype=float)

        n_time, n_state = xf.shape
        xs = np.zeros_like(xf)
        ps = np.zeros_like(pf)
        xs[-1] = xf[-1]
        ps[-1] = pf[-1]

        for t in range(n_time - 2, -1, -1):
            c = pf[t] @ a.T @ np.linalg.inv(pp[t + 1])
            xs[t] = xf[t] + c @ (xs[t + 1] - xp[t + 1])
            ps[t] = pf[t] + c @ (ps[t + 1] - pp[t + 1]) @ c.T

        return xs, ps
