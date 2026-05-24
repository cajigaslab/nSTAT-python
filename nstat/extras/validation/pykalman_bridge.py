"""Cross-validate :meth:`nstat.DecodingAlgorithms.kalman_filter` against pykalman.

pykalman (https://github.com/pykalman/pykalman) is a BSD-3 pure-NumPy
Kalman implementation, community-resurrected in 2026 (v0.11.2, Jan
2026; 1.3k stars).  Because it's NumPy-only (no JAX), it's the
*lowest-friction* cross-validation reference for the Gaussian-only
Kalman path — perfect for CI parity tests.

This bridge addresses AUDIT D3 (the known smoother-index approximation
gap in :meth:`DecodingAlgorithms.kalman_fixedIntervalSmoother`) by
providing an independent reference implementation users can call from
the same script that uses nstat's Kalman primitives.

Install:
    pip install nstat-toolbox[test-parity]   # or just: pip install pykalman
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nstat import DecodingAlgorithms

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


_IMPORT_ERROR_MSG = (
    "nstat.extras.validation.pykalman_bridge requires the 'pykalman' package. "
    "Install with: pip install nstat-toolbox[test-parity]"
)


def _require_pykalman():
    try:
        from pykalman import KalmanFilter as PyKalmanFilter
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    return PyKalmanFilter


@dataclass(frozen=True)
class KalmanComparison:
    """Side-by-side Kalman filter / smoother fit comparison.

    Attributes
    ----------
    nstat_filtered_means
        Filtered state means from :meth:`DecodingAlgorithms.kalman_filter`,
        shape ``(T, Dx)``.
    pykalman_filtered_means
        Filtered state means from ``pykalman.KalmanFilter.filter()``,
        same shape and layout.
    nstat_smoothed_means
        Smoothed state means from :meth:`DecodingAlgorithms.kalman_fixedIntervalSmoother`
        if computed (else ``None``), shape ``(T, Dx)``.
    pykalman_smoothed_means
        Smoothed state means from ``pykalman.KalmanFilter.smooth()``,
        same shape (else ``None``).
    filtered_inf_norm
        :math:`\\max_t \\|\\hat{x}_t^{nstat} - \\hat{x}_t^{pykalman}\\|_\\infty`.
        Filtered means typically agree to ~1e-8 between any two correct
        implementations of the same linear-Gaussian model.
    smoothed_inf_norm
        Same for smoothed means; reflects the RTS smoother agreement.
        ``None`` if smoothing was not run.
    """

    nstat_filtered_means: np.ndarray
    pykalman_filtered_means: np.ndarray
    nstat_smoothed_means: np.ndarray | None
    pykalman_smoothed_means: np.ndarray | None
    filtered_inf_norm: float
    smoothed_inf_norm: float | None

    def assert_filtered_agree(self, atol: float = 1e-2) -> None:
        """Assert filtered means agree within ``atol``.

        Default ``atol=1e-2`` reflects the current empirical baseline
        on a 100×2 linear-Gaussian fixture (~2.6e-3, dominated by t=0
        initialization convention differences between nstat and
        pykalman).  Tighten to ``1e-8`` once nstat's filter is patched
        to match the pykalman initialization (prior at t=0 vs
        posterior at t=0).  Use this as a **regression guard**, not as
        a claim of exact agreement.
        """
        if self.filtered_inf_norm > atol:
            raise AssertionError(
                f"nstat vs pykalman filtered means disagree: "
                f"|Δx|_∞ = {self.filtered_inf_norm:.3e} > atol={atol}"
            )

    def assert_smoothed_agree(self, atol: float = 1.0) -> None:
        """Assert smoothed means agree within ``atol``.

        Default ``atol=1.0`` is intentionally **very** loose: it
        documents the current AUDIT D3 smoother-index approximation
        gap in :meth:`DecodingAlgorithms.kalman_fixedIntervalSmoother`
        (augmented-state smoother w/ finite lag vs full RTS smoother),
        which on the 100×2 reference fixture produces a ~0.4 unit
        disagreement.  Tighten to ``1e-6`` only after the smoother
        convergence work in AUDIT_REPORT.md §3.1 completes.
        """
        if self.smoothed_inf_norm is None:
            raise AssertionError(
                "Smoothed means were not computed; call "
                "cross_validate_kalman(..., compute_smoother=True)."
            )
        if self.smoothed_inf_norm > atol:
            raise AssertionError(
                f"nstat vs pykalman smoothed means disagree: "
                f"|Δx|_∞ = {self.smoothed_inf_norm:.3e} > atol={atol}"
            )


def cross_validate_kalman(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    transition_covariance: np.ndarray,
    observation_covariance: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
    *,
    compute_smoother: bool = True,
) -> KalmanComparison:
    """Fit a linear-Gaussian Kalman filter (+ smoother) in both nstat and pykalman.

    Parameters
    ----------
    observations
        Observation time series, shape ``(T, Dy)``.
    transition_matrix
        State transition matrix :math:`A`, shape ``(Dx, Dx)``.
    observation_matrix
        Observation matrix :math:`C`, shape ``(Dy, Dx)``.
    transition_covariance
        Process noise covariance :math:`Q`, shape ``(Dx, Dx)``.
    observation_covariance
        Measurement noise covariance :math:`R`, shape ``(Dy, Dy)``.
    initial_state_mean
        Initial state estimate :math:`\\hat{x}_0`, shape ``(Dx,)``.
    initial_state_covariance
        Initial state covariance :math:`P_0`, shape ``(Dx, Dx)``.
    compute_smoother
        If ``True`` (default), also fit the RTS smoother in both
        libraries and populate the smoothed-means fields.

    Returns
    -------
    KalmanComparison

    Notes
    -----
    The bridge calls nstat with the Pythonic keyword signature
    (``observations=``, ``transition=``, …) — see
    :meth:`DecodingAlgorithms.kalman_filter` for the MATLAB-compatible
    positional alternative.
    """
    PyKalmanFilter = _require_pykalman()

    observations = np.asarray(observations, dtype=float)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    A = np.asarray(transition_matrix, dtype=float)
    C = np.asarray(observation_matrix, dtype=float)
    Q = np.asarray(transition_covariance, dtype=float)
    R = np.asarray(observation_covariance, dtype=float)
    x0 = np.asarray(initial_state_mean, dtype=float).ravel()
    P0 = np.asarray(initial_state_covariance, dtype=float)

    # nstat filter
    nstat_result = DecodingAlgorithms.kalman_filter(
        observations=observations,
        transition=A,
        observation_matrix=C,
        q_cov=Q,
        r_cov=R,
        x0=x0,
        p0=P0,
    )
    nstat_filtered = np.asarray(nstat_result["state"], dtype=float)

    # pykalman filter
    pyk = PyKalmanFilter(
        transition_matrices=A,
        observation_matrices=C,
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_mean=x0,
        initial_state_covariance=P0,
    )
    pyk_filtered, _ = pyk.filter(observations)

    filtered_diff = float(np.max(np.abs(nstat_filtered - pyk_filtered)))

    nstat_smoothed = None
    pyk_smoothed = None
    smoothed_diff = None
    if compute_smoother:
        # nstat's smoother is MATLAB-style positional only and takes a
        # ``lags`` arg (augmented-state smoother — AUDIT D3).  We use
        # lags=T-1 to approximate full-window RTS smoothing, then
        # compare against pykalman's RTS smoother on the same model.
        lags = max(1, observations.shape[0] - 1)
        smoother_out = DecodingAlgorithms.kalman_fixedIntervalSmoother(
            A, C, Q, R, P0, x0, observations, lags
        )
        # Returns 4-tuple: (x_pLag, Pe_pLag, x_uLag, Pe_uLag).
        # x_uLag is the smoothed (updated, lagged) state, shape (N, Dx).
        nstat_smoothed = np.asarray(smoother_out[2], dtype=float)
        pyk_smoothed, _ = pyk.smooth(observations)
        # Length alignment: pykalman returns T points; nstat may differ
        # by 1 depending on lag semantics.  Compare the common prefix.
        common = min(nstat_smoothed.shape[0], pyk_smoothed.shape[0])
        smoothed_diff = float(
            np.max(np.abs(nstat_smoothed[:common] - pyk_smoothed[:common]))
        )

    return KalmanComparison(
        nstat_filtered_means=nstat_filtered,
        pykalman_filtered_means=pyk_filtered,
        nstat_smoothed_means=nstat_smoothed,
        pykalman_smoothed_means=pyk_smoothed,
        filtered_inf_norm=filtered_diff,
        smoothed_inf_norm=smoothed_diff,
    )


__all__ = ["KalmanComparison", "cross_validate_kalman"]
