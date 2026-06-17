r"""Smoothed point-process residuals (pure NumPy/SciPy).

Kernel-smoothed Pearson-style residuals for a binned point-process
model, used to surface time-localised mis-specification that the
discrete-time-rescaling KS test averages over.

Given observed per-bin counts :math:`N_k` (typically 0/1 for narrow
bins) and a model's per-bin expected count
:math:`\hat\lambda_k\,\Delta`, the raw residual

.. math::

    e_k = N_k - \hat\lambda_k\,\Delta

is a mean-zero, finite-variance sequence under the true model (Brown et
al. 2002; Truccolo et al. 2005).  Smoothing :math:`e_k` with a
normalised Gaussian kernel of bandwidth :math:`b` bins gives a
*time-local* mis-fit indicator that should hover around zero under a
correct model and reveal a drift wherever the model under- or
over-predicts.  In the spatial / clusterless analogues the same
construction reduces to Andersen's (1997) cumulative residual process
after integration.

References
----------
- Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002). *The
  time-rescaling theorem and its application to neural spike train data
  analysis.* Neural Computation 14(2):325-346.
- Andersen PK (1997). *Statistical Models Based on Counting Processes.*
  Springer.
- Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005). *A
  point process framework for relating neural spiking activity to
  spiking history, neural ensemble, and extrinsic covariate effects.*
  Journal of Neurophysiology 93(2):1074-1089.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _gaussian_kernel(bandwidth_bins: float, truncate: float = 4.0) -> np.ndarray:
    """Normalised discrete Gaussian kernel of standard deviation ``bandwidth_bins``.

    Truncated at ``truncate * bandwidth`` bins each side; matches the
    convention of :func:`scipy.ndimage.gaussian_filter1d`.  The kernel
    is normalised to sum to 1 so that smoothing preserves the mean of
    the residual sequence.
    """
    b = float(bandwidth_bins)
    half = max(1, int(np.ceil(truncate * b)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / b) ** 2)
    k /= k.sum()
    return k


def pp_residuals_smoothed(
    spike_bins: np.ndarray,
    lam_per_bin: np.ndarray,
    bandwidth: float,
    *,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Kernel-smoothed point-process residuals on a uniform time grid.

    Builds the per-bin count vector
    :math:`N_k = \mathrm{bincount}(\text{spike\_bins},\,\text{minlength}=T)`,
    forms the raw residual :math:`e_k = N_k - \hat\lambda_k\,\Delta`
    using the model rate ``lam_per_bin`` (already multiplied by the
    bin width when the input is a per-bin probability), and convolves
    :math:`e_k` with a normalised discrete Gaussian kernel of standard
    deviation ``bandwidth`` *bins*.

    Parameters
    ----------
    spike_bins
        Integer bin indices at which events occurred.  Sorted or not —
        :func:`numpy.bincount` is order-invariant.
    lam_per_bin
        Per-bin model-expected count :math:`\hat\lambda_k\,\Delta`,
        length :math:`T`.  Must be non-negative.  In the discrete-time
        idiom of :mod:`nstat.extras.spatial.marked_gof` this is the
        spike *probability* :math:`p_k = \hat\lambda_k\,\Delta`.
    bandwidth
        Gaussian-kernel standard deviation in *bins*.  Must be strictly
        positive.  Wider bandwidth → smoother residual, less time
        resolution.
    dt
        Bin width in seconds.  Used only to build the returned time
        vector ``t_grid``; the residual values are independent of
        ``dt``.

    Returns
    -------
    t_grid : np.ndarray
        Bin-centre time vector, shape ``(T,)``.
    residuals : np.ndarray
        Smoothed residual sequence, shape ``(T,)``.  Centred at zero
        under the true model; a sustained departure from zero flags a
        time-localised mis-fit.

    Raises
    ------
    ValueError
        If ``bandwidth <= 0`` or ``lam_per_bin`` contains a negative
        value.

    Notes
    -----
    *Confidence: high* on the centred-at-zero behaviour under the true
    model and on the sign of the drift under a mis-specified model.
    The smoothing here is in *time* only; the same convolution
    transposes naturally to a 2-D Gaussian for spatial residuals.
    """
    lam = np.asarray(lam_per_bin, dtype=float).ravel()
    if lam.size == 0:
        raise ValueError("lam_per_bin must be non-empty")
    if np.any(lam < 0):
        raise ValueError("lam_per_bin must be non-negative")
    b = float(bandwidth)
    if b <= 0:
        raise ValueError(f"bandwidth must be positive; got {bandwidth!r}")
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt!r}")

    T = lam.size
    sb = np.asarray(spike_bins, dtype=int).ravel()
    if sb.size:
        if sb.min() < 0 or sb.max() >= T:
            raise ValueError(
                f"spike_bins must lie in [0, T); got min={sb.min()}, "
                f"max={sb.max()}, T={T}"
            )
        counts = np.bincount(sb, minlength=T).astype(float)
    else:
        counts = np.zeros(T, dtype=float)

    raw = counts - lam
    kernel = _gaussian_kernel(b)
    smoothed = fftconvolve(raw, kernel, mode="same")

    t_grid = (np.arange(T, dtype=float) + 0.5) * dt
    return t_grid, smoothed


__all__ = [
    "pp_residuals_smoothed",
]
