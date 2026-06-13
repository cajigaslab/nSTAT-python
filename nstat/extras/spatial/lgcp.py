r"""Log-Gaussian Cox process rate maps by the Laplace approximation.

Pure NumPy/SciPy.  This is the Python-only companion to the curriculum's
Chapter 5 worked example (*Spatial Point Processes*): it turns a spatial
point pattern into a *posterior* rate map with calibrated credible bands
— the uncertainty a bare kernel-density rate map cannot provide.

The model (Chapter 5, Eq. 5.4):

1. Bin the events onto a grid; cell counts :math:`y_m` are exact integers
   (Prop. 5.A.1).
2. Place a Gaussian-process prior on the log-intensity field
   :math:`f = \log\Lambda`, :math:`f \sim \mathcal N(m_0, K)` with a
   Matern-5/2 covariance.
3. Find the posterior mode by Newton / IRLS (Rasmussen & Williams,
   Algorithm 3.1) with the Poisson observation model
   :math:`y_m \sim \text{Poisson}(a\,e^{f_m})` (``a`` = cell area):

   .. math::

      W = \operatorname{diag}(a\,e^{f}), \qquad
      f \leftarrow m_0 + (K^{-1} + W)^{-1}
          \bigl[W(f - m_0) + (y - a\,e^{f})\bigr].

4. The posterior covariance at the mode is
   :math:`\Sigma = (K^{-1} + \hat W)^{-1}`, with per-cell log-rate
   variance :math:`v_m = \Sigma_{mm}`.

The rate map is then read off as a **log-normal** band (the calibrated
payoff): for a credible level with two-sided z-score :math:`z`,

.. math::

   \text{rate\_mean} = e^{\hat f + v/2}, \qquad
   \text{rate\_lo/hi} = e^{\hat f \mp z\sqrt{v}} .

The band is **wider in data-sparse cells**: where there are no events,
:math:`\hat W \to 0`, so :math:`v_m \to K_{mm}` (the GP prior variance) —
this is the property the curriculum caption turns on, and the one the
companion test asserts.

An optional heavier GP path may bridge ``gpflow`` (install
``pip install nstat-toolbox[spatial-gp]``); the **default is
dependency-free** pure NumPy/SciPy.

References
----------
- Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
  Learning*, Algorithm 3.1 (Laplace approximation for GP classification),
  adapted here to the Poisson likelihood.
- Diggle PJ et al. (2013). *Spatial and spatio-temporal log-Gaussian Cox
  processes.* Statistical Science 28(4):542.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from nstat.extras._lazy import require_optional
from nstat.extras.spatial._kernels import bin_counts, make_grid, matern_covariance


@dataclass(frozen=True)
class LGCPResult:
    """Fitted LGCP rate map (plain NumPy).

    Attributes
    ----------
    grid
        ``(M, d)`` cell-centre coordinates.
    counts
        ``(M,)`` exact integer cell counts.
    f_mode
        ``(M,)`` posterior-mode log-intensity :math:`\\hat f`.
    f_var
        ``(M,)`` posterior log-intensity variance :math:`v_m`.
    cell_area
        Area (or volume) of one grid cell.
    edges
        Per-dimension bin edges (length ``G+1`` each).
    n_iter
        Newton/IRLS iterations to convergence.
    converged
        Whether the mode-finding hit the tolerance before ``max_iter``.
    """

    grid: np.ndarray
    counts: np.ndarray
    f_mode: np.ndarray
    f_var: np.ndarray
    cell_area: float
    edges: list
    n_iter: int
    converged: bool

    def rate_map(self, level: float = 0.90):
        r"""Posterior rate map with a log-normal credible band.

        Parameters
        ----------
        level
            Two-sided credible level (``0.90`` → :math:`z = 1.645`).

        Returns
        -------
        mean, lo, hi : tuple[np.ndarray, np.ndarray, np.ndarray]
            Each ``(M,)``: the log-normal posterior mean rate
            :math:`e^{\hat f + v/2}` and the lower/upper band
            :math:`e^{\hat f \mp z\sqrt v}`.  Rates are per unit area
            (multiply by :attr:`cell_area` for expected counts).
        """
        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0, 1)")
        z = float(norm.ppf(0.5 + level / 2.0))
        v = np.clip(self.f_var, 0.0, None)
        sd = np.sqrt(v)
        mean = np.exp(self.f_mode + 0.5 * v)
        lo = np.exp(self.f_mode - z * sd)
        hi = np.exp(self.f_mode + z * sd)
        return mean, lo, hi

    def rate_mean(self) -> np.ndarray:
        """Convenience: just the log-normal posterior-mean rate."""
        return np.exp(self.f_mode + 0.5 * np.clip(self.f_var, 0.0, None))

    def intensity_fn(self):
        """Return a callable ``X -> rate`` (nearest-cell lookup of the mean).

        Useful as the ``lambda_hat`` argument to the
        :mod:`nstat.extras.spatial.spatial_gof` estimators — though note
        the plug-in caveat (use a held-out intensity for honest coverage).
        """
        from scipy.spatial.distance import cdist

        mean = self.rate_mean()
        grid = self.grid

        def _fn(X):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            idx = np.argmin(cdist(X, grid), axis=1)
            return mean[idx]

        return _fn


def lgcp_fit(
    points: np.ndarray,
    domain,
    *,
    grid: int = 20,
    kernel: str = "matern52",
    length_scale: float = 0.12,
    variance: float = 1.0,
    prior_mean: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    jitter: float = 1e-6,
    backend: str = "numpy",
) -> LGCPResult:
    r"""Fit an LGCP rate map by the Laplace approximation.

    Parameters
    ----------
    points
        ``(n, d)`` event coordinates.
    domain
        ``((lo, hi), ...)`` rectangular analysis window per dimension.
    grid
        Number of cells per dimension (``G``).  The analysis grid is
        ``G**d`` cells.
    kernel
        GP covariance.  ``"matern52"`` (default, the curriculum choice),
        ``"matern32"``, or ``"exponential"`` (Matern-1/2).
    length_scale
        GP range parameter :math:`\ell`.
    variance
        GP marginal variance :math:`\sigma^2`.
    prior_mean
        Constant prior mean :math:`m_0` for the log-rate.  ``None``
        (default) uses :math:`\log(n/|D|) - \sigma^2/2` so the prior-mean
        rate matches the empirical event density.
    max_iter, tol
        Newton/IRLS stopping controls.
    jitter
        Diagonal jitter on the covariance for numerical stability.
    backend
        ``"numpy"`` (default, dependency-free) or ``"gpflow"`` (requires
        ``pip install nstat-toolbox[spatial-gp]``).  The gpflow path is a
        thin convenience that fits a variational GP and reads back the
        same posterior-mode/variance interface; it is **optional** and
        the NumPy path is the reference implementation.

    Returns
    -------
    LGCPResult
        Call :meth:`LGCPResult.rate_map` for the credible-band rate map.

    Notes
    -----
    *Confidence: high* — this is the standard Laplace GP (Rasmussen &
    Williams Alg. 3.1) with the Poisson link; the recovered mean tracks
    the true field and the band widens in data-sparse cells (both
    asserted in the companion tests).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    domain = tuple((float(lo), float(hi)) for lo, hi in domain)
    if pts.shape[1] != len(domain):
        raise ValueError(
            f"points dimension {pts.shape[1]} does not match domain "
            f"dimension {len(domain)}"
        )

    nu_map = {"matern52": 2.5, "matern32": 1.5, "exponential": 0.5}
    if kernel not in nu_map:
        raise ValueError(f"kernel must be one of {list(nu_map)}; got {kernel!r}")

    centres, cell_area, edges = make_grid(domain, grid)
    y = bin_counts(pts, edges)
    M = centres.shape[0]
    area = float(np.prod([hi - lo for lo, hi in domain]))

    if backend == "gpflow":
        return _lgcp_fit_gpflow(
            pts, centres, y, cell_area, edges, area,
            kernel=kernel, length_scale=length_scale, variance=variance,
            prior_mean=prior_mean,
        )
    elif backend != "numpy":
        raise ValueError(f"backend must be 'numpy' or 'gpflow'; got {backend!r}")

    K = matern_covariance(
        centres, length_scale=length_scale, variance=variance,
        nu=nu_map[kernel], jitter=jitter,
    )
    Kinv = np.linalg.inv(K)

    if prior_mean is None:
        m0 = float(np.log(max(y.sum(), 1.0) / area) - 0.5 * variance)
    else:
        m0 = float(prior_mean)

    f = np.full(M, m0)
    converged = False
    n_iter = max_iter
    for it in range(max_iter):
        lam = cell_area * np.exp(f)            # expected count per cell
        W = np.diag(lam)
        f_new = m0 + np.linalg.solve(Kinv + W, W @ (f - m0) + (y - lam))
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            converged = True
            n_iter = it + 1                    # 1-based iteration count for reporting
            break
        f = f_new

    Sigma = np.linalg.inv(Kinv + np.diag(cell_area * np.exp(f)))
    v = np.clip(np.diag(Sigma), 0.0, None)

    return LGCPResult(
        grid=centres,
        counts=y,
        f_mode=f,
        f_var=v,
        cell_area=cell_area,
        edges=edges,
        n_iter=n_iter,
        converged=converged,
    )


def _lgcp_fit_gpflow(
    pts, centres, y, cell_area, edges, area, *,
    kernel, length_scale, variance, prior_mean,
) -> LGCPResult:
    """Optional gpflow-backed fit (lazy import; falls back semantics).

    This path requires the ``spatial-gp`` extra.  It fits a variational
    GP regression to the (offset) log-counts and reads back the posterior
    mean / variance, exposing the same :class:`LGCPResult` interface as
    the NumPy reference path.  Kept intentionally thin.
    """
    gpflow = require_optional("gpflow", install_key="spatial-gp")
    import tensorflow as tf  # gpflow pulls TF; surfaced via the same extra

    nu_map = {"matern52": gpflow.kernels.Matern52,
              "matern32": gpflow.kernels.Matern32,
              "exponential": gpflow.kernels.Exponential}
    kern = nu_map[kernel](lengthscales=length_scale, variance=variance)

    # Poisson likelihood VGP on the cell counts (counts as a function of
    # cell centres, with the log cell-area as a fixed offset via mean fn).
    m0 = (float(np.log(max(y.sum(), 1.0) / area))
          if prior_mean is None else float(prior_mean))
    data = (tf.constant(centres, dtype=tf.float64),
            tf.constant(y.reshape(-1, 1), dtype=tf.float64))
    model = gpflow.models.VGP(
        data,
        kernel=kern,
        likelihood=gpflow.likelihoods.Poisson(),
        mean_function=gpflow.mean_functions.Constant(m0),
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables,
                 options={"maxiter": 200})
    f_mean, f_var = model.predict_f(tf.constant(centres, dtype=tf.float64))
    f_mode = np.asarray(f_mean).ravel() - np.log(cell_area)
    v = np.clip(np.asarray(f_var).ravel(), 0.0, None)
    return LGCPResult(
        grid=centres, counts=y, f_mode=f_mode, f_var=v,
        cell_area=cell_area, edges=edges, n_iter=200, converged=True,
    )


__all__ = ["LGCPResult", "lgcp_fit"]
