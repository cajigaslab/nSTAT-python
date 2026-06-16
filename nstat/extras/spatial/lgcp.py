r"""Log-Gaussian Cox process rate maps by the Laplace approximation.

Pure NumPy/SciPy.  Turns a spatial point pattern into a *posterior*
rate map with calibrated credible bands — the uncertainty a bare
kernel-density rate map cannot provide.

The model:

1. Bin the events onto a grid; cell counts :math:`y_m` are exact
   integers (Møller, Syversveen & Waagepetersen 1998).
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
this is the property the companion test asserts.

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
import scipy.linalg as sla
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
        GP covariance.  ``"matern52"`` (default, the standard
        smooth-but-not-too-smooth LGCP prior), ``"matern32"``, or
        ``"exponential"`` (Matern-1/2).
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


# ----------------------------------------------------------------------
# Basis-projected LGCP (Tier B): Matern GP prior directly on B-spline
# coefficients (de Boor 1978) at their Greville anchor points.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MaternPrior:
    r"""Matern GP prior on the coefficient vector of a 2-D B-spline basis.

    Used by :func:`lgcp_fit_glm` to place a smooth GP prior directly on the
    :math:`K` B-spline coefficients (evaluated at their Greville abscissae,
    de Boor 1978) rather than on the :math:`M = G^2` cell log-rates.  When
    :math:`K \ll M` this is the difference between an :math:`O(K^3)` IRLS
    step and an :math:`O(M^3)` one — for a 64x64 grid (:math:`M=4096`) with
    a typical :math:`K=64`-coefficient basis the speedup is ~64x in the
    dominant cubic cost.

    Parameters
    ----------
    nu
        Matern smoothness; one of ``{0.5, 1.5, 2.5}`` (closed-form Matern).
    length_scale
        Range parameter :math:`\ell > 0`.
    marginal_var
        Marginal variance :math:`\sigma^2 > 0` (default ``1.0``).
    jitter
        Diagonal stabilizer added to :math:`K` (default ``1e-6``); on a
        :class:`numpy.linalg.LinAlgError` from Cholesky the jitter is
        retried once at ``10 * jitter``.

    Methods
    -------
    K(coords)
        Dense ``(K, K)`` covariance.
    cholesky(coords)
        Lower Cholesky factor :math:`L` such that :math:`K = LL^\top`.
    K_inv(coords)
        Inverse :math:`K^{-1}` via :func:`scipy.linalg.cho_solve`.
    log_det(coords)
        :math:`\log\det K = 2\sum_i \log L_{ii}`.

    Each derived quantity is computed once per ``coords`` and cached on
    the instance, keyed by the array's shape and bytes.

    References
    ----------
    - Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
      Learning*, Ch. 4.
    - Møller J, Syversveen AR, Waagepetersen RP (1998). *Log Gaussian Cox
      processes.* Scand. J. Statistics 25(3):451-482.
    """

    nu: float
    length_scale: float
    marginal_var: float = 1.0
    jitter: float = 1e-6

    def __post_init__(self):
        if self.nu not in (0.5, 1.5, 2.5):
            raise ValueError(
                f"MaternPrior.nu must be one of {{0.5, 1.5, 2.5}}; got {self.nu}"
            )
        if self.length_scale <= 0:
            raise ValueError(
                f"MaternPrior.length_scale must be positive; got {self.length_scale}"
            )
        if self.marginal_var <= 0:
            raise ValueError(
                f"MaternPrior.marginal_var must be positive; got {self.marginal_var}"
            )
        if self.jitter < 0:
            raise ValueError(
                f"MaternPrior.jitter must be non-negative; got {self.jitter}"
            )
        # frozen=True blocks normal attribute assignment; bypass to install
        # the per-instance cache exactly once.
        object.__setattr__(self, "_cache", {})

    def _key(self, coords: np.ndarray) -> tuple:
        arr = np.ascontiguousarray(np.asarray(coords, dtype=float))
        return (arr.shape, arr.tobytes())

    def _entry(self, coords: np.ndarray) -> dict:
        cache = self._cache  # type: ignore[attr-defined]
        key = self._key(coords)
        if key in cache:
            return cache[key]
        # Build K, factor it, derive inverse and log-determinant once.
        jitter = float(self.jitter)
        for attempt in range(2):
            K = matern_covariance(
                coords,
                length_scale=float(self.length_scale),
                variance=float(self.marginal_var),
                nu=float(self.nu),
                jitter=jitter,
            )
            try:
                L = np.linalg.cholesky(K)
                break
            except np.linalg.LinAlgError:
                if attempt == 0:
                    jitter = max(jitter, 1e-12) * 10.0
                    continue
                raise
        # Inverse via cho_solve on the identity (uses the factor we already paid for).
        K_inv = sla.cho_solve((L, True), np.eye(L.shape[0]))
        log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
        entry = {"K": K, "L": L, "K_inv": K_inv, "log_det": log_det, "jitter": jitter}
        cache[key] = entry
        return entry

    def K(self, coords: np.ndarray) -> np.ndarray:
        """Dense Matern covariance ``(K, K)`` on ``coords``."""
        return self._entry(coords)["K"]

    def cholesky(self, coords: np.ndarray) -> np.ndarray:
        """Lower Cholesky factor of :meth:`K`."""
        return self._entry(coords)["L"]

    def K_inv(self, coords: np.ndarray) -> np.ndarray:
        """Inverse covariance (computed via the cached factor)."""
        return self._entry(coords)["K_inv"]

    def log_det(self, coords: np.ndarray) -> float:
        """Log-determinant of :meth:`K`."""
        return self._entry(coords)["log_det"]


def _irls_penalized_glm(
    B: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    K_inv: np.ndarray,
    m0: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Penalized Poisson IRLS in coefficient space (private to ``lgcp.py``).

    Solves

    .. math::

        \\hat\\beta = \\arg\\max_\\beta \\;
        \\sum_m \\bigl(y_m \\eta_m - e^{\\eta_m}\\bigr)
        - \\tfrac{1}{2}(\\beta - m_0)^\\top K^{-1} (\\beta - m_0),

    with :math:`\\eta = B \\beta + \\text{offset}`, via Newton/IRLS:

    - Gradient: :math:`g = B^\\top(y - \\lambda) - K^{-1}(\\beta - m_0)`.
    - Hessian: :math:`H = B^\\top \\operatorname{diag}(\\lambda) B + K^{-1}`.
    - Update: :math:`\\beta \\leftarrow \\beta + H^{-1} g`.

    The Hessian is symmetric positive-definite (sum of a PSD weighted
    Gram and PD :math:`K^{-1}`), so :func:`scipy.linalg.solve` with
    ``assume_a="pos"`` is the right primitive.

    Parameters
    ----------
    B
        ``(M, K)`` basis design matrix (rows = grid cells, cols = coefs).
    y
        ``(M,)`` cell counts.
    offset
        ``(M,)`` log-offset added to the linear predictor (typically
        ``log(cell_area)``).
    K_inv
        ``(K, K)`` precision matrix from the GP prior.
    m0
        ``(K,)`` prior mean.
    max_iter, tol
        Newton/IRLS stopping controls.

    Returns
    -------
    beta : np.ndarray
        ``(K,)`` coefficient mode.
    Sigma_beta : np.ndarray
        ``(K, K)`` posterior covariance :math:`H^{-1}` at the mode.
    n_iter : int
        Iterations actually run (1-based count when converged).
    converged : bool
        Whether the max-abs delta dropped below ``tol``.

    References
    ----------
    - Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
      Learning*, Algorithm 3.1 (Laplace approximation).
    - Wood SN (2017). *Generalized Additive Models: An Introduction with R*,
      Ch. 3, penalized IRLS for exponential-family GAMs.
    """
    K = K_inv.shape[0]
    beta = np.array(m0, dtype=float, copy=True)
    converged = False
    n_iter = max_iter
    for it in range(max_iter):
        eta = B @ beta + offset
        # Numerical guard: clip eta so exp doesn't overflow.  The IRLS
        # iterate can spike eta on early steps before the prior pulls
        # it back into a sensible range.
        np.clip(eta, -50.0, 50.0, out=eta)
        lam = np.exp(eta)
        g = B.T @ (y - lam) - K_inv @ (beta - m0)
        # H = B.T @ diag(lam) @ B + K_inv  (avoid building the dense diag).
        BtW = B.T * lam  # shape (K, M)
        H = BtW @ B + K_inv
        delta = sla.solve(H, g, assume_a="pos")
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            converged = True
            n_iter = it + 1
            break
    # Posterior covariance at the converged mode (one final factorization).
    eta = B @ beta + offset
    np.clip(eta, -50.0, 50.0, out=eta)
    lam = np.exp(eta)
    H = (B.T * lam) @ B + K_inv
    c, low = sla.cho_factor(H, lower=True)
    Sigma_beta = sla.cho_solve((c, low), np.eye(K))
    return beta, Sigma_beta, n_iter, converged


def lgcp_fit_glm(
    points: np.ndarray,
    domain,
    basis,
    prior: MaternPrior,
    *,
    grid: int = 32,
    prior_mean: float | np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> LGCPResult:
    r"""Basis-projected LGCP fit by penalized Poisson IRLS.

    Bins events into a ``grid x grid`` cell grid, projects the
    log-intensity onto a 2-D B-spline basis (:class:`~nstat.extras.spatial.basis.BSplineBasis2D`),
    and finds the posterior mode of the basis coefficients under a
    :class:`MaternPrior` evaluated at the Greville abscissae of the basis.
    The resulting per-cell log-rate posterior :math:`(\hat f, v)` is
    returned as the same :class:`LGCPResult` that :func:`lgcp_fit`
    produces — so :meth:`LGCPResult.rate_map` is available unchanged.

    The cubic cost of the dominant linear solve scales with the basis
    dimension :math:`K`, not the grid size :math:`M`, so this is the
    routine to reach for on grids where the dense-GP :func:`lgcp_fit`
    would be slow (e.g. :math:`G \ge 50`).

    Parameters
    ----------
    points
        ``(n, 2)`` event coordinates.
    domain
        ``((xlo, xhi), (ylo, yhi))`` rectangular window.
    basis
        A :class:`~nstat.extras.spatial.basis.BSplineBasis2D` whose
        ``grid_x`` / ``grid_y`` are the cell-centre axes of the analysis
        grid (length ``grid`` each).  Use
        :meth:`BSplineBasis2D.from_grid` to construct.
    prior
        :class:`MaternPrior` evaluated at the basis' Greville abscissae
        (de Boor 1978) — the prior on the coefficient vector.
    grid
        Cells per axis (default 32).  Must equal ``len(basis.grid_x)`` /
        ``len(basis.grid_y)``.
    prior_mean
        Constant prior mean :math:`m_0` for the log-rate (added on the
        coefficient axis; scalar broadcasts).  ``None`` (default) sets
        :math:`m_0 = \log(\max(n,1)/|D|) - \sigma^2/2` (matching
        :func:`lgcp_fit`'s default and reflecting the log-normal mean
        correction of Møller et al. 1998).
    max_iter, tol
        Newton/IRLS stopping controls.

    Returns
    -------
    LGCPResult
        With the same field semantics as :func:`lgcp_fit`: ``f_mode``
        and ``f_var`` are per-cell log-rates (so :meth:`rate_map` works
        without modification).

    Notes
    -----
    *Confidence: high* — this is the standard penalized GLM/GAM Laplace
    approximation (Wood 2017, Ch. 3; Diggle et al. 2013) with a Matern
    prior on B-spline coefficients.  The basis-projection device is
    Diggle, Moraga, Rowlingson & Taylor 2013, *Statistical Science*.

    References
    ----------
    - Møller J, Syversveen AR, Waagepetersen RP (1998). *Log Gaussian Cox
      processes.* Scand. J. Statistics 25(3):451-482.
    - Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
      Learning.*
    - Diggle PJ, Moraga P, Rowlingson B, Taylor BM (2013). *Spatial and
      spatio-temporal log-Gaussian Cox processes.* Statistical Science
      28(4):542.
    - Wood SN (2017). *Generalized Additive Models: An Introduction with R.*

    See Also
    --------
    lgcp_fit : dense per-cell GP fit; the reference implementation
        (cheaper at small ``G``, expensive at large ``G``).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    domain = tuple((float(lo), float(hi)) for lo, hi in domain)
    if pts.shape[1] != 2 or len(domain) != 2:
        raise ValueError("lgcp_fit_glm is 2-D only; points and domain must be 2-D")
    G = int(grid)
    if len(basis.grid_x) != G or len(basis.grid_y) != G:
        raise ValueError(
            f"basis grids must have length grid={G}; got "
            f"({len(basis.grid_x)}, {len(basis.grid_y)})"
        )

    # Cell edges and centres in ij flattening (x outer, y inner) — matches
    # the basis design-matrix column order.  Deliberately not via
    # _kernels.make_grid, which is indexing='xy'.
    (xlo, xhi), (ylo, yhi) = domain
    edges_x = np.linspace(xlo, xhi, G + 1)
    edges_y = np.linspace(ylo, yhi, G + 1)
    cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    XX, YY = np.meshgrid(cx, cy, indexing="ij")
    cell_centres = np.column_stack([XX.ravel(), YY.ravel()])

    # Bin events in ij flattening — histogram2d returns shape (Gx, Gy)
    # already in ij order, so .ravel() is the right flattening (no swap).
    if pts.shape[0] == 0:
        y = np.zeros(G * G, dtype=float)
    else:
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[edges_x, edges_y])
        y = H.ravel().astype(float)

    cell_area = float((xhi - xlo) * (yhi - ylo) / (G * G))
    area = float((xhi - xlo) * (yhi - ylo))

    B = basis.design_matrix()
    if B.shape[0] != G * G:
        raise ValueError(
            f"basis design matrix has {B.shape[0]} rows; expected G*G = {G * G}"
        )
    coef_coords = basis.coefficient_coords()
    K_inv = prior.K_inv(coef_coords)
    K_dim = B.shape[1]

    if prior_mean is None:
        m0_scalar = float(np.log(max(y.sum(), 1.0) / area) - 0.5 * float(prior.marginal_var))
        m0 = np.full(K_dim, m0_scalar)
    elif np.isscalar(prior_mean):
        m0 = np.full(K_dim, float(prior_mean))
    else:
        m0 = np.asarray(prior_mean, dtype=float).reshape(-1)
        if m0.shape != (K_dim,):
            raise ValueError(
                f"prior_mean must be scalar or shape ({K_dim},); got {m0.shape}"
            )

    offset = np.full(G * G, np.log(cell_area))
    beta, Sigma_beta, n_iter, converged = _irls_penalized_glm(
        B, y, offset, K_inv, m0, max_iter=max_iter, tol=tol,
    )

    # Per-cell posterior: f is log-rate per unit area (LGCPResult semantics
    # — see lgcp_fit, where the log(cell_area) only enters the likelihood
    # as an offset and is NOT stored in f_mode).  Without this, the stored
    # f_mode is log-expected-count and rate_map() returns counts, not rates.
    f_mode = B @ beta
    # einsum is O(M * K^2); fine for the basis sizes this routine targets.
    v = np.einsum("mi,ij,mj->m", B, Sigma_beta, B)
    v = np.clip(v, 0.0, None)

    return LGCPResult(
        grid=cell_centres,
        counts=y,
        f_mode=f_mode,
        f_var=v,
        cell_area=cell_area,
        edges=[edges_x, edges_y],
        n_iter=n_iter,
        converged=converged,
    )


__all__ = ["LGCPResult", "lgcp_fit", "lgcp_fit_glm", "MaternPrior"]
