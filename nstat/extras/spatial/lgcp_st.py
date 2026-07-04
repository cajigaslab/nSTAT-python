r"""Spatiotemporal log-Gaussian Cox process by a Kronecker Laplace approximation.

Pure NumPy/SciPy.  Extends :mod:`nstat.extras.spatial.lgcp` from a static
spatial rate map to a **time-varying** rate :math:`\Lambda(x, y, t)` on a
3-D ``(x, y, t)`` grid, with a *separable* space x time Matern-GP prior on
the log-intensity and a Kronecker-structured Laplace (Newton/IRLS)
posterior-mode solve.  ``lgcp_fit`` (spatial-only) is the ``Gt = 1``
special case of this model.

Model
-----

1. Bin events onto a 3-D grid; cell counts :math:`y_m` are exact
   integers (Moller, Syversveen & Waagepetersen 1998), same as
   :func:`nstat.extras.spatial.lgcp.lgcp_fit`.
2. Place a *separable* GP prior on the log-intensity field
   :math:`f = \log \Lambda`, :math:`f \sim \mathcal N(m_0, K)` with

   .. math::

      K = K_x \otimes K_y \otimes K_t,

   the Kronecker product of three independent 1-D Matern covariances
   (one per axis; Rasmussen & Williams 2006 Sec. 4.2 for the closed-form
   Matern kernel; Saatci 2011 Ch. 5 and Wilson & Nickisch 2015 for
   Kronecker/separable GP-grid priors).
3. Find the posterior mode by Newton / IRLS (Rasmussen & Williams 2006,
   Algorithm 3.1) with the Poisson observation model
   :math:`y_m \sim \mathrm{Poisson}(v\, e^{f_m})` (``v`` = cell volume):

   .. math::

      W = \operatorname{diag}(v\, e^{f}), \qquad
      f \leftarrow m_0 + (K^{-1} + W)^{-1}
          \bigl[W(f - m_0) + (y - v\, e^{f})\bigr].

4. The posterior covariance at the mode is
   :math:`\Sigma = (K^{-1} + \hat W)^{-1}`; per-cell log-rate variance
   :math:`v_m = \Sigma_{mm}` feeds the same log-normal credible band as
   :class:`~nstat.extras.spatial.lgcp.LGCPResult`.

The Kronecker trick
--------------------

The whole point of factoring the prior is to **never form the dense**
:math:`(G_x G_y G_t)^2` **matrix**.  Each 1-D Matern factor
:math:`K_d` is eigendecomposed, :math:`K_d = Q_d \Lambda_d Q_d^\top`
(``numpy.linalg.eigh``, :math:`d \in \{x, y, t\}`).  Because
Kronecker-product eigenpairs are themselves products
(standard multilinear algebra; Saatci 2011 Sec. 5.1),

.. math::

   K = (Q_x \otimes Q_y \otimes Q_t)\,
       (\Lambda_x \otimes \Lambda_y \otimes \Lambda_t)\,
       (Q_x \otimes Q_y \otimes Q_t)^\top ,

and a Kronecker matrix-vector product :math:`(A \otimes B \otimes C) v`
is computed by reshaping :math:`v` into a 3-way tensor and applying
:math:`A`, :math:`B`, :math:`C` as **mode products** along their
respective axes (:func:`_kron_matvec`) — :math:`O(G_x G_y G_t \cdot
\max(G_x, G_y, G_t))` work and no :math:`(G_x G_y G_t)^2` storage,
versus :math:`O((G_x G_y G_t)^3)` for a naive dense solve.

:math:`K^{-1}` is Kronecker-structured (the eigenbasis diagonalizes the
whole product at once) but :math:`K^{-1} + W` is **not** — the diagonal
observation-precision :math:`W` does not commute with a single small
eigenbasis.  Per Newton step we therefore solve
:math:`(K^{-1} + W) x = \mathrm{rhs}` with a *few* conjugate-gradient
iterations (:func:`scipy.sparse.linalg.cg`) against a
:class:`scipy.sparse.linalg.LinearOperator` whose ``matvec`` chains the
fast Kronecker :math:`K^{-1}` product with an elementwise :math:`W v` —
the "GP-grid" approach (Saatci 2011; Wilson & Nickisch 2015, *Kernel
Interpolation for Scalable Structured Gaussian Processes*, ICML).  This
keeps the whole mode-finding loop matrix-free at any grid size,
including the ``(24, 24, 24)`` default (:math:`\approx 1.4\times 10^4`
cells, for which the dense :math:`(K^{-1}+W)^{-1}` matrix alone would be
~1.5 GB and its Cholesky factorization :math:`O(10^{12})` flops).

The posterior-variance diagonal :math:`v_m = \Sigma_{mm}` has no closed
Kronecker form once :math:`W` is added, so it is estimated **matrix-free**
by the stochastic Hutchinson diagonal estimator (Hutchinson 1990; Bekas,
Kokiopoulou & Saad 2007): for Rademacher probe vectors :math:`z_i`
(:math:`\pm 1` i.i.d.), solve :math:`(K^{-1}+W) x_i = z_i` by the same
CG operator and average :math:`z_i \odot x_i`
(:math:`\mathbb E[z \odot \Sigma z] = \operatorname{diag}(\Sigma)` for
independent, zero-mean, unit-variance ``z`` entries).

Axis-order convention
----------------------

Internally the grid tensor is flattened ``(y, x, t)`` — ``y`` slow, ``x``
fast — to match the exact flattening convention already used by
:func:`nstat.extras.spatial._kernels.bin_counts` /
:func:`~nstat.extras.spatial._kernels.make_grid` (``indexing='xy'``),
with ``t`` appended as a third, fastest-varying axis.  This is the
*same* separable model as the :math:`K_x \otimes K_y \otimes K_t`
written above — Kronecker factors commute under simultaneous
relabelling of factors and tensor axes — just re-indexed so
:func:`nstat.extras.spatial._kernels.bin_counts` can be reused verbatim
with no post-hoc transpose.  :func:`_kron_matvec`'s companion test
verifies the fast path against an explicit ``numpy.kron`` construction
in this exact axis order.

References
----------
- Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
  Learning*, Algorithm 3.1 and Sec. 4.2 (Matern kernels).
- Moller J, Syversveen AR, Waagepetersen RP (1998). *Log Gaussian Cox
  processes.* Scand. J. Statistics 25(3):451-482.
- Diggle PJ, Moraga P, Rowlingson B, Taylor BM (2013). *Spatial and
  spatio-temporal log-Gaussian Cox processes.* Statistical Science
  28(4):542.
- Saatci Y (2011). *Scalable Inference for Structured Gaussian Process
  Models.* PhD thesis, University of Cambridge (Kronecker/GP-grid
  inference, Ch. 5).
- Wilson AG, Nickisch H (2015). *Kernel Interpolation for Scalable
  Structured Gaussian Processes (KISS-GP).* ICML.
- Hutchinson MF (1990). *A stochastic estimator of the trace of the
  influence matrix for Laplacian smoothing splines.* Comm. Statist.
  Simulation Comput. 19(2):433-450.
- Bekas C, Kokiopoulou E, Saad Y (2007). *An estimator for the diagonal
  of a matrix.* Applied Numerical Mathematics 57(11-12):1214-1229.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist
from scipy.stats import norm

from nstat.extras.spatial._kernels import bin_counts, matern_covariance

# Matrix-free numerics: conjugate-gradient tolerances for the Newton/IRLS
# precision solve, and the Hutchinson stochastic-diagonal estimator used
# for the posterior variance (no closed Kronecker form once the diagonal
# observation precision W is added to K^-1; see module docstring).
_CG_RTOL = 1e-10
_CG_MAXITER = 500
_N_VARIANCE_PROBES = 64
_VARIANCE_RNG_SEED = 0
_ETA_CLIP = 50.0


# ----------------------------------------------------------------------
# Kronecker / mode-product machinery
# ----------------------------------------------------------------------


def _axis_grid(lo: float, hi: float, n: int):
    """1-D cell edges / centres / width on ``[lo, hi]`` split into ``n`` cells.

    Same edge/centre construction as
    :func:`nstat.extras.spatial._kernels.make_grid`, generalized to a
    single axis so per-axis grid counts ``(Gx, Gy, Gt)`` need not be
    equal (``make_grid`` takes one ``n_per_dim`` shared across all axes).
    """
    lo = float(lo)
    hi = float(hi)
    n = int(n)
    edges = np.linspace(lo, hi, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = (hi - lo) / n
    return centres, edges, width


def _kron_matvec(v: np.ndarray, mats: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    r"""Fast Kronecker matrix-vector product via mode products.

    Computes :math:`(\text{mats}[0] \otimes \text{mats}[1] \otimes
    \cdots) v` **without** forming the dense Kronecker matrix, by
    reshaping the flat vector ``v`` into a tensor of shape ``shape`` and
    contracting each factor along its own axis (standard multilinear
    algebra; Saatci 2011 Sec. 5.1).  For factor sizes
    :math:`(n_0, n_1, \dots)` with :math:`M = \prod_k n_k`, this costs
    :math:`O(M \cdot \max_k n_k)` instead of :math:`O(M^2)` for a dense
    matrix-vector product.

    Parameters
    ----------
    v
        Flat vector of length ``prod(shape)``.
    mats
        One square matrix per axis of ``shape`` (``mats[k]`` is
        ``(shape[k], shape[k])``).
    shape
        The tensor shape ``v`` is reshaped to (C/row-major).

    Returns
    -------
    np.ndarray
        Flat vector of length ``prod(shape)``.

    Notes
    -----
    Correctness (verified to machine precision against
    ``numpy.kron`` in the companion test): for a tensor ``F`` of shape
    ``shape`` flattened in C order, the Kronecker product
    ``mats[0] (x) mats[1] (x) ...`` acting on ``vec(F)`` is *exactly* the
    tensor obtained by contracting ``mats[k]`` along axis ``k`` of
    ``F`` for every ``k`` — no axis-order correction needed, because
    C-order flattening puts axis 0 slowest / last axis fastest, which
    matches the standard (row-major) multi-index definition of the
    Kronecker product used here.
    """
    F = np.asarray(v, dtype=float).reshape(shape)
    for axis, K in enumerate(mats):
        F = np.moveaxis(F, axis, 0)
        n, rest_shape = F.shape[0], F.shape[1:]
        # Reshape to a plain 2-D matrix and use `@` (maps directly to a
        # BLAS gemm call) rather than `np.tensordot`, which is faster in
        # practice for this "one small factor x one big flattened batch"
        # shape (mode product along axis 0 of an N-way tensor).
        F2 = K @ np.ascontiguousarray(F).reshape(n, -1)
        F = F2.reshape((K.shape[0],) + rest_shape)
        F = np.moveaxis(F, 0, axis)
    return F.reshape(-1)


def _make_kinv_matvec(
    Qy: np.ndarray, Qx: np.ndarray, Qt: np.ndarray,
    inv_eig_flat: np.ndarray, shape: tuple[int, int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    r"""Fast :math:`K^{-1} v` via the eigendecomposition of each factor.

    :math:`K^{-1} = (Q_y \otimes Q_x \otimes Q_t)\,
    \operatorname{diag}(1/\lambda)\,(Q_y \otimes Q_x \otimes Q_t)^\top`,
    applied as two :func:`_kron_matvec` calls (transform to the
    eigenbasis, scale, transform back) plus one elementwise divide — the
    Kronecker matvec path used for the GP prior term in the Newton/IRLS
    solve (see module docstring).
    """
    QyT, QxT, QtT = Qy.T, Qx.T, Qt.T

    def _matvec(v: np.ndarray) -> np.ndarray:
        u = _kron_matvec(v, [QyT, QxT, QtT], shape)
        u = u * inv_eig_flat
        u = _kron_matvec(u, [Qy, Qx, Qt], shape)
        return u

    return _matvec


def _to_txy(flat: np.ndarray, Gy: int, Gx: int, Gt: int) -> np.ndarray:
    """Reshape a ``(y, x, t)``-flattened vector to ``(Gt, Gy*Gx)``.

    ``Gy*Gx`` columns are in the same ``y``-slow / ``x``-fast order as
    :func:`_spatial_grid` (matching
    :func:`nstat.extras.spatial._kernels.make_grid`'s convention).
    """
    return flat.reshape(Gy, Gx, Gt).transpose(2, 0, 1).reshape(Gt, Gy * Gx)


def _spatial_grid(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """``(Gx*Gy, 2)`` spatial centres, ``y`` slow / ``x`` fast (meshgrid ``'xy'``)."""
    XX, YY = np.meshgrid(cx, cy, indexing="xy")
    return np.column_stack([XX.ravel(), YY.ravel()])


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LGCPSTResult:
    """Fitted spatiotemporal LGCP rate field (plain NumPy).

    Attributes
    ----------
    grid_x
        ``(Gx*Gy, 2)`` spatial cell-centre coordinates (``y`` slow,
        ``x`` fast — see module docstring).
    grid_t
        ``(Gt,)`` time cell-centre coordinates.
    counts
        ``(Gt, Gx*Gy)`` exact integer cell counts, row ``k`` matching
        ``grid_t[k]`` and columns matching ``grid_x``.
    f_mode
        ``(Gt, Gx*Gy)`` posterior-mode log-intensity :math:`\\hat f`.
    f_var
        ``(Gt, Gx*Gy)`` posterior log-intensity variance
        (Hutchinson stochastic-diagonal estimate; see module docstring).
    cell_volume
        Volume (area x duration) of one grid cell.
    n_iter
        Newton/IRLS iterations to convergence.
    converged
        Whether the mode-finding hit the tolerance before ``max_iter``.
    """

    grid_x: np.ndarray
    grid_t: np.ndarray
    counts: np.ndarray
    f_mode: np.ndarray
    f_var: np.ndarray
    cell_volume: float
    n_iter: int
    converged: bool

    def rate_map(self, t: float, level: float = 0.90):
        r"""Posterior spatial rate map at time ``t``, log-normal credible band.

        Parameters
        ----------
        t
            Query time; snapped to the nearest entry of :attr:`grid_t`.
        level
            Two-sided credible level (``0.90`` -> :math:`z = 1.645`).

        Returns
        -------
        mean, lo, hi : tuple[np.ndarray, np.ndarray, np.ndarray]
            Each ``(Gx*Gy,)``: the log-normal posterior mean rate
            :math:`e^{\hat f + v/2}` and the lower/upper band
            :math:`e^{\hat f \mp z\sqrt v}` at the nearest time bin to
            ``t``.  Rates are per unit volume (multiply by
            :attr:`cell_volume` for expected counts).
        """
        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0, 1)")
        t_idx = int(np.argmin(np.abs(self.grid_t - float(t))))
        z = float(norm.ppf(0.5 + level / 2.0))
        v = np.clip(self.f_var[t_idx], 0.0, None)
        sd = np.sqrt(v)
        f = self.f_mode[t_idx]
        mean = np.exp(f + 0.5 * v)
        lo = np.exp(f - z * sd)
        hi = np.exp(f + z * sd)
        return mean, lo, hi

    def intensity_fn(self) -> Callable[[np.ndarray, "float | np.ndarray"], np.ndarray]:
        """Return a callable ``fn(X, t) -> rate`` (nearest-cell lookup).

        Parameters of the returned callable
        ------------------------------------
        X : array_like, shape ``(n, 2)`` (or ``(2,)`` for one point)
            Spatial query coordinates.
        t : float or array_like, shape ``(n,)``
            Query time(s).  A scalar broadcasts to every row of ``X``.

        Returns
        -------
        np.ndarray
            ``(n,)`` log-normal posterior-mean rate
            :math:`e^{\\hat f + v/2}` at the nearest ``(space, time)``
            cell to each query.

        Useful as a Cox-process background-rate function, e.g. the
        ``mu(x, t)`` argument of a spatiotemporal Hawkes background.
        """
        mean_full = np.exp(self.f_mode + 0.5 * np.clip(self.f_var, 0.0, None))
        grid_x = self.grid_x
        grid_t = self.grid_t

        def _fn(X, t):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            n = X.shape[0]
            t_arr = np.asarray(t, dtype=float)
            if t_arr.ndim == 0:
                t_arr = np.full(n, float(t_arr))
            else:
                t_arr = np.broadcast_to(t_arr.reshape(-1), (n,))
            sp_idx = np.argmin(cdist(X, grid_x), axis=1)
            t_idx = np.argmin(np.abs(grid_t[None, :] - t_arr[:, None]), axis=1)
            return mean_full[t_idx, sp_idx]

        return _fn


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------


def lgcp_st_fit(
    points: np.ndarray,
    times: np.ndarray,
    *,
    domain,
    period,
    grid: tuple[int, int, int] = (24, 24, 24),
    length_scale_space: float = 0.12,
    length_scale_time: float = 0.1,
    nu: float = 1.5,
    variance: float = 1.0,
    prior_mean: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    jitter: float = 1e-6,
) -> LGCPSTResult:
    r"""Fit a spatiotemporal LGCP rate field by a Kronecker Laplace approximation.

    Parameters
    ----------
    points
        ``(n, 2)`` spatial event coordinates.
    times
        ``(n,)`` event times, paired with ``points`` row-for-row.
    domain
        ``((xlo, xhi), (ylo, yhi))`` rectangular spatial analysis window.
    period
        ``(tlo, thi)`` analysis time window.
    grid
        ``(Gx, Gy, Gt)`` cells per axis.  The analysis grid is
        ``Gx * Gy * Gt`` cells; never densely materialized (see module
        docstring) — the default ``(24, 24, 24)`` (~1.4e4 cells) is fit
        entirely matrix-free.
    length_scale_space, length_scale_time
        Matern range parameters :math:`\ell_{xy}` (shared by the ``x``
        and ``y`` axes — an isotropic-in-space, separable-in-time prior)
        and :math:`\ell_t`.
    nu
        Shared Matern smoothness for all three axis kernels; one of
        ``{0.5, 1.5, 2.5}`` (the closed-form orders implemented by
        :func:`~nstat.extras.spatial._kernels.matern_covariance`).
    variance
        Marginal variance of the full separable prior,
        :math:`\operatorname{Var}[f_m] = K_{mm}`.  Split as the cube
        root across the three axis factors
        (:math:`\sigma_x^2 = \sigma_y^2 = \sigma_t^2 = \text{variance}^{1/3}`)
        so the product reproduces ``variance`` exactly and a single
        ``variance`` knob scales the informativeness of *all three* axes
        together (not just one) — e.g. ``variance -> inf`` yields a
        flat/uninformative prior on every axis at once, collapsing the
        Newton/IRLS mode to the per-cell Poisson MLE
        :math:`\log(y_m / \text{cell\_volume})`.
    prior_mean
        Constant prior mean :math:`m_0` for the log-rate.  ``None``
        (default) uses :math:`\log(n/|D|) - \text{variance}/2`
        (:math:`|D|` = spatial area x time span), the same log-normal
        mean correction as :func:`~nstat.extras.spatial.lgcp.lgcp_fit`
        (Moller, Syversveen & Waagepetersen 1998).
    max_iter, tol
        Newton/IRLS stopping controls (outer loop; ``tol`` on the max
        absolute change in ``f`` between iterations).
    jitter
        Diagonal jitter added to each 1-D Matern factor for numerical
        positive-definiteness.

    Returns
    -------
    LGCPSTResult
        Call :meth:`LGCPSTResult.rate_map` for the credible-band spatial
        map at a chosen time, or :meth:`LGCPSTResult.intensity_fn` for a
        callable ``(x, t) -> rate``.

    Notes
    -----
    *Confidence: high* for the Laplace/Newton-IRLS mechanics (identical
    to :func:`~nstat.extras.spatial.lgcp.lgcp_fit`, Rasmussen & Williams
    2006 Alg. 3.1) and for the Kronecker matvec (verified to machine
    precision against a dense ``numpy.kron`` construction in the
    companion test).  *Confidence: moderate* for the posterior-variance
    magnitude specifically — it is a stochastic (Hutchinson) estimate,
    not exact, so credible bands are directionally but not numerically
    exact relative to a hypothetical dense solve.

    References
    ----------
    See the module docstring.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (n, 2); got shape {pts.shape}")
    t_arr = np.asarray(times, dtype=float).reshape(-1)
    if t_arr.shape[0] != pts.shape[0]:
        raise ValueError(
            f"points has {pts.shape[0]} rows but times has {t_arr.shape[0]} entries"
        )

    # Accept either a tuple or a list of (lo, hi) pairs (matching the
    # flexible convention of the sibling spatial modules) and raise a
    # clean ValueError — not a raw TypeError — on malformed input.
    try:
        domain = tuple((float(lo), float(hi)) for lo, hi in domain)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "domain must be a sequence of (lo, hi) pairs for the (x, y) axes, "
            f"e.g. ((0.0, 1.0), (0.0, 1.0)); got {domain!r}"
        ) from exc
    if len(domain) != 2:
        raise ValueError("domain must be a 2-tuple of (lo, hi) pairs for the (x, y) axes")
    for axis, (lo, hi) in zip(("x", "y"), domain):
        if not (hi > lo):
            raise ValueError(
                f"domain {axis}-axis must have hi > lo; got ({lo}, {hi})"
            )
    try:
        period = (float(period[0]), float(period[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"period must be a (tlo, thi) pair; got {period!r}"
        ) from exc
    if not (period[1] > period[0]):
        raise ValueError(
            f"period must have thi > tlo; got {period}"
        )

    grid_norm = tuple(int(g) for g in grid)
    if len(grid_norm) != 3:
        raise ValueError("grid must be a 3-tuple (Gx, Gy, Gt)")
    Gx, Gy, Gt = grid_norm
    if Gx < 1 or Gy < 1 or Gt < 1:
        raise ValueError(f"grid entries must be >= 1; got {grid}")

    nu = float(nu)
    variance = float(variance)
    if variance <= 0:
        raise ValueError("variance must be positive")

    (xlo, xhi), (ylo, yhi) = domain
    tlo, thi = period
    cx, ex, dx = _axis_grid(xlo, xhi, Gx)
    cy, ey, dy = _axis_grid(ylo, yhi, Gy)
    ct, et, dt = _axis_grid(tlo, thi, Gt)
    cell_volume = float(dx * dy * dt)

    grid_x = _spatial_grid(cx, cy)
    grid_t = ct

    # Bin events onto the 3-D (x, y, t) grid.  bin_counts' histogramdd +
    # swapaxes(0, 1) already produces a flat array in exactly the (y, x, t)
    # order this module uses internally (y slow, x mid, t fast) — see
    # module docstring "Axis-order convention" — so no post-hoc transpose
    # is needed here.
    pts3d = np.column_stack([pts[:, 0], pts[:, 1], t_arr])
    y_flat = bin_counts(pts3d, [ex, ey, et])
    shape = (Gy, Gx, Gt)
    M = Gy * Gx * Gt

    # Marginal variance split evenly (cube root) across the three axis
    # factors so a single `variance` scales all axes' informativeness
    # together (see docstring above) while the product reproduces the
    # requested overall marginal variance exactly.
    axis_variance = variance ** (1.0 / 3.0)
    Kx = matern_covariance(cx.reshape(-1, 1), length_scale=length_scale_space,
                            variance=axis_variance, nu=nu, jitter=jitter)
    Ky = matern_covariance(cy.reshape(-1, 1), length_scale=length_scale_space,
                            variance=axis_variance, nu=nu, jitter=jitter)
    Kt = matern_covariance(ct.reshape(-1, 1), length_scale=length_scale_time,
                            variance=axis_variance, nu=nu, jitter=jitter)

    ly, Qy = np.linalg.eigh(Ky)
    lx, Qx = np.linalg.eigh(Kx)
    lt, Qt = np.linalg.eigh(Kt)
    inv_eig = 1.0 / (ly[:, None, None] * lx[None, :, None] * lt[None, None, :])
    inv_eig_flat = inv_eig.reshape(-1)

    kinv_matvec = _make_kinv_matvec(Qy, Qx, Qt, inv_eig_flat, shape)

    total_volume = float((xhi - xlo) * (yhi - ylo) * (thi - tlo))
    if prior_mean is None:
        m0 = float(np.log(max(pts.shape[0], 1) / total_volume) - 0.5 * variance)
    else:
        m0 = float(prior_mean)

    f = np.full(M, m0)
    x_prev = np.zeros(M)
    converged = False
    n_iter = max_iter
    for it in range(max_iter):
        f_clip = np.clip(f, -_ETA_CLIP, _ETA_CLIP)
        lam = cell_volume * np.exp(f_clip)
        rhs = lam * (f - m0) + (y_flat - lam)

        def _precision_matvec(v, _lam=lam):
            return kinv_matvec(v) + _lam * v

        op = LinearOperator((M, M), matvec=_precision_matvec, dtype=float)
        x, _info = cg(op, rhs, x0=x_prev, rtol=_CG_RTOL, maxiter=_CG_MAXITER)
        f_new = m0 + x
        delta = np.max(np.abs(f_new - f))
        x_prev = x
        f = f_new
        if delta < tol:
            converged = True
            n_iter = it + 1
            break

    # Posterior variance diagonal at the converged mode: matrix-free
    # Hutchinson stochastic-diagonal estimator (Hutchinson 1990; Bekas,
    # Kokiopoulou & Saad 2007) against the same CG operator — never forms
    # Sigma = (K^-1 + W)^-1 densely.
    f_clip = np.clip(f, -_ETA_CLIP, _ETA_CLIP)
    lam_final = cell_volume * np.exp(f_clip)

    def _precision_matvec_final(v, _lam=lam_final):
        return kinv_matvec(v) + _lam * v

    op_final = LinearOperator((M, M), matvec=_precision_matvec_final, dtype=float)
    rng = np.random.default_rng(_VARIANCE_RNG_SEED)
    diag_acc = np.zeros(M)
    for _ in range(_N_VARIANCE_PROBES):
        z = rng.choice(np.array([-1.0, 1.0]), size=M)
        xz, _info = cg(op_final, z, rtol=1e-8, maxiter=_CG_MAXITER)
        diag_acc += z * xz
    v_flat = np.clip(diag_acc / _N_VARIANCE_PROBES, 0.0, None)

    counts_out = _to_txy(y_flat, Gy, Gx, Gt)
    f_mode_out = _to_txy(f, Gy, Gx, Gt)
    f_var_out = _to_txy(v_flat, Gy, Gx, Gt)

    return LGCPSTResult(
        grid_x=grid_x,
        grid_t=grid_t,
        counts=counts_out,
        f_mode=f_mode_out,
        f_var=f_var_out,
        cell_volume=cell_volume,
        n_iter=n_iter,
        converged=converged,
    )


__all__ = ["LGCPSTResult", "lgcp_st_fit"]
