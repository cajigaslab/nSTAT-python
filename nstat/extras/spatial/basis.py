r"""Tensor-product B-spline basis on a rectangular spatial domain.

A dependency-free factory for log-intensity bases on a rectangle, designed
to feed :func:`nstat.glm.fit_poisson_glm` as the ``x`` argument.  The 2-D
basis is the tensor product of two clamped uniform 1-D B-spline bases
(`de Boor 1978 <https://doi.org/10.1007/978-1-4612-6333-3>`_); the
companion :meth:`BSplineBasis2D.gram` returns the **P-spline second-difference
penalty** of Eilers & Marx (1996), so a downstream caller can either fit
the unpenalized basis directly or add ``rho * gram`` to the Hessian for a
smooth penalty.

Conventions
-----------
Row order in the 2-D design matrix is ``indexing="ij"``: row ``i*Ny + j``
evaluates at ``(grid_x[i], grid_y[j])``.  This is **different** from
:func:`nstat.extras.spatial._kernels.make_grid`, which uses
``indexing="xy"``.  Reshape with::

    rate_field = (B @ coef + intercept).reshape(len(grid_x), len(grid_y))

Knot placement
--------------
Uniform only.  ``clamped=True`` (default) repeats endpoints ``degree+1``
times so the partition of unity holds at the boundary; ``clamped=False``
uses a non-clamped uniform knot vector spanning the domain (still useful
when the basis is intended for a wider region).

References
----------
- de Boor C (1978). *A Practical Guide to Splines.* Springer.
- Eilers PHC, Marx BD (1996). *Flexible Smoothing with B-splines and
  Penalties.* Statistical Science 11(2):89-121.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline


# ----------------------------------------------------------------------
# Knot vector construction
# ----------------------------------------------------------------------


def _clamped_uniform_knots(
    a: float, b: float, n_knots: int, degree: int
) -> np.ndarray:
    """Clamped uniform knot vector on ``[a, b]``.

    The returned vector has length ``n_knots + degree + 1`` with each
    endpoint repeated ``degree + 1`` times and ``n_knots - degree - 1``
    evenly-spaced interior breaks.
    """
    n_interior = n_knots - degree - 1
    if n_interior < 0:
        # Caller validates n_knots >= degree + 1 first; reaching here is a bug.
        raise ValueError(
            f"n_knots must be >= degree + 1; got n_knots={n_knots}, degree={degree}"
        )
    if n_interior == 0:
        interior = np.empty(0, dtype=float)
    else:
        # Place interior breaks strictly inside (a, b) — exclude endpoints.
        interior = np.linspace(a, b, n_interior + 2)[1:-1]
    return np.concatenate(
        [np.full(degree + 1, float(a)), interior, np.full(degree + 1, float(b))]
    )


def _nonclamped_uniform_knots(
    a: float, b: float, n_knots: int, degree: int
) -> np.ndarray:
    """Non-clamped uniform knot vector covering ``[a, b]``.

    The knots are uniformly spaced with step ``(b - a) / (n_knots - degree)``;
    ``t[degree] == a`` and ``t[n_knots] == b``, so the basis evaluated on
    the grid is well-defined with ``extrapolate=False``.
    """
    if n_knots <= degree:
        raise ValueError(
            f"n_knots must be > degree for non-clamped uniform; got {n_knots} <= {degree}"
        )
    dx = (b - a) / (n_knots - degree)
    nt = n_knots + degree + 1
    return a + (np.arange(nt, dtype=float) - degree) * dx


# ----------------------------------------------------------------------
# 1-D and 2-D basis builders
# ----------------------------------------------------------------------


def bspline_basis_1d(
    grid: np.ndarray,
    n_knots: int,
    degree: int = 3,
    clamped: bool = True,
) -> np.ndarray:
    r"""Evaluate a 1-D B-spline basis on a grid.

    Parameters
    ----------
    grid
        Sorted 1-D evaluation points; must lie in the basis support
        (``[grid.min(), grid.max()]`` defines the support).
    n_knots
        Number of basis functions (i.e. spline coefficients).  Must
        satisfy ``n_knots >= degree + 1``.
    degree
        Spline polynomial degree (default 3 → cubic).
    clamped
        If ``True`` (default), endpoints are repeated ``degree+1`` times
        so the partition of unity holds at the boundary.  If ``False``,
        a uniform non-clamped knot vector is used.

    Returns
    -------
    np.ndarray
        Dense ``(N, n_knots)`` design matrix.  Row ``i`` evaluates each
        basis function at ``grid[i]``.  The rows sum to 1 (partition of
        unity) up to floating-point error.

    Raises
    ------
    ValueError
        If ``degree < 0`` or ``n_knots < degree + 1``.

    Notes
    -----
    Implemented via :func:`scipy.interpolate.BSpline.design_matrix`.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0; got {degree}")
    if n_knots < degree + 1:
        raise ValueError(
            f"n_knots must be >= degree + 1; got n_knots={n_knots}, degree={degree}"
        )

    g = np.asarray(grid, dtype=float).ravel()
    if g.size == 0:
        return np.zeros((0, n_knots), dtype=float)
    a, b = float(g.min()), float(g.max())
    if not (b > a):
        # Degenerate grid (single point or all-equal); just return ones at
        # the partition-of-unity slot.  Build knots over [a, a + eps] and
        # evaluate to recover a sensible row vector by extension.
        a_eff, b_eff = a, a + 1.0
    else:
        a_eff, b_eff = a, b

    if clamped:
        knots = _clamped_uniform_knots(a_eff, b_eff, n_knots, degree)
    else:
        knots = _nonclamped_uniform_knots(a_eff, b_eff, n_knots, degree)

    # design_matrix requires x within [t[k], t[n_knots]]; clip away tiny
    # float drift at the right endpoint that would otherwise raise.
    eps = 1e-12 * max(1.0, abs(b_eff - a_eff))
    g_clipped = np.clip(g, a_eff, b_eff - eps if b_eff > a_eff else b_eff)
    B = BSpline.design_matrix(g_clipped, knots, degree).toarray()
    return np.asarray(B, dtype=float)


def bspline_basis_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_knots: int | tuple[int, int],
    degree: int = 3,
    domain: str = "rect",
    clamped: bool = True,
) -> np.ndarray:
    r"""Tensor-product 2-D B-spline basis on a rectangular grid.

    The returned design matrix has ``Nx * Ny`` rows and ``nx * ny``
    columns, where ``Nx = len(grid_x)``, ``Ny = len(grid_y)``, and
    ``(nx, ny)`` is ``n_knots`` (a single int is broadcast to both axes).
    Row ``i*Ny + j`` evaluates each tensor-product basis function at
    ``(grid_x[i], grid_y[j])`` — i.e. ``indexing="ij"`` flattening
    (x-axis outer, y-axis inner).  **Note**: this is different from
    :func:`nstat.extras.spatial._kernels.make_grid`, which uses
    ``indexing="xy"``.

    To reshape a length-``Nx*Ny`` predicted-rate vector back to a 2-D
    field::

        rate2d = pred.reshape(len(grid_x), len(grid_y))   # ij layout

    Parameters
    ----------
    grid_x, grid_y
        Sorted 1-D evaluation grids per axis.
    n_knots
        Number of basis functions per axis.  Scalar or ``(nx, ny)``.
    degree
        Polynomial degree (default 3).
    domain
        ``"rect"`` for the tensor product on a rectangle (only mode
        implemented).  ``"circular"`` raises :class:`NotImplementedError`.
    clamped
        Whether to use clamped (endpoint-repeating) uniform knots
        (default ``True``).

    Returns
    -------
    np.ndarray
        Dense ``(Nx*Ny, nx*ny)`` design matrix.  Suitable as the ``x``
        argument to :func:`nstat.glm.fit_poisson_glm`.
    """
    if domain == "circular":
        raise NotImplementedError("circular domain stub — use rect for now")
    if domain != "rect":
        raise ValueError(f"domain must be 'rect' or 'circular'; got {domain!r}")
    if isinstance(n_knots, tuple):
        nx, ny = int(n_knots[0]), int(n_knots[1])
    else:
        nx = ny = int(n_knots)

    Bx = bspline_basis_1d(grid_x, nx, degree=degree, clamped=clamped)
    By = bspline_basis_1d(grid_y, ny, degree=degree, clamped=clamped)
    Nx, Ny = Bx.shape[0], By.shape[0]
    K = nx * ny
    # Tensor product with ij flattening: out[i*Ny + j, a*ny + b] = Bx[i,a] * By[j,b].
    out = (Bx[:, None, :, None] * By[None, :, None, :]).reshape(Nx * Ny, K)
    return out


# ----------------------------------------------------------------------
# Frozen dataclass facade
# ----------------------------------------------------------------------


def _difference_penalty(n: int, order: int = 2) -> np.ndarray:
    """Return :math:`D` such that ``D @ beta`` is the ``order``-th forward
    difference.  ``D.T @ D`` is the P-spline penalty (Eilers-Marx 1996)."""
    return np.diff(np.eye(n), n=order, axis=0)


def _greville(
    grid: np.ndarray, n_knots: int, degree: int, clamped: bool
) -> np.ndarray:
    """Return Greville abscissae for a 1-D B-spline basis (de Boor 1978).

    The :math:`i`-th Greville abscissa is the mean of the ``degree``
    interior knots ``t[i+1], ..., t[i+degree]`` from the same knot vector
    :func:`bspline_basis_1d` uses; it is the canonical anchor location of
    the :math:`i`-th basis coefficient.

    Parameters mirror :func:`bspline_basis_1d`; the knot vector is
    reconstructed via :func:`_clamped_uniform_knots` / :func:`_nonclamped_uniform_knots`
    so the abscissae stay consistent with the design matrix even if the
    knot helpers ever change.
    """
    g = np.asarray(grid, dtype=float).ravel()
    if g.size == 0:
        a_eff, b_eff = 0.0, 1.0
    else:
        a, b = float(g.min()), float(g.max())
        a_eff, b_eff = (a, b) if b > a else (a, a + 1.0)
    if clamped:
        t = _clamped_uniform_knots(a_eff, b_eff, n_knots, degree)
    else:
        t = _nonclamped_uniform_knots(a_eff, b_eff, n_knots, degree)
    if degree == 0:
        # Greville reduces to the midpoint of [t[i], t[i+1]] when degree=0;
        # the formula t[i+1:i+1].mean() of an empty slice is NaN, so guard.
        return 0.5 * (t[:-1] + t[1:])
    return np.array(
        [float(t[i + 1 : i + degree + 1].mean()) for i in range(n_knots)],
        dtype=float,
    )



@dataclass(frozen=True)
class BSplineBasis2D:
    """Frozen container for a 2-D tensor-product B-spline basis.

    Attributes
    ----------
    grid_x, grid_y
        The 1-D evaluation grids the basis was built on.
    n_knots_x, n_knots_y
        Number of basis functions per axis.
    degree
        Polynomial degree.
    clamped
        Whether the underlying knot vectors are clamped.
    _design
        Cached ``(Nx*Ny, nx*ny)`` design matrix (ij-flattened — see
        :func:`bspline_basis_2d`).
    """

    grid_x: np.ndarray
    grid_y: np.ndarray
    n_knots_x: int
    n_knots_y: int
    degree: int
    clamped: bool
    _design: np.ndarray = field(repr=False, compare=False)

    @classmethod
    def from_grid(
        cls,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        n_knots: int | tuple[int, int],
        degree: int = 3,
        clamped: bool = True,
    ) -> "BSplineBasis2D":
        """Build a basis on ``(grid_x, grid_y)`` with ``n_knots`` per axis."""
        if isinstance(n_knots, tuple):
            nx, ny = int(n_knots[0]), int(n_knots[1])
        else:
            nx = ny = int(n_knots)
        B = bspline_basis_2d(
            grid_x, grid_y, (nx, ny), degree=degree, domain="rect", clamped=clamped
        )
        return cls(
            grid_x=np.asarray(grid_x, dtype=float),
            grid_y=np.asarray(grid_y, dtype=float),
            n_knots_x=nx,
            n_knots_y=ny,
            degree=degree,
            clamped=clamped,
            _design=B,
        )

    def design_matrix(self) -> np.ndarray:
        """Return the dense ``(Nx*Ny, nx*ny)`` design matrix (ij layout)."""
        return self._design

    def coefficient_coords(self) -> np.ndarray:
        r"""Greville-abscissa anchor points for the basis coefficients.

        Returns the tensor product of per-axis Greville abscissae (de Boor
        1978) in **ij flattening** (x outer, y inner), matching the column
        order of :meth:`design_matrix`: column ``a*ny + b`` has its anchor
        at ``(greville_x[a], greville_y[b])``.

        These coordinates are the canonical spatial locations of the
        coefficients themselves — used by
        :func:`nstat.extras.spatial.lgcp.lgcp_fit_glm` to evaluate a
        Matern GP prior directly on the coefficient vector, sidestepping
        the per-cell ``K`` matrix and producing an :math:`O(K^3)` rather
        than :math:`O(M^3)` cost.

        Returns
        -------
        np.ndarray
            ``(n_knots_x * n_knots_y, 2)`` array of coefficient anchor
            coordinates.

        References
        ----------
        de Boor C (1978). *A Practical Guide to Splines.* Springer.
        """
        ax = _greville(self.grid_x, self.n_knots_x, self.degree, self.clamped)
        ay = _greville(self.grid_y, self.n_knots_y, self.degree, self.clamped)
        XX, YY = np.meshgrid(ax, ay, indexing="ij")
        return np.column_stack([XX.ravel(), YY.ravel()])


    def gram(self) -> np.ndarray:
        r"""P-spline 2-D second-difference penalty (Eilers & Marx 1996).

        Returns the ``(nx*ny, nx*ny)`` matrix

        .. math::

            P = D_x^\top D_x \otimes I_y + I_x \otimes D_y^\top D_y ,

        where :math:`D_x`, :math:`D_y` are second-order forward-difference
        operators on the marginal coefficient vectors.  Symmetric and
        positive semi-definite by construction.
        """
        nx, ny = self.n_knots_x, self.n_knots_y
        Dx = _difference_penalty(nx, order=2)
        Dy = _difference_penalty(ny, order=2)
        Ix = np.eye(nx)
        Iy = np.eye(ny)
        Px = np.kron(Dx.T @ Dx, Iy)
        Py = np.kron(Ix, Dy.T @ Dy)
        return Px + Py


__all__ = [
    "bspline_basis_1d",
    "bspline_basis_2d",
    "BSplineBasis2D",
]
