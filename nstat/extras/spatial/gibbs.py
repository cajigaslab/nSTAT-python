r"""Gibbs interaction point-process models and Berman-Turner pseudo-likelihood.

Three pairwise-interaction Gibbs models with closed-form Papangelou
conditional intensities, paired with a Berman-Turner (1992) device that
reformulates the pseudo-likelihood (Besag 1977) as a logistic-Poisson
GLM and solves it by composition with :func:`nstat.glm.fit_poisson_glm`.

Implemented processes:

- :class:`GibbsStrauss` — Strauss (1975) pairwise model with interaction
  radius ``R`` and inhibition parameter ``gamma in (0, 1]`` (``gamma=1``
  recovers the homogeneous Poisson; ``gamma -> 0`` recovers a hard-core
  pattern).
- :class:`HardcoreProcess` — Strauss limit ``gamma = 0``: no two points
  may lie within radius ``R``.
- :class:`AreaInteractionProcess` — Widom & Rowlinson (1970) / Baddeley
  & van Lieshout (1995) area-interaction model with cluster/inhibition
  scalar ``eta > 0`` and discretized union-of-discs sufficient statistic.

Inference:

- :func:`simulate_strauss_birth_death` and the area-interaction sampler
  it dispatches to implement a Metropolis-Hastings birth-death chain
  (Geyer 1999).  :func:`simulate_hardcore_rejection` uses dart-throwing
  with a documented birth-death fallback hint.
- :func:`pseudo_likelihood_fit` — Berman-Turner (1992) quadrature device
  recasting the Besag (1977) pseudo-likelihood as a Poisson GLM.  The
  data + dummy quadrature design feeds straight into
  :func:`nstat.glm.fit_poisson_glm` (proof of composition: this is the
  only entry from ``nstat.extras.spatial.gibbs`` into the core package).

References
----------
- Strauss, D. J. (1975). *A model for clustering.* Biometrika 62(2):467.
- Besag, J. (1977). *Some methods of statistical analysis for spatial
  data.* Bull. Inst. Internat. Statist. 47:77.
- Baddeley, A. & Turner, R. (2000). *Practical maximum pseudolikelihood
  for spatial point patterns.* Aust. N. Z. J. Stat. 42(3):283.
- Berman, M. & Turner, T. R. (1992). *Approximating point process
  likelihoods with GLIM.* Appl. Stat. 41(1):31.
- Geyer, C. J. (1999). *Likelihood inference for spatial point
  processes.* In *Stochastic Geometry: Likelihood and Computation*.
- Widom, B. & Rowlinson, J. S. (1970). *New model for the study of
  liquid-vapor phase transitions.* J. Chem. Phys. 52(4):1670.
- Baddeley, A. J. & van Lieshout, M. N. M. (1995). *Area-interaction
  point processes.* Ann. Inst. Statist. Math. 47(4):601.
- Baddeley, A., Rubak, E. & Turner, R. (2015). *Spatial Point Patterns:
  Methodology and Applications with R.* CRC §13.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from nstat.glm import PoissonGLMResult, fit_poisson_glm
from nstat.extras.spatial.cluster_cox import _validate_window, _window_area


# ----------------------------------------------------------------------
# Parameter dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GibbsStrauss:
    r"""Strauss (1975) pairwise-interaction Gibbs process parameters.

    The Papangelou conditional intensity at a candidate ``u`` given the
    existing pattern ``x`` is

    .. math::

        \lambda^*(u \mid x) = \beta\,\gamma^{t_R(u, x)},

    where :math:`t_R(u, x)` is the number of points of ``x`` within
    distance ``R`` of ``u``.  ``gamma in (0, 1]`` enforces inhibition;
    ``gamma = 1`` recovers the homogeneous Poisson process with rate
    ``beta``.

    Parameters
    ----------
    beta
        First-order activity parameter ``beta > 0``.  Units: events per
        unit area.
    gamma
        Pairwise interaction parameter, ``0 < gamma <= 1``.
    R
        Interaction radius ``R > 0``.

    References
    ----------
    Strauss (1975); Baddeley-Rubak-Turner (2015) §13.4.
    """
    beta: float
    gamma: float
    R: float

    def __post_init__(self) -> None:
        if not (self.beta > 0):
            raise ValueError(f"beta must be positive; got {self.beta}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                f"gamma must lie in (0, 1]; got {self.gamma}"
            )
        if not (self.R > 0):
            raise ValueError(f"R must be positive; got {self.R}")


@dataclass(frozen=True)
class HardcoreProcess:
    r"""Hard-core Gibbs process parameters — Strauss limit ``gamma = 0``.

    No two points may lie within distance ``R``; equivalently, the
    Papangelou conditional intensity is ``beta`` when ``u`` has no
    neighbour within ``R`` and ``0`` otherwise.

    Parameters
    ----------
    beta
        First-order activity parameter ``beta > 0``.
    R
        Hard-core radius ``R > 0``.
    """
    beta: float
    R: float

    def __post_init__(self) -> None:
        if not (self.beta > 0):
            raise ValueError(f"beta must be positive; got {self.beta}")
        if not (self.R > 0):
            raise ValueError(f"R must be positive; got {self.R}")


@dataclass(frozen=True)
class AreaInteractionProcess:
    r"""Widom-Rowlinson / Baddeley-van Lieshout area-interaction process.

    The Papangelou conditional intensity is

    .. math::

        \lambda^*(u \mid x) = \beta\,\eta^{-\Delta U(u, x)},

    where :math:`\Delta U(u, x)` is the differential area of the union
    of radius-``R`` discs centred on the points (Widom-Rowlinson 1970;
    Baddeley-van Lieshout 1995).  ``eta > 1`` favours clustering;
    ``eta < 1`` produces inhibition; ``eta = 1`` recovers Poisson.

    Parameters
    ----------
    beta
        First-order activity parameter ``beta > 0``.
    eta
        Interaction strength ``eta > 0`` (``>1`` clusters, ``<1`` repels).
    R
        Disc radius ``R > 0`` defining the area sufficient statistic.
    """
    beta: float
    eta: float
    R: float

    def __post_init__(self) -> None:
        if not (self.beta > 0):
            raise ValueError(f"beta must be positive; got {self.beta}")
        if not (self.eta > 0):
            raise ValueError(f"eta must be positive; got {self.eta}")
        if not (self.R > 0):
            raise ValueError(f"R must be positive; got {self.R}")


# ----------------------------------------------------------------------
# Sufficient-statistic helpers
# ----------------------------------------------------------------------


def _pairwise_within_R(
    u: np.ndarray, x: np.ndarray, R: float
) -> np.ndarray:
    """Boolean mask: which rows of ``x`` lie within ``R`` of point ``u``.

    ``u`` is a length-2 vector; ``x`` is ``(m, 2)``.
    """
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    d2 = np.sum((x - u) ** 2, axis=1)
    return d2 < R * R


def _strauss_t(u: np.ndarray, x: np.ndarray, R: float) -> int:
    """``t_R(u, x)`` — count of x-points strictly within ``R`` of ``u``."""
    return int(np.sum(_pairwise_within_R(u, x, R)))


def _occupancy_grid(
    x: np.ndarray,
    window: tuple[float, float, float, float],
    R: float,
    pixel_resolution: int,
) -> tuple[np.ndarray, float]:
    """Return ``(C, pixel_area)`` for an ``int16`` disc-occupancy grid.

    ``C[i, j]`` is the number of points whose radius-``R`` disc covers
    the centre of pixel ``(i, j)``.  Returned grid has shape
    ``(pixel_resolution, pixel_resolution)``; the pixel area is the
    factor needed to convert disc-overlap pixel counts into area.
    """
    xmin, ymin, xmax, ymax = window
    nx = ny = int(pixel_resolution)
    px = (xmax - xmin) / nx
    py = (ymax - ymin) / ny
    pixel_area = px * py
    grid = np.zeros((ny, nx), dtype=np.int16)
    if x.shape[0] == 0:
        return grid, pixel_area
    # Discretized disc stamp: precompute the relative pixel offsets that
    # fall within radius R of a disc centre.  At pixel_resolution = 256
    # with R/(xmax-xmin) = 0.05 the stamp is ~13x13.
    rx = max(int(np.ceil(R / px)), 1)
    ry = max(int(np.ceil(R / py)), 1)
    oy, ox = np.meshgrid(
        np.arange(-ry, ry + 1), np.arange(-rx, rx + 1), indexing="ij"
    )
    stamp_mask = (ox * px) ** 2 + (oy * py) ** 2 <= R * R
    so_y, so_x = np.where(stamp_mask)
    # Offset shifted to (-ry, -rx) origin.
    so_y = so_y - ry
    so_x = so_x - rx
    for pt in x:
        cx = int(np.floor((pt[0] - xmin) / px))
        cy = int(np.floor((pt[1] - ymin) / py))
        py_idx = cy + so_y
        px_idx = cx + so_x
        valid = (
            (py_idx >= 0) & (py_idx < ny) & (px_idx >= 0) & (px_idx < nx)
        )
        grid[py_idx[valid], px_idx[valid]] += 1
    return grid, pixel_area


def _area_increment(
    u: np.ndarray,
    grid: np.ndarray,
    pixel_area: float,
    window: tuple[float, float, float, float],
    R: float,
) -> float:
    r"""Differential union area :math:`\Delta U(u, x)` when adding ``u``.

    Computes how many *new* pixels (currently uncovered) would be
    covered by the radius-``R`` disc around ``u``, multiplied by
    ``pixel_area``.  Pixels already covered by another disc do not
    contribute to the increment.
    """
    xmin, ymin, xmax, ymax = window
    ny, nx = grid.shape
    px = (xmax - xmin) / nx
    py = (ymax - ymin) / ny
    rx = max(int(np.ceil(R / px)), 1)
    ry = max(int(np.ceil(R / py)), 1)
    cx = int(np.floor((u[0] - xmin) / px))
    cy = int(np.floor((u[1] - ymin) / py))
    y0 = max(cy - ry, 0)
    y1 = min(cy + ry + 1, ny)
    x0 = max(cx - rx, 0)
    x1 = min(cx + rx + 1, nx)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    sub = grid[y0:y1, x0:x1]
    # Pixel centres in the sub-window.
    iy = np.arange(y0, y1) - cy
    ix = np.arange(x0, x1) - cx
    yy, xx = np.meshgrid(iy, ix, indexing="ij")
    disc_mask = (xx * px) ** 2 + (yy * py) ** 2 <= R * R
    new_mask = disc_mask & (sub == 0)
    return float(np.count_nonzero(new_mask)) * pixel_area


def _stamp_grid(
    u: np.ndarray,
    grid: np.ndarray,
    window: tuple[float, float, float, float],
    R: float,
    *,
    sign: int = +1,
) -> None:
    """In-place add/remove a radius-``R`` disc stamp centred at ``u``."""
    xmin, ymin, xmax, ymax = window
    ny, nx = grid.shape
    px = (xmax - xmin) / nx
    py = (ymax - ymin) / ny
    rx = max(int(np.ceil(R / px)), 1)
    ry = max(int(np.ceil(R / py)), 1)
    cx = int(np.floor((u[0] - xmin) / px))
    cy = int(np.floor((u[1] - ymin) / py))
    y0 = max(cy - ry, 0)
    y1 = min(cy + ry + 1, ny)
    x0 = max(cx - rx, 0)
    x1 = min(cx + rx + 1, nx)
    if y1 <= y0 or x1 <= x0:
        return
    iy = np.arange(y0, y1) - cy
    ix = np.arange(x0, x1) - cx
    yy, xx = np.meshgrid(iy, ix, indexing="ij")
    disc_mask = (xx * px) ** 2 + (yy * py) ** 2 <= R * R
    grid[y0:y1, x0:x1] = grid[y0:y1, x0:x1] + sign * disc_mask.astype(
        np.int16
    )


# ----------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------


def simulate_strauss_birth_death(
    process: GibbsStrauss | AreaInteractionProcess,
    window,
    *,
    n_steps: int = 5000,
    pixel_resolution: int = 256,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Metropolis-Hastings birth-death sampler for Gibbs interaction models.

    For Strauss (Strauss 1975):

    .. math::

        \alpha_b = \min\!\Bigl(1,\; \tfrac{|W|\,\beta\,\gamma^{t_R(u, x)}}{n + 1}\Bigr),

    with the reciprocal for the death move.  For area-interaction the
    factor is :math:`\eta^{-\Delta U(u, x)}` and the discretized
    occupancy grid is maintained incrementally.

    Burn-in is the first 50% of ``n_steps`` (Geyer 1999).

    Parameters
    ----------
    process
        :class:`GibbsStrauss` or :class:`AreaInteractionProcess`.
    window
        ``(xmin, ymin, xmax, ymax)`` rectangle.
    n_steps
        Total birth-death proposals.  Half are dropped as burn-in.
    pixel_resolution
        Side length of the area-interaction occupancy grid.  Ignored for
        Strauss.  Must satisfy ``R >= 2 * pixel_size`` (raise otherwise).
    rng
        ``np.random.Generator`` instance.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` float64 array of points in the window after burn-in.
    """
    if not isinstance(
        process, (GibbsStrauss, AreaInteractionProcess)
    ):
        raise TypeError(
            "process must be GibbsStrauss or AreaInteractionProcess; "
            f"got {type(process).__name__}"
        )
    if not (n_steps >= 2):
        raise ValueError(f"n_steps must be >= 2; got {n_steps}")
    win = _validate_window(window)
    area = _window_area(win)
    xmin, ymin, xmax, ymax = win

    is_area = isinstance(process, AreaInteractionProcess)
    grid: np.ndarray | None = None
    pixel_area = 0.0
    if is_area:
        nx = int(pixel_resolution)
        px = (xmax - xmin) / nx
        py = (ymax - ymin) / nx
        pixel_size = max(px, py)
        if process.R < 2.0 * pixel_size:
            raise ValueError(
                "AreaInteractionProcess R must satisfy R >= 2 * pixel_size "
                f"(pixel_size={pixel_size:.4g}, R={process.R:.4g}); "
                "increase pixel_resolution or R"
            )
        grid = np.zeros((nx, nx), dtype=np.int16)
        pixel_area = px * py

    # Start with an empty pattern.  At equilibrium the chain forgets the
    # initial state; the 50% burn-in is conservative for n_steps >= 2000
    # at the default rates (Geyer 1999).
    points: list[np.ndarray] = []
    burn = n_steps // 2
    snapshot: np.ndarray | None = None
    snapshot_grid: np.ndarray | None = None

    for step in range(int(n_steps)):
        n = len(points)
        # Choose birth or death with 0.5 / 0.5.
        if n == 0 or rng.uniform() < 0.5:
            # Birth proposal.
            u = np.array(
                [
                    rng.uniform(xmin, xmax),
                    rng.uniform(ymin, ymax),
                ],
                dtype=float,
            )
            if isinstance(process, GibbsStrauss):
                x_arr = (
                    np.asarray(points, dtype=float)
                    if n > 0
                    else np.zeros((0, 2), dtype=float)
                )
                t = _strauss_t(u, x_arr, process.R)
                ratio = (
                    area * process.beta * process.gamma**t / (n + 1)
                )
            else:
                # Area-interaction.
                du = _area_increment(
                    u, grid, pixel_area, win, process.R  # type: ignore[arg-type]
                )
                ratio = (
                    area * process.beta * process.eta ** (-du) / (n + 1)
                )
            alpha = min(1.0, float(ratio))
            if rng.uniform() < alpha:
                points.append(u)
                if is_area:
                    _stamp_grid(u, grid, win, process.R, sign=+1)  # type: ignore[arg-type]
        else:
            # Death proposal: pick a uniformly random existing point.
            j = int(rng.integers(0, n))
            u = points[j]
            others = (
                np.asarray(
                    points[:j] + points[j + 1 :], dtype=float
                )
                if n > 1
                else np.zeros((0, 2), dtype=float)
            )
            if isinstance(process, GibbsStrauss):
                t = _strauss_t(u, others, process.R)
                # Reciprocal of birth ratio with the SAME numerator
                # before/after — death is birth^{-1} at the configuration
                # without u.
                ratio = n / (
                    area * process.beta * process.gamma**t
                )
            else:
                # Remove u from the grid temporarily to recompute the
                # increment it brings back upon hypothetical re-birth.
                _stamp_grid(u, grid, win, process.R, sign=-1)  # type: ignore[arg-type]
                du = _area_increment(
                    u, grid, pixel_area, win, process.R  # type: ignore[arg-type]
                )
                ratio = n / (
                    area * process.beta * process.eta ** (-du)
                )
                # Restore — we may not accept the death.
                _stamp_grid(u, grid, win, process.R, sign=+1)  # type: ignore[arg-type]
            alpha = min(1.0, float(ratio))
            if rng.uniform() < alpha:
                if is_area:
                    _stamp_grid(u, grid, win, process.R, sign=-1)  # type: ignore[arg-type]
                points.pop(j)

        if step == burn:
            snapshot = (
                np.asarray(points, dtype=float).copy()
                if points
                else np.zeros((0, 2), dtype=float)
            )
            if is_area:
                snapshot_grid = grid.copy()  # type: ignore[union-attr]

    # Return the final state (post-burn-in).
    out = (
        np.asarray(points, dtype=float)
        if points
        else np.zeros((0, 2), dtype=float)
    )
    # Sanity: snapshot is captured so future verifier tests can ask for
    # the burn-in midpoint; we currently return only the final state.
    del snapshot, snapshot_grid
    return np.ascontiguousarray(out, dtype=np.float64)


def simulate_hardcore_rejection(
    process: HardcoreProcess,
    window,
    *,
    rng: np.random.Generator,
    max_attempts: int = 10000,
) -> np.ndarray:
    r"""Dart-throwing rejection sampler for a hard-core Gibbs process.

    Proposes Poisson(``beta * |W|``) candidates uniformly in the window
    and keeps a candidate only if it sits at distance ``>= R`` from every
    accepted point.  If the acceptance ratio drops below ``0.1`` after
    ``max_attempts`` proposals the function raises with a hint to switch
    to :func:`simulate_strauss_birth_death` with ``gamma=0`` and a
    Strauss-style birth-death chain.

    Parameters
    ----------
    process
        :class:`HardcoreProcess` parameters.
    window
        ``(xmin, ymin, xmax, ymax)`` rectangle.
    rng
        ``np.random.Generator`` instance.
    max_attempts
        Maximum proposals before fallback hint.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` float64 array of points in the window.
    """
    if not isinstance(process, HardcoreProcess):
        raise TypeError(
            "process must be HardcoreProcess; got "
            f"{type(process).__name__}"
        )
    win = _validate_window(window)
    area = _window_area(win)
    xmin, ymin, xmax, ymax = win

    # Target N from the homogeneous-Poisson baseline.  The hard-core
    # constraint removes some — that's expected.
    target_n = max(int(rng.poisson(process.beta * area)), 1)
    accepted: list[np.ndarray] = []
    R2 = process.R * process.R
    attempts = 0
    while attempts < max_attempts and len(accepted) < target_n:
        u = np.array(
            [
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax),
            ],
            dtype=float,
        )
        attempts += 1
        if accepted:
            arr = np.asarray(accepted, dtype=float)
            d2 = np.sum((arr - u) ** 2, axis=1)
            if np.min(d2) < R2:
                continue
        accepted.append(u)

    if attempts > 0 and (len(accepted) / attempts) < 0.1:
        raise RuntimeError(
            f"hard-core dart-throwing acceptance ratio "
            f"({len(accepted)}/{attempts} = "
            f"{len(accepted) / attempts:.3f}) fell below 0.1 after "
            f"{attempts} proposals — the configuration may be too dense. "
            "Switch to simulate_strauss_birth_death with a GibbsStrauss(gamma "
            "near 0) for a birth-death-chain fallback."
        )
    out = (
        np.asarray(accepted, dtype=float)
        if accepted
        else np.zeros((0, 2), dtype=float)
    )
    return np.ascontiguousarray(out, dtype=np.float64)


# ----------------------------------------------------------------------
# Berman-Turner pseudo-likelihood
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GibbsFitResult:
    r"""Output of :func:`pseudo_likelihood_fit`.

    Attributes
    ----------
    model_type
        ``"strauss"``, ``"hardcore"``, or ``"area_interaction"``.
    params
        Decoded interaction parameters; for ``"strauss"`` the keys are
        ``{"beta", "gamma"}``; for ``"hardcore"`` only ``{"beta"}``; for
        ``"area_interaction"`` ``{"beta", "eta"}``.
    R
        Interaction radius used in the design.
    pseudo_log_likelihood
        Besag (1977) pseudo-log-likelihood
        :math:`\sum_i \log \lambda^*(x_i \mid x \setminus \{x_i\}) - \sum_k w_k \lambda_k`,
        recomputed from ``glm_result.coefficients`` per the architect's
        brief §6 (NOT read off the GLM's optimizer state).
    n_data
        Number of data events.
    n_dummy
        Number of Berman-Turner quadrature dummies.
    glm_result
        Underlying :class:`nstat.glm.PoissonGLMResult`.
    """
    model_type: str
    params: dict
    R: float
    pseudo_log_likelihood: float
    n_data: int
    n_dummy: int
    glm_result: PoissonGLMResult


def _leave_one_out_t_strauss(points: np.ndarray, R: float) -> np.ndarray:
    """``t_R(x_i, x \\ {x_i})`` for every data point (vectorised)."""
    n = points.shape[0]
    if n < 2:
        return np.zeros(n, dtype=float)
    # Pairwise squared distances.
    d2 = np.sum(
        (points[:, None, :] - points[None, :, :]) ** 2, axis=2
    )
    mask = d2 < R * R
    np.fill_diagonal(mask, False)
    return mask.sum(axis=1).astype(float)


def _t_data_to_dummies(
    dummies: np.ndarray, data: np.ndarray, R: float
) -> np.ndarray:
    """``t_R(z_k, data)`` for each dummy ``z_k``."""
    if dummies.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if data.shape[0] == 0:
        return np.zeros(dummies.shape[0], dtype=float)
    d2 = np.sum(
        (dummies[:, None, :] - data[None, :, :]) ** 2, axis=2
    )
    return (d2 < R * R).sum(axis=1).astype(float)


def _min_distance(dummies: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Minimum distance from each dummy to any data point."""
    if dummies.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if data.shape[0] == 0:
        return np.full(dummies.shape[0], np.inf, dtype=float)
    d2 = np.sum(
        (dummies[:, None, :] - data[None, :, :]) ** 2, axis=2
    )
    return np.sqrt(np.min(d2, axis=1))


def _area_increment_leave_one_out(
    points: np.ndarray,
    window: tuple[float, float, float, float],
    R: float,
    pixel_resolution: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    r"""Per-data ``\Delta U(x_i, x \ {x_i})`` plus a full occupancy grid.

    Returns ``(delta_data, grid_full, pixel_area, _)``.  The grid is
    built once over all events; per-point increments are read by
    temporarily flipping the i-th disc off.
    """
    grid, pixel_area = _occupancy_grid(
        points, window, R, pixel_resolution
    )
    delta = np.zeros(points.shape[0], dtype=float)
    for i, pt in enumerate(points):
        _stamp_grid(pt, grid, window, R, sign=-1)
        delta[i] = _area_increment(pt, grid, pixel_area, window, R)
        _stamp_grid(pt, grid, window, R, sign=+1)
    return delta, grid, pixel_area, np.array([])


def _area_increment_dummies(
    dummies: np.ndarray,
    full_grid: np.ndarray,
    pixel_area: float,
    window: tuple[float, float, float, float],
    R: float,
) -> np.ndarray:
    """``\\Delta U(z_k, data)`` for each dummy ``z_k`` against full grid."""
    if dummies.shape[0] == 0:
        return np.zeros(0, dtype=float)
    return np.array(
        [
            _area_increment(z, full_grid, pixel_area, window, R)
            for z in dummies
        ],
        dtype=float,
    )


def pseudo_likelihood_fit(
    points: np.ndarray,
    model_type: str,
    window,
    *,
    R: float,
    n_dummy_per_event: int = 10,
    l2: float = 1e-6,
    pixel_resolution: int = 256,
    rng: np.random.Generator | None = None,
) -> GibbsFitResult:
    r"""Berman-Turner (1992) pseudo-likelihood fit of a Gibbs model.

    Reformulates Besag's (1977) pseudo-likelihood as a Poisson GLM by
    augmenting the data events with quadrature dummies (Baddeley-Turner
    2000).  For each model the design matrix has the canonical
    sufficient-statistic columns; intercept absorbs ``log beta``.

    The fit is delegated to :func:`nstat.glm.fit_poisson_glm` via the
    ``(y/w, offset=log w)`` reformulation that avoids needing an
    explicit ``weights`` kwarg on the GLM (architect's brief §3).

    Parameters
    ----------
    points
        ``(n, 2)`` array of observed events.
    model_type
        ``"strauss"``, ``"hardcore"``, or ``"area_interaction"``.
    window
        ``(xmin, ymin, xmax, ymax)`` rectangle.
    R
        Interaction radius.
    n_dummy_per_event
        Number of quadrature dummies per data event.  Total dummies
        ``m = n_dummy_per_event * max(n, 1)``.
    l2
        Ridge term passed to :func:`nstat.glm.fit_poisson_glm`.
    pixel_resolution
        Side length of the area-interaction occupancy grid.  Ignored
        for Strauss / hard-core.
    rng
        ``np.random.Generator`` used to draw the dummies.  If ``None``
        a fresh ``np.random.default_rng()`` is used.

    Returns
    -------
    GibbsFitResult
    """
    if model_type not in {"strauss", "hardcore", "area_interaction"}:
        raise ValueError(
            "model_type must be 'strauss', 'hardcore', or "
            f"'area_interaction'; got {model_type!r}"
        )
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points must have shape (n, 2); got {pts.shape}"
        )
    if not (R > 0):
        raise ValueError(f"R must be positive; got {R}")
    if not (n_dummy_per_event >= 1):
        raise ValueError(
            f"n_dummy_per_event must be >= 1; got {n_dummy_per_event}"
        )
    win = _validate_window(window)
    area = _window_area(win)
    if rng is None:
        rng = np.random.default_rng()

    n = pts.shape[0]
    m = int(n_dummy_per_event * max(n, 1))
    xmin, ymin, xmax, ymax = win
    dummies = np.column_stack(
        [
            rng.uniform(xmin, xmax, size=m),
            rng.uniform(ymin, ymax, size=m),
        ]
    )

    # Quadrature weights.  Dummies share |W| / m; data points get a
    # symbolic weight w_dummy / 1e6 (Baddeley-Turner 2000).
    w_dummy = area / m
    w_data = w_dummy / 1e6
    w = np.concatenate(
        [np.full(n, w_data, dtype=float), np.full(m, w_dummy, dtype=float)]
    )
    # Berman-Turner (1992) Poisson-GLM reformulation.  The weighted
    # pseudo-likelihood ``Σ_k w_k [z_k/w_k · log λ_k - λ_k]`` (with
    # ``z_k = 1[data]``) reduces to the standard unweighted Poisson log-
    # likelihood ``Σ_k [Y_k η_k - exp(η_k + offset_k)]`` with
    # ``Y_k = z_k`` (1 for data, 0 for dummies) and ``offset_k = log w_k``
    # — the mean becomes ``μ_k = w_k λ_k`` and the data term picks up
    # ``log λ_k`` as desired.  Architect's brief §3 calls this the
    # "(y/w, offset=log w)" reformulation; the practical form for a GLM
    # without weights is the equivalent ``Y = z`` (no division), since
    # ``w_k · z_k/w_k = z_k`` exactly.
    y_response = np.concatenate(
        [
            np.full(n, 1.0, dtype=float),
            np.zeros(m, dtype=float),
        ]
    )
    offset = np.log(w)

    if model_type == "hardcore":
        # Validate: no two data within R; restrict dummies to those whose
        # nearest data is > R.
        if n >= 2:
            d2 = np.sum(
                (pts[:, None, :] - pts[None, :, :]) ** 2, axis=2
            )
            np.fill_diagonal(d2, np.inf)
            if np.any(d2 < R * R):
                raise ValueError(
                    "hardcore model: data contains pairs within R="
                    f"{R}; not a valid hard-core configuration"
                )
        min_d = _min_distance(dummies, pts)
        keep_dummy = min_d > R
        m_kept = int(np.count_nonzero(keep_dummy))
        keep = np.concatenate(
            [np.ones(n, dtype=bool), keep_dummy]
        )
        # Rebuild design with no extra columns — intercept only.
        X = np.zeros((n + m, 0), dtype=float)
        glm = fit_poisson_glm(
            X[keep],
            y_response[keep],
            offset=offset[keep],
            include_intercept=True,
            l2=l2,
        )
        beta_hat = float(np.exp(glm.intercept))
        params = {"beta": beta_hat}
        # Recompute pseudo-log-likelihood from coefficients.
        # lambda*(x_i | rest) = beta for kept data; for dropped dummies
        # (within R of some data) lambda* = 0 by hard-core.  The
        # Σ w_k λ_k integral sums only over kept rows.
        log_lam_data = np.full(n, glm.intercept, dtype=float)
        log_lam_dummy = np.full(m_kept, glm.intercept, dtype=float)
        lam_data = np.exp(np.clip(log_lam_data, -20.0, 20.0))
        lam_dummy = np.exp(np.clip(log_lam_dummy, -20.0, 20.0))
        pll = float(
            np.sum(log_lam_data)
            - np.sum(np.full(n, w_data) * lam_data)
            - np.sum(np.full(m_kept, w_dummy) * lam_dummy)
        )
        return GibbsFitResult(
            model_type=model_type,
            params=params,
            R=float(R),
            pseudo_log_likelihood=pll,
            n_data=n,
            n_dummy=m_kept,
            glm_result=glm,
        )

    if model_type == "strauss":
        # Column: t_R for data (leave-one-out) and for dummies (against
        # full data).
        t_data = _leave_one_out_t_strauss(pts, R)
        t_dummy = _t_data_to_dummies(dummies, pts, R)
        X = np.concatenate([t_data, t_dummy])[:, None]
        glm = fit_poisson_glm(
            X,
            y_response,
            offset=offset,
            include_intercept=True,
            l2=l2,
        )
        beta_hat = float(np.exp(glm.intercept))
        gamma_raw = float(np.exp(glm.coefficients[0]))
        if gamma_raw > 1.0:
            warnings.warn(
                "Strauss pseudo-likelihood fit produced gamma > 1 "
                f"(raw {gamma_raw:.4f}) — data appears clustered; "
                "consider fit_thomas",
                UserWarning,
                stacklevel=2,
            )
        gamma_hat = float(min(gamma_raw, 1.0))
        params = {"beta": beta_hat, "gamma": gamma_hat}
        # PLL recomputed from coefficients via the GLM-predicted rate.
        eta_all = glm.intercept + (X @ glm.coefficients).ravel()
        # offset is part of the Poisson log-density via log(w); the
        # *intensity* lambda* is exp(eta_all) WITHOUT the offset.
        lam_all = np.exp(np.clip(eta_all, -20.0, 20.0))
        log_lam_data = eta_all[:n]
        pll = float(
            np.sum(log_lam_data) - np.sum(w * lam_all)
        )
        return GibbsFitResult(
            model_type=model_type,
            params=params,
            R=float(R),
            pseudo_log_likelihood=pll,
            n_data=n,
            n_dummy=m,
            glm_result=glm,
        )

    # area_interaction
    # Pixel-area floor: same rule as the sampler.
    nx = int(pixel_resolution)
    px = (xmax - xmin) / nx
    py = (ymax - ymin) / nx
    pixel_size = max(px, py)
    if R < 2.0 * pixel_size:
        raise ValueError(
            "area_interaction R must satisfy R >= 2 * pixel_size "
            f"(pixel_size={pixel_size:.4g}, R={R:.4g}); increase "
            "pixel_resolution or R"
        )
    delta_data, full_grid, pixel_area, _ = _area_increment_leave_one_out(
        pts, win, R, pixel_resolution
    )
    delta_dummy = _area_increment_dummies(
        dummies, full_grid, pixel_area, win, R
    )
    X = np.concatenate([delta_data, delta_dummy])[:, None]
    # In the area-interaction Papangelou: lambda* = beta * eta^{-Delta U}.
    # So log lambda* = log beta - (log eta) * Delta U; the regression
    # coefficient on Delta U is -log(eta).
    glm = fit_poisson_glm(
        X,
        y_response,
        offset=offset,
        include_intercept=True,
        l2=l2,
    )
    beta_hat = float(np.exp(glm.intercept))
    eta_hat = float(np.exp(-glm.coefficients[0]))
    params = {"beta": beta_hat, "eta": eta_hat}
    eta_all = glm.intercept + (X @ glm.coefficients).ravel()
    lam_all = np.exp(np.clip(eta_all, -20.0, 20.0))
    log_lam_data = eta_all[:n]
    pll = float(np.sum(log_lam_data) - np.sum(w * lam_all))
    return GibbsFitResult(
        model_type=model_type,
        params=params,
        R=float(R),
        pseudo_log_likelihood=pll,
        n_data=n,
        n_dummy=m,
        glm_result=glm,
    )


__all__ = [
    "GibbsStrauss",
    "HardcoreProcess",
    "AreaInteractionProcess",
    "simulate_strauss_birth_death",
    "simulate_hardcore_rejection",
    "pseudo_likelihood_fit",
    "GibbsFitResult",
]
