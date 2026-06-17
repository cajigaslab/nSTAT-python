r"""Cluster Cox processes — Thomas, Matérn-cluster, and generic Neyman-Scott.

A pure-NumPy/SciPy simulator and closed-form pair-correlation library for
the three canonical *cluster* Cox processes:

- :class:`ThomasProcess` — homogeneous Poisson parents with
  :math:`{\rm Poisson}(\mu)` offspring counts displaced by an isotropic
  2-D Gaussian of standard deviation :math:`\sigma` (Thomas 1949).
- :class:`MaternClusterProcess` — same parent model with offspring
  uniformly displaced inside a disc of radius :math:`R` (Matérn 1986).
- :class:`NeymanScottCox` — the generic Neyman-Scott Cox process
  (Neyman & Scott 1958), parameterised by a user-supplied offspring
  displacement kernel.

These complete the :mod:`nstat.extras.spatial` Cox-process catalogue
alongside the :class:`~nstat.extras.spatial.lgcp.LGCPResult`
log-Gaussian Cox process: the LGCP captures *smoothly varying* intensity,
while the cluster models capture *concentrated bursts* — different
biological hypotheses about non-Poisson clustering in neural place fields
and afferent dendrite ensembles.

The closed-form pair correlations are the targets a minimum-contrast
estimator (see :mod:`nstat.extras.spatial.inference`) recovers from an
empirical :math:`\hat g(r)` measured by
:func:`nstat.extras.spatial.spatial_gof.pair_correlation`.

References
----------
- Thomas, M. (1949). *A generalization of Poisson's binomial limit for use in
  ecology.* Biometrika 36(1/2):18.
- Matérn, B. (1986). *Spatial Variation* (2nd ed.). Springer Lecture Notes
  in Statistics 36.
- Neyman, J. & Scott, E. L. (1958). *Statistical approach to problems of
  cosmology.* J. R. Stat. Soc. B 20(1):1.
- Møller, J. & Waagepetersen, R. P. (2003). *Statistical Inference and
  Simulation for Spatial Point Processes.* Chapman & Hall §5.3.
- Diggle, P. J. (2013). *Statistical Analysis of Spatial and
  Spatio-Temporal Point Patterns* (3rd ed.). CRC §6.2.1.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ----------------------------------------------------------------------
# Window validation
# ----------------------------------------------------------------------


def _validate_window(window) -> tuple[float, float, float, float]:
    """Return a validated ``(xmin, ymin, xmax, ymax)`` rectangle.

    The cluster-process simulators use the flat 4-tuple convention
    because they parent-buffer (``window`` padded by an ``r_pad``) and
    that arithmetic is cleaner with ``(xmin, ymin, xmax, ymax)`` than
    with the nested ``((xmin, xmax), (ymin, ymax))`` form used by the
    SOIRS goodness-of-fit module.
    """
    arr = tuple(float(v) for v in window)
    if len(arr) != 4:
        raise ValueError(
            "window must be a 4-tuple (xmin, ymin, xmax, ymax); "
            f"got length {len(arr)}"
        )
    xmin, ymin, xmax, ymax = arr
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(
            "window must satisfy xmax > xmin and ymax > ymin; "
            f"got ({xmin}, {ymin}, {xmax}, {ymax})"
        )
    return xmin, ymin, xmax, ymax


def _window_area(window) -> float:
    xmin, ymin, xmax, ymax = _validate_window(window)
    return (xmax - xmin) * (ymax - ymin)


# ----------------------------------------------------------------------
# Process parameter classes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ThomasProcess:
    r"""Parameters of a Thomas cluster Cox process.

    Parents are a homogeneous Poisson process at intensity
    ``intensity_parent``; each parent emits an independent
    :math:`{\rm Poisson}(\mu)` cluster whose offspring are displaced by
    an isotropic 2-D Gaussian of standard deviation ``sigma`` (Thomas
    1949; Møller-Waagepetersen 2003 §5.3).

    Parameters
    ----------
    intensity_parent
        Parent Poisson intensity :math:`\lambda_p` (events per unit area).
    mu_offspring
        Mean offspring count per parent :math:`\mu`.
    sigma
        Gaussian displacement standard deviation :math:`\sigma`.

    Notes
    -----
    The mean intensity of the resulting Cox process is
    :math:`\lambda = \lambda_p\,\mu`; the closed-form pair correlation is

    .. math::

        g(r) = 1 + \frac{\exp(-r^2 / (4\sigma^2))}{4\pi\sigma^2 \lambda_p}.
    """
    intensity_parent: float
    mu_offspring: float
    sigma: float

    def __post_init__(self) -> None:
        if not (self.intensity_parent > 0):
            raise ValueError(
                f"intensity_parent must be positive; got {self.intensity_parent}"
            )
        if not (self.mu_offspring > 0):
            raise ValueError(
                f"mu_offspring must be positive; got {self.mu_offspring}"
            )
        if not (self.sigma > 0):
            raise ValueError(f"sigma must be positive; got {self.sigma}")


@dataclass(frozen=True)
class MaternClusterProcess:
    r"""Parameters of a Matérn cluster Cox process.

    Parents are a homogeneous Poisson process at intensity
    ``intensity_parent``; each parent emits
    :math:`{\rm Poisson}(\mu)` offspring uniformly inside a disc of
    radius ``radius`` (Matérn 1986; Møller-Waagepetersen 2003 §5.3).

    Parameters
    ----------
    intensity_parent
        Parent Poisson intensity :math:`\lambda_p`.
    mu_offspring
        Mean offspring count per parent :math:`\mu`.
    radius
        Disc radius :math:`R` of the uniform offspring displacement.
    """
    intensity_parent: float
    mu_offspring: float
    radius: float

    def __post_init__(self) -> None:
        if not (self.intensity_parent > 0):
            raise ValueError(
                f"intensity_parent must be positive; got {self.intensity_parent}"
            )
        if not (self.mu_offspring > 0):
            raise ValueError(
                f"mu_offspring must be positive; got {self.mu_offspring}"
            )
        if not (self.radius > 0):
            raise ValueError(f"radius must be positive; got {self.radius}")


@dataclass(frozen=True)
class NeymanScottCox:
    r"""Generic Neyman-Scott cluster Cox process.

    Parents are a homogeneous Poisson process at intensity
    ``intensity_parent``; each parent emits :math:`{\rm Poisson}(\mu)`
    offspring whose 2-D displacements are sampled by a user-supplied
    ``offspring_kernel(n, rng)`` returning an ``(n, 2)`` array
    (Neyman-Scott 1958; Møller-Waagepetersen 2003 §5.3).

    Parameters
    ----------
    intensity_parent
        Parent Poisson intensity :math:`\lambda_p`.
    mu_offspring
        Mean offspring count per parent :math:`\mu`.
    offspring_kernel
        Callable ``(n_offspring, rng) -> (n_offspring, 2)`` ndarray of
        offspring displacements from the parent location.
    pad
        Parent-window buffer.  If ``0`` the simulator emits a warning
        because the kernel's effective support is unknown and the
        in-window pattern will be edge-biased.

    Notes
    -----
    Pass an explicit ``pad`` greater than the kernel's effective support
    (e.g. the 99.9% quantile of the displacement magnitude) to avoid
    boundary bias.  For Thomas / Matérn-cluster the bundled simulators
    set ``pad = 3*sigma`` / ``pad = radius`` automatically.
    """
    intensity_parent: float
    mu_offspring: float
    offspring_kernel: Callable[[int, np.random.Generator], np.ndarray]
    pad: float = 0.0

    def __post_init__(self) -> None:
        if not (self.intensity_parent > 0):
            raise ValueError(
                f"intensity_parent must be positive; got {self.intensity_parent}"
            )
        if not (self.mu_offspring > 0):
            raise ValueError(
                f"mu_offspring must be positive; got {self.mu_offspring}"
            )
        if not callable(self.offspring_kernel):
            raise ValueError(
                "offspring_kernel must be a callable (n, rng) -> (n, 2)"
            )
        if self.pad < 0:
            raise ValueError(f"pad must be non-negative; got {self.pad}")


# ----------------------------------------------------------------------
# Closed-form pair correlations
# ----------------------------------------------------------------------


def thomas_pair_correlation(
    r: np.ndarray,
    sigma: float,
    intensity_parent: float,
    mu_offspring: float,  # noqa: ARG001 (kept for API symmetry; see Notes)
) -> np.ndarray:
    r"""Closed-form pair correlation of a Thomas process.

    .. math::

        g(r) = 1 + \frac{\exp(-r^2 / (4\sigma^2))}{4\pi\,\sigma^2\,\lambda_p}

    (Thomas 1949; Møller-Waagepetersen 2003 §5.3; Diggle 2013 §6.2.1).
    Note ``mu_offspring`` does not enter the pair correlation — it is
    accepted only so the closed-form ``g_model_fn`` shares a signature
    with the simulator constructor when scripting parameter sweeps.

    Parameters
    ----------
    r
        Distances at which to evaluate :math:`g`.
    sigma
        Gaussian displacement standard deviation.
    intensity_parent
        Parent Poisson intensity :math:`\lambda_p`.
    mu_offspring
        Unused; accepted for API symmetry only.

    Returns
    -------
    np.ndarray
        :math:`g(r)` at each requested distance.
    """
    if not (sigma > 0):
        raise ValueError(f"sigma must be positive; got {sigma}")
    if not (intensity_parent > 0):
        raise ValueError(
            f"intensity_parent must be positive; got {intensity_parent}"
        )
    r = np.asarray(r, dtype=float)
    return 1.0 + np.exp(-(r**2) / (4.0 * sigma**2)) / (
        4.0 * np.pi * sigma**2 * intensity_parent
    )


def matern_cluster_pair_correlation(
    r: np.ndarray,
    radius: float,
    intensity_parent: float,
    mu_offspring: float,  # noqa: ARG001 (kept for API symmetry; see Notes)
) -> np.ndarray:
    r"""Closed-form pair correlation of a Matérn cluster process.

    For ``r <= 2R``,

    .. math::

        g(r) = 1 + \frac{h(r; R)}{\pi R^2 \lambda_p},
        \qquad
        h(r; R) = \frac{1}{\pi}\Bigl[
            2\arccos\!\bigl(\tfrac{r}{2R}\bigr)
            - \tfrac{r}{R}\sqrt{1 - \tfrac{r^2}{4R^2}}
        \Bigr],

    and :math:`g(r) = 1` for :math:`r > 2R` (Matérn 1986;
    Møller-Waagepetersen 2003 §5.3; Diggle 2013 §6.2.1).
    """
    if not (radius > 0):
        raise ValueError(f"radius must be positive; got {radius}")
    if not (intensity_parent > 0):
        raise ValueError(
            f"intensity_parent must be positive; got {intensity_parent}"
        )
    r = np.asarray(r, dtype=float)
    g = np.ones_like(r)
    mask = r <= 2.0 * radius
    if np.any(mask):
        rm = r[mask]
        ratio = rm / (2.0 * radius)
        # Clip to keep arccos and sqrt arguments numerically valid at the
        # boundary (small float overshoots from r == 2R).
        ratio_c = np.clip(ratio, 0.0, 1.0)
        h = (1.0 / np.pi) * (
            2.0 * np.arccos(ratio_c)
            - (rm / radius) * np.sqrt(np.clip(1.0 - ratio_c**2, 0.0, None))
        )
        g[mask] = 1.0 + h / (np.pi * radius**2 * intensity_parent)
    return g


# ----------------------------------------------------------------------
# Simulators
# ----------------------------------------------------------------------


def _draw_parents(
    intensity_parent: float,
    window: tuple[float, float, float, float],
    pad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Homogeneous-Poisson parents on the padded window."""
    xmin, ymin, xmax, ymax = window
    pxmin, pymin = xmin - pad, ymin - pad
    pxmax, pymax = xmax + pad, ymax + pad
    parent_area = (pxmax - pxmin) * (pymax - pymin)
    n_parents = rng.poisson(intensity_parent * parent_area)
    if n_parents == 0:
        return np.zeros((0, 2), dtype=float)
    px = rng.uniform(pxmin, pxmax, size=n_parents)
    py = rng.uniform(pymin, pymax, size=n_parents)
    return np.column_stack([px, py])


def _gaussian_offspring(
    n: int, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Isotropic 2-D Gaussian displacements."""
    return rng.normal(0.0, sigma, size=(n, 2))


def _uniform_disc_offspring(
    n: int, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Uniformly distributed displacements on the closed disc of radius ``R``."""
    # Polar form with the sqrt(U) radial draw — uniform-density on the disc.
    u = rng.uniform(0.0, 1.0, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = radius * np.sqrt(u)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _scatter_offspring(
    parents: np.ndarray,
    mu_offspring: float,
    offspring_kernel: Callable[[int, np.random.Generator], np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw Poisson(mu) offspring per parent and apply ``offspring_kernel``.

    Returns ``(offspring_xy, parent_index)`` so callers that want the
    parent label per kept offspring can reconstruct it without redoing
    the count.
    """
    if parents.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)
    counts = rng.poisson(mu_offspring, size=parents.shape[0])
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)
    parent_index = np.repeat(np.arange(parents.shape[0]), counts)
    disp = offspring_kernel(total, rng)
    disp = np.asarray(disp, dtype=float)
    if disp.shape != (total, 2):
        raise ValueError(
            "offspring_kernel must return shape (n, 2); "
            f"got {disp.shape} for n={total}"
        )
    xy = parents[parent_index] + disp
    return xy, parent_index


def _crop_to_window(
    xy: np.ndarray, window: tuple[float, float, float, float]
) -> np.ndarray:
    xmin, ymin, xmax, ymax = window
    if xy.shape[0] == 0:
        return xy
    mask = (
        (xy[:, 0] >= xmin)
        & (xy[:, 0] <= xmax)
        & (xy[:, 1] >= ymin)
        & (xy[:, 1] <= ymax)
    )
    return xy[mask]


def simulate_thomas(
    intensity_parent: float,
    mu_offspring: float,
    sigma: float,
    window,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Simulate a Thomas process on a rectangular window.

    Parents are drawn from a homogeneous Poisson process on the window
    padded by ``3*sigma`` (so 99.7% of clusters spawned outside the
    window can still place offspring inside it).  Each parent emits
    :math:`{\rm Poisson}(\mu)` offspring displaced by an isotropic
    Gaussian; only offspring within the window are returned.

    Parameters
    ----------
    intensity_parent, mu_offspring, sigma
        Thomas parameters; see :class:`ThomasProcess`.
    window
        ``(xmin, ymin, xmax, ymax)`` rectangle.
    rng
        ``np.random.Generator`` instance.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` float64 array of points in the window.
    """
    if not (intensity_parent > 0):
        raise ValueError(
            f"intensity_parent must be positive; got {intensity_parent}"
        )
    if not (mu_offspring > 0):
        raise ValueError(f"mu_offspring must be positive; got {mu_offspring}")
    if not (sigma > 0):
        raise ValueError(f"sigma must be positive; got {sigma}")
    win = _validate_window(window)
    pad = 3.0 * sigma
    parents = _draw_parents(intensity_parent, win, pad, rng)
    xy, _ = _scatter_offspring(
        parents,
        mu_offspring,
        lambda n, r: _gaussian_offspring(n, sigma, r),
        rng,
    )
    return np.ascontiguousarray(_crop_to_window(xy, win), dtype=np.float64)


def simulate_matern_cluster(
    intensity_parent: float,
    mu_offspring: float,
    radius: float,
    window,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Simulate a Matérn cluster process on a rectangular window.

    Parents are drawn on the window padded by ``radius`` (the kernel's
    exact support); offspring are uniform on the disc of that radius.
    """
    if not (intensity_parent > 0):
        raise ValueError(
            f"intensity_parent must be positive; got {intensity_parent}"
        )
    if not (mu_offspring > 0):
        raise ValueError(f"mu_offspring must be positive; got {mu_offspring}")
    if not (radius > 0):
        raise ValueError(f"radius must be positive; got {radius}")
    win = _validate_window(window)
    pad = radius
    parents = _draw_parents(intensity_parent, win, pad, rng)
    xy, _ = _scatter_offspring(
        parents,
        mu_offspring,
        lambda n, r: _uniform_disc_offspring(n, radius, r),
        rng,
    )
    return np.ascontiguousarray(_crop_to_window(xy, win), dtype=np.float64)


def simulate_neyman_scott(
    process: NeymanScottCox,
    window,
    *,
    rng: np.random.Generator,
    return_parents: bool = False,
):
    r"""Simulate a generic Neyman-Scott cluster Cox process.

    Parameters
    ----------
    process
        :class:`NeymanScottCox` carrying the parent intensity, offspring
        rate, displacement kernel, and parent-window pad.
    window
        ``(xmin, ymin, xmax, ymax)`` rectangle.
    rng
        ``np.random.Generator``.
    return_parents
        If ``True``, return ``(offspring_xy, parents_xy)`` — the parent
        locations are *before* cropping; offspring is cropped to
        ``window``.

    Returns
    -------
    np.ndarray or tuple
        Offspring ``(n, 2)`` array, or ``(offspring, parents)`` when
        ``return_parents=True``.
    """
    if not isinstance(process, NeymanScottCox):
        raise TypeError(
            "process must be a NeymanScottCox instance; got "
            f"{type(process).__name__}"
        )
    win = _validate_window(window)
    if process.pad == 0.0:
        warnings.warn(
            "NeymanScottCox.pad == 0 — the parent window is not buffered. "
            "Offspring kernels with support outside the window will produce "
            "an edge-biased pattern. Pass an explicit `pad` greater than "
            "the kernel's effective support.",
            stacklevel=2,
        )
    parents = _draw_parents(process.intensity_parent, win, process.pad, rng)
    xy, _ = _scatter_offspring(
        parents, process.mu_offspring, process.offspring_kernel, rng
    )
    cropped = np.ascontiguousarray(_crop_to_window(xy, win), dtype=np.float64)
    if return_parents:
        return cropped, np.ascontiguousarray(parents, dtype=np.float64)
    return cropped


__all__ = [
    "ThomasProcess",
    "MaternClusterProcess",
    "NeymanScottCox",
    "thomas_pair_correlation",
    "matern_cluster_pair_correlation",
    "simulate_thomas",
    "simulate_matern_cluster",
    "simulate_neyman_scott",
]
