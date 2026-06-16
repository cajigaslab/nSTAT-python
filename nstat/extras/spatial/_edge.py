r"""Edge-correction primitives for inhomogeneous second-order estimators.

Closed-form per-pair edge weights for a rectangular observation window
``W = [xlo, xhi] x [ylo, yhi]``.  Used by
:mod:`nstat.extras.spatial.spatial_gof` when the ``edge_correction``
keyword selects one of ``"isotropic"``, ``"translation"``, or ``"border"``.

Three published corrections are implemented:

- **Isotropic** (Ripley 1976, 1977): for a pair :math:`(i, j)` separated
  by distance :math:`r`, weight by the inverse fraction of the circle of
  radius :math:`r` centred at one event that lies inside :math:`W`.  The
  closed form for a rectangle is the sum of central angles subtended by
  the four boundary chords.
- **Translation** (Ohser 1983): weight by :math:`|W| / |W \cap W_h|`,
  where :math:`W_h` is :math:`W` shifted by the inter-event offset
  :math:`h = x_i - x_j`.  Symmetric in :math:`i \leftrightarrow j`.
- **Border** (Baddeley, Rubak & Turner 2015, eq. 7.5): restrict the
  estimator to events whose distance to the boundary of :math:`W` is at
  least :math:`r`.  Returns a boolean usability mask per event.

References
----------
- Ripley BD (1976). *The second-order analysis of stationary point
  processes.* J. Appl. Probab. 13(2):255-266.
- Ripley BD (1977). *Modelling spatial patterns.* JRSS-B 39(2):172-212.
- Ohser J (1983). *On estimators for the reduced second moment measure
  of point processes.* Math. Operationsforsch. Statist., Ser. Statist.
  14(1):63-71.
- Baddeley A, Rubak E, Turner R (2015). *Spatial Point Patterns:
  Methodology and Applications with R.* CRC Press.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Ripley isotropic correction
# ----------------------------------------------------------------------


def _arc_fraction_inside_rect(
    cx: float, cy: float, r: float, xlo: float, xhi: float, ylo: float, yhi: float
) -> float:
    """Fraction of the circle of radius ``r`` centred at ``(cx, cy)`` that
    lies inside the rectangle ``[xlo, xhi] x [ylo, yhi]``.

    Computed by inclusion-exclusion on the four boundary half-planes:
    sum the four single-side cut arcs, then add back the four pairwise
    corner overlaps (the doubly-excluded arc that wraps a corner).
    Triple overlaps are zero whenever the rectangle is at least as wide
    and tall as ``2r``, which the border-correction precondition already
    guarantees in practice.  Returns a value in ``[0, 1]``.
    """
    if r <= 0:
        return 1.0
    # Signed distances from the centre to each of the four sides
    # (positive when the centre is inside the rectangle relative to that
    # side).  A side cuts the circle iff ``|d| < r``.
    dL, dR = cx - xlo, xhi - cx
    dB, dT = cy - ylo, yhi - cy

    def arc_outside(d: float) -> float:
        """Full arc-angle of the circle lying outside the half-plane
        defined by a single side at signed perpendicular distance ``d``
        from the centre.  ``2 * arccos(d/r)`` when the side cuts the
        circle; ``0`` if it does not; ``2*pi`` if the centre is entirely
        outside that half-plane."""
        if d >= r:
            return 0.0
        if d <= -r:
            return 2.0 * np.pi
        return 2.0 * np.arccos(d / r)

    # First-order: total angle the circle spends outside each half-plane
    # (sum of full arc-angles, not half-angles).
    excluded = arc_outside(dL) + arc_outside(dR) + arc_outside(dB) + arc_outside(dT)

    # Second-order: subtract back the doubly-counted corner overlaps.  A
    # corner at perpendicular distances ``(dx, dy) > 0`` has an overlap
    # angle iff ``sqrt(dx^2 + dy^2) < r``; the closed-form overlap is
    # ``pi/2 - arccos(dx/r) - arccos(dy/r)`` (the small arc that wraps
    # the corner outside BOTH half-planes).
    def corner_overlap(dx: float, dy: float) -> float:
        if dx >= r or dy >= r:
            return 0.0
        if dx * dx + dy * dy >= r * r:
            return 0.0
        a = np.arccos(min(max(dx / r, -1.0), 1.0))
        b = np.arccos(min(max(dy / r, -1.0), 1.0))
        return max(0.0, a + b - 0.5 * np.pi)

    # Four corners: (xlo, ylo), (xhi, ylo), (xlo, yhi), (xhi, yhi).
    # The relevant perpendicular distances at each corner are the
    # signed distances to its two adjacent sides — clipped at zero (a
    # centre OUTSIDE one of those sides is handled correctly by the
    # half_angle term and contributes no extra corner overlap).
    overlaps = (
        corner_overlap(max(dL, 0.0), max(dB, 0.0))
        + corner_overlap(max(dR, 0.0), max(dB, 0.0))
        + corner_overlap(max(dL, 0.0), max(dT, 0.0))
        + corner_overlap(max(dR, 0.0), max(dT, 0.0))
    )

    frac_outside = (excluded - overlaps) / (2.0 * np.pi)
    # Numerical clamp to [0, 1].
    frac = min(1.0, max(0.0, 1.0 - frac_outside))
    return float(frac)


def frac_disc_in_rect(p: np.ndarray, r: float, domain) -> float:
    """Ripley isotropic correction (fraction of circle inside the window).

    Parameters
    ----------
    p
        ``(2,)`` event coordinates.
    r
        Pair-distance radius (must be positive).
    domain
        Rectangular window ``((xlo, xhi), (ylo, yhi))``.

    Returns
    -------
    float
        Fraction in ``(0, 1]``.  Never returns 0; the smallest positive
        value is clipped at ``1 / (2*pi*r)``-scale lower bound to keep the
        reciprocal weight finite when an event sits exactly at a corner.
    """
    (xlo, xhi), (ylo, yhi) = domain
    cx, cy = float(p[0]), float(p[1])
    frac = _arc_fraction_inside_rect(cx, cy, float(r), xlo, xhi, ylo, yhi)
    # Numerical floor — at most three arcs can vanish the disc; the
    # reciprocal is unbounded only if the event sits exactly at a corner.
    return max(frac, 1e-12)


# ----------------------------------------------------------------------
# Ohser translation correction
# ----------------------------------------------------------------------


def frac_translation_rect(dx: float, dy: float, domain) -> float:
    r"""Translation edge correction (Ohser 1983) for a rectangle.

    Returns :math:`|W| / |W \cap W_h|`, where :math:`W_h` is the window
    shifted by ``(dx, dy)``.  Symmetric in the sign of ``(dx, dy)``
    because the intersection area depends only on ``|dx|`` and ``|dy|``.

    Returns
    -------
    float
        Weight in ``[1, +inf]``.  ``+inf`` if the shifted window is
        disjoint (``|dx| >= xhi-xlo`` or ``|dy| >= yhi-ylo``).
    """
    (xlo, xhi), (ylo, yhi) = domain
    Lx, Ly = float(xhi - xlo), float(yhi - ylo)
    ax, ay = abs(float(dx)), abs(float(dy))
    if ax >= Lx or ay >= Ly:
        return float("inf")
    overlap = (Lx - ax) * (Ly - ay)
    area = Lx * Ly
    if overlap <= 0:
        return float("inf")
    return area / overlap


# ----------------------------------------------------------------------
# Border-method usable mask
# ----------------------------------------------------------------------


def border_usable_mask(pts: np.ndarray, domain, r: float) -> np.ndarray:
    """Boolean ``(n,)`` mask of events with boundary-distance ``>= r``.

    These are the events that can be used as the *focal* point in a
    border-corrected second-order estimator at radius ``r`` — no part of
    a disc of that radius can leave the window.
    """
    pts = np.asarray(pts, dtype=float)
    (xlo, xhi), (ylo, yhi) = domain
    dx = np.minimum(pts[:, 0] - xlo, xhi - pts[:, 0])
    dy = np.minimum(pts[:, 1] - ylo, yhi - pts[:, 1])
    bdist = np.minimum(dx, dy)
    return bdist >= float(r)


__all__ = [
    "frac_disc_in_rect",
    "frac_translation_rect",
    "border_usable_mask",
]
