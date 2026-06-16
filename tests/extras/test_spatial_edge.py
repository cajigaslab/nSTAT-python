"""Tests for nstat.extras.spatial._edge — per-pair edge-correction weights.

Synthetic data only (np.random.default_rng); no patient data.

The closed-form Ripley isotropic weight (:func:`frac_disc_in_rect`) is
validated against a Monte-Carlo estimate at the configurations that
matter most: an event near a corner whose disc clips BOTH adjacent
sides (requires the inclusion-exclusion second-order term to be
correct), the trivial interior case, and a side-only clip.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial._edge import (
    border_usable_mask,
    frac_disc_in_rect,
    frac_translation_rect,
)

DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def _mc_frac_disc_in_rect(p, r, domain, n: int, rng):
    """Monte-Carlo estimate of the fraction of the circle of radius r
    centred at p that lies inside the rectangular domain.  Samples
    uniform angles on the circle (the disc's perimeter), so this is the
    *arc-length* fraction — exactly what the closed form computes."""
    (xlo, xhi), (ylo, yhi) = domain
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    xs = p[0] + r * np.cos(theta)
    ys = p[1] + r * np.sin(theta)
    inside = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
    return float(np.mean(inside))


def test_frac_disc_in_rect_interior_event_is_one():
    """Disc entirely inside the window -> fraction = 1.0."""
    # Centre at (0.5, 0.5) with r=0.1 fits well inside [0,1]^2.
    f = frac_disc_in_rect(np.array([0.5, 0.5]), r=0.1, domain=DOMAIN)
    assert abs(f - 1.0) < 1e-12


def test_frac_disc_in_rect_side_only_clip_matches_mc():
    """Event near a single side (not a corner) — disc clips one half-plane only."""
    rng = np.random.default_rng(0)
    # Centre 0.05 from the left side, r=0.10; far from top/bottom/right.
    p = np.array([0.05, 0.5])
    r = 0.10
    f_closed = frac_disc_in_rect(p, r=r, domain=DOMAIN)
    f_mc = _mc_frac_disc_in_rect(p, r, DOMAIN, n=20_000, rng=rng)
    assert abs(f_closed - f_mc) < 0.02, (
        f"closed={f_closed:.4f} vs MC={f_mc:.4f} differ by more than 2%"
    )


def test_frac_disc_in_rect_corner_overlap():
    """Event near a corner whose disc clips BOTH adjacent sides AND wraps
    the corner (i.e. sqrt(dx^2 + dy^2) < r).  Requires the inclusion-
    exclusion corner term in the closed form to be correct; the test is
    against a Monte-Carlo perimeter estimate within +/- 2%."""
    rng = np.random.default_rng(7)
    # Centre 0.03 from BOTH the left and bottom sides; r=0.10 so:
    #   - the disc clips both left and bottom half-planes (0.03 < 0.10), AND
    #   - sqrt(0.03^2 + 0.03^2) ~ 0.0424 < 0.10, so the disc wraps the corner.
    p = np.array([0.03, 0.03])
    r = 0.10
    f_closed = frac_disc_in_rect(p, r=r, domain=DOMAIN)
    f_mc = _mc_frac_disc_in_rect(p, r, DOMAIN, n=50_000, rng=rng)
    assert 0.0 < f_closed <= 1.0
    assert abs(f_closed - f_mc) < 0.02, (
        f"corner-overlap closed form ({f_closed:.4f}) disagrees with MC "
        f"({f_mc:.4f}) by more than 2% — the inclusion-exclusion second-"
        f"order correction is probably wrong."
    )


def test_frac_disc_in_rect_corner_no_wrap():
    """Event near a corner where the disc clips both adjacent sides but
    does NOT wrap the corner (sqrt(dx^2+dy^2) >= r).  The corner overlap
    term must contribute zero — verify the closed form still tracks MC."""
    rng = np.random.default_rng(11)
    # dx = dy = 0.08; sqrt(0.08^2 + 0.08^2) = 0.1131 > r = 0.10.
    p = np.array([0.08, 0.08])
    r = 0.10
    f_closed = frac_disc_in_rect(p, r=r, domain=DOMAIN)
    f_mc = _mc_frac_disc_in_rect(p, r, DOMAIN, n=50_000, rng=rng)
    assert abs(f_closed - f_mc) < 0.02


def test_frac_translation_rect_symmetric_and_positive():
    """Ohser weight is symmetric in sign of the offset and >= 1."""
    domain = ((0.0, 2.0), (0.0, 3.0))
    w_pp = frac_translation_rect(0.5, 0.7, domain)
    w_nn = frac_translation_rect(-0.5, -0.7, domain)
    w_pn = frac_translation_rect(0.5, -0.7, domain)
    assert w_pp == pytest.approx(w_nn)
    assert w_pp == pytest.approx(w_pn)
    assert w_pp >= 1.0


def test_frac_translation_rect_disjoint_is_inf():
    """Offset larger than the window side -> intersection empty -> +inf."""
    domain = ((0.0, 1.0), (0.0, 1.0))
    assert np.isinf(frac_translation_rect(1.5, 0.0, domain))
    assert np.isinf(frac_translation_rect(0.0, 2.0, domain))


def test_border_usable_mask_basic():
    """Boundary-distance >= r: only the deep-interior point qualifies."""
    pts = np.array(
        [
            [0.5, 0.5],   # boundary distance 0.5
            [0.02, 0.5],  # boundary distance 0.02
            [0.5, 0.99],  # boundary distance 0.01
        ]
    )
    m = border_usable_mask(pts, ((0.0, 1.0), (0.0, 1.0)), r=0.1)
    assert m.tolist() == [True, False, False]
