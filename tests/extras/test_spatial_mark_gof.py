"""Tests for nstat.extras.spatial.mark_gof — mark correlation / variogram.

Synthetic data only.  The estimators are Nadaraya-Watson smooths over
pair distances (Stoyan-Stoyan 1994; Schlather 2001; Cressie-Hawkins
1980).
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial import mark_correlation, mark_variogram


DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def test_mark_correlation_unmarked_is_one():
    """Marks i.i.d. Lognormal independent of the pattern -> the
    Schlather mark correlation hovers around 1 at every lag (Schlather
    2001).  Averaging over n_sim realisations keeps the noise small."""
    rng = np.random.default_rng(21)
    n = 200
    r_grid = np.linspace(0.05, 0.20, 6)
    n_sim = 25
    avg = np.zeros(len(r_grid))
    for s in range(n_sim):
        pts = rng.uniform(0, 1, size=(n, 2))
        marks = rng.lognormal(mean=0.0, sigma=0.5, size=n)
        avg = avg + mark_correlation(pts, marks, r_grid)
    avg /= n_sim
    assert np.all(np.abs(avg - 1.0) < 0.1), (
        f"k_f(r) under independent marks should average to 1; got {avg}"
    )


def test_mark_correlation_detects_mark_clustering():
    """Marks tied to a spatial covariate -> mark correlation at short
    lags departs from 1.  Use marks = f(position) where f is a smooth
    spatial gradient so nearby events have similar marks (positive mark
    dependence)."""
    rng = np.random.default_rng(23)
    n = 500
    pts = rng.uniform(0, 1, size=(n, 2))
    # Smooth spatial mark gradient: a high-frequency sinusoid means
    # nearby pairs have *very* similar marks while far pairs decorrelate.
    marks = 3.0 + np.sin(2.0 * np.pi * 3.0 * pts[:, 0]) + np.sin(
        2.0 * np.pi * 3.0 * pts[:, 1]
    )
    # All marks positive (the sinusoid pair has amplitude <= 2, baseline 3).
    r_grid = np.linspace(0.02, 0.30, 10)
    k = mark_correlation(pts, marks, r_grid)
    # Mark correlation should be greater at short lags than at long lags
    # when marks are spatially smooth (positive mark dependence at small r).
    assert k[0] > k[-1] + 0.02, (
        f"mark correlation should decay with r when marks track position; "
        f"got k={k}"
    )


def test_mark_variogram_constant_mark_is_zero():
    """If every event carries the same mark, the variogram is identically
    zero by construction."""
    rng = np.random.default_rng(25)
    n = 100
    pts = rng.uniform(0, 1, size=(n, 2))
    marks = np.full(n, 3.14)
    r_grid = np.linspace(0.05, 0.20, 6)
    gamma = mark_variogram(pts, marks, r_grid)
    finite = np.isfinite(gamma)
    assert finite.any()
    assert np.allclose(gamma[finite], 0.0)


def test_mark_correlation_kernel_choice():
    """The Schlather and Isham kernels disagree numerically; the "none"
    kernel returns raw (un-normalised) products.  Test that the three
    branches produce distinct numeric output on the same input."""
    rng = np.random.default_rng(27)
    n = 150
    pts = rng.uniform(0, 1, size=(n, 2))
    marks = rng.lognormal(mean=0.5, sigma=0.4, size=n)
    r_grid = np.linspace(0.05, 0.15, 5)
    k_schl = mark_correlation(pts, marks, r_grid, kernel="schlather")
    k_isham = mark_correlation(pts, marks, r_grid, kernel="isham")
    k_none = mark_correlation(pts, marks, r_grid, kernel="none")
    # Different normalisations -> different curves.
    assert not np.allclose(k_schl, k_isham)
    assert not np.allclose(k_schl, k_none)
    # An unknown kernel name is a ValueError.
    with pytest.raises(ValueError, match="kernel must be"):
        mark_correlation(pts, marks, r_grid, kernel="bogus")


def test_mark_correlation_input_shape_mismatch():
    """If marks and points disagree on length, raise ValueError."""
    rng = np.random.default_rng(29)
    pts = rng.uniform(0, 1, size=(30, 2))
    marks = rng.lognormal(size=20)  # wrong length
    r_grid = np.linspace(0.05, 0.15, 4)
    with pytest.raises(ValueError, match="marks must align"):
        mark_correlation(pts, marks, r_grid)
    with pytest.raises(ValueError, match="marks must align"):
        mark_variogram(pts, marks, r_grid)
    # Non-planar points are also rejected.
    pts3 = rng.uniform(0, 1, size=(30, 3))
    marks3 = rng.lognormal(size=30)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        mark_correlation(pts3, marks3, r_grid)
