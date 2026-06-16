r"""Tests for nstat.extras.spatial.bartlett.bartlett_density_from_pcf.

Hankel-zero transform of (g(r) - 1) -- the spatial Bartlett spectral
density (Bartlett 1964; Stein 1999 §3; Moller-Syversveen-Waagepetersen
1998 for the LGCP).  Synthetic data only.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial import bartlett_density_from_pcf


def test_bartlett_density_lgcp_closed_form():
    r"""For a 2-D isotropic Gaussian covariance C(r) = sigma^2 exp(-r^2 /
    (2 ell^2)), the Hankel-zero transform is the closed form

        S(k) = 2 pi sigma^2 ell^2 exp(-k^2 ell^2 / 2).

    For an LGCP with small log-rate variance, g(r) - 1 ~= C(r), so the
    estimator must match the closed form across the *body* of the
    wavenumber grid (excluding the top decile near k_max, which is
    sensitive to the lag-grid truncation per Stein 1999 §3).
    """
    sigma2 = 0.05  # small -> g - 1 ~ C is a good approximation
    ell = 0.10
    # Lag grid: dense, long enough that C(r_max) ~ 0.
    r = np.linspace(0.005, 0.6, 600)
    C = sigma2 * np.exp(-(r**2) / (2.0 * ell**2))
    g = 1.0 + C  # use g - 1 = C exactly so we test the Hankel quadrature

    # Restrict the comparison to the BODY of the default wavenumber grid
    # where the closed form is non-negligible.  Above 10% of S(0) is the
    # signal-bearing band of the log-spaced grid; below that the closed
    # form decays exponentially and a relative-error metric loses meaning.
    k, S_hat = bartlett_density_from_pcf(r, g)
    S_true = 2.0 * np.pi * sigma2 * (ell**2) * np.exp(-(k**2) * (ell**2) / 2.0)

    body = S_true > 0.01 * S_true[0]
    # Exclude the top decile of THAT body if it touches k_max (Stein 1999 §3
    # truncation sensitivity).
    body_idx = np.where(body)[0]
    cutoff = body_idx[int(0.9 * body_idx.size)] if body_idx.size else 0
    use = np.zeros_like(body)
    use[:cutoff] = body[:cutoff]

    err = np.abs(S_hat[use] - S_true[use]) / np.abs(S_true[use])
    assert np.median(err) < 0.05, (
        f"Hankel quadrature deviates from the Gaussian closed form: "
        f"median rel-err = {np.median(err):.4f}; max body err = {err.max():.4f}"
    )


def test_bartlett_density_default_k_grid_returned():
    """When ``k_grid=None`` the default 64-point log-spaced grid spans
    from pi / r_max to pi / dr_min."""
    r = np.linspace(0.01, 0.5, 200)
    g = np.ones_like(r) + 0.01 * np.exp(-(r**2) / (2 * 0.05**2))
    k, S = bartlett_density_from_pcf(r, g)
    assert k.shape == S.shape
    assert k.size == 64
    dr_min = float(np.diff(r).min())
    assert abs(k[0] - np.pi / r[-1]) < 1e-9
    assert abs(k[-1] - np.pi / dr_min) < 1e-6
    # Strictly increasing (log-spaced).
    assert np.all(np.diff(k) > 0)


def test_bartlett_density_input_validation():
    """Reject misaligned/empty/non-positive inputs."""
    r = np.linspace(0.01, 0.5, 100)
    g = np.ones_like(r)
    # Mis-aligned shapes.
    with pytest.raises(ValueError, match="align"):
        bartlett_density_from_pcf(r, g[:-1])
    # Non-positive r.
    bad_r = r.copy()
    bad_r[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        bartlett_density_from_pcf(bad_r, g)
    # Non-monotone r.
    bad_r2 = r.copy()
    bad_r2[10] = bad_r2[9]
    with pytest.raises(ValueError, match="strictly increasing"):
        bartlett_density_from_pcf(bad_r2, g)
    # Too few lags.
    with pytest.raises(ValueError, match="at least 2"):
        bartlett_density_from_pcf(np.array([0.1]), np.array([1.0]))
    # Non-positive k_grid.
    with pytest.raises(ValueError, match="strictly positive"):
        bartlett_density_from_pcf(r, g, k_grid=np.array([1.0, 0.0, 2.0]))
    # Empty k_grid.
    with pytest.raises(ValueError, match="non-empty"):
        bartlett_density_from_pcf(r, g, k_grid=np.array([]))
