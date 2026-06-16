"""Tests for nstat.extras.spatial.marked_gof.rescaled_acf.

Lag-autocorrelation of the rescaled-time variates with a Bartlett band
(Brown et al. 2002; Andersen 1997; Truccolo et al. 2005).  Synthetic
data only.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial import RescaledACFResult, rescaled_acf


def test_rescaled_acf_iid_unif_within_band():
    """Under i.i.d. Uniform(0,1) -- the rescaled-time null -- the sample
    ACF at every lag is inside the +/- 1.96/sqrt(n) Bartlett band on
    average."""
    rng = np.random.default_rng(11)
    n = 2000
    u = rng.uniform(size=n)
    res = rescaled_acf(u, n_lags=15)
    # Asymptotic: in-band at >= 95% of lags under the null.
    assert res.inside_band.mean() >= 0.85
    # The band is tied to the sample size.
    assert abs(res.band - 1.96 / np.sqrt(n)) < 1e-12


def test_rescaled_acf_correlated_uniforms_exceeds_band():
    """A strongly serially correlated z-series -> at least one positive
    short-lag exceeds the asymptotic band."""
    rng = np.random.default_rng(13)
    n = 1500
    z = rng.standard_normal(n)
    # AR(1) coupling with phi=0.7 -> short-lag autocorr ~ 0.7.
    phi = 0.7
    for k in range(1, n):
        z[k] = phi * z[k - 1] + np.sqrt(1 - phi**2) * z[k]
    # Back-transform z (standard normal) into uniforms.
    from scipy.stats import norm
    u = norm.cdf(z)
    res = rescaled_acf(u, n_lags=10)
    # Lag-1 sample autocorr is well outside the band.
    assert abs(res.acf[0]) > res.band
    # And explicitly: not all lags are inside.
    assert not res.inside_band.all()


def test_rescaled_acf_shape_and_dataclass():
    """Frozen-dataclass contract: shapes, dtypes, and field types."""
    rng = np.random.default_rng(17)
    n_lags = 8
    u = rng.uniform(size=200)
    res = rescaled_acf(u, n_lags=n_lags)
    assert isinstance(res, RescaledACFResult)
    assert res.lags.shape == (n_lags,)
    assert res.acf.shape == (n_lags,)
    assert res.inside_band.shape == (n_lags,)
    assert res.lags.dtype.kind == "i"
    assert np.array_equal(res.lags, np.arange(1, n_lags + 1))
    assert isinstance(res.band, float)
    # Frozen: cannot reassign.
    with pytest.raises(Exception):
        res.band = 0.0


def test_rescaled_acf_rejects_short_input():
    """If u has fewer than n_lags + 2 values, raise ValueError."""
    u = np.linspace(0.1, 0.9, 5)
    with pytest.raises(ValueError, match="n_lags"):
        rescaled_acf(u, n_lags=10)
    with pytest.raises(ValueError, match="n_lags"):
        rescaled_acf(np.linspace(0.1, 0.9, 100), n_lags=0)
