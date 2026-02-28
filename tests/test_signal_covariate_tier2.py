from __future__ import annotations

import numpy as np

from nstat.signal import Covariate, Signal



def test_signal_tier2_methods() -> None:
    t = np.linspace(0.0, 1.0, 101)
    y = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    sig = Signal(time=t, data=y, name="sig")

    merged = sig.merge(sig)
    assert merged.n_channels == 4

    ds = sig.derivative()
    integ = sig.integral()
    assert ds.data.shape == sig.data.shape
    assert integ.shape == sig.data.shape

    rs = sig.resample(50.0)
    assert np.isclose(rs.sample_rate_hz, 50.0, atol=1e-6)

    sub = sig.get_sub_signal([0])
    assert sub.n_channels == 1



def test_covariate_tier2_methods() -> None:
    t = np.linspace(0.0, 1.0, 51)
    y = np.column_stack([np.sin(2 * np.pi * t), np.sin(2 * np.pi * t + 0.2)])
    cov = Covariate(time=t, data=y, name="stim", labels=["s1", "s2"])

    mu, lo, hi = cov.compute_mean_plus_ci(axis=1)
    assert mu.shape == t.shape
    assert lo.shape == t.shape
    assert hi.shape == t.shape

    sub = cov.get_sub_signal("s2")
    assert sub.labels == ["s2"]

    payload = cov.to_structure()
    rec = Covariate.from_structure(payload)
    assert rec.labels == cov.labels
    assert np.allclose(rec.data, cov.data)
