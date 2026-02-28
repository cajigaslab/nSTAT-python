import numpy as np

from nstat.signal import Covariate, Signal


def test_signal_shape_and_rate() -> None:
    t = np.linspace(0.0, 1.0, 101)
    s = Signal(time=t, data=np.sin(2 * np.pi * t), name="sig")
    assert s.n_samples == 101
    assert s.n_channels == 1
    assert 99.0 < s.sample_rate_hz < 101.0


def test_covariate_labels() -> None:
    t = np.linspace(0.0, 1.0, 21)
    x = np.column_stack([t, t**2])
    c = Covariate(time=t, data=x, name="poly", labels=["x", "x2"])
    assert c.n_channels == 2
    assert c.labels == ["x", "x2"]
