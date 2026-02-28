import numpy as np

from nstat.confidence import ConfidenceInterval


def test_confidence_width_and_contains() -> None:
    t = np.linspace(0.0, 1.0, 11)
    ci = ConfidenceInterval(time=t, lower=np.zeros_like(t), upper=np.ones_like(t))
    assert np.allclose(ci.width(), 1.0)
    assert np.all(ci.contains(0.5 * np.ones_like(t)))
