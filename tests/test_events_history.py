import numpy as np

from nstat.events import Events
from nstat.history import HistoryBasis


def test_events_subset() -> None:
    e = Events(times=np.array([0.1, 0.3, 0.7]), labels=["a", "b", "c"])
    sub = e.subset(0.2, 1.0)
    assert sub.times.tolist() == [0.3, 0.7]


def test_history_design_matrix() -> None:
    hb = HistoryBasis(bin_edges_s=np.array([0.0, 0.05, 0.1]))
    mat = hb.design_matrix(spike_times_s=np.array([0.15, 0.22]), time_grid_s=np.array([0.25, 0.3]))
    assert mat.shape == (2, 2)
