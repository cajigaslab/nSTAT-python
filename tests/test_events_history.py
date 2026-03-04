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


def test_history_design_matrix_matches_naive_reference() -> None:
    rng = np.random.default_rng(7)
    spikes = np.sort(rng.random(400) * 2.0)
    grid = np.linspace(0.0, 2.0, 250)
    hb = HistoryBasis(bin_edges_s=np.array([0.0, 0.01, 0.03, 0.07, 0.1]))

    fast = hb.design_matrix(spike_times_s=spikes, time_grid_s=grid)

    ref = np.zeros_like(fast)
    for i, t_now in enumerate(grid):
        lags = t_now - spikes
        for j in range(hb.n_bins):
            lo = hb.bin_edges_s[j]
            hi = hb.bin_edges_s[j + 1]
            ref[i, j] = float(np.sum((lags > lo) & (lags <= hi)))

    np.testing.assert_allclose(fast, ref, atol=0.0, rtol=0.0)
