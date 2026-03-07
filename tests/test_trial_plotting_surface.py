from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from nstat import CovColl, Covariate, Events, History, Trial, nspikeTrain, nstColl


def test_history_and_events_plot_surface_returns_axes() -> None:
    hist = History([0.0, 0.1, 0.2, 0.4])
    events = Events([0.2, 0.7], ["E1", "E2"])

    ax_hist = hist.plot()
    ax_events = events.plot()

    assert hasattr(ax_hist, "broken_barh")
    assert hasattr(ax_events, "vlines")
    plt.close("all")


def test_covcoll_nstcoll_and_trial_plot_surface_return_matplotlib_objects() -> None:
    t = np.linspace(0.0, 1.0, 1001)
    position = Covariate(t, np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]), "Position", "time", "s", "a.u.", ["x", "y"])
    force = Covariate(t, np.column_stack([np.cos(4 * np.pi * t), np.sin(4 * np.pi * t)]), "Force", "time", "s", "a.u.", ["f_x", "f_y"])
    covs = CovColl([position, force])

    spikes = nstColl(
        [
            nspikeTrain([0.1, 0.2, 0.5, 0.8], name="1", minTime=0.0, maxTime=1.0, makePlots=-1),
            nspikeTrain([0.15, 0.25, 0.45, 0.9], name="2", minTime=0.0, maxTime=1.0, makePlots=-1),
        ]
    )
    trial = Trial(spikes, covs, Events([0.3], ["cue"]), History([0.0, 0.05, 0.1]))

    fig_cov = covs.plot()
    ax_spikes = spikes.plot()
    fig_trial = trial.plot()

    assert len(fig_cov.axes) == 2
    assert hasattr(ax_spikes, "vlines")
    assert len(fig_trial.axes) == 4
    plt.close("all")
