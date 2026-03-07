from __future__ import annotations

import numpy as np

from nstat.simulators import simulate_point_process, simulate_two_neuron_network


def test_simulate_two_neuron_network_matches_matlab_tutorial_defaults() -> None:
    sim = simulate_two_neuron_network(seed=4)

    assert sim.time.shape == (50001,)
    np.testing.assert_allclose(sim.baseline_mu, np.array([-3.0, -3.0], dtype=float))
    np.testing.assert_allclose(sim.history_kernel, np.array([-4.0, -2.0, -1.0], dtype=float))
    np.testing.assert_allclose(sim.stimulus_kernel, np.array([1.0, -1.0], dtype=float))
    np.testing.assert_allclose(sim.ensemble_kernel, np.array([1.0, -4.0], dtype=float))
    np.testing.assert_allclose(sim.actual_network, np.array([[0.0, 1.0], [-4.0, 0.0]], dtype=float))
    np.testing.assert_allclose(
        sim.lambda_delta[:5],
        np.array(
            [
                [0.04742587, 0.04742587],
                [0.04771053, 0.04714283],
                [0.0479968, 0.0468614],
                [0.04828468, 0.04658159],
                [0.04857417, 0.0463034],
            ],
            dtype=float,
        ),
        rtol=1e-7,
        atol=1e-9,
    )
    assert [sim.spikes.getNST(i + 1).n_spikes for i in range(2)] == [2590, 2365]


def test_simulate_point_process_retains_rate_and_time_shape() -> None:
    time = np.array([0.0, 0.1, 0.2, 0.3], dtype=float)
    rate = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)

    sim = simulate_point_process(time, rate, seed=1)

    np.testing.assert_allclose(sim.time, time)
    np.testing.assert_allclose(sim.rate_hz, rate)
    assert np.all(sim.spikes.spikeTimes >= time.min())
    assert np.all(sim.spikes.spikeTimes <= time.max())
