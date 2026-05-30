"""Tests for ``nstat.extras.decoding.clusterless_bridge``.

The bridge wraps
`replay_trajectory_classification <https://github.com/Eden-Kramer-Lab/replay_trajectory_classification>`_
(Denovellis 2021, eLife; MIT).  Like the other JAX-heavy bridges (e.g.
the dynamax EM bridge) the functional path is gated on the optional
library being installed — the import-hint contract is what the base
suite enforces; the functional smoke test runs only when
``[clusterless]`` is present.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_clusterless_bridge_emits_install_hint_when_missing() -> None:
    """When ``replay_trajectory_classification`` is absent, the bridge
    raises a clear ImportError naming the pip-install extras key."""
    try:
        import replay_trajectory_classification  # noqa: F401
        pytest.skip("replay_trajectory_classification is installed; import-error path unreachable")
    except ImportError:
        pass

    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_decoder

    with pytest.raises(ImportError) as excinfo:
        fit_clusterless_decoder(
            position=np.zeros((10, 1)),
            multiunits=np.zeros((10, 4, 2)),
        )
    assert "pip install nstat-toolbox[clusterless]" in str(excinfo.value)


def test_clusterless_bridge_module_imports_without_backing_library() -> None:
    """Importing the bridge module must NOT require the optional dep —
    only *calling* the public functions does.  This is the same
    discipline as :mod:`nstat.extras.em.dynamax_bridge`.
    """
    import nstat.extras.decoding.clusterless_bridge as m

    assert hasattr(m, "fit_clusterless_decoder")
    assert hasattr(m, "fit_clusterless_classifier")
    assert hasattr(m, "ClusterlessDecoderResult")
    assert hasattr(m, "ClusterlessClassifierResult")
    assert "fit_clusterless_decoder" in m.__all__


def test_input_validation_rejects_bad_shapes() -> None:
    """Shape contracts are checked before the optional library is
    touched — so we get a clean ``ValueError`` instead of an opaque
    upstream traceback."""
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_decoder

    # position must be 1- or 2-D
    with pytest.raises(ValueError, match="position must be shape"):
        fit_clusterless_decoder(
            position=np.zeros((10, 2, 3)),
            multiunits=np.zeros((10, 4, 2)),
        )
    # multiunits must be 3-D (T, n_marks, n_electrodes)
    with pytest.raises(ValueError, match="multiunits must be shape"):
        fit_clusterless_decoder(
            position=np.zeros((10, 1)),
            multiunits=np.zeros((10, 4)),
        )
    # time axes must match
    with pytest.raises(ValueError, match="share the same n_time"):
        fit_clusterless_decoder(
            position=np.zeros((10, 1)),
            multiunits=np.zeros((9, 4, 2)),
        )


def test_classifier_state_names_validation() -> None:
    """Empty ``state_names`` is rejected upfront."""
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_classifier

    with pytest.raises(ValueError, match="at least one name"):
        fit_clusterless_classifier(
            position=np.zeros((10, 1)),
            multiunits=np.zeros((10, 4, 2)),
            state_names=[],
        )


def test_clusterless_decoder_runs_on_synthetic_data() -> None:
    """Functional smoke: a small synthetic 1-D trajectory + multiunit
    cube runs end-to-end and returns a well-shaped posterior.

    Gated on ``replay_trajectory_classification`` being installed —
    skipped in the default base CI; exercised under
    ``pip install nstat-toolbox[clusterless]``.
    """
    pytest.importorskip("replay_trajectory_classification")
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_decoder

    rng = np.random.default_rng(0)
    n_time, n_marks, n_electrodes = 200, 4, 3
    # A simple back-and-forth 1-D trajectory in [0, 100].
    t = np.arange(n_time)
    position = 50.0 + 45.0 * np.sin(2 * np.pi * t / n_time).reshape(-1, 1)
    # Multiunits: mostly NaN (no spikes); a sparse Poisson of mark
    # vectors tied loosely to position (small numerical-stability fudge).
    multiunits = np.full((n_time, n_marks, n_electrodes), np.nan)
    spike_mask = rng.random(n_time) < 0.3
    for t_i in np.flatnonzero(spike_mask):
        for e in range(n_electrodes):
            if rng.random() < 0.5:
                # Mark features as small Gaussians scaled by position bin index.
                multiunits[t_i, :, e] = rng.normal(loc=position[t_i, 0] / 20.0, size=n_marks)

    result = fit_clusterless_decoder(
        position, multiunits,
        place_bin_size=5.0,
    )

    # Shape contract: time axis present; posterior + MAP have aligned T.
    assert result.posterior.shape[0] == n_time
    assert result.map_position.shape[0] == n_time
    assert result.causal_posterior.shape == result.posterior.shape
    assert np.all(np.isfinite(result.posterior))
    # Posterior is a probability distribution along the position axes.
    sums = result.posterior.reshape(n_time, -1).sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), (
        f"posterior should be normalized; row sums in "
        f"[{sums.min():.4f}, {sums.max():.4f}]"
    )


def test_clusterless_classifier_smoke_and_state_marginals() -> None:
    """Functional smoke: the classifier decode + classify, and the
    marginal state probabilities are a valid distribution per time."""
    pytest.importorskip("replay_trajectory_classification")
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_classifier

    rng = np.random.default_rng(1)
    n_time, n_marks, n_electrodes = 150, 4, 3
    t = np.arange(n_time)
    position = 50.0 + 40.0 * np.cos(2 * np.pi * t / n_time).reshape(-1, 1)
    multiunits = np.full((n_time, n_marks, n_electrodes), np.nan)
    spike_mask = rng.random(n_time) < 0.25
    for t_i in np.flatnonzero(spike_mask):
        for e in range(n_electrodes):
            if rng.random() < 0.5:
                multiunits[t_i, :, e] = rng.normal(loc=position[t_i, 0] / 20.0, size=n_marks)

    result = fit_clusterless_classifier(
        position, multiunits,
        place_bin_size=5.0,
        state_names=["continuous", "fragmented"],
    )

    assert result.state_probabilities.shape == (n_time, 2)
    assert result.state_names == ["continuous", "fragmented"]
    sums = result.state_probabilities.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), (
        f"state marginals should sum to 1; row sums in "
        f"[{sums.min():.4f}, {sums.max():.4f}]"
    )
