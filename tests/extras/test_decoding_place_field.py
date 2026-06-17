"""Tests for ``nstat.extras.decoding.place_field_decoder``.

Covers config validation, end-to-end decoding on a synthetic 2-D
place-cell trial, silent-cell handling, filter / CIF dispatch, and
position shape / length validation.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from nstat import (
    Covariate,
    CovariateCollection,
    SpikeTrainCollection,
    Trial,
    nspikeTrain,
)
from nstat.extras.decoding import (
    PlaceFieldDecoderConfig,
    PlaceFieldDecoderResult,
    fit_place_field_decoder,
)


# ---------------------------------------------------------------------
# Synthetic-trial fixture helper
# ---------------------------------------------------------------------


def _ou_walk(n: int, dt: float, *, theta: float = 0.3, sigma: float = 0.25,
             seed: int = 0) -> np.ndarray:
    """Ornstein-Uhlenbeck walk in [0.05, 0.95], mean-reverting to 0.5."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    x[0] = 0.5
    for k in range(1, n):
        x[k] = (
            x[k - 1]
            + theta * (0.5 - x[k - 1]) * dt
            + sigma * np.sqrt(dt) * rng.standard_normal()
        )
    return np.clip(x, 0.05, 0.95)


def _make_synthetic_trial(
    *,
    duration_s: float = 60.0,
    fs: float = 50.0,
    cell_centres: tuple[tuple[float, float], ...] = (
        (0.25, 0.25), (0.75, 0.75), (0.5, 0.5),
    ),
    peak_rate: float = 20.0,
    sigma_pf: float = 0.15,
    seed: int = 20260617,
    inject_silent: bool = False,
) -> tuple[Trial, np.ndarray]:
    """Construct a synthetic 2-D place-cell Trial + aligned position.

    Returns ``(trial, position)`` where ``position`` is shape ``(n_time, 2)``
    aligned to the trial's covariate sample grid.  When ``inject_silent``
    is True, a no-spike cell is appended as the first cell.
    """
    rng = np.random.default_rng(seed)
    n_time = int(duration_s * fs)
    t = np.arange(n_time, dtype=float) / fs
    x_pos = _ou_walk(n_time, 1.0 / fs, seed=seed + 1)
    y_pos = _ou_walk(n_time, 1.0 / fs, seed=seed + 2)

    # Spike-generation lattice for the synthetic Poisson process.
    bin_width = 0.020
    n_bins = int(duration_s / bin_width)
    bc = (np.arange(n_bins) + 0.5) * bin_width
    x_bin = np.interp(bc, t, x_pos)
    y_bin = np.interp(bc, t, y_pos)

    nstrains: list[Any] = []
    if inject_silent:
        nstrains.append(nspikeTrain(np.array([], dtype=float),
                                    minTime=0.0, maxTime=duration_s))
    for cx, cy in cell_centres:
        rates = peak_rate * np.exp(
            -((x_bin - cx) ** 2 + (y_bin - cy) ** 2) / (2.0 * sigma_pf ** 2)
        )
        counts = rng.poisson(rates * bin_width)
        spike_times: list[float] = []
        for k, c in enumerate(counts):
            for _ in range(int(c)):
                spike_times.append(
                    float(bc[k] + rng.uniform(-bin_width / 2, bin_width / 2))
                )
        nstrains.append(
            nspikeTrain(np.sort(np.asarray(spike_times, dtype=float)),
                        minTime=0.0, maxTime=duration_s)
        )

    spikes = SpikeTrainCollection(nstrains)
    x_cov = Covariate(t, x_pos, "x_pos", "time", "s", "m", ["x"])
    y_cov = Covariate(t, y_pos, "y_pos", "time", "s", "m", ["y"])
    trial = Trial(
        spike_collection=spikes,
        covariate_collection=CovariateCollection([x_cov, y_cov]),
    )
    # Pull the (resampled) position back out of the trial so its length
    # matches the trial covariate grid — Trial.__init__ upsamples to
    # ``sampleRate`` (usually 1000 Hz).
    x_resampled = np.asarray(
        trial.covarColl.getCov(0).data, dtype=float
    ).reshape(-1)
    y_resampled = np.asarray(
        trial.covarColl.getCov(1).data, dtype=float
    ).reshape(-1)
    position = np.column_stack([x_resampled, y_resampled])
    return trial, position


# ---------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------


def test_config_validation() -> None:
    """Each invalid configuration field raises ValueError."""
    with pytest.raises(ValueError, match="bin_width_s"):
        PlaceFieldDecoderConfig(bin_width_s=0.0)
    with pytest.raises(ValueError, match="bin_width_s"):
        PlaceFieldDecoderConfig(bin_width_s=-0.02)
    with pytest.raises(ValueError, match="n_basis_per_dim"):
        PlaceFieldDecoderConfig(n_basis_per_dim=1)
    with pytest.raises(ValueError, match="spline_order"):
        PlaceFieldDecoderConfig(spline_order=0)
    with pytest.raises(ValueError, match="cif_kind"):
        PlaceFieldDecoderConfig(cif_kind="cubic")
    with pytest.raises(ValueError, match="decode_filter"):
        PlaceFieldDecoderConfig(decode_filter="kalman")
    with pytest.raises(ValueError, match="min_n_spikes_per_cell"):
        PlaceFieldDecoderConfig(min_n_spikes_per_cell=-1)


# ---------------------------------------------------------------------
# 2. End-to-end smoke
# ---------------------------------------------------------------------


def test_fit_place_field_decoder_smoke() -> None:
    """Synthetic 3-cell trial → wrapper returns a sensible decode.

    Uses the nonlinear filter branch (analytical CIF evaluation): the
    linear-filter branch relies on a quadratic-Taylor linearisation
    at the trajectory mean that diverges when |beta| is large and the
    filter strays from the linearisation point — fine on the canonical
    example08 dataset (smooth 1460 s trajectory, sub-Hz cells) but
    too tight for short synthetic trials.  The dispatch test
    (``test_linear_vs_nonlinear_filter_dispatch``) exercises both
    branches separately.
    """
    cfg = PlaceFieldDecoderConfig(decode_filter="nonlinear")
    trial, position = _make_synthetic_trial()
    result = fit_place_field_decoder(trial, position, config=cfg)

    assert isinstance(result, PlaceFieldDecoderResult)
    n_bins = result.decoded_position.shape[0]
    assert result.decoded_position.shape == (n_bins, 2)
    assert result.decoded_covariance.shape == (n_bins, 2, 2)
    assert result.decoding_error.shape == (n_bins,)
    assert np.isfinite(result.mean_decoding_error)
    # On a 3-cell well-tiled synthetic test, mean Euclidean error stays
    # well under 0.5 position units (unit-box trajectory).
    assert result.mean_decoding_error < 0.5, (
        f"mean_decoding_error={result.mean_decoding_error}"
    )
    assert result.cell_indices_kept == [0, 1, 2]
    assert result.cell_indices_skipped == []
    assert result.n_basis_per_dim == 8
    assert result.bin_width_s == pytest.approx(0.020)
    assert len(result.spline_coefs) == 3
    assert len(result.quadratic_coefs) == 3
    assert result.quadratic_coefs[0].shape == (6,)
    assert result.spline_coefs[0].shape == (8 * 8 + 1,)


# ---------------------------------------------------------------------
# 3. Silent-cell handling
# ---------------------------------------------------------------------


def test_decoder_handles_silent_cells() -> None:
    """A no-spike cell is skipped, others still decode, UserWarning fires."""
    trial, position = _make_synthetic_trial(inject_silent=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_place_field_decoder(trial, position)

    assert 0 in result.cell_indices_skipped
    assert 0 not in result.cell_indices_kept
    # Three sensible cells follow the silent dummy.
    assert sorted(result.cell_indices_kept) == [1, 2, 3]
    # The skip emits a UserWarning mentioning the cell index and threshold.
    silent_warnings = [
        w for w in captured
        if issubclass(w.category, UserWarning)
        and "cell 0" in str(w.message)
    ]
    assert silent_warnings, (
        "expected a UserWarning for the silent cell; "
        f"captured={[str(w.message) for w in captured]}"
    )
    # Decoder still produced a usable trajectory.
    assert np.all(np.isfinite(result.decoded_position))


# ---------------------------------------------------------------------
# 4. All-silent → clear error
# ---------------------------------------------------------------------


def test_decoder_handles_all_silent() -> None:
    """When no cell passes the min-spike filter, raise a clear ValueError."""
    # Build a trial with two empty spike trains.
    duration = 30.0
    fs = 50.0
    n_time = int(duration * fs)
    t = np.arange(n_time, dtype=float) / fs
    x_pos = _ou_walk(n_time, 1.0 / fs, seed=1)
    y_pos = _ou_walk(n_time, 1.0 / fs, seed=2)
    empty = nspikeTrain(np.array([], dtype=float), minTime=0.0, maxTime=duration)
    spikes = SpikeTrainCollection([empty, empty])
    x_cov = Covariate(t, x_pos, "x_pos", "time", "s", "m", ["x"])
    y_cov = Covariate(t, y_pos, "y_pos", "time", "s", "m", ["y"])
    trial = Trial(
        spike_collection=spikes,
        covariate_collection=CovariateCollection([x_cov, y_cov]),
    )
    position = np.column_stack([
        np.asarray(trial.covarColl.getCov(0).data, dtype=float).reshape(-1),
        np.asarray(trial.covarColl.getCov(1).data, dtype=float).reshape(-1),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no cells kept"):
            fit_place_field_decoder(trial, position)


# ---------------------------------------------------------------------
# 5. Linear vs nonlinear filter dispatch
# ---------------------------------------------------------------------


def test_linear_vs_nonlinear_filter_dispatch(monkeypatch) -> None:
    """``decode_filter`` selects PPDecodeFilterLinear vs PPDecodeFilter."""
    trial, position = _make_synthetic_trial(duration_s=20.0)
    from nstat.extras.decoding import place_field_decoder as mod

    counter_linear: dict[str, int] = {"n": 0}
    counter_nonlinear: dict[str, int] = {"n": 0}
    real_linear = mod.DecodingAlgorithms.PPDecodeFilterLinear
    real_nonlinear = mod.DecodingAlgorithms.PPDecodeFilter

    def wrap_linear(*args, **kwargs):
        counter_linear["n"] += 1
        return real_linear(*args, **kwargs)

    def wrap_nonlinear(*args, **kwargs):
        counter_nonlinear["n"] += 1
        return real_nonlinear(*args, **kwargs)

    monkeypatch.setattr(
        mod.DecodingAlgorithms, "PPDecodeFilterLinear", wrap_linear
    )
    monkeypatch.setattr(
        mod.DecodingAlgorithms, "PPDecodeFilter", wrap_nonlinear
    )

    cfg_l = PlaceFieldDecoderConfig(bin_width_s=0.10, decode_filter="linear")
    fit_place_field_decoder(trial, position, config=cfg_l)
    assert counter_linear["n"] == 1
    assert counter_nonlinear["n"] == 0

    cfg_n = PlaceFieldDecoderConfig(bin_width_s=0.10, decode_filter="nonlinear")
    fit_place_field_decoder(trial, position, config=cfg_n)
    assert counter_nonlinear["n"] == 1
    # ``PPDecodeFilter`` does not re-enter ``PPDecodeFilterLinear`` for
    # history-free symbolic CIFs (no target branch requested).
    assert counter_linear["n"] == 1


# ---------------------------------------------------------------------
# 6. Quadratic vs linear CIF dispatch
# ---------------------------------------------------------------------


def test_quadratic_vs_linear_cif_dispatch() -> None:
    """``cif_kind`` controls coefficient vector dimensionality."""
    trial, position = _make_synthetic_trial(duration_s=30.0)
    cfg_q = PlaceFieldDecoderConfig(cif_kind="quadratic", bin_width_s=0.10)
    cfg_l = PlaceFieldDecoderConfig(cif_kind="linear", bin_width_s=0.10)
    r_q = fit_place_field_decoder(trial, position, config=cfg_q)
    r_l = fit_place_field_decoder(trial, position, config=cfg_l)

    assert all(c.shape == (6,) for c in r_q.quadratic_coefs)
    assert all(c.shape == (3,) for c in r_l.quadratic_coefs)


# ---------------------------------------------------------------------
# 7. Position shape validation
# ---------------------------------------------------------------------


def test_decoder_position_shape_validation() -> None:
    """Position with the wrong dimensionality raises ValueError early."""
    trial, position = _make_synthetic_trial(duration_s=15.0)

    # 1-D
    with pytest.raises(ValueError, match="2-D"):
        fit_place_field_decoder(trial, position[:, 0])
    # (n, 3)
    bad = np.column_stack([position, position[:, 0]])
    with pytest.raises(ValueError, match="2 columns"):
        fit_place_field_decoder(trial, bad)


# ---------------------------------------------------------------------
# 8. Position time mismatch
# ---------------------------------------------------------------------


def test_decoder_position_time_mismatch() -> None:
    """Position length not matching trial length raises ValueError."""
    trial, position = _make_synthetic_trial(duration_s=15.0)
    # Truncate position by one sample → length mismatch.
    with pytest.raises(ValueError, match="does not match trial length"):
        fit_place_field_decoder(trial, position[:-1])
