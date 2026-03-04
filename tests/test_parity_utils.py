from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from tests.parity_utils import (
    assert_allclose_scaled,
    assert_event_times_close,
    assert_matching_nan_inf_locations,
    assert_same_shape,
    canonicalize_numeric,
    loadmat_normalized,
    matlab_rng_command,
    set_deterministic_seeds,
)


def test_set_deterministic_seeds_reproducible() -> None:
    g1 = set_deterministic_seeds(1234)
    a1 = g1.normal(size=8)
    g2 = set_deterministic_seeds(1234)
    a2 = g2.normal(size=8)
    assert np.array_equal(a1, a2)
    assert matlab_rng_command(1234) == "rng(1234, 'twister');"


def test_loadmat_normalized_converts_structs_and_cells(tmp_path: Path) -> None:
    matlab_struct = {"field_a": np.array([1.0, 2.0]), "field_b": np.array([[3.0]])}
    cell_like = np.empty((1, 2), dtype=object)
    cell_like[0, 0] = np.array([4.0, 5.0])
    cell_like[0, 1] = "x"
    path = tmp_path / "fixture.mat"
    scipy.io.savemat(path, {"S": matlab_struct, "C": cell_like})

    payload = loadmat_normalized(path)
    assert "S" in payload and "C" in payload
    assert isinstance(payload["S"], dict)
    assert payload["S"]["field_a"].shape == (1, 2)
    assert isinstance(payload["C"], list)


def test_canonicalize_and_shape_helpers() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    col = canonicalize_numeric(v, vector_shape="column")
    row = canonicalize_numeric(v, vector_shape="row")
    assert col.dtype == np.float64
    assert col.shape == (3, 1)
    assert row.shape == (1, 3)
    assert_same_shape(col, np.zeros((3, 1)))


def test_nan_inf_and_scaled_allclose_helpers() -> None:
    expected = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
    actual = np.array([1.0 + 1e-10, np.nan, np.inf, -np.inf, 5.0 + 1e-10])
    assert_matching_nan_inf_locations(actual, expected)
    assert_allclose_scaled(actual, expected, rtol=1e-7, atol=1e-9, scale="maxabs")


def test_event_time_helper_sorts_and_compares() -> None:
    a = np.array([0.3000000001, 0.1, 0.2])
    b = np.array([0.1, 0.2, 0.3])
    assert_event_times_close(a, b, atol=1e-8, sort_values=True)
