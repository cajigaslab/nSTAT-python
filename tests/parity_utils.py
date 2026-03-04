from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


def set_deterministic_seeds(seed: int) -> np.random.Generator:
    """Set deterministic Python + NumPy RNG state and return a Generator."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    return np.random.default_rng(int(seed))


def matlab_rng_command(seed: int, generator: str = "twister") -> str:
    """Return the MATLAB RNG statement used for fixture generation scripts."""
    return f"rng({int(seed)}, '{generator}');"


def _convert_matlab_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 1:
            return _convert_matlab_value(value.reshape(-1)[0])
        if value.ndim == 0:
            return _convert_matlab_value(value.item())
        if value.ndim == 1:
            return [_convert_matlab_value(x) for x in value.tolist()]
        if value.ndim == 2:
            return [[_convert_matlab_value(x) for x in row] for row in value.tolist()]
        return [_convert_matlab_value(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "_fieldnames"):
        out: dict[str, Any] = {}
        for name in getattr(value, "_fieldnames", []):
            out[str(name)] = _convert_matlab_value(getattr(value, name))
        return out
    return value


def loadmat_normalized(
    path: str | Path,
    *,
    squeeze_me: bool = False,
    keep_metadata: bool = False,
) -> dict[str, Any]:
    """Load a MATLAB .mat file and normalize structs/cells into Python types."""
    payload = scipy.io.loadmat(
        str(path),
        squeeze_me=squeeze_me,
        struct_as_record=False,
    )
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if not keep_metadata and key.startswith("__"):
            continue
        out[key] = _convert_matlab_value(value)
    return out


def canonicalize_numeric(value: Any, *, vector_shape: str = "preserve") -> np.ndarray:
    """Canonicalize numeric values for parity comparisons (dtype + vector orientation)."""
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64, copy=False)
    if arr.ndim == 1:
        if vector_shape == "column":
            arr = arr[:, None]
        elif vector_shape == "row":
            arr = arr[None, :]
    return arr


def assert_same_shape(actual: Any, expected: Any) -> None:
    a = np.asarray(actual)
    b = np.asarray(expected)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: actual={a.shape} expected={b.shape}")


def assert_matching_nan_inf_locations(actual: Any, expected: Any) -> None:
    a = canonicalize_numeric(actual)
    b = canonicalize_numeric(expected)
    assert_same_shape(a, b)

    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        raise AssertionError("NaN locations do not match")

    a_pos_inf = np.isposinf(a)
    b_pos_inf = np.isposinf(b)
    if not np.array_equal(a_pos_inf, b_pos_inf):
        raise AssertionError("+Inf locations do not match")

    a_neg_inf = np.isneginf(a)
    b_neg_inf = np.isneginf(b)
    if not np.array_equal(a_neg_inf, b_neg_inf):
        raise AssertionError("-Inf locations do not match")


def _scale_from_expected(expected: np.ndarray, mode: str) -> float:
    finite = np.isfinite(expected)
    if not np.any(finite):
        return 1.0
    vals = np.abs(expected[finite])
    if mode == "maxabs":
        return float(max(np.max(vals), 1.0))
    if mode == "rms":
        return float(max(np.sqrt(np.mean(expected[finite] ** 2)), 1.0))
    if mode == "range":
        rng = float(np.max(expected[finite]) - np.min(expected[finite]))
        return max(rng, 1.0)
    raise ValueError(f"unsupported scale mode: {mode}")


def assert_allclose_scaled(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    scale: str = "maxabs",
) -> None:
    a = canonicalize_numeric(actual)
    b = canonicalize_numeric(expected)
    assert_same_shape(a, b)
    assert_matching_nan_inf_locations(a, b)

    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return

    scale_val = _scale_from_expected(b, scale)
    np.testing.assert_allclose(
        a[finite],
        b[finite],
        rtol=float(rtol),
        atol=float(atol) * scale_val,
    )


def assert_event_times_close(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1.0e-9,
    sort_values: bool = True,
) -> None:
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(expected, dtype=np.float64).reshape(-1)
    if sort_values:
        a = np.sort(a)
        b = np.sort(b)
    assert_same_shape(a, b)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=float(atol))
