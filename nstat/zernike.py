from __future__ import annotations

from math import comb, sqrt
from typing import Iterable

import numpy as np


DEFAULT_ZERNIKE_MODES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, -1),
    (1, 1),
    (2, -2),
    (2, 0),
    (2, 2),
    (3, -3),
    (3, -1),
    (3, 1),
    (3, 3),
)


def _as_mode_vector(values: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=int).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return array


def _as_float_vector(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return array


def _radial_polynomial(n: int, m_abs: int, r: np.ndarray) -> np.ndarray:
    radial = np.zeros(r.shape[0], dtype=float)
    for s in range((n - m_abs) // 2 + 1):
        coefficient = ((-1) ** s) * comb(n - s, s) * comb(n - 2 * s, (n - m_abs) // 2 - s)
        radial += float(coefficient) * np.power(r, n - 2 * s, dtype=float)
    return radial


def zernfun(
    n: Iterable[int] | np.ndarray,
    m: Iterable[int] | np.ndarray,
    r: Iterable[float] | np.ndarray,
    theta: Iterable[float] | np.ndarray,
    *,
    normalized: bool = False,
) -> np.ndarray:
    """Evaluate MATLAB-style Zernike functions on the unit disk."""

    n_vec = _as_mode_vector(n, "n")
    m_vec = _as_mode_vector(m, "m")
    r_vec = _as_float_vector(r, "r")
    theta_vec = _as_float_vector(theta, "theta")

    if n_vec.shape != m_vec.shape:
        raise ValueError("n and m must be the same length")
    if r_vec.shape != theta_vec.shape:
        raise ValueError("r and theta must be the same length")
    if np.any((r_vec < 0.0) | (r_vec > 1.0)):
        raise ValueError("All radial coordinates must be between 0 and 1")

    out = np.zeros((r_vec.shape[0], n_vec.shape[0]), dtype=float)
    for column, (n_val, m_val) in enumerate(zip(n_vec.tolist(), m_vec.tolist(), strict=False)):
        m_abs = abs(int(m_val))
        if n_val < 0:
            raise ValueError("All n values must be non-negative")
        if m_abs > n_val or (n_val - m_abs) % 2 != 0:
            raise ValueError("Each |m| must be <= n and differ from n by a multiple of 2")

        radial = _radial_polynomial(int(n_val), m_abs, r_vec)
        if normalized:
            radial *= sqrt(float(n_val + 1)) if m_val == 0 else sqrt(float(2 * (n_val + 1)))

        if m_val > 0:
            out[:, column] = radial * np.sin(theta_vec * float(m_val))
        elif m_val < 0:
            out[:, column] = radial * np.cos(theta_vec * float(m_val))
        else:
            out[:, column] = radial
    return out


def zernike_basis_from_cartesian(
    x: Iterable[float] | np.ndarray,
    y: Iterable[float] | np.ndarray,
    *,
    modes: tuple[tuple[int, int], ...] = DEFAULT_ZERNIKE_MODES,
    normalized: bool = True,
    fill_value: float | None = None,
) -> np.ndarray:
    x_vec = _as_float_vector(x, "x")
    y_vec = _as_float_vector(y, "y")
    if x_vec.shape != y_vec.shape:
        raise ValueError("x and y must be the same length")

    theta = np.arctan2(y_vec, x_vec)
    r = np.sqrt(x_vec * x_vec + y_vec * y_vec)
    inside = r <= (1.0 + 1e-12)
    n = np.asarray([mode[0] for mode in modes], dtype=int)
    m = np.asarray([mode[1] for mode in modes], dtype=int)

    if fill_value is None:
        if not np.all(inside):
            raise ValueError("Cartesian coordinates must lie within the unit disk")
        return zernfun(n, m, np.clip(r, 0.0, 1.0), theta, normalized=normalized)

    out = np.full((x_vec.shape[0], len(modes)), float(fill_value), dtype=float)
    if np.any(inside):
        out[inside] = zernfun(n, m, np.clip(r[inside], 0.0, 1.0), theta[inside], normalized=normalized)
    return out


__all__ = ["DEFAULT_ZERNIKE_MODES", "zernfun", "zernike_basis_from_cartesian"]
