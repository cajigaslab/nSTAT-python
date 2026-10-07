"""Reusable fixtures for building EXACTLY singular matrices in tests.

2026-10 EM / release cycle incident: ``tests/test_em_singular_solves.py``
built its singular ``Sxkm1xkm1`` from a matrix with every column identical,
which is rank-deficient only *up to floating-point rounding* — Accelerate's
LU hit an exact zero pivot (test passed on macOS), Linux OpenBLAS hit a tiny
nonzero pivot instead (test failed in CI on Linux). No local gate caught the
BLAS-dependence until it broke CI.

Use :func:`exactly_singular_gram` instead of hand-rolling a "should be
singular" matrix (identical/near-identical columns, rank-1 outer products
without a structural zero, etc.). It is singular *by construction* — a
Gram matrix with an exact zero row and column — so every LAPACK build meets
an exact zero pivot, not merely a small one. Pair it with
:func:`assert_exactly_singular` to self-check a fixture before trusting it.

See also: the "Testing singular matrices" note in CONTRIBUTING.md.
"""
from __future__ import annotations

import numpy as np


def exactly_singular_gram(
    dim: int,
    n_samples: int,
    rng: np.random.Generator,
    *,
    zero_index: int = -1,
) -> np.ndarray:
    """Return a ``dim x dim`` Gram matrix with an exact zero row/column.

    Built as ``x @ x.T`` where row ``zero_index`` of ``x`` (shape
    ``(dim, n_samples)``) is forced to exactly ``0.0`` *before* the product.
    That zero row makes the Gram matrix's ``zero_index`` row and column
    exactly ``0.0`` too (not merely small), so any LU factorization meets an
    exact zero pivot there regardless of BLAS/LAPACK build.

    This is BLAS-independent by construction. Do NOT substitute a matrix
    that is singular only up to floating-point rounding (e.g. identical or
    near-identical columns/rows) — that reproduces on some BLAS builds and
    silently passes finite, wrong-looking values on others.
    """
    x = rng.standard_normal((dim, n_samples))
    x[zero_index, :] = 0.0
    return x @ x.T


def assert_exactly_singular(matrix: np.ndarray, *, zero_index: int = -1) -> None:
    """Assert ``matrix`` has an exact zero row AND column at ``zero_index``.

    A sanity check for fixtures built by :func:`exactly_singular_gram` (or
    an equivalent hand construction). Catches the mistake of building a
    matrix that is singular only up to rounding -- such a matrix will NOT
    have an exact zero row/column, so this assertion fails on it, which is
    the point: it flags a BLAS-dependent fixture before it ships.
    """
    row = matrix[zero_index, :]
    col = matrix[:, zero_index]
    assert np.all(row == 0.0), (
        f"row {zero_index} is not exactly zero ({row!r}) -- this matrix is "
        "singular only up to rounding, which is BLAS-dependent; use "
        "exactly_singular_gram() instead."
    )
    assert np.all(col == 0.0), (
        f"column {zero_index} is not exactly zero ({col!r}) -- this matrix "
        "is singular only up to rounding, which is BLAS-dependent; use "
        "exactly_singular_gram() instead."
    )
