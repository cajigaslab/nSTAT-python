"""MATLAB-aligned Mersenne Twister RNG wrapper for deterministic MC parity.

This module exposes :class:`MatlabRNG`, a thin wrapper around NumPy's legacy
:class:`numpy.random.RandomState` (MT19937), seeded the same way MATLAB's
``rng(N)`` seeds its default ``twister`` generator.  The goal is to give
Python recipes a deterministic stream that lines up with MATLAB's uniform
draws bit-for-bit, while normals are produced by a documented Python-side
algorithm (Box-Muller from the same MT19937 stream) — *not* MATLAB's
Ziggurat.

Use cases (v13 iter 60)
-----------------------
- Tightening tolerances on Case-C drift entries (``PPLFP_MStep``,
  ``PPLFP_EM``, ``v9_PPSS_EM``) by seeding the recipe-side RNG global
  state so Python output is run-to-run reproducible.
- Cross-checks where MATLAB's ``rand`` (uniform) stream is the relevant
  artefact — those match bit-for-bit because the MT19937 state init
  agrees between MATLAB ``rng(N)`` and ``numpy.random.RandomState(N)``.

What is bit-equivalent to MATLAB
--------------------------------
- The underlying MT19937 state after seeding (verified iter 60).
- The uniform sequence ``rand()`` (verified: MATLAB ``rand`` and
  ``RandomState(seed).rand`` match to float64 round-off).

What is NOT bit-equivalent to MATLAB
------------------------------------
- ``randn`` / ``normrnd`` — MATLAB uses Marsaglia–Tsang Ziggurat
  (256-region) since R14sp1; NumPy's legacy ``randn`` uses polar
  Marsaglia.  Both draw from the same MT19937 stream but consume
  different numbers of uint32 words per output.  This module's
  :meth:`MatlabRNG.randn` uses the Box–Muller transform from the
  uniform stream — *statistically equivalent* to MATLAB normals
  (same distribution, same MT state) but not bit-equivalent to any
  specific MATLAB output.

For users who need bit-equivalence with NumPy's legacy ``randn``,
:meth:`MatlabRNG.legacy_randn` exposes the underlying
``RandomState.randn``; for full bit-equivalence with MATLAB normrnd a
proper Ziggurat port is required (out of scope for v13).

Example
-------
>>> from nstat.extras.matlab_rng import MatlabRNG
>>> r = MatlabRNG(42)
>>> u = r.rand(3)            # matches MATLAB rand(1,3) under rng(42)
>>> n = r.randn(3)           # deterministic Python-side normals
>>> r2 = MatlabRNG(42)
>>> bool((r2.rand(3) == u).all())
True
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import numpy as np

__all__ = ["MatlabRNG", "seeded_global_rng"]


class MatlabRNG:
    """Deterministic RNG with MATLAB-aligned MT19937 state.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.  Mirrors MATLAB ``rng(seed)`` for the
        underlying MT19937 state.

    Attributes
    ----------
    seed : int
        The seed used to initialise the generator.

    Notes
    -----
    The internal generator is ``numpy.random.RandomState(seed)``.  NumPy's
    legacy ``RandomState`` implements the same ``init_by_array`` Mersenne
    Twister scheme as MATLAB's default ``twister`` generator, so the
    raw 624-word state and the uniform-double stream are bit-equivalent
    to MATLAB ``rand()`` under matching seed.

    Normals are drawn via the Box-Muller transform from pairs of
    uniforms drawn from the same MT19937 stream.  This produces
    deterministic, machine-independent normal samples that have the
    same standard-normal distribution as MATLAB's Ziggurat output but
    are NOT bit-equivalent — MATLAB Ziggurat consumes a variable number
    of uint32 words per sample and uses different acceptance regions.
    """

    __slots__ = ("_seed", "_rs")

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
        seed_int = int(seed)
        if seed_int < 0:
            raise ValueError(f"seed must be non-negative, got {seed_int}")
        self._seed = seed_int
        self._rs = np.random.RandomState(seed_int)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def seed(self) -> int:
        """The seed this generator was initialised with."""
        return self._seed

    @property
    def random_state(self) -> np.random.RandomState:
        """The underlying :class:`numpy.random.RandomState` (MT19937)."""
        return self._rs

    # ------------------------------------------------------------------
    # Uniform draws — bit-equivalent to MATLAB ``rand``
    # ------------------------------------------------------------------
    def rand(self, *shape: int) -> np.ndarray:
        """Uniform [0, 1) draws.  Bit-equivalent to MATLAB ``rand``."""
        if not shape:
            return float(self._rs.rand())
        return self._rs.rand(*shape)

    # ------------------------------------------------------------------
    # Normal draws — statistically equivalent to MATLAB normrnd
    # ------------------------------------------------------------------
    def randn(self, *shape: int) -> np.ndarray | float:
        """Standard-normal draws via Box-Muller from the MT19937 stream.

        Returns
        -------
        float or numpy.ndarray
            Scalar when called with no shape; array of the requested
            shape otherwise.

        Notes
        -----
        Uses the standard Box-Muller transform:
        ``z1 = sqrt(-2 ln u1) cos(2*pi*u2)``,
        ``z2 = sqrt(-2 ln u1) sin(2*pi*u2)``.
        Two normals are produced per pair of uniforms; for odd-length
        requests the second normal of the final pair is discarded
        (no across-call cache, to keep call-pattern reproducibility
        simple: ``randn(n)`` always consumes ``2 * ceil(n/2)`` uniforms).

        This is *not* bit-equivalent to MATLAB ``randn`` (which uses
        Ziggurat) but is statistically identical (same N(0,1)
        distribution) and deterministic across machines / NumPy
        versions for a given seed.
        """
        n = 1
        for s in shape:
            n *= int(s)
        if n == 0:
            return np.zeros(shape, dtype=float)

        n_pairs = (n + 1) // 2
        u1 = self._rs.rand(n_pairs)
        u2 = self._rs.rand(n_pairs)
        # Replace any exact zeros (RandomState.rand can return 0.0) to
        # avoid log(0).  Probability ~ 2^-53 per draw.
        zero_mask = u1 == 0.0
        while zero_mask.any():
            u1 = np.where(zero_mask, self._rs.rand(int(zero_mask.sum())), u1)
            zero_mask = u1 == 0.0
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        z1 = r * np.cos(theta)
        z2 = r * np.sin(theta)

        out = np.empty(2 * n_pairs, dtype=float)
        out[0::2] = z1
        out[1::2] = z2
        out = out[:n]

        if not shape:
            return float(out[0])
        return out.reshape(shape)

    def legacy_randn(self, *shape: int) -> np.ndarray | float:
        """NumPy's legacy ``RandomState.randn`` (polar Marsaglia).

        Provided for callers who want bit-equivalence with NumPy's
        legacy ``np.random.randn`` rather than the Box-Muller stream
        produced by :meth:`randn`.  Not bit-equivalent to MATLAB.
        """
        if not shape:
            return float(self._rs.randn())
        return self._rs.randn(*shape)

    def normrnd(
        self,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        *shape: int,
    ) -> np.ndarray | float:
        """MATLAB-style ``normrnd(mu, sigma, ...)`` using :meth:`randn`."""
        return mu + sigma * self.randn(*shape)

    def standard_normal(self, size=None) -> np.ndarray | float:
        """NumPy-Generator-style alias for :meth:`randn`.

        Accepts either a scalar size or a tuple/list shape (matching
        the :class:`numpy.random.Generator` API).
        """
        if size is None:
            return self.randn()
        if isinstance(size, (int, np.integer)):
            return self.randn(int(size))
        return self.randn(*tuple(int(s) for s in size))


@contextlib.contextmanager
def seeded_global_rng(seed: int) -> Iterator[MatlabRNG]:
    """Context manager: seed NumPy global RNG state for the with-block.

    Used by the numerical-drift recipes to make Python output
    deterministic run-to-run when the underlying nstat function draws
    via ``np.random.default_rng()`` (no seed) or the legacy
    ``np.random.randn`` global path.

    On entry: saves the current global ``np.random`` state, seeds
    ``np.random.seed(seed)``, and monkey-patches
    ``np.random.default_rng`` to return a deterministically-seeded
    :class:`numpy.random.Generator` (PCG64) derived from ``seed``.
    On exit: restores both.

    Yields a :class:`MatlabRNG` for callers that want to draw
    explicitly inside the block.

    Example
    -------
    >>> import numpy as np
    >>> from nstat.extras.matlab_rng import seeded_global_rng
    >>> with seeded_global_rng(42) as rng:
    ...     a = np.random.randn(3)        # deterministic
    ...     g = np.random.default_rng()   # also deterministic
    ...     b = g.standard_normal(3)
    >>> with seeded_global_rng(42) as rng:
    ...     a2 = np.random.randn(3)
    >>> bool((a == a2).all())
    True
    """
    saved_state = np.random.get_state()
    saved_default_rng = np.random.default_rng

    np.random.seed(seed)
    # Patch default_rng so that any call inside the block returns the
    # same deterministic Generator (PCG64-seeded by `seed`).  Successive
    # calls return *fresh* generators with the same seed — matching the
    # current intent of "deterministic Python output, run-to-run".
    def _seeded_default_rng(*a, **kw):  # type: ignore[no-untyped-def]
        # Honour explicit seeds passed by callers (rare in the MC paths
        # we target); otherwise force our deterministic seed.
        if a or kw:
            return saved_default_rng(*a, **kw)
        return saved_default_rng(seed)

    np.random.default_rng = _seeded_default_rng  # type: ignore[assignment]
    try:
        yield MatlabRNG(seed)
    finally:
        np.random.default_rng = saved_default_rng  # type: ignore[assignment]
        np.random.set_state(saved_state)
