"""Tests for nstat.extras.matlab_rng.MatlabRNG.

The MT19937 state is verified to match MATLAB ``rng(seed)`` for the
uniform stream (bit-equivalent).  Normals are produced via Box-Muller
from the same MT state and are checked for determinism only — they
are *not* bit-equivalent to MATLAB ``randn`` (Ziggurat).

Captured MATLAB reference (under ``rng(42)``) for first 5 ``rand``:
    [0.374540118847362, 0.950714306409916, 0.731993941811405,
     0.598658484197037, 0.156018640442437]
(captured 2026-06-19 via /tmp/capture_randn.m on R2024b; the rand
values are deterministic across MATLAB releases since R14sp1.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nstat.extras.matlab_rng import MatlabRNG, seeded_global_rng


# MATLAB rng(42); rand(1, 5) — verified at the MT19937 state level.
MATLAB_RAND_SEED42_FIRST5 = np.array(
    [
        0.374540118847362,
        0.950714306409916,
        0.731993941811405,
        0.598658484197037,
        0.156018640442437,
    ],
    dtype=float,
)


class TestSeeding:
    def test_seed_must_be_int(self) -> None:
        with pytest.raises(TypeError):
            MatlabRNG(0.5)  # type: ignore[arg-type]

    def test_seed_must_be_nonneg(self) -> None:
        with pytest.raises(ValueError):
            MatlabRNG(-1)

    def test_seed_exposed(self) -> None:
        rng = MatlabRNG(42)
        assert rng.seed == 42

    def test_underlying_random_state_is_mt19937(self) -> None:
        rng = MatlabRNG(42)
        st = rng.random_state.get_state()
        # MT19937 state tuple: ('MT19937', key_array, pos, has_gauss, cached_gauss)
        assert st[0] == "MT19937"
        assert st[1].shape == (624,)
        # First key element is the seed itself for MATLAB-compatible init.
        assert int(st[1][0]) == 42


class TestUniform:
    def test_rand_matches_matlab_first_5(self) -> None:
        """Uniform stream is bit-equivalent to MATLAB rand under rng(42)."""
        rng = MatlabRNG(42)
        py = rng.rand(5)
        np.testing.assert_allclose(py, MATLAB_RAND_SEED42_FIRST5, atol=1e-15)

    def test_rand_scalar(self) -> None:
        rng = MatlabRNG(42)
        x = rng.rand()
        assert isinstance(x, float)
        np.testing.assert_allclose(x, MATLAB_RAND_SEED42_FIRST5[0], atol=1e-15)

    def test_rand_shape(self) -> None:
        rng = MatlabRNG(0)
        a = rng.rand(2, 3)
        assert a.shape == (2, 3)


class TestNormal:
    def test_randn_deterministic(self) -> None:
        """Same seed -> same normal sequence (regardless of MATLAB matching)."""
        a = MatlabRNG(42).randn(100)
        b = MatlabRNG(42).randn(100)
        np.testing.assert_array_equal(a, b)

    def test_randn_distribution(self) -> None:
        """Sample mean / std are consistent with N(0, 1) at large N."""
        rng = MatlabRNG(2026)
        x = rng.randn(20000)
        assert abs(float(np.mean(x))) < 0.05
        assert abs(float(np.std(x)) - 1.0) < 0.05

    def test_randn_scalar(self) -> None:
        rng = MatlabRNG(42)
        x = rng.randn()
        assert isinstance(x, float)

    def test_randn_shape(self) -> None:
        rng = MatlabRNG(7)
        a = rng.randn(3, 4)
        assert a.shape == (3, 4)

    def test_randn_single_call_deterministic_shape_invariant(self) -> None:
        """A single ``randn(n, m)`` call yields the same result as ``randn(n*m)`` reshaped."""
        a = MatlabRNG(42).randn(8).reshape(2, 4)
        b = MatlabRNG(42).randn(2, 4)
        np.testing.assert_array_equal(a, b)

    def test_normrnd_uses_randn(self) -> None:
        rng = MatlabRNG(42)
        n1 = rng.normrnd(5.0, 2.0, 100)
        rng2 = MatlabRNG(42)
        n2 = 5.0 + 2.0 * rng2.randn(100)
        np.testing.assert_array_equal(n1, n2)

    def test_standard_normal_alias(self) -> None:
        rng = MatlabRNG(42)
        a = rng.standard_normal(5)
        rng2 = MatlabRNG(42)
        b = rng2.randn(5)
        np.testing.assert_array_equal(a, b)

        rng3 = MatlabRNG(42)
        c = rng3.standard_normal((2, 3))
        assert c.shape == (2, 3)


class TestMatlabRandnReference:
    """Compare against a captured MATLAB ``rng(42); randn(...)`` sequence
    when the reference fixture exists.

    The MatlabRNG does NOT match MATLAB ``randn`` bit-for-bit (MATLAB uses
    Ziggurat; this class uses Box-Muller).  This test merely documents
    the *expected* divergence so that future iters wanting strict
    bit-equivalence have a visible baseline to start from.
    """

    REF_PATH = Path("/tmp/randn_ref.mat")

    def test_known_divergence_from_matlab_randn(self) -> None:
        if not self.REF_PATH.exists():
            pytest.skip("MATLAB reference /tmp/randn_ref.mat not present")
        from scipy.io import loadmat

        ref = loadmat(str(self.REF_PATH))
        ml_r = ref["r"].ravel()
        py_r = MatlabRNG(42).randn(ml_r.size)
        # Confirm the documented divergence: no element matches.
        n_match = int(np.sum(np.isclose(ml_r, py_r, atol=1e-12)))
        # We expect 0 matches (Box-Muller vs Ziggurat consume different
        # numbers of uniforms per normal); a future bit-equivalent
        # implementation would flip this assertion.
        assert n_match == 0, (
            "MatlabRNG.randn unexpectedly matches MATLAB randn — if a "
            "future iter shipped a real Ziggurat port, update this test "
            "to assert full match."
        )


class TestSeededGlobalRng:
    def test_seeds_legacy_global(self) -> None:
        with seeded_global_rng(42):
            a = np.random.randn(5)
        with seeded_global_rng(42):
            b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)

    def test_seeds_default_rng(self) -> None:
        with seeded_global_rng(42):
            g1 = np.random.default_rng()
            a = g1.standard_normal(5)
        with seeded_global_rng(42):
            g2 = np.random.default_rng()
            b = g2.standard_normal(5)
        np.testing.assert_array_equal(a, b)

    def test_restores_state_on_exit(self) -> None:
        # Capture pre-state, run a with-block, confirm post-state matches.
        pre = np.random.get_state()
        pre_default = np.random.default_rng
        with seeded_global_rng(123):
            np.random.randn(5)
        post = np.random.get_state()
        # MT keys should be identical (legacy state restored verbatim).
        np.testing.assert_array_equal(pre[1], post[1])
        assert np.random.default_rng is pre_default

    def test_yields_matlab_rng(self) -> None:
        with seeded_global_rng(42) as rng:
            assert isinstance(rng, MatlabRNG)
            assert rng.seed == 42
