"""A numba whose ``njit`` decoration raises must never break nstat."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, importlib.machinery, os, sys

    CALLS = []
    # Decorations 1..FAIL_FROM-1 "succeed" (identity), FAIL_FROM onward raise.
    FAIL_FROM = int(os.environ["NJIT_FAIL_FROM"])

    class L(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            def njit(*a, **k):
                def deco(f):
                    CALLS.append(f.__name__)
                    if len(CALLS) >= FAIL_FROM:
                        raise RuntimeError("cannot cache function: no locator available")
                    return f
                return deco
            module.njit = njit

    class F(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "numba":
                return importlib.machinery.ModuleSpec(name, L())

    sys.meta_path.insert(0, F())

    import numpy as np
    import nstat
    from nstat import DecodingAlgorithms as DA

    def run():
        rng = np.random.default_rng(0)
        C = rng.standard_normal((1, 2))
        y = rng.standard_normal((1, 50))
        kf = DA.kalman_filter(np.eye(2), C, 0.01 * np.eye(2), 0.05 * np.eye(1),
                              np.eye(2), np.zeros(2), y)
        dN = (rng.uniform(size=(2, 60)) < 0.05).astype(float)
        pp = DA.PPDecodeFilterLinear(
            np.eye(2), 0.01 * np.eye(2), dN, np.array([-1.0, -1.0]),
            np.array([[0.5, 0.2], [0.1, 0.4]]), "binomial", 0.001)
        return kf, pp

    def flat(res):
        out = []
        for r in res:
            out.extend(flat(r) if isinstance(r, (tuple, list)) else [np.asarray(r)])
        return out

    first = run()
    second = run()  # a second call must not re-run the kernels module body
    nk = sys.modules["nstat.extras._numba_kernels"]
    assert nk._NUMBA_AVAILABLE is False and nk._NUMBA_IMPORT_ERROR is not None
    # module body ran once: decorations stop at the first failure
    assert len(CALLS) == FAIL_FROM, CALLS
    nk._NUMBA_AVAILABLE = False
    ref = run()
    for a, b, c in zip(flat(first), flat(second), flat(ref)):
        for x in (a, b):
            if x.dtype.kind == "f":
                assert np.array_equal(x, c, equal_nan=True)
            else:
                assert repr(x) == repr(c)
    print("OK")
    """
)


@pytest.mark.parametrize("fail_from", [1, 2], ids=["every-decoration", "later-decoration"])
def test_failing_njit_decoration_falls_back_to_pure_python(fail_from):
    """fail_from=1: every decoration raises. fail_from=2: the first kernel
    decorates and a later one raises, so no mixed JIT/Python state may be used."""
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo), "NJIT_FAIL_FROM": str(fail_from)},
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")
