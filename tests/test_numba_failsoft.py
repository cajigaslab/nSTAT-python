"""A numba whose ``njit`` decoration raises must never break nstat."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, importlib.machinery, sys

    CALLS = []

    class L(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            def njit(*a, **k):
                def deco(f):
                    CALLS.append(f.__name__)
                    raise RuntimeError("cannot cache function: no locator available")
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
    assert len(CALLS) == 1, CALLS  # module body ran once (first decoration failed)
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


def test_failing_njit_decoration_falls_back_to_pure_python():
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        cwd=repo,
        env={**__import__("os").environ, "PYTHONPATH": str(repo)},
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")
