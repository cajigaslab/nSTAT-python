"""The opt-in numba accelerator is probed lazily and fails soft.

Codebase review 2026-10-04 (extras F10 / B5, plus the tests-T1 follow-up):

* ``import nstat`` must not import ``nstat.extras._numba_kernels`` (a
  core -> extras dependency) or probe ``numba``; the probe runs at the first
  fast-path dispatch in ``DecodingAlgorithms``.
* An installed-but-broken numba (e.g. built against a different NumPy ABI)
  can raise ``ValueError`` / ``RuntimeError`` from inside its own import.
  That must disable the fast path (``_NUMBA_AVAILABLE = False``, reason kept
  in ``_NUMBA_IMPORT_ERROR``) instead of crashing ``import nstat`` or the
  decoders.
* The dispatch reads ``nstat.extras._numba_kernels._NUMBA_AVAILABLE`` at
  call time, so monkeypatching it still selects the path
  (``tests/test_decoding_algorithms_fidelity.py`` relies on this).

The import-state tests run in a fresh interpreter because the pytest process
has usually imported ``numba`` / ``nstat.extras`` already.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

import nstat
from nstat.DecodingAlgorithms import DecodingAlgorithms

_REPO_ROOT = Path(nstat.__file__).resolve().parents[1]


def _run_fresh(script: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _kalman_inputs():
    rng = np.random.default_rng(0)
    n_steps, n_state, n_obs = 120, 3, 2
    A = np.eye(n_state) + 0.01 * rng.standard_normal((n_state, n_state))
    C = rng.standard_normal((n_obs, n_state))
    Pv = 0.01 * np.eye(n_state)
    Pw = 0.05 * np.eye(n_obs)
    Px0 = np.eye(n_state)
    x0 = np.zeros(n_state)
    y = rng.standard_normal((n_obs, n_steps))
    return A, C, Pv, Pw, Px0, x0, y


def _ppaf_inputs():
    rng = np.random.default_rng(0)
    n_steps = 200
    a = np.eye(2)
    q = 0.01 * np.eye(2)
    mu = np.array([-1.0, -1.0], dtype=float)
    beta = np.array([[0.5, 0.2], [0.1, 0.4]], dtype=float)
    dN = (rng.uniform(0.0, 1.0, size=(2, n_steps)) < 0.05).astype(float)
    return a, q, dN, mu, beta


def test_import_nstat_does_not_probe_numba(tmp_path: Path) -> None:
    proc = _run_fresh(
        f"""
        import sys
        import nstat
        assert nstat.__file__.startswith({str(_REPO_ROOT)!r}), nstat.__file__
        loaded = sorted(m for m in ("numba", "nstat.extras._numba_kernels") if m in sys.modules)
        assert not loaded, loaded
        # The extras namespace stays reachable as an attribute.
        assert nstat.extras.__name__ == "nstat.extras"
        print("OK")
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")


def test_abi_broken_numba_falls_back_to_pure_python(tmp_path: Path, monkeypatch) -> None:
    """A numba whose import raises ValueError must not break nstat."""
    out_path = tmp_path / "fallback_outputs.npz"
    proc = _run_fresh(
        f"""
        import importlib.abc
        import importlib.machinery
        import sys

        class _BrokenNumbaLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise ValueError("simulated numba/NumPy ABI incompatibility")

        class _BrokenNumbaFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "numba" or fullname.startswith("numba."):
                    return importlib.machinery.ModuleSpec(fullname, _BrokenNumbaLoader())
                return None

        sys.meta_path.insert(0, _BrokenNumbaFinder())

        import numpy as np
        import nstat
        from nstat import DecodingAlgorithms
        from nstat.extras import _numba_kernels as nk

        assert nstat.__file__.startswith({str(_REPO_ROOT)!r}), nstat.__file__
        assert nk._NUMBA_AVAILABLE is False
        assert isinstance(nk._NUMBA_IMPORT_ERROR, ValueError), repr(nk._NUMBA_IMPORT_ERROR)
        assert "simulated numba/NumPy ABI" in str(nk._NUMBA_IMPORT_ERROR)
        assert "numba" not in sys.modules

        rng = np.random.default_rng(0)
        n_steps, n_state, n_obs = 120, 3, 2
        A = np.eye(n_state) + 0.01 * rng.standard_normal((n_state, n_state))
        C = rng.standard_normal((n_obs, n_state))
        kf = DecodingAlgorithms.kalman_filter(
            A, C, 0.01 * np.eye(n_state), 0.05 * np.eye(n_obs), np.eye(n_state),
            np.zeros(n_state), rng.standard_normal((n_obs, n_steps)),
        )

        rng = np.random.default_rng(0)
        dN = (rng.uniform(0.0, 1.0, size=(2, 200)) < 0.05).astype(float)
        pp = DecodingAlgorithms.PPDecodeFilterLinear(
            np.eye(2), 0.01 * np.eye(2), dN, np.array([-1.0, -1.0]),
            np.array([[0.5, 0.2], [0.1, 0.4]]), "binomial", 0.001,
        )
        np.savez(
            {str(out_path)!r},
            **{{f"kf{{i}}": np.asarray(v) for i, v in enumerate(kf[:5])}},
            **{{f"pp{{i}}": np.asarray(v) for i, v in enumerate(pp[:4])}},
        )
        print("OK")
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")

    # The subprocess outputs must equal this process's forced pure-Python path.
    monkeypatch.setattr("nstat.extras._numba_kernels._NUMBA_AVAILABLE", False)
    kf = DecodingAlgorithms.kalman_filter(*_kalman_inputs())
    pp = DecodingAlgorithms.PPDecodeFilterLinear(*_ppaf_inputs(), "binomial", 0.001)
    saved = np.load(out_path)
    for i in range(5):
        np.testing.assert_array_equal(saved[f"kf{i}"], np.asarray(kf[i]))
    for i in range(4):
        np.testing.assert_array_equal(saved[f"pp{i}"], np.asarray(pp[i]))


def test_fast_path_dispatch_reads_flag_at_call_time(monkeypatch) -> None:
    """Flipping ``_NUMBA_AVAILABLE`` after import still selects the path."""
    from nstat.extras import _numba_kernels as nk

    calls: list[str] = []

    def _spy(name):
        def _kernel(*args, **kwargs):
            calls.append(name)
            # Raising sends the dispatch to its pure-Python fallback.
            raise RuntimeError("spy kernel")

        return _kernel

    monkeypatch.setattr(nk, "kalman_filter_loop", _spy("kalman"))
    monkeypatch.setattr(nk, "ppdecode_linear_loop", _spy("ppaf"))

    monkeypatch.setattr(nk, "_NUMBA_AVAILABLE", True)
    DecodingAlgorithms.kalman_filter(*_kalman_inputs())
    DecodingAlgorithms.PPDecodeFilterLinear(*_ppaf_inputs(), "binomial", 0.001)
    assert calls == ["kalman", "ppaf"]

    monkeypatch.setattr(nk, "_NUMBA_AVAILABLE", False)
    DecodingAlgorithms.kalman_filter(*_kalman_inputs())
    DecodingAlgorithms.PPDecodeFilterLinear(*_ppaf_inputs(), "binomial", 0.001)
    assert calls == ["kalman", "ppaf"]
