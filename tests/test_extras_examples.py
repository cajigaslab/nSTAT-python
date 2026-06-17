"""Smoke + run tests for ``examples/extras/*.py``.

Two layers:

1. **Compile-import**: every example script must parse cleanly under
   the current Python version (catches accidental breakage when bridges
   evolve).  Runs in every CI environment, regardless of opt-deps.
2. **Run-as-main**: when the example's backing opt-dep is installed,
   the script must exit cleanly (return code 0) — a real end-to-end
   smoke check that the bridge actually works.

Adding a new example file is the only required step; this test
auto-discovers ``examples/extras/*.py`` and applies both layers.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "extras"


# Map: example script stem → name of the opt-dep package it requires.
# Used to skip the full run-as-main pass when the dep is absent.
EXAMPLE_BACKING_PACKAGE: dict[str, str] = {
    "interop_neo_demo": "neo",
    "interop_pynapple_demo": "pynapple",
    "interop_nwb_demo": "pynwb",
    "validation_nemos_demo": "nemos",
    "validation_pykalman_demo": "pykalman",
    "validation_statsmodels_demo": "statsmodels",
    "metrics_spike_distances_demo": "pyspike",
    "em_dynamax_demo": "dynamax",
    "decoding_clusterless_demo": "replay_trajectory_classification",
    # place_field_decoder is pure-core (numpy + scipy already required);
    # "numpy" forces the run-as-main test to execute the demo end-to-end.
    "decoding_place_field_demo": "numpy",
    "latents_gpfa_demo": "elephant",
    # nstat.extras.spatial cluster-Cox + Gibbs demos: pure NumPy/SciPy
    # (no opt-dep beyond the core stack), so "numpy" is the always-present
    # marker that forces the run-as-main test to execute the demo.
    "spatial_cluster_cox_demo": "numpy",
    "spatial_gibbs_demo": "numpy",
}


def _discover_examples() -> list[Path]:
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", _discover_examples(), ids=lambda p: p.stem)
def test_example_script_imports_cleanly(script: Path) -> None:
    """Every examples/extras/*.py must parse + import without raising
    at module load time (i.e., top-level imports succeed even with no
    opt-deps installed).

    This catches the bug class where a refactor accidentally hoists an
    opt-dep import out of the lazy gate into the module top-level —
    which would crash every user who hasn't installed the dep.
    """
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Loading must not raise.  Specifically: no top-level access to
    # neo / pynapple / pynwb / nemos / pykalman / pyspike.
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), f"{script.name} missing main()"


@pytest.mark.parametrize("script", _discover_examples(), ids=lambda p: p.stem)
def test_example_runs_when_backing_dep_installed(script: Path) -> None:
    """When the example's opt-dep is installed, ``python <script>`` exits 0.

    Skips silently when the dep is absent (mirrors the
    ``pytest.importorskip`` pattern in the per-bridge functional tests).
    """
    pkg = EXAMPLE_BACKING_PACKAGE.get(script.stem)
    if pkg is None:
        pytest.skip(f"No backing-package mapping for {script.stem}")
    pytest.importorskip(pkg)

    # Many opt-deps (notably nemos) lazily pull in jax, which in turn
    # is sensitive to numpy version mismatches.  When the env can't load
    # jax at all, the subprocess would crash with a confusing
    # AttributeError rather than the expected "package missing" path.
    # Treat transitive ImportErrors as skip-worthy rather than failures.
    _TRANSITIVE_DEPS = {
        "nemos": ("jax",),
        "em_dynamax_demo": ("jax", "dynamax"),
    }
    for transitive in _TRANSITIVE_DEPS.get(pkg, ()):
        try:
            __import__(transitive)
        except Exception as exc:
            pytest.skip(
                f"Transitive dep {transitive!r} for {pkg!r} unavailable: {exc}"
            )

    # PYTHONPATH must include REPO_ROOT — ``python examples/extras/X.py``
    # adds the *script's* directory to sys.path[0] (not the cwd), so the
    # editable nstat install is invisible without this.
    import os
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{script.name} exited with code {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_every_example_has_a_backing_package_mapping() -> None:
    """Every example script must have an entry in EXAMPLE_BACKING_PACKAGE
    so the run-as-main test can decide whether to skip.

    Catches drift where a new example lands but the test's mapping
    isn't updated — the script would silently skip its end-to-end check.
    """
    unmapped: list[str] = []
    for script in _discover_examples():
        if script.stem not in EXAMPLE_BACKING_PACKAGE:
            unmapped.append(script.stem)
    assert not unmapped, (
        f"examples/extras/ scripts without an EXAMPLE_BACKING_PACKAGE "
        f"entry: {unmapped}.  Add them to the dict in this test."
    )
