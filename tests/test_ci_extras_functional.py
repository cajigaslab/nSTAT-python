"""Drift guard: the ``extras-functional`` CI job exists in ``ci.yml``.

The job installs the package with ``[all-extras]`` and runs every
``nstat.extras.*`` bridge against its real backing library — closing
the false-safety gap where ``pytest.importorskip``'d tests stay green
in the default ``[dev]``-only CI matrix.

This test catches the bug class where the job is accidentally deleted
or renamed during workflow refactoring.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load_ci_jobs() -> dict:
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))["jobs"]


def test_extras_functional_job_exists() -> None:
    """``ci.yml`` must define a job that installs [all-extras] and runs
    the extras functional tests with their backing libs present.
    """
    jobs = _load_ci_jobs()
    assert "extras-functional" in jobs, (
        "ci.yml is missing the 'extras-functional' job that installs "
        "[all-extras] and exercises every nstat.extras bridge with its "
        "real backing library.  Without this job, the importorskip-gated "
        "round-trip tests are silently invisible in CI."
    )


def test_extras_functional_installs_all_extras() -> None:
    """The job must actually install ``[all-extras]`` (or equivalent)."""
    jobs = _load_ci_jobs()
    job = jobs.get("extras-functional", {})
    steps = job.get("steps", [])
    run_blocks = [s.get("run", "") for s in steps if isinstance(s, dict)]
    combined = "\n".join(run_blocks)
    assert "[dev,all-extras]" in combined or "[all-extras]" in combined, (
        "extras-functional job does not pip-install the [all-extras] group."
    )


def test_extras_functional_runs_the_extras_tests() -> None:
    """The job must invoke ``pytest tests/extras/`` (the per-bridge
    functional tests) — otherwise the install is wasted.
    """
    jobs = _load_ci_jobs()
    job = jobs.get("extras-functional", {})
    steps = job.get("steps", [])
    run_blocks = [s.get("run", "") for s in steps if isinstance(s, dict)]
    combined = "\n".join(run_blocks)
    assert "tests/extras" in combined, (
        "extras-functional job does not run tests under tests/extras/."
    )


def test_extras_functional_smokes_every_bridge_import() -> None:
    """The job's sanity-check step must import every shipped bridge —
    catches the case where the import-error pathway is broken but
    pytest only tests the happy path.
    """
    jobs = _load_ci_jobs()
    job = jobs.get("extras-functional", {})
    steps = job.get("steps", [])
    run_blocks = [s.get("run", "") for s in steps if isinstance(s, dict)]
    combined = "\n".join(run_blocks)

    REQUIRED_IMPORT_TARGETS = (
        "nstat.extras.interop.neo",
        "nstat.extras.interop.pynapple",
        "nstat.extras.interop.nwb",
        "nstat.extras.validation.nemos_bridge",
        "nstat.extras.validation.pykalman_bridge",
        "nstat.extras.metrics.spike_distances",
    )
    missing = [t for t in REQUIRED_IMPORT_TARGETS if t not in combined]
    assert not missing, (
        f"extras-functional job's sanity-check step doesn't import "
        f"every bridge.  Missing: {missing}.  Add ``python -c \"from "
        f"<module> import <symbol>\"`` lines for each."
    )
