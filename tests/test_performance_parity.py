"""Schema-only smoke test for the performance-parity baseline.

This test deliberately does NOT run the timing harness: that needs
MATLAB and the local MATLAB checkout (see ``tools/parity/perf_check.py``
and ``docs/parity/runbook.md`` "Performance parity").  It only
validates that ``parity/performance_baseline.yml`` exists, parses, and
records the five hot paths in the expected shape, so that schema
regressions in the baseline file get caught by ``make test-smoke``.

When a builder edits ``tools/parity/perf_check.py`` and re-captures the
baseline with ``make perf-check-capture``, these assertions defend the
shape consumers (this test, the runbook table, future CI gates) depend
on.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "parity" / "performance_baseline.yml"

EXPECTED_PATH_NAMES = {
    "analysis_run_for_neuron",
    "pp_decode_filter_linear",
    "kalman_filter",
    "simulate_point_process",
    "history_compute_history",
}


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE_PATH.is_file(), (
        f"performance baseline missing at {BASELINE_PATH}; "
        "run `make perf-check-capture` to regenerate (needs MATLAB)."
    )
    with open(BASELINE_PATH) as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), "performance_baseline.yml must parse as a mapping"
    return data


def test_baseline_top_level_schema(baseline: dict) -> None:
    """Top-level keys are stable contract for the runbook + future CI gates."""
    for key in ("version", "captured", "matlab_repo", "python_version", "hardware", "paths"):
        assert key in baseline, f"performance baseline missing top-level key: {key}"
    assert baseline["version"] == 1
    assert isinstance(baseline["paths"], list)


def test_baseline_lists_all_five_hot_paths(baseline: dict) -> None:
    """The five hot paths from v11 iter 53 are the contract; drift is a fail."""
    names = {entry["name"] for entry in baseline["paths"]}
    assert names == EXPECTED_PATH_NAMES, (
        f"performance baseline must list exactly the five hot paths; "
        f"missing={EXPECTED_PATH_NAMES - names}, extra={names - EXPECTED_PATH_NAMES}"
    )


@pytest.mark.parametrize("expected_name", sorted(EXPECTED_PATH_NAMES))
def test_baseline_entry_schema(baseline: dict, expected_name: str) -> None:
    """Each path entry exposes the fields the runbook consumes."""
    entry = next(e for e in baseline["paths"] if e["name"] == expected_name)
    for key in (
        "python_function",
        "matlab_function",
        "input_size",
        "python",
        "matlab",
        "target_for_competitive",
        "target_for_parity",
        "flag",
    ):
        assert key in entry, f"{expected_name}: missing field {key}"
    # python sub-mapping must contain a median
    assert "median_sec" in entry["python"], f"{expected_name}: python.median_sec missing"
    assert isinstance(entry["python"].get("runs_sec"), list)
    assert len(entry["python"]["runs_sec"]) >= 1
