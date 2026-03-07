from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "parity" / "simulink_fidelity.yml"
MATLAB_REPO_ROOT = REPO_ROOT.parent / "nSTAT"
VALID_STRATEGIES = {
    "native_python",
    "generated_code_wrapped",
    "packaged_runtime",
    "matlab_engine_fallback",
    "unsupported",
    "reference_only",
}


def _load_audit() -> dict:
    payload = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    assert payload.get("items"), "parity/simulink_fidelity.yml is empty"
    return payload


def test_simulink_fidelity_audit_uses_known_strategy_values() -> None:
    payload = _load_audit()
    for row in payload["items"]:
        assert row["python_strategy"] in VALID_STRATEGIES


def test_simulink_fidelity_audit_records_required_execution_fields() -> None:
    payload = _load_audit()
    for row in payload["items"]:
        assert row["model_path"]
        assert row["purpose"]
        assert row["chosen_interoperability_strategy"]
        assert row["validation_plan"]


def test_simulink_fidelity_audit_paths_exist_when_matlab_repo_is_available() -> None:
    if not MATLAB_REPO_ROOT.exists():
        pytest.skip(f"MATLAB reference repo not available at {MATLAB_REPO_ROOT}")

    payload = _load_audit()
    missing = [row["model_path"] for row in payload["items"] if not (MATLAB_REPO_ROOT / row["model_path"]).exists()]
    assert not missing, f"Missing Simulink audit paths in MATLAB repo: {missing}"


def test_simulink_fidelity_audit_has_no_partial_or_missing_behavioral_paths() -> None:
    payload = _load_audit()
    outstanding = {
        row["model_name"]
        for row in payload["items"]
        if row["model_name"] in {"PointProcessSimulation", "SimulatedNetwork2"}
        and row["current_python_status"] in {"partial", "missing", "unsupported"}
    }
    assert not outstanding
