from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "parity" / "class_fidelity.yml"
VALID_STATUSES = {"exact", "high_fidelity", "partial", "shim_only", "missing", "not_applicable"}
PRIORITY_CLASSES = {
    "SignalObj",
    "Covariate",
    "Trial",
    "TrialConfig",
    "ConfigColl",
    "nspikeTrain",
    "nstColl",
    "Analysis",
    "FitResult",
    "FitResSummary",
    "CIF",
    "DecodingAlgorithms",
    "History",
    "Events",
}


def _load_audit() -> dict:
    payload = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    assert payload.get("items"), "parity/class_fidelity.yml is empty"
    return payload


def test_class_fidelity_audit_covers_priority_classes() -> None:
    payload = _load_audit()
    names = {str(item.get("matlab_name", "")).strip() for item in payload["items"]}
    assert PRIORITY_CLASSES.issubset(names)


def test_class_fidelity_audit_uses_known_status_values() -> None:
    payload = _load_audit()
    for item in payload["items"]:
        assert item["status"] in VALID_STATUSES


def test_core_matlab_facing_classes_are_not_shim_only() -> None:
    payload = _load_audit()
    audit_by_name = {str(item["matlab_name"]): item for item in payload["items"]}

    for name in ("SignalObj", "Covariate", "nspikeTrain"):
        row = audit_by_name[name]
        assert row["status"] not in {"shim_only", "missing"}
        assert row["python_path"] == "nstat/core.py"


def test_class_fidelity_audit_has_unique_matlab_names() -> None:
    payload = _load_audit()
    names = [str(item.get("matlab_name", "")).strip() for item in payload["items"]]
    assert len(names) == len(set(names))
