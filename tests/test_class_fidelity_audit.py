from __future__ import annotations

from pathlib import Path

import yaml

from nstat.class_fidelity import iter_symbol_presence_mismatches


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "parity" / "class_fidelity.yml"
VALID_STATUSES = {"exact", "high_fidelity", "partial", "wrapper_only", "missing", "not_applicable"}
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


def test_core_matlab_facing_classes_are_not_wrapper_only() -> None:
    payload = _load_audit()
    audit_by_name = {str(item["matlab_name"]): item for item in payload["items"]}

    for name in ("SignalObj", "Covariate", "nspikeTrain"):
        row = audit_by_name[name]
        assert row["status"] not in {"wrapper_only", "missing"}
        assert row["python_impl_path"] == "nstat/core.py"


def test_class_fidelity_audit_has_unique_matlab_names() -> None:
    payload = _load_audit()
    names = [str(item.get("matlab_name", "")).strip() for item in payload["items"]]
    assert len(names) == len(set(names))


def test_class_fidelity_audit_uses_requested_field_names() -> None:
    payload = _load_audit()
    required = {
        "matlab_name",
        "matlab_path",
        "python_public_name",
        "python_impl_path",
        "status",
        "constructor_parity",
        "property_parity",
        "method_parity",
        "defaults_parity",
        "indexing_parity",
        "symbol_presence_verified",
        "plotting_report_parity",
        "known_remaining_differences",
        "required_remediation",
    }
    for item in payload["items"]:
        assert required <= set(item), f"Missing required class-fidelity fields for {item.get('matlab_name')}"


def test_class_fidelity_audit_uses_yes_no_symbol_presence_flags() -> None:
    payload = _load_audit()
    for item in payload["items"]:
        assert str(item["symbol_presence_verified"]).strip().lower() in {"true", "false", "yes", "no"}


def test_class_fidelity_symbol_presence_matches_runtime_resolution() -> None:
    payload = _load_audit()
    assert not iter_symbol_presence_mismatches(payload)
