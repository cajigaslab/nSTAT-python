from __future__ import annotations

import json
from pathlib import Path


def test_class_equivalence_inventory_exists_and_has_full_class_coverage() -> None:
    inventory_path = Path("parity/class_equivalence_inventory.json")
    assert inventory_path.exists(), "Missing parity/class_equivalence_inventory.json"

    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = payload.get("class_rows", [])
    assert len(rows) >= 16

    statuses = {str(row.get("status", "")) for row in rows}
    assert "gap_missing_mapping" not in statuses

    required = {
        "SignalObj",
        "Covariate",
        "ConfidenceInterval",
        "Events",
        "History",
        "nspikeTrain",
        "nstColl",
        "CovColl",
        "TrialConfig",
        "ConfigColl",
        "Trial",
        "CIF",
        "Analysis",
        "FitResult",
        "FitResSummary",
        "DecodingAlgorithms",
    }
    covered = {str(row["matlab_class"]) for row in rows}
    assert required.issubset(covered)

    summary = payload.get("summary", {})
    assert summary.get("required_classes_missing_from_matlab_scan", []) == []
    assert summary.get("classes_missing_mapping", []) == []


def test_class_equivalence_report_exists_and_is_clean() -> None:
    report_path = Path("parity/class_equivalence_report.json")
    assert report_path.exists(), "Missing parity/class_equivalence_report.json"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    assert int(summary.get("total_classes", 0)) >= 16
    assert int(summary.get("gap_classes", 1)) == 0
    assert bool(summary.get("required_class_coverage_ok", False))

    top_methods = payload.get("top_critical_methods_tested", {})
    assert len(top_methods) >= 16
    for cls, methods in top_methods.items():
        assert methods, f"missing critical methods for {cls}"
