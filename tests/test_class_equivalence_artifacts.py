from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from tools.parity.generate_class_equivalence_inventory import build_inventory


def _resolve_matlab_root(repo_root: Path) -> Path | None:
    env_candidates = [
        os.environ.get("NSTAT_MATLAB_ROOT"),
        os.environ.get("MATLAB_NSTAT_ROOT"),
    ]
    for candidate in env_candidates:
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                return path

    checkout_meta = repo_root / "parity/matlab_reference_checkout.json"
    if checkout_meta.exists():
        payload = json.loads(checkout_meta.read_text(encoding="utf-8"))
        path = Path(str(payload.get("dest", ""))).expanduser().resolve()
        if path.exists():
            return path

    default_path = Path("/tmp/upstream-nstat")
    if default_path.exists():
        return default_path

    return None


def _build_runtime_artifacts(tmp_path: Path) -> tuple[dict, dict]:
    repo_root = Path(__file__).resolve().parents[1]
    matlab_root = _resolve_matlab_root(repo_root)
    if matlab_root is None:
        pytest.skip(
            "MATLAB reference checkout not available; set NSTAT_MATLAB_ROOT "
            "or provide /tmp/upstream-nstat to generate class-equivalence artifacts."
        )

    args = Namespace(
        repo_root=repo_root,
        matlab_root=matlab_root,
        method_mapping=Path("parity/method_mapping.yaml"),
        method_exclusions=Path("parity/method_exclusions.yml"),
        class_contracts=Path("parity/class_contracts.yml"),
        fixture_spec=Path("parity/class_fixture_export_spec.yml"),
        behavior_contracts=[Path("tests/parity/class_behavior_specs.yml"), Path("tests/parity/compat_behavior_specs.yml")],
        out_inventory=tmp_path / "class_equivalence_inventory.json",
        out_report=tmp_path / "class_equivalence_report.json",
    )
    inventory, report = build_inventory(args)
    args.out_inventory.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory, report


def test_class_equivalence_inventory_runtime_generation_has_full_class_coverage(tmp_path: Path) -> None:
    payload, _ = _build_runtime_artifacts(tmp_path)
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


def test_class_equivalence_report_runtime_generation_is_clean(tmp_path: Path) -> None:
    _, payload = _build_runtime_artifacts(tmp_path)
    summary = payload.get("summary", {})
    assert int(summary.get("total_classes", 0)) >= 16
    assert int(summary.get("gap_classes", 1)) == 0
    assert bool(summary.get("required_class_coverage_ok", False))

    top_methods = payload.get("top_critical_methods_tested", {})
    assert len(top_methods) >= 16
    for cls, methods in top_methods.items():
        assert methods, f"missing critical methods for {cls}"
