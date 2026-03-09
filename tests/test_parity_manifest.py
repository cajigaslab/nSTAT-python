from __future__ import annotations

from pathlib import Path

from collections import Counter

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "parity" / "manifest.yml"
NOTEBOOK_AUDIT_PATH = REPO_ROOT / "parity" / "notebook_fidelity.yml"
CLASS_AUDIT_PATH = REPO_ROOT / "parity" / "class_fidelity.yml"
SIMULINK_AUDIT_PATH = REPO_ROOT / "parity" / "simulink_fidelity.yml"

EXPECTED_MATLAB_PUBLIC_API = {
    "Analysis",
    "CIF",
    "ConfidenceInterval",
    "ConfigColl",
    "CovColl",
    "Covariate",
    "DecodingAlgorithms",
    "Events",
    "FitResSummary",
    "FitResult",
    "History",
    "SignalObj",
    "Trial",
    "TrialConfig",
    "getPaperDataDirs",
    "nSTAT_Install",
    "nspikeTrain",
    "nstColl",
    "nstatOpenHelpPage",
}

EXPECTED_HELP_WORKFLOWS = {
    "AnalysisExamples",
    "AnalysisExamples2",
    "ClassDefinitions",
    "ConfidenceIntervalOverview",
    "ConfigCollExamples",
    "CovCollExamples",
    "CovariateExamples",
    "DecodingExample",
    "DecodingExampleWithHist",
    "DocumentationSetup2025b",
    "EventsExamples",
    "Examples",
    "ExplicitStimulusWhiskerData",
    "FitResSummaryExamples",
    "FitResultExamples",
    "FitResultReference",
    "HippocampalPlaceCellExample",
    "HistoryExamples",
    "HybridFilterExample",
    "NetworkTutorial",
    "NeuralSpikeAnalysis_top",
    "PaperOverview",
    "PPSimExample",
    "PPThinning",
    "PSTHEstimation",
    "SignalObjExamples",
    "StimulusDecode2D",
    "TrialConfigExamples",
    "TrialExamples",
    "ValidationDataSet",
    "mEPSCAnalysis",
    "nSTATPaperExamples",
    "nSpikeTrainExamples",
    "nstCollExamples",
}

VALID_STATUSES = {"mapped", "partial", "missing", "not_applicable"}


def _load_manifest() -> dict:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_parity_manifest_public_api_coverage() -> None:
    payload = _load_manifest()
    entries = payload["public_api"]
    names = {row["matlab"] for row in entries}
    assert names == EXPECTED_MATLAB_PUBLIC_API


def test_parity_manifest_help_workflow_coverage() -> None:
    payload = _load_manifest()
    entries = payload["help_workflows"]
    names = {row["matlab"] for row in entries}
    assert names == EXPECTED_HELP_WORKFLOWS


def test_parity_manifest_statuses_and_mapped_targets_are_valid() -> None:
    payload = _load_manifest()
    for section_name in ("public_api", "help_workflows", "paper_examples", "docs_gallery", "installer_setup", "repo_structure"):
        for row in payload[section_name]:
            status = row["status"]
            assert status in VALID_STATUSES
            target = row.get("python_target")
            if status == "mapped":
                assert target, f"Mapped item in {section_name} is missing a python_target: {row}"


def test_manifest_help_workflows_align_with_notebook_fidelity_audit() -> None:
    manifest = _load_manifest()
    notebook_audit = yaml.safe_load(NOTEBOOK_AUDIT_PATH.read_text(encoding="utf-8")) or {}
    audit_rows = {row["topic"]: row for row in notebook_audit.get("items", [])}

    manifest_help_rows = {
        row["matlab"]: row
        for row in manifest["help_workflows"]
        if str(row.get("python_target", "")).startswith("notebooks/")
    }

    assert set(audit_rows) <= set(manifest_help_rows)
    for topic, audit_row in audit_rows.items():
        manifest_row = manifest_help_rows[topic]
        audit_row = audit_rows[topic]
        if manifest_row["status"] == "mapped":
            assert audit_row["status"] in {"high_fidelity", "exact"}


def test_manifest_fidelity_summary_matches_detailed_audits() -> None:
    manifest = _load_manifest()
    class_audit = yaml.safe_load(CLASS_AUDIT_PATH.read_text(encoding="utf-8")) or {}
    notebook_audit = yaml.safe_load(NOTEBOOK_AUDIT_PATH.read_text(encoding="utf-8")) or {}
    simulink_audit = yaml.safe_load(SIMULINK_AUDIT_PATH.read_text(encoding="utf-8")) or {}

    expected = {
        "class_fidelity": dict(Counter(str(row.get("status", "")).strip() for row in class_audit.get("items", []))),
        "notebook_fidelity": dict(Counter(str(row.get("status", "")).strip() for row in notebook_audit.get("items", []))),
        "simulink_fidelity": dict(Counter(str(row.get("status", "")).strip() for row in simulink_audit.get("items", []))),
    }
    assert manifest["fidelity_summary"] == expected
