from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_MATLAB_CLASSES = {
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


def test_class_method_inventory_has_all_required_classes() -> None:
    payload = yaml.safe_load(Path("baseline/class_method_inventory.yml").read_text(encoding="utf-8"))
    classes = payload["classes"]
    mapped = {row["matlab_class"] for row in classes}
    assert REQUIRED_MATLAB_CLASSES.issubset(mapped)



def test_example_inventory_has_expected_topics() -> None:
    payload = yaml.safe_load(Path("baseline/example_workflow_inventory.yml").read_text(encoding="utf-8"))
    workflows = payload["workflows"]
    assert len(workflows) == 25
    assert any(row["topic"] == "nSTATPaperExamples" for row in workflows)



def test_paper_mapping_mentions_reference() -> None:
    payload = yaml.safe_load(Path("baseline/paper_section_mapping.yml").read_text(encoding="utf-8"))
    assert payload["paper"]["doi"] == "10.1016/j.jneumeth.2012.08.009"


def test_parity_manifest_exists_and_includes_vertical_slice() -> None:
    payload = yaml.safe_load(Path("baseline/parity_manifest.yml").read_text(encoding="utf-8"))
    assert payload["policy"]["clean_room"] is True
    assert len(payload["classes"]) == 16
    assert any(row["topic"] == "nSTATPaperExamples" for row in payload["workflows"])
