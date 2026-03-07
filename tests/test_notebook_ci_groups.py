from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_MANIFEST_PATH = REPO_ROOT / "tools" / "notebooks" / "notebook_manifest.yml"
TOPIC_GROUPS_PATH = REPO_ROOT / "tools" / "notebooks" / "topic_groups.yml"

REQUIRED_CI_SMOKE_TOPICS = {
    "ConfidenceIntervalOverview",
}
REQUIRED_PARITY_CORE_TOPICS = {
    "AnalysisExamples",
    "DecodingExample",
    "DecodingExampleWithHist",
    "ExplicitStimulusWhiskerData",
    "HippocampalPlaceCellExample",
    "HybridFilterExample",
    "PPSimExample",
    "SignalObjExamples",
    "StimulusDecode2D",
    "TrialExamples",
    "ValidationDataSet",
    "nSTATPaperExamples",
    "nSpikeTrainExamples",
}
REQUIRED_HELPFILE_FULL_TOPICS = {
    "AnalysisExamples",
    "DecodingExample",
    "DecodingExampleWithHist",
    "ExplicitStimulusWhiskerData",
    "HippocampalPlaceCellExample",
    "HybridFilterExample",
    "PPSimExample",
    "StimulusDecode2D",
    "TrialExamples",
    "ValidationDataSet",
    "nSTATPaperExamples",
}


def test_ci_smoke_group_covers_required_parity_notebooks() -> None:
    notebook_manifest = yaml.safe_load(NOTEBOOK_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    notebook_topics = {row["topic"] for row in notebook_manifest.get("notebooks", [])}

    groups_payload = yaml.safe_load(TOPIC_GROUPS_PATH.read_text(encoding="utf-8")) or {}
    ci_smoke = set(groups_payload.get("groups", {}).get("ci_smoke", []))

    assert REQUIRED_CI_SMOKE_TOPICS <= notebook_topics
    assert REQUIRED_CI_SMOKE_TOPICS <= ci_smoke


def test_ci_smoke_group_topics_exist_in_notebook_manifest() -> None:
    notebook_manifest = yaml.safe_load(NOTEBOOK_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    notebook_topics = {row["topic"] for row in notebook_manifest.get("notebooks", [])}

    groups_payload = yaml.safe_load(TOPIC_GROUPS_PATH.read_text(encoding="utf-8")) or {}
    ci_smoke = groups_payload.get("groups", {}).get("ci_smoke", [])

    missing = [topic for topic in ci_smoke if topic not in notebook_topics]
    assert not missing, f"CI smoke group references unknown notebook topics: {missing}"


def test_parity_core_group_covers_required_helpfile_parity_notebooks() -> None:
    notebook_manifest = yaml.safe_load(NOTEBOOK_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    notebook_topics = {row["topic"] for row in notebook_manifest.get("notebooks", [])}

    groups_payload = yaml.safe_load(TOPIC_GROUPS_PATH.read_text(encoding="utf-8")) or {}
    parity_core = set(groups_payload.get("groups", {}).get("parity_core", []))

    assert REQUIRED_PARITY_CORE_TOPICS <= notebook_topics
    assert REQUIRED_PARITY_CORE_TOPICS <= parity_core


def test_parity_core_group_extends_ci_smoke_coverage() -> None:
    groups_payload = yaml.safe_load(TOPIC_GROUPS_PATH.read_text(encoding="utf-8")) or {}
    groups = groups_payload.get("groups", {})
    ci_smoke = set(groups.get("ci_smoke", []))
    parity_core = set(groups.get("parity_core", []))

    assert ci_smoke < parity_core


def test_parity_core_group_topics_exist_in_notebook_manifest() -> None:
    notebook_manifest = yaml.safe_load(NOTEBOOK_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    notebook_topics = {row["topic"] for row in notebook_manifest.get("notebooks", [])}

    groups_payload = yaml.safe_load(TOPIC_GROUPS_PATH.read_text(encoding="utf-8")) or {}
    parity_core = groups_payload.get("groups", {}).get("parity_core", [])

    missing = [topic for topic in parity_core if topic not in notebook_topics]
    assert not missing, f"parity_core group references unknown notebook topics: {missing}"


def test_helpfile_full_group_covers_all_tracked_helpfile_ports() -> None:
    notebook_manifest = yaml.safe_load(NOTEBOOK_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    notebook_topics = {row["topic"] for row in notebook_manifest.get("notebooks", [])}

    groups_payload = yaml.safe_load(TOPIC_GROUPS_PATH.read_text(encoding="utf-8")) or {}
    helpfile_full = set(groups_payload.get("groups", {}).get("helpfile_full", []))

    assert REQUIRED_HELPFILE_FULL_TOPICS <= notebook_topics
    assert REQUIRED_HELPFILE_FULL_TOPICS == helpfile_full
