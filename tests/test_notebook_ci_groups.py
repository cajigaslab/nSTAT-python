from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_MANIFEST_PATH = REPO_ROOT / "tools" / "notebooks" / "notebook_manifest.yml"
TOPIC_GROUPS_PATH = REPO_ROOT / "tools" / "notebooks" / "topic_groups.yml"

REQUIRED_CI_SMOKE_TOPICS = {
    "ConfidenceIntervalOverview",
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
