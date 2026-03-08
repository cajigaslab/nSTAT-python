from __future__ import annotations

from pathlib import Path

import nbformat
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTES_PATH = REPO_ROOT / "tools" / "notebooks" / "parity_notes.yml"
TOPIC_GROUPS_PATH = REPO_ROOT / "tools" / "notebooks" / "topic_groups.yml"
MARKER = "<!-- parity-note -->"


def _load_notes() -> list[dict[str, str]]:
    payload = yaml.safe_load(NOTES_PATH.read_text(encoding="utf-8")) or {}
    return list(payload.get("notes", []))


def test_parity_notes_topics_are_covered_by_parity_core_group() -> None:
    topic_groups = yaml.safe_load(TOPIC_GROUPS_PATH.read_text(encoding="utf-8")) or {}
    parity_core = set(topic_groups.get("groups", {}).get("parity_core", []))
    note_topics = {row["topic"] for row in _load_notes()}
    assert note_topics <= parity_core


def test_target_notebooks_start_with_machine_readable_parity_note() -> None:
    for row in _load_notes():
        notebook_path = REPO_ROOT / row["file"]
        notebook = nbformat.read(notebook_path, as_version=4)
        first_cell = notebook.cells[0]
        source = "".join(first_cell.get("source", ""))

        assert first_cell.cell_type == "markdown", f"{notebook_path} must start with a markdown parity note"
        assert MARKER in source, f"{notebook_path} is missing the parity note marker"
        assert row["source_matlab"] in source
        assert row["fidelity_status"] in source
        assert row["remaining_differences"] in source


def test_notebook_parity_notes_track_only_known_partial_statuses() -> None:
    partial = [row["topic"] for row in _load_notes() if row["fidelity_status"] == "partial"]
    assert partial == []


def test_high_fidelity_parity_notes_do_not_admit_placeholder_or_tracker_only_status() -> None:
    forbidden = ("placeholder", "tracker-only", "partial fidelity", "stubbed")
    for row in _load_notes():
        if row["fidelity_status"] not in {"high_fidelity", "exact"}:
            continue
        notebook_path = REPO_ROOT / row["file"]
        notebook = nbformat.read(notebook_path, as_version=4)
        source = "".join(notebook.cells[0].get("source", "")).lower()
        assert not any(term in source for term in forbidden), f"{notebook_path} still self-reports reduced fidelity"
