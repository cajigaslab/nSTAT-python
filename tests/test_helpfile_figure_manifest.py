from __future__ import annotations

import json
from pathlib import Path

import nbformat
import yaml


FIG_MANIFEST = Path("parity/helpfile_figure_manifest.json")
HELPFILE_MANIFEST = Path("parity/helpfile_notebook_manifest.yml")


def test_helpfile_figure_manifest_schema() -> None:
    payload = json.loads(FIG_MANIFEST.read_text(encoding="utf-8"))
    assert int(payload.get("schema_version", 0)) >= 1
    topics = payload.get("topics", {})
    assert isinstance(topics, dict) and topics
    for topic, row in topics.items():
        assert isinstance(topic, str) and topic
        assert isinstance(row, dict)
        assert "matlab_helpfile_path" in row
        assert int(row.get("total_figures_expected", -1)) >= 0
        events = row.get("events", [])
        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, dict)
            assert int(event.get("section_index", 0)) >= 1
            assert int(event.get("matlab_line_number", 0)) >= 1
            assert str(event.get("event_type", "")) in {"new_figure", "add_to_current"}
            assert int(event.get("figure_ordinal", 0)) >= 1


def test_helpfile_manifest_figure_counts_match_figure_manifest() -> None:
    fig_payload = json.loads(FIG_MANIFEST.read_text(encoding="utf-8"))
    fig_topics = fig_payload.get("topics", {})
    help_rows = (yaml.safe_load(HELPFILE_MANIFEST.read_text(encoding="utf-8")) or {}).get("notebooks", [])
    for row in help_rows:
        topic = str(row["topic"])
        assert topic in fig_topics
        expected = int(fig_topics[topic]["total_figures_expected"])
        assert int(row["expected_min_figures"]) == expected
        assert int(row["expected_figure_count"]) == expected


def test_notebooks_include_figure_tracker_assertion() -> None:
    help_rows = (yaml.safe_load(HELPFILE_MANIFEST.read_text(encoding="utf-8")) or {}).get("notebooks", [])
    for row in help_rows:
        nb = nbformat.read(Path(str(row["notebook_path"])), as_version=4)
        code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "FigureTracker" in code, f"{row['topic']}: missing FigureTracker usage"
        assert "FIGURE_TRACKER.finalize()" in code, f"{row['topic']}: missing FigureTracker.finalize()"
