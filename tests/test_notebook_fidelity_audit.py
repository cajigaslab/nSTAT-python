from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nstat.notebook_fidelity_audit import default_matlab_repo_root, render_notebook_fidelity_audit
from nstat.notebook_parity import load_notebook_parity_notes


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "parity" / "notebook_fidelity.yml"


def test_notebook_fidelity_audit_covers_all_parity_notes() -> None:
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    notes = load_notebook_parity_notes(REPO_ROOT)

    audit_topics = {row["topic"] for row in audit.get("items", [])}
    note_topics = {row["topic"] for row in notes}
    assert audit_topics == note_topics


def test_notebook_fidelity_audit_has_structural_counts() -> None:
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    for row in audit.get("items", []):
        assert row["status"] in {"high_fidelity", "exact", "partial", "missing"}
        assert isinstance(row["executable_in_ci"], bool)
        assert "current_run_group" in row
        assert isinstance(row["fixture_backed"], bool)
        assert "python_sections" in row
        assert "python_expected_figures" in row
        assert row["python_expected_figures"] >= 0
        assert isinstance(row["python_has_finalize_call"], bool)
        assert "python_placeholder_cells" in row
        assert "python_tracker_only_cells" in row


def test_notebook_fidelity_audit_marks_upgraded_ports_as_high_fidelity() -> None:
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    upgraded_topics = {row["topic"] for row in audit.get("items", []) if row["status"] in {"high_fidelity", "exact"}}
    assert {
        "AnalysisExamples",
        "AnalysisExamples2",
        "NetworkTutorial",
        "PPSimExample",
        "nSTATPaperExamples",
    } <= upgraded_topics


def test_notebook_fidelity_audit_tracks_only_known_partial_notebooks() -> None:
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    partial_topics = {row["topic"] for row in audit.get("items", []) if row["status"] in {"partial", "missing"}}
    assert partial_topics == set()


def test_high_fidelity_notebooks_have_no_placeholder_or_tracker_only_cells() -> None:
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    for row in audit.get("items", []):
        if row["status"] not in {"high_fidelity", "exact"}:
            continue
        assert not row["python_contains_placeholders"], f"{row['topic']} still contains placeholder code"
        assert not row["python_contains_tracker_only_cells"], f"{row['topic']} still contains tracker-only cells"


def test_high_fidelity_notebooks_have_near_matlab_structural_counts() -> None:
    # Known structural deltas with documented justification:
    #  - nSTATPaperExamples section_delta=3: added CIF/reach/hybrid setup figure cells
    #  - StimulusDecode2D section_delta=-1: removed raster section with no MATLAB equivalent
    #  - DecodingExample figure_delta=-2: MATLAB plotResults publishes 3 images from 1 call
    SECTION_TOLERANCE = 3
    FIGURE_TOLERANCE = 2
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    for row in audit.get("items", []):
        if row["status"] not in {"high_fidelity", "exact"}:
            continue
        if row.get("section_delta") is None or row.get("figure_delta") is None:
            continue
        assert abs(int(row["section_delta"])) <= SECTION_TOLERANCE, f"{row['topic']} has a large MATLAB section delta ({row['section_delta']})"
        assert abs(int(row["figure_delta"])) <= FIGURE_TOLERANCE, f"{row['topic']} has a large MATLAB figure delta ({row['figure_delta']})"


def test_required_notebook_ports_are_executable_in_ci() -> None:
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8")) or {}
    for row in audit.get("items", []):
        assert row["executable_in_ci"] is True
        assert row["current_run_group"] in {"helpfile_full", "parity_core", "ci_smoke", "core", "smoke"}


def test_notebook_fidelity_audit_matches_generator_when_matlab_repo_is_available() -> None:
    matlab_repo = default_matlab_repo_root(REPO_ROOT)
    if not matlab_repo.exists():
        pytest.skip(f"MATLAB reference repo not available at {matlab_repo}")
    committed = AUDIT_PATH.read_text(encoding="utf-8")
    assert committed == render_notebook_fidelity_audit(REPO_ROOT, matlab_repo_root=matlab_repo)
