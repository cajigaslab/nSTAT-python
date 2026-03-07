from __future__ import annotations

from pathlib import Path

from nstat.parity_report import render_parity_report


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "parity" / "report.md"


def test_committed_parity_report_matches_generator() -> None:
    committed = REPORT_PATH.read_text(encoding="utf-8")
    assert committed == render_parity_report(REPO_ROOT)


def test_parity_report_highlights_current_constraints() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "no missing MATLAB public APIs remain" in text
    assert "paper examples and docs gallery" in text.lower()
    assert "nstatOpenHelpPage" in text
