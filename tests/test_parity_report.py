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
    assert "all canonical paper examples and committed gallery directories are mapped" in text
    assert "class fidelity" in text.lower()
    assert "Notebook Fidelity Summary" in text
    assert "Simulink Fidelity Summary" in text
    assert "Remaining Notebook-Fidelity Deltas" in text
    assert "workflow coverage is complete, but 1 MATLAB-helpfile notebook ports are still marked partial" in text
    assert "`StimulusDecode2D` -> `notebooks/StimulusDecode2D.ipynb` [partial]" in text
    assert "No partial or missing items remain in the mapping inventory." in text
    assert "Remaining Class-Fidelity Deltas" in text
    assert "the class audit reports no partial, wrapper-only, or missing items" in text
    assert "No partial, wrapper-only, or missing class-fidelity items remain." in text
    assert "Simulink Fidelity Deltas" in text
    assert "native Python coverage exists for the required published workflows" in text
    assert "reference-only" in text
    assert "nstatOpenHelpPage" in text
