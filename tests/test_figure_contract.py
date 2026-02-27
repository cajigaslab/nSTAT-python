from __future__ import annotations

import json
from pathlib import Path


def test_figure_contract_completeness(project_root: Path) -> None:
    contract_path = project_root / "examples" / "help_topics" / "figure_contract.json"
    data = json.loads(contract_path.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    assert isinstance(topics, dict)
    assert len(topics) == 25

    zero_topics = {name for name, info in topics.items() if int(info.get("expected_figures", 0)) == 0}
    assert zero_topics == {
        "TrialConfigExamples",
        "ConfigCollExamples",
        "FitResultExamples",
        "FitResSummaryExamples",
    }
