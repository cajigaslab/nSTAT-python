from __future__ import annotations

from pathlib import Path

import nbformat


TOPICS = [
    "AnalysisExamples2",
    "DocumentationSetup2025b",
    "FitResultReference",
    "HybridFilterExample",
    "publish_all_helpfiles",
]



def test_new_topics_have_assertion_cells() -> None:
    for topic in TOPICS:
        path = Path("notebooks") / f"{topic}.ipynb"
        nb = nbformat.read(path, as_version=4)
        code = "\n".join(
            cell.source for cell in nb.cells if cell.cell_type == "code"
        )
        assert "Notebook checkpoints passed" in code
        assert "Topic-specific checkpoint" in code
