from __future__ import annotations

from pathlib import Path



def test_docs_notebook_catalog_exists() -> None:
    catalog = Path("docs/notebooks.md")
    assert catalog.exists()

    text = catalog.read_text(encoding="utf-8")
    assert "AnalysisExamples.ipynb" in text
    assert "nSTATPaperExamples.ipynb" in text
