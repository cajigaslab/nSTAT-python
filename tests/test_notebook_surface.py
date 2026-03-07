from __future__ import annotations

from pathlib import Path

import nbformat
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_python_facing() -> None:
    notebook_dir = REPO_ROOT / "notebooks"
    for path in sorted(notebook_dir.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        meta = nb.metadata.get("nstat", {})
        assert "source_file" not in meta, f"{path.name} still exposes source_file metadata"
        assert "source_type" not in meta, f"{path.name} still exposes source_type metadata"
        assert "strict_section_cell_mapping" not in meta, f"{path.name} still exposes MATLAB mapping metadata"

        text = "\n".join(cell.source for cell in nb.cells)
        assert "# MATLAB L" not in text, f"{path.name} still contains MATLAB line comments"
        assert "_matlab(" not in text, f"{path.name} still contains MATLAB placeholder calls"
        assert "AUTO-GENERATED FROM MATLAB" not in text, f"{path.name} still advertises MATLAB generation"


def test_readme_catalog_is_python_facing() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Notebook generated from MATLAB help source" not in readme


def test_confidence_interval_overview_is_catalogued() -> None:
    notebook_manifest = yaml.safe_load((REPO_ROOT / "tools" / "notebooks" / "notebook_manifest.yml").read_text(encoding="utf-8"))
    topics = {row["topic"] for row in notebook_manifest["notebooks"]}
    assert "ConfidenceIntervalOverview" in topics

    example_manifest = yaml.safe_load((REPO_ROOT / "examples" / "nSTATPaperExamples" / "manifest.yml").read_text(encoding="utf-8"))
    names = {row["name"] for row in example_manifest["examples"]}
    assert "ConfidenceIntervalOverview" in names


def test_hybrid_filter_notebook_does_not_require_example_data_download() -> None:
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "HybridFilterExample.ipynb", as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)

    assert "ensure_example_data(download=True)" not in text
    assert "from nstat.data_manager import ensure_example_data" not in text
