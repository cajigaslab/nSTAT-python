from __future__ import annotations

from pathlib import Path

import nbformat


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
