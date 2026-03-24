from __future__ import annotations

from pathlib import Path

import nbformat

from tools.notebooks.build_network_tutorial_notebook import build_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "NetworkTutorial.ipynb"


def _normalize_notebook(notebook) -> None:
    for cell in notebook.cells:
        cell["id"] = "normalized"
        cell["execution_count"] = None
        cell["outputs"] = []
        # Strip execution-related cell metadata added by nbconvert
        cell.get("metadata", {}).pop("execution", None)
    # Strip kernel-specific metadata that changes after execution
    notebook.metadata.pop("language_info", None)


def _cell_payload(cell) -> tuple[str, str, dict]:
    return cell.cell_type, "".join(cell.get("source", "")), dict(cell.get("metadata", {}))


def test_network_tutorial_builder_matches_committed_notebook() -> None:
    committed = nbformat.read(NOTEBOOK_PATH, as_version=4)
    generated = build_notebook()
    _normalize_notebook(committed)
    _normalize_notebook(generated)
    assert committed.metadata == generated.metadata
    assert len(committed.cells) == len(generated.cells)
    assert [_cell_payload(cell) for cell in committed.cells] == [_cell_payload(cell) for cell in generated.cells]
