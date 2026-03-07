#!/usr/bin/env python3
"""Synchronize MATLAB parity note cells into selected notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTES_PATH = REPO_ROOT / "tools" / "notebooks" / "parity_notes.yml"
MARKER = "<!-- parity-note -->"


def build_note(source_matlab: str, fidelity_status: str, remaining_differences: str) -> str:
    return "\n".join(
        [
            MARKER,
            "## MATLAB Parity Note",
            f"- Source MATLAB helpfile: `{source_matlab}`",
            f"- Fidelity status: `{fidelity_status}`",
            f"- Remaining justified differences: {remaining_differences}",
        ]
    )


def sync_notebook(path: Path, note_text: str) -> None:
    notebook = nbformat.read(path, as_version=4)
    parity_cell = nbformat.v4.new_markdown_cell(note_text)
    if notebook.cells and notebook.cells[0].cell_type == "markdown" and MARKER in "".join(notebook.cells[0].get("source", "")):
        notebook.cells[0] = parity_cell
    else:
        notebook.cells.insert(0, parity_cell)
    nbformat.write(notebook, path)


def main() -> int:
    payload = yaml.safe_load(NOTES_PATH.read_text(encoding="utf-8")) or {}
    for row in payload.get("notes", []):
        path = REPO_ROOT / str(row["file"])
        sync_notebook(
            path,
            build_note(
                str(row["source_matlab"]),
                str(row["fidelity_status"]),
                str(row["remaining_differences"]),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
