"""Shared writer for the historical helpfile-bootstrap notebook generators.

Factored out of the byte-identical ``_write_notebook`` that was duplicated in
``build_analysis_help_notebooks.py``, ``build_foundational_help_notebooks.py``
and ``build_helpfile_fidelity_notebooks.py``.

Those generators are *historical bootstrap scaffolding* — see this directory's
README.md.  The committed notebooks under ``notebooks/`` are the source of
truth and have diverged from generator output; do not re-run the generators.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

LANGUAGE_METADATA = {"language_info": {"name": "python"}}


def write_notebook(
    path: Path,
    *,
    topic: str,
    expected_figures: int,
    markdown_note: str,
    code_cells: list[str],
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell(markdown_note),
            *[new_code_cell(dedent(cell).strip() + "\n") for cell in code_cells],
        ],
        metadata={
            **LANGUAGE_METADATA,
            "nstat": {
                "expected_figures": expected_figures,
                "run_group": "smoke",
                "style": "python-example",
                "topic": topic,
            },
        },
    )
    path.write_text(nbformat.writes(notebook), encoding="utf-8")
