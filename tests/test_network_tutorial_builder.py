from __future__ import annotations

"""Smoke test for the historical NetworkTutorial bootstrap generator.

The generator ``tools/notebook_build/build_network_tutorial_notebook.py``
originally scaffolded ``notebooks/NetworkTutorial.ipynb``.  That notebook has
since been hand-refined, sanitized, parity-annotated, and executed, and is now
the source of truth (see ``tools/notebook_build/README.md``).  Re-running the
generator would *overwrite and corrupt* the curated committed notebook, so the
generator output has intentionally diverged.

This test no longer asserts cell-for-cell equality (that would force a revert
of the curated parity work).  Instead it verifies the generator still imports
cleanly and produces a structurally valid notebook with the same top-level
language metadata, and that the committed notebook still carries its MATLAB
parity-note banner.  That keeps the historical generator covered without
fighting the curated notebook.
"""

from pathlib import Path

import nbformat

from tools.notebook_build.build_network_tutorial_notebook import build_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "NetworkTutorial.ipynb"


def test_network_tutorial_builder_still_runs() -> None:
    """The historical generator must still import and produce a valid notebook."""
    generated = build_notebook()
    nbformat.validate(generated)
    assert len(generated.cells) > 0
    assert generated.metadata.get("language_info", {}).get("name") == "python"


def test_committed_notebook_carries_parity_note() -> None:
    """The committed notebook keeps a MATLAB parity-note banner cell."""
    committed = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(committed)
    first = committed.cells[0]
    assert first.cell_type == "markdown"
    src = "".join(first.get("source", ""))
    assert "parity-note" in src or "MATLAB Parity" in src
