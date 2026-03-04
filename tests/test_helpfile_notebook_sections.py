from __future__ import annotations

import os
import re
from pathlib import Path

import nbformat
import yaml


HELPFILE_MANIFEST = Path("parity/helpfile_notebook_manifest.yml")


def _load_manifest_rows() -> list[dict]:
    payload = yaml.safe_load(HELPFILE_MANIFEST.read_text(encoding="utf-8")) or {}
    return [dict(row) for row in payload.get("notebooks", [])]


def _resolve_matlab_help_root() -> Path | None:
    env = os.environ.get("NSTAT_MATLAB_HELP_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.exists():
            return candidate
    for candidate in (
        Path("/tmp/upstream-nstat/helpfiles"),
        Path.home()
        / "Library"
        / "CloudStorage"
        / "Dropbox"
        / "Research"
        / "Matlab"
        / "nSTAT_currentRelease_Local"
        / "helpfiles",
    ):
        if candidate.exists():
            return candidate
    return None


def _section_count_from_helpfile(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return 1
    sections = 0
    current_has_lines = False
    for line in lines:
        if re.match(r"^\s*%%", line):
            if current_has_lines:
                sections += 1
            current_has_lines = True
        else:
            current_has_lines = True
    if current_has_lines:
        sections += 1
    return sections


def test_helpfile_manifest_has_required_fields() -> None:
    rows = _load_manifest_rows()
    assert rows, "helpfile notebook manifest is empty"
    for row in rows:
        assert "topic" in row
        assert "file" in row
        assert "run_group" in row
        assert "matlab_helpfile" in row
        assert int(row["section_count"]) >= 1
        assert int(row["cell_count"]) >= 1
        assert int(row["expected_figure_count"]) >= 1


def test_helpfile_notebooks_are_code_only_and_cell_counts_match_sections() -> None:
    rows = _load_manifest_rows()
    for row in rows:
        topic = str(row["topic"])
        notebook_path = Path(str(row["file"]))
        nb = nbformat.read(notebook_path, as_version=4)

        assert nb.cells, f"{topic}: no notebook cells"
        assert all(cell.cell_type == "code" for cell in nb.cells), f"{topic}: contains non-code cells"

        section_count = int(row["section_count"])
        cell_count = int(row["cell_count"])
        assert len(nb.cells) == section_count, f"{topic}: notebook cell count != section_count"
        assert len(nb.cells) == cell_count, f"{topic}: notebook cell count != manifest cell_count"

        code = "\n".join(cell.source for cell in nb.cells)
        assert "# MATLAB" in code or "# %" in code, f"{topic}: missing MATLAB trace comments"


def test_helpfile_section_counts_match_matlab_when_available() -> None:
    matlab_help_root = _resolve_matlab_help_root()
    if matlab_help_root is None:
        return

    rows = _load_manifest_rows()
    for row in rows:
        topic = str(row["topic"])
        helpfile_rel = str(row["matlab_helpfile"])
        helpfile_path = matlab_help_root / helpfile_rel
        assert helpfile_path.exists(), f"{topic}: missing MATLAB helpfile {helpfile_path}"
        expected = _section_count_from_helpfile(helpfile_path)
        actual = int(row["section_count"])
        assert actual == expected, f"{topic}: section_count mismatch (manifest={actual}, matlab={expected})"
