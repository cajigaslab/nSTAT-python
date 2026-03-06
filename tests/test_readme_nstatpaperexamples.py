from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


def test_readme_states_python_repo_is_standalone_from_matlab_repo() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "does not require a MATLAB checkout" in text
    assert "cajigaslab/nSTAT" in text
    assert "nstat.compat.matlab" in text
