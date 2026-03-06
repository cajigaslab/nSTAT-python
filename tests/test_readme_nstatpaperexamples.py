from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


def test_readme_includes_nstatpaperexamples_code_and_figure() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    match = re.search(r"### nSTATPaperExamples\n(.*?)\n## Documentation\n", text, flags=re.S)
    assert match, "README is missing the nSTATPaperExamples block."
    block = match.group(1)
    assert "examples/readme_examples/example4_nstatpaperexamples_overview.py" in block
    assert "from nstat.paper_examples import run_paper_examples" in block
    assert "readme_example4_nstatpaperexamples_overview.png" in block
