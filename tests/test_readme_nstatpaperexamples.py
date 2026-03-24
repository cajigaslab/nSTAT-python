from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


def test_readme_links_to_generated_paper_gallery_and_figure_dirs() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"## Paper Examples \(Self-Contained\)\n(.*?)\nPlot style policy:",
        text,
        flags=re.S,
    )
    assert match, "README is missing the paper examples block."
    block = match.group(1)

    assert "[docs/paper_examples.md](docs/paper_examples.md)" in block

    for example_id in ("example01", "example02", "example03", "example04", "example05"):
        link = f"[Figures](docs/figures/{example_id}/)"
        assert block.count(link) == 1, f"README must link each canonical figure directory exactly once: {link}"


def test_readme_no_longer_uses_legacy_nstatpaperexamples_overview_block() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "### nSTATPaperExamples" not in text
    assert "example4_nstatpaperexamples_overview.py" not in text
    assert "readme_example4_nstatpaperexamples_overview.png" not in text
