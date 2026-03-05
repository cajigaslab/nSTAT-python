from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


def _examples_block(text: str) -> str:
    match = re.search(r"## Examples\n(.*?)\n## Documentation\n", text, flags=re.S)
    assert match, "README missing Examples section"
    return match.group(1)


def test_readme_includes_nstatpaperexamples_code_and_figure() -> None:
    block = _examples_block(README.read_text(encoding="utf-8"))
    start = block.find("### nSTATPaperExamples")
    assert start >= 0, "README missing nSTATPaperExamples section"
    section = block[start:]

    assert "```python" in section, "nSTATPaperExamples section must include a Python code block"
    assert "![nSTATPaperExamples" in section, "nSTATPaperExamples section must embed at least one figure"
    assert "notebooks/nSTATPaperExamples.ipynb" in section, "Section must link runnable notebook"
    assert "examples/nSTATPaperExamples.py" in section, "Section must link runnable script"


def test_readme_nstatpaperexamples_image_exists() -> None:
    readme = README.read_text(encoding="utf-8")
    match = re.search(r"!\[nSTATPaperExamples[^\]]*\]\(([^)]+)\)", readme)
    assert match, "README missing embedded nSTATPaperExamples image"
    image_path = (REPO_ROOT / match.group(1)).resolve()
    assert image_path.exists(), f"Referenced README image missing: {image_path}"
