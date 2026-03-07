from __future__ import annotations

import json
from pathlib import Path

from nstat.paper_gallery import build_gallery_manifest, render_paper_examples_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paper_gallery_manifest_is_generated_from_source_manifest() -> None:
    committed = json.loads((REPO_ROOT / "docs" / "figures" / "manifest.json").read_text(encoding="utf-8"))
    built = build_gallery_manifest(REPO_ROOT)
    assert committed["figure_root"] == built["figure_root"]
    assert committed["examples"] == built["examples"]


def test_paper_examples_markdown_matches_generator() -> None:
    committed = (REPO_ROOT / "docs" / "paper_examples.md").read_text(encoding="utf-8")
    assert committed == render_paper_examples_markdown(REPO_ROOT)


def test_paper_gallery_directories_exist() -> None:
    for example_id in ("example01", "example02", "example03", "example04", "example05"):
        path = REPO_ROOT / "docs" / "figures" / example_id
        assert path.is_dir()
        assert (path / "README.md").exists()
