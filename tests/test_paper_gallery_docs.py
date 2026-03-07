from __future__ import annotations

import json
from pathlib import Path

import yaml

from nstat.paper_gallery import build_gallery_manifest, render_paper_examples_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paper_gallery_manifest_is_generated_from_source_manifest() -> None:
    committed = json.loads((REPO_ROOT / "docs" / "figures" / "manifest.json").read_text(encoding="utf-8"))
    built = build_gallery_manifest(REPO_ROOT)
    assert committed == built


def test_paper_examples_markdown_matches_generator() -> None:
    committed = (REPO_ROOT / "docs" / "paper_examples.md").read_text(encoding="utf-8")
    assert committed == render_paper_examples_markdown(REPO_ROOT)


def test_paper_gallery_directories_exist() -> None:
    for example_id in ("example01", "example02", "example03", "example04", "example05"):
        path = REPO_ROOT / "docs" / "figures" / example_id
        assert path.is_dir()
        assert (path / "README.md").exists()


def test_mapped_gallery_directories_have_committed_figure_sets() -> None:
    parity = yaml.safe_load((REPO_ROOT / "parity" / "manifest.yml").read_text(encoding="utf-8")) or {}
    paper_manifest = yaml.safe_load((REPO_ROOT / "examples" / "paper" / "manifest.yml").read_text(encoding="utf-8")) or {}
    figure_map = {row["example_id"]: row["figure_files"] for row in paper_manifest.get("examples", [])}

    for row in parity.get("docs_gallery", []):
        target = str(row.get("python_target", ""))
        if not target.startswith("docs/figures/example") or row.get("status") != "mapped":
            continue
        example_id = Path(target).name
        for rel_path in figure_map.get(example_id, []):
            assert (REPO_ROOT / rel_path).exists(), f"Missing committed gallery figure for mapped example: {rel_path}"
