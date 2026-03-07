from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import yaml

from nstat.paper_figures import list_named_paper_figures


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "examples" / "paper" / "manifest.yml"


def _load_examples() -> list[dict[str, object]]:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(payload["examples"])


def test_figure_builder_registry_matches_manifest() -> None:
    for row in _load_examples():
        expected = [Path(path).name for path in row["figure_files"]]
        assert list_named_paper_figures(row["example_id"]) == expected


def test_committed_png_sets_match_manifest_exactly() -> None:
    for row in _load_examples():
        example_id = row["example_id"]
        figure_dir = REPO_ROOT / "docs" / "figures" / example_id
        expected = sorted(Path(path).name for path in row["figure_files"])
        committed = sorted(path.name for path in figure_dir.glob("*.png"))
        assert committed == expected


def test_committed_pngs_are_readable_images() -> None:
    for row in _load_examples():
        for rel_path in row["figure_files"]:
            image = mpimg.imread(REPO_ROOT / rel_path)
            assert image.ndim in (2, 3)
            assert image.shape[0] > 100
            assert image.shape[1] > 100


def test_gallery_readmes_embed_every_expected_figure() -> None:
    for row in _load_examples():
        example_id = row["example_id"]
        readme = (REPO_ROOT / "docs" / "figures" / example_id / "README.md").read_text(encoding="utf-8")
        for rel_path in row["figure_files"]:
            filename = Path(rel_path).name
            stem = Path(rel_path).stem
            assert f"### {filename}" in readme
            assert f"![{stem}](./{filename})" in readme
