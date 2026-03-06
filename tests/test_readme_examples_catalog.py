from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


def test_readme_tracks_python_port_top_level_sections() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    for heading in (
        "How to install nSTAT-python",
        "Quickstart (Python 3.10+)",
        "Paper Examples (Self-Contained)",
        "Paper-aligned toolbox map",
        "Standalone Python repository",
    ):
        assert heading in text


def test_readme_documents_automatic_dataset_download() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "downloads it automatically" in lowered or "downloads the figshare dataset automatically" in lowered
    assert "10.6084/m9.figshare.4834640.v3" in text
    assert "NSTAT_DATA_DIR" in text


def test_readme_lists_core_paper_examples_and_runner() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    for label in ("Example 01", "Example 02", "Example 03", "Example 04", "Example 05"):
        assert label in text
    assert "python examples/nSTATPaperExamples.py --repo-root ." in text
