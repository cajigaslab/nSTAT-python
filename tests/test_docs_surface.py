from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_CONF_PATH = REPO_ROOT / "docs" / "conf.py"
DOCS_INDEX_PATH = REPO_ROOT / "docs" / "index.rst"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_docs_index_includes_paper_examples_page() -> None:
    text = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    assert "paper_examples" in text


def test_docs_conf_enables_markdown_support() -> None:
    text = DOCS_CONF_PATH.read_text(encoding="utf-8")
    assert 'extensions = ["myst_parser"]' in text


def test_ci_runs_docs_build_job() -> None:
    payload = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8")) or {}
    jobs = payload.get("jobs", {})
    assert "docs-build" in jobs
    steps = jobs["docs-build"].get("steps", [])
    run_lines = [step.get("run", "") for step in steps if isinstance(step, dict)]
    assert any("build_gallery.py" in line for line in run_lines)
    assert any("python -m sphinx -W -b html docs docs/_build/html" in line for line in run_lines)
