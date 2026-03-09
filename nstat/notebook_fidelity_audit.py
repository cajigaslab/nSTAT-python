from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import nbformat
import yaml

from nstat.notebook_parity import (
    audit_notebook_placeholders,
    extract_figure_contract,
    load_notebook_parity_notes,
)


IMG_SRC_RE = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)
SECTION_RE = re.compile(r"^%%", re.MULTILINE)
PYTHON_SECTION_RE = re.compile(r"^# SECTION\b", re.MULTILINE)
CI_GROUP_ORDER = ("helpfile_full", "parity_core", "ci_smoke", "core", "smoke")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_matlab_repo_root(repo_root: Path | None = None) -> Path:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    return base.parent / "nSTAT"


def _count_matlab_sections(matlab_m_path: Path) -> int:
    text = matlab_m_path.read_text(encoding="utf-8", errors="ignore")
    return len(SECTION_RE.findall(text))


def _count_matlab_published_figures(matlab_html_path: Path) -> int:
    text = matlab_html_path.read_text(encoding="utf-8", errors="ignore")
    return len(IMG_SRC_RE.findall(text))


def _count_python_sections(notebook_path: Path) -> int:
    notebook = nbformat.read(notebook_path, as_version=4)
    text = "\n".join(str(cell.get("source", "")) for cell in notebook.cells)
    return len(PYTHON_SECTION_RE.findall(text))


def _load_notebook_groups(base: Path) -> dict[str, set[str]]:
    payload = yaml.safe_load((base / "tools" / "notebooks" / "topic_groups.yml").read_text(encoding="utf-8")) or {}
    groups = payload.get("groups", {})
    return {str(name): {str(topic) for topic in values or []} for name, values in groups.items()}


def build_notebook_fidelity_audit(
    repo_root: Path | None = None,
    *,
    matlab_repo_root: Path | None = None,
) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    matlab_root = default_matlab_repo_root(base) if matlab_repo_root is None else matlab_repo_root.resolve()
    help_root = matlab_root / "helpfiles"
    notes = load_notebook_parity_notes(base)
    topic_groups = _load_notebook_groups(base)

    items: list[dict[str, Any]] = []
    for row in notes:
        topic = str(row["topic"])
        notebook_path = base / str(row["file"])
        figure_contract = extract_figure_contract(notebook_path)
        placeholder_audit = audit_notebook_placeholders(notebook_path)
        python_sections = _count_python_sections(notebook_path)
        matlab_stem = Path(str(row["source_matlab"])).stem
        matlab_m_path = help_root / f"{matlab_stem}.m"
        matlab_html_path = help_root / f"{matlab_stem}.html"
        matlab_available = matlab_m_path.exists() and matlab_html_path.exists()
        current_run_group = next((group for group in CI_GROUP_ORDER if topic in topic_groups.get(group, set())), None)

        item: dict[str, Any] = {
            "topic": topic,
            "source_matlab": str(row["source_matlab"]),
            "python_notebook": str(row["file"]),
            "status": str(row["fidelity_status"]),
            "fidelity_status": str(row["fidelity_status"]),
            "executable_in_ci": current_run_group is not None,
            "current_run_group": current_run_group,
            "fixture_backed": False,
            "remaining_differences": str(row["remaining_differences"]),
            "python_sections": python_sections,
            "python_expected_figures": int(figure_contract.expected_count) if figure_contract else 0,
            "python_uses_figure_tracker": figure_contract is not None,
            "python_has_finalize_call": bool(figure_contract.has_finalize_call) if figure_contract else False,
            "python_placeholder_cells": placeholder_audit.placeholder_cells,
            "python_tracker_only_cells": placeholder_audit.tracker_only_cells,
            "python_contains_placeholders": placeholder_audit.contains_placeholders,
            "python_contains_tracker_only_cells": placeholder_audit.contains_tracker_only_cells,
        }
        if matlab_available:
            matlab_sections = _count_matlab_sections(matlab_m_path)
            matlab_figures = _count_matlab_published_figures(matlab_html_path)
            item.update(
                {
                    "matlab_repo_root": str(matlab_root),
                    "matlab_sections": matlab_sections,
                    "matlab_published_figures": matlab_figures,
                    "section_delta": python_sections - matlab_sections,
                    "figure_delta": int(figure_contract.expected_count) - matlab_figures if figure_contract else -matlab_figures,
                }
            )
        else:
            item.update(
                {
                    "matlab_repo_root": str(matlab_root),
                    "matlab_sections": None,
                    "matlab_published_figures": None,
                    "section_delta": None,
                    "figure_delta": None,
                }
            )
        items.append(item)

    return {
        "version": 1,
        "generated_on": str(date.today()),
        "source_repositories": {
            "matlab": "https://github.com/cajigaslab/nSTAT",
            "python": "https://github.com/cajigaslab/nSTAT-python",
        },
        "status_legend": ["exact", "high_fidelity", "partial", "missing"],
        "matlab_repo_root": str(matlab_root),
        "items": items,
    }


def render_notebook_fidelity_audit(
    repo_root: Path | None = None,
    *,
    matlab_repo_root: Path | None = None,
) -> str:
    payload = build_notebook_fidelity_audit(repo_root, matlab_repo_root=matlab_repo_root)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def write_notebook_fidelity_audit(
    repo_root: Path | None = None,
    *,
    matlab_repo_root: Path | None = None,
) -> Path:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    out = base / "parity" / "notebook_fidelity.yml"
    out.write_text(
        render_notebook_fidelity_audit(base, matlab_repo_root=matlab_repo_root),
        encoding="utf-8",
    )
    return out


__all__ = [
    "build_notebook_fidelity_audit",
    "default_matlab_repo_root",
    "render_notebook_fidelity_audit",
    "write_notebook_fidelity_audit",
]
