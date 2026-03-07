from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SUMMARY_SECTIONS = (
    "public_api",
    "help_workflows",
    "paper_examples",
    "docs_gallery",
    "installer_setup",
    "repo_structure",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_parity_manifest(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "parity" / "manifest.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _iter_outstanding_rows(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for section_name in SUMMARY_SECTIONS:
        for row in payload.get(section_name, []):
            if row.get("status") in {"partial", "missing"}:
                rows.append((section_name, row))
    return rows


def _iter_non_applicable_rows(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for section_name in SUMMARY_SECTIONS:
        for row in payload.get(section_name, []):
            if row.get("status") == "not_applicable":
                rows.append((section_name, row))
    return rows


def render_parity_report(repo_root: Path | None = None) -> str:
    payload = load_parity_manifest(repo_root)
    lines = [
        "# nSTAT Python Parity Report",
        "",
        "Generated from `parity/manifest.yml`.",
        "",
        f"- MATLAB reference: {payload['source_repositories']['matlab']}",
        f"- Python target: {payload['source_repositories']['python']}",
        f"- Inventory version: {payload['version']}",
        f"- Generated on: {payload['generated_on']}",
        "",
        "## Summary",
        "",
        "| Section | Mapped | Partial | Missing | Not Applicable |",
        "|---|---:|---:|---:|---:|",
    ]
    for section_name in SUMMARY_SECTIONS:
        counts = payload["summary"][section_name]
        label = section_name.replace("_", " ")
        lines.append(
            f"| `{label}` | {counts['mapped']} | {counts['partial']} | {counts['missing']} | {counts['not_applicable']} |"
        )

    lines.extend(
        [
            "",
            "## Coverage Notes",
            "",
            "- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable.",
            "- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.",
            "- Paper examples and docs gallery: canonical structure is present, but dataset-backed outputs and figure files are still partial.",
            "",
            "## Remaining Deltas",
            "",
        ]
    )

    outstanding = _iter_outstanding_rows(payload)
    if not outstanding:
        lines.append("No partial or missing items remain.")
    else:
        current_section = ""
        for section_name, row in outstanding:
            if section_name != current_section:
                if current_section:
                    lines.append("")
                lines.append(f"### `{section_name}`")
                lines.append("")
                current_section = section_name
            label = row.get("matlab") or row.get("python_target") or row.get("path") or row.get("name")
            python_target = row.get("python_target") or row.get("python_script") or row.get("python_path") or row.get("path")
            notes = row.get("notes", "")
            lines.append(f"- `{label}` -> `{python_target}`: {notes}")

    lines.extend(["", "## Justified Non-Applicable Items", ""])
    non_applicable = _iter_non_applicable_rows(payload)
    if not non_applicable:
        lines.append("None.")
    else:
        for section_name, row in non_applicable:
            label = row.get("matlab") or row.get("path") or row.get("name")
            notes = row.get("notes", "")
            lines.append(f"- `{section_name}`: `{label}`. {notes}")

    lines.append("")
    return "\n".join(lines)


def write_parity_report(repo_root: Path | None = None) -> Path:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    report_path = base / "parity" / "report.md"
    report_path.write_text(render_parity_report(base), encoding="utf-8")
    return report_path


__all__ = [
    "load_parity_manifest",
    "render_parity_report",
    "write_parity_report",
]
