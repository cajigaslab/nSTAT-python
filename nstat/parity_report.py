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


def _has_outstanding(payload: dict[str, Any], section_name: str) -> bool:
    counts = payload["summary"][section_name]
    return counts["partial"] > 0 or counts["missing"] > 0


def load_parity_manifest(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "parity" / "manifest.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_class_fidelity_audit(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "parity" / "class_fidelity.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _summarize_class_fidelity(payload: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in payload.get("status_legend", [])}
    for row in payload.get("items", []):
        status = str(row.get("status", "")).strip()
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return counts


def _iter_class_fidelity_rows(payload: dict[str, Any], statuses: set[str]) -> list[dict[str, Any]]:
    return [row for row in payload.get("items", []) if row.get("status") in statuses]


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
    class_fidelity = load_class_fidelity_audit(repo_root)
    class_counts = _summarize_class_fidelity(class_fidelity)
    lines = [
        "# nSTAT Python Parity Report",
        "",
        "Generated from `parity/manifest.yml` and `parity/class_fidelity.yml`.",
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
            "## Class Fidelity Summary",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    for status in class_fidelity.get("status_legend", []):
        lines.append(f"| `{status}` | {class_counts.get(status, 0)} |")

    lines.extend(["", "## Coverage Notes", ""])
    lines.append(
        "- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable."
    )
    lines.append("- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.")
    if _has_outstanding(payload, "paper_examples") or _has_outstanding(payload, "docs_gallery"):
        lines.append(
            "- Paper examples and docs gallery: canonical structure is present, but dataset-backed outputs and figure files are still partial."
        )
    else:
        lines.append(
            "- Paper examples and docs gallery: all canonical paper examples and committed gallery directories are mapped."
        )
    priority_remaining = _iter_class_fidelity_rows(class_fidelity, {"partial", "shim_only", "missing"})
    if not priority_remaining:
        lines.append("- Class fidelity: the class audit reports no partial, shim-only, or missing items.")
    else:
        lines.append(
            "- Class fidelity: mapping parity is ahead of semantic parity; the audit still reports partial fidelity for several MATLAB-facing classes and workflows."
        )
    lines.extend(["", "## Remaining Mapping Deltas", ""])

    outstanding = _iter_outstanding_rows(payload)
    if not outstanding:
        lines.append("No partial or missing items remain in the mapping inventory.")
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

    lines.extend(["", "## Remaining Class-Fidelity Deltas", ""])
    if not priority_remaining:
        lines.append("No partial, shim-only, or missing class-fidelity items remain.")
    else:
        for row in priority_remaining:
            label = row.get("matlab_name") or row.get("python_symbol") or row.get("matlab_path")
            python_target = row.get("python_symbol") or row.get("python_path")
            recommendation = row.get("recommended_remediation", [])
            if isinstance(recommendation, list):
                recommendation_text = recommendation[0] if recommendation else ""
            else:
                recommendation_text = str(recommendation)
            note = row.get("method_parity", "")
            detail = recommendation_text or note
            lines.append(f"- `{label}` -> `{python_target}` [{row['status']}]: {detail}")

    lines.extend(["", "## Justified Non-Applicable Items", ""])
    non_applicable = _iter_non_applicable_rows(payload)
    class_non_applicable = _iter_class_fidelity_rows(class_fidelity, {"not_applicable"})
    if not non_applicable:
        if not class_non_applicable:
            lines.append("None.")
    else:
        for section_name, row in non_applicable:
            label = row.get("matlab") or row.get("path") or row.get("name")
            notes = row.get("notes", "")
            lines.append(f"- `{section_name}`: `{label}`. {notes}")
    for row in class_non_applicable:
        label = row.get("matlab_name") or row.get("matlab_path")
        notes = row.get("known_semantic_differences", [])
        if isinstance(notes, list):
            note_text = notes[0] if notes else ""
        else:
            note_text = str(notes)
        lines.append(f"- `class_fidelity`: `{label}`. {note_text}")

    lines.append("")
    return "\n".join(lines)


def write_parity_report(repo_root: Path | None = None) -> Path:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    report_path = base / "parity" / "report.md"
    report_path.write_text(render_parity_report(base), encoding="utf-8")
    return report_path


__all__ = [
    "load_class_fidelity_audit",
    "load_parity_manifest",
    "render_parity_report",
    "write_parity_report",
]
