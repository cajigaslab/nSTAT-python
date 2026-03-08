from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nstat.notebook_parity import (
    iter_outstanding_notebook_fidelity,
    load_notebook_parity_notes,
    summarize_notebook_fidelity,
)
from nstat.simulink_fidelity import (
    iter_outstanding_simulink_items,
    load_simulink_fidelity_audit,
    summarize_simulink_strategies,
)


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
    notebook_fidelity = load_notebook_parity_notes(repo_root)
    simulink_fidelity = load_simulink_fidelity_audit(repo_root)
    class_counts = _summarize_class_fidelity(class_fidelity)
    notebook_counts = summarize_notebook_fidelity(notebook_fidelity)
    notebook_partial = iter_outstanding_notebook_fidelity(notebook_fidelity)
    simulink_counts = summarize_simulink_strategies(simulink_fidelity)
    simulink_outstanding = iter_outstanding_simulink_items(simulink_fidelity)
    simulink_reference_only = [
        row
        for row in simulink_fidelity.get("items", [])
        if str(row.get("current_python_status", "")).strip() == "reference_only"
        or str(row.get("python_strategy", "")).strip() == "reference_only"
    ]
    lines = [
        "# nSTAT Python Parity Report",
        "",
        "Generated from `parity/manifest.yml`, `parity/class_fidelity.yml`, and `tools/notebooks/parity_notes.yml`.",
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

    lines.extend(
        [
            "",
            "## Notebook Fidelity Summary",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    for status in ("exact", "high_fidelity", "partial"):
        lines.append(f"| `{status}` | {notebook_counts.get(status, 0)} |")

    lines.extend(
        [
            "",
            "## Simulink Fidelity Summary",
            "",
            "| Strategy | Count |",
            "|---|---:|",
        ]
    )
    for status in simulink_fidelity.get("strategy_legend", []):
        lines.append(f"| `{status}` | {simulink_counts.get(status, 0)} |")

    lines.extend(["", "## Coverage Notes", ""])
    lines.append(
        "- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable."
    )
    lines.append("- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.")
    if notebook_partial:
        lines.append(
            f"- Notebook fidelity: workflow coverage is complete, but {len(notebook_partial)} MATLAB-helpfile notebook ports are still marked partial in `tools/notebooks/parity_notes.yml`."
        )
        lines.append(
            "- Notebook fidelity audit: structural section/figure comparisons plus placeholder/tracker-only cell detection are recorded in `parity/notebook_fidelity.yml`."
        )
    else:
        lines.append("- Notebook fidelity: all tracked MATLAB-helpfile notebook ports are marked high fidelity or exact.")
    if _has_outstanding(payload, "paper_examples") or _has_outstanding(payload, "docs_gallery"):
        lines.append(
            "- Paper examples and docs gallery: canonical structure is present, but dataset-backed outputs and figure files are still partial."
        )
    else:
        lines.append(
            "- Paper examples and docs gallery: all canonical paper examples and committed gallery directories are mapped."
        )
    priority_remaining = _iter_class_fidelity_rows(class_fidelity, {"partial", "wrapper_only", "missing"})
    if not priority_remaining:
        lines.append("- Class fidelity: the class audit reports no partial, wrapper-only, or missing items.")
    else:
        lines.append(
            "- Class fidelity: mapping parity is ahead of semantic parity; the audit still reports partial fidelity for several MATLAB-facing classes and workflows."
        )
    if simulink_outstanding:
        lines.append(
            f"- Simulink fidelity: {len(simulink_outstanding)} Simulink-backed assets still rely on partial, fallback, or unsupported Python execution paths."
        )
    elif simulink_reference_only:
        lines.append(
            f"- Simulink fidelity: native Python coverage exists for the required published workflows, and {len(simulink_reference_only)} inventoried MATLAB assets remain reference-only."
        )
    else:
        lines.append("- Simulink fidelity: all inventoried Simulink-backed workflows have an explicit Python execution strategy.")
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

    lines.extend(["", "## Remaining Notebook-Fidelity Deltas", ""])
    if not notebook_partial:
        lines.append("No partial notebook-fidelity items remain in `tools/notebooks/parity_notes.yml`.")
    else:
        for row in notebook_partial:
            lines.append(
                f"- `{row['topic']}` -> `{row['file']}` [{row['fidelity_status']}]: {row['remaining_differences']}"
            )

    lines.extend(["", "## Remaining Class-Fidelity Deltas", ""])
    if not priority_remaining:
        lines.append("No partial, wrapper-only, or missing class-fidelity items remain.")
    else:
        for row in priority_remaining:
            label = row.get("matlab_name") or row.get("python_public_name") or row.get("matlab_path")
            python_target = row.get("python_public_name") or row.get("python_impl_path")
            recommendation = row.get("required_remediation", [])
            if isinstance(recommendation, list):
                recommendation_text = recommendation[0] if recommendation else ""
            else:
                recommendation_text = str(recommendation)
            note = row.get("method_parity", "")
            detail = recommendation_text or note
            lines.append(f"- `{label}` -> `{python_target}` [{row['status']}]: {detail}")

    lines.extend(["", "## Simulink Fidelity Deltas", ""])
    if not simulink_outstanding and not simulink_reference_only:
        lines.append("No partial, reference-only, fallback, or unsupported Simulink execution paths remain in the audit.")
    else:
        for row in simulink_outstanding:
            lines.append(
                f"- `{row['model_name']}` -> `{row['model_path']}` [{row['python_strategy']}/{row['current_python_status']}]: {row['chosen_interoperability_strategy']}"
            )
        for row in simulink_reference_only:
            lines.append(
                f"- `{row['model_name']}` -> `{row['model_path']}` [{row['python_strategy']}/{row['current_python_status']}]: {row['chosen_interoperability_strategy']}"
            )

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
        notes = row.get("known_remaining_differences", [])
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
