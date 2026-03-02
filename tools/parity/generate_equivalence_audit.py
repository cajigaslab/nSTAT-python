#!/usr/bin/env python3
"""Generate method-level and example-level equivalence audit artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import nbformat
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI envs
    nbformat = None


IMG_SRC_RE = re.compile(r'<img[^>]+src="([^"]+)"', flags=re.IGNORECASE)


@dataclass(slots=True)
class MatlabCodeStats:
    total_code_lines: int
    blocks: list[dict[str, Any]]


@dataclass(slots=True)
class NotebookCodeStats:
    total_code_lines: int
    cells: list[dict[str, Any]]


@dataclass(slots=True)
class NotebookValidationStats:
    has_topic_checkpoint: bool
    assertion_count: int
    has_plot_call: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--matlab-root", type=Path, required=True)
    parser.add_argument("--method-mapping", type=Path, default=Path("parity/method_mapping.yaml"))
    parser.add_argument("--example-mapping", type=Path, default=Path("parity/example_mapping.yaml"))
    parser.add_argument("--matlab-inventory", type=Path, default=Path("parity/matlab_api_inventory.json"))
    parser.add_argument("--python-inventory", type=Path, default=Path("parity/python_api_inventory.json"))
    parser.add_argument(
        "--method-exclusions",
        type=Path,
        default=Path("parity/method_exclusions.yml"),
        help="Optional YAML file listing MATLAB methods intentionally excluded (e.g., documented stubs).",
    )
    parser.add_argument(
        "--probe-report",
        type=Path,
        default=Path("parity/method_probe_report.json"),
        help="Optional method probe JSON report used to mark additional methods as functionally verified.",
    )
    parser.add_argument(
        "--class-contracts",
        type=Path,
        action="append",
        default=[
            Path("tests/parity/class_behavior_specs.yml"),
            Path("tests/parity/compat_behavior_specs.yml"),
        ],
        help="YAML behavior contract specs to include for functional coverage.",
    )
    parser.add_argument(
        "--validation-image-root",
        type=Path,
        default=Path("tmp/pdfs/validation_report/notebook_images"),
        help="Root directory containing extracted notebook figure images by topic",
    )
    parser.add_argument(
        "--fallback-validation-image-root",
        type=Path,
        default=Path("baseline/validation/notebook_images"),
        help="Fallback root directory for tracked validation images when tmp outputs are absent.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Output report path",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _matlab_methods_by_class(inventory: dict[str, Any]) -> dict[str, list[str]]:
    return {str(row["matlab_class"]): list(row["methods"]) for row in inventory["classes"]}


def _python_surfaces_by_class(inventory: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for row in inventory["classes"]:
        cls = str(row["matlab_class"])
        py_surface = set(row["python"]["methods"]) | set(row["python"]["properties"]) | set(
            row["python"]["fields"]
        )
        compat_surface = set(row["compat"]["methods"]) | set(row["compat"]["properties"]) | set(
            row["compat"]["fields"]
        )
        out[cls] = {"python": py_surface, "compat": compat_surface}
    return out


def _contract_members_by_class(specs: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for spec in specs:
        for row in spec.get("classes", []):
            cls = str(row["matlab_class"])
            members = {str(contract["member"]) for contract in row.get("contracts", [])}
            out.setdefault(cls, set()).update(members)
    return out


def _probe_verified_members_by_class(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    out: dict[str, set[str]] = {}
    for row in payload.get("class_rows", []):
        klass = str(row.get("matlab_class", ""))
        if not klass:
            continue
        methods = {str(item) for item in row.get("success_methods", [])}
        out[klass] = methods
    return out


def _excluded_methods_by_class(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    payload = _load_yaml(path)
    out: dict[str, set[str]] = {}
    for row in payload.get("classes", []):
        klass = str(row.get("matlab_class", ""))
        if not klass:
            continue
        methods = {str(method) for method in row.get("methods", [])}
        out[klass] = methods
    return out


def _extract_matlab_code_stats(path: Path) -> MatlabCodeStats:
    if not path.exists():
        return MatlabCodeStats(total_code_lines=0, blocks=[])

    code_lines: list[tuple[int, str]] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            continue
        code_lines.append((idx, line))

    blocks: list[dict[str, Any]] = []
    if code_lines:
        block_start, first_line = code_lines[0]
        prev = block_start
        count = 1
        preview = first_line
        for idx, line in code_lines[1:]:
            if idx == prev + 1:
                count += 1
                prev = idx
                continue
            blocks.append(
                {
                    "start_line": block_start,
                    "end_line": prev,
                    "line_count": count,
                    "preview": preview[:140],
                }
            )
            block_start = idx
            prev = idx
            count = 1
            preview = line
        blocks.append(
            {
                "start_line": block_start,
                "end_line": prev,
                "line_count": count,
                "preview": preview[:140],
            }
        )

    return MatlabCodeStats(total_code_lines=len(code_lines), blocks=blocks)


def _extract_notebook_code_stats(path: Path) -> NotebookCodeStats:
    if not path.exists():
        return NotebookCodeStats(total_code_lines=0, cells=[])

    if nbformat is not None:
        payload = nbformat.read(path, as_version=4)
        notebook_cells = payload.cells
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        notebook_cells = raw.get("cells", [])

    cells: list[dict[str, Any]] = []
    total = 0
    for i, cell in enumerate(notebook_cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        src_raw = cell.get("source", "")
        if isinstance(src_raw, list):
            src = "".join(str(part) for part in src_raw)
        else:
            src = str(src_raw)
        filtered = []
        for line in src.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            filtered.append(stripped)
        line_count = len(filtered)
        total += line_count
        if line_count == 0:
            continue
        cells.append(
            {
                "cell_index": i,
                "line_count": line_count,
                "preview": filtered[0][:140],
            }
        )
    return NotebookCodeStats(total_code_lines=total, cells=cells)


def _load_notebook_cells(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if nbformat is not None:
        payload = nbformat.read(path, as_version=4)
        return list(payload.cells)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("cells", []))


def _extract_notebook_validation_stats(path: Path) -> NotebookValidationStats:
    code = []
    for cell in _load_notebook_cells(path):
        if cell.get("cell_type") != "code":
            continue
        src_raw = cell.get("source", "")
        if isinstance(src_raw, list):
            src = "".join(str(part) for part in src_raw)
        else:
            src = str(src_raw)
        code.append(src)

    code_text = "\n".join(code)
    has_topic_checkpoint = "Topic-specific checkpoint" in code_text and "Notebook checkpoints passed" in code_text
    assertion_count = sum(1 for line in code_text.splitlines() if line.strip().startswith("assert "))
    has_plot_call = any(
        token in code_text
        for token in ("plt.", ".plot(", "imshow(", "pcolor(", "mesh(", "scatter(", "hist(")
    )
    return NotebookValidationStats(
        has_topic_checkpoint=has_topic_checkpoint,
        assertion_count=assertion_count,
        has_plot_call=has_plot_call,
    )


def _collect_matlab_reference_images(help_root: Path, topic: str) -> list[str]:
    topic_lower = topic.lower()
    found: list[Path] = []
    seen: set[Path] = set()

    def add_if_valid(path: Path) -> None:
        if not path.exists():
            return
        name = path.name.lower()
        if name.startswith(f"{topic_lower}_eq"):
            return
        if "eq" in name and name.startswith(topic_lower):
            return
        if name.startswith("logo"):
            return
        if path in seen:
            return
        seen.add(path)
        found.append(path)

    html_path = help_root / f"{topic}.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        for src in IMG_SRC_RE.findall(html):
            candidate = help_root / Path(src).name
            add_if_valid(candidate)

    for pattern in (f"{topic}_*.png", f"{topic}.png", f"{topic}-*.png"):
        for candidate in sorted(help_root.glob(pattern)):
            add_if_valid(candidate)

    ordered = sorted(found, key=lambda p: p.name.lower())
    return [str(path) for path in ordered]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    matlab_root = args.matlab_root.resolve()
    help_root = matlab_root / "helpfiles"

    method_mapping = _load_yaml((repo_root / args.method_mapping).resolve())
    example_mapping = _load_yaml((repo_root / args.example_mapping).resolve())
    matlab_inventory = _load_json((repo_root / args.matlab_inventory).resolve())
    python_inventory = _load_json((repo_root / args.python_inventory).resolve())
    class_contract_specs: list[dict[str, Any]] = []
    for spec_path in args.class_contracts:
        resolved = (repo_root / spec_path).resolve()
        if not resolved.exists():
            continue
        class_contract_specs.append(_load_yaml(resolved))

    matlab_methods = _matlab_methods_by_class(matlab_inventory)
    python_surfaces = _python_surfaces_by_class(python_inventory)
    contract_members = _contract_members_by_class(class_contract_specs)
    probe_members = _probe_verified_members_by_class((repo_root / args.probe_report).resolve())
    excluded_methods = _excluded_methods_by_class((repo_root / args.method_exclusions).resolve())

    class_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for row in method_mapping["classes"]:
        matlab_class = str(row["matlab_class"])
        alias_methods = dict(row.get("alias_methods", {}))
        py_surface = python_surfaces[matlab_class]["python"]
        compat_surface = python_surfaces[matlab_class]["compat"]
        covered_contracts = contract_members.get(matlab_class, set())
        covered_probe = probe_members.get(matlab_class, set())

        class_methods = matlab_methods[matlab_class]
        missing_symbol_count = 0
        verified_count = 0
        unverified_count = 0
        excluded_count = 0

        for method in class_methods:
            mapped_member = str(alias_methods.get(method, method))
            in_python = mapped_member in py_surface
            in_compat = mapped_member in compat_surface
            symbol_ok = in_python or in_compat
            is_excluded = method in excluded_methods.get(matlab_class, set())
            if is_excluded:
                status = "excluded_matlab_stub"
                excluded_count += 1
            elif not symbol_ok:
                status = "missing_symbol"
                missing_symbol_count += 1
            elif mapped_member in covered_contracts:
                status = "contract_verified"
                verified_count += 1
            elif mapped_member in covered_probe:
                status = "probe_verified"
                verified_count += 1
            else:
                status = "unverified_behavior"
                unverified_count += 1

            method_rows.append(
                {
                    "matlab_class": matlab_class,
                    "matlab_method": method,
                    "mapped_python_member": mapped_member,
                    "mapped_via_alias": method in alias_methods,
                    "present_in_python_surface": in_python,
                    "present_in_compat_surface": in_compat,
                    "excluded_method": is_excluded,
                    "has_behavior_contract": mapped_member in covered_contracts,
                    "has_probe_verification": mapped_member in covered_probe,
                    "functional_status": status,
                }
            )

        class_rows.append(
            {
                "matlab_class": matlab_class,
                "matlab_method_count": len(class_methods),
                "contract_verified_count": verified_count,
                "unverified_behavior_count": unverified_count,
                "missing_symbol_count": missing_symbol_count,
                "excluded_method_count": excluded_count,
                "contract_coverage_ratio": float(verified_count / max(len(class_methods), 1)),
                "probe_verified_count": sum(
                    1
                    for row in method_rows
                    if row["matlab_class"] == matlab_class and row["functional_status"] == "probe_verified"
                ),
                "eligible_method_count": len(class_methods) - excluded_count,
                "eligible_verified_ratio": float(verified_count / max(len(class_methods) - excluded_count, 1)),
            }
        )

    example_rows: list[dict[str, Any]] = []
    validation_root = (repo_root / args.validation_image_root).resolve()
    fallback_validation_root = (repo_root / args.fallback_validation_image_root).resolve()
    for row in example_mapping["examples"]:
        topic = str(row["matlab_topic"])
        matlab_file = help_root / f"{topic}.m"
        python_nb = (repo_root / str(row["python_notebook"])).resolve()

        matlab_stats = _extract_matlab_code_stats(matlab_file)
        notebook_stats = _extract_notebook_code_stats(python_nb)
        notebook_validation = _extract_notebook_validation_stats(python_nb)

        reference_images = _collect_matlab_reference_images(help_root, topic)
        python_img_dir = validation_root / topic
        if python_img_dir.exists():
            python_images = sorted(str(path) for path in python_img_dir.glob("*.png"))
        else:
            python_images = []
        if not python_images:
            fallback_img_dir = fallback_validation_root / topic
            if fallback_img_dir.exists():
                python_images = sorted(str(path) for path in fallback_img_dir.glob("*.png"))

        if not matlab_file.exists() or not python_nb.exists():
            alignment_status = "missing_artifact"
        elif matlab_stats.total_code_lines == 0 and notebook_stats.total_code_lines == 0:
            alignment_status = "doc_only"
        elif matlab_stats.total_code_lines == 0 and notebook_stats.total_code_lines > 0:
            alignment_status = "matlab_doc_only"
        elif matlab_stats.total_code_lines > 0 and notebook_stats.total_code_lines == 0:
            alignment_status = "missing_executable_content"
        else:
            if (
                notebook_validation.has_topic_checkpoint
                and notebook_validation.assertion_count >= 2
                and len(python_images) >= 1
            ):
                alignment_status = "validated"
            else:
                alignment_status = "pending_manual_review"

        line_ratio = (
            float(notebook_stats.total_code_lines / matlab_stats.total_code_lines)
            if matlab_stats.total_code_lines > 0
            else None
        )

        example_rows.append(
            {
                "topic": topic,
                "matlab_file": str(matlab_file),
                "python_notebook": str(python_nb),
                "alignment_status": alignment_status,
                "matlab_code_lines": matlab_stats.total_code_lines,
                "python_code_lines": notebook_stats.total_code_lines,
                "python_to_matlab_line_ratio": line_ratio,
                "matlab_code_blocks": matlab_stats.blocks,
                "python_code_cells": notebook_stats.cells,
                "has_topic_checkpoint": notebook_validation.has_topic_checkpoint,
                "assertion_count": notebook_validation.assertion_count,
                "has_plot_call": notebook_validation.has_plot_call,
                "matlab_reference_image_count": len(reference_images),
                "python_validation_image_count": len(python_images),
                "matlab_reference_images": reference_images[:12],
                "python_validation_images": python_images[:12],
            }
        )

    total_methods = len(method_rows)
    total_verified = sum(
        1 for row in method_rows if row["functional_status"] in {"contract_verified", "probe_verified"}
    )
    total_contract_verified = sum(1 for row in method_rows if row["functional_status"] == "contract_verified")
    total_probe_verified = sum(1 for row in method_rows if row["functional_status"] == "probe_verified")
    total_excluded = sum(1 for row in method_rows if row["functional_status"] == "excluded_matlab_stub")
    total_unverified = sum(1 for row in method_rows if row["functional_status"] == "unverified_behavior")
    total_missing = sum(1 for row in method_rows if row["functional_status"] == "missing_symbol")
    total_eligible = total_methods - total_excluded

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matlab_root": str(matlab_root),
        "repo_root": str(repo_root),
        "method_functional_audit": {
            "summary": {
                "total_methods": total_methods,
                "contract_verified_methods": total_verified,
                "contract_explicit_verified_methods": total_contract_verified,
                "probe_verified_methods": total_probe_verified,
                "excluded_methods": total_excluded,
                "eligible_methods": total_eligible,
                "unverified_behavior_methods": total_unverified,
                "missing_symbol_methods": total_missing,
                "contract_verified_ratio": float(total_verified / max(total_methods, 1)),
                "eligible_verified_ratio": float(total_verified / max(total_eligible, 1)),
            },
            "class_summary": class_rows,
            "method_rows": method_rows,
        },
        "example_line_alignment_audit": {
            "summary": {
                "total_topics": len(example_rows),
                "missing_artifact_topics": sum(
                    1 for row in example_rows if row["alignment_status"] == "missing_artifact"
                ),
                "missing_executable_topics": sum(
                    1 for row in example_rows if row["alignment_status"] == "missing_executable_content"
                ),
                "pending_manual_review_topics": sum(
                    1 for row in example_rows if row["alignment_status"] == "pending_manual_review"
                ),
                "validated_topics": sum(1 for row in example_rows if row["alignment_status"] == "validated"),
                "matlab_doc_only_topics": sum(
                    1 for row in example_rows if row["alignment_status"] == "matlab_doc_only"
                ),
                "doc_only_topics": sum(1 for row in example_rows if row["alignment_status"] == "doc_only"),
            },
            "topic_rows": example_rows,
        },
    }

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote equivalence audit report to {out_path}")
    print(
        "Method functional audit: "
        f"total={total_methods}, excluded={total_excluded}, verified={total_verified}, "
        f"unverified={total_unverified}, missing={total_missing}"
    )
    print(
        "Example alignment audit: "
        f"topics={len(example_rows)}, pending_manual_review="
        f"{sum(1 for row in example_rows if row['alignment_status'] == 'pending_manual_review')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
