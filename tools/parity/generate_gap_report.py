#!/usr/bin/env python3
"""Generate machine-readable parity gap report from inventories and mappings."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--method-mapping", type=Path, default=Path("parity/method_mapping.yaml"))
    parser.add_argument("--example-mapping", type=Path, default=Path("parity/example_mapping.yaml"))
    parser.add_argument("--matlab-inventory", type=Path, default=Path("parity/matlab_api_inventory.json"))
    parser.add_argument("--python-inventory", type=Path, default=Path("parity/python_api_inventory.json"))
    parser.add_argument("--help-toc", type=Path, default=Path("docs/help/helptoc.yml"))
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("parity/parity_gap_report.json"),
        help="Output JSON report",
    )
    parser.add_argument(
        "--fail-on",
        choices=["none", "low", "medium", "high"],
        default="high",
        help="Fail if any issue at or above this severity exists",
    )
    return parser.parse_args()


def _add_issue(issues: list[dict[str, Any]], severity: str, check: str, message: str, details: dict[str, Any]) -> None:
    issues.append(
        {
            "severity": severity,
            "check": check,
            "message": message,
            "details": details,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_lookup(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows}


def _exists_symbol(path: str, repo_root: Path) -> bool:
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    try:
        module_name, attr_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return hasattr(module, attr_name)
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    method_mapping = _load_yaml((repo_root / args.method_mapping).resolve())
    example_mapping = _load_yaml((repo_root / args.example_mapping).resolve())
    matlab_inventory = _load_json((repo_root / args.matlab_inventory).resolve())
    python_inventory = _load_json((repo_root / args.python_inventory).resolve())
    help_toc = _load_yaml((repo_root / args.help_toc).resolve())

    issues: list[dict[str, Any]] = []

    matlab_by_class = _build_lookup(matlab_inventory["classes"], "matlab_class")
    python_by_class = _build_lookup(python_inventory["classes"], "matlab_class")

    mapped_classes = method_mapping["classes"]
    class_coverage: list[dict[str, Any]] = []

    help_toc_text = json.dumps(help_toc).lower()

    for row in mapped_classes:
        matlab_class = str(row["matlab_class"])
        python_class = str(row["python_class"])
        compat_class = str(row["compat_class"])
        aliases: dict[str, str] = dict(row.get("alias_methods", {}))

        if matlab_class not in matlab_by_class:
            _add_issue(
                issues,
                "high",
                "missing_matlab_class_inventory",
                f"MATLAB class missing from inventory: {matlab_class}",
                {"matlab_class": matlab_class},
            )
            continue

        if matlab_class not in python_by_class:
            _add_issue(
                issues,
                "high",
                "missing_python_class_inventory",
                f"Python class missing from inventory: {matlab_class}",
                {"matlab_class": matlab_class, "python_class": python_class},
            )
            continue

        if not _exists_symbol(python_class, repo_root):
            _add_issue(
                issues,
                "high",
                "missing_python_class",
                f"Python class symbol not importable: {python_class}",
                {"matlab_class": matlab_class, "python_class": python_class},
            )

        if not _exists_symbol(compat_class, repo_root):
            _add_issue(
                issues,
                "high",
                "missing_compat_class",
                f"Compatibility class symbol not importable: {compat_class}",
                {"matlab_class": matlab_class, "compat_class": compat_class},
            )

        matlab_methods = set(matlab_by_class[matlab_class]["methods"])
        py_row = python_by_class[matlab_class]
        python_surface = set(py_row["python"]["methods"]) | set(py_row["python"]["properties"]) | set(
            py_row["python"]["fields"]
        )
        compat_surface = set(py_row["compat"]["methods"]) | set(py_row["compat"]["properties"]) | set(
            py_row["compat"]["fields"]
        )

        missing_methods: list[str] = []
        mapped_count = 0

        for method in sorted(matlab_methods):
            if method in aliases:
                target = aliases[method]
                if target in python_surface or target in compat_surface:
                    mapped_count += 1
                else:
                    missing_methods.append(method)
                continue

            if method in python_surface or method in compat_surface:
                mapped_count += 1
            else:
                missing_methods.append(method)

        if missing_methods:
            _add_issue(
                issues,
                "medium",
                "missing_method_mappings",
                f"{matlab_class} has {len(missing_methods)} unmapped MATLAB methods",
                {
                    "matlab_class": matlab_class,
                    "missing_methods": missing_methods,
                    "matlab_method_count": len(matlab_methods),
                },
            )

        class_help = repo_root / "docs" / "help" / "classes" / f"{matlab_class}.md"
        if not class_help.exists():
            _add_issue(
                issues,
                "high",
                "missing_class_help",
                f"Missing class help page for {matlab_class}",
                {"class_help": str(class_help)},
            )

        class_coverage.append(
            {
                "matlab_class": matlab_class,
                "matlab_method_count": len(matlab_methods),
                "mapped_method_count": mapped_count,
                "missing_method_count": len(missing_methods),
                "coverage_ratio": float(mapped_count / max(len(matlab_methods), 1)),
            }
        )

    example_rows = example_mapping["examples"]
    example_coverage: list[dict[str, Any]] = []

    for row in example_rows:
        topic = str(row["matlab_topic"])
        notebook = (repo_root / str(row["python_notebook"])).resolve()
        help_page = (repo_root / str(row["python_help_page"])).resolve()

        notebook_exists = notebook.exists()
        help_exists = help_page.exists()

        if not notebook_exists:
            _add_issue(
                issues,
                "high",
                "missing_notebook",
                f"Missing notebook for topic {topic}",
                {"topic": topic, "notebook": str(notebook)},
            )
        if not help_exists:
            _add_issue(
                issues,
                "high",
                "missing_example_help",
                f"Missing help page for topic {topic}",
                {"topic": topic, "help_page": str(help_page)},
            )

        topic_token = topic.lower()
        if topic_token not in help_toc_text:
            _add_issue(
                issues,
                "medium",
                "topic_missing_from_help_toc",
                f"Topic not present in docs/help/helptoc.yml: {topic}",
                {"topic": topic},
            )

        example_coverage.append(
            {
                "topic": topic,
                "notebook_exists": notebook_exists,
                "help_exists": help_exists,
            }
        )

    summary = {
        "high": sum(1 for i in issues if i["severity"] == "high"),
        "medium": sum(1 for i in issues if i["severity"] == "medium"),
        "low": sum(1 for i in issues if i["severity"] == "low"),
    }
    summary["total"] = summary["high"] + summary["medium"] + summary["low"]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fail_on": args.fail_on,
        "summary": summary,
        "class_coverage": class_coverage,
        "example_coverage": example_coverage,
        "issues": issues,
    }

    out_path = (repo_root / args.report_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote parity gap report to {out_path}")
    print(f"Issue summary: high={summary['high']}, medium={summary['medium']}, low={summary['low']}")

    threshold = SEVERITY_ORDER[args.fail_on]
    max_issue = max((SEVERITY_ORDER[i["severity"]] for i in issues), default=0)
    if max_issue >= threshold and threshold > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
