#!/usr/bin/env python3
"""Enforce strict MATLAB->Python mapped method coverage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--method-mapping", type=Path, default=Path("parity/method_mapping.yaml"))
    parser.add_argument("--method-exclusions", type=Path, default=Path("parity/method_exclusions.yml"))
    parser.add_argument("--matlab-inventory", type=Path, default=Path("parity/matlab_api_inventory.json"))
    parser.add_argument("--python-inventory", type=Path, default=Path("parity/python_api_inventory.json"))
    parser.add_argument("--report-out", type=Path, default=Path("parity/method_mapping_gate_report.json"))
    parser.add_argument(
        "--fail-on-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return non-zero when any mapped MATLAB method is missing from Python surfaces.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusion_lookup(payload: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in payload.get("classes", []):
        matlab_class = str(row.get("matlab_class", "")).strip()
        methods = {str(method) for method in row.get("methods", [])}
        if matlab_class:
            out[matlab_class] = methods
    return out


def _python_surface(class_row: dict[str, Any]) -> set[str]:
    py = class_row.get("python", {})
    compat = class_row.get("compat", {})
    return set(py.get("methods", [])) | set(py.get("properties", [])) | set(py.get("fields", [])) | set(
        compat.get("methods", [])
    ) | set(compat.get("properties", [])) | set(compat.get("fields", []))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    method_mapping = _load_yaml((repo_root / args.method_mapping).resolve())
    matlab_inventory = _load_json((repo_root / args.matlab_inventory).resolve())
    python_inventory = _load_json((repo_root / args.python_inventory).resolve())

    exclusions: dict[str, set[str]] = {}
    exclusions_path = (repo_root / args.method_exclusions).resolve()
    if exclusions_path.exists():
        exclusions = _exclusion_lookup(_load_yaml(exclusions_path))

    matlab_by_class = {str(row["matlab_class"]): row for row in matlab_inventory.get("classes", [])}
    python_by_class = {str(row["matlab_class"]): row for row in python_inventory.get("classes", [])}

    class_rows: list[dict[str, Any]] = []
    missing_total = 0
    considered_total = 0

    for row in method_mapping.get("classes", []):
        matlab_class = str(row.get("matlab_class", ""))
        aliases = dict(row.get("alias_methods", {}))
        matlab_row = matlab_by_class.get(matlab_class, {})
        python_row = python_by_class.get(matlab_class, {})

        matlab_methods = set(str(method) for method in matlab_row.get("methods", []))
        excluded_methods = exclusions.get(matlab_class, set())
        considered_methods = sorted(method for method in matlab_methods if method not in excluded_methods)
        surface = _python_surface(python_row)

        stale_aliases = sorted(method for method in aliases if method not in matlab_methods)
        missing_methods: list[dict[str, str]] = []
        covered_methods: list[dict[str, str]] = []

        for method in considered_methods:
            target = str(aliases.get(method, method))
            if target in surface:
                covered_methods.append({"matlab_method": method, "python_member": target})
            else:
                missing_methods.append({"matlab_method": method, "python_member": target})

        missing_count = len(missing_methods)
        considered_count = len(considered_methods)
        covered_count = len(covered_methods)
        considered_total += considered_count
        missing_total += missing_count

        class_rows.append(
            {
                "matlab_class": matlab_class,
                "considered_method_count": considered_count,
                "covered_method_count": covered_count,
                "missing_method_count": missing_count,
                "coverage_ratio": float(covered_count / max(considered_count, 1)),
                "missing_methods": missing_methods,
                "stale_alias_methods": stale_aliases,
                "excluded_method_count": len(excluded_methods),
            }
        )

    missing_classes = sum(1 for row in class_rows if row["missing_method_count"] > 0)
    summary = {
        "total_classes": len(class_rows),
        "classes_with_missing_methods": missing_classes,
        "total_considered_methods": considered_total,
        "total_missing_methods": missing_total,
        "overall_coverage_ratio": float((considered_total - missing_total) / max(considered_total, 1)),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_mapping": str((repo_root / args.method_mapping).resolve()),
        "method_exclusions": str(exclusions_path),
        "matlab_inventory": str((repo_root / args.matlab_inventory).resolve()),
        "python_inventory": str((repo_root / args.python_inventory).resolve()),
        "summary": summary,
        "class_rows": class_rows,
    }

    out_path = (repo_root / args.report_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote strict method coverage report: {out_path}")
    print(
        "Strict method coverage summary: "
        f"classes_with_missing={summary['classes_with_missing_methods']} "
        f"total_missing_methods={summary['total_missing_methods']} "
        f"overall_coverage={summary['overall_coverage_ratio']:.4f}"
    )

    if args.fail_on_missing and missing_total > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
