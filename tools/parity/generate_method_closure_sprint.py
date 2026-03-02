#!/usr/bin/env python3
"""Generate a targeted method-closure sprint backlog from parity audit output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Functional equivalence audit report JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parity/method_closure_sprint.md"),
        help="Markdown backlog output path.",
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=8,
        help="Number of highest probe-verified classes to prioritize.",
    )
    parser.add_argument(
        "--max-methods-per-class",
        type=int,
        default=20,
        help="Maximum listed methods per class in the backlog section.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))

    method_audit = payload["method_functional_audit"]
    summary = method_audit["summary"]
    class_rows = method_audit["class_summary"]
    method_rows = method_audit["method_rows"]

    probe_only_by_class: dict[str, list[str]] = defaultdict(list)
    excluded_by_class: dict[str, list[str]] = defaultdict(list)

    for row in method_rows:
        klass = str(row["matlab_class"])
        method_name = str(row["matlab_method"])
        status = str(row.get("functional_status", ""))
        has_contract = bool(row.get("has_behavior_contract", False))
        if status == "probe_verified" and not has_contract:
            probe_only_by_class[klass].append(method_name)
        if bool(row.get("excluded_method", False)):
            excluded_by_class[klass].append(method_name)

    priority_rows = sorted(
        class_rows,
        key=lambda row: (
            int(row.get("probe_verified_count", 0)),
            int(row.get("contract_verified_count", 0)),
        ),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("# Method Closure Sprint Backlog")
    lines.append("")
    lines.append(
        "This sprint backlog targets methods that are probe-verified but not yet "
        "explicitly covered by behavior contracts."
    )
    lines.append("")
    lines.append("## Functional Summary")
    lines.append(f"- Total methods: `{int(summary.get('total_methods', 0))}`")
    lines.append(
        f"- Contract-explicit verified methods: `{int(summary.get('contract_explicit_verified_methods', 0))}`"
    )
    lines.append(f"- Probe-verified methods: `{int(summary.get('probe_verified_methods', 0))}`")
    lines.append(
        f"- Eligible verified ratio: `{float(summary.get('eligible_verified_ratio', 0.0)):.3f}`"
    )
    lines.append(f"- Excluded methods: `{int(summary.get('excluded_methods', 0))}`")
    lines.append("")
    lines.append("## Priority Class Queue")
    lines.append("| Class | Probe-verified | Contract-verified | Probe-only methods |")
    lines.append("|---|---:|---:|---:|")
    for row in priority_rows[: args.top_classes]:
        klass = str(row["matlab_class"])
        lines.append(
            f"| {klass} | {int(row.get('probe_verified_count', 0))} | "
            f"{int(row.get('contract_verified_count', 0))} | "
            f"{len(probe_only_by_class.get(klass, []))} |"
        )
    lines.append("")

    lines.append("## Sprint Work Packages")
    lines.append("")
    for row in priority_rows[: args.top_classes]:
        klass = str(row["matlab_class"])
        methods = sorted(probe_only_by_class.get(klass, []))
        if not methods:
            continue
        lines.append(f"### {klass}")
        lines.append(
            "- Goal: Convert probe-only functional verification to explicit behavior contracts."
        )
        lines.append("- Candidate methods:")
        for method_name in methods[: args.max_methods_per_class]:
            lines.append(f"  - `{method_name}`")
        extra = len(methods) - args.max_methods_per_class
        if extra > 0:
            lines.append(f"  - `... (+{extra} additional methods)`")
        lines.append("")

    lines.append("## Excluded MATLAB Stub Methods")
    if not excluded_by_class:
        lines.append("- None")
    else:
        for klass in sorted(excluded_by_class):
            lines.append(f"- `{klass}`")
            for method_name in sorted(excluded_by_class[klass]):
                lines.append(f"  - `{method_name}`")
    lines.append("")

    lines.append("## Exit Criteria")
    lines.append("- Each listed method has an explicit behavior contract in parity audit generation.")
    lines.append("- New/updated contract tests are added and pass in CI.")
    lines.append(
        "- Functional parity summary increases `contract_explicit_verified_methods` and preserves gate pass."
    )
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote method closure sprint backlog: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
