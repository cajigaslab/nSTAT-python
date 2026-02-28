#!/usr/bin/env python3
"""Generate Tier-1 parity backlog markdown from parity gap report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TIER1_CLASSES = [
    "DecodingAlgorithms",
    "Analysis",
    "Trial",
    "CovColl",
    "nstColl",
    "FitResult",
    "FitResSummary",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity/parity_gap_report.json"),
        help="Parity gap report JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parity/TIER1_PORT_BACKLOG.md"),
        help="Output markdown backlog path.",
    )
    return parser.parse_args()



def _priority(missing_count: int) -> str:
    if missing_count >= 50:
        return "P0"
    if missing_count >= 25:
        return "P1"
    return "P2"



def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))

    coverage_rows = {row["matlab_class"]: row for row in report["class_coverage"]}
    issues_rows = {
        issue["details"].get("matlab_class"): issue
        for issue in report["issues"]
        if issue.get("check") == "missing_method_mappings"
    }

    lines: list[str] = []
    lines.append("# Tier-1 Port Backlog")
    lines.append("")
    lines.append("Generated from `parity/parity_gap_report.json`.")
    lines.append("")
    lines.append("| Priority | MATLAB Class | Missing Methods | Coverage | Next Implementation Focus |")
    lines.append("|---|---:|---:|---:|---|")

    for klass in TIER1_CLASSES:
        cov = coverage_rows.get(klass, {})
        missing = int(cov.get("missing_method_count", 0))
        ratio = float(cov.get("coverage_ratio", 0.0))
        issue = issues_rows.get(klass, {})
        method_preview = issue.get("details", {}).get("missing_methods", [])[:6]
        preview = ", ".join(method_preview) if method_preview else "n/a"
        lines.append(
            f"| {_priority(missing)} | `{klass}` | {missing} | {ratio:.3f} | {preview} |"
        )

    lines.append("")
    lines.append("## Implementation Order")
    lines.append("1. `DecodingAlgorithms` and `Analysis` (numerical behavior parity)")
    lines.append("2. `Trial` / `CovColl` / `nstColl` (data plumbing parity)")
    lines.append("3. `FitResult` / `FitResSummary` (model diagnostics parity)")
    lines.append("")
    lines.append("## Acceptance for Each Tier-1 Class")
    lines.append("- Implement method aliases in `nstat.compat.matlab`.")
    lines.append("- Add numerical fixture checks when outputs are deterministic.")
    lines.append("- Add unit tests for method signatures and key behavior.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote backlog: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
