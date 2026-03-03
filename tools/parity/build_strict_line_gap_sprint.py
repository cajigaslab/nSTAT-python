#!/usr/bin/env python3
"""Build a strict-line parity sprint backlog from equivalence audit output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Equivalence audit report JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parity/strict_line_gap_sprint.md"),
        help="Output markdown backlog path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of strict line-gap topics to include.",
    )
    return parser.parse_args()


def _f(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def main() -> int:
    args = parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    audit = payload.get("example_line_alignment_audit", {})
    summary = audit.get("summary", {})
    rows = list(audit.get("topic_rows", []))

    gaps = [row for row in rows if row.get("strict_line_status") == "line_port_gap"]
    gaps.sort(
        key=lambda row: (
            float(row.get("line_port_coverage", 0.0)),
            float(row.get("line_port_function_recall", 0.0)),
            -float(row.get("matlab_code_lines", 0.0)),
        )
    )
    selected = gaps[: max(0, args.top_n)]

    lines: list[str] = []
    lines.append("# Strict Line-Port Gap Sprint")
    lines.append("")
    lines.append(f"- Source report: `{args.report}`")
    lines.append(f"- Total topics: `{summary.get('total_topics', 0)}`")
    lines.append(
        "- Strict summary: "
        f"verified={summary.get('strict_line_verified_topics', 0)}, "
        f"partial={summary.get('strict_line_partial_topics', 0)}, "
        f"gap={summary.get('strict_line_gap_topics', len(gaps))}"
    )
    lines.append("")
    lines.append("## Priority Queue")
    lines.append(
        "| Priority | Topic | Coverage | Function recall | Code-line ratio (Py/MATLAB) | MATLAB lines | Python lines |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(selected, start=1):
        lines.append(
            "| "
            f"{i} | {row.get('topic', '-')}"
            f" | {_f(row.get('line_port_coverage'))}"
            f" | {_f(row.get('line_port_function_recall'))}"
            f" | {_f(row.get('python_to_matlab_line_ratio'))}"
            f" | {int(row.get('matlab_code_lines', 0))}"
            f" | {int(row.get('python_code_lines', 0))} |"
        )
    lines.append("")
    lines.append("## Execution Checklist")
    lines.append("- Export executable-line snapshots for each gap topic.")
    lines.append("- Regenerate notebooks with snapshot anchors.")
    lines.append("- Re-run `tools/parity/sync_parity_artifacts.py`.")
    lines.append("- Target strict status: `line_port_partial` or `line_port_verified`.")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote strict-line sprint backlog: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
