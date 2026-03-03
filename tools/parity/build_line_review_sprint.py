#!/usr/bin/env python3
"""Build a prioritized line-by-line parity sprint backlog from review JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity/line_by_line_review_report.json"),
        help="Path to line-by-line review JSON report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parity/line_review_sprint.md"),
        help="Path to markdown backlog output.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of highest-priority needs_review topics to include.",
    )
    return parser.parse_args()


def _f(v: object | None) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def main() -> int:
    args = parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    rows = list(payload.get("topic_rows", []))

    needs_review = [row for row in rows if row.get("line_review_status") == "needs_review"]
    needs_review.sort(
        key=lambda row: (
            float(row.get("line_alignment_ratio") or 0.0),
            -int(row.get("missing_matlab_step_count") or 0),
        )
    )
    top_rows = needs_review[: max(0, args.top_n)]

    lines: list[str] = []
    lines.append("# Line Review Sprint Backlog")
    lines.append("")
    lines.append(f"- Source report: `{args.report}`")
    lines.append(f"- Generated at: `{summary.get('generated_at_utc', '-')}`")
    lines.append(f"- Total topics: `{summary.get('total_topics', 0)}`")
    lines.append(f"- Needs review: `{summary.get('needs_review_topics', len(needs_review))}`")
    lines.append(f"- Average line alignment ratio: `{_f(summary.get('average_line_alignment_ratio'))}`")
    lines.append("")
    lines.append("## Priority Queue")
    lines.append(
        "| Priority | Topic | Status | Line ratio | Step recall | Step precision | Missing MATLAB steps |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for i, row in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{i} | {row.get('topic', '-')}"
            f" | {row.get('line_review_status', '-')}"
            f" | {_f(row.get('line_alignment_ratio'))}"
            f" | {_f(row.get('matlab_step_recall'))}"
            f" | {_f(row.get('python_step_precision'))}"
            f" | {int(row.get('missing_matlab_step_count') or 0)} |"
        )

    lines.append("")
    lines.append("## Execution Notes")
    lines.append("- Address topics in queue order unless a dependency forces reordering.")
    lines.append(
        "- For each topic, update notebook logic first, then rerun `review_line_by_line_equivalence.py`."
    )
    lines.append("- Keep MATLAB/Python operation ordering aligned before adjusting numeric thresholds.")
    lines.append(
        "- After each topic fix, regenerate and commit: `parity/line_by_line_review_report.json` and this backlog."
    )
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote sprint backlog: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
