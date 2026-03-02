#!/usr/bin/env python3
"""Generate release-candidate notes from parity artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--commit", type=str, default="")
    parser.add_argument("--run-url", type=str, default="")
    parser.add_argument("--validation-pdf", type=str, default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gap-report",
        type=Path,
        default=Path("parity/parity_gap_report.json"),
    )
    parser.add_argument(
        "--equivalence-report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
    )
    parser.add_argument(
        "--numeric-drift-report",
        type=Path,
        default=Path("parity/numeric_drift_report.json"),
    )
    parser.add_argument(
        "--example-output-spec",
        type=Path,
        default=Path("parity/example_output_spec.yml"),
    )
    parser.add_argument(
        "--previous-tag",
        type=str,
        default="",
        help="Optional previous RC tag (for explicit RC-to-RC deltas).",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _read_json_from_tag(repo_root: Path, tag: str, relpath: Path) -> dict:
    if not tag:
        return {}
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{tag}:{relpath.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def _delta_line(label: str, current: int, previous: int) -> str:
    delta = current - previous
    sign = "+" if delta >= 0 else ""
    return f"- {label}: `{previous} -> {current}` (`{sign}{delta}`)"


def _latest_snapshot(parity_dir: Path) -> Path | None:
    candidates = sorted(parity_dir.glob("matlab_gold_snapshot_*.yml"))
    if not candidates:
        return None
    return candidates[-1]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    gap_report = _read_json(repo_root / args.gap_report)
    eq_report = _read_json(repo_root / args.equivalence_report)
    drift_report = _read_json(repo_root / args.numeric_drift_report)
    spec = _read_yaml(repo_root / args.example_output_spec)
    snapshot_path = _latest_snapshot(repo_root / "parity")
    snapshot = _read_yaml(snapshot_path) if snapshot_path is not None else {}
    previous_tag = args.previous_tag.strip()
    previous_eq = _read_json_from_tag(repo_root, previous_tag, args.equivalence_report)
    previous_drift = _read_json_from_tag(repo_root, previous_tag, args.numeric_drift_report)
    previous_gap = _read_json_from_tag(repo_root, previous_tag, args.gap_report)

    gap = gap_report.get("summary", {})
    method = eq_report.get("method_functional_audit", {}).get("summary", {})
    example = eq_report.get("example_line_alignment_audit", {}).get("summary", {})
    drift = drift_report.get("summary", {})
    out_of_scope_topics = spec.get("out_of_scope_topics", [])

    lines: list[str] = []
    lines.append(f"## nSTAT-python {args.tag}")
    lines.append("")
    lines.append(
        "Automated release-candidate notes generated from parity, functional, "
        "and numeric-drift artifacts."
    )
    lines.append("")
    lines.append("### Structural parity")
    lines.append(f"- High gaps: `{int(gap.get('high', 0))}`")
    lines.append(f"- Medium gaps: `{int(gap.get('medium', 0))}`")
    lines.append(f"- Low gaps: `{int(gap.get('low', 0))}`")
    lines.append(f"- Total gaps: `{int(gap.get('total', 0))}`")
    lines.append("")
    lines.append("### Functional parity")
    lines.append(f"- Total methods: `{int(method.get('total_methods', 0))}`")
    lines.append(
        f"- Contract-verified methods: `{int(method.get('contract_verified_methods', 0))}`"
    )
    lines.append(
        "- Contract-explicit verified methods: "
        f"`{int(method.get('contract_explicit_verified_methods', 0))}`"
    )
    lines.append(f"- Probe-verified methods: `{int(method.get('probe_verified_methods', 0))}`")
    lines.append(f"- Excluded methods: `{int(method.get('excluded_methods', 0))}`")
    lines.append(f"- Missing symbol methods: `{int(method.get('missing_symbol_methods', 0))}`")
    lines.append(
        f"- Unverified behavior methods: `{int(method.get('unverified_behavior_methods', 0))}`"
    )
    lines.append("")
    lines.append("### Example parity")
    lines.append(f"- Total topics: `{int(example.get('total_topics', 0))}`")
    lines.append(f"- Validated topics: `{int(example.get('validated_topics', 0))}`")
    lines.append(f"- MATLAB doc-only topics: `{int(example.get('matlab_doc_only_topics', 0))}`")
    lines.append(
        f"- Pending manual review topics: `{int(example.get('pending_manual_review_topics', 0))}`"
    )
    if out_of_scope_topics:
        lines.append("- Out-of-scope topics:")
        for topic in out_of_scope_topics:
            lines.append(f"  - `{topic}`")
    lines.append("")
    lines.append("### Numeric drift")
    lines.append(f"- Topics checked: `{int(drift.get('topics', 0))}`")
    lines.append(f"- Topics passed: `{int(drift.get('passed_topics', 0))}`")
    lines.append(f"- Topics failed: `{int(drift.get('failed_topics', 0))}`")
    lines.append(f"- Metrics checked: `{int(drift.get('checked_metrics', 0))}`")
    lines.append(f"- Metrics failed: `{int(drift.get('failed_metrics', 0))}`")
    lines.append("")

    if previous_tag:
        prev_gap = previous_gap.get("summary", {})
        prev_method = previous_eq.get("method_functional_audit", {}).get("summary", {})
        prev_example = previous_eq.get("example_line_alignment_audit", {}).get("summary", {})
        prev_drift = previous_drift.get("summary", {})
        lines.append(f"### RC delta vs `{previous_tag}`")
        lines.append(
            _delta_line(
                "Structural high gaps",
                int(gap.get("high", 0)),
                int(prev_gap.get("high", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Structural medium gaps",
                int(gap.get("medium", 0)),
                int(prev_gap.get("medium", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Validated example topics",
                int(example.get("validated_topics", 0)),
                int(prev_example.get("validated_topics", 0)),
            )
        )
        lines.append(
            _delta_line(
                "MATLAB doc-only topics",
                int(example.get("matlab_doc_only_topics", 0)),
                int(prev_example.get("matlab_doc_only_topics", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Contract-explicit verified methods",
                int(method.get("contract_explicit_verified_methods", 0)),
                int(prev_method.get("contract_explicit_verified_methods", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Probe-verified methods",
                int(method.get("probe_verified_methods", 0)),
                int(prev_method.get("probe_verified_methods", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Unverified behavior methods",
                int(method.get("unverified_behavior_methods", 0)),
                int(prev_method.get("unverified_behavior_methods", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Numeric topics checked",
                int(drift.get("topics", 0)),
                int(prev_drift.get("topics", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Numeric topics passed",
                int(drift.get("passed_topics", 0)),
                int(prev_drift.get("passed_topics", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Numeric topics failed",
                int(drift.get("failed_topics", 0)),
                int(prev_drift.get("failed_topics", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Numeric metrics checked",
                int(drift.get("checked_metrics", 0)),
                int(prev_drift.get("checked_metrics", 0)),
            )
        )
        lines.append(
            _delta_line(
                "Numeric metrics failed",
                int(drift.get("failed_metrics", 0)),
                int(prev_drift.get("failed_metrics", 0)),
            )
        )
        lines.append("")

    if snapshot:
        source = snapshot.get("source", {})
        mirror = snapshot.get("mirror", {})
        lines.append("### Frozen MATLAB snapshot")
        lines.append(
            f"- Snapshot id: `{snapshot.get('snapshot_id', '-')}` "
            f"(`{snapshot.get('captured_on', '-')}`)"
        )
        lines.append(
            f"- Source manifest: `{source.get('manifest_path', '-')}` "
            f"(sha256 `{source.get('manifest_sha256', '-')}`)"
        )
        lines.append(
            f"- Mirror manifest: `{mirror.get('manifest_path', '-')}` "
            f"(sha256 `{mirror.get('manifest_sha256', '-')}`)"
        )
        lines.append(f"- File count: `{mirror.get('file_count', '-')}`")
        lines.append(f"- Total size bytes: `{mirror.get('total_size_bytes', '-')}`")
        lines.append("")

    if args.validation_pdf:
        lines.append("### Validation asset")
        lines.append(f"- PDF: `{args.validation_pdf}`")
        lines.append("")

    if args.commit:
        lines.append("### Build metadata")
        lines.append(f"- Commit: `{args.commit}`")
        if args.run_url:
            lines.append(f"- Workflow run: {args.run_url}")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote RC release notes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
