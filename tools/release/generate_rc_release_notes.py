#!/usr/bin/env python3
"""Generate release-candidate notes from parity artifacts."""

from __future__ import annotations

import argparse
import json
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
