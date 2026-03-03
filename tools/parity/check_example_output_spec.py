#!/usr/bin/env python3
"""Validate notebook/example output readiness against a policy spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Equivalence audit report path.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("parity/example_output_spec.yml"),
        help="Example output policy spec path.",
    )
    return parser.parse_args()


def _topic_cfg(spec: dict, topic: str) -> dict:
    merged = dict(spec.get("defaults", {}))
    merged.update(spec.get("topics", {}).get(topic, {}))
    return merged


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    rows = report["example_line_alignment_audit"]["topic_rows"]
    out_of_scope_topics = set(spec.get("out_of_scope_topics", []))

    failures: list[str] = []

    for row in rows:
        topic = str(row["topic"])
        cfg = _topic_cfg(spec, topic)
        is_out_of_scope = topic in out_of_scope_topics

        if is_out_of_scope:
            allowed = set(cfg.get("out_of_scope_allowed_alignment_statuses", []))
            if not allowed:
                allowed = set(cfg.get("allowed_alignment_statuses", []))
        else:
            allowed = set(cfg.get("allowed_alignment_statuses", []))

        status = str(row["alignment_status"])
        if allowed and status not in allowed:
            scope_label = "out-of-scope" if is_out_of_scope else "in-scope"
            failures.append(
                f"{topic}: {scope_label} alignment_status '{status}' not in allowed set {sorted(allowed)}"
            )

        min_code_lines = int(cfg.get("min_python_code_lines", 0))
        py_lines = int(row.get("python_code_lines", 0))
        if py_lines < min_code_lines:
            failures.append(f"{topic}: python_code_lines={py_lines} below required {min_code_lines}")

        min_code_cells = int(cfg.get("min_python_code_cells", 0))
        py_cells = len(row.get("python_code_cells", []))
        if py_cells < min_code_cells:
            failures.append(f"{topic}: python_code_cells={py_cells} below required {min_code_cells}")

        if bool(cfg.get("require_topic_checkpoint", False)):
            has_checkpoint = bool(row.get("has_topic_checkpoint", False))
            if not has_checkpoint:
                failures.append(f"{topic}: missing topic checkpoint cell marker")

        min_assertions = int(cfg.get("min_assertion_count", 0))
        if min_assertions > 0:
            assertion_count = int(row.get("assertion_count", 0))
            if assertion_count < min_assertions:
                failures.append(
                    f"{topic}: assertion_count={assertion_count} below required {min_assertions}"
                )

        if bool(cfg.get("require_plot_call", False)):
            has_plot = bool(row.get("has_plot_call", False))
            if not has_plot:
                failures.append(f"{topic}: no plotting call detected in notebook code")

        min_mat_refs = int(cfg.get("min_matlab_reference_images", 0))
        mat_refs = int(row.get("matlab_reference_image_count", 0))
        if mat_refs < min_mat_refs:
            failures.append(f"{topic}: matlab_reference_image_count={mat_refs} below required {min_mat_refs}")

        if bool(cfg.get("enforce_validation_images", False)):
            min_py_imgs = int(cfg.get("min_python_validation_images", 0))
            py_imgs = int(row.get("python_validation_image_count", 0))
            if py_imgs < min_py_imgs:
                failures.append(
                    f"{topic}: python_validation_image_count={py_imgs} below required {min_py_imgs}"
                )

        if bool(cfg.get("require_line_port_audit", False)):
            strict_status = str(row.get("strict_line_status", ""))
            allowed_strict = set(cfg.get("allowed_strict_line_statuses", []))
            if allowed_strict and strict_status not in allowed_strict:
                failures.append(
                    f"{topic}: strict_line_status '{strict_status}' not in allowed set {sorted(allowed_strict)}"
                )

            min_coverage = float(cfg.get("min_line_port_coverage", 0.0))
            coverage = float(row.get("line_port_coverage", 0.0))
            if coverage < min_coverage:
                failures.append(
                    f"{topic}: line_port_coverage={coverage:.4f} below required {min_coverage:.4f}"
                )

            min_func_recall = float(cfg.get("min_line_port_function_recall", 0.0))
            func_recall = float(row.get("line_port_function_recall", 0.0))
            if func_recall < min_func_recall:
                failures.append(
                    f"{topic}: line_port_function_recall={func_recall:.4f} below required {min_func_recall:.4f}"
                )

    if failures:
        print("Example output spec check FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Example output spec check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
