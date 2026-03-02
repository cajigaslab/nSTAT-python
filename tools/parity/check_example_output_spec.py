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

    failures: list[str] = []

    for row in rows:
        topic = str(row["topic"])
        cfg = _topic_cfg(spec, topic)

        allowed = set(cfg.get("allowed_alignment_statuses", []))
        status = str(row["alignment_status"])
        if allowed and status not in allowed:
            failures.append(f"{topic}: alignment_status '{status}' not in allowed set {sorted(allowed)}")

        min_code_lines = int(cfg.get("min_python_code_lines", 0))
        py_lines = int(row.get("python_code_lines", 0))
        if py_lines < min_code_lines:
            failures.append(f"{topic}: python_code_lines={py_lines} below required {min_code_lines}")

        min_code_cells = int(cfg.get("min_python_code_cells", 0))
        py_cells = len(row.get("python_code_cells", []))
        if py_cells < min_code_cells:
            failures.append(f"{topic}: python_code_cells={py_cells} below required {min_code_cells}")

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

    if failures:
        print("Example output spec check FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Example output spec check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
