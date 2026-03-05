#!/usr/bin/env python3
"""Fail fast when expected CI artifacts are missing before upload."""

from __future__ import annotations

import argparse
from pathlib import Path


def _matches(pattern: str) -> list[Path]:
    return sorted(Path().glob(pattern))


def _must_have(pattern: str) -> tuple[str, list[Path]]:
    hits = _matches(pattern)
    if not hits:
        raise FileNotFoundError(f"missing required artifacts for pattern: {pattern}")
    return pattern, hits


def _maybe_have(pattern: str) -> tuple[str, list[Path]]:
    return pattern, _matches(pattern)


def _validate(kind: str) -> list[tuple[str, list[Path]]]:
    rows: list[tuple[str, list[Path]]] = []
    if kind == "validation":
        rows.append(_must_have("output/pdf/*.pdf"))
        rows.append(_must_have("output/pdf/*.json"))
        rows.append(_must_have("output/pdf/*.csv"))
        rows.append(_maybe_have("output/pdf/validation_gate_mode_latest.json"))
        rows.append(_maybe_have("output/pdf/validation_gate_mode_latest.csv"))
        return rows
    if kind == "image":
        rows.append(_must_have("output/pdf/image_mode_parity/summary.json"))
        rows.append(_must_have("output/pdf/image_mode_parity/**/*"))
        rows.append(_must_have("output/pdf/*.json"))
        return rows
    if kind == "performance":
        rows.append(_must_have("output/performance/performance_parity_report.json"))
        rows.append(_must_have("output/performance/performance_parity_report.csv"))
        rows.append(_must_have("output/performance/*.json"))
        rows.append(_must_have("output/performance/*.csv"))
        return rows
    raise ValueError(f"Unsupported kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["validation", "image", "performance"],
        help="Artifact group to validate before upload.",
    )
    args = parser.parse_args()

    rows = _validate(args.kind)
    print(f"Artifact check [{args.kind}] passed.")
    for pattern, hits in rows:
        print(f"- {pattern}: {len(hits)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
