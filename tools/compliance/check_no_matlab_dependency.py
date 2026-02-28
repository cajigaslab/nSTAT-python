#!/usr/bin/env python3
"""Verify repository source code has no MATLAB runtime dependency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PATTERNS = [
    re.compile(r"\bmatlab\.engine\b"),
    re.compile(r"\bimport\s+matlab\b"),
    re.compile(r"\bfrom\s+matlab\b"),
    re.compile(r"\bmfilename\("),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    offenders: list[tuple[str, int, str]] = []

    for path in args.root.joinpath("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                if pattern.search(line):
                    offenders.append((path.relative_to(args.root).as_posix(), line_no, line.strip()))

    if offenders:
        print("MATLAB dependency check FAILED.")
        for rel, line_no, line in offenders:
            print(f"  {rel}:{line_no}: {line}")
        return 1

    print("MATLAB dependency check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
