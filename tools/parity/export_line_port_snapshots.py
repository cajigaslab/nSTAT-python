#!/usr/bin/env python3
"""Export MATLAB executable-line snapshots for notebook strict line-port anchors."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TOPICS = (
    "nSTATPaperExamples",
    "HippocampalPlaceCellExample",
    "publish_all_helpfiles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="nSTAT-python repository root.",
    )
    parser.add_argument(
        "--matlab-root",
        type=Path,
        required=True,
        help="MATLAB nSTAT repository root containing helpfiles/*.m.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=list(DEFAULT_TOPICS),
        help="MATLAB help topics to export.",
    )
    return parser.parse_args()


def _extract_exec_lines(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("%"):
            continue
        out.append(stripped)
    return out


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    matlab_root = args.matlab_root.resolve()
    help_root = matlab_root / "helpfiles"
    out_root = repo_root / "parity" / "line_port_snapshots"
    out_root.mkdir(parents=True, exist_ok=True)

    for topic in args.topics:
        src = help_root / f"{topic}.m"
        if not src.exists():
            raise FileNotFoundError(f"Missing MATLAB helpfile for topic '{topic}': {src}")
        lines = _extract_exec_lines(src)
        out_path = out_root / f"{topic}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Wrote {out_path} ({len(lines)} executable lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
