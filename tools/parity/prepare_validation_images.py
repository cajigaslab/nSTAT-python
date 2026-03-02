#!/usr/bin/env python3
"""Prepare deterministic notebook validation images for parity checks.

This utility verifies that every manifest topic has at least one tracked
baseline image and then syncs those images into the runtime tmp directory
consumed by equivalence-audit tooling.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tools/notebooks/notebook_manifest.yml"),
        help="Notebook manifest with topic list.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("baseline/validation/notebook_images"),
        help="Tracked baseline validation image root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("tmp/pdfs/validation_report/notebook_images"),
        help="Output directory used by parity audit tooling.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="Minimum number of PNGs required per topic in baseline.",
    )
    return parser.parse_args()


def _load_topics(manifest_path: Path) -> list[str]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return [str(row["topic"]) for row in payload.get("notebooks", [])]


def main() -> int:
    args = parse_args()
    topics = _load_topics(args.manifest)
    if not topics:
        raise RuntimeError("no topics found in notebook manifest")

    missing: list[str] = []
    for topic in topics:
        topic_dir = args.baseline_root / topic
        pngs = sorted(topic_dir.glob("*.png")) if topic_dir.exists() else []
        if len(pngs) < args.min_images:
            missing.append(topic)

    if missing:
        print("Validation image baseline check FAILED")
        for topic in missing:
            print(f"  - missing or insufficient baseline images: {topic}")
        return 1

    args.out_root.parent.mkdir(parents=True, exist_ok=True)
    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    shutil.copytree(args.baseline_root, args.out_root)

    copied_count = len(list(args.out_root.glob("*/*.png")))
    print("Validation image baseline check passed.")
    print(f"Copied baseline images to {args.out_root} ({copied_count} png files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
