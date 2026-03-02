#!/usr/bin/env python3
"""CI gate for validation report visual quality.

Fails when:
- any topic has fewer than the configured minimum unique notebook figures
- rendered PDF pages contain duplicates (same visual hash)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-pdf",
        required=True,
        help="PDF file path or glob pattern (e.g., output/pdf/*.pdf).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("tmp/pdfs/validation_report/notebook_images"),
        help="Root directory containing per-topic notebook images.",
    )
    parser.add_argument(
        "--min-unique-images-per-topic",
        type=int,
        default=1,
        help="Minimum required unique PNG images per topic.",
    )
    parser.add_argument(
        "--max-duplicate-pdf-pages",
        type=int,
        default=0,
        help="Maximum allowed duplicate rendered PDF pages.",
    )
    return parser.parse_args()


def _resolve_pdf(path_or_glob: str) -> Path:
    cand = sorted(Path().glob(path_or_glob))
    if not cand:
        p = Path(path_or_glob)
        if p.exists():
            return p
        raise FileNotFoundError(f"No PDF matches: {path_or_glob}")
    return max(cand, key=lambda p: p.stat().st_mtime)


def _image_fingerprint(path: Path) -> str:
    arr = np.asarray(
        Image.open(path).convert("L").resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _check_topic_images(images_root: Path, min_unique: int) -> tuple[list[str], dict[str, tuple[int, int]]]:
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    failures: list[str] = []
    stats: dict[str, tuple[int, int]] = {}
    topic_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    if not topic_dirs:
        failures.append("no topic image directories found")
        return failures, stats

    for topic_dir in topic_dirs:
        pngs = sorted(topic_dir.glob("*.png"))
        hashes = [_image_fingerprint(p) for p in pngs]
        unique = len(set(hashes))
        stats[topic_dir.name] = (len(pngs), unique)
        if unique < min_unique:
            failures.append(
                f"topic={topic_dir.name}: unique_images={unique} < min_required={min_unique}"
            )
    return failures, stats


def _check_pdf_page_duplicates(pdf_path: Path, max_dupes: int) -> tuple[list[str], int, int]:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required for PDF visual gate but was not found in PATH")

    with tempfile.TemporaryDirectory(prefix="nstat_pdf_gate_") as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", str(pdf_path), str(out_prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
        page_pngs = sorted(Path(tmp).glob("page-*.png"))
        if not page_pngs:
            return ["pdf rendered to zero pages"], 0, 0

        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in page_pngs]
        total = len(hashes)
        unique = len(set(hashes))
        dupes = total - unique
        failures = []
        if dupes > max_dupes:
            failures.append(f"duplicate_pdf_pages={dupes} > max_allowed={max_dupes}")
        return failures, total, dupes


def main() -> int:
    args = parse_args()
    pdf_path = _resolve_pdf(args.report_pdf)

    image_failures, topic_stats = _check_topic_images(
        images_root=args.images_root,
        min_unique=args.min_unique_images_per_topic,
    )
    pdf_failures, total_pages, duplicate_pages = _check_pdf_page_duplicates(
        pdf_path=pdf_path,
        max_dupes=args.max_duplicate_pdf_pages,
    )

    print(f"Validation PDF gate: {pdf_path}")
    print(f"Topic coverage: {len(topic_stats)} topics")
    for topic, (total, unique) in sorted(topic_stats.items()):
        print(f"  - {topic}: total_images={total} unique_images={unique}")
    print(f"PDF pages: total={total_pages} duplicate_pages={duplicate_pages}")

    failures = image_failures + pdf_failures
    if failures:
        print("Visual gate failures:")
        for row in failures:
            print(f"  - {row}")
        return 1

    print("Visual validation gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
