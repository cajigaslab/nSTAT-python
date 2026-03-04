#!/usr/bin/env python3
"""Page-by-page image-mode parity gate for MATLAB-vs-Python validation PDFs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # Optional dependency; workflow installs it.
    import fitz  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyMuPDF (fitz) is required: {exc}") from exc

try:  # Optional fallback handled below.
    from skimage.metrics import structural_similarity as skimage_ssim  # type: ignore
except Exception:  # pragma: no cover
    skimage_ssim = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-pdf", type=Path, required=True, help="Rendered Python validation PDF")
    parser.add_argument("--matlab-pdf", type=Path, required=True, help="Rendered MATLAB reference PDF")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/pdf/image_mode_parity"),
        help="Directory for parity artifacts",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI")
    parser.add_argument("--ssim-threshold", type=float, default=0.90, help="Minimum SSIM to pass")
    parser.add_argument(
        "--nrmse-threshold",
        type=float,
        default=0.20,
        help="Maximum normalized RMSE when SSIM backend is unavailable",
    )
    parser.add_argument(
        "--max-failing-pages",
        type=int,
        default=0,
        help="Allow up to this many failing pages before non-zero exit",
    )
    parser.add_argument(
        "--ignore-pages",
        type=str,
        default="",
        help="Comma-separated 1-based page numbers to ignore, e.g. '1,2,10'",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path (defaults to <out-dir>/summary.json)",
    )
    return parser.parse_args()


@dataclass
class PageParity:
    page: int
    ignored: bool
    metric: str
    score: float
    passed: bool
    python_shape: tuple[int, int]
    matlab_shape: tuple[int, int]
    diff_image: str | None


def _parse_ignore_pages(raw: str) -> set[int]:
    out: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.add(int(token))
    return out


def _render_pdf_grayscale(pdf_path: Path, dpi: int) -> list[np.ndarray]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    scale = float(dpi) / 72.0
    matrix = fitz.Matrix(scale, scale)
    doc = fitz.open(str(pdf_path))
    try:
        pages: list[np.ndarray] = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n >= 3:
                rgb = arr[:, :, :3].astype(np.float32)
                gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
            else:
                gray = arr[:, :, 0].astype(np.float32) / 255.0
            pages.append(np.clip(gray, 0.0, 1.0))
        return pages
    finally:
        doc.close()


def _resize_to_match(src: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if src.shape == shape:
        return src
    img = Image.fromarray(np.clip(src * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized = img.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    denom = max(float(np.max([a.max() - a.min(), b.max() - b.min()])), 1e-12)
    return rmse / denom


def _save_diff_image(py: np.ndarray, mat: np.ndarray, out_path: Path) -> None:
    py_u8 = np.clip(py * 255.0, 0.0, 255.0).astype(np.uint8)
    mat_u8 = np.clip(mat * 255.0, 0.0, 255.0).astype(np.uint8)
    diff = np.abs(py_u8.astype(np.int16) - mat_u8.astype(np.int16)).astype(np.uint8)

    py_rgb = np.stack([py_u8, py_u8, py_u8], axis=2)
    mat_rgb = np.stack([mat_u8, mat_u8, mat_u8], axis=2)
    diff_rgb = np.stack([diff, np.zeros_like(diff), np.zeros_like(diff)], axis=2)
    panel = np.concatenate([py_rgb, mat_rgb, diff_rgb], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel, mode="RGB").save(out_path)


def _environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "fitz": getattr(fitz, "__doc__", "").split()[1] if getattr(fitz, "__doc__", "") else "unknown",
    }
    try:
        import scipy  # type: ignore

        metadata["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        metadata["scipy"] = "unavailable"
    metadata["ssim_backend"] = "skimage" if skimage_ssim is not None else "nrmse"
    return metadata


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (args.summary_json.resolve() if args.summary_json else out_dir / "summary.json")

    ignore_pages = _parse_ignore_pages(args.ignore_pages)
    py_pages = _render_pdf_grayscale(args.python_pdf.resolve(), args.dpi)
    matlab_pages = _render_pdf_grayscale(args.matlab_pdf.resolve(), args.dpi)

    compare_pages = min(len(py_pages), len(matlab_pages))
    rows: list[PageParity] = []
    diff_dir = out_dir / "diff"

    for idx in range(compare_pages):
        page_num = idx + 1
        py = py_pages[idx]
        mat = _resize_to_match(matlab_pages[idx], py.shape)
        ignored = page_num in ignore_pages

        if skimage_ssim is not None:
            metric = "ssim"
            score = float(skimage_ssim(py, mat, data_range=1.0))
            passed = (score >= args.ssim_threshold) or ignored
        else:
            metric = "nrmse"
            score = float(_nrmse(py, mat))
            passed = (score <= args.nrmse_threshold) or ignored

        diff_path: Path | None = None
        if not passed and not ignored:
            diff_path = diff_dir / f"page_{page_num:03d}.png"
            _save_diff_image(py, mat, diff_path)

        rows.append(
            PageParity(
                page=page_num,
                ignored=ignored,
                metric=metric,
                score=score,
                passed=passed,
                python_shape=tuple(int(v) for v in py.shape),
                matlab_shape=tuple(int(v) for v in mat.shape),
                diff_image=(str(diff_path) if diff_path is not None else None),
            )
        )

    failed = [r for r in rows if not r.passed and not r.ignored]
    count_mismatch = len(py_pages) != len(matlab_pages)
    page_count_failure = 1 if count_mismatch else 0

    if skimage_ssim is not None:
        worst = sorted(rows, key=lambda r: r.score)[: min(10, len(rows))]
    else:
        worst = sorted(rows, key=lambda r: r.score, reverse=True)[: min(10, len(rows))]

    summary = {
        "schema_version": 1,
        "python_pdf": str(args.python_pdf.resolve()),
        "matlab_pdf": str(args.matlab_pdf.resolve()),
        "dpi": int(args.dpi),
        "thresholds": {
            "ssim_threshold": float(args.ssim_threshold),
            "nrmse_threshold": float(args.nrmse_threshold),
            "max_failing_pages": int(args.max_failing_pages),
        },
        "environment": _environment_metadata(),
        "page_counts": {
            "python": len(py_pages),
            "matlab": len(matlab_pages),
            "compared": compare_pages,
            "mismatch": bool(count_mismatch),
        },
        "failed_page_count": len(failed),
        "worst_pages": [
            {"page": r.page, "metric": r.metric, "score": r.score, "passed": r.passed, "ignored": r.ignored}
            for r in worst
        ],
        "pages": [
            {
                "page": r.page,
                "ignored": r.ignored,
                "metric": r.metric,
                "score": r.score,
                "passed": r.passed,
                "python_shape": list(r.python_shape),
                "matlab_shape": list(r.matlab_shape),
                "diff_image": r.diff_image,
            }
            for r in rows
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote image-mode parity summary: {summary_path}")
    print(f"Compared pages: {compare_pages} (python={len(py_pages)} matlab={len(matlab_pages)})")
    print(f"Failed pages: {len(failed)}")
    if count_mismatch:
        print("Page-count mismatch detected between Python and MATLAB PDFs")

    if page_count_failure > 0:
        return 1
    return 0 if len(failed) <= args.max_failing_pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
