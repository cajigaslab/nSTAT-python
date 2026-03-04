#!/usr/bin/env python3
"""Build paired MATLAB/Python image-sequence PDFs for page-by-page parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True, help="Validation summary JSON from generate_validation_pdf.py")
    parser.add_argument(
        "--python-out",
        type=Path,
        default=Path("output/pdf/image_mode_parity/python_pages.pdf"),
        help="Output PDF containing Python images",
    )
    parser.add_argument(
        "--matlab-out",
        type=Path,
        default=Path("output/pdf/image_mode_parity/matlab_pages.pdf"),
        help="Output PDF containing MATLAB images",
    )
    parser.add_argument(
        "--pairs-json",
        type=Path,
        default=Path("output/pdf/image_mode_parity/pairs.json"),
        help="Output JSON containing selected per-topic image pairs",
    )
    return parser.parse_args()


def _resolve_img(path_str: str) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.exists() else None


def _select_pair(row: dict) -> tuple[Path | None, Path | None]:
    py = _resolve_img(str(row.get("matched_python_image") or ""))
    mat = _resolve_img(str(row.get("matched_matlab_image") or ""))

    if py is None:
        py_list = row.get("python_images") or []
        if py_list:
            py = _resolve_img(str(py_list[0]))

    if mat is None:
        mat_list = row.get("matlab_reference_images") or []
        if mat_list:
            mat = _resolve_img(str(mat_list[0]))

    return py, mat


def _draw_page(pdf: canvas.Canvas, *, topic: str, image_path: Path | None, label: str) -> None:
    w, h = letter
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(36, h - 44, f"{label}: {topic}")

    if image_path is None:
        pdf.setFont("Helvetica", 10)
        pdf.drawString(36, h - 72, "Missing image")
        pdf.showPage()
        return

    pdf.setFont("Helvetica", 9)
    pdf.drawString(36, h - 62, str(image_path))

    with Image.open(image_path) as img:
        iw, ih = img.size
    max_w = w - 72
    max_h = h - 120
    scale = min(max_w / iw, max_h / ih)
    draw_w = iw * scale
    draw_h = ih * scale
    x = (w - draw_w) / 2.0
    y = (h - 90 - draw_h) / 2.0
    pdf.drawImage(str(image_path), x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
    pdf.showPage()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.report_json.read_text(encoding="utf-8"))
    rows = payload.get("notebooks", [])

    pairs: list[dict] = []
    for row in rows:
        topic = str(row.get("topic", ""))
        py, mat = _select_pair(row)
        pairs.append(
            {
                "topic": topic,
                "python_image": str(py) if py is not None else "",
                "matlab_image": str(mat) if mat is not None else "",
            }
        )

    args.python_out.parent.mkdir(parents=True, exist_ok=True)
    args.matlab_out.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_json.parent.mkdir(parents=True, exist_ok=True)

    pdf_py = canvas.Canvas(str(args.python_out), pagesize=letter)
    pdf_mat = canvas.Canvas(str(args.matlab_out), pagesize=letter)

    for pair in pairs:
        topic = pair["topic"]
        py = Path(pair["python_image"]) if pair["python_image"] else None
        mat = Path(pair["matlab_image"]) if pair["matlab_image"] else None
        _draw_page(pdf_py, topic=topic, image_path=py, label="Python")
        _draw_page(pdf_mat, topic=topic, image_path=mat, label="MATLAB")

    pdf_py.save()
    pdf_mat.save()
    args.pairs_json.write_text(json.dumps({"schema_version": 1, "pairs": pairs}, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote Python PDF: {args.python_out}")
    print(f"Wrote MATLAB PDF: {args.matlab_out}")
    print(f"Wrote pairs JSON: {args.pairs_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
