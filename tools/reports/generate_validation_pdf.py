#!/usr/bin/env python3
"""Generate a complete visual PDF validation report for nSTAT-python examples."""

from __future__ import annotations

import argparse
import base64
import functools
import json
import os
import re
import subprocess
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nbformat
import numpy as np
import yaml
from nbclient import NotebookClient
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float
    stdout_tail: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(slots=True)
class NotebookTarget:
    topic: str
    file: Path
    run_group: str


@dataclass(slots=True)
class NotebookReport:
    topic: str
    file: Path
    run_group: str
    executed: bool
    duration_s: float
    image_paths: list[Path]
    image_count: int
    text_snippet: str
    error: str
    matlab_ref_images: list[Path]
    similarity_score: float | None
    parity_pass: bool | None
    alignment_status: str | None
    matched_python_image: Path | None
    matched_matlab_image: Path | None
    parity_metrics: dict[str, object] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to nSTAT-python repository root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "tools" / "notebooks" / "notebook_manifest.yml",
        help="Notebook manifest path.",
    )
    parser.add_argument(
        "--matlab-help-root",
        type=Path,
        default=None,
        help="Path to MATLAB nSTAT helpfiles folder (for reference parity images).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "pdf",
        help="Directory for final PDF.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "pdfs" / "validation_report",
        help="Directory for intermediate images.",
    )
    parser.add_argument(
        "--notebook-group",
        choices=["smoke", "full", "all"],
        default="full",
        help="Notebook group to include in the report.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-cell timeout in seconds when executing notebooks.",
    )
    parser.add_argument(
        "--parity-threshold",
        type=float,
        default=0.80,
        help="Minimum image similarity score in [0,1] for Python-vs-MATLAB pass.",
    )
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help="Skip MATLAB-reference image parity scoring.",
    )
    parser.add_argument(
        "--parity-mode",
        choices=["gate", "image"],
        default="gate",
        help=(
            "Parity pass/fail mode: "
            "'gate' follows parity/function_example_alignment_report.json statuses; "
            "'image' uses Python-vs-MATLAB image similarity threshold."
        ),
    )
    parser.add_argument(
        "--equivalence-report",
        type=Path,
        default=REPO_ROOT / "parity" / "function_example_alignment_report.json",
        help="Equivalence audit report JSON used when --parity-mode=gate.",
    )
    parser.add_argument(
        "--example-output-spec",
        type=Path,
        default=REPO_ROOT / "parity" / "example_output_spec.yml",
        help="Example output policy spec used to resolve allowed alignment statuses.",
    )
    parser.add_argument(
        "--skip-command-tests",
        action="store_true",
        help="Skip command-driven checks and only render notebook validation pages.",
    )
    return parser.parse_args()


def run_command(name: str, cmd: list[str], cwd: Path) -> CommandResult:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    raw_lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    filtered: list[str] = []
    skip_tokens = (
        "Debugger warning:",
        "PYDEVD_DISABLE_FILE_VALIDATION",
        "-Xfrozen_modules=off",
    )
    for line in raw_lines:
        stripped = line.strip()
        if any(token in stripped for token in skip_tokens):
            continue
        if stripped.startswith("0.00s -"):
            continue
        filtered.append(stripped)
    tail = filtered[-20:]
    return CommandResult(
        name=name,
        command=cmd,
        returncode=proc.returncode,
        duration_s=elapsed,
        stdout_tail="\n".join(tail),
    )


def load_targets(manifest_path: Path, repo_root: Path, group: str) -> list[NotebookTarget]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    all_targets: list[NotebookTarget] = []
    for row in payload.get("notebooks", []):
        all_targets.append(
            NotebookTarget(
                topic=str(row["topic"]),
                file=repo_root / str(row["file"]),
                run_group=str(row["run_group"]),
            )
        )
    if group in {"full", "all"}:
        return all_targets
    return [target for target in all_targets if target.run_group == "smoke"]


def load_parity_gate_status(
    equivalence_report: Path,
    example_output_spec: Path,
) -> dict[str, tuple[str, bool]]:
    if not equivalence_report.exists() or not example_output_spec.exists():
        return {}

    report = json.loads(equivalence_report.read_text(encoding="utf-8"))
    spec = yaml.safe_load(example_output_spec.read_text(encoding="utf-8")) or {}

    defaults = spec.get("defaults", {})
    per_topic = spec.get("topics", {})
    rows = report.get("example_line_alignment_audit", {}).get("topic_rows", [])

    out: dict[str, tuple[str, bool]] = {}
    for row in rows:
        topic = str(row.get("topic", ""))
        if not topic:
            continue
        status = str(row.get("alignment_status", ""))
        cfg = dict(defaults)
        cfg.update(per_topic.get(topic, {}))
        allowed = set(cfg.get("allowed_alignment_statuses", []))
        allowed_ok = status in allowed if allowed else True
        out[topic] = (status, allowed_ok)
    return out


def load_parity_topic_metrics(equivalence_report: Path) -> dict[str, dict[str, object]]:
    """Load per-topic parity metrics used in the PDF comparison table."""

    if not equivalence_report.exists():
        return {}
    report = json.loads(equivalence_report.read_text(encoding="utf-8"))
    rows = report.get("example_line_alignment_audit", {}).get("topic_rows", [])
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        topic = str(row.get("topic", ""))
        if not topic:
            continue
        out[topic] = {
            "matlab_code_lines": row.get("matlab_code_lines"),
            "python_code_lines": row.get("python_code_lines"),
            "python_to_matlab_line_ratio": row.get("python_to_matlab_line_ratio"),
            "matlab_reference_image_count": row.get("matlab_reference_image_count"),
            "python_validation_image_count": row.get("python_validation_image_count"),
            "assertion_count": row.get("assertion_count"),
            "has_plot_call": row.get("has_plot_call"),
            "has_topic_checkpoint": row.get("has_topic_checkpoint"),
        }
    return out


def _short_text(output_text: str, max_chars: int = 280) -> str:
    clean = " ".join(output_text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def resolve_matlab_help_root(repo_root: Path, provided: Path | None) -> Path | None:
    candidates: list[Path] = []
    if provided is not None:
        candidates.append(provided)

    env_help = os.environ.get("NSTAT_MATLAB_HELP_ROOT")
    if env_help:
        candidates.append(Path(env_help))

    env_root = os.environ.get("NSTAT_MATLAB_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "helpfiles")

    candidates.append(
        Path.home()
        / "Library"
        / "CloudStorage"
        / "Dropbox"
        / "Research"
        / "Matlab"
        / "nSTAT_currentRelease_Local"
        / "helpfiles"
    )
    candidates.append(repo_root / ".." / "nSTAT_currentRelease_Local" / "helpfiles")

    for cand in candidates:
        resolved = cand.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    return None


def collect_matlab_reference_images(topic: str, matlab_help_root: Path | None) -> list[Path]:
    if matlab_help_root is None:
        return []

    topic_lower = topic.lower()
    found: list[Path] = []
    seen: set[Path] = set()

    def add_if_valid(path: Path) -> None:
        if not path.exists():
            return
        name = path.name.lower()
        if name.startswith(f"{topic_lower}_eq"):
            return
        if "eq" in name and name.startswith(topic_lower):
            return
        if name.startswith("logo"):
            return
        if path not in seen:
            seen.add(path)
            found.append(path)

    html_path = matlab_help_root / f"{topic}.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        srcs = re.findall(r'<img[^>]+src="([^"]+)"', html, flags=re.IGNORECASE)
        for src in srcs:
            src_name = Path(src).name
            ext = src_name.lower().rsplit(".", 1)[-1] if "." in src_name else ""
            if ext not in {"png", "jpg", "jpeg", "gif"}:
                continue
            candidate = matlab_help_root / src_name
            add_if_valid(candidate)

    for pattern in (f"{topic}_*.png", f"{topic}.png", f"{topic}-*.png"):
        for candidate in sorted(matlab_help_root.glob(pattern)):
            add_if_valid(candidate)

    if found:
        priority: list[Path] = []
        secondary: list[Path] = []
        for path in found:
            name = path.name.lower()
            if name.startswith(f"{topic_lower}_") or name == f"{topic_lower}.png":
                priority.append(path)
            else:
                secondary.append(path)
        found = priority + secondary

    return found[:8]


@functools.lru_cache(maxsize=1024)
def _load_similarity_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _ncc_score(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    va = arr_a.ravel() - float(np.mean(arr_a))
    vb = arr_b.ravel() - float(np.mean(arr_b))
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return 0.0
    ncc = float(np.dot(va, vb) / denom)
    return max(0.0, min(1.0, (ncc + 1.0) / 2.0))


def compute_image_similarity(path_a: Path, path_b: Path, max_shift_px: int = 12) -> float:
    arr_a = _load_similarity_array(path_a)
    arr_b = _load_similarity_array(path_b)

    # Allow small translation tolerance so margin/layout shifts do not
    # dominate similarity scoring for otherwise equivalent plots.
    best = 0.0
    min_overlap = 96
    for dy in range(-max_shift_px, max_shift_px + 1):
        for dx in range(-max_shift_px, max_shift_px + 1):
            y1a = max(0, dy)
            y2a = min(256, 256 + dy)
            x1a = max(0, dx)
            x2a = min(256, 256 + dx)

            y1b = max(0, -dy)
            y2b = min(256, 256 - dy)
            x1b = max(0, -dx)
            x2b = min(256, 256 - dx)

            if (y2a - y1a) < min_overlap or (x2a - x1a) < min_overlap:
                continue

            score = _ncc_score(arr_a[y1a:y2a, x1a:x2a], arr_b[y1b:y2b, x1b:x2b])
            if score > best:
                best = score
    return best


def execute_notebook_capture(
    target: NotebookTarget,
    tmp_dir: Path,
    timeout: int,
    matlab_help_root: Path | None,
    parity_threshold: float,
    skip_parity_check: bool,
    parity_mode: str,
    gate_status: tuple[str, bool] | None,
    parity_metrics: dict[str, object] | None,
) -> NotebookReport:
    start = time.perf_counter()
    image_dir = tmp_dir / "notebook_images" / target.topic
    image_dir.mkdir(parents=True, exist_ok=True)

    matlab_ref_images = collect_matlab_reference_images(target.topic, matlab_help_root)

    if not target.file.exists():
        duration = time.perf_counter() - start
        return NotebookReport(
            topic=target.topic,
            file=target.file,
            run_group=target.run_group,
            executed=False,
            duration_s=duration,
            image_paths=[],
            image_count=0,
            text_snippet="",
            error=f"Notebook not found: {target.file}",
            matlab_ref_images=matlab_ref_images,
            similarity_score=None,
            parity_pass=None,
            alignment_status=(gate_status[0] if gate_status is not None else None),
            matched_python_image=None,
            matched_matlab_image=None,
            parity_metrics=parity_metrics,
        )

    notebook = nbformat.read(target.file, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(target.file.parent)}},
    )

    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - start
        return NotebookReport(
            topic=target.topic,
            file=target.file,
            run_group=target.run_group,
            executed=False,
            duration_s=duration,
            image_paths=[],
            image_count=0,
            text_snippet="",
            error=str(exc),
            matlab_ref_images=matlab_ref_images,
            similarity_score=None,
            parity_pass=None,
            alignment_status=(gate_status[0] if gate_status is not None else None),
            matched_python_image=None,
            matched_matlab_image=None,
            parity_metrics=parity_metrics,
        )

    image_paths: list[Path] = []
    text_snippet = ""
    image_index = 0

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            output_type = output.get("output_type", "")
            if output_type in {"display_data", "execute_result"}:
                data = output.get("data", {})
                png_b64 = data.get("image/png")
                if png_b64 is not None:
                    if isinstance(png_b64, list):
                        png_b64 = "".join(png_b64)
                    try:
                        png_bytes = base64.b64decode(png_b64)
                    except Exception:  # noqa: BLE001
                        png_bytes = b""
                    if png_bytes:
                        image_index += 1
                        image_path = image_dir / f"{target.topic}_{image_index:03d}.png"
                        image_path.write_bytes(png_bytes)
                        image_paths.append(image_path)
                if not text_snippet and "text/plain" in data:
                    text_plain = data["text/plain"]
                    if isinstance(text_plain, list):
                        text_plain = "\n".join(text_plain)
                    text_snippet = _short_text(str(text_plain))
            elif output_type == "stream" and not text_snippet:
                text_snippet = _short_text(str(output.get("text", "")))

    similarity_score: float | None = None
    parity_pass: bool | None = None
    alignment_status: str | None = gate_status[0] if gate_status is not None else None
    matched_python_image: Path | None = None
    matched_matlab_image: Path | None = None

    if parity_mode == "gate":
        if gate_status is not None:
            parity_pass = bool(gate_status[1])
        else:
            parity_pass = False
    else:
        if not skip_parity_check:
            if image_paths and matlab_ref_images:
                best = -1.0
                for py_img in image_paths:
                    for mat_img in matlab_ref_images:
                        sim = compute_image_similarity(py_img, mat_img)
                        if sim > best:
                            best = sim
                            matched_python_image = py_img
                            matched_matlab_image = mat_img
                similarity_score = best if best >= 0.0 else None
                parity_pass = similarity_score >= parity_threshold
            else:
                parity_pass = None

    duration = time.perf_counter() - start
    return NotebookReport(
        topic=target.topic,
        file=target.file,
        run_group=target.run_group,
        executed=True,
        duration_s=duration,
        image_paths=image_paths,
        image_count=len(image_paths),
        text_snippet=text_snippet,
        error="",
        matlab_ref_images=matlab_ref_images,
        similarity_score=similarity_score,
        parity_pass=parity_pass,
        alignment_status=alignment_status,
        matched_python_image=matched_python_image,
        matched_matlab_image=matched_matlab_image,
        parity_metrics=parity_metrics,
    )


def _draw_wrapped_lines(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    wrap_width: int = 100,
    line_step: float = 12.0,
) -> float:
    lines: list[str] = []
    for block in text.splitlines() or [""]:
        wrapped = textwrap.wrap(block, width=wrap_width) or [""]
        lines.extend(wrapped)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= line_step
    return y


def _draw_image_fit(pdf: canvas.Canvas, image_path: Path, x: float, y: float, max_w: float, max_h: float) -> None:
    reader = ImageReader(str(image_path))
    iw, ih = reader.getSize()
    scale = min(max_w / iw, max_h / ih)
    w = iw * scale
    h = ih * scale
    pdf.drawImage(reader, x, y, width=w, height=h)


def _draw_image_gallery(
    pdf: canvas.Canvas,
    images: list[Path],
    x: float,
    y: float,
    width: float,
    height: float,
    max_items: int = 4,
) -> None:
    subset = images[:max_items]
    if not subset:
        pdf.setFont("Helvetica", 9)
        pdf.drawString(x, y + height - 12, "No images available.")
        return

    n = len(subset)
    if n == 1:
        _draw_image_fit(pdf, subset[0], x, y, width, height)
        return

    cols = 2
    rows = 2 if n > 2 else 1
    cell_w = (width - 8) / cols
    cell_h = (height - 8) / rows

    for idx, image_path in enumerate(subset):
        col = idx % cols
        row = idx // cols
        if row >= rows:
            break
        cell_x = x + col * (cell_w + 8)
        cell_y = y + (rows - 1 - row) * (cell_h + 8)
        _draw_image_fit(pdf, image_path, cell_x, cell_y, cell_w, cell_h)


def _format_metric_value(value: object | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _draw_metrics_table(
    pdf: canvas.Canvas,
    metrics: dict[str, object] | None,
    *,
    x: float,
    top_y: float,
    width: float,
) -> None:
    rows = [
        ("matlab_code_lines", "MATLAB code lines"),
        ("python_code_lines", "Python code lines"),
        ("python_to_matlab_line_ratio", "Python/MATLAB line ratio"),
        ("matlab_reference_image_count", "MATLAB reference images"),
        ("python_validation_image_count", "Python validation images"),
        ("assertion_count", "Checkpoint assertions"),
        ("has_plot_call", "Contains plotting logic"),
        ("has_topic_checkpoint", "Has topic checkpoint"),
    ]
    row_h = 12.0
    table_h = row_h * (len(rows) + 1)
    key_col_w = width * 0.68

    pdf.setLineWidth(0.6)
    pdf.rect(x, top_y - table_h, width, table_h)
    pdf.line(x + key_col_w, top_y, x + key_col_w, top_y - table_h)
    for idx in range(1, len(rows) + 1):
        y = top_y - idx * row_h
        pdf.line(x, y, x + width, y)

    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(x + 4, top_y - 9, "Metric")
    pdf.drawString(x + key_col_w + 4, top_y - 9, "Value")

    pdf.setFont("Helvetica", 8)
    if metrics is None:
        pdf.drawString(x + 4, top_y - row_h - 9, "No parity metric row available")
        return

    for idx, (key, label) in enumerate(rows, start=1):
        y = top_y - idx * row_h - 9
        pdf.drawString(x + 4, y, label)
        pdf.drawString(x + key_col_w + 4, y, _format_metric_value(metrics.get(key)))


def draw_cover_page(
    pdf: canvas.Canvas,
    repo_root: Path,
    commit: str,
    generated_at: str,
    notebook_group: str,
    selected_count: int,
    command_results: list[CommandResult],
    matlab_help_root: Path | None,
    parity_threshold: float,
    skip_parity_check: bool,
    parity_mode: str,
) -> None:
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, 760, "nSTAT-python Validation Report (All Examples)")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, 736, f"Generated: {generated_at}")
    pdf.drawString(40, 720, f"Repository: {repo_root}")
    pdf.drawString(40, 704, f"Commit: {commit}")
    pdf.drawString(40, 688, f"Notebook group: {notebook_group}")
    pdf.drawString(40, 672, f"Examples included: {selected_count}")
    matlab_root_msg = str(matlab_help_root) if matlab_help_root is not None else "NOT FOUND"
    pdf.drawString(40, 656, f"MATLAB helpfiles root: {matlab_root_msg}")
    if parity_mode == "gate":
        parity_msg = "gate-status (equivalence audit + output spec)"
    else:
        parity_msg = "SKIPPED" if skip_parity_check else f"image similarity threshold={parity_threshold:.2f}"
    pdf.drawString(40, 640, f"Parity mode: {parity_mode} ({parity_msg})")

    y = 612
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, y, "Command checks")
    y -= 18
    pdf.setFont("Helvetica", 10)

    if not command_results:
        pdf.drawString(46, y, "- Skipped by --skip-command-tests")
        y -= 14
    else:
        for result in command_results:
            status = "PASS" if result.passed else "FAIL"
            pdf.drawString(46, y, f"- {result.name}: {status} ({result.duration_s:.2f}s)")
            y -= 14
            if result.stdout_tail:
                y = _draw_wrapped_lines(pdf, 58, y, result.stdout_tail, wrap_width=90)
                y -= 6
            if y < 90:
                pdf.showPage()
                y = 760
                pdf.setFont("Helvetica", 10)

    pdf.showPage()


def draw_summary_pages(
    pdf: canvas.Canvas,
    reports: list[NotebookReport],
    skip_parity_check: bool,
    parity_mode: str,
) -> None:
    total = len(reports)
    executed = sum(1 for report in reports if report.executed)
    failed_exec = total - executed
    with_py_images = sum(1 for report in reports if report.image_count > 0)
    with_matlab_refs = sum(1 for report in reports if len(report.matlab_ref_images) > 0)
    parity_checked = sum(1 for report in reports if report.parity_pass is not None)
    parity_passed = sum(1 for report in reports if report.parity_pass is True)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 760, "Example Coverage Summary")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, 738, f"Total examples: {total}")
    pdf.drawString(180, 738, f"Executed: {executed}")
    pdf.drawString(300, 738, f"Exec failures: {failed_exec}")
    pdf.drawString(430, 738, f"Py figs: {with_py_images}")
    pdf.drawString(40, 722, f"MATLAB refs available: {with_matlab_refs}")
    if parity_mode == "gate":
        pdf.drawString(260, 722, f"Parity gate pass: {parity_passed}/{parity_checked}")
    elif skip_parity_check:
        pdf.drawString(260, 722, "Parity scoring: skipped")
    else:
        pdf.drawString(260, 722, f"Parity pass: {parity_passed}/{parity_checked}")

    y = 694
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(40, y, "Exec")
    pdf.drawString(74, y, "Parity")
    pdf.drawString(126, y, "Topic")
    pdf.drawString(308, y, "Py")
    pdf.drawString(332, y, "MAT")
    pdf.drawString(360, y, "Score")
    pdf.drawString(402, y, "Run")
    pdf.drawString(438, y, "Sec")
    pdf.drawString(472, y, "Status")
    y -= 12
    pdf.setFont("Helvetica", 9)

    for report in reports:
        if y < 70:
            pdf.showPage()
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(40, 760, "Exec")
            pdf.drawString(74, 760, "Parity")
            pdf.drawString(126, 760, "Topic")
            pdf.drawString(308, 760, "Py")
            pdf.drawString(332, 760, "MAT")
            pdf.drawString(360, 760, "Score")
            pdf.drawString(402, 760, "Run")
            pdf.drawString(438, 760, "Sec")
            pdf.drawString(472, 760, "Status")
            y = 748
            pdf.setFont("Helvetica", 9)

        exec_status = "PASS" if report.executed else "FAIL"
        if report.parity_pass is None:
            parity_status = "N/A"
        else:
            parity_status = "PASS" if report.parity_pass else "FAIL"

        score_text = f"{report.similarity_score:.3f}" if report.similarity_score is not None else "-"

        pdf.drawString(40, y, exec_status)
        pdf.drawString(74, y, parity_status)
        pdf.drawString(126, y, report.topic[:30])
        pdf.drawString(308, y, str(report.image_count))
        pdf.drawString(332, y, str(len(report.matlab_ref_images)))
        pdf.drawString(360, y, score_text)
        pdf.drawString(402, y, report.run_group)
        pdf.drawString(438, y, f"{report.duration_s:.2f}")
        status_text = report.alignment_status if report.alignment_status is not None else "-"
        pdf.drawString(472, y, status_text[:16])
        y -= 12

    pdf.showPage()


def draw_example_page(pdf: canvas.Canvas, report: NotebookReport, index: int, total: int) -> None:
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(40, 760, f"Example {index}/{total}: {report.topic}")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, 742, f"Notebook: {report.file}")
    pdf.drawString(40, 728, f"Run group: {report.run_group}")

    exec_status = "PASS" if report.executed else "FAIL"
    if report.parity_pass is None:
        parity_status = "N/A"
    else:
        parity_status = "PASS" if report.parity_pass else "FAIL"

    score_text = f"{report.similarity_score:.3f}" if report.similarity_score is not None else "-"
    header = (
        f"Execution: {exec_status} | Parity: {parity_status} | Similarity: {score_text} | "
        f"Runtime: {report.duration_s:.2f}s"
    )
    pdf.drawString(40, 714, header)
    if report.alignment_status is not None:
        pdf.drawString(40, 700, f"Equivalence status: {report.alignment_status}")
        y_header = 686
    else:
        y_header = 700
    pdf.drawString(
        40,
        y_header,
        f"Python figs: {report.image_count} | MATLAB refs: {len(report.matlab_ref_images)}",
    )

    y = y_header - 20
    if report.error:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y, "Execution error")
        y -= 16
        pdf.setFont("Helvetica", 10)
        _draw_wrapped_lines(pdf, 48, y, report.error, wrap_width=92)
        pdf.showPage()
        return

    # Side-by-side galleries so each example page contains distinct visual evidence.
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, 664, "Python output gallery")
    _draw_image_gallery(pdf, report.image_paths, 40, 350, 250, 300, max_items=4)

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(310, 664, "MATLAB reference gallery")
    _draw_image_gallery(pdf, report.matlab_ref_images, 310, 350, 250, 300, max_items=4)

    pdf.setFont("Helvetica", 8)
    pdf.drawString(40, 336, f"Python figures shown: {min(report.image_count, 4)} / {report.image_count}")
    pdf.drawString(
        310,
        336,
        f"MATLAB refs shown: {min(len(report.matlab_ref_images), 4)} / {len(report.matlab_ref_images)}",
    )

    if report.matched_python_image is not None and report.matched_matlab_image is not None:
        pdf.setFont("Helvetica", 8)
        py_name = report.matched_python_image.name
        mat_name = report.matched_matlab_image.name
        pdf.drawString(40, 322, f"Best-match pair: {py_name} vs {mat_name}")

    if report.text_snippet:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, 304, "Output snippet")
        pdf.setFont("Helvetica", 9)
        _draw_wrapped_lines(pdf, 48, 290, report.text_snippet, wrap_width=102, line_step=10)

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, 190, "MATLAB vs Python key metrics")
    _draw_metrics_table(
        pdf,
        report.parity_metrics,
        x=40,
        top_y=182,
        width=520,
    )

    pdf.showPage()


def generate_pdf_report(
    repo_root: Path,
    manifest_path: Path,
    output_pdf: Path,
    tmp_dir: Path,
    notebook_group: str,
    timeout: int,
    run_commands: bool,
    matlab_help_root: Path | None,
    parity_threshold: float,
    skip_parity_check: bool,
    parity_mode: str,
    equivalence_report: Path,
    example_output_spec: Path,
) -> tuple[Path, list[NotebookReport], list[CommandResult], Path | None]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    command_results: list[CommandResult] = []
    if run_commands:
        commands = [
            (
                "Unit tests",
                ["pytest", "-q", "tests/test_parity_numerics.py", "tests/test_behavior_contracts.py"],
            ),
            (
                "No MATLAB dependency gate",
                ["python", "tools/compliance/check_no_matlab_dependency.py"],
            ),
        ]
        for name, cmd in commands:
            command_results.append(run_command(name=name, cmd=cmd, cwd=repo_root))

    resolved_matlab_help_root = resolve_matlab_help_root(repo_root, matlab_help_root)
    parity_gate_status = load_parity_gate_status(equivalence_report, example_output_spec)
    parity_topic_metrics = load_parity_topic_metrics(equivalence_report)

    targets = load_targets(manifest_path, repo_root, notebook_group)
    reports: list[NotebookReport] = []
    for target in targets:
        reports.append(
            execute_notebook_capture(
                target=target,
                tmp_dir=tmp_dir,
                timeout=timeout,
                matlab_help_root=resolved_matlab_help_root,
                parity_threshold=parity_threshold,
                skip_parity_check=skip_parity_check,
                parity_mode=parity_mode,
                gate_status=parity_gate_status.get(target.topic),
                parity_metrics=parity_topic_metrics.get(target.topic),
            )
        )

    commit = (
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True, text=True)
        .stdout.strip()
        or "unknown"
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = canvas.Canvas(str(output_pdf), pagesize=letter)
    pdf.setTitle("nSTAT-python Validation Report")

    draw_cover_page(
        pdf=pdf,
        repo_root=repo_root,
        commit=commit,
        generated_at=generated_at,
        notebook_group=notebook_group,
        selected_count=len(targets),
        command_results=command_results,
        matlab_help_root=resolved_matlab_help_root,
        parity_threshold=parity_threshold,
        skip_parity_check=skip_parity_check,
        parity_mode=parity_mode,
    )
    draw_summary_pages(
        pdf=pdf,
        reports=reports,
        skip_parity_check=skip_parity_check,
        parity_mode=parity_mode,
    )

    total = len(reports)
    for index, report in enumerate(reports, start=1):
        draw_example_page(pdf=pdf, report=report, index=index, total=total)

    pdf.save()
    return output_pdf, reports, command_results, resolved_matlab_help_root


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pdf = args.output_dir / f"nstat_python_validation_report_{stamp}.pdf"

    report_path, reports, command_results, matlab_help_root = generate_pdf_report(
        repo_root=args.repo_root,
        manifest_path=args.manifest,
        output_pdf=output_pdf,
        tmp_dir=args.tmp_dir,
        notebook_group=args.notebook_group,
        timeout=args.timeout,
        run_commands=not args.skip_command_tests,
        matlab_help_root=args.matlab_help_root,
        parity_threshold=args.parity_threshold,
        skip_parity_check=args.skip_parity_check,
        parity_mode=args.parity_mode,
        equivalence_report=args.equivalence_report,
        example_output_spec=args.example_output_spec,
    )

    executed = sum(1 for report in reports if report.executed)
    exec_failures = len(reports) - executed
    with_images = sum(1 for report in reports if report.image_count > 0)
    parity_checked = sum(1 for report in reports if report.parity_pass is not None)
    parity_failures = sum(1 for report in reports if report.parity_pass is False)
    command_failures = sum(1 for result in command_results if not result.passed)

    print(f"Generated PDF report: {report_path}")
    print(f"MATLAB help root: {matlab_help_root}")
    print(
        f"Notebook results: total={len(reports)} executed={executed} exec_failures={exec_failures} "
        f"with_images={with_images}"
    )
    print(f"Parity results ({args.parity_mode} mode): checked={parity_checked} failures={parity_failures}")
    print(f"Command checks: total={len(command_results)} failed={command_failures}")

    return 0 if exec_failures == 0 and command_failures == 0 and parity_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
