from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "helpfiles").exists() else PROJECT_ROOT.parent
TOC_PATH = REPO_ROOT / "helpfiles" / "helptoc.xml"
CONTRACT_PATH = PROJECT_ROOT / "examples" / "help_topics" / "figure_contract.json"
MLX_METADATA_PATH = PROJECT_ROOT / "examples" / "help_topics" / "matlab_mlx_metadata.json"
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "helpfiles"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "published_notebook_check.json"
REQUIRED_NOTEBOOK_SECTIONS = [
    "## What this example demonstrates",
    "## Data and assumptions",
    "## Step-by-step workflow",
    "## Expected figures and interpretation",
    "## Debug tips",
    "## MATLAB Live Script alignment",
    "## Paper terminology and section references",
]


def _example_topics() -> list[tuple[str, str]]:
    tree = ET.parse(TOC_PATH)
    root = tree.getroot()
    examples = None
    for item in root.iter("tocitem"):
        if item.attrib.get("id") == "nstat_examples":
            examples = item
            break
    if examples is None:
        raise RuntimeError("Could not find examples section in helptoc.xml")

    topics: list[tuple[str, str]] = []
    for item in examples.findall("tocitem"):
        title = " ".join((item.text or "").split()) or Path(item.attrib.get("target", "")).stem
        target = item.attrib.get("target", "")
        if target:
            topics.append((title, Path(target).stem))
    return topics


def _load_contract() -> dict[str, dict[str, Any]]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    if not isinstance(topics, dict) or not topics:
        raise RuntimeError(f"Invalid figure contract: {CONTRACT_PATH}")
    return topics


def _load_mlx_metadata() -> dict[str, dict[str, Any]]:
    if not MLX_METADATA_PATH.exists():
        return {}
    data = json.loads(MLX_METADATA_PATH.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    return topics if isinstance(topics, dict) else {}


def _extract_markdown(nb: Any) -> str:
    blocks: list[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        blocks.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(blocks)


def _count_outputs(nb: Any) -> tuple[int, int, int]:
    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    with_outputs = 0
    image_outputs = 0
    stream_outputs = 0
    for cell in code_cells:
        outputs = cell.get("outputs", [])
        if outputs:
            with_outputs += 1
        for out in outputs:
            if out.get("output_type") == "stream":
                stream_outputs += 1
            if out.get("output_type") not in {"display_data", "execute_result"}:
                continue
            data = out.get("data", {})
            if isinstance(data, dict) and "image/png" in data:
                image_outputs += 1
    return len(code_cells), with_outputs, image_outputs + stream_outputs


def _count_image_outputs(nb: Any) -> int:
    image_outputs = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") not in {"display_data", "execute_result"}:
                continue
            data = out.get("data", {})
            if isinstance(data, dict) and "image/png" in data:
                image_outputs += 1
    return image_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify published notebooks contain required narrative sections and embedded outputs")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()

    topics = _example_topics()
    contract = _load_contract()
    mlx_metadata = _load_mlx_metadata()

    rows: list[dict[str, Any]] = []
    ok_count = 0
    for title, stem in topics:
        path = NOTEBOOK_ROOT / f"{stem}.ipynb"
        expected_figures = int(contract.get(stem, {}).get("expected_figures", 0))
        if not path.exists():
            rows.append(
                {
                    "topic": stem,
                    "title": title,
                    "ok": False,
                    "error": "notebook_missing",
                    "path": str(path.relative_to(REPO_ROOT)),
                }
            )
            continue

        nb = nbformat.read(path, as_version=4)
        md = _extract_markdown(nb)
        missing_sections = [section for section in REQUIRED_NOTEBOOK_SECTIONS if section not in md]
        has_mlx_reference = "MATLAB Live Script source:" in md
        has_paper_reference = "pmc.ncbi.nlm.nih.gov/articles/PMC3491120" in md
        mlx_topic = mlx_metadata.get(stem, {})
        mlx_headings = mlx_topic.get("headings", []) if isinstance(mlx_topic, dict) else []
        if not isinstance(mlx_headings, list):
            mlx_headings = []
        mlx_headings = [str(h).strip() for h in mlx_headings if str(h).strip()]
        if not mlx_headings:
            mlx_heading_alignment = has_mlx_reference
        else:
            # Require at least one source MLX heading to appear in notebook narrative.
            mlx_heading_alignment = any(heading in md for heading in mlx_headings[:3])
        code_cells, with_outputs, total_outputs = _count_outputs(nb)
        image_outputs = _count_image_outputs(nb)

        if expected_figures == 0:
            output_ok = image_outputs == 0 and with_outputs > 0
        else:
            output_ok = image_outputs >= expected_figures and with_outputs > 0

        row_ok = (
            (len(missing_sections) == 0)
            and output_ok
            and has_mlx_reference
            and has_paper_reference
            and mlx_heading_alignment
        )
        if row_ok:
            ok_count += 1

        rows.append(
            {
                "topic": stem,
                "title": title,
                "path": str(path.relative_to(REPO_ROOT)),
                "expected_figures": expected_figures,
                "missing_sections": missing_sections,
                "code_cells": code_cells,
                "code_cells_with_outputs": with_outputs,
                "total_outputs": total_outputs,
                "image_outputs": image_outputs,
                "output_ok": output_ok,
                "has_mlx_reference": has_mlx_reference,
                "has_paper_reference": has_paper_reference,
                "mlx_heading_alignment": mlx_heading_alignment,
                "ok": row_ok,
            }
        )

    summary = {
        "topics": len(topics),
        "topics_ok": ok_count,
        "required_sections": REQUIRED_NOTEBOOK_SECTIONS,
        "pass": ok_count == len(topics),
    }

    report = Path(args.report).expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print(json.dumps({**summary, "report": str(report.relative_to(REPO_ROOT))}, indent=2))

    if args.enforce_gate and not summary["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
