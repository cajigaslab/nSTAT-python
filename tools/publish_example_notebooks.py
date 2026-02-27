from __future__ import annotations

import argparse
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "helpfiles").exists() else PROJECT_ROOT.parent
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "helpfiles"
TOC_PATH = REPO_ROOT / "helpfiles" / "helptoc.xml"
REPORT_PATH = PROJECT_ROOT / "reports" / "published_notebook_execution.json"


def _example_topics() -> list[str]:
    tree = ET.parse(TOC_PATH)
    root = tree.getroot()
    examples = None
    for item in root.iter("tocitem"):
        if item.attrib.get("id") == "nstat_examples":
            examples = item
            break
    if examples is None:
        raise RuntimeError("Could not find examples section in helptoc.xml")

    out: list[str] = []
    for item in examples.findall("tocitem"):
        target = item.attrib.get("target", "")
        if target:
            out.append(Path(target).stem)
    return out


def _count_outputs(nb: Any) -> tuple[int, int]:
    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    cells_with_outputs = 0
    image_outputs = 0
    for cell in code_cells:
        outputs = cell.get("outputs", [])
        if outputs:
            cells_with_outputs += 1
        for out in outputs:
            if out.get("output_type") not in {"display_data", "execute_result"}:
                continue
            data = out.get("data", {})
            if isinstance(data, dict) and "image/png" in data:
                image_outputs += 1
    return cells_with_outputs, image_outputs


def _execute_notebook(path: Path, timeout_s: int) -> dict[str, Any]:
    nb = nbformat.read(path, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout_s,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "path": str(path.relative_to(REPO_ROOT)),
        }

    nbformat.write(nb, path)
    cells_with_outputs, image_outputs = _count_outputs(nb)
    return {
        "ok": True,
        "path": str(path.relative_to(REPO_ROOT)),
        "code_cells_with_outputs": cells_with_outputs,
        "image_outputs": image_outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute 25 help-example notebooks and persist outputs for GitHub rendering")
    parser.add_argument("--timeout", type=int, default=900, help="Per-notebook execution timeout in seconds")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Output JSON report path")
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")

    topics = _example_topics()
    rows: list[dict[str, Any]] = []
    ok_count = 0

    for stem in topics:
        path = NOTEBOOK_ROOT / f"{stem}.ipynb"
        if not path.exists():
            rows.append(
                {
                    "topic": stem,
                    "ok": False,
                    "error": "notebook_missing",
                    "path": str(path.relative_to(REPO_ROOT)),
                }
            )
            continue

        out = _execute_notebook(path=path, timeout_s=args.timeout)
        out["topic"] = stem
        rows.append(out)
        if out.get("ok"):
            ok_count += 1

    report = Path(args.report).expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "topics": len(topics),
        "executed": ok_count,
        "pass": ok_count == len(topics),
    }
    report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "report": str(report.relative_to(REPO_ROOT))}, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
