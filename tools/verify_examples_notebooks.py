from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "helpfiles").exists() else PROJECT_ROOT.parent
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "helpfiles"
SRC_ROOT = PROJECT_ROOT / "examples" / "help_topics"
REPORT_DIR = PROJECT_ROOT / "reports"
TOC_PATH = REPO_ROOT / "helpfiles" / "helptoc.xml"
FIGURE_CONTRACT_PATH = SRC_ROOT / "figure_contract.json"
FIGURE_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "figures"
PY_ROOT = PROJECT_ROOT
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")


def _load_contract() -> dict[str, dict[str, object]]:
    data = json.loads(FIGURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    if not isinstance(topics, dict) or not topics:
        raise RuntimeError(f"Invalid figure contract: {FIGURE_CONTRACT_PATH}")
    return topics


def example_topics() -> list[tuple[str, str]]:
    tree = ET.parse(TOC_PATH)
    root = tree.getroot()
    examples = None
    for item in root.iter("tocitem"):
        if item.attrib.get("id") == "nstat_examples":
            examples = item
            break
    if examples is None:
        raise RuntimeError("Unable to locate examples node in helptoc.xml")

    out: list[tuple[str, str]] = []
    for item in examples.findall("tocitem"):
        title = " ".join((item.text or "").split()) or Path(item.attrib.get("target", "")).stem
        target = item.attrib.get("target", "")
        if target:
            out.append((title, target))
    return out


def _clear_figure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.glob("fig_*.png"):
        p.unlink()


def _to_repo_relative(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _validate_figure_payload(output: dict[str, Any] | None, expected: int) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {
            "ok": False,
            "figure_count": -1,
            "figures_len": -1,
            "missing": ["<missing output dict>"],
        }
    figure_count = int(output.get("figure_count", -1))
    figures = output.get("figures", [])
    if not isinstance(figures, list):
        figures = []
    missing = [str(p) for p in figures if not Path(p).exists()]
    contract_expected = int(output.get("figure_contract_expected", -1))
    ok = (
        contract_expected == expected
        and figure_count == expected
        and len(figures) == expected
        and len(missing) == 0
    )
    return {
        "ok": ok,
        "figure_count": figure_count,
        "figures_len": len(figures),
        "figure_contract_expected": contract_expected,
        "missing": [_to_repo_relative(p) for p in missing],
        "figures": [_to_repo_relative(p) for p in figures],
    }


def run_python_module(stem: str, expected_figures: int) -> dict[str, Any]:
    try:
        mod = importlib.import_module(f"examples.help_topics.{stem}")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"import_error: {exc}"}

    if not hasattr(mod, "run"):
        return {"ok": False, "error": "missing run()"}

    module_fig_dir = FIGURE_OUTPUT_ROOT / "modules" / stem
    _clear_figure_dir(module_fig_dir)

    try:
        out = mod.run(repo_root=REPO_ROOT, figure_dir=module_fig_dir, render_figures=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }

    figures = _validate_figure_payload(out if isinstance(out, dict) else None, expected_figures)
    figures["directory_files"] = len(list(module_fig_dir.glob("fig_*.png")))
    figures["directory"] = _to_repo_relative(module_fig_dir)

    return {
        "ok": bool(figures["ok"]),
        "error": "" if figures["ok"] else "figure_contract_failed",
        "output": out,
        "figures": figures,
    }


def execute_notebook(path: Path, topic: str, expected_figures: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "ok": False,
            "error": "notebook_missing",
            "stdout": "",
            "code_cells": 0,
            "output": None,
            "figures": {"ok": False, "figure_count": -1, "figures_len": -1, "missing": ["<notebook missing>"]},
        }

    notebook_fig_dir = FIGURE_OUTPUT_ROOT / "notebooks" / topic
    _clear_figure_dir(notebook_fig_dir)

    data = json.loads(path.read_text(encoding="utf-8"))
    ns: dict[str, Any] = {}
    buf = io.StringIO()
    code_cells = 0
    try:
        for idx, cell in enumerate(data.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            code = "".join(cell.get("source", []))
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, f"{path}:{idx}", "exec"), ns, ns)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "stdout": buf.getvalue(),
            "code_cells": code_cells,
            "output": ns.get("out"),
            "figures": _validate_figure_payload(ns.get("out"), expected_figures),
        }

    figures = _validate_figure_payload(ns.get("out"), expected_figures)
    figures["directory_files"] = len(list(notebook_fig_dir.glob("fig_*.png")))
    figures["directory"] = _to_repo_relative(notebook_fig_dir)
    return {
        "ok": bool(figures["ok"]),
        "error": "" if figures["ok"] else "figure_contract_failed",
        "stdout": buf.getvalue(),
        "code_cells": code_cells,
        "output": ns.get("out"),
        "figures": figures,
    }


def main() -> int:
    topics = example_topics()
    contract = _load_contract()
    expected_total = int(os.environ.get("NSTAT_EXPECTED_EXAMPLE_NOTEBOOKS", "25"))

    rows: list[dict[str, Any]] = []
    summary = {
        "total_examples": len(topics),
        "python_modules_ok": 0,
        "notebooks_ok": 0,
        "topic_alignment_ok": 0,
        "figure_contract_ok": 0,
        "zero_figure_contract_ok": 0,
    }

    for title, target in topics:
        stem = Path(target).stem
        if stem not in contract:
            rows.append(
                {
                    "example": stem,
                    "title": title,
                    "matlab_target": target,
                    "python_module_ok": False,
                    "python_module_error": "missing_figure_contract",
                    "notebook_ok": False,
                    "notebook_error": "missing_figure_contract",
                    "topic_alignment_ok": False,
                    "figure_contract_expected": None,
                }
            )
            continue

        expected_figures = int(contract[stem].get("expected_figures", 0))
        py = run_python_module(stem, expected_figures)
        nb = execute_notebook(NOTEBOOK_ROOT / f"{stem}.ipynb", topic=stem, expected_figures=expected_figures)

        if py["ok"]:
            summary["python_modules_ok"] += 1
        if nb["ok"]:
            summary["notebooks_ok"] += 1

        topic_ok = bool(py.get("output") and isinstance(py["output"], dict) and py["output"].get("topic") == stem)
        if topic_ok:
            summary["topic_alignment_ok"] += 1

        figure_ok = bool(py.get("figures", {}).get("ok") and nb.get("figures", {}).get("ok"))
        if figure_ok:
            summary["figure_contract_ok"] += 1

        if expected_figures == 0 and figure_ok:
            summary["zero_figure_contract_ok"] += 1

        rows.append(
            {
                "example": stem,
                "title": title,
                "matlab_target": target,
                "figure_contract_expected": expected_figures,
                "python_module_ok": py["ok"],
                "python_module_error": py.get("error", ""),
                "python_output_keys": sorted(list(py.get("output", {}).keys())) if isinstance(py.get("output"), dict) else [],
                "python_figures": py.get("figures", {}),
                "notebook_ok": nb["ok"],
                "notebook_error": nb.get("error", ""),
                "notebook_code_cells": nb.get("code_cells"),
                "notebook_figures": nb.get("figures", {}),
                "topic_alignment_ok": topic_ok,
                "figure_contract_ok": figure_ok,
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "examples_notebook_verification.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    zero_topics_expected = sum(1 for info in contract.values() if int(info.get("expected_figures", 0)) == 0)
    pass_gate = (
        summary["total_examples"] == expected_total
        and summary["python_modules_ok"] == expected_total
        and summary["notebooks_ok"] == expected_total
        and summary["topic_alignment_ok"] == expected_total
        and summary["figure_contract_ok"] == expected_total
        and summary["zero_figure_contract_ok"] == zero_topics_expected
    )

    print(
        json.dumps(
            {
                "report": str(out.relative_to(REPO_ROOT)),
                "expected_examples": expected_total,
                "expected_zero_figure_topics": zero_topics_expected,
                "pass": pass_gate,
                **summary,
            },
            indent=2,
        )
    )
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
