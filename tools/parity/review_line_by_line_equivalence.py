#!/usr/bin/env python3
"""Review MATLAB example files against Python notebooks line-by-line."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

try:
    import nbformat
except ModuleNotFoundError:  # pragma: no cover
    nbformat = None


MATLAB_COMMENT_RE = re.compile(r"%.*$")
STRING_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b")
CALL_RE = re.compile(r"(?<![.\w])([A-Za-z_]\w*)\s*\(")
METHOD_CALL_RE = re.compile(r"\.\s*([A-Za-z_]\w*)\s*\(")
INDEX_LIKE_TOKENS = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "lt",
}

MATLAB_OP_CANON = {
    "glmfit": "fit_glm",
    "glmval": "predict",
    "imagesc": "imagesc",
    "imshow": "imagesc",
    "figure": "figure",
    "subplot": "subplot",
    "plot": "plot",
    "plot3": "plot",
    "pcolor": "pcolor",
    "mesh": "mesh",
    "scatter": "plot",
    "hist": "hist",
    "histogram": "hist",
    "load": "load",
    "save": "save",
    "assert": "assert",
}

PY_OP_CANON = {
    "fit_glm": "fit_glm",
    "glmfit": "fit_glm",
    "predict": "predict",
    "imagesc": "imagesc",
    "imshow": "imagesc",
    "figure": "figure",
    "subplots": "subplot",
    "subplot": "subplot",
    "plot": "plot",
    "plot3": "plot",
    "pcolor": "pcolor",
    "pcolormesh": "pcolor",
    "mesh": "mesh",
    "scatter": "plot",
    "hist": "hist",
    "histogram": "hist",
    "load": "load",
    "save": "save",
    "assert": "assert",
}


@dataclass(slots=True)
class CodeLine:
    line_no: int
    raw: str
    norm: str
    ops: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--matlab-root", type=Path, required=True)
    parser.add_argument("--example-mapping", type=Path, default=Path("parity/example_mapping.yaml"))
    parser.add_argument("--method-mapping", type=Path, default=Path("parity/method_mapping.yaml"))
    parser.add_argument("--out-json", type=Path, default=Path("parity/line_by_line_review_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("parity/line_by_line_review.md"))
    parser.add_argument(
        "--fail-on-needs-review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero when any topic remains in needs_review status.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_notebook_cells(path: Path) -> list[dict[str, Any]]:
    if nbformat is not None:
        payload = nbformat.read(path, as_version=4)
        return list(payload.cells)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("cells", []))


def _normalize_line(raw: str, *, matlab: bool) -> str:
    text = raw.strip()
    if matlab:
        text = MATLAB_COMMENT_RE.sub("", text).strip()
    text = STRING_RE.sub("<str>", text)
    text = NUMBER_RE.sub("<num>", text)
    text = text.replace(";", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _canonical_op(name: str, *, matlab: bool, alias_map: dict[str, str]) -> str:
    lname = name.strip().lower()
    if matlab:
        mapped = alias_map.get(name, alias_map.get(lname, name))
        mapped_l = str(mapped).strip().lower()
        return MATLAB_OP_CANON.get(mapped_l, mapped_l)
    return PY_OP_CANON.get(lname, lname)


def _extract_ops(line: str, *, matlab: bool, alias_map: dict[str, str]) -> list[str]:
    ops: list[str] = []
    for token in CALL_RE.findall(line):
        if token.lower() in {"if", "for", "while", "switch", "catch", "function"}:
            continue
        if token.lower() in INDEX_LIKE_TOKENS:
            continue
        ops.append(_canonical_op(token, matlab=matlab, alias_map=alias_map))
    for token in METHOD_CALL_RE.findall(line):
        ops.append(_canonical_op(token, matlab=matlab, alias_map=alias_map))
    # Deduplicate while preserving order for this line.
    seen: set[str] = set()
    deduped: list[str] = []
    for op in ops:
        if op in seen:
            continue
        seen.add(op)
        deduped.append(op)
    return deduped


def _extract_matlab_lines(path: Path, alias_map: dict[str, str]) -> list[CodeLine]:
    if not path.exists():
        return []
    out: list[CodeLine] = []
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    first_code = ""
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("%"):
            continue
        first_code = stripped.lower()
        break
    is_function_file = first_code.startswith("function ")
    seen_primary_function = False

    for idx, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if is_function_file and stripped.lower().startswith("function "):
            if seen_primary_function:
                # Ignore local helper-function implementations; parity should
                # compare the published top-level workflow body.
                break
            seen_primary_function = True
            continue
        if not stripped or stripped.startswith("%"):
            continue
        if stripped == "end":
            continue
        norm = _normalize_line(raw, matlab=True)
        if not norm:
            continue
        ops = _extract_ops(raw, matlab=True, alias_map=alias_map)
        out.append(CodeLine(line_no=idx, raw=stripped[:220], norm=norm, ops=ops))
    return out


def _extract_python_lines(path: Path, alias_map: dict[str, str]) -> list[CodeLine]:
    if not path.exists():
        return []
    out: list[CodeLine] = []
    line_no = 0
    for cell in _load_notebook_cells(path):
        if cell.get("cell_type") != "code":
            continue
        src_raw = cell.get("source", "")
        src = "".join(str(part) for part in src_raw) if isinstance(src_raw, list) else str(src_raw)
        if "validate_numeric_checkpoints" in src and "TOPIC =" in src:
            # Shared setup boilerplate is common across all notebooks and
            # does not represent topic-specific parity.
            continue
        if "validate_numeric_checkpoints(CHECKPOINT_METRICS, CHECKPOINT_LIMITS, TOPIC)" in src:
            # Shared CI assertion cell is intentionally standardized.
            continue
        for raw in src.splitlines():
            line_no += 1
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            norm = _normalize_line(raw, matlab=False)
            if not norm:
                continue
            ops = _extract_ops(raw, matlab=False, alias_map=alias_map)
            out.append(CodeLine(line_no=line_no, raw=stripped[:220], norm=norm, ops=ops))
    return out


def _alignment_metrics(matlab_lines: list[CodeLine], python_lines: list[CodeLine]) -> dict[str, Any]:
    def build_steps(lines: list[CodeLine]) -> list[tuple[str, str]]:
        steps: list[tuple[str, str]] = []
        for row in lines:
            if row.ops:
                for op in row.ops:
                    steps.append((op, row.raw))
                continue
            if row.norm.startswith("for "):
                steps.append(("for", row.raw))
            elif row.norm.startswith("if "):
                steps.append(("if", row.raw))
            elif row.norm.startswith("while "):
                steps.append(("while", row.raw))
            elif row.norm.startswith("assert "):
                steps.append(("assert", row.raw))
            elif row.norm.startswith("load(") or row.norm.startswith("load "):
                steps.append(("load", row.raw))
            elif row.norm.startswith("save(") or row.norm.startswith("save "):
                steps.append(("save", row.raw))
        return steps

    m_steps = build_steps(matlab_lines)
    p_steps = build_steps(python_lines)
    m_norm = [row[0] for row in m_steps]
    p_norm = [row[0] for row in p_steps]

    if not m_norm and not p_norm:
        return {
            "line_alignment_ratio": 1.0,
            "missing_matlab_steps": [],
            "extra_python_steps": [],
            "missing_matlab_step_count": 0,
            "extra_python_step_count": 0,
        }
    if not m_norm or not p_norm:
        return {
            "line_alignment_ratio": 0.0,
            "missing_matlab_steps": [f"{row[0]} :: {row[1]}" for row in m_steps[:25]],
            "extra_python_steps": [f"{row[0]} :: {row[1]}" for row in p_steps[:25]],
            "missing_matlab_step_count": len(m_steps),
            "extra_python_step_count": len(p_steps),
        }

    matcher = SequenceMatcher(a=m_norm, b=p_norm, autojunk=False)
    matched_m: set[int] = set()
    matched_p: set[int] = set()
    matched_lines = 0
    for block in matcher.get_matching_blocks():
        if block.size <= 0:
            continue
        matched_lines += block.size
        matched_m.update(range(block.a, block.a + block.size))
        matched_p.update(range(block.b, block.b + block.size))

    ratio = float((2.0 * matched_lines) / max(len(m_norm) + len(p_norm), 1))
    missing = [f"{m_steps[i][0]} :: {m_steps[i][1]}" for i in range(len(m_steps)) if i not in matched_m]
    extra = [f"{p_steps[i][0]} :: {p_steps[i][1]}" for i in range(len(p_steps)) if i not in matched_p]
    return {
        "line_alignment_ratio": ratio,
        "missing_matlab_steps": missing[:25],
        "extra_python_steps": extra[:25],
        "missing_matlab_step_count": len(missing),
        "extra_python_step_count": len(extra),
    }


def _step_metrics(matlab_lines: list[CodeLine], python_lines: list[CodeLine]) -> dict[str, Any]:
    mat_ops = [op for row in matlab_lines for op in row.ops]
    py_ops = [op for row in python_lines for op in row.ops]
    mat_counter = Counter(mat_ops)
    py_counter = Counter(py_ops)
    shared = sum(min(mat_counter[key], py_counter[key]) for key in mat_counter)
    mat_total = sum(mat_counter.values())
    py_total = sum(py_counter.values())
    recall = float(shared / mat_total) if mat_total > 0 else None
    precision = float(shared / py_total) if py_total > 0 else None

    missing_ops = []
    for op, count in mat_counter.items():
        gap = count - py_counter.get(op, 0)
        if gap > 0:
            missing_ops.append({"op": op, "missing_count": int(gap), "matlab_count": int(count)})
    missing_ops.sort(key=lambda row: row["missing_count"], reverse=True)

    extra_ops = []
    for op, count in py_counter.items():
        gap = count - mat_counter.get(op, 0)
        if gap > 0:
            extra_ops.append({"op": op, "extra_count": int(gap), "python_count": int(count)})
    extra_ops.sort(key=lambda row: row["extra_count"], reverse=True)

    return {
        "matlab_step_recall": recall,
        "python_step_precision": precision,
        "matlab_op_total": mat_total,
        "python_op_total": py_total,
        "missing_matlab_ops": missing_ops[:20],
        "extra_python_ops": extra_ops[:20],
    }


def _status_for_row(row: dict[str, Any]) -> str:
    if not row.get("matlab_exists") or not row.get("python_exists"):
        return "missing_artifact"
    if int(row.get("matlab_line_count", 0)) == 0:
        # MATLAB topic is documentation-only; no executable parity sequence
        # exists to compare line-by-line.
        return "doc_only"
    ratio = float(row.get("line_alignment_ratio") or 0.0)
    recall = float(row.get("matlab_step_recall") or 0.0)
    precision = float(row.get("python_step_precision") or 0.0)
    if ratio >= 0.40 and recall >= 0.50 and precision >= 0.35:
        return "aligned"
    if ratio >= 0.28 and recall >= 0.35:
        return "partially_aligned"
    return "needs_review"


def _write_markdown(out_path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Line-by-Line Equivalence Review")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at_utc']}")
    lines.append(f"- Topics: {summary['total_topics']}")
    lines.append(f"- Aligned: {summary['aligned_topics']}")
    lines.append(f"- Partially aligned: {summary['partially_aligned_topics']}")
    lines.append(f"- Doc-only (MATLAB): {summary['doc_only_topics']}")
    lines.append(f"- Needs review: {summary['needs_review_topics']}")
    lines.append("")
    lines.append("| Topic | Status | Line ratio | Step recall | Step precision | Missing MATLAB steps |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        ratio = row.get("line_alignment_ratio")
        recall = row.get("matlab_step_recall")
        precision = row.get("python_step_precision")
        ratio_text = f"{float(ratio):.3f}" if isinstance(ratio, (int, float)) else "-"
        recall_text = f"{float(recall):.3f}" if isinstance(recall, (int, float)) else "-"
        precision_text = f"{float(precision):.3f}" if isinstance(precision, (int, float)) else "-"
        lines.append(
            "| "
            f"{row['topic']} | {row['line_review_status']} | "
            f"{ratio_text} | {recall_text} | {precision_text} | {row['missing_matlab_step_count']} |"
        )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    matlab_root = args.matlab_root.resolve()
    help_root = matlab_root / "helpfiles"

    example_mapping = _load_yaml((repo_root / args.example_mapping).resolve())
    method_mapping = _load_yaml((repo_root / args.method_mapping).resolve())

    alias_map: dict[str, str] = {}
    for row in method_mapping.get("classes", []):
        for matlab_name, py_name in dict(row.get("alias_methods", {})).items():
            alias_map[str(matlab_name)] = str(py_name)
            alias_map[str(matlab_name).lower()] = str(py_name)

    topic_rows: list[dict[str, Any]] = []
    for row in example_mapping.get("examples", []):
        topic = str(row.get("matlab_topic", "")).strip()
        if not topic:
            continue
        matlab_file = help_root / f"{topic}.m"
        python_nb = (repo_root / str(row.get("python_notebook", ""))).resolve()

        matlab_lines = _extract_matlab_lines(matlab_file, alias_map)
        python_lines = _extract_python_lines(python_nb, alias_map)
        line_metrics = _alignment_metrics(matlab_lines, python_lines)
        step_metrics = _step_metrics(matlab_lines, python_lines)

        out_row: dict[str, Any] = {
            "topic": topic,
            "matlab_file": str(matlab_file),
            "python_notebook": str(python_nb),
            "matlab_exists": matlab_file.exists(),
            "python_exists": python_nb.exists(),
            "matlab_line_count": len(matlab_lines),
            "python_line_count": len(python_lines),
            **line_metrics,
            **step_metrics,
        }
        out_row["line_review_status"] = _status_for_row(out_row)
        topic_rows.append(out_row)

    aligned = sum(1 for row in topic_rows if row["line_review_status"] == "aligned")
    partially = sum(1 for row in topic_rows if row["line_review_status"] == "partially_aligned")
    needs = sum(1 for row in topic_rows if row["line_review_status"] == "needs_review")
    doc_only = sum(1 for row in topic_rows if row["line_review_status"] == "doc_only")
    missing = sum(1 for row in topic_rows if row["line_review_status"] == "missing_artifact")
    avg_ratio = float(
        sum(float(row.get("line_alignment_ratio") or 0.0) for row in topic_rows) / max(len(topic_rows), 1)
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_topics": len(topic_rows),
        "aligned_topics": aligned,
        "partially_aligned_topics": partially,
        "needs_review_topics": needs,
        "doc_only_topics": doc_only,
        "missing_artifact_topics": missing,
        "average_line_alignment_ratio": avg_ratio,
    }

    out_payload = {
        "summary": summary,
        "topic_rows": sorted(topic_rows, key=lambda row: (row["line_review_status"], row["topic"])),
    }

    out_json = (repo_root / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_md = (repo_root / args.out_md).resolve()
    _write_markdown(out_md, summary, out_payload["topic_rows"])

    print(f"Wrote line-by-line review JSON: {out_json}")
    print(f"Wrote line-by-line review markdown: {out_md}")
    print(
        "Line-by-line summary: "
        f"topics={summary['total_topics']} "
        f"aligned={summary['aligned_topics']} "
        f"partial={summary['partially_aligned_topics']} "
        f"needs_review={summary['needs_review_topics']} "
        f"avg_ratio={summary['average_line_alignment_ratio']:.3f}"
    )
    if args.fail_on_needs_review and summary["needs_review_topics"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
