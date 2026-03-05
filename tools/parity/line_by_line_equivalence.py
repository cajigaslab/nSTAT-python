#!/usr/bin/env python3
"""Check MATLAB↔Python line-structure equivalence for mapped entities."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat
import yaml


MATLAB_COMMENT_RE = re.compile(r"^\s*#\s*MATLAB(?:\s+L(?P<line>\d+))?:", re.IGNORECASE)


@dataclass(slots=True)
class Rule:
    entity_type: str
    min_comment_ratio: float
    min_pair_ratio: float
    max_order_violations: int


@dataclass(slots=True)
class Result:
    entity_name: str
    entity_type: str
    matlab_path: str
    python_path: str
    matlab_code_lines: int
    matlab_comment_markers: int
    mirrored_pairs: int
    comment_ratio: float
    pair_ratio: float
    order_violations: int
    pass_check: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--mapping", type=Path, default=Path("parity/port_mapping.yml"))
    parser.add_argument("--policy", type=Path, default=Path("parity/line_equivalence_policy.yml"))
    parser.add_argument("--out-json", type=Path, default=Path("parity/line_equivalence_report.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("parity/line_equivalence_report.csv"))
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload at {path}")
    return payload


def _load_generator_module(repo_root: Path):
    mod_path = repo_root / "tools" / "notebooks" / "generate_helpfile_notebooks.py"
    spec = importlib.util.spec_from_file_location("generate_helpfile_notebooks", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import notebook generator from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _matlab_code_line_count(matlab_path: Path, generator_mod: Any) -> int:
    source_type = "mlx" if matlab_path.suffix.lower() == ".mlx" else "m"
    sections = generator_mod._extract_sections(matlab_path, source_type)
    return sum(1 for section in sections for line in section.lines if line.is_code)


def _python_lines(path: Path) -> list[str]:
    if path.suffix.lower() == ".ipynb":
        nb = nbformat.read(path, as_version=4)
        lines: list[str] = []
        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            src = str(cell.get("source", ""))
            lines.extend(src.splitlines())
        return lines
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _evaluate(path: Path, matlab_code_lines: int) -> tuple[int, int, float, float, int]:
    lines = _python_lines(path)
    marker_idxs: list[int] = []
    marker_line_numbers: list[int] = []

    for idx, line in enumerate(lines):
        match = MATLAB_COMMENT_RE.match(line)
        if not match:
            continue
        marker_idxs.append(idx)
        if match.group("line"):
            marker_line_numbers.append(int(match.group("line")))

    markers = len(marker_idxs)

    mirrored_pairs = 0
    for idx in marker_idxs:
        nxt = idx + 1
        while nxt < len(lines) and not lines[nxt].strip():
            nxt += 1
        if nxt >= len(lines):
            continue
        if not lines[nxt].lstrip().startswith("#"):
            mirrored_pairs += 1

    order_violations = 0
    for i in range(1, len(marker_line_numbers)):
        if marker_line_numbers[i] < marker_line_numbers[i - 1]:
            order_violations += 1

    denominator = max(matlab_code_lines, 1)
    comment_ratio = markers / denominator
    pair_ratio = mirrored_pairs / max(markers, 1)
    return markers, mirrored_pairs, comment_ratio, pair_ratio, order_violations


def _rules(policy: dict[str, Any]) -> dict[str, Rule]:
    default = policy.get("default", {}) if isinstance(policy.get("default", {}), dict) else {}
    default_rule = Rule(
        entity_type="default",
        min_comment_ratio=float(default.get("min_comment_ratio", 0.0)),
        min_pair_ratio=float(default.get("min_pair_ratio", 0.0)),
        max_order_violations=int(default.get("max_order_violations", 10_000)),
    )
    out: dict[str, Rule] = {"default": default_rule}
    entity_rules = policy.get("entity_rules", {}) if isinstance(policy.get("entity_rules", {}), dict) else {}
    for entity_type, row in entity_rules.items():
        if not isinstance(row, dict):
            continue
        out[entity_type] = Rule(
            entity_type=entity_type,
            min_comment_ratio=float(row.get("min_comment_ratio", default_rule.min_comment_ratio)),
            min_pair_ratio=float(row.get("min_pair_ratio", default_rule.min_pair_ratio)),
            max_order_violations=int(row.get("max_order_violations", default_rule.max_order_violations)),
        )
    return out


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    mapping_payload = _load_yaml((repo_root / args.mapping).resolve())
    policy_payload = _load_yaml((repo_root / args.policy).resolve())
    rules = _rules(policy_payload)
    generator_mod = _load_generator_module(repo_root)

    results: list[Result] = []
    entries = mapping_payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("port_mapping.yml must contain an entries list")

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).strip().lower()
        if status != "mapped":
            continue
        matlab_path = Path(str(entry.get("matlab_path", "")).strip())
        python_rel = str(entry.get("python_path", "")).strip()
        if not matlab_path.exists() or not python_rel:
            continue
        python_path = (repo_root / python_rel).resolve()
        if not python_path.exists():
            continue

        matlab_code_lines = _matlab_code_line_count(matlab_path, generator_mod)
        markers, pairs, comment_ratio, pair_ratio, ordering = _evaluate(python_path, matlab_code_lines)
        entity_type = str(entry.get("entity_type", "unknown")).strip()
        rule = rules.get(entity_type, rules["default"])
        pass_check = (
            comment_ratio >= rule.min_comment_ratio
            and pair_ratio >= rule.min_pair_ratio
            and ordering <= rule.max_order_violations
        )
        results.append(
            Result(
                entity_name=str(entry.get("entity_name", "")),
                entity_type=entity_type,
                matlab_path=str(matlab_path),
                python_path=str(python_path),
                matlab_code_lines=matlab_code_lines,
                matlab_comment_markers=markers,
                mirrored_pairs=pairs,
                comment_ratio=comment_ratio,
                pair_ratio=pair_ratio,
                order_violations=ordering,
                pass_check=pass_check,
            )
        )

    failures = [row for row in results if not row.pass_check]
    payload = {
        "policy": str((repo_root / args.policy).resolve()),
        "mapping": str((repo_root / args.mapping).resolve()),
        "total": len(results),
        "failures": len(failures),
        "results": [asdict(row) for row in results],
    }

    out_json = (repo_root / args.out_json).resolve()
    out_csv = (repo_root / args.out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "entity_name",
                "entity_type",
                "matlab_path",
                "python_path",
                "matlab_code_lines",
                "matlab_comment_markers",
                "mirrored_pairs",
                "comment_ratio",
                "pair_ratio",
                "order_violations",
                "pass_check",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    print(json.dumps({"total": len(results), "failures": len(failures)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
