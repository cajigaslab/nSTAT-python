#!/usr/bin/env python3
"""Build MATLAB↔Python port inventory and mapping manifests."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml


CORE_CLASS_NAMES = [
    "SignalObj",
    "Covariate",
    "ConfidenceInterval",
    "Events",
    "History",
    "nspikeTrain",
    "nstColl",
    "CovColl",
    "TrialConfig",
    "ConfigColl",
    "Trial",
    "CIF",
    "Analysis",
    "FitResult",
    "FitResSummary",
    "DecodingAlgorithms",
]

CORE_FUNCTION_NAMES = [
    "getPaperDataDirs",
    "nSTAT_Install",
    "nstatOpenHelpPage",
]

CLASS_PATH_OVERRIDES = {
    "SignalObj": "nstat/SignalObj.py",
    "Covariate": "nstat/Covariate.py",
    "ConfidenceInterval": "nstat/ConfidenceInterval.py",
    "Events": "nstat/events.py",
    "History": "nstat/history.py",
    "nspikeTrain": "nstat/nspikeTrain.py",
    "nstColl": "nstat/nstColl.py",
    "CovColl": "nstat/CovColl.py",
    "TrialConfig": "nstat/TrialConfig.py",
    "ConfigColl": "nstat/ConfigColl.py",
    "Trial": "nstat/trial.py",
    "CIF": "nstat/cif.py",
    "Analysis": "nstat/analysis.py",
    "FitResult": "nstat/FitResult.py",
    "FitResSummary": "nstat/FitResSummary.py",
    "DecodingAlgorithms": "nstat/DecodingAlgorithms.py",
}

FUNCTION_PATH_OVERRIDES = {
    "getPaperDataDirs": "nstat/paper_examples.py",
    "nSTAT_Install": "nstat/nstat_install.py",
    "nstatOpenHelpPage": "nstat/nstat_install.py",
}


@dataclass(slots=True)
class MappingRow:
    matlab_path: str
    entity_name: str
    entity_type: str
    python_path: str
    status: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=Path("parity/matlab_reference.yml"),
    )
    parser.add_argument(
        "--notebook-manifest",
        type=Path,
        default=Path("tools/notebooks/notebook_manifest.yml"),
    )
    parser.add_argument(
        "--out-mapping",
        type=Path,
        default=Path("parity/port_mapping.yml"),
    )
    parser.add_argument(
        "--out-inventory",
        type=Path,
        default=Path("parity/port_inventory.json"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("parity/port_inventory_summary.json"),
    )
    return parser.parse_args()


def _resolve_matlab_root(repo_root: Path, config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    reference = payload.get("reference", {}) if isinstance(payload, dict) else {}
    local_path = str(reference.get("local_path", "")).strip()
    if local_path:
        candidate = Path(local_path)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.exists():
            return candidate
    default_candidate = (repo_root.parent / "nSTAT-matlab-cleanup").resolve()
    if default_candidate.exists():
        return default_candidate
    raise FileNotFoundError("Unable to resolve MATLAB reference root")


def _iter_python_entities(py_root: Path) -> dict[str, dict[str, list[str]]]:
    entities: dict[str, dict[str, list[str]]] = {}
    for path in sorted(py_root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        rel = path.relative_to(py_root.parent).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        classes: list[str] = []
        functions: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        entities[rel] = {"classes": classes, "functions": functions}
    return entities


def _notebook_topic_map(manifest_path: Path) -> dict[str, str]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for row in payload.get("notebooks", []):
        topic = str(row.get("topic", "")).strip()
        file_path = str(row.get("file", "")).strip()
        if topic and file_path:
            out[topic] = file_path
    return out


def _detect_matlab_entities(matlab_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    class_paths: list[Path] = []
    function_paths: list[Path] = []
    help_paths: list[Path] = []

    for path in sorted(matlab_root.glob("*.m")):
        stem = path.stem
        if stem in CORE_CLASS_NAMES:
            class_paths.append(path)
        elif stem in CORE_FUNCTION_NAMES:
            function_paths.append(path)

    help_root = matlab_root / "helpfiles"
    if help_root.exists():
        help_paths.extend(sorted(help_root.glob("*.m")))
        help_paths.extend(sorted(help_root.glob("*.mlx")))
    return class_paths, function_paths, help_paths


def _path_exists(repo_root: Path, rel_path: str) -> bool:
    if not rel_path:
        return False
    return (repo_root / rel_path).exists()


def _build_mapping_rows(
    repo_root: Path,
    matlab_root: Path,
    class_paths: Iterable[Path],
    function_paths: Iterable[Path],
    help_paths: Iterable[Path],
    notebook_topics: dict[str, str],
) -> list[MappingRow]:
    rows: list[MappingRow] = []

    for path in class_paths:
        name = path.stem
        py_path = CLASS_PATH_OVERRIDES.get(name, "")
        status = "mapped" if _path_exists(repo_root, py_path) else "missing"
        notes = "core class mapping"
        rows.append(
            MappingRow(
                matlab_path=str(path),
                entity_name=name,
                entity_type="class",
                python_path=py_path,
                status=status,
                notes=notes,
            )
        )

    for path in function_paths:
        name = path.stem
        py_path = FUNCTION_PATH_OVERRIDES.get(name, "")
        status = "mapped" if _path_exists(repo_root, py_path) else "missing"
        rows.append(
            MappingRow(
                matlab_path=str(path),
                entity_name=name,
                entity_type="function",
                python_path=py_path,
                status=status,
                notes="core function mapping",
            )
        )

    for path in help_paths:
        topic = path.stem
        py_path = notebook_topics.get(topic, "")
        status = "mapped" if _path_exists(repo_root, py_path) else "missing"
        entity_type = "help_mlx" if path.suffix.lower() == ".mlx" else "help_m"
        rows.append(
            MappingRow(
                matlab_path=str(path),
                entity_name=topic,
                entity_type=entity_type,
                python_path=py_path,
                status=status,
                notes="help source to notebook mapping",
            )
        )

    rows.sort(key=lambda row: (row.entity_type, row.entity_name.lower(), row.matlab_path.lower()))
    return rows


def _summary(rows: list[MappingRow], python_entities: dict[str, dict[str, list[str]]]) -> dict[str, object]:
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_type.setdefault(row.entity_type, {"mapped": 0, "missing": 0, "needs_review": 0, "total": 0})
        bucket["total"] += 1
        bucket[row.status] = bucket.get(row.status, 0) + 1

    core_missing = [
        row.entity_name
        for row in rows
        if row.entity_type in {"class", "function"} and row.status == "missing"
    ]
    help_missing = [row.entity_name for row in rows if row.entity_type.startswith("help_") and row.status != "mapped"]

    return {
        "total_entities": len(rows),
        "by_type": by_type,
        "core_missing": sorted(set(core_missing)),
        "help_missing": sorted(set(help_missing)),
        "python_modules_indexed": len(python_entities),
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ref_cfg = (repo_root / args.reference_config).resolve()
    matlab_root = _resolve_matlab_root(repo_root, ref_cfg)

    class_paths, function_paths, help_paths = _detect_matlab_entities(matlab_root)
    notebook_topics = _notebook_topic_map((repo_root / args.notebook_manifest).resolve())
    python_entities = _iter_python_entities((repo_root / "nstat").resolve())

    rows = _build_mapping_rows(
        repo_root=repo_root,
        matlab_root=matlab_root,
        class_paths=class_paths,
        function_paths=function_paths,
        help_paths=help_paths,
        notebook_topics=notebook_topics,
    )

    mapping_payload = {
        "version": 1,
        "matlab_root": str(matlab_root),
        "python_root": str((repo_root / "nstat").resolve()),
        "entries": [asdict(row) for row in rows],
    }
    inventory_payload = {
        "version": 1,
        "matlab": {
            "classes": [str(path) for path in class_paths],
            "functions": [str(path) for path in function_paths],
            "help_sources": [str(path) for path in help_paths],
        },
        "python": {
            "root": str((repo_root / "nstat").resolve()),
            "entities": python_entities,
        },
        "mapping": [asdict(row) for row in rows],
    }
    summary_payload = _summary(rows, python_entities)

    out_mapping = (repo_root / args.out_mapping).resolve()
    out_inventory = (repo_root / args.out_inventory).resolve()
    out_summary = (repo_root / args.out_summary).resolve()
    out_mapping.parent.mkdir(parents=True, exist_ok=True)
    out_inventory.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    out_mapping.write_text(yaml.safe_dump(mapping_payload, sort_keys=False), encoding="utf-8")
    out_inventory.write_text(json.dumps(inventory_payload, indent=2), encoding="utf-8")
    out_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "matlab_root": str(matlab_root),
                "entries": len(rows),
                "core_missing": summary_payload["core_missing"],
                "help_missing": summary_payload["help_missing"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
