#!/usr/bin/env python3
"""Generate MATLAB and Python API inventories used for parity analysis."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


FUNCTION_RE = re.compile(
    r"^\s*function\b(?:\s+\[[^\]]*\]\s*=|\s+[^=\n\r]+\s*=)?\s*([A-Za-z]\w*)\s*\(",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-mapping",
        type=Path,
        default=Path("parity/method_mapping.yaml"),
        help="Method mapping YAML",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="nSTAT-python repository root",
    )
    parser.add_argument(
        "--matlab-root",
        type=Path,
        default=Path(""),
        help="MATLAB nSTAT repository root (required)",
    )
    parser.add_argument(
        "--matlab-out",
        type=Path,
        default=Path("parity/matlab_api_inventory.json"),
        help="Output path for MATLAB API inventory",
    )
    parser.add_argument(
        "--python-out",
        type=Path,
        default=Path("parity/python_api_inventory.json"),
        help="Output path for Python API inventory",
    )
    return parser.parse_args()


def _public_python_members(obj: Any) -> dict[str, list[str]]:
    methods: list[str] = []
    properties: list[str] = []
    fields: list[str] = []

    for name, member in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        static_member = inspect.getattr_static(obj, name)
        if isinstance(static_member, property):
            properties.append(name)
            continue
        if inspect.isroutine(member):
            methods.append(name)

    dataclass_fields = getattr(obj, "__dataclass_fields__", {})
    fields.extend(sorted(name for name in dataclass_fields if not name.startswith("_")))

    return {
        "methods": sorted(set(methods)),
        "properties": sorted(set(properties)),
        "fields": sorted(set(fields)),
    }


def _parse_matlab_methods(matlab_file: Path) -> list[str]:
    if not matlab_file.exists():
        return []
    names: list[str] = []
    for line in matlab_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = FUNCTION_RE.match(line)
        if match:
            names.append(match.group(1))
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _resolve_module_head(repo_root: Path) -> str:
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return str(src_root)


def _load_mapping(mapping_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    return payload["classes"]


def build_matlab_inventory(classes: list[dict[str, Any]], matlab_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in classes:
        matlab_class = str(row["matlab_class"])
        matlab_file = matlab_root / str(row.get("matlab_file", f"{matlab_class}.m"))
        methods = _parse_matlab_methods(matlab_file)
        records.append(
            {
                "matlab_class": matlab_class,
                "matlab_file": str(matlab_file),
                "method_count": len(methods),
                "methods": methods,
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matlab_root": str(matlab_root),
        "classes": records,
    }


def build_python_inventory(classes: list[dict[str, Any]], repo_root: Path) -> dict[str, Any]:
    _resolve_module_head(repo_root)

    records: list[dict[str, Any]] = []
    for row in classes:
        python_class = str(row["python_class"])
        compat_class = str(row["compat_class"])

        p_module, p_attr = python_class.rsplit(".", 1)
        py_obj = getattr(importlib.import_module(p_module), p_attr)
        py_members = _public_python_members(py_obj)

        c_module, c_attr = compat_class.rsplit(".", 1)
        compat_obj = getattr(importlib.import_module(c_module), c_attr)
        compat_members = _public_python_members(compat_obj)

        records.append(
            {
                "matlab_class": str(row["matlab_class"]),
                "python_class": python_class,
                "compat_class": compat_class,
                "python": {
                    "method_count": len(py_members["methods"]),
                    "methods": py_members["methods"],
                    "properties": py_members["properties"],
                    "fields": py_members["fields"],
                },
                "compat": {
                    "method_count": len(compat_members["methods"]),
                    "methods": compat_members["methods"],
                    "properties": compat_members["properties"],
                    "fields": compat_members["fields"],
                },
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_root": str(repo_root),
        "classes": records,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    mapping = args.method_mapping.resolve()

    matlab_root = args.matlab_root.resolve() if str(args.matlab_root) else Path()
    if not str(matlab_root) or not matlab_root.exists():
        raise FileNotFoundError(
            "--matlab-root is required and must point to the MATLAB nSTAT repository"
        )

    classes = _load_mapping(mapping)

    matlab_inventory = build_matlab_inventory(classes=classes, matlab_root=matlab_root)
    python_inventory = build_python_inventory(classes=classes, repo_root=repo_root)

    _write_json((repo_root / args.matlab_out).resolve(), matlab_inventory)
    _write_json((repo_root / args.python_out).resolve(), python_inventory)

    print(f"Wrote MATLAB inventory to {(repo_root / args.matlab_out).resolve()}")
    print(f"Wrote Python inventory to {(repo_root / args.python_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
