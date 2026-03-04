#!/usr/bin/env python3
"""Generate class-level MATLAB/Python equivalence inventory and summary report."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CLASSDEF_RE = re.compile(r"^\s*classdef\s+([A-Za-z]\w*)(?:\s*<\s*([^\n%]+))?", flags=re.IGNORECASE)
METHODS_BLOCK_RE = re.compile(r"^\s*methods(?:\s*\(([^)]*)\))?\s*$", flags=re.IGNORECASE)
PROPERTIES_BLOCK_RE = re.compile(r"^\s*properties(?:\s*\(([^)]*)\))?\s*$", flags=re.IGNORECASE)
FUNCTION_RE = re.compile(
    r"^\s*function\b(?:\s+\[[^\]]*\]\s*=|\s+[^=\n\r]+\s*=)?\s*([A-Za-z]\w*)\s*(?:\(([^)]*)\))?",
    flags=re.IGNORECASE,
)
PROPERTY_RE = re.compile(r"^\s*([A-Za-z]\w*)")
ACCESS_RE = re.compile(r"access\s*=\s*([A-Za-z]+)", flags=re.IGNORECASE)


REQUIRED_CLASSES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--matlab-root", type=Path, required=True)
    parser.add_argument("--method-mapping", type=Path, default=Path("parity/method_mapping.yaml"))
    parser.add_argument("--method-exclusions", type=Path, default=Path("parity/method_exclusions.yml"))
    parser.add_argument("--class-contracts", type=Path, default=Path("parity/class_contracts.yml"))
    parser.add_argument("--fixture-spec", type=Path, default=Path("parity/class_fixture_export_spec.yml"))
    parser.add_argument(
        "--behavior-contracts",
        type=Path,
        action="append",
        default=[Path("tests/parity/class_behavior_specs.yml"), Path("tests/parity/compat_behavior_specs.yml")],
    )
    parser.add_argument("--out-inventory", type=Path, default=Path("parity/class_equivalence_inventory.json"))
    parser.add_argument("--out-report", type=Path, default=Path("parity/class_equivalence_report.json"))
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _signature_text(callable_obj: Any) -> str:
    try:
        return str(inspect.signature(callable_obj))
    except (TypeError, ValueError):
        return "(unavailable)"


def _public_python_surface(obj: Any) -> dict[str, Any]:
    methods: dict[str, str] = {}
    properties: list[str] = []
    fields: list[str] = []

    for name, member in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        raw = inspect.getattr_static(obj, name)
        if isinstance(raw, property):
            properties.append(name)
            continue
        if inspect.isroutine(member):
            methods[name] = _signature_text(member)

    dataclass_fields = getattr(obj, "__dataclass_fields__", {})
    for name in sorted(dataclass_fields):
        if not name.startswith("_"):
            fields.append(name)

    init_sig = "(unavailable)"
    if hasattr(obj, "__init__"):
        init_sig = _signature_text(obj.__init__)

    return {
        "constructor_signature": init_sig,
        "methods": dict(sorted(methods.items())),
        "properties": sorted(set(properties)),
        "fields": sorted(set(fields)),
    }


def _parse_attrs(attrs: str | None) -> tuple[str, bool]:
    if not attrs:
        return "public", False
    access_match = ACCESS_RE.search(attrs)
    access = access_match.group(1).lower() if access_match else "public"
    is_static = "static" in attrs.lower()
    return access, is_static


def _parse_matlab_class(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    class_name = path.stem
    superclass = ""
    class_match = None
    for line in lines:
        class_match = CLASSDEF_RE.match(line)
        if class_match:
            class_name = class_match.group(1)
            superclass = (class_match.group(2) or "").strip()
            break

    properties: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []

    in_properties = False
    current_prop_access = "public"
    current_methods_access = "public"
    current_methods_static = False
    in_methods = False

    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue

        pm = PROPERTIES_BLOCK_RE.match(line)
        if pm:
            in_properties = True
            in_methods = False
            current_prop_access, _ = _parse_attrs(pm.group(1))
            continue

        mm = METHODS_BLOCK_RE.match(line)
        if mm:
            in_methods = True
            in_properties = False
            current_methods_access, current_methods_static = _parse_attrs(mm.group(1))
            continue

        if stripped == "end":
            if in_properties:
                in_properties = False
            # Keep method block mode sticky across nested function `end`.
            continue

        if in_properties:
            prop_match = PROPERTY_RE.match(line)
            if prop_match:
                prop_name = prop_match.group(1)
                if prop_name.lower() not in {"properties", "methods", "classdef", "end"}:
                    properties.append(
                        {
                            "name": prop_name,
                            "access": current_prop_access,
                            "line": idx,
                        }
                    )
            continue

        func_match = FUNCTION_RE.match(line)
        if func_match and in_methods:
            name = func_match.group(1)
            args_raw = (func_match.group(2) or "").strip()
            args = [a.strip() for a in args_raw.split(",") if a.strip()] if args_raw else []
            methods.append(
                {
                    "name": name,
                    "args": args,
                    "line": idx,
                    "access": current_methods_access,
                    "static": current_methods_static,
                }
            )

    constructor = next((m for m in methods if m["name"] == class_name), None)
    public_methods = [m["name"] for m in methods if m["access"] not in {"private"}]
    private_methods = [m["name"] for m in methods if m["access"] == "private"]

    return {
        "matlab_class": class_name,
        "matlab_file": str(path),
        "superclass": superclass,
        "constructor": constructor or {"name": class_name, "args": [], "line": None, "access": "public", "static": False},
        "properties": properties,
        "methods": methods,
        "public_methods": sorted(dict.fromkeys(public_methods)),
        "private_methods": sorted(dict.fromkeys(private_methods)),
    }


def _contract_members(paths: list[Path]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            continue
        payload = _load_yaml(path)
        for cls in payload.get("classes", []):
            name = str(cls["matlab_class"])
            for contract in cls.get("contracts", []):
                out[name].add(str(contract["member"]))
    return out


def _fixture_lookup(fixture_spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in fixture_spec.get("classes", []):
        out[str(row["matlab_class"])] = {
            "fixture_path": str(row["fixture_path"]),
            "generator": str(row["generator"]),
        }
    return out


def _excluded_methods(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    payload = _load_yaml(path)
    out: dict[str, set[str]] = {}
    for row in payload.get("classes", []):
        cls = str(row.get("matlab_class", ""))
        if not cls:
            continue
        out[cls] = {str(method) for method in row.get("methods", [])}
    return out


def _load_python_class(path: str) -> Any:
    module, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def build_inventory(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    matlab_root = args.matlab_root.resolve()
    mapping = _load_yaml((repo_root / args.method_mapping).resolve())
    class_contracts = _load_yaml((repo_root / args.class_contracts).resolve())
    fixture_spec = _load_yaml((repo_root / args.fixture_spec).resolve())
    exclusions = _excluded_methods((repo_root / args.method_exclusions).resolve())
    behavior_contracts = _contract_members([(repo_root / p).resolve() for p in args.behavior_contracts])

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    matlab_files = sorted(matlab_root.glob("*.m"))
    matlab_classes = {}
    for f in matlab_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        if CLASSDEF_RE.search(text):
            entry = _parse_matlab_class(f)
            matlab_classes[entry["matlab_class"]] = entry

    mapping_rows = {str(row["matlab_class"]): row for row in mapping.get("classes", [])}
    contract_rows = {str(row["matlab_class"]): row for row in class_contracts.get("classes", [])}
    fixture_rows = _fixture_lookup(fixture_spec)

    class_rows: list[dict[str, Any]] = []
    missing_classes: list[str] = []
    missing_mappings: list[str] = []
    missing_fixtures: list[str] = []
    missing_method_symbols: dict[str, list[str]] = {}

    for matlab_class in sorted(matlab_classes):
        matlab_meta = matlab_classes[matlab_class]
        map_row = mapping_rows.get(matlab_class)
        contract_row = contract_rows.get(matlab_class)
        fixture_row = fixture_rows.get(matlab_class)

        if map_row is None:
            missing_mappings.append(matlab_class)
            class_rows.append(
                {
                    "matlab_class": matlab_class,
                    "status": "gap_missing_mapping",
                    "matlab": matlab_meta,
                }
            )
            continue

        python_class_path = str(map_row["python_class"])
        compat_class_path = str(map_row["compat_class"])
        alias_methods = {str(k): str(v) for k, v in map_row.get("alias_methods", {}).items()}

        py_cls = _load_python_class(python_class_path)
        compat_cls = _load_python_class(compat_class_path)
        py_surface = _public_python_surface(py_cls)
        compat_surface = _public_python_surface(compat_cls)

        py_members = set(py_surface["methods"]).union(py_surface["properties"]).union(py_surface["fields"])
        compat_members = set(compat_surface["methods"]).union(compat_surface["properties"]).union(compat_surface["fields"])

        method_rows: list[dict[str, Any]] = []
        missing_for_class: list[str] = []
        for matlab_method in matlab_meta["public_methods"]:
            mapped = alias_methods.get(matlab_method, matlab_method)
            in_python = mapped in py_members
            in_compat = mapped in compat_members
            symbol_ok = in_python or in_compat
            is_excluded = matlab_method in exclusions.get(matlab_class, set())
            if (not symbol_ok) and (not is_excluded):
                missing_for_class.append(matlab_method)
            method_rows.append(
                {
                    "matlab_method": matlab_method,
                    "mapped_member": mapped,
                    "mapped_via_alias": matlab_method in alias_methods,
                    "present_in_python_surface": in_python,
                    "present_in_compat_surface": in_compat,
                    "excluded_method": is_excluded,
                    "covered_by_behavior_contract": mapped in behavior_contracts.get(matlab_class, set()),
                    "covered_by_class_contract": mapped in set(contract_row.get("key_methods", [])) if contract_row else False,
                }
            )

        if missing_for_class:
            missing_method_symbols[matlab_class] = missing_for_class

        fixture_path = ""
        fixture_exists = False
        fixture_generator = ""
        if fixture_row:
            fixture_path = fixture_row["fixture_path"]
            fixture_generator = fixture_row["generator"]
            fixture_exists = (repo_root / fixture_path).exists()
            if not fixture_exists:
                missing_fixtures.append(matlab_class)
        else:
            missing_fixtures.append(matlab_class)

        constructor_contract = {}
        if contract_row is not None:
            constructor_contract = {
                "python_class": str(contract_row.get("python_class", "")),
                "compat_class": str(contract_row.get("compat_class", "")),
                "fixture_path": str(contract_row.get("fixture_path", "")),
                "key_methods": list(contract_row.get("key_methods", [])),
            }

        status = "verified"
        if missing_for_class:
            status = "partial_missing_method_symbols"
        if not fixture_exists:
            status = "partial_missing_fixture"

        class_rows.append(
            {
                "matlab_class": matlab_class,
                "status": status,
                "matlab": matlab_meta,
                "mapping": {
                    "python_class": python_class_path,
                    "compat_class": compat_class_path,
                    "alias_methods": alias_methods,
                },
                "python": py_surface,
                "compat": compat_surface,
                "fixture": {
                    "path": fixture_path,
                    "exists": fixture_exists,
                    "generator": fixture_generator,
                },
                "class_contract": constructor_contract,
                "method_rows": method_rows,
            }
        )

    for required in REQUIRED_CLASSES:
        if required not in matlab_classes:
            missing_classes.append(required)

    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matlab_root": str(matlab_root),
        "python_root": str(repo_root),
        "required_classes": REQUIRED_CLASSES,
        "class_rows": class_rows,
        "summary": {
            "total_matlab_classes": len(matlab_classes),
            "required_classes_missing_from_matlab_scan": sorted(missing_classes),
            "classes_missing_mapping": sorted(missing_mappings),
            "classes_missing_fixtures": sorted(missing_fixtures),
            "classes_with_missing_method_symbols": sorted(missing_method_symbols.keys()),
        },
    }

    verified = [row for row in class_rows if row["status"] == "verified"]
    partial = [row for row in class_rows if row["status"].startswith("partial_")]
    gaps = [row for row in class_rows if row["status"].startswith("gap_")]

    top_methods = {}
    for row in class_rows:
        cls = row["matlab_class"]
        key_methods = row.get("class_contract", {}).get("key_methods", []) or []
        if key_methods:
            top_methods[cls] = key_methods[:10]
        else:
            top_methods[cls] = [m["matlab_method"] for m in row["method_rows"][:10]]

    remaining_issues: list[dict[str, Any]] = []
    for cls in sorted(missing_method_symbols):
        remaining_issues.append(
            {
                "matlab_class": cls,
                "issue": "missing_method_symbol",
                "missing_methods": sorted(missing_method_symbols[cls]),
            }
        )
    for cls in sorted(missing_fixtures):
        remaining_issues.append(
            {
                "matlab_class": cls,
                "issue": "missing_fixture",
                "repro": f"python tools/parity/export_matlab_gold_fixtures.py --matlab-root {matlab_root}",
            }
        )
    for cls in sorted(missing_mappings):
        remaining_issues.append(
            {
                "matlab_class": cls,
                "issue": "missing_mapping",
                "repro": "Update parity/method_mapping.yaml with python_class/compat_class mapping.",
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "verified_classes": len(verified),
            "partial_classes": len(partial),
            "gap_classes": len(gaps),
            "total_classes": len(class_rows),
            "required_class_coverage_ok": len(missing_classes) == 0,
        },
        "top_critical_methods_tested": top_methods,
        "remaining_issues": remaining_issues,
    }
    return inventory, report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    inventory, report = build_inventory(args)

    repo_root = args.repo_root.resolve()
    out_inventory = (repo_root / args.out_inventory).resolve()
    out_report = (repo_root / args.out_report).resolve()
    _write_json(out_inventory, inventory)
    _write_json(out_report, report)

    print(f"Wrote class inventory: {out_inventory}")
    print(f"Wrote class report:    {out_report}")
    print(
        "Summary: "
        f"{report['summary']['verified_classes']} verified, "
        f"{report['summary']['partial_classes']} partial, "
        f"{report['summary']['gap_classes']} gaps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
