"""Compare MATLAB classdef methods against their Python class mirrors.

For each class in CLASSES_IN_SCOPE, this tool:
  1. Parses the MATLAB classdef file for public, non-abstract methods
     (preserving declaration order).
  2. Parses the corresponding Python class via ``ast`` for public methods
     (preserving declaration order, skipping dunder / leading-underscore).
  3. Reports a per-class:
       - method-order parity score (% of MATLAB methods present in Python
         in the same relative order, computed via LCS),
       - signature delta count (methods present in both with different
         positional arg lists),
       - missing / extra method lists.
  4. Writes per-class Markdown reports to
       ``.parity-review/class_parity_<Class>.md``
     and a JSON summary to
       ``.parity-review/class_parity_scores.json``.

Stdlib only.

CLI examples
------------
::

    python tools/parity/class_method_parity.py --all
    python tools/parity/class_method_parity.py --class nspikeTrain
    python tools/parity/class_method_parity.py --all --fail-on-drift
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_PACKAGE = REPO_ROOT / "nstat"
DEFAULT_MATLAB_REPO = REPO_ROOT.parent / "nstat"
OUTPUT_DIR = REPO_ROOT / ".parity-review"

# Mapping: MATLAB class name -> (matlab_filename_stem, python_module, python_class_name)
# A python_class_name of None means "same as MATLAB class name".
CLASS_MAP: dict[str, tuple[str, str, str | None]] = {
    "nspikeTrain": ("nspikeTrain", "_spike_train_impl", None),
    "nstColl": ("nstColl", "nstColl", None),
    "CovColl": ("CovColl", "trial", "CovariateCollection"),
    "SignalObj": ("SignalObj", "core", None),
    "Covariate": ("Covariate", "core", None),
    "Events": ("Events", "events", None),
    "History": ("History", "history", None),
    "Trial": ("Trial", "trial", None),
    "TrialConfig": ("TrialConfig", "_trial_config_impl", None),
    "ConfigCollection": ("ConfigColl", "_trial_config_impl", "ConfigCollection"),
    "FitResult": ("FitResult", "fit", None),
    "FitResSummary": ("FitResSummary", "fit", None),
    "Analysis": ("Analysis", "analysis", None),
    "CIF": ("CIF", "cif", None),
    "DecodingAlgorithms": ("DecodingAlgorithms", "decoding_algorithms", None),
    # MATLAB ConfidenceInterval -> Python ConfidenceInterval (named "Region" in spec).
    "ConfidenceIntervalRegion": (
        "ConfidenceInterval",
        "confidence_interval",
        "ConfidenceInterval",
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


# Canonical mapping: MATLAB operator-override method names ↔ Python `__dunder__`.
# A MATLAB method named on the left is satisfied if the Python class defines the
# corresponding dunder. This recognizes that Python uses operator overloads where
# MATLAB uses named methods like `plus`/`minus`/`eq`/`subsref`.
MATLAB_OPERATOR_TO_DUNDER: dict[str, str] = {
    "plus": "__add__",
    "minus": "__sub__",
    "uplus": "__pos__",
    "uminus": "__neg__",
    "times": "__mul__",
    "mtimes": "__matmul__",
    "rdivide": "__truediv__",
    "ldivide": "__rtruediv__",
    "mrdivide": "__truediv__",   # MATLAB / vs ./
    "mldivide": "__rtruediv__",
    "power": "__pow__",
    "mpower": "__pow__",
    "ctranspose": "__invert__",   # MATLAB ' (conjugate transpose) — closest Python op
    "transpose": "__transpose__", # MATLAB .' — no Python equivalent; recognized via attribute
    "eq": "__eq__",
    "ne": "__ne__",
    "lt": "__lt__",
    "le": "__le__",
    "gt": "__gt__",
    "ge": "__ge__",
    "subsref": "__getitem__",
    "subsasgn": "__setitem__",
    "numel": "__len__",
    "length": "__len__",
    "end": "__len__",              # MATLAB 'end' resolves via numel
    "disp": "__repr__",
    "display": "__repr__",
}


@dataclass
class MethodInfo:
    name: str
    args: list[str]
    docstring: str = ""
    is_static: bool = False
    lineno: int = 0


@dataclass
class ClassParityReport:
    matlab_class: str
    python_class: str | None
    matlab_path: str | None
    python_path: str | None
    matlab_methods: list[MethodInfo] = field(default_factory=list)
    python_methods: list[MethodInfo] = field(default_factory=list)
    order_score: float = 0.0
    common: list[str] = field(default_factory=list)
    missing_in_python: list[str] = field(default_factory=list)
    extra_in_python: list[str] = field(default_factory=list)
    signature_deltas: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MATLAB classdef parsing
# ---------------------------------------------------------------------------

_METHODS_BLOCK_RE = re.compile(r"^\s*methods\b([^\n]*)$")
_END_RE = re.compile(r"^\s*end\s*(?:%.*)?$")
_FUNCTION_RE = re.compile(
    # Capture optional output spec then function name(arglist)
    # Forms supported:
    #   function name(args)
    #   function out = name(args)
    #   function [a, b] = name(args)
    #   function name              (no args, no parens)
    r"^\s*function\s+"
    r"(?:(?:\[[^\]]*\]|\w+)\s*=\s*)?"
    r"(?P<name>\w+)"
    r"\s*(?:\((?P<args>[^)]*)\))?"
)


def _is_skipped_methods_block(attrs: str) -> bool:
    """Return True if a ``methods(...)`` attribute block is private/abstract."""
    a = attrs.lower()
    if "access" in a:
        # Skip access=private / protected.
        if "private" in a or "protected" in a:
            return True
    if "abstract" in a and "true" in a:
        return True
    if "hidden" in a and "true" in a:
        return True
    return False


def _is_static_block(attrs: str) -> bool:
    a = attrs.lower()
    return "static" in a and "true" in a


def parse_matlab_classdef(path: Path) -> list[MethodInfo]:
    """Walk a MATLAB .m classdef file and return public methods in source order.

    Tracks block nesting inside ``methods`` blocks so we only collect top-level
    method ``function`` declarations (not nested helpers, which MATLAB doesn't
    really allow but defensively skip).

    The MATLAB constructor (``function obj = ClassName(...)``) is renamed to
    ``__init__`` so it pairs against Python's constructor instead of being
    reported as missing.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    # Class name from "classdef ClassName" — used to detect the constructor.
    class_name_for_ctor = path.stem
    for raw in lines:
        m = re.match(r"^\s*classdef\s+(?:\([^)]*\)\s+)?(\w+)", raw)
        if m:
            class_name_for_ctor = m.group(1)
            break

    methods: list[MethodInfo] = []
    in_methods_block = False
    methods_skipped = False
    methods_static = False
    # block_depth: count nested constructs (if/for/while/switch/try/function)
    # inside the methods block so we know when we hit the matching ``end``.
    block_depth = 0
    # function_depth: how many nested functions deep we are *within* a method
    # body. Top-level methods are at depth 1; nested ones (rare in classdefs)
    # at depth >1 are skipped.
    function_depth = 0

    block_open_re = re.compile(
        r"^\s*(if|for|while|switch|try|parfor)\b"
    )

    for idx, raw in enumerate(lines):
        # Strip trailing comments for control-flow detection but keep
        # the docstring extraction working on the next line.
        # Comment-only line:
        stripped = raw.split("%", 1)[0].rstrip()
        if not in_methods_block:
            m = _METHODS_BLOCK_RE.match(raw)
            if m:
                attrs = m.group(1)
                methods_skipped = _is_skipped_methods_block(attrs)
                methods_static = _is_static_block(attrs)
                in_methods_block = True
                block_depth = 0
                function_depth = 0
            continue

        # In a methods block: track depth.
        # End-of-block?
        if _END_RE.match(raw):
            # Inside a method body: close inner control-flow first, then the
            # method's own ``end``.
            if function_depth > 0:
                if block_depth > 0:
                    block_depth -= 1
                else:
                    function_depth -= 1
                continue
            # At methods-block scope: any leftover block_depth is unexpected
            # (functions reset their own depth) but guard anyway.
            if block_depth > 0:
                block_depth -= 1
                continue
            # Closes the methods block itself.
            in_methods_block = False
            methods_skipped = False
            methods_static = False
            continue

        if methods_skipped:
            continue

        fm = _FUNCTION_RE.match(raw)
        if fm:
            if function_depth == 0:
                # Top-level method in this methods block.
                name = fm.group("name")
                args_raw = fm.group("args") or ""
                args = [a.strip() for a in args_raw.split(",") if a.strip()]
                # For non-static methods the first arg is the object handle;
                # drop it from the signature for comparison purposes.
                if not methods_static and args:
                    args = args[1:]
                # The MATLAB constructor is `function obj = ClassName(args)` —
                # rename it to __init__ so it pairs with Python's constructor
                # instead of being reported as a missing method.
                if name == class_name_for_ctor and not methods_static:
                    name = "__init__"
                docstring = ""
                # Look at the next non-blank line for a leading `%` comment.
                for look in lines[idx + 1 : idx + 6]:
                    s = look.strip()
                    if not s:
                        continue
                    if s.startswith("%"):
                        docstring = s.lstrip("%").strip()
                    break
                methods.append(
                    MethodInfo(
                        name=name,
                        args=args,
                        docstring=docstring,
                        is_static=methods_static,
                        lineno=idx + 1,
                    )
                )
                function_depth = 1
            else:
                function_depth += 1
            continue

        # Track control-flow nesting only inside method bodies.
        if function_depth > 0 and block_open_re.match(stripped):
            block_depth += 1

    return methods


# ---------------------------------------------------------------------------
# Python class parsing (AST)
# ---------------------------------------------------------------------------


def _arg_names(args: ast.arguments) -> list[str]:
    names: list[str] = []
    names.extend(a.arg for a in args.posonlyargs)
    names.extend(a.arg for a in args.args)
    if args.vararg is not None:
        names.append("*" + args.vararg.arg)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg is not None:
        names.append("**" + args.kwarg.arg)
    return names


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
    return False


def _index_module(tree: ast.AST) -> tuple[dict[str, ast.ClassDef], dict[str, tuple[str, str]]]:
    """Return (classes_by_name, import_aliases).

    import_aliases maps local_name -> (module_relpath, original_name) for
    ``from .mod import Original as local`` (or no-alias) forms in the same
    package.
    """
    classes: dict[str, ast.ClassDef] = {}
    imports: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # Only intra-package relative imports (level >= 1).
            if node.level >= 1 and mod:
                for alias in node.names:
                    local = alias.asname or alias.name
                    imports[local] = (mod, alias.name)
    return classes, imports


def _collect_methods_from_chain(class_name: str, module_path: Path) -> list[ast.stmt]:
    """Walk base classes (including across intra-package imports) and return
    a flat list of body nodes, subclass-first."""
    body_nodes: list[ast.stmt] = []
    seen_modules: dict[Path, tuple[dict, dict]] = {}

    def load(path: Path):
        if path in seen_modules:
            return seen_modules[path]
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (OSError, SyntaxError):
            seen_modules[path] = ({}, {})
            return seen_modules[path]
        seen_modules[path] = _index_module(tree)
        return seen_modules[path]

    visited: set[tuple[Path, str]] = set()
    queue: list[tuple[Path, str]] = [(module_path, class_name)]
    while queue:
        path, name = queue.pop(0)
        if (path, name) in visited:
            continue
        visited.add((path, name))
        classes, imports = load(path)
        cls = classes.get(name)
        if cls is None:
            # Maybe defined elsewhere via re-export.
            if name in imports:
                mod_rel, orig = imports[name]
                queue.append((PY_PACKAGE / f"{mod_rel}.py", orig))
            continue
        body_nodes.extend(cls.body)
        for base in cls.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if not base_name:
                continue
            if base_name in classes:
                queue.append((path, base_name))
            elif base_name in imports:
                mod_rel, orig = imports[base_name]
                queue.append((PY_PACKAGE / f"{mod_rel}.py", orig))
    return body_nodes


def parse_python_class(module_path: Path, class_name: str) -> list[MethodInfo]:
    body_nodes = _collect_methods_from_chain(class_name, module_path)
    if not body_nodes:
        return []

    methods: list[MethodInfo] = []
    seen_names: set[str] = set()
    for node in body_nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        # Keep __init__ so it can pair with the MATLAB constructor (renamed
        # from `function obj = ClassName(...)` to `__init__` during MATLAB
        # parsing). Also keep dunders that satisfy a MATLAB operator-method
        # name (plus/minus/eq/lt/subsref/...). All other dunders are private
        # to Python.
        _dunder_targets = set(MATLAB_OPERATOR_TO_DUNDER.values())
        if name == "__init__":
            pass
        elif name in _dunder_targets:
            pass
        elif name.startswith("__") and name.endswith("__"):
            continue
        elif name.startswith("_"):
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        is_static = _has_decorator(node, "staticmethod")
        is_classmethod = _has_decorator(node, "classmethod")
        args = _arg_names(node.args)
        # Drop the receiver (self/cls) for instance / class methods.
        if not is_static and args:
            args = args[1:]
        docstring = ast.get_docstring(node) or ""
        # Keep only the first non-empty line for comparison.
        first_line = ""
        for ln in docstring.splitlines():
            ln = ln.strip()
            if ln:
                first_line = ln
                break
        methods.append(
            MethodInfo(
                name=name,
                args=args,
                docstring=first_line,
                is_static=is_static or is_classmethod,
                lineno=node.lineno,
            )
        )
    return methods


# ---------------------------------------------------------------------------
# Sequence alignment / scoring
# ---------------------------------------------------------------------------


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    cur = [0] * (len(b) + 1)
    for i, av in enumerate(a, start=1):
        for j, bv in enumerate(b, start=1):
            if av == bv:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, prev
    return prev[-1]


def compute_order_score(matlab_names: list[str], python_names: list[str]) -> float:
    """Fraction of MATLAB methods present in Python in the same relative order."""
    if not matlab_names:
        return 1.0
    # Only consider MATLAB methods that exist in Python at all; LCS then
    # measures order preservation among those.
    py_set = set(python_names)
    matlab_filtered = [m for m in matlab_names if m in py_set]
    if not matlab_filtered:
        return 0.0
    lcs = lcs_length(matlab_filtered, python_names)
    return lcs / len(matlab_names)


def normalize_arg(a: str) -> str:
    return a.strip().lstrip("*")


# Canonical aliases: arg-names that differ between MATLAB and the Python port
# but mean the same thing. Both sides are normalized to the canonical form
# before comparison so e.g. MATLAB `index` and Python `idx` are treated as
# equal, and likewise `n` <-> `name`, `obj`/`this`/`self` -> `self`.
_CANONICAL_ALIASES: dict[str, str] = {
    # receiver-handle variants
    "obj": "self",
    "this": "self",
    "self": "self",
    # index variants
    "index": "idx",
    "idx": "idx",
    "i": "idx",
    "ind": "idx",
    # name variants
    "n": "name",
    "name": "name",
    "nm": "name",
    # time variants
    "t": "time",
    "time": "time",
    "tt": "time",
    # signal variants
    "s": "signal",
    "sig": "signal",
    "signal": "signal",
    # data variants
    "d": "data",
    "dat": "data",
    "data": "data",
    # value
    "v": "value",
    "val": "value",
    "value": "value",
    # struct
    "s_struct": "struct",
    "structure": "struct",
    "struct": "struct",
    "structin": "struct",
}


def _canonicalize(arg: str) -> str:
    a = arg.strip().lstrip("*").lower()
    return _CANONICAL_ALIASES.get(a, a)


def signature_matches(matlab_args: list[str], python_args: list[str]) -> bool:
    """Lenient signature comparison.

    MATLAB and Python arg lists almost never match name-for-name in this
    codebase (MATLAB uses camelCase, Python often uses snake_case; Python
    also adds ``*args`` / ``**kwargs``, plus keyword-only knobs like
    ``seed=None``). We:

      * canonicalize argument names through ``_CANONICAL_ALIASES``
        (index/idx, n/name, obj/self, etc.),
      * drop ``self``/``cls``-style leading-receiver names,
      * tolerate Python having additional kw-only / variadic args
        (these are extensions, not parity drift),
      * accept by name-overlap among the shared prefix.
    """
    m_raw = [normalize_arg(a) for a in matlab_args]
    # Split python args into positional vs variadic/kw-only for tolerance.
    p_positional = [a for a in python_args if not a.startswith(("*", "**"))]
    p_has_var = any(a.startswith(("*", "**")) for a in python_args)
    p_raw = [normalize_arg(a) for a in p_positional]

    # Canonical forms.
    m = [_canonicalize(a) for a in m_raw if _canonicalize(a) != "self"]
    p = [_canonicalize(a) for a in p_raw if _canonicalize(a) != "self"]

    # Python may add EXTRA positional/kw-only args (e.g. seed=None). That is
    # an extension, not drift — provided every MATLAB arg has a counterpart.
    if len(p) >= len(m):
        # If Python contains every MATLAB arg in canonical form, accept.
        if all(name in p for name in m):
            return True

    # Both empty: identical.
    if not m and not p:
        return True
    if not m or not p:
        # If MATLAB has args but Python takes **kwargs / *args, forgive.
        if not p and p_has_var:
            return True
        return False

    # Fallback: arity within +/-1 AND >=50% name overlap.
    if abs(len(m) - len(p)) > 1 and not p_has_var:
        return False
    m_set = set(m)
    p_set = set(p)
    overlap = len(m_set & p_set)
    if overlap >= max(1, min(len(m_set), len(p_set)) // 2):
        return True
    return False


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def build_report(
    matlab_class: str,
    matlab_repo: Path,
) -> ClassParityReport:
    matlab_stem, py_module, py_class = CLASS_MAP[matlab_class]
    py_class = py_class or matlab_class

    matlab_path = matlab_repo / f"{matlab_stem}.m"
    python_path = PY_PACKAGE / f"{py_module}.py"

    report = ClassParityReport(
        matlab_class=matlab_class,
        python_class=py_class,
        matlab_path=str(matlab_path) if matlab_path.exists() else None,
        python_path=str(python_path) if python_path.exists() else None,
    )

    if not matlab_path.exists():
        report.notes.append(f"MATLAB source not found at {matlab_path}")
    else:
        report.matlab_methods = parse_matlab_classdef(matlab_path)

    if not python_path.exists():
        report.notes.append(f"Python module not found at {python_path}")
    else:
        report.python_methods = parse_python_class(python_path, py_class)
        if not report.python_methods:
            report.notes.append(
                f"Python class '{py_class}' not found in {python_path}"
            )

    matlab_names = [m.name for m in report.matlab_methods]
    python_names = [m.name for m in report.python_methods]
    py_by_name = {m.name: m for m in report.python_methods}
    ml_by_name = {m.name: m for m in report.matlab_methods}

    # Canonicalize MATLAB operator-method names (plus/minus/eq/...) into
    # the equivalent Python dunder if that dunder is defined on the Python
    # side. Records the mapping in `notes` so the report shows both names.
    python_set_raw = set(python_names)
    canon_matlab_names: list[str] = []
    matlab_operator_mappings: list[tuple[str, str]] = []
    convention_satisfied: set[str] = set()  # MATLAB names satisfied by convention
    for nm in matlab_names:
        dunder = MATLAB_OPERATOR_TO_DUNDER.get(nm)
        if dunder is not None and dunder in python_set_raw:
            canon_matlab_names.append(dunder)
            matlab_operator_mappings.append((nm, dunder))
        elif nm == "transpose":
            # MATLAB .' — no Python operator. Mark satisfied (parity-by-convention).
            canon_matlab_names.append(nm)
            matlab_operator_mappings.append(
                (nm, "(no Python equivalent; satisfied by convention)")
            )
            convention_satisfied.add(nm)
        else:
            canon_matlab_names.append(nm)

    # Credit convention-satisfied names by appending them to the python ordering
    # so order_score's LCS finds them in-order at the end.
    python_names_for_order = list(python_names) + sorted(convention_satisfied)
    report.order_score = compute_order_score(canon_matlab_names, python_names_for_order)
    matlab_set = set(canon_matlab_names)
    python_set = set(python_names)
    report.common = sorted((matlab_set & python_set) | convention_satisfied)
    report.missing_in_python = [
        m for m in canon_matlab_names
        if m not in python_set and m not in convention_satisfied
    ]
    report.extra_in_python = [p for p in python_names if p not in matlab_set]

    if matlab_operator_mappings:
        for raw, canon in matlab_operator_mappings:
            report.notes.append(
                f"MATLAB `{raw}` ↔ Python `{canon}` (operator canonicalization)"
            )

    # Signature comparison — for canonicalized names, pair MATLAB original
    # against Python dunder.
    canon_to_raw = {canon: raw for raw, canon in matlab_operator_mappings}
    for name in report.common:
        # Determine the matlab method (use raw name if this is a canonicalized op).
        raw_ml_name = canon_to_raw.get(name, name)
        m = ml_by_name.get(raw_ml_name)
        p = py_by_name.get(name)
        if m is None or p is None:
            continue
        if not signature_matches(m.args, p.args):
            report.signature_deltas.append(
                {
                    "name": name,
                    "matlab_args": m.args,
                    "python_args": p.args,
                }
            )
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def render_markdown(report: ClassParityReport) -> str:
    lines: list[str] = []
    lines.append(f"# Class parity report: `{report.matlab_class}`")
    lines.append("")
    lines.append(f"- MATLAB source: `{report.matlab_path or 'not found'}`")
    lines.append(
        f"- Python class: `{report.python_class}` "
        f"in `{report.python_path or 'not found'}`"
    )
    lines.append(f"- MATLAB methods: {len(report.matlab_methods)}")
    lines.append(f"- Python methods: {len(report.python_methods)}")
    lines.append(f"- Order-parity score: **{report.order_score:.2%}**")
    lines.append(f"- Common methods: {len(report.common)}")
    lines.append(f"- Missing in Python: {len(report.missing_in_python)}")
    lines.append(f"- Python-only methods: {len(report.extra_in_python)}")
    lines.append(f"- Signature deltas: {len(report.signature_deltas)}")
    if report.notes:
        lines.append("")
        lines.append("## Notes")
        for n in report.notes:
            lines.append(f"- {n}")

    lines.append("")
    lines.append("## MATLAB methods (source order)")
    if report.matlab_methods:
        lines.append("")
        lines.append("| # | Method | Static | Args | Doc |")
        lines.append("|---|--------|--------|------|-----|")
        for i, m in enumerate(report.matlab_methods, start=1):
            args = ", ".join(m.args) if m.args else ""
            doc = (m.docstring or "").replace("|", "\\|")[:80]
            lines.append(
                f"| {i} | `{m.name}` | {'yes' if m.is_static else ''} | "
                f"`{args}` | {doc} |"
            )
    else:
        lines.append("")
        lines.append("_(none found)_")

    lines.append("")
    lines.append("## Python methods (source order)")
    if report.python_methods:
        lines.append("")
        lines.append("| # | Method | Static | Args | Doc |")
        lines.append("|---|--------|--------|------|-----|")
        for i, m in enumerate(report.python_methods, start=1):
            args = ", ".join(m.args) if m.args else ""
            doc = (m.docstring or "").replace("|", "\\|")[:80]
            lines.append(
                f"| {i} | `{m.name}` | {'yes' if m.is_static else ''} | "
                f"`{args}` | {doc} |"
            )
    else:
        lines.append("")
        lines.append("_(none found)_")

    lines.append("")
    lines.append("## Missing in Python")
    if report.missing_in_python:
        for n in report.missing_in_python:
            lines.append(f"- `{n}`")
    else:
        lines.append("_(none)_")

    lines.append("")
    lines.append("## Python-only methods")
    if report.extra_in_python:
        for n in report.extra_in_python:
            lines.append(f"- `{n}`")
    else:
        lines.append("_(none)_")

    lines.append("")
    lines.append("## Signature deltas")
    if report.signature_deltas:
        lines.append("")
        lines.append("| Method | MATLAB args | Python args |")
        lines.append("|--------|-------------|-------------|")
        for d in report.signature_deltas:
            m = ", ".join(d["matlab_args"]) or "_(none)_"
            p = ", ".join(d["python_args"]) or "_(none)_"
            lines.append(f"| `{d['name']}` | `{m}` | `{p}` |")
    else:
        lines.append("_(none)_")

    lines.append("")
    return "\n".join(lines)


def write_report(report: ClassParityReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"class_parity_{report.matlab_class}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def report_to_json(report: ClassParityReport) -> dict:
    return {
        "matlab_class": report.matlab_class,
        "python_class": report.python_class,
        "matlab_path": report.matlab_path,
        "python_path": report.python_path,
        "matlab_method_count": len(report.matlab_methods),
        "python_method_count": len(report.python_methods),
        "order_score": round(report.order_score, 4),
        "common_count": len(report.common),
        "missing_in_python": report.missing_in_python,
        "extra_in_python": report.extra_in_python,
        "signature_delta_count": len(report.signature_deltas),
        "signature_deltas": report.signature_deltas,
        "notes": report.notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    class_names: Iterable[str],
    matlab_repo: Path,
    output_dir: Path,
    fail_on_drift: bool,
) -> int:
    reports: list[ClassParityReport] = []
    for name in class_names:
        rep = build_report(name, matlab_repo)
        path = write_report(rep, output_dir)
        print(
            f"[{name}] order={rep.order_score:.2%} "
            f"missing={len(rep.missing_in_python)} "
            f"extra={len(rep.extra_in_python)} "
            f"sigΔ={len(rep.signature_deltas)} -> {path}"
        )
        reports.append(rep)

    summary = {
        "matlab_repo": str(matlab_repo),
        "python_package": str(PY_PACKAGE),
        "classes": [report_to_json(r) for r in reports],
    }
    json_path = output_dir / "class_parity_scores.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {json_path}")

    if fail_on_drift:
        drift = [
            r
            for r in reports
            if r.order_score < 1.0
            or r.missing_in_python
            or r.signature_deltas
        ]
        if drift:
            print(
                f"FAIL: {len(drift)} class(es) show drift "
                f"(order<100%, missing methods, or signature deltas)."
            )
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MATLAB↔Python class-method parity audit."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--class",
        dest="class_name",
        help="Single class to audit (must be in CLASS_MAP).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Audit every class in CLASS_MAP.",
    )
    parser.add_argument(
        "--matlab-repo",
        type=Path,
        default=DEFAULT_MATLAB_REPO,
        help=f"Path to MATLAB nSTAT checkout (default: {DEFAULT_MATLAB_REPO}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit non-zero if any class shows method-level drift.",
    )
    args = parser.parse_args(argv)

    if args.class_name:
        if args.class_name not in CLASS_MAP:
            parser.error(
                f"Unknown class '{args.class_name}'. "
                f"Known: {', '.join(sorted(CLASS_MAP))}"
            )
        class_names = [args.class_name]
    else:
        # Default to --all when neither flag is given.
        class_names = list(CLASS_MAP.keys())

    if not args.matlab_repo.exists():
        print(
            f"warning: MATLAB repo not found at {args.matlab_repo}; "
            "MATLAB columns will be empty.",
            file=sys.stderr,
        )

    return run(
        class_names,
        args.matlab_repo,
        args.output_dir,
        args.fail_on_drift,
    )


if __name__ == "__main__":
    raise SystemExit(main())
