"""Test guard for the v0.5.0 0-based indexing convention.

Implements hard rules HR2 and HR3 from
``docs/proposals/2026-06-11-zero-based-indexing.md``:

- HR2: no ``range(1, ...)`` loops in ``nstat/``
- HR3: no ``arr[<identifier> - 1]`` subscript patterns in ``nstat/``

Uses AST-based scanning (not regex) so:
- ``len(x) - 1`` outside a subscript does not trigger,
- ``n - 1`` in math expressions does not trigger,
- only literal-``1`` subtractions inside subscripts are flagged.

Each violation is keyed by ``(file, source_line_stripped)`` so the test
is robust to line-number shifts during the sweep.  A site that is
intentionally 1-based (e.g. MATLAB-faithful display label formatting)
goes into ``ALLOWLIST`` with a reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


NSTAT_DIR = Path(__file__).resolve().parents[1] / "nstat"


# Files whose 1-based patterns are intentionally allowed.
#
# ``nstat/compat/matlab/`` is the MATLAB-style import-alias shim — any
# 1-based pattern there is by design (matches the MATLAB surface
# verbatim).
ALLOWED_FILES: set[str] = {
    # all .py under nstat/compat/matlab/ are allowed by prefix below
}
ALLOWED_FILE_PREFIXES: tuple[str, ...] = ("compat/matlab/",)


# Variable-name allowlist for ``[<var> - 1]`` patterns that are algorithmic
# (time recursion, last-element access, EM counters), not 1-based-to-0-based
# parameter translation.  Keyed by (relative_path, var_name).
#
# Use ``("*", var)`` to apply across every file.  Use a specific path to scope
# tightly when the same name has different meaning elsewhere.
VARIABLE_ALLOWLIST: set[tuple[str, str]] = {
    # decoding_algorithms.py: state-space Kalman / EM recursion variables.
    # `k`, `time_index`, `num_steps`, `K`, `N` are time/length indices;
    # ``[k - 1]`` is "previous time step", ``[N - 1]`` is "last element".
    # ``cnt`` is the EM iteration counter; ``[cnt - 1]`` is "previous iter".
    ("decoding_algorithms.py", "k"),
    ("decoding_algorithms.py", "num_steps"),
    ("decoding_algorithms.py", "K"),
    ("decoding_algorithms.py", "N"),
    ("decoding_algorithms.py", "n"),
    ("decoding_algorithms.py", "cnt"),
}


# Site-level allowlist for individual lines that must keep their pattern.
# Key: (relative_path, source_line_stripped); Value: reason.
ALLOWLIST: dict[tuple[str, str], str] = {
    # ---- Algorithmic patterns (NOT 1-based-to-0-based translation) ----
    # Time-recursive update: ``post[t - 1]`` reads the PREVIOUS time step.
    ("paper_examples_full.py", "pred0 = max(post[t - 1, 0] * p_ij[0, 0] + post[t - 1, 1] * p_ij[1, 0], 1e-15)"): "time recursion",
    ("paper_examples_full.py", "pred1 = max(post[t - 1, 0] * p_ij[0, 1] + post[t - 1, 1] * p_ij[1, 1], 1e-15)"): "time recursion",
    ("paper_examples_full.py", "for t in range(1, n_t):"): "time recursion: t=0 is initial condition",
    # ``corr[N - 1]`` is the last element of an N-length array — algorithmic.
    ("core.py", "corr = corr / corr[N - 1] if corr[N - 1] != 0 else corr"): "last-element normalization",
    # Retry counter / iteration counter: 1-based by display intent.
    ("data_manager.py", "for attempt in range(1, retries + 1):"): "retry counter (display)",
    ("glm.py", "for n_iter in range(1, max_iter + 1):"): "iteration counter (display)",
    # Time-recursive ensemble effects: ``spikes[i - 1, :]`` = previous time step.
    ("simulators.py", "ens_effect[0] = ensemble_kernel_arr[0] * float(spikes[i - 1, 1])"): "previous-timestep ensemble effect",
    ("simulators.py", "ens_effect[1] = ensemble_kernel_arr[1] * float(spikes[i - 1, 0])"): "previous-timestep ensemble effect",
    # History covariates: lag values start at 1 (lag 0 = the spike itself).
    ("paper_examples_full.py", "full_hist = _history_matrix(y_sel, list(range(1, int(candidate_q[-1]) + 1)))"): "history lags start at 1",
    # Lag iteration: lag values start at 1 (lag 0 = current step).
    ("decoding_algorithms.py", "for k in range(1, lag_count + 1):"): "lag values start at 1",
}


def _is_allowed_file(rel_path: str) -> bool:
    if rel_path in ALLOWED_FILES:
        return True
    return any(rel_path.startswith(prefix) for prefix in ALLOWED_FILE_PREFIXES)


class _Visitor(ast.NodeVisitor):
    """Collects HR2 (range-from-1) and HR3 ([x - 1] subscript) violations."""

    def __init__(self, source_lines: list[str], rel_path: str) -> None:
        self.source_lines = source_lines
        self.rel_path = rel_path
        self.range_hits: list[tuple[int, str]] = []  # (lineno, source)
        self.minus_one_hits: list[tuple[int, str]] = []

    def _line(self, lineno: int) -> str:
        try:
            return self.source_lines[lineno - 1].strip()
        except IndexError:
            return ""

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 1
        ):
            self.range_hits.append((node.lineno, self._line(node.lineno)))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        for sub_expr in self._iter_index_components(node.slice):
            if self._is_identifier_minus_one(sub_expr):
                ident = self._extract_identifier(sub_expr)
                if ident is not None and (
                    (self.rel_path, ident) in VARIABLE_ALLOWLIST
                    or ("*", ident) in VARIABLE_ALLOWLIST
                ):
                    continue
                self.minus_one_hits.append((sub_expr.lineno, self._line(sub_expr.lineno)))
        self.generic_visit(node)

    @staticmethod
    def _extract_identifier(node: ast.AST) -> str | None:
        if isinstance(node, ast.BinOp) and isinstance(node.left, ast.Name):
            return node.left.id
        return None

    @staticmethod
    def _iter_index_components(slice_node: ast.AST):
        if isinstance(slice_node, ast.Tuple):
            yield from slice_node.elts
        elif isinstance(slice_node, ast.Slice):
            for part in (slice_node.lower, slice_node.upper, slice_node.step):
                if part is not None:
                    yield part
        else:
            yield slice_node

    @staticmethod
    def _is_identifier_minus_one(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and isinstance(node.left, (ast.Name, ast.Attribute, ast.Subscript))
            and isinstance(node.right, ast.Constant)
            and node.right.value == 1
        )


def _collect_violations() -> tuple[list[str], list[str]]:
    """Return (range_violations, minus_one_violations) as formatted lines."""
    range_violations: list[str] = []
    minus_one_violations: list[str] = []

    for py_file in sorted(NSTAT_DIR.rglob("*.py")):
        rel_path = py_file.relative_to(NSTAT_DIR).as_posix()
        if _is_allowed_file(rel_path):
            continue

        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        visitor = _Visitor(source.splitlines(), rel_path)
        visitor.visit(tree)

        for lineno, line in visitor.range_hits:
            if (rel_path, line) in ALLOWLIST:
                continue
            range_violations.append(f"  nstat/{rel_path}:{lineno}: {line}")

        for lineno, line in visitor.minus_one_hits:
            if (rel_path, line) in ALLOWLIST:
                continue
            minus_one_violations.append(f"  nstat/{rel_path}:{lineno}: {line}")

    return range_violations, minus_one_violations


def test_no_range_from_one_in_nstat() -> None:
    """HR2: ``range(1, ...)`` loops indicate MATLAB-style 1-based indexing."""
    range_violations, _ = _collect_violations()
    if range_violations:
        preview = range_violations[:15]
        tail = (
            f"\n  ... and {len(range_violations) - 15} more" if len(range_violations) > 15 else ""
        )
        pytest.fail(
            "HR2 violations (`range(1, ...)` loops) in nstat/:\n"
            + "\n".join(preview)
            + tail
            + "\n\nSweep to `range(N)` or `enumerate(...)`.  "
            "See docs/proposals/2026-06-11-zero-based-indexing.md."
        )


def test_no_identifier_minus_one_subscript_in_nstat() -> None:
    """HR3: ``arr[idx - 1]`` patterns are MATLAB-style index translation."""
    _, minus_one_violations = _collect_violations()
    if minus_one_violations:
        preview = minus_one_violations[:15]
        tail = (
            f"\n  ... and {len(minus_one_violations) - 15} more"
            if len(minus_one_violations) > 15
            else ""
        )
        pytest.fail(
            "HR3 violations (`arr[<ident> - 1]` subscripts) in nstat/:\n"
            + "\n".join(preview)
            + tail
            + "\n\nFlip the call sites to 0-based indexing.  "
            "See docs/proposals/2026-06-11-zero-based-indexing.md."
        )
