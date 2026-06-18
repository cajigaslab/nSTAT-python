"""Test guard for the v0.5.0 0-based indexing convention.

Implements hard rules from ``docs/proposals/2026-06-11-zero-based-indexing.md``:

- HR2: no ``range(1, ...)`` loops
- HR3: no ``arr[<identifier> - 1]`` subscript patterns
- HR4 (literal-arg): no hardcoded ``1``/``2``/``3`` passed as a selector
  to known 0-based public methods (``getNST``, ``getCov``, ``getCoeffs``,
  ``getSubSignal``, etc.)

Scans (Tier 1 hardening from the v0.5.0 retrospective):

- ``nstat/**/*.py``  — package source
- ``examples/**/*.py``  — example scripts (paper + extras)
- ``tools/**/*.py``  — notebook builders / regen scripts
- ``notebooks/**/*.ipynb``  — extracts code cells via ``nbformat`` and
  scans their AST

Uses AST scanning (not regex) so ``len(x) - 1`` outside a subscript and
``n - 1`` in math expressions do not trigger.

Each violation is keyed by ``(file, source_line_stripped)`` so the test
is robust to line-number shifts during a sweep.  A site that is
intentionally 1-based (e.g. MATLAB-faithful display label formatting)
goes into ``ALLOWLIST`` with a reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NSTAT_DIR = REPO_ROOT / "nstat"
EXAMPLES_DIR = REPO_ROOT / "examples"
TOOLS_DIR = REPO_ROOT / "tools"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


# HR4: methods that take a 0-based selector as a positional/keyword arg.
# Literal ``1``/``2``/``3`` passed to one of these gets flagged.  Each
# entry says which arg position is the selector (0-indexed: 0 = first
# positional, 1 = second positional, etc.); negative means "any literal
# 1/2/3 passed positionally anywhere in this call is suspect."
#
# Kept tight on purpose — false positives are worse than false negatives
# here (we'd rather miss a legit literal than reject a benign one).
KNOWN_ZERO_BASED_METHODS: dict[str, int] = {
    # Single-selector accessors (selector is the first positional arg)
    "getNST": 0,
    "getCov": 0,
    "getCoeffs": 0,
    "getCoeffsWithLabels": 0,
    "getHistCoeffs": 0,
    "getHistCoeffsWithLabels": 0,
    "getSubSignal": 0,
    "getConfig": 0,
    "getSubSignalFromInd": 0,
    "computeKSStats": 0,
    "computeFitResidual": 0,
    "evalLambda": 0,
    "getSigCoeffs": 0,
    "getNeighbors": 0,
    "getDesignMatrix": 0,
    "getHistMatrices": 0,
    "getEnsCovMatrix": 0,
    # Two-arg analysis helpers: trial is first, selector is second.
    "RunAnalysisForNeuron": 1,
    "run_analysis_for_neuron": 1,
    "computeHistLag": 1,
    "computeNeighbors": 1,
    "spikeTrigAvg": 1,
    "compHistEnsCoeff": 1,
    # ConfigCollection.setConfig(trial, idx) — second positional arg.
    "setConfig": 1,
}


def _is_literal_int(node: ast.AST, values: set[int]) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in values


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
    # PPHybridFilterLinear time recursion: MU_u[:, time_index - 1] reads
    # previous-step model probabilities (M22 fix); same in the m-step path.
    ("decoding_algorithms.py", "time_index"),
    ("decoding_algorithms.py", "K"),
    ("decoding_algorithms.py", "N"),
    ("decoding_algorithms.py", "n"),
    ("decoding_algorithms.py", "cnt"),
    # Tier 1 hardening — examples/notebooks/tools that use canonical
    # time-recursion variables (``t``, ``k``, ``lt``, ``row``).
    ("em_dynamax_demo.py", "t"),
    ("validation_pykalman_demo.py", "t"),
    ("example03_psth_and_ssglm.py", "lt"),
    ("example05_decoding_ppaf_pphf.py", "k"),
    # ValidationDataSet raster: ``axs[row - 1]`` because row is the
    # display label (1-based math convention) while axs is 0-based.
    ("ValidationDataSet.ipynb", "row"),
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
    # alignSubSignals merge loop: dimension 0 is handled before the loop,
    # then dims 1..N-1 are merged in.
    ("core.py", "for i in range(1, self.dimension):"): "skip first dimension (handled before loop)",
    # to_collapsed_train: ``selected_trains[idx - 1]`` is the previous train,
    # not a 1-based index translation; idx here is the enumerate counter.
    ("trial.py", "prev_train = selected_trains[idx - 1]"): "previous-element lookup",
    # Display tick positions stay 1-indexed (math convention) for raster plots.
    ("trial.py", "ax.set_yticks(range(1, len(selected) + 1), [str(item) for item in selected])"): "ytick display positions stay 1-indexed",
    # nstCollExamples: raster ytick positions/labels use 1..N for human-readable
    # neuron numbering (Neuron1, Neuron2, ...) matching MATLAB raster convention.
    ("nstCollExamples.ipynb", "_ax.set_yticks(list(range(1, 21)))"): "raster ytick positions 1..N for MATLAB-style display",
    ("nstCollExamples.ipynb", '_ax.set_yticklabels([f"Neuron{i}" for i in range(1, 21)], fontstyle="italic")'): "Neuron1..NeuronN labels mirror MATLAB raster",
    ("nstCollExamples.ipynb", "_ax.set_yticks(list(range(1, len(labels) + 1)))"): "raster ytick positions 1..N for MATLAB-style display",
    # HR4 baseline: legitimate 0-based selectors > 0 (intentional, not stale 1-based).
    # README example uses getNST(1) as a fallback after getNST(0) fails — supports
    # objects with either indexing convention.
    ("example2_simulate_cif_spiketrain_10s.py", "first_train = spike_obj.getNST(1)"): "fallback after getNST(0) failed",
    # AnalysisExamples2: comparing nstat fit against glmfit reference using the
    # second fit (0-based index 1, was MATLAB b{2}).
    ("AnalysisExamples2.ipynb", "b_diff = b - fitResults.getCoeffs(1)[0]"): "MATLAB b{2} = 0-based fit 1",
    # NetworkTutorial: third fit (0-based 2) is the connectivity-included model.
    ("NetworkTutorial.ipynb", "coeffs, labels, _ = fit.getCoeffsWithLabels(2)"): "intentional 3rd fit (0-based)",
    # NetworkTutorial: comparing first vs second spike train counts.
    ("NetworkTutorial.ipynb", '"spike_counts": [spikeColl.getNST(0).n_spikes, spikeColl.getNST(1).n_spikes],'): "first + second train",
    # PPSimExample: third subsignal (0-based 2) is the binomial model's lambda.
    ("PPSimExample.ipynb", "results[0].lambdaSignal.getSubSignal(2).plot(handle=ax)"): "intentional 3rd subsignal (0-based)",
    # Example01: second subsignal of lambda (0-based 1) for the green-curve plot.
    ("example01_mepsc_poisson.py", 'ax.plot(lam.time, lam.getSubSignal(1).data[:, 0], "g", linewidth=2)'): "second subsignal (0-based)",
    # Time-recursion `range(1, T)` patterns in examples and tools — t=0 is initial condition.
    ("em_dynamax_demo.py", "for t in range(1, T):"): "time recursion: t=0 is initial",
    ("validation_pykalman_demo.py", "for t in range(1, T):"): "time recursion: t=0 is initial",
    ("example05_decoding_ppaf_pphf.py", "for k in range(1, T):"): "time recursion: k=0 is initial",
    # run_helpfile: i is the 1-based figure number (display convention).
    ("run_helpfile.py", "for i in range(1, n_figures + 1):"): "figure numbering (display)",
    # DecodingExampleWithHist: idx is the canonical time recursion index.
    ("DecodingExampleWithHist.ipynb", "for idx in range(1, len(time)):"): "time recursion",
}


def _is_allowed_file(rel_path: str) -> bool:
    if rel_path in ALLOWED_FILES:
        return True
    return any(rel_path.startswith(prefix) for prefix in ALLOWED_FILE_PREFIXES)


class _Visitor(ast.NodeVisitor):
    """Collects HR2 (range-from-1), HR3 ([x - 1] subscript), HR4 (literal-arg) violations."""

    def __init__(self, source_lines: list[str], rel_path: str) -> None:
        self.source_lines = source_lines
        self.rel_path = rel_path
        self.range_hits: list[tuple[int, str]] = []  # (lineno, source)
        self.minus_one_hits: list[tuple[int, str]] = []
        self.literal_arg_hits: list[tuple[int, str]] = []

    def _line(self, lineno: int) -> str:
        try:
            return self.source_lines[lineno - 1].strip()
        except IndexError:
            return ""

    def visit_Call(self, node: ast.Call) -> None:
        # HR2: range(1, ...) loops
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 1
        ):
            self.range_hits.append((node.lineno, self._line(node.lineno)))

        # HR4: literal 1/2/3 passed as selector arg to a known 0-based method.
        method_name: str | None = None
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            method_name = node.func.id
        if method_name in KNOWN_ZERO_BASED_METHODS:
            sel_pos = KNOWN_ZERO_BASED_METHODS[method_name]
            if sel_pos < len(node.args) and _is_literal_int(node.args[sel_pos], {1, 2, 3}):
                self.literal_arg_hits.append((node.lineno, self._line(node.lineno)))

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        for sub_expr in self._iter_index_components(node.slice):
            if self._is_identifier_minus_one(sub_expr):
                ident = self._extract_identifier(sub_expr)
                if ident is not None and (
                    (self.rel_path, ident) in VARIABLE_ALLOWLIST
                    or (Path(self.rel_path).name, ident) in VARIABLE_ALLOWLIST
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


def _notebook_source(nb_path: Path) -> str:
    """Concatenate all code-cell source from an .ipynb into a synthetic .py.

    Each cell is preceded by a blank line so line numbers don't collide
    in the AST (rough; line-number reporting points to the start of the
    notebook rather than the cell, but the source-line key in ALLOWLIST
    still identifies the exact violating statement).
    """
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception:
        return ""
    parts: list[str] = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            parts.append(cell.source)
    return "\n\n".join(parts)


def _iter_scan_targets() -> list[tuple[str, Path, str]]:
    """Yield ``(display_root, source_path, rel_display)`` for every file/notebook to scan."""
    targets: list[tuple[str, Path, str]] = []
    for root, name in [
        (NSTAT_DIR, "nstat"),
        (EXAMPLES_DIR, "examples"),
        (TOOLS_DIR, "tools"),
    ]:
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            rel = py_file.relative_to(root).as_posix()
            targets.append((name, py_file, rel))
    if NOTEBOOKS_DIR.exists():
        for nb_file in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
            rel = nb_file.relative_to(NOTEBOOKS_DIR).as_posix()
            targets.append(("notebooks", nb_file, rel))
    return targets


def _collect_violations() -> tuple[list[str], list[str], list[str]]:
    """Return (range_violations, minus_one_violations, literal_arg_violations)."""
    range_violations: list[str] = []
    minus_one_violations: list[str] = []
    literal_arg_violations: list[str] = []

    for display_root, src_path, rel in _iter_scan_targets():
        # File-prefix allowlist still applies to nstat/ (compat/matlab shim).
        if display_root == "nstat" and _is_allowed_file(rel):
            continue

        if src_path.suffix == ".ipynb":
            source = _notebook_source(src_path)
            if not source:
                continue
        else:
            source = src_path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source, filename=str(src_path))
        except SyntaxError:
            continue

        visitor = _Visitor(source.splitlines(), rel)
        visitor.visit(tree)

        for lineno, line in visitor.range_hits:
            if (rel, line) in ALLOWLIST or (Path(rel).name, line) in ALLOWLIST:
                continue
            range_violations.append(f"  {display_root}/{rel}:{lineno}: {line}")

        for lineno, line in visitor.minus_one_hits:
            if (rel, line) in ALLOWLIST or (Path(rel).name, line) in ALLOWLIST:
                continue
            minus_one_violations.append(f"  {display_root}/{rel}:{lineno}: {line}")

        for lineno, line in visitor.literal_arg_hits:
            if (rel, line) in ALLOWLIST or (Path(rel).name, line) in ALLOWLIST:
                continue
            literal_arg_violations.append(f"  {display_root}/{rel}:{lineno}: {line}")

    return range_violations, minus_one_violations, literal_arg_violations


def _format_failure(rule: str, hint: str, violations: list[str]) -> str:
    preview = violations[:15]
    tail = f"\n  ... and {len(violations) - 15} more" if len(violations) > 15 else ""
    return (
        f"{rule} violations:\n"
        + "\n".join(preview)
        + tail
        + f"\n\n{hint}  "
        "See docs/proposals/2026-06-11-zero-based-indexing.md."
    )


def test_no_range_from_one() -> None:
    """HR2: ``range(1, ...)`` loops indicate MATLAB-style 1-based indexing.

    Scans nstat/, examples/, tools/, notebooks/.
    """
    range_violations, _, _ = _collect_violations()
    if range_violations:
        pytest.fail(
            _format_failure(
                "HR2 (`range(1, ...)` loops)",
                "Sweep to `range(N)` or `enumerate(...)`.",
                range_violations,
            )
        )


def test_no_identifier_minus_one_subscript() -> None:
    """HR3: ``arr[idx - 1]`` patterns are MATLAB-style index translation.

    Scans nstat/, examples/, tools/, notebooks/.
    """
    _, minus_one_violations, _ = _collect_violations()
    if minus_one_violations:
        pytest.fail(
            _format_failure(
                "HR3 (`arr[<ident> - 1]` subscripts)",
                "Flip the call sites to 0-based indexing.",
                minus_one_violations,
            )
        )


def test_no_literal_arg_to_known_zero_based_method() -> None:
    """HR4: literal ``1``/``2``/``3`` passed as a selector arg to a method
    known to take 0-based indices (``getNST``, ``getCov``, ``getCoeffs``,
    ``RunAnalysisForNeuron``, ``computeHistLag``, ``setConfig``, etc.).

    This catches the drift that the AST scanner's HR2/HR3 patterns miss:
    hardcoded selector literals in notebooks/examples/tools that look
    fine to a regex sweep but are out-of-bounds after the v0.5.0 flip.
    """
    _, _, literal_arg_violations = _collect_violations()
    if literal_arg_violations:
        pytest.fail(
            _format_failure(
                "HR4 (literal 1/2/3 passed to a 0-based method)",
                "Pass 0, 1, 2 instead.  If the literal is intentional"
                " (e.g. fit_num=2 to select the third fit, 0-based), add it"
                " to ALLOWLIST keyed by (file, source_line).",
                literal_arg_violations,
            )
        )
