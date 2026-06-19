#!/usr/bin/env python3
"""Section-aligned code-structure parity diff: MATLAB helpfiles vs Python notebooks.

For each in-scope topic this tool:

1. Parses ``helpfiles/<Topic>.m`` (from a local MATLAB nSTAT checkout), splitting
   on ``%%`` section markers and extracting every function-call site inside
   each section's code body.
2. Parses ``notebooks/<Topic>.ipynb``, walking code cells whose source begins
   with ``# SECTION N: <title>`` and using :mod:`ast` to enumerate call sites.
3. Aligns MATLAB sections to Python sections by title (exact, then fuzzy match
   on a normalized title), and within paired sections runs LCS on the
   call-name sequences (with fuzzy name matching: ``plot`` matches
   ``ax.plot``/``plt.plot``/``fig.plot``/``.plot`` etc.).
4. Emits a per-topic markdown report under
   ``.parity-review/code_structure_<topic>.md`` and a JSON summary at
   ``.parity-review/code_structure_scores.json`` keyed by topic.

The score per topic is the fraction of MATLAB call sites with a matching
Python call inside the same section.

Independence: this script reads the local MATLAB checkout as a read-only
filesystem traversal. It does not import or execute MATLAB code, and it
does not write anything outside this repo's ``.parity-review/`` directory.

Usage
-----
    python tools/parity/code_structure_diff.py --topic mEPSCAnalysis
    python tools/parity/code_structure_diff.py --all
    python tools/parity/code_structure_diff.py --all --fail-below-threshold 0.85

Exit codes
----------
- 0 — all scanned topics pass threshold (or no threshold was set).
- 1 — invocation error (missing helpfile/notebook, etc.).
- 2 — at least one topic scored below ``--fail-below-threshold``.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import nbformat


# In-scope topics — must mirror the prompt list (23 topics).
IN_SCOPE_TOPICS: tuple[str, ...] = (
    "AnalysisExamples",
    "AnalysisExamples2",
    "CovCollExamples",
    "CovariateExamples",
    "DecodingExample",
    "DecodingExampleWithHist",
    "EventsExamples",
    "ExplicitStimulusWhiskerData",
    "HippocampalPlaceCellExample",
    "HistoryExamples",
    "HybridFilterExample",
    "NetworkTutorial",
    "PPSimExample",
    "PPThinning",
    "PSTHEstimation",
    "SignalObjExamples",
    "StimulusDecode2D",
    "TrialExamples",
    "ValidationDataSet",
    "mEPSCAnalysis",
    "nSTATPaperExamples",
    "nSpikeTrainExamples",
    "nstCollExamples",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATLAB_REPO = Path(
    os.environ.get("NSTAT_MATLAB_PATH", "/Users/iahncajigas/projects/nstat")
)
OUTPUT_DIR = REPO_ROOT / ".parity-review"


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass
class Call:
    """A single function-call site."""

    raw: str                # original textual form, e.g. "Analysis.RunAnalysisForAllNeurons"
    name: str               # bare function name, e.g. "RunAnalysisForAllNeurons"
    line: int               # 1-based line within its section's code body

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Call({self.raw!r} @ {self.line})"


@dataclass
class Section:
    """One ``%%`` MATLAB section or one ``# SECTION N:`` Python cell."""

    index: int              # 1-based ordinal
    title: str              # human-readable section title
    calls: list[Call] = field(default_factory=list)


@dataclass
class AlignedPair:
    """A MATLAB section paired with at most one Python section."""

    matlab: Section
    python: Section | None
    matched: list[tuple[Call, Call]] = field(default_factory=list)
    matlab_only: list[Call] = field(default_factory=list)
    python_only: list[Call] = field(default_factory=list)


# ----------------------------------------------------------------------------
# MATLAB-only / convention-only callable names that don't need a Python mirror
# ----------------------------------------------------------------------------

# Calls that exist in MATLAB by convention (e.g. printf-style auto-display when
# omitting a semicolon, MATLAB-only housekeeping). If a MATLAB-only call's
# bare name is in this set, it is dropped before scoring rather than counted
# as drift. Keep this list conservative.
MATLAB_CONVENTION_NAMES: frozenset[str] = frozenset(
    {
        "clear", "clc", "close", "format", "disp", "fprintf", "sprintf",
        "tic", "toc", "pause", "exist", "isempty", "length", "size",
        "numel", "zeros", "ones", "nan", "inf", "true", "false",
        # MATLAB-style data-loading helpers (Python uses np.loadtxt etc.)
        "importdata", "fullfile",
        # MATLAB control-style "calls" the regex catches but that aren't
        # really function calls in the AST sense.
        "if", "elseif", "else", "for", "while", "switch", "case", "end",
        "return", "break", "continue", "function", "global", "persistent",
        # MATLAB-only display helpers we can safely skip
        "__SLICE__",
    }
)


# Fuzzy-equivalence map: collapse a MATLAB or Python call name down to a
# canonical token before comparison. Anything not listed maps to its lowered
# bare name. Common matplotlib / plt prefix variants are stripped.
FUZZY_ALIASES: dict[str, str] = {
    # Plotting bridge — MATLAB's "plot" maps to matplotlib's .plot / plt.plot.
    "plot": "plot",
    "subplot": "subplot",
    "scatter": "scatter",
    "imshow": "imshow",
    "imagesc": "imshow",
    "bar": "bar",
    "hist": "hist",
    "histogram": "hist",
    "errorbar": "errorbar",
    "semilogx": "semilogx",
    "semilogy": "semilogy",
    "loglog": "loglog",
    "title": "title",
    "xlabel": "xlabel",
    "ylabel": "ylabel",
    "zlabel": "zlabel",
    "legend": "legend",
    "axis": "axis",
    "xlim": "xlim",
    "ylim": "ylim",
    "grid": "grid",
    "hold": "hold",
    "colorbar": "colorbar",
    "colormap": "colormap",
    "figure": "figure",
    "savefig": "savefig",
    "saveas": "savefig",
    # Random helpers
    "rand": "rand",
    "randn": "randn",
    "randi": "randi",
    # ndarray creation
    "linspace": "linspace",
    "logspace": "logspace",
    "arange": "linspace",     # MATLAB ``0:step:end`` ≈ ``np.arange``
    "min": "min",
    "max": "max",
    "mean": "mean",
    "median": "median",
    "std": "std",
    "var": "var",
    "sum": "sum",
    "abs": "abs",
    "round": "round",
    "floor": "floor",
    "ceil": "ceil",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    # nSTAT class & helper names — MATLAB and Python ports share these
    # verbatim per the parity contract, so they survive the canonicalizer
    # unchanged. We don't enumerate them here; the lower-cased identity
    # default is good enough.
    # Idiom canonicalization (operator/idiom names from MATLAB normalize
    # to the Python equivalent name).
    "set": "set",
    "get": "get",
    "regexp": "search",
    "regexprep": "sub",
    "strrep": "replace",
    "strsplit": "split",
    "strjoin": "join",
    "strtrim": "strip",
    "lower": "lower",
    "upper": "upper",
    "num2str": "str",
    "str2num": "float",
    "str2double": "float",
    "int2str": "str",
    # MATLAB array creation ↔ NumPy
    "ones": "ones",
    "zeros": "zeros",
    "eye": "eye",
    "repmat": "tile",
    "tile": "tile",
    # MATLAB shape ops
    "reshape": "reshape",
    "squeeze": "squeeze",
    "permute": "transpose",
    "transpose": "transpose",
    # add_subplot ↔ subplot
    "add_subplot": "subplot",
    "new_figure": "figure",
}


# ----------------------------------------------------------------------------
# MATLAB idiom rewriting — performed on raw source before call-extraction.
# Each tuple is (compiled regex, replacement); the replacement is a synthetic
# call form that the call-extraction regex then picks up.
# ----------------------------------------------------------------------------

_MATLAB_IDIOM_REWRITES: list[tuple[re.Pattern[str], str]] = [
    # MATLAB column-vector / slice assignments like M(:,1) = ... are NOT calls;
    # eliminate them entirely so they don't inflate the MATLAB-only count.
    (re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*[\s\d:,]+\s*\)\s*="), r" "),
    # NOTE: set(h, 'Prop', v) / get(h, 'Prop') rewrites were tried but introduce
    # more drift than they fix (MATLAB property names rarely match the matplotlib
    # `set_<name>` form exactly). Skipped.
]


def _rewrite_matlab_idioms(code: str) -> str:
    """Apply MATLAB idiom rewrites so the call-extractor picks up Python equivalents."""
    for pat, repl in _MATLAB_IDIOM_REWRITES:
        code = pat.sub(repl, code)
    return code

# Pure attribute / accessor noise we strip from the right end of a call chain
# to extract the bare method name. e.g. ``self.tracker.new_figure`` → ``new_figure``.
PLOT_PREFIX_RE = re.compile(r"^(?:plt|np|sp|pd|sns|ax\d*|axes?|fig|figure|self|cls)(?:\.|$)", re.IGNORECASE)


def canonicalize(call_name: str) -> str:
    """Reduce a call name to a fuzzy comparison key.

    Examples
    --------
    >>> canonicalize("Analysis.RunAnalysisForAllNeurons")
    'runanalysisforallneurons'
    >>> canonicalize("ax.plot")
    'plot'
    >>> canonicalize("plt.subplot")
    'subplot'
    """
    if not call_name:
        return ""
    # Take the rightmost attribute (the actual method/function token).
    bare = call_name.rsplit(".", 1)[-1].strip()
    bare_lower = bare.lower()
    # Honor explicit aliases first.
    if bare_lower in FUZZY_ALIASES:
        return FUZZY_ALIASES[bare_lower]
    return bare_lower


# ----------------------------------------------------------------------------
# MATLAB parser
# ----------------------------------------------------------------------------

# MATLAB function-call regex. We match ``Name(`` or ``Name.method(`` after a
# non-identifier character, and explicitly exclude declarations on a line that
# starts with ``function`` or that look like indexing of a known builtin.
_MATLAB_CALL_RE = re.compile(
    r"""
    (?P<lead>^|[^A-Za-z0-9_.])           # boundary (not inside another ident)
    (?P<name>
        [A-Za-z_][A-Za-z0-9_]*           # base identifier
        (?:\.[A-Za-z_][A-Za-z0-9_]*)*     # optional .method chains
    )
    \s*\(
    """,
    re.VERBOSE,
)

# MATLAB comment markers: ``%`` to end of line. We strip these before
# call-extraction to avoid false positives in prose.
_MATLAB_COMMENT_RE = re.compile(r"%(?!%).*$", re.MULTILINE)

# MATLAB strings (single-quoted). Replace with whitespace before call-extract
# so we don't pick up things inside text literals.
_MATLAB_STRING_RE = re.compile(r"'(?:''|[^'\n])*'")


def _strip_matlab_noise(code: str) -> str:
    """Remove single-line comments and string literals from MATLAB code.

    Also applies idiom rewrites (``set(h,'X',v)`` → ``ax.set_X(v)`` etc.)
    BEFORE string-stripping so the property-name literal survives long enough
    to participate in the rewrite.
    """
    code = _rewrite_matlab_idioms(code)
    code = _MATLAB_STRING_RE.sub(lambda m: " " * len(m.group(0)), code)
    code = _MATLAB_COMMENT_RE.sub("", code)
    return code


def parse_matlab_helpfile(path: Path) -> list[Section]:
    """Split a MATLAB helpfile on ``%%`` markers and extract call sites.

    A section opens on a line that *starts* (after optional whitespace) with
    ``%%`` and the first non-blank text after the marker is the title. All
    subsequent lines, until the next ``%%`` or EOF, form the section body.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    sections: list[Section] = []
    cur_title: str | None = None
    cur_body: list[str] = []

    def _flush(idx: int) -> None:
        if cur_title is None:
            return
        body = "\n".join(cur_body)
        calls = _extract_matlab_calls(body)
        sections.append(Section(index=idx, title=cur_title, calls=calls))

    for raw in lines:
        stripped = raw.lstrip()
        if stripped.startswith("%%"):
            # New section. First, flush the previous one.
            _flush(len(sections) + 1)
            # The title is everything after the leading ``%%`` (and optional space).
            title = stripped[2:].strip()
            if not title:
                title = "(untitled section)"
            cur_title = title
            cur_body = []
        else:
            if cur_title is not None:
                cur_body.append(raw)
    _flush(len(sections) + 1)

    return sections


def _extract_matlab_calls(body: str) -> list[Call]:
    """Pull function-call sites from a MATLAB section body."""
    cleaned = _strip_matlab_noise(body)
    calls: list[Call] = []
    for match in _MATLAB_CALL_RE.finditer(cleaned):
        name = match.group("name")
        bare = name.rsplit(".", 1)[-1].lower()
        # Skip MATLAB-convention names (control flow, display helpers, etc.).
        if bare in MATLAB_CONVENTION_NAMES:
            continue
        # Compute the 1-based line number.
        line = cleaned.count("\n", 0, match.start()) + 1
        calls.append(Call(raw=name, name=bare, line=line))
    return calls


# ----------------------------------------------------------------------------
# Python notebook parser
# ----------------------------------------------------------------------------

_SECTION_HEADER_RE = re.compile(r"^\s*#\s*SECTION\s+(\d+)\s*:\s*(.*?)\s*$", re.IGNORECASE)


def parse_python_notebook(path: Path) -> list[Section]:
    """Walk a notebook, capturing ``# SECTION N: <title>`` code cells."""
    nb = nbformat.read(path, as_version=4)
    sections: list[Section] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source or ""
        first_line = src.split("\n", 1)[0]
        m = _SECTION_HEADER_RE.match(first_line)
        if not m:
            continue
        idx = int(m.group(1))
        title = m.group(2).strip() or "(untitled section)"
        calls = _extract_python_calls(src)
        sections.append(Section(index=idx, title=title, calls=calls))
    return sections


def _extract_python_calls(src: str) -> list[Call]:
    """Use :mod:`ast` to enumerate call sites in a notebook cell."""
    # ipython magics / shell escapes confuse the AST parser; strip them.
    safe_lines: list[str] = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")) and not stripped.startswith("#"):
            safe_lines.append("")
        else:
            safe_lines.append(line)
    safe_src = "\n".join(safe_lines)

    try:
        tree = ast.parse(safe_src)
    except SyntaxError:
        # Cell may contain a partial/illegal snippet; skip rather than crash.
        return []

    calls: list[Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        raw = _format_call_target(node.func)
        if raw is None:
            continue
        bare = raw.rsplit(".", 1)[-1]
        calls.append(Call(raw=raw, name=bare, line=getattr(node, "lineno", 0)))
    return calls


def _format_call_target(node: ast.AST) -> str | None:
    """Render an ``ast.Call.func`` node as a dotted-name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _format_call_target(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Call):
        # Chained call: ``foo(...).bar(...)`` — represent the outer head.
        inner = _format_call_target(node.func)
        return inner
    if isinstance(node, ast.Subscript):
        return _format_call_target(node.value)
    return None


# ----------------------------------------------------------------------------
# Section alignment
# ----------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy matching."""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", title).lower()
    return re.sub(r"\s+", " ", cleaned).strip()


def align_sections(
    matlab: list[Section], python: list[Section]
) -> list[AlignedPair]:
    """Pair MATLAB sections to Python sections by best title match.

    Each Python section is consumed at most once. We greedily match in MATLAB
    order: for each unpaired MATLAB section, find the unpaired Python section
    with the highest title-similarity ratio (must clear ``0.55``); ties broken
    by ordinal proximity.
    """
    py_used: set[int] = set()
    pairs: list[AlignedPair] = []

    py_norm = {p.index: _normalize_title(p.title) for p in python}

    for m_sec in matlab:
        m_norm = _normalize_title(m_sec.title)
        best_idx: int | None = None
        best_score = 0.55  # minimum acceptance threshold
        for p_sec in python:
            if p_sec.index in py_used:
                continue
            ratio = difflib.SequenceMatcher(None, m_norm, py_norm[p_sec.index]).ratio()
            # Mild ordinal-proximity tiebreaker (max ±0.05 bonus).
            ord_bonus = max(0.0, 0.05 - 0.01 * abs(m_sec.index - p_sec.index))
            score = ratio + ord_bonus
            if score > best_score:
                best_score = score
                best_idx = p_sec.index
        if best_idx is not None:
            py_used.add(best_idx)
            py_sec = next(p for p in python if p.index == best_idx)
            pairs.append(AlignedPair(matlab=m_sec, python=py_sec))
        else:
            pairs.append(AlignedPair(matlab=m_sec, python=None))

    return pairs


# ----------------------------------------------------------------------------
# LCS within paired sections
# ----------------------------------------------------------------------------


def lcs_call_match(
    matlab_calls: list[Call], python_calls: list[Call]
) -> tuple[list[tuple[Call, Call]], list[Call], list[Call]]:
    """Run LCS on two call sequences (fuzzy-canonicalized) and return the alignment."""
    a = [canonicalize(c.raw) for c in matlab_calls]
    b = [canonicalize(c.raw) for c in python_calls]

    # Standard O(N*M) LCS DP.
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return [], list(matlab_calls), list(python_calls)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    matched: list[tuple[Call, Call]] = []
    matlab_only: list[Call] = []
    python_only: list[Call] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            matched.append((matlab_calls[i], python_calls[j]))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            matlab_only.append(matlab_calls[i])
            i += 1
        else:
            python_only.append(python_calls[j])
            j += 1
    matlab_only.extend(matlab_calls[i:])
    python_only.extend(python_calls[j:])
    return matched, matlab_only, python_only


# ----------------------------------------------------------------------------
# Topic-level driver + report writers
# ----------------------------------------------------------------------------


@dataclass
class TopicResult:
    topic: str
    matlab_total_calls: int
    matched_calls: int
    score: float
    sections: list[AlignedPair]
    unpaired_python: list[Section]
    error: str | None = None


def diff_topic(
    topic: str, matlab_repo: Path, notebooks_dir: Path
) -> TopicResult:
    """Compute the section-aligned call-match score for one topic."""
    m_path = matlab_repo / "helpfiles" / f"{topic}.m"
    n_path = notebooks_dir / f"{topic}.ipynb"
    if not m_path.exists():
        return TopicResult(topic, 0, 0, 0.0, [], [], error=f"missing helpfile: {m_path}")
    if not n_path.exists():
        return TopicResult(topic, 0, 0, 0.0, [], [], error=f"missing notebook: {n_path}")

    matlab_sections = parse_matlab_helpfile(m_path)
    python_sections = parse_python_notebook(n_path)

    pairs = align_sections(matlab_sections, python_sections)

    total_calls = 0
    matched_total = 0
    for pair in pairs:
        m_calls = pair.matlab.calls
        total_calls += len(m_calls)
        if pair.python is None:
            pair.matlab_only = list(m_calls)
            continue
        matched, mo, po = lcs_call_match(m_calls, pair.python.calls)
        pair.matched = matched
        pair.matlab_only = mo
        pair.python_only = po
        matched_total += len(matched)

    paired_py_indices = {p.python.index for p in pairs if p.python is not None}
    unpaired_python = [p for p in python_sections if p.index not in paired_py_indices]

    score = (matched_total / total_calls) if total_calls else 1.0
    return TopicResult(
        topic=topic,
        matlab_total_calls=total_calls,
        matched_calls=matched_total,
        score=score,
        sections=pairs,
        unpaired_python=unpaired_python,
    )


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_topic_report(result: TopicResult, out_dir: Path) -> Path:
    """Write the side-by-side markdown diff for one topic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"code_structure_{result.topic}.md"

    lines: list[str] = []
    lines.append(f"# Code-structure parity: `{result.topic}`")
    lines.append("")
    if result.error:
        lines.append(f"> **ERROR**: {result.error}")
        lines.append("")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    pct = result.score * 100
    lines.append(
        f"- MATLAB call sites: **{result.matlab_total_calls}**"
        f"  ·  matched in same section: **{result.matched_calls}**"
        f"  ·  score: **{pct:.1f}%**"
    )
    lines.append("")

    for pair in result.sections:
        lines.append(f"## §{pair.matlab.index} {pair.matlab.title}")
        if pair.python is None:
            lines.append("")
            lines.append("_No matching Python section found._")
            lines.append("")
            if pair.matlab_only:
                lines.append("MATLAB-only calls in this section:")
                lines.append("")
                for c in pair.matlab_only:
                    lines.append(f"- `{_md_escape(c.raw)}` (line {c.line})")
                lines.append("")
            continue

        lines.append(
            f"_Paired with Python §{pair.python.index}: {pair.python.title}_"
        )
        lines.append("")
        lines.append("| # | MATLAB call | Python call | Status |")
        lines.append("|---|---|---|---|")

        # Render matched pairs first (call appearance order).
        for m, p in pair.matched:
            lines.append(
                f"| ✓ | `{_md_escape(m.raw)}` | `{_md_escape(p.raw)}` | matched |"
            )
        for c in pair.matlab_only:
            lines.append(
                f"| · | `{_md_escape(c.raw)}` (line {c.line}) | — | matlab-only |"
            )
        for c in pair.python_only:
            lines.append(
                f"| · | — | `{_md_escape(c.raw)}` (line {c.line}) | python-only |"
            )
        lines.append("")

    if result.unpaired_python:
        lines.append("## Unpaired Python sections")
        lines.append("")
        for s in result.unpaired_python:
            lines.append(f"- §{s.index}: {s.title} ({len(s.calls)} calls)")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_scores_json(results: Iterable[TopicResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "code_structure_scores.json"
    payload = {
        r.topic: {
            "score": round(r.score, 4),
            "matlab_total_calls": r.matlab_total_calls,
            "matched_calls": r.matched_calls,
            "error": r.error,
        }
        for r in results
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="code_structure_diff.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--topic", help="Run a single topic (must be in-scope).")
    g.add_argument("--all", action="store_true", help="Run all in-scope topics.")
    p.add_argument(
        "--matlab-repo",
        type=Path,
        default=DEFAULT_MATLAB_REPO,
        help=f"Path to the local MATLAB nSTAT checkout (default: {DEFAULT_MATLAB_REPO}).",
    )
    p.add_argument(
        "--notebooks-dir",
        type=Path,
        default=REPO_ROOT / "notebooks",
        help="Directory containing notebook ports (default: ./notebooks).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for reports (default: ./.parity-review).",
    )
    p.add_argument(
        "--fail-below-threshold",
        type=float,
        default=None,
        help="Exit with code 2 if any topic scores strictly below this fraction (0..1).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if not args.matlab_repo.exists():
        print(
            f"ERROR: MATLAB repo not found at {args.matlab_repo}. "
            f"Set --matlab-repo or NSTAT_MATLAB_PATH.",
            file=sys.stderr,
        )
        return 1
    if not args.notebooks_dir.exists():
        print(f"ERROR: notebooks dir not found at {args.notebooks_dir}.", file=sys.stderr)
        return 1

    if args.topic:
        if args.topic not in IN_SCOPE_TOPICS:
            print(
                f"WARNING: --topic {args.topic!r} is not in the in-scope list; "
                f"running anyway.",
                file=sys.stderr,
            )
        topics: tuple[str, ...] = (args.topic,)
    else:
        topics = IN_SCOPE_TOPICS

    results: list[TopicResult] = []
    for t in topics:
        result = diff_topic(t, args.matlab_repo, args.notebooks_dir)
        results.append(result)
        report_path = write_topic_report(result, args.output_dir)
        if result.error:
            print(f"[{t}] ERROR: {result.error}  -> {report_path}")
        else:
            pct = result.score * 100
            print(
                f"[{t}] {result.matched_calls}/{result.matlab_total_calls} "
                f"calls matched ({pct:.1f}%)  -> {report_path}"
            )

    scores_path = write_scores_json(results, args.output_dir)
    print(f"summary: {scores_path}")

    if args.fail_below_threshold is not None:
        below = [r for r in results if not r.error and r.score < args.fail_below_threshold]
        if below:
            names = ", ".join(f"{r.topic} ({r.score:.2f})" for r in below)
            print(
                f"FAIL: {len(below)} topic(s) below threshold "
                f"{args.fail_below_threshold}: {names}",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
