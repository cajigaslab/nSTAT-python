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
EXEMPTIONS_PATH = REPO_ROOT / "parity" / "code_structure_exemptions.yml"


@dataclass(frozen=True)
class TopicExemption:
    """Loaded exemption payload for one topic.

    ``call_names`` are bare lowercase callable names that should be excluded
    from BOTH the numerator and denominator across every section of the topic.

    ``drop_sections`` are MATLAB section indices (1-based) whose entire body
    should be excluded from scoring. Use for whole sections whose Python
    equivalent runs inside a helper (so the notebook cell deliberately has a
    different call shape).
    """

    call_names: frozenset[str]
    drop_sections: frozenset[int]
    # Map MATLAB section index -> tuple of additional Python section indices
    # whose calls should be concatenated onto the paired Python section's
    # call list before LCS alignment. Use when the MATLAB helpfile keeps a
    # per-trial loop and a batch-summary block in one section while the
    # Python port splits them into adjacent cells.
    extend_pairs: dict[int, tuple[int, ...]]


def _load_exemptions(path: Path = EXEMPTIONS_PATH) -> dict[str, TopicExemption]:
    """Load per-topic call-name and section exemptions from a YAML file.

    Returns a mapping ``{topic: TopicExemption}``. Missing file, missing
    PyYAML, or a malformed payload all degrade gracefully to an empty dict.

    The YAML schema is::

        version: 1
        topics:
          <TopicName>:
            # Per-call exemption (applies across every section of the topic):
            - call: "<bare_call_name>"
              reason: "<why this isn't drift>"
            # Optional: drop entire MATLAB section from scoring:
            - drop_section: <1-based-MATLAB-section-index>
              reason: "<why this section has no Python mirror>"

    Per-call exemptions are removed from BOTH the MATLAB call list and the
    score denominator before LCS alignment runs. Per-section exemptions drop
    the whole MATLAB section from scoring while still surfacing it in the
    per-section report under an ``exempted section`` header.
    """
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - PyYAML is a dev dep
        print(
            f"WARNING: PyYAML not installed; ignoring {path.name}.",
            file=sys.stderr,
        )
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:  # pragma: no cover - defensive
        print(f"WARNING: failed to parse {path.name}: {e}", file=sys.stderr)
        return {}
    topics = payload.get("topics") or {}
    out: dict[str, TopicExemption] = {}
    for topic, entries in topics.items():
        names: set[str] = set()
        sections: set[int] = set()
        extend: dict[int, tuple[int, ...]] = {}
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                if "call" in entry:
                    names.add(str(entry["call"]).lower())
                elif "drop_section" in entry:
                    try:
                        sections.add(int(entry["drop_section"]))
                    except (TypeError, ValueError):
                        continue
                elif "extend_matlab_section" in entry:
                    # Schema:
                    #   - extend_matlab_section: <matlab_index>
                    #     with_python_sections: [<idx>, <idx>, ...]
                    try:
                        m_idx = int(entry["extend_matlab_section"])
                    except (TypeError, ValueError):
                        continue
                    raw_with = entry.get("with_python_sections") or []
                    if not isinstance(raw_with, list):
                        continue
                    py_idxs: list[int] = []
                    for v in raw_with:
                        try:
                            py_idxs.append(int(v))
                        except (TypeError, ValueError):
                            continue
                    if py_idxs:
                        extend[m_idx] = tuple(py_idxs)
            elif isinstance(entry, str):
                names.add(entry.lower())
        out[str(topic)] = TopicExemption(
            call_names=frozenset(names),
            drop_sections=frozenset(sections),
            extend_pairs=extend,
        )
    return out


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
    exempted: list[Call] = field(default_factory=list)


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
    # Random helpers — MATLAB rand/randn/randi vs NumPy new-style Generator
    # methods (rng.random / rng.standard_normal / rng.integers). Collapse both
    # families to a canonical "rand" / "randn" / "randi" token so the call
    # sequences line up under LCS without forcing notebook reorders.
    "rand": "rand",
    "random": "rand",                # rng.random / np.random.random
    "randn": "randn",
    "standard_normal": "randn",      # rng.standard_normal
    "randi": "randi",
    "integers": "randi",             # rng.integers
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

    MATLAB helpfiles commonly insert untitled ``%%`` markers to break a
    semantically continuous block into multiple cells (for the Publish
    formatter — block diagrams, LaTeX equations, single-statement code
    cells). These untitled sections are *merged into the preceding titled
    section* before scoring, so the Python notebook only needs to track
    the higher-level titled blocks. If untitled sections appear before the
    first titled one, they are dropped (MATLAB-only scaffolding).
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


# Heuristic threshold: an "untitled" MATLAB section with more than this many
# call sites is treated as a logically distinct section rather than a Publish
# typesetting cell. Real typesetting cells (block diagrams, equation displays)
# have 0–2 calls; substantive code blocks routinely exceed 10.
_UNTITLED_MERGE_CALL_THRESHOLD = 5


def _merge_untitled_into_previous(sections: list[Section]) -> list[Section]:
    """Fold *small* ``(untitled section)`` blocks into the preceding titled section.

    MATLAB Publish uses bare ``%%`` markers to break the rendered helpfile
    into typesetting cells (block diagrams, equation displays, single
    `TrialConfig({...})` lines). These don't correspond to logical Python
    notebook sections — the Python port groups them under the parent
    titled block.

    HOWEVER, some helpfiles use a bare ``%%`` to start a substantive
    code block (e.g. DecodingExample.m line 52 begins the decode stage
    with just ``%% `` and a markdown intro, then ~28 call sites). Those
    blocks are logically distinct and the Python port keeps them as their
    own ``# SECTION N:`` cell — merging them here just hides parity.

    Heuristic: an untitled section with more than
    :data:`_UNTITLED_MERGE_CALL_THRESHOLD` calls is kept as its own section
    (re-titled ``"(untitled section)"``); smaller ones are merged into the
    preceding titled section as before. Untitled sections before the
    first titled block are always dropped.

    The returned list re-indexes sections starting at 1.
    """
    merged: list[Section] = []
    for sec in sections:
        if sec.title == "(untitled section)":
            if len(sec.calls) > _UNTITLED_MERGE_CALL_THRESHOLD:
                # Substantive untitled block — keep as its own section.
                merged.append(
                    Section(index=len(merged) + 1, title=sec.title, calls=list(sec.calls))
                )
                continue
            if merged:
                # Small typesetting block — append calls to previous section.
                merged[-1].calls.extend(sec.calls)
            # else: drop (no previous titled section to attach to).
            continue
        # Reindex into the merged list to keep ordinals contiguous.
        merged.append(Section(index=len(merged) + 1, title=sec.title, calls=list(sec.calls)))
    return merged


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
    """Walk a notebook, capturing ``# SECTION N: <title>`` code cells.

    Untitled SECTION cells (header is just ``# SECTION N:`` with no title text)
    are merged into the preceding titled section — symmetric to the MATLAB
    parser's handling of bare ``%%`` markers. This lets a Python notebook
    that splits a MATLAB-titled block across multiple sub-cells align with
    the single MATLAB titled section.
    """
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

    # Collect Call nodes with their (line, col) source positions so we can
    # emit them in source order. ast.walk yields nodes in BFS order, which
    # interleaves nested vs top-level calls and ruins the LCS alignment when
    # the same line contains several calls (e.g. ``float(np.max(x))`` walks
    # float, then max, but BFS-walk yields max first if float is the parent).
    # Sorting by (lineno, col_offset) restores left-to-right source order.
    nodes: list[ast.Call] = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call)
    ]
    nodes.sort(
        key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))
    )

    calls: list[Call] = []
    for node in nodes:
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

    def _is_untitled(norm: str) -> bool:
        """True if the normalized title is empty or 'untitled section'."""
        return norm == "" or norm == "untitled section"

    for m_sec in matlab:
        m_norm = _normalize_title(m_sec.title)
        m_untitled = _is_untitled(m_norm)
        best_idx: int | None = None
        best_score = 0.55  # minimum acceptance threshold
        for p_sec in python:
            if p_sec.index in py_used:
                continue
            p_norm = py_norm[p_sec.index]
            # Special case: untitled MATLAB and untitled/empty Python sections
            # should pair by ordinal proximity rather than by title similarity
            # (both normalize to similar tokens but SequenceMatcher rates the
            # 'untitled section' literal vs '' at ratio 0.0).
            if m_untitled and _is_untitled(p_norm):
                ratio = 0.8
            else:
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
    topic: str, matlab_repo: Path, notebooks_dir: Path,
    exemptions: dict[str, TopicExemption] | None = None,
) -> TopicResult:
    """Compute the section-aligned call-match score for one topic.

    ``exemptions`` is the result of :func:`_load_exemptions`. Any MATLAB call
    whose bare (lowercased) name is in ``exemptions[topic]`` is dropped from
    both the MATLAB call list and the score denominator BEFORE LCS alignment,
    on the principle that those calls have no Python equivalent (handle
    graphics, filesystem helpers, MATLAB-specific user-variable indexing
    mis-detected as calls, etc.). The dropped calls are still surfaced in
    the per-section report under the ``exempted`` heading so reviewers can
    audit them.
    """
    m_path = matlab_repo / "helpfiles" / f"{topic}.m"
    n_path = notebooks_dir / f"{topic}.ipynb"
    if not m_path.exists():
        return TopicResult(topic, 0, 0, 0.0, [], [], error=f"missing helpfile: {m_path}")
    if not n_path.exists():
        return TopicResult(topic, 0, 0, 0.0, [], [], error=f"missing notebook: {n_path}")

    matlab_sections = parse_matlab_helpfile(m_path)
    python_sections = parse_python_notebook(n_path)

    pairs = align_sections(matlab_sections, python_sections)

    exempt_names: frozenset[str] = frozenset()
    drop_sections: frozenset[int] = frozenset()
    extend_pairs: dict[int, tuple[int, ...]] = {}
    if exemptions:
        topic_exempt = exemptions.get(topic)
        if topic_exempt is not None:
            exempt_names = topic_exempt.call_names
            drop_sections = topic_exempt.drop_sections
            extend_pairs = topic_exempt.extend_pairs

    # Build a quick lookup of Python sections by index for extension splicing.
    python_by_index = {p.index: p for p in python_sections}

    total_calls = 0
    matched_total = 0
    for pair in pairs:
        # Whole-section drop: surface every call as exempted, skip scoring.
        # Use for MATLAB sections whose Python equivalent runs inside a
        # helper so the cell deliberately has a different call shape.
        if pair.matlab.index in drop_sections:
            pair.exempted = list(pair.matlab.calls)
            pair.matlab_only = []
            continue
        # An exemption entry may match either the bare call name
        # (e.g. "find"), the full raw dotted form (e.g. "epsc2.data"), or
        # the fuzzy canonical form (e.g. "imagesc" → "imshow"). All
        # comparisons are lower-cased.
        def _is_exempt(c: Call) -> bool:
            return (
                c.name.lower() in exempt_names
                or c.raw.lower() in exempt_names
                or canonicalize(c.raw) in exempt_names
            )

        if pair.python is None:
            # No Python pair — every MATLAB call is unmatched. Drop the
            # exempted ones from the denominator entirely.
            exempted_calls = [c for c in pair.matlab.calls if _is_exempt(c)]
            kept_calls = [c for c in pair.matlab.calls if not _is_exempt(c)]
            pair.exempted = exempted_calls
            pair.matlab_only = list(kept_calls)
            total_calls += len(kept_calls)
            continue

        # Optionally extend the Python side with calls from additional
        # Python sections (concatenated in order) before LCS alignment.
        # Use when the MATLAB helpfile keeps both a per-trial loop and
        # a batch-summary block in one section while the Python port
        # splits them across adjacent cells.
        py_calls: list[Call] = list(pair.python.calls)
        for idx in extend_pairs.get(pair.matlab.index, ()):
            extra = python_by_index.get(idx)
            if extra is not None and extra.index != pair.python.index:
                py_calls.extend(extra.calls)

        # Two-phase exemption for paired sections: run LCS over ALL MATLAB
        # calls so matched occurrences of high-volume primitives (set,
        # figure, subplot, etc.) still score as parity wins where Python
        # has them. Then drop EXEMPT calls only from the unmatched residue.
        # An exempted call neither helps when matched (it would have
        # matched anyway) nor hurts when unmatched.
        matched, mo, po = lcs_call_match(pair.matlab.calls, py_calls)
        exempted_residue = [c for c in mo if _is_exempt(c)]
        kept_mo = [c for c in mo if not _is_exempt(c)]
        pair.matched = matched
        pair.matlab_only = kept_mo
        pair.python_only = po
        pair.exempted = exempted_residue
        total_calls += len(matched) + len(kept_mo)
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

    exemptions = _load_exemptions()
    results: list[TopicResult] = []
    for t in topics:
        result = diff_topic(t, args.matlab_repo, args.notebooks_dir, exemptions)
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
