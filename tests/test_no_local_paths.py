"""Guard against local absolute paths leaking into tracked files.

2026-10 EM / release cycle: a scratch path landed in a public docstring, a
local MATLAB checkout path landed in ``parity/matlab_defects.yml``, and local
machine paths leaked into several *generated* artifacts
(``parity/visual_fidelity_results.json``, ``docs/parity/visual_comparison.md``,
``parity/matlab_audit_xref.md``) — none of it caught by any local gate.
``git grep -I -E "/Users/[a-z]|/private/tmp/|scratchpad"`` found ~35 hits.

This test scans every tracked text file for absolute user/tmp paths and dev
scratch markers, including Jupyter notebook cell *outputs* (not just
*source*) — a committed output can leak a path just as easily as source
does, as `notebooks/nSTATPaperExamples.ipynb` did (a printed dataset-root
dict, and a matplotlib warning whose default formatting embeds the
triggering module's absolute path). Fix the underlying cause (print a
relative/short value; suppress or relocate the warning) and re-execute the
notebook through the repo's execution path so the committed output is
clean, rather than exempting outputs from this guard.

Fix leaks at the source, never here:
  - hand-edited file -> generic wording / an env-var name
    (``$NSTAT_MATLAB_PATH``), or a sibling-checkout-relative default.
  - generated file -> fix the generator, then regenerate with the repo tool
    (``make regen`` / the specific ``tools/parity/build_*.py``); never
    hand-edit a regenerated artifact.

Only genuinely historical, dated documents may be allowlisted below, with a
one-line justification per entry — not as a way to avoid fixing new leaks.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()

# Matches /Users/<name>, /home/<name>, /private/tmp/..., /tmp/claude...,
# C:\Users / C:\\Users (JSON-escaped) / C:/Users, the dev scratchpad
# directory name, and the <SCRATCH> placeholder token used in run briefs.
_LOCAL_PATH_RE = re.compile(
    r"/Users/[A-Za-z]"
    r"|/home/[A-Za-z]"
    r"|/private/tmp/"
    r"|/tmp/claude"
    r"|[Ss]cratchpad"
    r"|<SCRATCH>"
    r"|C:\\\\?Users"
    r"|C:/Users"
)

# Dated historical documents that record a specific past investigation
# session (its working paths are part of the historical record, not a
# leak to fix). Every entry needs a one-line justification.
_ALLOWLIST: dict[str, str] = {
    "proposals/2026-07-04-extras-neuroscience-grounding-plan.md": (
        "dated proposal recording a literal verification command run from "
        "the maintainer's checkout at the time; historical record, not a "
        "live instruction."
    ),
    "proposals/2026-10-04-codebase-review-findings.json": (
        "dated codebase-review findings; 'verification_method' fields record "
        "the literal scratch scripts a past review session wrote and ran, "
        "kept verbatim as the audit trail."
    ),
}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8192]
    except OSError:
        return True
    return b"\x00" in chunk


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _notebook_source_hits(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    hits: list[str] = []
    for ci, cell in enumerate(payload.get("cells", [])):
        source_text = _as_text(cell.get("source", ""))
        for lineno, line in enumerate(source_text.splitlines(), start=1):
            if _LOCAL_PATH_RE.search(line):
                hits.append(f"cell {ci} source line {lineno}: {line.strip()[:160]}")
        for oi, output in enumerate(cell.get("outputs", []) or []):
            texts = [_as_text(output.get("text"))]
            data = output.get("data")
            if isinstance(data, dict):
                for key in ("text/plain", "text/html"):
                    if key in data:
                        texts.append(_as_text(data[key]))
            ename = output.get("ename")
            if ename:
                texts.append(_as_text(ename))
            texts.append(_as_text(output.get("evalue")))
            texts.append(_as_text(output.get("traceback")))
            for text in texts:
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if _LOCAL_PATH_RE.search(line):
                        hits.append(
                            f"cell {ci} output {oi} line {lineno}: {line.strip()[:160]}"
                        )
    return hits


def _plain_file_hits(path: Path) -> list[str]:
    hits: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _LOCAL_PATH_RE.search(line):
            hits.append(f"line {lineno}: {line.strip()[:160]}")
    return hits


def _collect_violations() -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for rel in _tracked_files():
        if rel == THIS_FILE or rel in _ALLOWLIST:
            continue
        path = REPO_ROOT / rel
        if not path.is_file() or _is_binary(path):
            continue
        if rel.endswith(".ipynb"):
            hits = _notebook_source_hits(path)
        else:
            hits = _plain_file_hits(path)
        if hits:
            violations[rel] = hits
    return violations


def test_no_local_paths_in_tracked_files() -> None:
    violations = _collect_violations()
    if violations:
        lines = ["Local absolute paths / scratch markers leaked into tracked files:"]
        for rel, hits in sorted(violations.items()):
            lines.append(f"  {rel}:")
            for hit in hits[:5]:
                lines.append(f"    {hit}")
        lines.append(
            "\nFix at the source (generic wording / $NSTAT_MATLAB_PATH for "
            "hand-edited files; fix-the-generator-then-regen for generated "
            "artifacts). Only dated historical documents may be added to "
            "the _ALLOWLIST in this test, with a justification."
        )
        pytest.fail("\n".join(lines))


def test_allowlist_entries_still_exist_and_are_tracked() -> None:
    # Keeps the allowlist from silently rotting into dead entries that hide
    # a future, unrelated leak in the same file.
    tracked = set(_tracked_files())
    for rel in _ALLOWLIST:
        assert rel in tracked, f"allowlisted path {rel!r} is no longer tracked"
