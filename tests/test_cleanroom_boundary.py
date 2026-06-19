from __future__ import annotations

import re
from pathlib import Path

import nstat


REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_RUNTIME_PATTERNS = [
    re.compile(r"\bimport\s+matlab\b"),
    re.compile(r"\bimport\s+matlab\.engine\b"),
    re.compile(r"\bmatlab\.engine\b"),
    re.compile(r"\bstart_matlab\b"),
    re.compile(r"\bsubprocess\.(run|Popen)\([^\\n]*matlab"),
    re.compile(r"\bshutil\.which\(['\"]matlab['\"]\)"),
]

# matlab_engine.py is the *official* MATLAB Engine bridge module — it is
# *allowed* to import matlab.engine.  All other package files must remain
# cleanroom (no MATLAB runtime dependency).
BRIDGE_MODULE_ALLOWLIST = {"matlab_engine.py"}

# ``tools/parity/matlab/`` houses the MATLAB-side parity tooling: it is
# deliberately MATLAB-dependent (regenerates gold fixtures, runs MATLAB
# helpfiles against the corresponding Python code).  These scripts are
# not shipped with the installable package and run only on developers
# who have MATLAB.  Skipped from the cleanroom scan.
PARITY_MATLAB_TOOLS_DIR = REPO_ROOT / "tools" / "parity" / "matlab"

# Per-file allowlist of developer-only parity tools that legitimately
# shell out to MATLAB.  These are NOT shipped with the installable
# package and run only on developers who have MATLAB + the local nSTAT
# checkout.  Add new entries here only when the script's sole purpose
# is parity measurement (timing, fixture capture, diff).
DEVELOPER_PARITY_TOOLS_ALLOWLIST = {
    # v11 iter 53: performance-parity baseline harness — invokes
    # /opt/homebrew/bin/matlab -batch with a tic/toc script and parses
    # elapsed seconds from stdout.  Documented in
    # docs/parity/runbook.md "Performance parity".
    "perf_check.py",
}


def _assert_clean(paths: list[Path], *, allowlist: set[str] | None = None) -> None:
    allowlist = allowlist or set()
    violations: list[str] = []
    for path in paths:
        if path.name in allowlist:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_RUNTIME_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPO_ROOT)} :: {pattern.pattern}")
    assert not violations, "Disallowed MATLAB runtime references found:\n" + "\n".join(violations)


def test_installable_package_has_no_matlab_runtime_dependency() -> None:
    package_paths = sorted((REPO_ROOT / "nstat").glob("**/*.py"))
    _assert_clean(package_paths, allowlist=BRIDGE_MODULE_ALLOWLIST)


def test_notebooks_examples_and_ci_do_not_shell_out_to_matlab() -> None:
    targets = sorted((REPO_ROOT / "examples").glob("**/*.py"))
    # Exclude tools/parity/matlab/ — the MATLAB-side parity tooling is
    # deliberately MATLAB-dependent (regenerates gold fixtures, runs
    # helpfiles).  See PARITY_MATLAB_TOOLS_DIR.
    targets += [
        p
        for p in sorted((REPO_ROOT / "tools").glob("**/*.py"))
        if PARITY_MATLAB_TOOLS_DIR not in p.parents
        and p.name not in DEVELOPER_PARITY_TOOLS_ALLOWLIST
    ]
    targets += sorted((REPO_ROOT / "notebooks").glob("**/*.ipynb"))
    targets += sorted((REPO_ROOT / ".github" / "workflows").glob("**/*.yml"))
    _assert_clean(targets)


def test_package_root_does_not_export_matlab_reference_helpers() -> None:
    for name in ("matlab_engine_available", "run_point_process_reference", "run_simulated_network_reference"):
        assert not hasattr(nstat, name), f"nstat should not export MATLAB reference helper {name}"
