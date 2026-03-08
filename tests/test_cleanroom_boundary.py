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


def _assert_clean(paths: list[Path]) -> None:
    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_RUNTIME_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPO_ROOT)} :: {pattern.pattern}")
    assert not violations, "Disallowed MATLAB runtime references found:\n" + "\n".join(violations)


def test_installable_package_has_no_matlab_runtime_dependency() -> None:
    package_paths = sorted((REPO_ROOT / "nstat").glob("**/*.py"))
    _assert_clean(package_paths)


def test_notebooks_examples_and_ci_do_not_shell_out_to_matlab() -> None:
    targets = sorted((REPO_ROOT / "examples").glob("**/*.py"))
    targets += sorted((REPO_ROOT / "tools").glob("**/*.py"))
    targets += sorted((REPO_ROOT / "notebooks").glob("**/*.ipynb"))
    targets += sorted((REPO_ROOT / ".github" / "workflows").glob("**/*.yml"))
    _assert_clean(targets)


def test_package_root_does_not_export_matlab_reference_helpers() -> None:
    for name in ("matlab_engine_available", "run_point_process_reference", "run_simulated_network_reference"):
        assert not hasattr(nstat, name), f"nstat should not export MATLAB reference helper {name}"
