from __future__ import annotations

import re
from pathlib import Path


PATTERNS = [
    re.compile(r"\bmatlab\.engine\b"),
    re.compile(r"\bimport\s+matlab\b"),
    re.compile(r"\bfrom\s+matlab\b"),
]



def test_source_has_no_matlab_runtime_dependency() -> None:
    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in PATTERNS:
            assert not pattern.search(text), f"MATLAB dependency found in {path}: {pattern.pattern}"
