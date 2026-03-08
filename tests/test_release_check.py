from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_TOOLS = REPO_ROOT / "tools" / "release"
if str(RELEASE_TOOLS) not in sys.path:
    sys.path.insert(0, str(RELEASE_TOOLS))

from release_gate_lib import build_release_gate_commands


def test_release_gate_is_pure_python() -> None:
    commands = build_release_gate_commands(REPO_ROOT)
    flattened = [" ".join(command) for command in commands]

    assert any("tests/test_cleanroom_boundary.py" in item for item in flattened)
    assert not any("matlab " in item.lower() for item in flattened)
    assert not any("export_matlab_gold_fixtures.py" in item for item in flattened)
