from __future__ import annotations

from pathlib import Path

from nstat.release_check import build_release_gate_commands


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_gate_includes_fixture_generation_and_matlab_suite() -> None:
    commands = build_release_gate_commands(REPO_ROOT, matlab_repo_root=REPO_ROOT.parent / "nSTAT")
    flattened = [" ".join(command) for command in commands]

    assert any("export_matlab_gold_fixtures.py" in item for item in flattened)
    assert any("tests/python_port_fidelity" in item for item in flattened)
    assert any("addpath(fullfile(pwd,'helpfiles'))" in item for item in flattened)


def test_release_gate_can_skip_matlab() -> None:
    commands = build_release_gate_commands(REPO_ROOT, skip_matlab=True)
    flattened = [" ".join(command) for command in commands]

    assert not any(item.startswith("matlab ") for item in flattened)
