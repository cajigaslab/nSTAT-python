from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_line_equivalence_policy() -> None:
    subprocess.run(
        ["python", "tools/parity/build_port_inventory.py", "--repo-root", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        check=True,
    )
    cmd = [
        "python",
        "tools/parity/line_by_line_equivalence.py",
        "--repo-root",
        str(REPO_ROOT),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    report = json.loads((REPO_ROOT / "parity" / "line_equivalence_report.json").read_text(encoding="utf-8"))
    assert int(report["total"]) > 0
    assert int(report["failures"]) == 0
