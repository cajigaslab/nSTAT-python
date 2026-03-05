from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_port_inventory_and_core_mapping() -> None:
    cmd = [
        "python",
        "tools/parity/build_port_inventory.py",
        "--repo-root",
        str(REPO_ROOT),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    summary = json.loads((REPO_ROOT / "parity" / "port_inventory_summary.json").read_text(encoding="utf-8"))
    assert summary["core_missing"] == []
    assert summary["help_missing"] == []
    assert int(summary["total_entities"]) > 0
