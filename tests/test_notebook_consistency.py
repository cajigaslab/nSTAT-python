from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_help_notebook_consistency_manifest() -> None:
    cmd = [
        "python",
        "tools/notebooks/check_notebook_consistency.py",
        "--repo-root",
        str(REPO_ROOT),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
