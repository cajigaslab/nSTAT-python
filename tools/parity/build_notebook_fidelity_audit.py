#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat.notebook_fidelity_audit import write_notebook_fidelity_audit


def main() -> int:
    path = write_notebook_fidelity_audit(REPO_ROOT)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
