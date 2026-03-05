#!/usr/bin/env python3
"""Compatibility wrapper for source-derived help notebook generation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.notebooks.generate_helpfile_notebooks import main


if __name__ == "__main__":
    raise SystemExit(main())
