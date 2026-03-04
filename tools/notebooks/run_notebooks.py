#!/usr/bin/env python3
"""Backward-compatible wrapper for notebook execution."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("execute_notebooks.py")), run_name="__main__")
