#!/usr/bin/env python3
"""Print resolved nSTAT example-data directory."""

from __future__ import annotations

from pathlib import Path

from nstat.data_manager import get_data_dir


def main() -> int:
    print(str(Path(get_data_dir()).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

