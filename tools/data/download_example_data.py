#!/usr/bin/env python3
"""Download/extract nSTAT example data (if needed) and print resolved data dir."""

from __future__ import annotations

from pathlib import Path

from nstat.data_manager import ensure_example_data


def main() -> int:
    path = ensure_example_data(download=True)
    print(str(Path(path).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

