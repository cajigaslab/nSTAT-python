"""Installation helpers for nSTAT-python.

`nstat_install` is a Python-side setup helper that prepares a local data cache
and returns deterministic setup metadata. This function is intentionally
MATLAB-free and serves as the recommended post-install check used in docs.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .datasets import get_cache_dir


@dataclass(frozen=True, slots=True)
class InstallReport:
    """Structured output from :func:`nstat_install`."""

    cache_dir: Path
    python_version: str
    package: str = "nstat"



def nstat_install(cache_dir: str | Path | None = None) -> InstallReport:
    """Run lightweight post-install setup for nSTAT-python.

    Parameters
    ----------
    cache_dir:
        Optional custom data-cache location.

    Returns
    -------
    InstallReport
        Report containing resolved cache directory and runtime metadata.
    """

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["NSTAT_DATA_CACHE"] = str(cache_path)

    resolved_cache = get_cache_dir()
    report = InstallReport(
        cache_dir=resolved_cache,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    return report



def main() -> int:
    parser = argparse.ArgumentParser(description="Run nSTAT-python post-install setup.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional custom data cache directory.",
    )
    args = parser.parse_args()

    report = nstat_install(cache_dir=args.cache_dir)
    print(f"nSTAT-python setup complete. Cache directory: {report.cache_dir}")
    print(f"Python runtime: {report.python_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
