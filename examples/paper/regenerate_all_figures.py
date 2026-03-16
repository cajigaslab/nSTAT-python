#!/usr/bin/env python3
"""Regenerate all paper example figures.

Runs each example script with --export-figures and saves PNGs to
docs/figures/exampleNN/.  Called by CI on every push and can also be
run locally:

    python examples/paper/regenerate_all_figures.py
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
FIGURES_ROOT = REPO_ROOT / "docs" / "figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES = [
    ("example01_mepsc_poisson", "example01", "run_example01"),
    ("example02_whisker_stimulus_thalamus", "example02", "run_example02"),
    ("example03_psth_and_ssglm", "example03", "run_example03"),
    ("example04_place_cells_continuous_stimulus", "example04", "run_example04"),
    ("example05_decoding_ppaf_pphf", "example05", "run_example05"),
]


def main() -> int:
    # Ensure example data is available
    from nstat.data_manager import ensure_example_data
    ensure_example_data(download=True)

    failed = 0
    for mod_name, dir_name, run_fn_name in EXAMPLES:
        export_dir = FIGURES_ROOT / dir_name
        print(f"\n{'='*60}")
        print(f"  {mod_name}")
        print(f"{'='*60}")
        try:
            mod = importlib.import_module(
                f"examples.paper.{mod_name}"
            )
            run_fn = getattr(mod, run_fn_name)
            run_fn(export_figures=True, export_dir=export_dir)
            print(f"  OK: figures saved to {export_dir}")
        except Exception:
            traceback.print_exc()
            failed += 1

    total = len(EXAMPLES)
    print(f"\n=== Done. {total - failed}/{total} examples succeeded ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
