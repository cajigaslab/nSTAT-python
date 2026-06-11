#!/usr/bin/env python3
"""Run a MATLAB nSTAT helpfile via the MATLAB Engine for Python.

Captures workspace variables and figure PNGs so the outputs can be
compared programmatically to Python notebook outputs.  This is the
helpfile-side of the parity validation harness.

Prerequisites
-------------
- MATLAB R2023b+ with the Python Engine for Python installed
  (``pip install /Applications/MATLAB_R<rel>.app/extern/engines/python/``).
- A local checkout of cajigaslab/nSTAT at the path given by
  ``--matlab-root`` (default: sibling ``../nSTAT``).

Outputs
-------
Each invocation writes to
``tests/parity/captured_matlab_helpfile_outputs/<helpfile>/``:
- ``workspace.mat`` — every numeric / string / struct variable left in
  the workspace after the helpfile finishes.
- ``fig_NN.png`` — each open MATLAB figure saved as PNG, numbered by
  figure handle order.
- ``stdout.txt`` — captured ``diary`` output.

Usage
-----
    python tools/parity/matlab/run_helpfile.py AnalysisExamples
    python tools/parity/matlab/run_helpfile.py CovariateExamples --headless

Useful for:
- Validating Python notebook numeric outputs against MATLAB ground truth
  without needing to run MATLAB interactively.
- Refreshing visual-parity baselines (the captured figures feed into
  ``tools/parity/build_visual_comparison.py`` as the MATLAB side).
- Identifying MATLAB-side bugs (M14-M21) by direct execution rather
  than code reading.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATLAB_ROOT = REPO_ROOT.parent / "nSTAT"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tests" / "parity" / "captured_matlab_helpfile_outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "helpfile",
        help="Name of the helpfile (without .m), e.g. 'AnalysisExamples'.",
    )
    p.add_argument(
        "--matlab-root",
        type=Path,
        default=DEFAULT_MATLAB_ROOT,
        help="MATLAB nSTAT checkout root (default: sibling ../nSTAT).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination for captured outputs.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Hide all MATLAB figure windows (still captures PNGs).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-helpfile MATLAB execution timeout in seconds.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import matlab.engine
    except ImportError as exc:
        print(
            f"ERROR: matlab.engine not installed.  Run "
            f"`pip install /Applications/MATLAB_R<rel>.app/extern/engines/python/`",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    helpfile_path = args.matlab_root / "helpfiles" / f"{args.helpfile}.m"
    if not helpfile_path.exists():
        print(f"ERROR: helpfile not found: {helpfile_path}", file=sys.stderr)
        return 1

    out_dir = args.output_root / args.helpfile
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting MATLAB engine (~10-20s)...")
    eng = matlab.engine.start_matlab("-nodesktop" if args.headless else "")
    print(f"OK, MATLAB R{eng.version('-release')!s}")
    try:
        # Set up path
        eng.addpath(str(args.matlab_root), nargout=0)
        eng.addpath(eng.genpath(str(args.matlab_root)), nargout=0)
        eng.cd(str(args.matlab_root / "helpfiles"), nargout=0)

        # Force headless if requested
        if args.headless:
            eng.eval("set(groot, 'DefaultFigureVisible', 'off');", nargout=0)

        # Start diary
        stdout_path = out_dir / "stdout.txt"
        eng.diary(str(stdout_path), nargout=0)

        # Read the .m file as text and ``eval`` it directly.  The
        # straightforward ``run(<name>)`` is blocked when MATLAB sees a
        # sibling ``<name>.mlx`` Live Script that "shadows" the .m file
        # (e.g. all helpfiles in cajigaslab/nSTAT).  ``eval(fileread(...))``
        # bypasses the shadowing check.
        print(f"Running {args.helpfile}.m (timeout {args.timeout}s)...")
        eng.eval(f"eval(fileread('{helpfile_path}'));", nargout=0)

        # Close diary
        eng.diary("off", nargout=0)

        # Save workspace
        workspace_path = out_dir / "workspace.mat"
        # ``save('-v7')`` is the format scipy.io.loadmat can read most reliably.
        eng.eval(f"save('{workspace_path}', '-v7');", nargout=0)
        print(f"Saved workspace → {workspace_path}")

        # Save each open figure as PNG
        figure_handles = eng.eval("get(0, 'Children')", nargout=1)
        # ``get(0, 'Children')`` returns numeric figure handles in a column
        n_figures = int(eng.eval("numel(get(0, 'Children'))", nargout=1))
        print(f"Capturing {n_figures} figure(s)...")
        for i in range(1, n_figures + 1):
            fig_path = out_dir / f"fig_{i:02d}.png"
            eng.eval(
                f"hFig = sort(get(0, 'Children')); "
                f"saveas(hFig({i}), '{fig_path}');",
                nargout=0,
            )
            print(f"  fig_{i:02d}.png saved")

        # Close all figures
        eng.eval("close all force; clear all;", nargout=0)
        print(f"\nDone.  Captures at {out_dir}")
        return 0
    finally:
        eng.quit()


if __name__ == "__main__":
    sys.exit(main())
