#!/usr/bin/env python3
"""Example 03 — PSTH and State-Space GLM Dynamics.

This example demonstrates:
  1) Simulating spike trains from a known sinusoidal CIF.
  2) Computing PSTH (histogram) and comparing with GLM-PSTH.
  3) State-space GLM (SSGLM) estimation with EM algorithm.
  4) Across-trial learning dynamics and stimulus-effect surfaces.

The example has two parts:
  Part A (experiment3): PSTH analysis — simulate 20 trials from sinusoidal
      CIF, load real data from ``data/PSTH/Results.mat``, compare histogram
      PSTH vs GLM-PSTH.
  Part B (experiment3b): SSGLM analysis — simulate 50-trial dataset with
      across-trial gain modulation, fit SSGLM via EM, visualise learning
      dynamics and 3-D stimulus-effect surfaces.

Expected outputs:
  - Figure 1: Simulated and real rasters.
  - Figure 2: PSTH comparison (histogram vs GLM).
  - Figure 3: SSGLM simulation summary.
  - Figure 4: SSGLM fit diagnostics.
  - Figure 5: Stimulus-effect surfaces (3-D).
  - Figure 6: Learning-trial comparison.

Paper mapping:
  Section 2.3.3 (PSTH) and Section 2.4 (SSGLM).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat.data_manager import ensure_example_data  # noqa: E402
from nstat.paper_examples_full import run_experiment3, run_experiment3b  # noqa: E402
from nstat.paper_figures import export_named_paper_figures  # noqa: E402


def run_example03(*, export_figures: bool = False, export_dir: Path | None = None):
    """Run Example 03: PSTH and SSGLM dynamics.

    Analysis workflow (mirrors Matlab example03_psth_and_ssglm.m):

    Part A — PSTH:
      1. Define sinusoidal CIF: lambda(t) = exp(b0 + b1*cos(2*pi*f*t)).
      2. Simulate 20 spike trains via CIF thinning.
      3. Load real multi-trial data from PSTH/Results.mat.
      4. Compute histogram PSTH and GLM-PSTH; compare.

    Part B — SSGLM:
      5. Simulate 50-trial population with across-trial stimulus gain.
      6. Fit SSGLM via EM (forward-backward Kalman + Newton M-step).
      7. Plot per-trial coefficient trajectories and confidence bands.
      8. Generate 3-D stimulus-effect surface and learning-trial figure.
    """
    data_dir = ensure_example_data(download=True)

    # --- Part A: PSTH analysis ---
    summary3, payload3 = run_experiment3(return_payload=True)

    # --- Part B: SSGLM analysis ---
    summary3b, payload3b = run_experiment3b(data_dir, return_payload=True)

    # Merge summaries for JSON output
    combined_summary = {
        "experiment3": summary3,
        "experiment3b": summary3b,
    }
    print(json.dumps(combined_summary, indent=2))

    if export_figures:
        if export_dir is None:
            export_dir = THIS_DIR / "figures" / "example03"
        # Figure generation needs the combined dicts (multi-section example)
        combined_payload = {
            "experiment3": payload3,
            "experiment3b": payload3b,
        }
        saved = export_named_paper_figures(
            "example03",
            summary=combined_summary,
            payload=combined_payload,
            export_dir=export_dir,
        )
        print(f"\nGenerated {len(saved)} figure(s):")
        for p in saved:
            print(f"  {p}")

    return combined_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 03: PSTH and SSGLM Dynamics"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_example03(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
