#!/usr/bin/env python3
"""Example 05 — Stimulus Decoding With PPAF and PPHF.

This example demonstrates:
  1) Univariate sinusoidal stimulus encoding and decoding via PPDecodeFilterLinear.
  2) 4-state arm-reach simulation with 20-cell population encoding.
  3) PPAF (Point-Process Adaptive Filter) decoding: free vs goal-informed.
  4) Hybrid filter (PPHybridFilterLinear) for joint discrete/continuous states.

The example has three parts:
  Part A (experiment5): Univariate sinusoidal stimulus — encode with 20
      neurons, decode with PPDecodeFilterLinear.
  Part B (experiment5b): 4-state arm reaching — simulate 20-cell population,
      compare PPAF vs PPAF+Goal across 20 simulations.
  Part C (experiment6): Hybrid filter — simulate 40-cell population with
      discrete reach states and continuous kinematics, decode with
      PPHybridFilterLinear.

Expected outputs:
  - Figure 1: Univariate stimulus setup (CIF tuning curves, simulated spikes).
  - Figure 2: Univariate decoding results (decoded stimulus vs true).
  - Figure 3: Reach setup and population encoding.
  - Figure 4: PPAF comparison (free vs goal-informed).
  - Figure 5: Hybrid filter setup.
  - Figure 6: Hybrid decoding summary.

Paper mapping:
  Section 2.5 (point-process adaptive filter) and Section 2.6 (hybrid filter).
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

from nstat.paper_examples_full import (  # noqa: E402
    run_experiment5,
    run_experiment5b,
    run_experiment6,
)
from nstat.paper_figures import export_named_paper_figures  # noqa: E402


def run_example05(*, export_figures: bool = False, export_dir: Path | None = None):
    """Run Example 05: PPAF and PPHF decoding.

    Analysis workflow (mirrors Matlab example05_decoding_ppaf_pphf.m):

    Part A — Univariate stimulus decoding:
      1. Define 20-cell population with sinusoidal tuning.
      2. Simulate spikes from sinusoidal stimulus CIF.
      3. Decode stimulus via PPDecodeFilterLinear.

    Part B — Arm-reach PPAF:
      4. Simulate 4-state reaching movements (position + velocity).
      5. Encode with 20-cell cosine-tuning population.
      6. Decode with PPAF (free) and PPAF+Goal; compare across 20 runs.

    Part C — Hybrid filter:
      7. Simulate 40-cell population with discrete reach-state modulation.
      8. Decode joint discrete/continuous state via PPHybridFilterLinear.
    """
    # --- Part A: Univariate sinusoidal stimulus ---
    summary5, payload5 = run_experiment5(return_payload=True)

    # --- Part B: Arm-reach PPAF ---
    summary5b, payload5b = run_experiment5b(return_payload=True)

    # --- Part C: Hybrid filter ---
    summary6, payload6 = run_experiment6(REPO_ROOT, return_payload=True)

    # Merge summaries for JSON output
    combined_summary = {
        "experiment5": summary5,
        "experiment5b": summary5b,
        "experiment6": summary6,
    }
    print(json.dumps(combined_summary, indent=2))

    if export_figures:
        if export_dir is None:
            export_dir = THIS_DIR / "figures" / "example05"
        combined_payload = {
            "experiment5": payload5,
            "experiment5b": payload5b,
            "experiment6": payload6,
        }
        saved = export_named_paper_figures(
            "example05",
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
        description="Example 05: Stimulus Decoding With PPAF and PPHF"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_example05(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
