#!/usr/bin/env python3
"""Example 02 — Whisker Stimulus GLM With Lag and History Selection.

This example demonstrates:
  1) Fitting an explicit-stimulus point-process GLM to thalamic spike data.
  2) Cross-correlation analysis to identify optimal stimulus lag.
  3) History-order selection via AIC/BIC sweeps.
  4) Model comparison: baseline vs stimulus vs stimulus+history.

Data provenance:
  Uses ``data/Explicit Stimulus/Dir3/Neuron1/Stim2/trngdataBis.mat``
  (whisker displacement ``t``, binary spike indicator ``y``, 1000 Hz).

Expected outputs:
  - Figure 1: Data overview (raster, stimulus, velocity).
  - Figure 2: Lag selection (CCF), history diagnostics, KS plot, coefficients.

Paper mapping:
  Section 2.3.2 (thalamic whisker-stimulus analysis).
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
from nstat.paper_examples_full import run_experiment2  # noqa: E402
from nstat.paper_figures import export_named_paper_figures  # noqa: E402


def run_example02(*, export_figures: bool = False, export_dir: Path | None = None):
    """Run Example 02: Whisker stimulus GLM.

    Analysis workflow (mirrors Matlab example02_whisker_stimulus_thalamus.m):
      1. Load trngdataBis.mat — stimulus displacement and spike indicator.
      2. Compute cross-covariance between residual spikes and stimulus.
      3. Identify peak lag; shift stimulus by optimal lag.
      4. Fit 3 nested GLMs:
         (a) baseline only,
         (b) baseline + stimulus + velocity,
         (c) baseline + stimulus + velocity + spike history.
      5. Sweep history orders 1..28 via AIC/BIC to select optimal lag.
      6. Generate figures comparing models.
    """
    data_dir = ensure_example_data(download=True)

    # Run analysis (returns summary statistics and figure payload)
    summary, payload = run_experiment2(data_dir, return_payload=True)

    print(json.dumps(summary, indent=2))

    if export_figures:
        if export_dir is None:
            export_dir = THIS_DIR / "figures" / "example02"
        saved = export_named_paper_figures(
            "example02", summary=summary, payload=payload, export_dir=export_dir
        )
        print(f"\nGenerated {len(saved)} figure(s):")
        for p in saved:
            print(f"  {p}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 02: Whisker Stimulus GLM")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_example02(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
