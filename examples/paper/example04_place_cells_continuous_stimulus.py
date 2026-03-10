#!/usr/bin/env python3
"""Example 04 — Place-Cell Receptive Fields (Gaussian vs Zernike).

This example demonstrates:
  1) Loading hippocampal place-cell data from two animals.
  2) Visualising spike locations overlaid on the animal's path.
  3) Fitting Gaussian and Zernike polynomial receptive-field models.
  4) Comparing model families via KS, AIC, and BIC statistics.
  5) Generating 2-D heatmaps and 3-D mesh plots of place fields.

Data provenance:
  Uses ``data/PlaceCellDataAnimal1.mat`` and ``data/PlaceCellDataAnimal2.mat``
  (position trajectories + multi-neuron spike times).

Expected outputs:
  - Figure 1: Example cells — spike locations over path (4 cells per animal).
  - Figure 2: Population model-comparison statistics (Delta-KS, Delta-AIC, Delta-BIC).
  - Figure 3: Gaussian receptive-field heatmaps (Animal 1).
  - Figure 4: Zernike receptive-field heatmaps (Animal 1).
  - Figure 5: Gaussian receptive-field heatmaps (Animal 2).
  - Figure 6: Zernike receptive-field heatmaps (Animal 2).
  - Figure 7: 3-D mesh comparison for selected example cells.

Paper mapping:
  Section 2.3.4 (place-cell continuous-stimulus analysis).
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
from nstat.paper_examples_full import run_experiment4  # noqa: E402
from nstat.paper_figures import export_named_paper_figures  # noqa: E402


def run_example04(*, export_figures: bool = False, export_dir: Path | None = None):
    """Run Example 04: Place-cell receptive fields.

    Analysis workflow (mirrors Matlab example04_place_cells_continuous_stimulus.m):
      1. Load PlaceCellDataAnimal1.mat and PlaceCellDataAnimal2.mat.
      2. For each animal, visualise 4 example neurons (spike locations on path).
      3. Load or compute precomputed fit results for all neurons.
      4. Compute per-neuron Delta-KS, Delta-AIC, Delta-BIC (Gaussian vs Zernike).
      5. Generate Gaussian receptive-field heatmaps for all neurons (both animals).
      6. Generate Zernike polynomial receptive-field heatmaps.
      7. Generate 3-D mesh comparison for selected example cells.
    """
    data_dir = ensure_example_data(download=True)

    # Run analysis (returns summary statistics and figure payload)
    summary, payload = run_experiment4(data_dir, return_payload=True)

    print(json.dumps(summary, indent=2))

    if export_figures:
        if export_dir is None:
            export_dir = THIS_DIR / "figures" / "example04"
        saved = export_named_paper_figures(
            "example04", summary=summary, payload=payload, export_dir=export_dir
        )
        print(f"\nGenerated {len(saved)} figure(s):")
        for p in saved:
            print(f"  {p}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 04: Place-Cell Receptive Fields"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_example04(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
