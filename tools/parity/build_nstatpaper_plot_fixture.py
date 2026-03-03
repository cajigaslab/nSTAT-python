#!/usr/bin/env python3
"""Build a consolidated MATLAB-gold fixture for nSTATPaperExamples plot arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="nSTAT-python repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/parity/fixtures/matlab_gold/nSTATPaperExamples_plot_gold.mat"),
        help="Output fixture path (relative to repo root if not absolute).",
    )
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    fixture_root = repo_root / "tests" / "parity" / "fixtures" / "matlab_gold"
    output_path = _resolve(repo_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ppsim = loadmat(fixture_root / "PPSimExample_gold.mat")
    dec_hist = loadmat(fixture_root / "DecodingExampleWithHist_gold.mat")
    place = loadmat(fixture_root / "HippocampalPlaceCellExample_gold.mat")
    psth = loadmat(fixture_root / "PSTHEstimation_gold.mat")
    mepsc = loadmat(fixture_root / "mEPSCAnalysis_gold.mat")

    payload = {
        "expected_rate_pp": np.asarray(ppsim["expected_rate"], dtype=float),
        "expected_decoded_hist": np.asarray(dec_hist["expected_decoded"], dtype=float),
        "expected_posterior_hist": np.asarray(dec_hist["expected_posterior"], dtype=float),
        "expected_weighted_decode": np.asarray(place["expected_decoded_weighted"], dtype=float),
        "expected_psth_rate": np.asarray(psth["expected_rate_psth"], dtype=float),
        "expected_psth_prob": np.asarray(psth["expected_prob_psth"], dtype=float),
        "expected_psth_sig": np.asarray(psth["expected_sig_psth"], dtype=float),
        "trace_mepsc": np.asarray(mepsc["trace_mepsc"], dtype=float),
        "time_mepsc": np.asarray(mepsc["time_mepsc"], dtype=float),
        "event_times_mepsc": np.asarray(mepsc["event_times_mepsc"], dtype=float),
    }
    savemat(output_path, payload)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
