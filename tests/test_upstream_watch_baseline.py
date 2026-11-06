"""Consistency checks for the parity-upstream-watch baseline.

``tools/parity/upstream_watch.py`` hashes ``TRACKED_FILES`` in the MATLAB
checkout and compares them with ``parity/upstream_watch_baseline.yml``.
A file listed in ``TRACKED_FILES`` but absent from the baseline makes every
weekly run report a spurious change; a baseline key that is not tracked is
reported as "deleted" forever.  These tests keep the two in step without
needing the MATLAB checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "tools" / "parity"
BASELINE = REPO_ROOT / "parity" / "upstream_watch_baseline.yml"

if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from upstream_watch import TRACKED_FILES  # noqa: E402


def _baseline() -> dict:
    return yaml.safe_load(BASELINE.read_text()) or {}


def test_tracked_files_have_no_duplicates() -> None:
    assert len(TRACKED_FILES) == len(set(TRACKED_FILES))


def test_baseline_keys_match_tracked_files() -> None:
    hashes = _baseline().get("hashes", {})
    assert list(hashes) == list(TRACKED_FILES)


def test_baseline_count_matches_tracked_files() -> None:
    assert _baseline().get("tracked_files") == len(TRACKED_FILES)


def test_collection_and_decoding_sources_are_tracked() -> None:
    # The Python mirrors of these MATLAB classes are covered by the class
    # parity audit, so an upstream edit to any of them must raise an alert.
    for rel in (
        "nstColl.m",
        "CovColl.m",
        "ConfigColl.m",
        "DecodingAlgorithms.m",
        "ConfidenceInterval.m",
        "LinearCIF.m",
    ):
        assert rel in TRACKED_FILES
        assert _baseline()["hashes"].get(rel), f"{rel} has no baseline hash"
