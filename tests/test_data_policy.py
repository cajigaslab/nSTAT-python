from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from nstat.datasets import fetch_dataset, fetch_matlab_gold_file, list_matlab_gold_files



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def test_shared_dataset_manifest_contains_mepsc_example() -> None:
    payload = json.loads(Path("data/datasets_manifest.json").read_text(encoding="utf-8"))
    dataset_names = {row["name"] for row in payload["datasets"]}
    assert "mEPSC-epsc2" in dataset_names


def test_datasets_manifest_contains_full_mirror_entries() -> None:
    payload = json.loads(Path("data/datasets_manifest.json").read_text(encoding="utf-8"))
    rows = payload["datasets"]
    by_name = {row["name"]: row for row in rows}

    mirror_manifest = json.loads(
        Path("data/shared/matlab_gold_20260302.manifest.json").read_text(encoding="utf-8")
    )
    files = mirror_manifest["files"]

    for row in files:
        rel = row["relative_path"]
        name = f"matlab_gold_20260302/{rel}"
        assert name in by_name, f"missing datasets manifest entry for {name}"
        ds = by_name[name]
        assert ds["filename"] == rel
        assert ds["sha256"] == row["sha256"]



def test_allowlisted_shared_data_file_matches_checksum() -> None:
    allowlist = yaml.safe_load(
        Path("tools/compliance/shared_data_allowlist.yml").read_text(encoding="utf-8")
    )

    for row in allowlist["shared_data"]:
        path = Path(row["python_path"])
        assert path.exists(), f"missing allowlisted data file: {path}"
        assert _sha256(path) == row["sha256"]


def test_fetch_dataset_prefers_local_matlab_mirror_for_mepsc() -> None:
    path = fetch_dataset("mEPSC-epsc2")
    resolved = path.resolve()
    assert "data/shared/matlab_gold_" in resolved.as_posix()
    assert resolved.name == "epsc2.txt"


def test_fetch_matlab_gold_file_and_listing_api() -> None:
    files = list_matlab_gold_files(version="20260302")
    assert "mEPSCs/epsc2.txt" in files
    assert "PlaceCellAnimal1Results.mat" in files

    path = fetch_matlab_gold_file("mEPSCs/epsc2.txt", version="20260302")
    assert path.exists()
    assert path.resolve().as_posix().endswith("/data/shared/matlab_gold_20260302/mEPSCs/epsc2.txt")
