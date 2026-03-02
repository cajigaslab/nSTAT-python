from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from nstat.datasets import fetch_dataset



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
