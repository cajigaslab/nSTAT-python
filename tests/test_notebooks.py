from __future__ import annotations

from pathlib import Path

import nbformat
import yaml



def test_notebook_manifest_entries_exist() -> None:
    manifest = yaml.safe_load(Path("tools/notebooks/notebook_manifest.yml").read_text(encoding="utf-8"))
    for row in manifest["notebooks"]:
        path = Path(row["file"])
        assert path.exists(), f"missing notebook file: {path}"



def test_notebook_metadata_matches_manifest() -> None:
    manifest = yaml.safe_load(Path("tools/notebooks/notebook_manifest.yml").read_text(encoding="utf-8"))
    for row in manifest["notebooks"]:
        path = Path(row["file"])
        nb = nbformat.read(path, as_version=4)
        meta = nb.metadata.get("nstat", {})
        assert meta.get("topic") == row["topic"]
        assert meta.get("run_group") == row["run_group"]
