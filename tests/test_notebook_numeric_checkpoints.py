from __future__ import annotations

from pathlib import Path

import nbformat
import yaml


MANIFEST = Path("tools/notebooks/notebook_manifest.yml")


def test_all_notebooks_define_numeric_checkpoints_and_validation_call() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for row in manifest.get("notebooks", []):
        topic = str(row["topic"])
        nb_path = Path(str(row["file"]))
        nb = nbformat.read(nb_path, as_version=4)
        code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

        assert "CHECKPOINT_METRICS" in code, f"{topic}: missing CHECKPOINT_METRICS"
        assert "CHECKPOINT_LIMITS" in code, f"{topic}: missing CHECKPOINT_LIMITS"
        assert "validate_numeric_checkpoints(CHECKPOINT_METRICS, CHECKPOINT_LIMITS, TOPIC)" in code, (
            f"{topic}: missing numeric checkpoint validation call"
        )
