from __future__ import annotations

import json
from pathlib import Path

import yaml

from nstat.notebook_figures import FigureTracker
from nstat.notebook_parity import extract_figure_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_MANIFEST_PATH = REPO_ROOT / "tools" / "notebooks" / "notebook_manifest.yml"


def test_notebook_figure_tracker_contracts_match_manifest_topics() -> None:
    payload = yaml.safe_load(NOTEBOOK_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    for row in payload.get("notebooks", []):
        notebook_path = REPO_ROOT / row["file"]
        contract = extract_figure_contract(notebook_path)
        if contract is None:
            continue
        assert contract.topic == row["topic"], f"{notebook_path} tracker topic drifted from notebook manifest"
        assert contract.expected_count >= 0
        assert contract.has_finalize_call, f"{notebook_path} uses FigureTracker but does not finalize it"


def test_figure_tracker_writes_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "notebook_images"
    tracker = FigureTracker(topic="ArtifactContractTest", output_root=output_root, expected_count=2)
    tracker.new_figure("first")
    tracker.new_figure("second")
    tracker.finalize()

    topic_dir = output_root / "ArtifactContractTest"
    manifest_path = topic_dir / "manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["topic"] == "ArtifactContractTest"
    assert payload["expected_count"] == 2
    assert payload["produced_count"] == 2
    assert payload["images"] == ["fig_001.png", "fig_002.png"]
