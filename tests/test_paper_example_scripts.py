from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "examples" / "paper" / "manifest.yml"

EXPECTED_SCRIPT_NAMES = {
    "example01_mepsc_poisson",
    "example02_whisker_stimulus_thalamus",
    "example03_psth_and_ssglm",
    "example04_place_cells_continuous_stimulus",
    "example05_decoding_ppaf_pphf",
}


def test_paper_example_manifest_covers_canonical_scripts() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = payload["examples"]
    names = {row["name"] for row in entries}
    assert names == EXPECTED_SCRIPT_NAMES
    assert [row["example_id"] for row in entries] == ["example01", "example02", "example03", "example04", "example05"]


def test_paper_example_scripts_exist_and_support_help() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    for row in payload["examples"]:
        script_path = REPO_ROOT / row["script"]
        assert script_path.exists(), f"Missing paper example script: {script_path}"
        proc = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--repo-root" in proc.stdout
        assert "--export-figures" in proc.stdout


def test_all_canonical_examples_support_figure_export(tmp_path: Path) -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    for row in payload["examples"][:5]:
        example_id = row["example_id"]
        script_path = REPO_ROOT / row["script"]
        export_dir = tmp_path / example_id
        proc = subprocess.run(
            [sys.executable, str(script_path), "--export-figures", "--export-dir", str(export_dir)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        for rel_path in row["figure_files"]:
            filename = Path(rel_path).name
            assert (export_dir / filename).exists(), f"Missing exported figure {filename} for {example_id}"
