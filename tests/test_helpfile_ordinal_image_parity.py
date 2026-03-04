from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest
import yaml


def _write_image(path: Path, value: float) -> None:
    arr = np.full((32, 32), value, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def _run_checker(
    manifest: Path,
    py_root: Path,
    mat_root: Path,
    out_json: Path,
    *,
    topics: str = "",
    threshold: str = "0.9",
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "python",
        "tools/reports/check_helpfile_ordinal_image_parity.py",
        "--manifest",
        str(manifest),
        "--python-image-root",
        str(py_root),
        "--matlab-image-root",
        str(mat_root),
        "--ssim-threshold",
        threshold,
        "--out-json",
        str(out_json),
    ]
    if topics:
        cmd.extend(["--topics", topics])
    return subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src:."},
    )


def test_ordinal_image_parity_passes_for_identical_pairs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yml"
    py_root = tmp_path / "python_images"
    mat_root = tmp_path / "matlab_images"
    out_json = tmp_path / "summary.json"

    _write_image(py_root / "TopicA" / "fig_001.png", 0.25)
    _write_image(mat_root / "TopicA" / "fig_001.png", 0.25)

    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "topics": [
                    {
                        "topic": "TopicA",
                        "expected_figure_count": 1,
                        "notebook_output_path": "notebooks/TopicA.ipynb",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _run_checker(manifest, py_root, mat_root, out_json)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"


def test_ordinal_image_parity_fails_on_count_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yml"
    py_root = tmp_path / "python_images"
    mat_root = tmp_path / "matlab_images"
    out_json = tmp_path / "summary.json"

    _write_image(py_root / "TopicA" / "fig_001.png", 0.25)
    _write_image(py_root / "TopicA" / "fig_002.png", 0.5)
    _write_image(mat_root / "TopicA" / "fig_001.png", 0.25)

    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "topics": [
                    {
                        "topic": "TopicA",
                        "expected_figure_count": 1,
                        "notebook_output_path": "notebooks/TopicA.ipynb",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _run_checker(manifest, py_root, mat_root, out_json)
    assert result.returncode == 1
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["failures"]


def test_analysisexamples_fig001_ssim_threshold_when_artifacts_present(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    py_img = repo_root / "output/notebook_images/AnalysisExamples/fig_001.png"
    mat_img = repo_root / "output/matlab_help_images/AnalysisExamples/fig_001.png"
    if not py_img.exists() or not mat_img.exists():
        pytest.skip("AnalysisExamples parity artifacts not present locally")

    manifest = repo_root / "parity/help_source_manifest.yml"
    out_json = tmp_path / "analysisexamples_summary.json"
    result = _run_checker(
        manifest,
        repo_root / "output/notebook_images",
        repo_root / "output/matlab_help_images",
        out_json,
        topics="AnalysisExamples",
        threshold="0.70",
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    topic_rows = [row for row in payload.get("topics", []) if row.get("topic") == "AnalysisExamples"]
    assert topic_rows, result.stdout + "\n" + result.stderr
    pairs = topic_rows[0].get("pairs", [])
    assert pairs, result.stdout + "\n" + result.stderr
    fig1 = next((pair for pair in pairs if int(pair.get("ordinal", -1)) == 1), None)
    assert fig1 is not None, result.stdout + "\n" + result.stderr
    assert float(fig1["score"]) >= 0.70, result.stdout + "\n" + result.stderr
