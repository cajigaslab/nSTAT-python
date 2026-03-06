from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path


def _write_lfs_pointer(path: Path, payload: bytes) -> None:
    digest = sha256(payload).hexdigest()
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest}\n"
        f"size {len(payload)}\n"
    )
    path.write_text(pointer, encoding="utf-8")


def test_hydrate_matlab_helpfile_assets_copies_verified_files(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    matlab_repo = tmp_path / "upstream-nstat-matlab"
    target_help = matlab_repo / "helpfiles"
    source_help = tmp_path / "materialized-helpfiles"
    report_json = tmp_path / "report.json"
    payload = b"mat-file-bytes"

    target_help.mkdir(parents=True)
    source_help.mkdir(parents=True)
    _write_lfs_pointer(target_help / "CovariateSample.mat", payload)
    (source_help / "CovariateSample.mat").write_bytes(payload)

    proc = subprocess.run(
        [
            "python3",
            "tools/reports/hydrate_matlab_helpfile_assets.py",
            "--matlab-repo",
            str(matlab_repo),
            "--source-help-root",
            str(source_help),
            "--report-json",
            str(report_json),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src:."},
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (target_help / "CovariateSample.mat").read_bytes() == payload
    payload_json = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload_json["status"] == "pass"
    assert payload_json["results"][0]["status"] == "hydrated"


def test_hydrate_matlab_helpfile_assets_fails_on_hash_mismatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    matlab_repo = tmp_path / "upstream-nstat-matlab"
    target_help = matlab_repo / "helpfiles"
    source_help = tmp_path / "materialized-helpfiles"
    report_json = tmp_path / "report.json"

    target_help.mkdir(parents=True)
    source_help.mkdir(parents=True)
    _write_lfs_pointer(target_help / "CovariateSample.mat", b"expected-payload")
    (source_help / "CovariateSample.mat").write_bytes(b"wrong-payload")

    proc = subprocess.run(
        [
            "python3",
            "tools/reports/hydrate_matlab_helpfile_assets.py",
            "--matlab-repo",
            str(matlab_repo),
            "--source-help-root",
            str(source_help),
            "--report-json",
            str(report_json),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src:."},
    )

    assert proc.returncode == 1
    payload_json = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload_json["status"] == "fail"
    assert payload_json["results"][0]["status"] == "hash_mismatch"
