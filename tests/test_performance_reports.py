from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


LINUX_BASELINE_DATED = Path("tests/performance/fixtures/python/performance_baseline_linux_20260304.json")
LINUX_BASELINE = Path("tests/performance/fixtures/python/performance_baseline_linux_latest.json")


def test_linux_latest_baseline_matches_dated_snapshot() -> None:
    latest = _load(LINUX_BASELINE)
    dated = _load(LINUX_BASELINE_DATED)
    assert latest == dated


def test_performance_fixture_coverage() -> None:
    matlab = _load(Path("tests/performance/fixtures/matlab/performance_baseline_470fde8.json"))
    python = _load(LINUX_BASELINE)

    matlab_pairs = {(row["case"], row["tier"]) for row in matlab["cases"]}
    python_pairs = {(row["case"], row["tier"]) for row in python["cases"]}
    assert matlab_pairs.issubset(python_pairs)
    assert len(matlab_pairs) == 15
    assert len(python_pairs) >= len(matlab_pairs)


def test_performance_comparator_runs(tmp_path: Path) -> None:
    out_json = tmp_path / "perf_report.json"
    out_csv = tmp_path / "perf_report.csv"
    cmd = [
        "python",
        "tools/performance/compare_matlab_python_performance.py",
        "--python-report",
        str(LINUX_BASELINE),
        "--matlab-report",
        "tests/performance/fixtures/matlab/performance_baseline_470fde8.json",
        "--policy",
        "parity/performance_gate_policy.yml",
        "--previous-python-report",
        str(LINUX_BASELINE),
        "--report-out",
        str(out_json),
        "--csv-out",
        str(out_csv),
        "--fail-on-regression",
        "--require-regression-env-match",
    ]
    subprocess.run(cmd, check=True)

    report = _load(out_json)
    assert report["counts"]["total_case_tiers"] >= 15
    assert report["counts"]["regression_failures"] == 0
    assert len(report["top_python_vs_matlab_gaps"]) <= 5


def test_performance_comparator_skips_regression_on_env_mismatch(tmp_path: Path) -> None:
    python_report = _load(LINUX_BASELINE)
    previous_report = _load(LINUX_BASELINE)

    # Force a would-be regression while also making previous env non-comparable.
    python_report["cases"][0]["median_runtime_ms"] = float(python_report["cases"][0]["median_runtime_ms"]) * 5.0
    previous_report["environment"]["platform"] = "Linux-test-x86_64"
    previous_report["environment"]["python"] = "3.11.9"

    python_path = tmp_path / "python_report.json"
    previous_path = tmp_path / "previous_report.json"
    python_path.write_text(json.dumps(python_report), encoding="utf-8")
    previous_path.write_text(json.dumps(previous_report), encoding="utf-8")

    out_json = tmp_path / "perf_report_env_mismatch.json"
    out_csv = tmp_path / "perf_report_env_mismatch.csv"
    cmd = [
        "python",
        "tools/performance/compare_matlab_python_performance.py",
        "--python-report",
        str(python_path),
        "--matlab-report",
        "tests/performance/fixtures/matlab/performance_baseline_470fde8.json",
        "--policy",
        "parity/performance_gate_policy.yml",
        "--previous-python-report",
        str(previous_path),
        "--report-out",
        str(out_json),
        "--csv-out",
        str(out_csv),
        "--fail-on-regression",
    ]
    subprocess.run(cmd, check=True)

    report = _load(out_json)
    assert report["policy"]["regression_env_compatible"] is False
    assert report["counts"]["regression_failures"] == 0


def test_performance_comparator_can_require_env_match(tmp_path: Path) -> None:
    python_report = _load(LINUX_BASELINE)
    previous_report = _load(LINUX_BASELINE)

    previous_report["environment"]["platform"] = "Linux-test-x86_64"
    previous_report["environment"]["python"] = "3.11.9"

    python_path = tmp_path / "python_report.json"
    previous_path = tmp_path / "previous_report.json"
    python_path.write_text(json.dumps(python_report), encoding="utf-8")
    previous_path.write_text(json.dumps(previous_report), encoding="utf-8")

    out_json = tmp_path / "perf_report_env_required.json"
    out_csv = tmp_path / "perf_report_env_required.csv"
    cmd = [
        "python",
        "tools/performance/compare_matlab_python_performance.py",
        "--python-report",
        str(python_path),
        "--matlab-report",
        "tests/performance/fixtures/matlab/performance_baseline_470fde8.json",
        "--policy",
        "parity/performance_gate_policy.yml",
        "--previous-python-report",
        str(previous_path),
        "--report-out",
        str(out_json),
        "--csv-out",
        str(out_csv),
        "--fail-on-regression",
        "--require-regression-env-match",
    ]
    proc = subprocess.run(cmd, check=False)
    assert proc.returncode != 0
