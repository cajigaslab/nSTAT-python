from __future__ import annotations

import os

import pytest

from nstat.performance_workloads import CASE_ORDER, run_python_workload

pytestmark = pytest.mark.skipif(
    os.getenv("NSTAT_RUN_PERF_BENCHMARKS", "0") != "1",
    reason="Performance benchmarks run only in dedicated CI jobs",
)


@pytest.mark.performance
@pytest.mark.parametrize("case", CASE_ORDER)
def test_benchmark_tier_s(benchmark: pytest.BenchmarkFixture, case: str) -> None:  # type: ignore[name-defined]
    summary = benchmark(run_python_workload, case, "S", 20260303)
    assert summary
    assert all(value == value for value in summary.values())
