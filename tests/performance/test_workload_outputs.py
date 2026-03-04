from __future__ import annotations

from nstat.performance_workloads import CASE_ORDER, TIER_ORDER, run_python_workload


def test_workloads_return_finite_metrics() -> None:
    for case in CASE_ORDER:
        for tier in TIER_ORDER:
            metrics = run_python_workload(case=case, tier=tier, seed=20260303)
            assert metrics, f"{case}/{tier} returned no metrics"
            for name, value in metrics.items():
                assert isinstance(value, float), f"{case}/{tier}:{name} must be float"
                assert value == value, f"{case}/{tier}:{name} is NaN"
                assert value != float("inf"), f"{case}/{tier}:{name} is inf"
