from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml

try:  # optional fixture support
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "parity" / "function_contracts.yml"
FIXTURE_ROOT = REPO_ROOT / "tests" / "parity" / "fixtures" / "matlab_gold" / "functions"


def _resolve_symbol(dotted: str):
    parts = dotted.split(".")
    for split_idx in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_idx])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for attr in parts[split_idx:]:
            obj = getattr(obj, attr)
        return obj
    raise ModuleNotFoundError(f"Unable to resolve symbol path: {dotted}")


def _factory(name: str):
    if name == "psth_small":
        from nstat.nspikeTrain import nspikeTrain

        trains = [nspikeTrain(np.array([0.1, 0.4])), nspikeTrain(np.array([0.2]))]
        edges = np.array([0.0, 0.25, 0.5], dtype=float)
        return (trains, edges)
    if name == "linear_decode_small":
        x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        return (x, y)
    if name == "kalman_scalar":
        obs = np.array([[1.0], [0.5], [0.2]], dtype=float)
        a = np.array([[1.0]], dtype=float)
        h = np.array([[1.0]], dtype=float)
        q = np.array([[0.01]], dtype=float)
        r = np.array([[0.04]], dtype=float)
        x0 = np.array([0.0], dtype=float)
        p0 = np.array([[1.0]], dtype=float)
        return (obs, a, h, q, r, x0, p0)
    raise KeyError(f"Unknown function factory: {name}")


def _assert_close(name: str, observed, expected) -> None:
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    assert obs.shape == exp.shape, f"{name}: shape mismatch {obs.shape} != {exp.shape}"
    np.testing.assert_allclose(obs, exp, rtol=1e-7, atol=1e-9)


def _assert_fixture(case_id: str, observed: dict[str, np.ndarray]) -> None:
    if loadmat is None:
        return
    fixture_path = FIXTURE_ROOT / case_id / "case.mat"
    if not fixture_path.exists():
        return
    data = loadmat(fixture_path)
    for key, value in observed.items():
        if key in data:
            _assert_close(key, value, np.asarray(data[key]))


def test_function_contracts_from_manifest() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    cases = payload.get("cases", [])
    assert cases, "function_contracts.yml is empty"

    for case in cases:
        case_id = str(case["id"])
        fn = _resolve_symbol(str(case["function"]))
        args = _factory(str(case["factory"]))
        expect = case.get("expect", {})

        out = fn(*args)
        if case_id == "psth_counts":
            rate, counts = out
            _assert_close("rate", rate, expect["rate"])
            _assert_close("counts", counts, expect["counts"])
            observed = {"rate": np.asarray(rate), "counts": np.asarray(counts)}
        elif case_id == "linear_decode_basic":
            _assert_close("coefficients", out["coefficients"], expect["coefficients"])
            _assert_close("decoded", out["decoded"], expect["decoded"])
            observed = {
                "coefficients": np.asarray(out["coefficients"]),
                "decoded": np.asarray(out["decoded"]),
            }
        elif case_id == "kalman_filter_scalar":
            _assert_close("state", out["state"].reshape(-1), expect["state"])
            _assert_close("cov", out["cov"].reshape(-1), expect["cov"])
            observed = {"state": np.asarray(out["state"]).reshape(-1), "cov": np.asarray(out["cov"]).reshape(-1)}
        else:
            raise AssertionError(f"Unhandled contract id: {case_id}")

        _assert_fixture(case_id, observed)


def test_function_contract_manifest_has_unique_ids() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    ids = [str(row.get("id", "")).strip() for row in payload.get("cases", [])]
    assert len(ids) == len(set(ids)), "Duplicate case id in parity/function_contracts.yml"
