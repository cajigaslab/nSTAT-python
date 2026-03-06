from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import yaml

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "parity" / "class_contracts.yml"
FIXTURE_ROOT = REPO_ROOT / "tests" / "parity" / "fixtures" / "matlab_gold" / "classes"


def _resolve_symbol(dotted: str):
    module_name, symbol_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, symbol_name)


def _factory(name: str):
    if name == "nst_coll_two_trains":
        from nstat.nspikeTrain import nspikeTrain
        from nstat.nstColl import nstColl

        return nstColl([nspikeTrain(np.array([0.1, 0.4])), nspikeTrain(np.array([0.2]))])
    raise KeyError(f"Unknown contract factory: {name}")


def _assert_expectations(obj, expect: dict):
    if "dimension" in expect:
        assert int(getattr(obj, "dimension")) == int(expect["dimension"])
    if "data_shape" in expect:
        assert list(np.asarray(getattr(obj, "data")).shape) == [int(v) for v in expect["data_shape"]]
    if "num_spike_trains" in expect:
        value = int(getattr(obj, "numSpikeTrains", getattr(obj, "num_spike_trains", -1)))
        assert value == int(expect["num_spike_trains"])


def _assert_fixture(case_id: str, obj) -> None:
    if loadmat is None:
        return
    fixture_path = FIXTURE_ROOT / case_id / "case.mat"
    if not fixture_path.exists():
        return
    data = loadmat(fixture_path)
    if "expectedDimension" in data and hasattr(obj, "dimension"):
        assert int(np.asarray(data["expectedDimension"]).reshape(-1)[0]) == int(getattr(obj, "dimension"))


def test_class_contracts_from_manifest() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    cases = payload.get("cases", [])
    assert cases, "class_contracts.yml is empty"

    for case in cases:
        case_id = str(case["id"])
        class_path = str(case["class"])
        ctor = case.get("constructor", {})
        expect = case.get("expect", {})

        cls = _resolve_symbol(class_path)

        if "factory" in ctor:
            obj = _factory(str(ctor["factory"]))
        else:
            args = ctor.get("args", [])
            obj = cls(*args)

        _assert_expectations(obj, expect)
        _assert_fixture(case_id, obj)


def test_class_contract_manifest_has_unique_ids() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    ids = [str(row.get("id", "")).strip() for row in payload.get("cases", [])]
    assert len(ids) == len(set(ids)), "Duplicate case id in parity/class_contracts.yml"
